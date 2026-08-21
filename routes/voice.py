import os
import time
import threading
import traceback
from datetime import datetime, timezone

from flask import (
    Blueprint,
    request,
    jsonify,
    Response,
    stream_with_context,
    current_app
)
from flask_login import login_required, current_user
from livekit.api import LiveKitAPI, CreateAgentDispatchRequest

from livekit_token import generate_livekit_token
from research.src.auth import db, User, Message
from research.src.memory import get_user_memory
from research.src.intent_classifier import classify_intent
from services.ai_service import (
    chatModel,
    classifierModel,
    retriever,
    is_prompt_injection,
    build_prompt
)
from services.chat_service import (
    build_history_text,
    update_memory_in_background,
    update_title_in_background,
    get_active_session_for_user,
    generate_voice_response
)

voice_bp = Blueprint("voice", __name__)


def _normalize_livekit_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return "wss://localhost:7880"
    if url.startswith("https://"):
        return "wss://" + url[8:]
    elif url.startswith("http://"):
        return "ws://" + url[7:]
    return url


@voice_bp.route("/voice_chat", methods=["POST"], endpoint="voice_chat")
def voice_chat():
    """Called by voice_worker.py — no login required (internal agent process)."""
    data = request.get_json(silent=True) or {}
    msg  = data.get("message", "").strip()
    user_id = data.get("user_id")
    stream = data.get("stream", False)

    if not msg:
        if stream:
            return Response("No message received", mimetype="text/plain")
        return jsonify({"response": "No message received"})
        
    if is_prompt_injection(msg):
        refusal_msg = "I cannot fulfill this request. I am a medical AI assistant, and my instructions cannot be overridden."
        if stream:
            def g(): yield refusal_msg
            return Response(stream_with_context(g()), mimetype="text/plain")
        return jsonify({"response": refusal_msg})

    # Fetch user if user_id is provided
    user = None
    if user_id:
        try:
            user = User.query.get(int(user_id))
        except Exception:
            pass

    if not stream:
        # non-streaming path
        try:
            if user:
                answer = generate_voice_response(msg, user=user)
            else:
                answer = generate_voice_response(msg)
        except Exception:
            traceback.print_exc()
            answer = "Sorry, something went wrong on my end."
        return jsonify({"response": answer})

    app_obj = current_app._get_current_object()

    # Streaming path
    def g():
        try:
            # 1. Fast Intent Classification
            intent = classify_intent(classifierModel, msg)
            print("=" * 50)
            print("VOICE USER:", msg)
            print("VOICE INTENT:", intent)
            print("=" * 50)

            # 2. Get active session and save user message if user is authenticated
            chat_session = None
            if user:
                chat_session = get_active_session_for_user(user.id)
                user_msg = Message(session_id=chat_session.id, role="user", content=msg)
                db.session.add(user_msg)
                db.session.commit()

                # Trigger background memory update asynchronously
                threading.Thread(
                    target=update_memory_in_background,
                    args=(app_obj, user.id, msg, build_history_text(chat_session)),
                    daemon=True
                ).start()

                # Trigger background title update asynchronously
                if chat_session.title == "New Consultation":
                    threading.Thread(
                        target=update_title_in_background,
                        args=(app_obj, chat_session.id, msg),
                        daemon=True
                    ).start()

            # 3. Setup prompt
            history_text = build_history_text(chat_session) if chat_session else ""
            user_memory = get_user_memory(user) if user else "Voice Session"
            dynamic_prompt = build_prompt(history_text, user_memory, user=user)

            # 4. Stream response from LLM
            full_response = []
            if intent == "medical_query":
                # Retrieve documents from Pinecone
                docs = retriever.invoke(msg)
                context = "\n\n".join([doc.page_content for doc in docs])
                formatted_prompt = dynamic_prompt.format(context=context, input=msg)
                
                for chunk in chatModel.stream(formatted_prompt):
                    text = chunk.content
                    if text:
                        yield text
                        full_response.append(text)
            else:
                if intent == "memory_recall":
                    prompt_val = f"User Memory:\n{user_memory}\n\nConversation History:\n{history_text}\n\nUser:\n{msg}"
                elif intent == "greeting":
                    prompt_val = f"Reply naturally to: {msg}"
                elif intent == "account_action":
                    prompt_val = "Please use the account controls available in the application."
                    yield prompt_val
                    full_response.append(prompt_val)
                    prompt_val = None
                elif intent == "general_chat":
                    prompt_val = f"Conversation History:\n{history_text}\n\nUser:\n{msg}"
                else:
                    prompt_val = msg

                if prompt_val:
                    for chunk in chatModel.stream(prompt_val):
                        text = chunk.content
                        if text:
                            yield text
                            full_response.append(text)

            # 5. Save assistant response to DB
            if chat_session and full_response:
                bot_answer = "".join(full_response)
                bot_msg = Message(session_id=chat_session.id, role="assistant", content=bot_answer)
                db.session.add(bot_msg)
                chat_session.updated_at = datetime.now(timezone.utc)
                db.session.commit()

        except Exception as e:
            traceback.print_exc()
            yield "Sorry, something went wrong on my end."

    return Response(stream_with_context(g()), mimetype="text/plain")


@voice_bp.route("/livekit_token", methods=["GET"], endpoint="livekit_token_route")
@login_required
def livekit_token_route():
    """Returns a LiveKit JWT so the browser can join a private voice room."""
    try:
        room_name = f"mediassist-voice-{current_user.id}-{int(time.time())}"

        token = generate_livekit_token(
            room_name=room_name,
            participant_identity=str(current_user.id),
            participant_name=current_user.name,
            dispatch_agent=False,
            agent_metadata=str(current_user.id),
        )

        livekit_url = _normalize_livekit_url(os.getenv("LIVEKIT_URL", "wss://localhost:7880"))

        return jsonify({
            "token":   token,
            "url":     livekit_url,
            "room":    room_name,
            "user_id": current_user.id,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "token": None}), 500


@voice_bp.route("/dispatch_agent", methods=["POST"], endpoint="dispatch_agent")
@login_required
def dispatch_agent():
    """Dispatches the medical-agent worker into the LiveKit room."""
    import asyncio

    data      = request.get_json(silent=True) or {}
    room_name = data.get("room") or f"mediassist-voice-{current_user.id}-{int(time.time())}"
    language  = data.get("language", "en").strip()

    livekit_url        = _normalize_livekit_url(os.getenv("LIVEKIT_URL", ""))
    livekit_api_key    = os.getenv("LIVEKIT_API_KEY",    "").strip()
    livekit_api_secret = os.getenv("LIVEKIT_API_SECRET", "").strip()

    if not livekit_api_key or not livekit_api_secret:
        return jsonify({
            "status":  "error",
            "message": "LIVEKIT_API_KEY or LIVEKIT_API_SECRET missing from .env"
        }), 500

    result = {}
    errors = []

    def run_dispatch():
        async def _dispatch():
            def has_active_job(dispatch) -> bool:
                state = getattr(dispatch, "state", None)
                if not state or getattr(state, "deleted_at", 0):
                    return False

                for job in getattr(state, "jobs", []):
                    job_state  = getattr(job, "state", None)
                    status     = getattr(job_state, "status",     None)
                    ended_at   = getattr(job_state, "ended_at",   0)
                    started_at = getattr(job_state, "started_at", 0)
                    if status == 1 and started_at and not ended_at:
                        return True
                return False

            try:
                async with LiveKitAPI(
                    url=livekit_url,
                    api_key=livekit_api_key,
                    api_secret=livekit_api_secret
                ) as lkapi:
                    dispatches = await lkapi.agent_dispatch.list_dispatch(room_name)
                    medical_dispatches = [
                        d for d in dispatches
                        if d.agent_name == "medical-agent"
                    ]

                    active_dispatches = [
                        d for d in medical_dispatches
                        if has_active_job(d)
                    ]

                    dispatch_to_reuse = active_dispatches[0] if active_dispatches else None
                    dispatches_to_delete = [
                        d for d in medical_dispatches
                        if d.id != getattr(dispatch_to_reuse, "id", None)
                    ]

                    for extra in dispatches_to_delete:
                        await lkapi.agent_dispatch.delete_dispatch(
                            extra.id,
                            room_name,
                        )
                        print(
                            f"[dispatch_agent] Removed stale dispatch "
                            f"'{extra.id}' from room '{room_name}'"
                        )

                    if dispatch_to_reuse:
                        result["status"]      = "ok"
                        result["dispatch_id"] = str(dispatch_to_reuse.id)
                        result["reused"]      = True
                        print(
                            f"[dispatch_agent] Reusing active dispatch "
                            f"'{dispatch_to_reuse.id}' for room '{room_name}'"
                        )
                        return

                    dispatch = await lkapi.agent_dispatch.create_dispatch(
                        CreateAgentDispatchRequest(
                            agent_name="medical-agent",
                            room=room_name,
                            metadata=language,
                        )
                    )
                    result["status"]      = "ok"
                    result["dispatch_id"] = str(dispatch.id)
                    result["reused"]      = False
                    print(
                        f"[dispatch_agent] Dispatched 'medical-agent' "
                        f"to room '{room_name}' -> id={dispatch.id}"
                    )

            except Exception as exc:
                traceback.print_exc()
                errors.append(str(exc))
                result["status"] = "error"

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_dispatch())
        finally:
            loop.close()

    t = threading.Thread(target=run_dispatch, daemon=True)
    t.start()
    t.join(timeout=10)

    if errors:
        return jsonify({"status": "error", "message": errors[0]}), 200

    return jsonify(result)
