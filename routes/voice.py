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
    build_prompt,
    PatientState,
    extract_patient_state,
    evaluate_triage_tier,
    check_medication_contraindications
)
from research.src.clinical_triage import check_mid_conversation_correction
from services.chat_service import (
    build_history_text,
    update_memory_in_background,
    update_title_in_background,
    get_active_session_for_user,
    generate_voice_response,
    load_patient_state,
    save_patient_state,
)
from services.task_dispatcher import (
    enqueue_memory_update,
    enqueue_title_update
)
from services.routing_service import route_query
from services.llm_classifier import (
    generate_greeting,
    _is_greeting_sane,
    get_static_greeting_fallback
)
from research.src.guardrails import apply_output_guardrails, NON_MEDICAL_REFUSAL
from services.state_store import get_state_store, check_and_apply_ttl, check_emergency_resolution
from services.output_guardrails import ClinicalOutputGuardrail
from services.audit_logger import AuditLogger
from research.src.clinical_parser import ClinicalEntityParser
from research.src.clinical_triage import ClinicalTriageEngine

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
        except Exception as e:
            current_app.logger.error(f"[voice_chat] Error generating voice response: {e}", exc_info=True)
            answer = "Sorry, something went wrong on my end."
        return jsonify({"response": answer})

    app_obj = current_app._get_current_object()

    # Streaming path
    def g():
        try:
            chat_session = None
            if user:
                chat_session = get_active_session_for_user(user.id)

            # Extract last 3 recent turns
            recent_turns = []
            if chat_session:
                try:
                    prior_messages = Message.query.filter_by(session_id=chat_session.id).order_by(Message.id.desc()).limit(3).all()
                    for m in reversed(prior_messages):
                        recent_turns.append({"role": m.role, "content": m.content})
                except Exception:
                    recent_turns = []

            # Load persisted patient state
            state_store = get_state_store()
            patient_state = (state_store.get(str(chat_session.id)) if chat_session else None) or load_patient_state(chat_session)

            # Unified route_query
            route_decision = route_query(
                msg=msg,
                session=None,
                patient_state=patient_state,
                recent_turns=recent_turns,
                session_id=str(chat_session.id if chat_session else "none")
            )

            # 1. BLOCKED_INJECTION
            if route_decision.action == "BLOCKED_INJECTION":
                yield route_decision.response or "Request blocked."
                return

            # 2. BLOCKED_EMERGENCY
            if route_decision.action == "BLOCKED_EMERGENCY":
                if chat_session:
                    try:
                        user_msg = Message(session_id=chat_session.id, role="user", content=msg)
                        bot_msg = Message(session_id=chat_session.id, role="assistant", content=route_decision.response)
                        db.session.add_all([user_msg, bot_msg])
                        db.session.commit()
                    except Exception:
                        pass
                yield route_decision.response or "Medical emergency detected."
                return

            # 3. NON_MEDICAL (Amendment 5: No memory, title, summarization, or eval tasks)
            if route_decision.action == "NON_MEDICAL":
                ans_refusal = route_decision.response or NON_MEDICAL_REFUSAL
                if chat_session:
                    try:
                        user_msg = Message(session_id=chat_session.id, role="user", content=msg)
                        bot_msg = Message(session_id=chat_session.id, role="assistant", content=ans_refusal)
                        db.session.add_all([user_msg, bot_msg])
                        db.session.commit()
                    except Exception:
                        pass
                yield ans_refusal
                return

            # 4. GREETING (Amendment 5 & 9: timeout 2.0s, no disclaimer consumed, no title, no memory)
            if route_decision.action == "GREETING":
                raw_greeting = generate_greeting(msg, recent_turns=recent_turns, language=route_decision.language, timeout=2.0)
                if not _is_greeting_sane(raw_greeting):
                    raw_greeting = get_static_greeting_fallback(route_decision.language)
                answer = apply_output_guardrails(raw_greeting, is_medical=False, show_disclaimer=False)
                if chat_session:
                    try:
                        user_msg = Message(session_id=chat_session.id, role="user", content=msg)
                        bot_msg = Message(session_id=chat_session.id, role="assistant", content=str(answer))
                        db.session.add_all([user_msg, bot_msg])
                        chat_session.updated_at = datetime.now(timezone.utc)
                        db.session.commit()
                    except Exception:
                        pass
                yield str(answer)
                return

            # 5. Handle ACCOUNT_ACTION turn
            if route_decision.action == "ACCOUNT_ACTION":
                ans_account = route_decision.response or "Please use the account controls available in the application."
                if chat_session:
                    try:
                        user_msg = Message(session_id=chat_session.id, role="user", content=msg)
                        bot_msg = Message(session_id=chat_session.id, role="assistant", content=ans_account)
                        db.session.add_all([user_msg, bot_msg])
                        chat_session.updated_at = datetime.now(timezone.utc)
                        db.session.commit()
                    except Exception:
                        pass
                yield ans_account
                return

            # 6. MEDICAL Path
            if chat_session:
                user_msg = Message(session_id=chat_session.id, role="user", content=msg)
                db.session.add(user_msg)
                db.session.commit()

            history_text = build_history_text(chat_session) if chat_session else ""
            user_memory = get_user_memory(user) if user else "Voice Session"

            if user and hasattr(user, "id"):
                enqueue_memory_update(user.id, msg, history_text)

            if chat_session and chat_session.title == "New Consultation":
                enqueue_title_update(chat_session.id, msg)

            if check_and_apply_ttl(patient_state) and chat_session:
                AuditLogger.log_ttl_expired(session_id=str(chat_session.id))
            if check_emergency_resolution(msg, patient_state) and chat_session:
                AuditLogger.log_emergency_resolved(session_id=str(chat_session.id))

            signals = ClinicalEntityParser.extract_clinical_signals(msg)
            patient_state.transient_symptoms = signals.get("active_symptoms", [])

            prev_snapshot = patient_state.get_diff_snapshot()
            patient_state = extract_patient_state(msg, patient_state)
            has_new_structured_fact = patient_state.has_state_diff(prev_snapshot)

            triage_eval = ClinicalTriageEngine.evaluate(msg, patient_state, signals)
            if triage_eval and triage_eval.get("tier") == "RESOLVED":
                if chat_session:
                    AuditLogger.log_emergency_resolved(session_id=str(chat_session.id))
                raw_answer = triage_eval["message"]
                answer = apply_output_guardrails(
                    raw_answer,
                    is_medical=True,
                    show_disclaimer=False,
                    patient_state=patient_state
                )
                if chat_session:
                    bot_msg = Message(session_id=chat_session.id, role="assistant", content=str(answer))
                    db.session.add(bot_msg)
                    save_patient_state(chat_session, patient_state)
                    state_store.set(str(chat_session.id), patient_state)
                    chat_session.updated_at = datetime.now(timezone.utc)
                    db.session.commit()
                yield str(answer)
                return

            if triage_eval and triage_eval.get("requires_escalation"):
                patient_state.emergency_override_served = True
                patient_state.risk_tier = "Emergency"
                em_type = triage_eval.get("emergency_type", "FEBRILE_NEUTROPENIA")
                patient_state.active_emergency = em_type
                raw_answer = triage_eval["message"]
                answer = apply_output_guardrails(
                    raw_answer,
                    is_medical=True,
                    show_disclaimer=False,
                    patient_state=patient_state,
                    triage_tier="Emergency"
                )
                answer, guardrail_triggered = ClinicalOutputGuardrail.sanitize_response(answer, patient_state)
                if chat_session:
                    if guardrail_triggered:
                        AuditLogger.log_guardrail_override(
                            session_id=str(chat_session.id),
                            condition="EMERGENCY_OUTPUT_GUARDRAIL",
                            rule_triggered=answer.trigger_reason or "GUARDRAIL_INTERCEPTION",
                            raw_tokens_intercepted=answer.intercepted_tokens
                        )
                    if em_type == "FEBRILE_NEUTROPENIA_CAREGIVER":
                        AuditLogger.log_caregiver_intercept(
                            session_id=str(chat_session.id),
                            condition="FEBRILE_NEUTROPENIA_CAREGIVER",
                            rule_triggered="CAREGIVER_TRIAGE_INTERCEPT"
                        )
                    else:
                        AuditLogger.log_emergency_triggered(
                            session_id=str(chat_session.id),
                            condition=em_type,
                            rule_triggered="CLINICAL_TRIAGE_ESCALATION"
                        )

                save_patient_state(chat_session, patient_state)
                if chat_session:
                    state_store.set(str(chat_session.id), patient_state)
                    bot_msg = Message(session_id=chat_session.id, role="assistant", content=str(answer))
                    db.session.add(bot_msg)
                    chat_session.updated_at = datetime.now(timezone.utc)
                    db.session.commit()
                yield str(answer)
                return

            # Intent Determination: Derived from route_decision with zero double LLM calls
            intent = classify_intent(
                classifierModel,
                msg,
                route_decision=route_decision,
                recent_turns=recent_turns,
                patient_state=patient_state
            )

            correction_alert = check_mid_conversation_correction(patient_state, history_text)
            dosing_blocked, dosing_refusal = check_medication_contraindications(patient_state, msg)
            risk_tier, red_flags, override_guidance = evaluate_triage_tier(
                patient_state, msg, has_new_structured_fact=has_new_structured_fact
            )
            patient_state.risk_tier = risk_tier
            patient_state.red_flags = red_flags
            dynamic_prompt = build_prompt(history_text, user_memory, user=user, patient_state=patient_state)

            full_response = []
            if override_guidance and risk_tier == "Emergency":
                text = override_guidance
                if correction_alert:
                    text = f"{correction_alert}\n\n{text}"
                yield text
                full_response.append(text)
            elif dosing_blocked:
                text = dosing_refusal
                if correction_alert:
                    text = f"{correction_alert}\n\n{text}"
                yield text
                full_response.append(text)
            elif intent == "medical_query":
                try:
                    docs = retriever.invoke(msg)
                    context = "\n\n".join([doc.page_content for doc in docs])
                except Exception:
                    context = "No reference material available; answer from general knowledge, do not cite sources."
                if not context.strip():
                    context = "No reference material available; answer from general knowledge, do not cite sources."
                formatted_prompt = dynamic_prompt.format(context=context, input=msg)
                
                if correction_alert:
                    yield correction_alert + "\n\n"
                    full_response.append(correction_alert + "\n\n")

                for chunk in chatModel.stream(formatted_prompt):
                    text = chunk.content
                    if text:
                        yield text
                        full_response.append(text)
                patient_state.disclaimer_shown = True
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
                else:
                    prompt_val = NON_MEDICAL_REFUSAL
                    yield prompt_val
                    full_response.append(prompt_val)
                    prompt_val = None

                if prompt_val:
                    for chunk in chatModel.stream(prompt_val):
                        text = chunk.content
                        if text:
                            yield text
                            full_response.append(text)

            save_patient_state(chat_session, patient_state)
            if chat_session:
                state_store.set(str(chat_session.id), patient_state)
            if chat_session and full_response:
                bot_answer = "".join(full_response)
                bot_msg = Message(session_id=chat_session.id, role="assistant", content=bot_answer)
                db.session.add(bot_msg)
                chat_session.updated_at = datetime.now(timezone.utc)
                db.session.commit()

        except Exception as e:
            current_app.logger.error(f"[voice_chat] Streaming error: {e}", exc_info=True)
            yield "Sorry, something went wrong on my end."
        finally:
            db.session.remove()

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
