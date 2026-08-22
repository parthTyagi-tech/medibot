import threading
import traceback
from datetime import datetime, timezone

from flask import (
    Blueprint,
    render_template,
    request,
    jsonify,
    session,
    current_app
)
from flask_login import login_required, current_user
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import app as app_module
from research.src.auth import db, ChatSession, Message
from research.src.memory import get_user_memory, clear_user_memory
from deepgram_tts import text_to_speech
from services.chat_service import (
    get_active_session,
    build_history_text,
    update_memory_in_background,
    update_title_in_background,
    summarize_session,
    summarize_session_in_background
)

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/health", methods=["GET"])
@chat_bp.route("/ping", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "MediAssist AI",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), 200


@chat_bp.route("/", endpoint="index")
@login_required
def index():
    past_sessions = ChatSession.query.filter_by(
        user_id=current_user.id
    ).order_by(ChatSession.updated_at.desc()).limit(10).all()

    return render_template(
        "chat.html",
        user=current_user,
        past_sessions=past_sessions,
        active_session_id=session.get("chat_session_id")
    )


@chat_bp.route("/new_chat", methods=["POST"], endpoint="new_chat")
@login_required
def new_chat():
    session_id = session.get("chat_session_id")
    if session_id:
        old_session = ChatSession.query.get(session_id)
        if old_session:
            summarize_session(old_session)

    session.pop("chat_session_id", None)
    return jsonify({"status": "ok"})


@chat_bp.route("/load_session/<int:session_id>", methods=["GET"], endpoint="load_session")
@login_required
def load_session(session_id):
    chat_session = ChatSession.query.filter_by(
        id=session_id, user_id=current_user.id
    ).first()

    if not chat_session:
        return jsonify({"error": "Session not found"}), 404

    session["chat_session_id"] = session_id
    print("LOADED SESSION:", session_id)

    messages = Message.query.filter_by(
        session_id=session_id
    ).order_by(Message.created_at).all()

    return jsonify({
        "session_id": session_id,
        "title":      chat_session.title,
        "messages":   [{"role": m.role, "content": m.content} for m in messages]
    })


@chat_bp.route("/get", methods=["POST"], endpoint="chat")
@login_required
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "Please enter a message."

    # 1. Run Input Guardrails (Prompt injection, Content safety, Medical emergency)
    is_blocked, category, guard_msg = app_module.apply_input_guardrails(msg)
    if is_blocked:
        # If medical emergency, save user message and emergency guidance in chat history
        if category == "medical_emergency":
            try:
                chat_session = get_active_session()
                user_msg = Message(session_id=chat_session.id, role="user", content=msg)
                bot_msg = Message(session_id=chat_session.id, role="assistant", content=guard_msg)
                db.session.add_all([user_msg, bot_msg])
                db.session.commit()
            except Exception:
                pass
        return guard_msg

    intent = app_module.classify_intent(app_module.classifierModel, msg)
    print("=" * 50)
    print("USER:", msg)
    print("INTENT:", intent)
    print("=" * 50)

    try:
        chat_session = get_active_session()

        user_msg = Message(session_id=chat_session.id, role="user", content=msg)
        db.session.add(user_msg)
        db.session.commit()

        history_text  = build_history_text(chat_session)
        user_memory   = get_user_memory(current_user)

        app_obj = current_app._get_current_object()

        # Trigger background memory update asynchronously (context-aware)
        threading.Thread(
            target=update_memory_in_background,
            args=(app_obj, current_user.id, msg, history_text),
            daemon=True
        ).start()

        # Trigger background title update asynchronously
        if chat_session.title == "New Consultation":
            threading.Thread(
                target=update_title_in_background,
                args=(app_obj, chat_session.id, msg),
                daemon=True
            ).start()

        # Trigger background context window summarization when history reaches 6+ messages
        msg_count = Message.query.filter_by(session_id=chat_session.id).count()
        if msg_count >= 6 and msg_count % 4 == 0:
            threading.Thread(
                target=summarize_session_in_background,
                args=(app_obj, chat_session.id),
                daemon=True
            ).start()


        dynamic_prompt = app_module.build_prompt(history_text, user_memory)

        if intent == "medical_query":
            question_answer_chain = create_stuff_documents_chain(app_module.chatModel, dynamic_prompt)
            rag_chain = create_retrieval_chain(app_module.retriever, question_answer_chain)
            response  = rag_chain.invoke({"input": msg})
            raw_answer = response.get("answer", "Sorry, I couldn't generate a response.")
            answer = app_module.apply_output_guardrails(raw_answer, is_medical=True)

        elif intent == "greeting":
            first_name = current_user.name.split()[0] if current_user and current_user.name else "there"
            greeting_prompt = (
                f"You are MediAssist, an empathetic medical AI assistant. "
                f"The user ({first_name}) said: '{msg}'. "
                f"Reply warmly in 1-2 friendly sentences and ask how you can assist with their health, symptoms, or medical questions today."
            )
            raw_resp = app_module.chatModel.invoke(greeting_prompt)
            raw_answer = raw_resp.content if hasattr(raw_resp, "content") else str(raw_resp)
            answer = app_module.apply_output_guardrails(raw_answer, is_medical=False)

        elif intent == "memory_recall":
            recall_prompt = (
                f"You are MediAssist. Answer the user's question about their medical history or previous discussion.\n"
                f"User Profile & Known Medical Memory:\n{user_memory}\n\n"
                f"Recent Conversation:\n{history_text}\n\n"
                f"User Query: {msg}"
            )
            raw_resp = app_module.chatModel.invoke(recall_prompt)
            raw_answer = raw_resp.content if hasattr(raw_resp, "content") else str(raw_resp)
            answer = app_module.apply_output_guardrails(raw_answer, is_medical=False)

        elif intent == "account_action":
            answer = "Please use the account controls available in the navigation bar to manage your account or consultation history."

        else:
            # Non-medical query: enforce strict medical specialization
            answer = app_module.NON_MEDICAL_REFUSAL

        bot_msg = Message(session_id=chat_session.id, role="assistant", content=answer)
        db.session.add(bot_msg)

        chat_session.updated_at = datetime.now(timezone.utc)
        db.session.commit()

        return answer

    except Exception:
        traceback.print_exc()
        return "Something went wrong."


@chat_bp.route("/delete_session/<int:session_id>", methods=["POST"], endpoint="delete_session")
@login_required
def delete_session(session_id):
    chat_session = ChatSession.query.filter_by(
        id=session_id, user_id=current_user.id
    ).first()

    if not chat_session:
        return jsonify({"success": False}), 404

    db.session.delete(chat_session)
    db.session.commit()

    if session.get("chat_session_id") == session_id:
        session.pop("chat_session_id", None)

    return jsonify({"success": True})


@chat_bp.route("/get_memory", methods=["GET"], endpoint="get_user_memory_route")
@login_required
def get_user_memory_route():
    return jsonify({"memory": get_user_memory(current_user)})


@chat_bp.route("/clear_memory", methods=["POST"], endpoint="clear_user_memory_route")
@login_required
def clear_user_memory_route():
    try:
        clear_user_memory(current_user)
        db.session.commit()
        return jsonify({"status": "ok"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


@chat_bp.route("/tts", methods=["POST"], endpoint="tts")
@login_required
def tts():
    text = request.form.get("text", "")
    filename = text_to_speech(text)
    return jsonify({"audio_url": f"/{filename}"})
