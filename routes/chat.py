import os
import re
import random
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

import logging

logger = logging.getLogger(__name__)

chat_bp = Blueprint("chat", __name__)

GREETING_PATTERNS = [
    r"^(hi|hello|hey|greetings|howdy|good\s+(morning|afternoon|evening|day)|salutations)\b",
    r"^how\s+(are\s+you|is\s+it\s+going|do\s+you\s+do|good\s+are\s+you)\b",
    r"^(what\'?s\s+up|sup|yo)\b",
]


def is_greeting_text(text: str) -> bool:
    cleaned = (text or "").strip().lower()
    return any(bool(re.search(pat, cleaned)) for pat in GREETING_PATTERNS)


from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from research.src.auth import db, ChatSession, Message
from research.src.memory import get_user_memory, clear_user_memory
from deepgram_tts import text_to_speech
from research.src.clinical_triage import (
    PatientState,
    ClinicalTriageEngine,
    format_context_aware_greeting,
    extract_patient_state,
    evaluate_triage_tier,
    check_medication_contraindications,
    check_mid_conversation_correction,
    friendly_emergency_label,
    maybe_append_emergency_reminder
)
from research.src.intent_classifier import is_third_party_query
from services.chat_service import (
    get_active_session,
    build_history_text,
    update_memory_in_background,
    update_title_in_background,
    summarize_session,
    summarize_session_in_background,
    load_patient_state,
    save_patient_state,
)
from services.task_dispatcher import (
    enqueue_memory_update,
    enqueue_title_update,
    enqueue_session_summarize,
    enqueue_eval_safety_check,
    enqueue_eval_turn
)
from services.state_store import (
    get_state_store,
    check_and_apply_ttl,
    check_emergency_resolution
)
from services.output_guardrails import ClinicalOutputGuardrail
from services.audit_logger import AuditLogger
from research.src.clinical_parser import ClinicalEntityParser
from services.routing_service import route_query, RouteDecision
from services.llm_classifier import (
    generate_greeting,
    _is_greeting_sane,
    get_static_greeting_fallback
)


@chat_bp.route("/health", methods=["GET"])
@chat_bp.route("/ping", methods=["GET"])
def health():
    db_status = "ok"
    try:
        from research.src.auth import db
        from sqlalchemy import text
        db.session.execute(text("SELECT 1"))
    except Exception as e:
        logger.warning("Database health check failed: %s", type(e).__name__)
        try:
            from research.src.auth import db
            db.session.rollback()
        except Exception as rollback_err:
            logger.warning("Database rollback failed: %s", type(rollback_err).__name__)
        return jsonify({
            "status": "degraded",
            "database": f"error: {type(e).__name__}",
            "service": "MediAssist AI",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 503

    return jsonify({
        "status": "healthy",
        "database": db_status,
        "service": "MediAssist AI",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), 200


@chat_bp.route("/", endpoint="index")
@login_required
def index():
    try:
        past_sessions = ChatSession.query.filter_by(
            user_id=current_user.id
        ).order_by(ChatSession.updated_at.desc()).limit(10).all()
    except Exception as e:
        logger.error("Error fetching past_sessions on index route: %s", type(e).__name__)
        db.session.rollback()
        past_sessions = []

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
    ).order_by(Message.created_at.asc(), Message.id.asc()).all()

    return jsonify({
        "session_id": session_id,
        "title":      chat_session.title,
        "messages":   [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.created_at.strftime("%I:%M %p") if m.created_at else ""
            }
            for m in messages
        ]
    })


def handle_chat_turn(msg: str, user=None) -> str:
    """
    Request lifecycle handler for chat messages.
    - Ensures session['patient_state'] persists across turns with default keys
    - Runs Pre-intent ClinicalTriageEngine evaluation
    - Intercepts greetings with context-aware check-in
    - Handles medical queries and safety guardrails
    """
    if not msg or not msg.strip():
        return "Please enter a message."
    msg = msg.strip()

    if user is None:
        user = current_user if current_user and current_user.is_authenticated else None

    # Ensure session["patient_state"] persists across turns with default keys
    try:
        if "patient_state" not in session or not isinstance(session.get("patient_state"), dict):
            session["patient_state"] = {
                "has_cancer_history": False,
                "active_emergency": None,
                "emergency_turn_count": 0,
                "current_symptoms": []
            }
        else:
            session["patient_state"].setdefault("has_cancer_history", False)
            session["patient_state"].setdefault("active_emergency", None)
            session["patient_state"].setdefault("emergency_turn_count", 0)
            session["patient_state"].setdefault("current_symptoms", [])
    except Exception as e:
        logger.warning("Session patient state initialization failed: %s", type(e).__name__)

    try:
        from services.chat_pipeline import run_turn
        return run_turn(msg, user=user, channel="web")
    except Exception:
        traceback.print_exc()
        return "Something went wrong."


handle_chat_message = handle_chat_turn


@chat_bp.route("/get", methods=["POST"], endpoint="chat")
@login_required
def chat():
    msg = request.form.get("msg", "").strip()
    return handle_chat_turn(msg, current_user)


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


def __getattr__(name):
    if name == "app_module":
        import app
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


try:
    import app as app_module
except Exception:
    pass

