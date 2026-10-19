import json
import logging
import traceback
from datetime import datetime, timezone
from flask import session
from flask_login import current_user

from research.src.auth import db, User, ChatSession, Message
from research.src.memory import get_user_memory, update_user_memory
from research.src.clinical_triage import PatientState

logger = logging.getLogger("voice-backend")


def load_patient_state(chat_session: ChatSession) -> PatientState:
    """Load patient state from DB column, falling back to empty state."""
    if chat_session and chat_session.patient_state_json:
        try:
            return PatientState.from_dict(json.loads(chat_session.patient_state_json))
        except Exception:
            pass
    return PatientState()


def save_patient_state(chat_session: ChatSession, patient_state: PatientState) -> None:
    """Persist patient state to DB column."""
    if chat_session:
        chat_session.patient_state_json = json.dumps(patient_state.to_dict())


def get_active_session() -> ChatSession:
    """
    Retrieves or creates the active ChatSession for the currently authenticated user.
    
    Returns:
        ChatSession: The active chat session instance.
    """
    print("ACTIVE SESSION:", session.get("chat_session_id"))
    session_id = session.get("chat_session_id")

    if session_id:
        chat_session = ChatSession.query.filter_by(
            id=session_id,
            user_id=current_user.id
        ).first()
        if chat_session:
            return chat_session

    chat_session = ChatSession(user_id=current_user.id)
    db.session.add(chat_session)
    db.session.commit()
    session["chat_session_id"] = chat_session.id
    return chat_session


def build_history_text(chat_session: ChatSession) -> str:
    """
    Builds a formatted rolling dialogue history string with session summary context.
    
    Args:
        chat_session (ChatSession): The session to extract dialogue history for.
        
    Returns:
        str: Formatted context string containing past summaries and recent dialogue turns.
    """
    messages = Message.query.filter_by(
        session_id=chat_session.id
    ).order_by(Message.created_at.asc(), Message.id.asc()).all()

    if not messages:
        return chat_session.summary or ""

    # Rolling window: keep last 14 messages for high-resolution dialogue context
    history = []
    for m in messages[-14:]:
        role = "Patient" if m.role == "user" else "MediAssist"
        history.append(f"{role}: {m.content}")

    if chat_session.summary:
        return (
            f"Summary of Earlier Consultation Context:\n"
            f"{chat_session.summary}\n\n"
            f"Recent Consultation Dialogue:\n" + "\n".join(history)
        )

    return "\n".join(history)


def update_session_title(chat_session: ChatSession, first_message: str) -> None:
    """
    Generates a concise 4-6 word title for the chat session based on the initial user query.
    
    Args:
        chat_session (ChatSession): The session whose title is being generated.
        first_message (str): The initial user query.
    """
    import app
    if chat_session.title == "New Consultation":
        try:
            title_prompt = (
                f"Generate a short 4-6 word title for a chat that starts with: '{first_message}'. "
                f"Return ONLY the title, no quotes, no punctuation at the end."
            )
            response = app.chatModel.invoke(title_prompt)
            title = response.content.strip()[:60]
            chat_session.title = title if title else first_message[:50]
        except Exception:
            chat_session.title = first_message[:50]


def process_memory_update(user_id: int, latest_message: str, history_text: str) -> None:
    """
    Executes the memory extraction and updates user's permanent medical profile.
    Can be called by the decoupled background worker or synchronous fallback.
    """
    import app
    user = User.query.get(user_id)
    if user:
        update_user_memory(user, app.chatModel, latest_message, history_text)
        db.session.commit()
        print(f"[BG Memory Update] Completed for user {user_id}")


def process_title_update(session_id: int, first_message: str) -> None:
    """
    Generates and persists a chat session title.
    Can be called by the decoupled background worker or synchronous fallback.
    """
    chat_session = ChatSession.query.get(session_id)
    if chat_session and chat_session.title == "New Consultation":
        update_session_title(chat_session, first_message)
        db.session.commit()
        print(f"[BG Title Update] Completed for session {session_id}")


def process_session_summarize(session_id: int) -> None:
    """
    Summarizes long consultations.
    Can be called by the decoupled background worker or synchronous fallback.
    """
    chat_session = ChatSession.query.get(session_id)
    if chat_session:
        summarize_session(chat_session)
        print(f"[BG Summarize] Completed for session {session_id}")


def update_memory_in_background(app_instance, user_id: int, latest_message: str, history_text: str) -> None:
    """
    Backwards-compatible wrapper for updating user memory in background.
    """
    with app_instance.app_context():
        try:
            process_memory_update(user_id, latest_message, history_text)
        except Exception as e:
            db.session.rollback()
            print(f"[BG Memory Update] Conflict or error: {e}")
        finally:
            db.session.remove()


def update_title_in_background(app_instance, session_id: int, first_message: str) -> None:
    """
    Backwards-compatible wrapper for updating session title in background.
    """
    with app_instance.app_context():
        try:
            process_title_update(session_id, first_message)
        except Exception as e:
            db.session.rollback()
            print(f"[BG Title Update] Conflict or error: {e}")
        finally:
            db.session.remove()


def summarize_session(chat_session: ChatSession) -> None:
    """
    Generates a concise bulleted clinical summary of long consultations for context preservation.
    
    Args:
        chat_session (ChatSession): The chat session to summarize.
    """
    import app
    messages = Message.query.filter_by(
        session_id=chat_session.id
    ).order_by(Message.created_at).all()

    if len(messages) < 4:
        return

    conversation = "\n".join([
        f"{'Patient' if m.role == 'user' else 'MediAssist'}: {m.content}"
        for m in messages
    ])

    try:
        summary_prompt = (
            "Summarize this clinical patient consultation in 2-3 concise bullet points focusing on:\n"
            "- Key symptoms reported & duration\n"
            "- Important medical context/history\n"
            "- Clinical advice or triage guidance provided\n\n"
            f"{conversation}\n\nSummary:"
        )
        response = app.chatModel.invoke(summary_prompt)
        content = response.content if hasattr(response, "content") else str(response)
        chat_session.summary = content.strip()
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"[Summarize Session] Error: {e}")


def summarize_session_in_background(app_instance, session_id: int) -> None:
    """
    Backwards-compatible wrapper for summarizing session in background.
    """
    with app_instance.app_context():
        try:
            process_session_summarize(session_id)
        except Exception as e:
            print(f"[BG Summarize] Error: {e}")
        finally:
            db.session.remove()



def get_active_session_for_user(user_id: int) -> ChatSession:
    """
    Retrieves the most recent ChatSession for a given user ID or initializes a new session.
    
    Args:
        user_id (int): The user ID.
        
    Returns:
        ChatSession: The active or newly created chat session.
    """
    chat_session = ChatSession.query.filter_by(user_id=user_id).order_by(ChatSession.updated_at.desc()).first()
    if not chat_session:
        chat_session = ChatSession(user_id=user_id)
        db.session.add(chat_session)
        db.session.commit()
    return chat_session


def generate_voice_response(msg: str, user=None) -> str:
    """
    Generates a non-streaming voice response with full clinical triage parity.
    Delegates to the unified chat pipeline in services/chat_pipeline.py.
    """
    from services.chat_pipeline import run_turn
    try:
        uid = getattr(user, 'id', None)
        logger.info(f"[generate_voice_response] START - msg='{msg[:60]}', user_id={uid}")
        answer = run_turn(msg, user=user, channel="voice")
        final_answer = answer or "I am ready to assist with your medical questions."
        logger.info(f"[generate_voice_response] DONE - returning ({len(final_answer)} chars): '{final_answer[:80]}...'")
        return final_answer
    except Exception as e:
        logger.error("[generate_voice_response] EXCEPTION: %s", type(e).__name__)
        return "I am ready to assist with your medical questions."
