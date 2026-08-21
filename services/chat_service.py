import traceback
from datetime import datetime, timezone
from flask import session
from flask_login import current_user
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from research.src.auth import db, User, ChatSession, Message
from research.src.memory import get_user_memory, update_user_memory


def get_active_session():
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


def build_history_text(chat_session):
    messages = Message.query.filter_by(
        session_id=chat_session.id
    ).order_by(Message.created_at).all()

    if not messages:
        return chat_session.summary or ""

    history = []
    for m in messages[-10:]:
        role = "User" if m.role == "user" else "MediAssist"
        history.append(f"{role}: {m.content}")

    if chat_session.summary:
        return (
            f"Summary of earlier conversation:\n"
            f"{chat_session.summary}\n\n"
            f"Recent messages:\n" + "\n".join(history)
        )

    return "\n".join(history)


def update_session_title(chat_session, first_message):
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


def update_memory_in_background(app_instance, user_id, latest_message, history_text):
    import app
    with app_instance.app_context():
        try:
            user = User.query.get(user_id)
            if user:
                update_user_memory(user, app.chatModel, latest_message, history_text)
                db.session.commit()
                print(f"[BG Memory Update] Completed for user {user_id}")
        except Exception as e:
            db.session.rollback()
            print(f"[BG Memory Update] Conflict or error: {e}")


def update_title_in_background(app_instance, session_id, first_message):
    with app_instance.app_context():
        try:
            chat_session = ChatSession.query.get(session_id)
            if chat_session and chat_session.title == "New Consultation":
                update_session_title(chat_session, first_message)
                db.session.commit()
                print(f"[BG Title Update] Completed for session {session_id}")
        except Exception as e:
            db.session.rollback()
            print(f"[BG Title Update] Conflict or error: {e}")


def summarize_session(chat_session):
    import app
    messages = Message.query.filter_by(
        session_id=chat_session.id
    ).order_by(Message.created_at).all()

    if len(messages) < 4:
        return

    conversation = "\n".join([
        f"{'User' if m.role == 'user' else 'MediAssist'}: {m.content}"
        for m in messages
    ])

    try:
        summary_prompt = (
            "Summarize this medical conversation "
            "in 2-3 sentences focusing on:\n"
            "- symptoms\n- concerns\n- advice given\n\n"
            f"{conversation}\n\nSummary:"
        )
        response = app.chatModel.invoke(summary_prompt)
        chat_session.summary = response.content
        db.session.commit()
    except Exception:
        pass


def get_active_session_for_user(user_id):
    chat_session = ChatSession.query.filter_by(user_id=user_id).order_by(ChatSession.updated_at.desc()).first()
    if not chat_session:
        chat_session = ChatSession(user_id=user_id)
        db.session.add(chat_session)
        db.session.commit()
    return chat_session


def generate_voice_response(msg: str, user=None) -> str:
    """Generates a non-streaming voice response for voice_worker.py."""
    import app
    try:
        intent = app.classify_intent(app.classifierModel, msg)
        chat_session = None
        if user:
            chat_session = get_active_session_for_user(user.id)
            user_msg = Message(session_id=chat_session.id, role="user", content=msg)
            db.session.add(user_msg)
            db.session.commit()

        history_text = build_history_text(chat_session) if chat_session else ""
        user_memory = get_user_memory(user) if user else "Voice Session"
        dynamic_prompt = app.build_prompt(history_text, user_memory, user=user)

        answer = ""
        if intent == "medical_query":
            docs = app.retriever.invoke(msg)
            context = "\n\n".join([doc.page_content for doc in docs])
            formatted_prompt = dynamic_prompt.format(context=context, input=msg)
            response_obj = app.chatModel.invoke(formatted_prompt)
            answer = response_obj.content if hasattr(response_obj, "content") else str(response_obj)
        else:
            if intent == "memory_recall":
                prompt_val = f"User Memory:\n{user_memory}\n\nConversation History:\n{history_text}\n\nUser:\n{msg}"
            elif intent == "greeting":
                prompt_val = f"Reply naturally to: {msg}"
            elif intent == "account_action":
                answer = "Please use the account controls available in the application."
                prompt_val = None
            elif intent == "general_chat":
                prompt_val = f"Conversation History:\n{history_text}\n\nUser:\n{msg}"
            else:
                prompt_val = msg

            if prompt_val:
                response_obj = app.chatModel.invoke(prompt_val)
                answer = response_obj.content if hasattr(response_obj, "content") else str(response_obj)

        if chat_session and answer:
            bot_msg = Message(session_id=chat_session.id, role="assistant", content=answer)
            db.session.add(bot_msg)
            chat_session.updated_at = datetime.now(timezone.utc)
            db.session.commit()

        return answer or "I am ready to assist with your medical questions."
    except Exception as e:
        traceback.print_exc()
        return "I am ready to assist with your medical questions."
