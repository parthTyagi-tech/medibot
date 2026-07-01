from dotenv import load_dotenv

load_dotenv()

import os
import time
import traceback
import secrets
import threading
from datetime import datetime, timezone, timedelta

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    session,
    Response,
    stream_with_context
)

from flask_mail import Mail, Message as MailMessage
from flask_dance.contrib.google import google
from flask_login import (
    login_required,
    logout_user,
    current_user,
    login_user
)
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"

from livekit_token import generate_livekit_token, voice_room_name
from livekit.api import LiveKitAPI, CreateAgentDispatchRequest
from deepgram_tts import text_to_speech

from research.src.intent_classifier import classify_intent
from research.src.auth import (
    db,
    login_manager,
    User,
    ChatSession,
    Message,
    create_google_blueprint
)
from research.src.memory import (
    get_user_memory,
    update_user_memory
)
from research.src.helper import download_embeddings

from pinecone import Pinecone
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import Any
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# ─────────────────────────────────────────────────────────────
# Flask App Setup
# ─────────────────────────────────────────────────────────────

app = Flask(__name__)

app.config["MAIL_SERVER"]        = "smtp.gmail.com"
app.config["MAIL_PORT"]          = 587
app.config["MAIL_USE_TLS"]       = True
app.config["MAIL_USERNAME"]      = "parthtyagi3389@gmail.com"
app.config["MAIL_PASSWORD"]      = os.getenv("MAIL_PASSWORD", "ajbb ekwo anvz kdwg")
app.config["MAIL_DEFAULT_SENDER"] = "parthtyagi3389@gmail.com"

mail = Mail(app)

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

app.secret_key = os.getenv("SECRET_KEY", "fallback-secret")
# Only set SERVER_NAME if explicitly provided (e.g. for local testing).
# Leave unset on Render to allow the container to bind to the Render domain automatically.
if os.getenv("SERVER_NAME"):
    app.config["SERVER_NAME"] = os.getenv("SERVER_NAME")

app.config["SQLALCHEMY_DATABASE_URI"]        = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)
with app.app_context():
    db.create_all()
login_manager.init_app(app)
login_manager.login_view = "login"

google_bp = create_google_blueprint()
app.register_blueprint(google_bp, url_prefix="/login")


# ─────────────────────────────────────────────────────────────
# LangChain Setup
# ─────────────────────────────────────────────────────────────

embedding = download_embeddings()

class CustomPineconeRetriever(BaseRetriever):
    index: Any
    embeddings: Any
    k: int = 3

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        try:
            query_vector = self.embeddings.embed_query(query)
            results = self.index.query(vector=query_vector, top_k=self.k, include_metadata=True)
            docs = []
            for match in results.get("matches", []):
                metadata = match.get("metadata", {})
                text = metadata.get("text", "")
                docs.append(Document(page_content=text, metadata=metadata))
            return docs
        except Exception as e:
            print("Pinecone Retrieval Error:", e)
            return []

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("medical-chatbot")

retriever = CustomPineconeRetriever(
    index=pinecone_index,
    embeddings=embedding,
    k=3
)

chatModel = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

classifierModel = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0
)


# ─────────────────────────────────────────────────────────────
# Prompt Builder
# ─────────────────────────────────────────────────────────────

def build_prompt(history_text, user_memory, user=None):

    if user is None:
        user = current_user if current_user.is_authenticated else None
    user_name  = user.name if user else "User"
    first_name = user_name.split()[0] if user_name else "User"

    history_part = f"Conversation so far:\n{history_text}\n" if history_text else ""

    system_prompt = (
        f"You are MediAssist, a friendly and knowledgeable "
        f"medical assistant — like a doctor friend who gives "
        f"clear, direct answers without unnecessary formality.\n\n"
        f"The user's name is {first_name}. "
        f"Use their name occasionally, not in every message.\n\n"
        f"{history_part}"
        f"Rules:\n"
        f"- Be natural and conversational.\n"
        f"- Never repeat or summarize what the user just said.\n"
        f"- For greetings or small talk, reply briefly and warmly.\n"
        f"- Don't ask multiple questions at once.\n"
        f"- For medical questions, give a clear direct answer first.\n"
        f"- Then add context if needed.\n"
        f"- Only use the retrieved context if genuinely relevant.\n"
        f"- If you don't know something, say so simply.\n"
        f"- Never invent medical facts.\n"
        f"- Never explain your own reasoning.\n"
        f"- Keep answers focused and concise.\n\n"
        f"User Memory:\n{user_memory}\n\n"
        f"Context:\n{{context}}"
    )

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])


# ─────────────────────────────────────────────────────────────
# Database Initialization 
# ─────────────────────────────────────────────────────────────

with app.app_context():
    db.create_all()


# ─────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────

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
    if chat_session.title == "New Consultation":
        try:
            title_prompt = (
                f"Generate a short 4-6 word title for a chat that starts with: '{first_message}'. "
                f"Return ONLY the title, no quotes, no punctuation at the end."
            )
            response = chatModel.invoke(title_prompt)
            title = response.content.strip()[:60]
            chat_session.title = title if title else first_message[:50]
        except Exception:
            chat_session.title = first_message[:50]


def update_memory_in_background(app_instance, user_id, latest_message, history_text):
    with app_instance.app_context():
        from research.src.auth import User, db
        try:
            user = User.query.get(user_id)
            if user:
                update_user_memory(user, chatModel, latest_message, history_text)
                db.session.commit()
                print(f"[BG Memory Update] Completed for user {user_id}")
        except Exception as e:
            db.session.rollback()
            print(f"[BG Memory Update] Conflict or error: {e}")


def update_title_in_background(app_instance, session_id, first_message):
    with app_instance.app_context():
        from research.src.auth import ChatSession, db
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
        response = chatModel.invoke(summary_prompt)
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


def generate_voice_response(msg, user=None):
    if user is None:
        intent = classify_intent(chatModel, msg)
        dynamic_prompt = build_prompt("", "Voice Session")

        if intent == "medical_query":
            question_answer_chain = create_stuff_documents_chain(chatModel, dynamic_prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            response = rag_chain.invoke({"input": msg})
            return response.get("answer", "Sorry, I couldn't generate a response.")

        elif intent == "greeting":
            return chatModel.invoke(f"Reply naturally to: {msg}").content

        else:
            return chatModel.invoke(msg).content

    # With user context
    intent = classify_intent(chatModel, msg)

    # Save user message to database
    chat_session = get_active_session_for_user(user.id)
    user_msg = Message(session_id=chat_session.id, role="user", content=msg)
    db.session.add(user_msg)
    db.session.commit()

    update_user_memory(user, chatModel, msg)
    db.session.commit()

    update_session_title(chat_session, msg)

    history_text  = build_history_text(chat_session)
    user_memory   = get_user_memory(user)
    dynamic_prompt = build_prompt(history_text, user_memory, user=user)

    if intent == "medical_query":
        question_answer_chain = create_stuff_documents_chain(chatModel, dynamic_prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response  = rag_chain.invoke({"input": msg})
        answer    = response.get("answer", "Sorry, I couldn't generate a response.")

    elif intent == "memory_recall":
        memory_prompt = f"User Memory:\n{user_memory}\n\nConversation History:\n{history_text}\n\nUser:\n{msg}"
        answer = chatModel.invoke(memory_prompt).content

    elif intent == "greeting":
        answer = chatModel.invoke(f"Reply naturally to: {msg}").content

    elif intent == "account_action":
        answer = "Please use the account controls available in the application."

    elif intent == "general_chat":
        general_prompt = f"Conversation History:\n{history_text}\n\nUser:\n{msg}"
        answer = chatModel.invoke(general_prompt).content

    else:
        answer = chatModel.invoke(msg).content

    bot_msg = Message(session_id=chat_session.id, role="assistant", content=answer)
    db.session.add(bot_msg)

    chat_session.updated_at = datetime.now(timezone.utc)
    db.session.commit()

    return answer


# ─────────────────────────────────────────────────────────────
# Routes — Pages
# ─────────────────────────────────────────────────────────────

@app.route("/")
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


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        email    = request.form["email"]
        password = request.form["password"]
        user     = User.query.filter_by(email=email).first()

        if user and user.password_hash and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for("index"))

        return render_template("login.html", error="Invalid email or password")

    success = None
    if request.args.get("reset") == "success":
        success = "Password reset successful. Please sign in with your new password."

    return render_template("login.html", success=success)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email    = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form.get("confirm_password")

        if password != confirm_password:
            return render_template("signup.html", error="Passwords do not match.")

        if len(password) < 8:
            return render_template("signup.html", error="Password must be at least 8 characters long.")

        if User.query.filter_by(email=email).first():
            return render_template("signup.html", error="Email already exists")

        user = User(
            email=email,
            name=email.split("@")[0],
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for("index"))

    return render_template("signup.html")


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        user  = User.query.filter_by(email=email).first()

        if not user:
            return render_template(
                "forgot_password.html",
                error="No MediAssist account exists for that email.",
                email=email,
            )

        otp = f"{secrets.randbelow(900000) + 100000}"
        user.reset_otp  = otp
        user.otp_expiry = datetime.utcnow() + timedelta(minutes=10)
        db.session.commit()

        msg      = MailMessage(
            subject="MediAssist Password Reset Verification",
            sender="parthtyagi3389@gmail.com",
            recipients=[email]
        )
        msg.body = f"""
🩺 MediAssist Password Reset

Hello,

We received a request to reset the password associated with your MediAssist account.

━━━━━━━━━━━━━━━━━━
RESET OTP: {otp}
━━━━━━━━━━━━━━━━━━

This code will expire in 10 minutes.

For your security:
• Never share this OTP with anyone.
• MediAssist will never ask for your OTP.
• If you did not request this reset, simply ignore this email.

Thank you for choosing MediAssist.

Best Regards,
MediAssist Team
Your AI Health Companion
"""
        try:
            mail.send(msg)
            print("EMAIL SENT SUCCESSFULLY")
        except Exception as e:
            print("EMAIL ERROR:", e)
            user.reset_otp = None
            user.otp_expiry = None
            db.session.commit()
            return render_template(
                "forgot_password.html",
                error="Could not send OTP right now. Please check mail settings and try again.",
                email=email,
            )

        session["reset_email"] = email
        session["reset_otp_verified"] = False
        return redirect(url_for("verify_otp", email=email))

    return render_template(
        "forgot_password.html",
        email=request.args.get("email", "").strip().lower(),
    )


@app.route("/verify-otp/<email>", methods=["GET", "POST"])
def verify_otp(email):
    email = email.strip().lower()
    user = User.query.filter_by(email=email).first()
    if not user:
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        otp = request.form["otp"].strip()

        if not user.reset_otp or not user.otp_expiry:
            return render_template(
                "verify_otp.html",
                email=email,
                error="No active OTP found. Please request a new code.",
            )

        if user.otp_expiry <= datetime.utcnow():
            user.reset_otp = None
            user.otp_expiry = None
            db.session.commit()
            return render_template(
                "verify_otp.html",
                email=email,
                error="OTP expired. Please request a new code.",
            )

        if secrets.compare_digest(user.reset_otp, otp):
            session["reset_email"] = email
            session["reset_otp_verified"] = True
            return redirect(url_for("reset_password", email=email))

        return render_template("verify_otp.html", email=email, error="Invalid OTP.")

    return render_template("verify_otp.html", email=email)


@app.route("/reset-password/<email>", methods=["GET", "POST"])
def reset_password(email):
    email = email.strip().lower()
    user = User.query.filter_by(email=email).first()
    if (
        not user
        or session.get("reset_email") != email
        or session.get("reset_otp_verified") is not True
    ):
        return redirect(url_for("forgot_password"))

    if not user.otp_expiry or user.otp_expiry <= datetime.utcnow():
        user.reset_otp = None
        user.otp_expiry = None
        db.session.commit()
        session.pop("reset_email", None)
        session.pop("reset_otp_verified", None)
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        password         = request.form["password"].strip()
        confirm_password = request.form["confirm_password"].strip()

        if password != confirm_password:
            return render_template(
                "reset_password.html",
                email=email,
                error="Passwords do not match.",
            )

        if len(password) < 8:
            return render_template(
                "reset_password.html",
                email=email,
                error="Password must be at least 8 characters long.",
            )

        user.password_hash  = generate_password_hash(password)
        user.reset_otp      = None
        user.otp_expiry     = None
        db.session.commit()

        session.pop("reset_email", None)
        session.pop("reset_otp_verified", None)
        if current_user.is_authenticated:
            logout_user()
        return redirect(url_for("login", reset="success"))

    return render_template("reset_password.html", email=email)


@app.route("/google_auth")
def google_auth():
    if not google.authorized:
        return redirect(url_for("google.login"))

    try:
        resp = google.get("/oauth2/v2/userinfo")
        if not resp.ok:
            return f"Google API error: {resp.text}", 400

        info  = resp.json()
        email = info.get("email")
        if not email:
            return "No email returned from Google.", 400

        user = User.query.filter_by(email=email).first()
        if not user:
            user = User(
                email=email,
                name=info.get("name", "User"),
                avatar=info.get("picture", "")
            )
            db.session.add(user)
            db.session.commit()

        login_user(user)
        return redirect(url_for("index"))

    except Exception as e:
        with open("error.log", "a") as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        return f"Something went wrong during login: {e}", 500


@app.route("/logout")
@login_required
def logout():
    session_id = session.get("chat_session_id")
    if session_id:
        chat_session = ChatSession.query.get(session_id)
        if chat_session:
            summarize_session(chat_session)

    session.pop("chat_session_id", None)
    logout_user()
    return redirect(url_for("login"))


# ─────────────────────────────────────────────────────────────
# Routes — Chat
# ─────────────────────────────────────────────────────────────

@app.route("/new_chat", methods=["POST"])
@login_required
def new_chat():
    session_id = session.get("chat_session_id")
    if session_id:
        old_session = ChatSession.query.get(session_id)
        if old_session:
            summarize_session(old_session)

    session.pop("chat_session_id", None)
    return jsonify({"status": "ok"})


@app.route("/load_session/<int:session_id>", methods=["GET"])
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


@app.route("/get", methods=["POST"])
@login_required
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "Please enter a message."

    intent = classify_intent(classifierModel, msg)
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

        # Trigger background memory update asynchronously (context-aware)
        threading.Thread(
            target=update_memory_in_background,
            args=(app, current_user.id, msg, history_text),
            daemon=True
        ).start()

        # Trigger background title update asynchronously
        if chat_session.title == "New Consultation":
            threading.Thread(
                target=update_title_in_background,
                args=(app, chat_session.id, msg),
                daemon=True
            ).start()

        dynamic_prompt = build_prompt(history_text, user_memory)

        if intent == "medical_query":
            question_answer_chain = create_stuff_documents_chain(chatModel, dynamic_prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            response  = rag_chain.invoke({"input": msg})
            answer    = response.get("answer", "Sorry, I couldn't generate a response.")

        elif intent == "memory_recall":
            memory_prompt = f"User Memory:\n{user_memory}\n\nConversation History:\n{history_text}\n\nUser:\n{msg}"
            answer = chatModel.invoke(memory_prompt).content

        elif intent == "greeting":
            answer = chatModel.invoke(f"Reply naturally to: {msg}").content

        elif intent == "account_action":
             answer = "Please use the account controls available in the application."

        elif intent == "general_chat":
            general_prompt = f"Conversation History:\n{history_text}\n\nUser:\n{msg}"
            answer = chatModel.invoke(general_prompt).content

        else:
            answer = chatModel.invoke(msg).content

        bot_msg = Message(session_id=chat_session.id, role="assistant", content=answer)
        db.session.add(bot_msg)

        chat_session.updated_at = datetime.now(timezone.utc)
        db.session.commit()

        return answer

    except Exception:
        traceback.print_exc()
        return "Something went wrong."


@app.route("/delete_session/<int:session_id>", methods=["POST"])
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


@app.route("/get_memory", methods=["GET"])
@login_required
def get_user_memory_route():
    from research.src.memory import get_user_memory
    return jsonify({"memory": get_user_memory(current_user)})


@app.route("/clear_memory", methods=["POST"])
@login_required
def clear_user_memory_route():
    from research.src.memory import clear_user_memory
    try:
        clear_user_memory(current_user)
        db.session.commit()
        return jsonify({"status": "ok"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/tts", methods=["POST"])
@login_required
def tts():
    text = request.form.get("text", "")
    filename = text_to_speech(text)
    return jsonify({"audio_url": f"/{filename}"})


# ─────────────────────────────────────────────────────────────
# Routes — Voice (LiveKit)
# ─────────────────────────────────────────────────────────────

@app.route("/voice_chat", methods=["POST"])
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
        except Exception:
            traceback.print_exc()
            answer = "Sorry, something went wrong on my end."
        return jsonify({"response": answer})

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
                    args=(app, user.id, msg, build_history_text(chat_session)),
                    daemon=True
                ).start()

                # Trigger background title update asynchronously
                if chat_session.title == "New Consultation":
                    threading.Thread(
                        target=update_title_in_background,
                        args=(app, chat_session.id, msg),
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


@app.route("/livekit_token", methods=["GET"])
@login_required
def livekit_token_route():
    """Returns a LiveKit JWT so the browser can join a private voice room.

    FIX: A fresh timestamp is appended to the room name on every call so
    that cancelled/disconnected sessions never collide with stale agent
    dispatches from a previous voice session.
    """
    try:
        # ✅ FIX 1: Fresh room name per session — timestamp prevents stale
        # agent dispatches from a previous (cancelled) voice session being
        # reused for a brand-new connection.
        room_name = f"mediassist-voice-{current_user.id}-{int(time.time())}"

        token = generate_livekit_token(
            room_name=room_name,
            participant_identity=str(current_user.id),
            participant_name=current_user.name,
            dispatch_agent=False,
            agent_metadata=str(current_user.id),
        )

        livekit_url = os.getenv("LIVEKIT_URL", "wss://localhost:7880")

        return jsonify({
            "token":   token,
            "url":     livekit_url,
            "room":    room_name,
            "user_id": current_user.id,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "token": None}), 500


@app.route("/dispatch_agent", methods=["POST"])
@login_required
def dispatch_agent():
    """Dispatches the medical-agent worker into the LiveKit room."""
    import asyncio

    data      = request.get_json(silent=True) or {}
    room_name = data.get("room") or f"mediassist-voice-{current_user.id}-{int(time.time())}"
    user_id   = str(current_user.id)

    livekit_url        = os.getenv("LIVEKIT_URL",        "").strip()
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

            # ✅ FIX 2: Only treat a job as active when it is genuinely
            # RUNNING (status == 1) AND has actually started (started_at > 0)
            # AND has not ended yet.  Previously status == 0 (PENDING) was
            # also counted as active, which caused zombie dispatches from a
            # cancelled session to block a fresh one from ever being created.
            def has_active_job(dispatch) -> bool:
                state = getattr(dispatch, "state", None)
                if not state or getattr(state, "deleted_at", 0):
                    return False

                for job in getattr(state, "jobs", []):
                    job_state  = getattr(job, "state", None)
                    status     = getattr(job_state, "status",     None)
                    ended_at   = getattr(job_state, "ended_at",   0)
                    started_at = getattr(job_state, "started_at", 0)
                    # status 1 == RUNNING; must have started and not ended
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
                            metadata=user_id,
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


@app.route("/debug_oauth")
def debug_oauth():
    from flask import url_for
    uri = url_for("google.authorized", _external=True)
    return f"Flask-Dance is using this redirect URI: <b>{uri}</b>"

# ─────────────────────────────────────────────────────────────
# Run App
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Start local LiveKit worker in dev mode
    import subprocess
    import sys
    print("Local startup: Launching LiveKit agent worker in background...")
    try:
        subprocess.Popen([sys.executable, "voice_worker.py", "dev"])
    except Exception as e:
        print(f"Failed to start local voice worker: {e}")

    app.run(host="localhost", port=5050, debug=True)