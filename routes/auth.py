import secrets
import threading
import traceback
from datetime import datetime, timezone, timedelta

from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    session,
    current_app
)
from flask_mail import Message as MailMessage
from flask_dance.contrib.google import google
from flask_login import (
    login_required,
    logout_user,
    current_user,
    login_user
)
from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from research.src.auth import db, User, ChatSession
from services.email_service import send_async_email
from services.chat_service import summarize_session

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/login", methods=["GET", "POST"], endpoint="login")
def login():
    if current_user.is_authenticated:
        return redirect(url_for("chat.index"))

    if request.method == "POST":
        email    = request.form["email"]
        password = request.form["password"]
        user     = User.query.filter_by(email=email).first()

        if user and user.password_hash and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for("chat.index"))

        return render_template("login.html", error="Invalid email or password")

    success = None
    if request.args.get("reset") == "success":
        success = "Password reset successful. Please sign in with your new password."

    return render_template("login.html", success=success)


@auth_bp.route("/signup", methods=["GET", "POST"], endpoint="signup")
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
        return redirect(url_for("chat.index"))

    return render_template("signup.html")


@auth_bp.route("/forgot-password", methods=["GET", "POST"], endpoint="forgot_password")
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
        user.otp_expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=10)
        db.session.commit()

        msg      = MailMessage(
            subject="MediAssist Password Reset Verification",
            sender="parthtyagi3389@gmail.com",
            recipients=[email]
        )
        msg.html = f"""
        <div style="font-family: sans-serif; max-width: 500px; margin: 0 auto; color: #333;">
            <h2 style="color: #2b5a8e;">🩺 MediAssist Password Reset</h2>
            <p>Hello,</p>
            <p>We received a request to reset the password associated with your MediAssist account.</p>
            <div style="background-color: #f4f7f6; padding: 15px; text-align: center; font-size: 24px; font-weight: bold; letter-spacing: 2px; border-radius: 8px; margin: 20px 0;">
                {otp}
            </div>
            <p>This code will expire in 10 minutes.</p>
            <p style="font-size: 12px; color: #777;">If you did not request this reset, simply ignore this email.</p>
            <p>Best Regards,<br>MediAssist Team</p>
        </div>
        """

        app_obj = current_app._get_current_object()
        from app import mail
        threading.Thread(target=send_async_email, args=(app_obj, msg, mail), daemon=True).start()
        
        session["reset_email"] = email
        session["reset_otp_verified"] = False
        return redirect(url_for("auth.verify_otp", email=email))

    return render_template(
        "forgot_password.html",
        email=request.args.get("email", "").strip().lower(),
    )


@auth_bp.route("/verify-otp/<email>", methods=["GET", "POST"], endpoint="verify_otp")
def verify_otp(email):
    email = email.strip().lower()
    user = User.query.filter_by(email=email).first()
    if not user:
        return redirect(url_for("auth.forgot_password"))

    if request.method == "POST":
        otp = request.form["otp"].strip()

        if not user.reset_otp or not user.otp_expiry:
            return render_template(
                "verify_otp.html",
                email=email,
                error="No active OTP found. Please request a new code.",
            )

        if user.otp_expiry <= datetime.now(timezone.utc).replace(tzinfo=None):
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
            return redirect(url_for("auth.reset_password", email=email))

        return render_template("verify_otp.html", email=email, error="Invalid OTP.")

    return render_template("verify_otp.html", email=email)


@auth_bp.route("/reset-password/<email>", methods=["GET", "POST"], endpoint="reset_password")
def reset_password(email):
    email = email.strip().lower()
    user = User.query.filter_by(email=email).first()
    if (
        not user
        or session.get("reset_email") != email
        or session.get("reset_otp_verified") is not True
    ):
        return redirect(url_for("auth.forgot_password"))

    if not user.otp_expiry or user.otp_expiry <= datetime.now(timezone.utc).replace(tzinfo=None):
        user.reset_otp = None
        user.otp_expiry = None
        db.session.commit()
        session.pop("reset_email", None)
        session.pop("reset_otp_verified", None)
        return redirect(url_for("auth.forgot_password"))

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
        return redirect(url_for("auth.login", reset="success"))

    return render_template("reset_password.html", email=email)


@auth_bp.route("/google_auth", endpoint="google_auth")
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
        return redirect(url_for("chat.index"))

    except Exception as e:
        with open("error.log", "a") as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        return f"Something went wrong during login: {e}", 500


@auth_bp.route("/logout", endpoint="logout")
@login_required
def logout():
    session_id = session.get("chat_session_id")
    if session_id:
        chat_session = ChatSession.query.get(session_id)
        if chat_session:
            summarize_session(chat_session)

    session.pop("chat_session_id", None)
    logout_user()
    return redirect(url_for("auth.login"))


@auth_bp.route("/debug_oauth", endpoint="debug_oauth")
def debug_oauth():
    uri = url_for("google.authorized", _external=True)
    return f"Flask-Dance is using this redirect URI: <b>{uri}</b>"
