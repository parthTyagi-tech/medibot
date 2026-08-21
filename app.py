from dotenv import load_dotenv
load_dotenv()

import os
import atexit
import subprocess
import sys

from flask import Flask
from flask_mail import Mail
from werkzeug.middleware.proxy_fix import ProxyFix

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"

from research.src.auth import (
    db,
    login_manager,
    create_google_blueprint
)
from research.src.intent_classifier import classify_intent
from services.ai_service import (
    chatModel,
    classifierModel,
    retriever,
    is_prompt_injection,
    detect_medical_emergency,
    apply_input_guardrails,
    apply_output_guardrails,
    build_prompt
)
from routes import register_routes

# ─────────────────────────────────────────────────────────────
# Flask App Factory / Setup
# ─────────────────────────────────────────────────────────────

app = Flask(__name__)

app.config["MAIL_SERVER"]        = "smtp.gmail.com"
app.config["MAIL_PORT"]          = 587
app.config["MAIL_USE_TLS"]       = True
app.config["MAIL_USE_SSL"]       = False
app.config["MAIL_USERNAME"]      = "parthtyagi3389@gmail.com"
app.config["MAIL_PASSWORD"]      = os.getenv("MAIL_PASSWORD", "ajbb ekwo anvz kdwg")
app.config["MAIL_DEFAULT_SENDER"] = "parthtyagi3389@gmail.com"

mail = Mail(app)

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

app.secret_key = os.getenv("SECRET_KEY", "fallback-secret")

if os.getenv("SERVER_NAME"):
    app.config["SERVER_NAME"] = os.getenv("SERVER_NAME")

db_url = os.getenv("DATABASE_URL", "sqlite:///users.db")
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"]        = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize extensions
db.init_app(app)
login_manager.init_app(app)
login_manager.login_view = "login"

with app.app_context():
    db.create_all()

# Register Blueprints
google_bp = create_google_blueprint()
app.register_blueprint(google_bp, url_prefix="/login")
register_routes(app)

# ─────────────────────────────────────────────────────────────
# Local Application Execution
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Local startup: Launching LiveKit agent worker in background...")
    try:
        proc = subprocess.Popen([sys.executable, "voice_worker.py", "dev"])
        @atexit.register
        def cleanup_worker():
            print("Local shutdown: Terminating LiveKit agent worker...")
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
    except Exception as e:
        print(f"Failed to start local voice worker: {e}")

    app.run(host="localhost", port=5050, debug=True)