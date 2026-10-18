import os
import sys
import time
import json
import hmac
import hashlib
import subprocess
import pytest
from unittest.mock import patch

from app import app, db
from research.src.auth import User


@pytest.fixture(autouse=True)
def setup_test_db():
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["WTF_CSRF_ENABLED"] = False
    app.config["SERVER_NAME"] = None

    with app.app_context():
        db.create_all()
        yield
        db.session.remove()
        db.drop_all()


def test_production_startup_without_secret_key_raises_runtime_error():
    """Verify fail-closed behavior: app startup in production raises RuntimeError if SECRET_KEY is unset."""
    code = (
        "import sys, os, dotenv;"
        "dotenv.load_dotenv = lambda *a, **k: None;"
        "os.environ.pop('SECRET_KEY', None);"
        "os.environ['FLASK_ENV'] = 'production';"
        "os.environ.pop('TESTING', None);"
        "import app"
    )
    cmd = [
        sys.executable,
        "-c",
        code
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "CRITICAL SECURITY CONFIGURATION ERROR: SECRET_KEY" in res.stderr


def test_debug_oauth_gated_behind_development():
    """Verify /debug_oauth returns 404 when FLASK_ENV is not development."""
    client = app.test_client()

    with patch.dict(os.environ, {"FLASK_ENV": "production"}):
        app.config["DEBUG"] = False
        resp = client.get("/debug_oauth")
        assert resp.status_code == 404

    with patch.dict(os.environ, {"FLASK_ENV": "development"}):
        resp = client.get("/debug_oauth")
        assert resp.status_code == 200
        assert b"Flask-Dance is using this redirect URI" in resp.data


def test_voice_chat_rejects_unsigned_request():
    """Verify /voice_chat rejects request without HMAC headers with 401."""
    client = app.test_client()
    resp = client.post(
        "/voice_chat",
        data=json.dumps({"message": "Hello doctor"}),
        content_type="application/json"
    )
    assert resp.status_code == 401
    data = resp.get_json()
    assert data["error"] == "Unauthorized"
    assert "Missing HMAC signature" in data["detail"]


def test_voice_chat_rejects_forged_signature():
    """Verify /voice_chat rejects request with invalid signature with 401."""
    client = app.test_client()
    ts = str(int(time.time()))
    resp = client.post(
        "/voice_chat",
        data=json.dumps({"message": "Hello doctor"}),
        content_type="application/json",
        headers={
            "X-Voice-Timestamp": ts,
            "X-Voice-Signature": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        }
    )
    assert resp.status_code == 401
    data = resp.get_json()
    assert data["error"] == "Unauthorized"
    assert "Signature mismatch" in data["detail"]


def test_voice_chat_rejects_stale_timestamp():
    """Verify /voice_chat rejects signatures older than 60s with 401."""
    client = app.test_client()
    secret = os.getenv("VOICE_INTERNAL_SECRET", "test-voice-secret-key-for-test-suites-32chars")
    stale_ts = str(int(time.time()) - 120)  # 2 minutes ago
    body_bytes = json.dumps({"message": "Hello doctor"}, separators=(',', ':')).encode("utf-8")
    canonical = f"{stale_ts}:/voice_chat::".encode("utf-8") + body_bytes
    sig = hmac.new(secret.encode("utf-8"), canonical, hashlib.sha256).hexdigest()

    resp = client.post(
        "/voice_chat",
        data=body_bytes,
        content_type="application/json",
        headers={
            "X-Voice-Timestamp": stale_ts,
            "X-Voice-Signature": sig
        }
    )
    assert resp.status_code == 401
    data = resp.get_json()
    assert "expired or invalid clock skew" in data["detail"]


def test_voice_chat_rejects_future_clock_skew():
    """Verify /voice_chat rejects signatures more than 5s in the future with 401."""
    client = app.test_client()
    secret = os.getenv("VOICE_INTERNAL_SECRET", "test-voice-secret-key-for-test-suites-32chars")
    future_ts = str(int(time.time()) + 30)  # 30 seconds ahead
    body_bytes = json.dumps({"message": "Hello doctor"}, separators=(',', ':')).encode("utf-8")
    canonical = f"{future_ts}:/voice_chat::".encode("utf-8") + body_bytes
    sig = hmac.new(secret.encode("utf-8"), canonical, hashlib.sha256).hexdigest()

    resp = client.post(
        "/voice_chat",
        data=body_bytes,
        content_type="application/json",
        headers={
            "X-Voice-Timestamp": future_ts,
            "X-Voice-Signature": sig
        }
    )
    assert resp.status_code == 401
    data = resp.get_json()
    assert "expired or invalid clock skew" in data["detail"]


def test_voice_chat_accepts_valid_signed_request():
    """Verify /voice_chat successfully authenticates valid HMAC signed request."""
    client = app.test_client()
    secret = os.getenv("VOICE_INTERNAL_SECRET", "test-voice-secret-key-for-test-suites-32chars")
    ts = str(int(time.time()))
    payload = {"message": "Hello doctor", "user_id": 42}
    body_bytes = json.dumps(payload, separators=(',', ':')).encode("utf-8")
    canonical = f"{ts}:/voice_chat:42:".encode("utf-8") + body_bytes
    sig = hmac.new(secret.encode("utf-8"), canonical, hashlib.sha256).hexdigest()

    with patch("routes.voice.generate_voice_response", return_value="Hello, how can I assist your health today?"):
        resp = client.post(
            "/voice_chat",
            data=body_bytes,
            content_type="application/json",
            headers={
                "X-Voice-Timestamp": ts,
                "X-Voice-Signature": sig
            }
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["response"] == "Hello, how can I assist your health today?"


def test_signup_validation_and_anti_enumeration():
    """Verify signup enforces email normalization, format regex, and anti-enumeration message."""
    client = app.test_client()

    # 1. Reject invalid email format
    resp_invalid = client.post("/signup", data={
        "email": "not-an-email",
        "password": "password123",
        "confirm_password": "password123"
    })
    assert resp_invalid.status_code == 200
    assert b"Please provide a valid email address." in resp_invalid.data

    # 2. Successfully register with unnormalized email (trailing spaces, uppercase)
    resp_valid = client.post("/signup", data={
        "email": "  TEST.Patient@Example.COM  ",
        "password": "password123",
        "confirm_password": "password123"
    }, follow_redirects=True)
    assert resp_valid.status_code == 200
    user = User.query.filter_by(email="test.patient@example.com").first()
    assert user is not None

    # 3. Duplicate registration returns generic message without revealing account existence details
    resp_duplicate = client.post("/signup", data={
        "email": "test.patient@example.com",
        "password": "password123",
        "confirm_password": "password123"
    })
    assert resp_duplicate.status_code == 200
    assert b"Unable to complete registration with that email address. Please verify your details or sign in." in resp_duplicate.data


def test_forgot_password_anti_enumeration():
    """Verify forgot-password behaves identically for existing and non-existing accounts."""
    client = app.test_client()

    # Create an existing user
    user = User(email="known.user@example.com", name="known", password_hash="hash")
    db.session.add(user)
    db.session.commit()

    with patch("routes.auth.send_async_email") as mock_send_email:
        # 1. Existing user -> sends email, redirects to /verify-otp
        resp_existing = client.post("/forgot-password", data={"email": "known.user@example.com"})
        assert resp_existing.status_code == 302
        assert resp_existing.headers["Location"].endswith("/verify-otp/known.user@example.com")
        assert mock_send_email.called

        # 2. Non-existent user -> does NOT send email, but ALSO redirects to /verify-otp
        mock_send_email.reset_mock()
        resp_nonexistent = client.post("/forgot-password", data={"email": "unknown.user@example.com"})
        assert resp_nonexistent.status_code == 302
        assert resp_nonexistent.headers["Location"].endswith("/verify-otp/unknown.user@example.com")
        assert not mock_send_email.called
