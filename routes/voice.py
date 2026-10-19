import os
import time
import hmac
import hashlib
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
from research.src.auth import User
from services.chat_service import generate_voice_response
from services.chat_pipeline import run_turn

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


def verify_voice_hmac(req) -> tuple[bool, str]:
    """
    Verifies that the incoming request to /voice_chat was signed by voice_worker
    using the shared secret VOICE_INTERNAL_SECRET.
    Validates timestamp freshness (<=60s) and bounds clock skew (<=5s).
    Compares raw bytes with hmac.compare_digest and fails closed in production.
    """
    secret = os.getenv("VOICE_INTERNAL_SECRET")
    is_dev = os.getenv("FLASK_ENV", "").strip().lower() == "development"
    if not secret:
        if is_dev:
            secret = "dev-voice-internal-secret-for-local-testing"
        else:
            return False, "VOICE_INTERNAL_SECRET unset in non-development mode"

    ts_str = req.headers.get("X-Voice-Timestamp")
    sig = req.headers.get("X-Voice-Signature")
    if not ts_str or not sig:
        return False, "Missing HMAC signature or timestamp headers"

    try:
        ts = float(ts_str)
        now = time.time()
        if (now - ts) > 60 or (ts - now) > 5:
            return False, "Signature timestamp expired or invalid clock skew"
    except (ValueError, TypeError):
        return False, "Invalid timestamp format"

    try:
        raw_body = req.get_data()
        data = req.get_json(silent=True) or {}
        user_id = data.get("user_id")
        user_id_str = str(user_id) if user_id is not None else ""
        canonical = f"{ts_str}:{req.path}:{user_id_str}:".encode("utf-8") + raw_body
        expected = hmac.new(secret.encode("utf-8"), canonical, hashlib.sha256).hexdigest()

        # Compare signatures as bytes; catch any TypeError/ValueError and return 401
        if not hmac.compare_digest(sig.strip().encode("utf-8"), expected.encode("utf-8")):
            return False, "Signature mismatch"
    except Exception as e:
        return False, f"Signature verification error: {type(e).__name__}"

    return True, "Valid"


@voice_bp.route("/voice_chat", methods=["POST"], endpoint="voice_chat")
def voice_chat():
    """Called by voice_worker.py — protected via HMAC request signing."""
    is_valid, reason = verify_voice_hmac(request)
    if not is_valid:
        current_app.logger.warning("[voice_chat] HMAC verification failed: %s", reason)
        return jsonify({"error": "Unauthorized", "detail": reason}), 401

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
        except Exception as e:
            current_app.logger.warning("Failed to fetch user by id in voice_chat: %s", type(e).__name__)

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

    from services.chat_pipeline import run_turn
    return Response(
        stream_with_context(run_turn(msg, user=user, channel="voice_stream")),
        mimetype="text/plain"
    )


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
