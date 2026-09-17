"""
Task Dispatcher Module (Decoupled Asynchronous Reliability)
============================================================
Pillars Implemented:
- Phase A: Enqueueing durable background jobs to Redis Streams (medical-background-tasks)
- Phase C: Idempotency Key deduplication preventing redundant or duplicate job execution
- Graceful Degradation: Handles missing or offline Redis broker without crashing HTTP routes
"""

import os
import json
import time
import hashlib
import logging
from typing import Optional, Dict, Any

try:
    import redis
except ImportError:
    redis = None

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "").strip()
TASK_STREAM = os.getenv("TASK_STREAM", "medical-background-tasks")
DLQ_STREAM = os.getenv("DLQ_STREAM", "medical-tasks-dlq")
CONSUMER_GROUP = os.getenv("TASK_CONSUMER_GROUP", "medical-task-workers")

_redis_client: Optional[Any] = None
_redis_available: Optional[bool] = None


def get_redis_client() -> Optional[Any]:
    """
    Returns a singleton Redis client instance, or None if Redis is not installed,
    not configured, or the server is unreachable.
    """
    global _redis_client, _redis_available
    if _redis_client is not None:
        return _redis_client

    if not REDIS_URL:
        _redis_available = False
        return None

    if redis is None:
        logger.warning("[TaskDispatcher] 'redis' library is not installed.")
        _redis_available = False
        return None

    try:
        client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        # Verify connection with a short timeout ping
        client.ping()
        _redis_client = client
        _redis_available = True
        return _redis_client
    except Exception as exc:
        logger.warning(f"[TaskDispatcher] Could not connect to Redis at {REDIS_URL}: {exc}")
        _redis_available = False
        return None


def generate_idempotency_key(task_type: str, **kwargs) -> str:
    """
    Generates a deterministic idempotency key for a given task type and parameters.
    
    Examples:
        - Memory update: idemp:mem:{user_id}:{sha256_of_msg}
        - Title update:  idemp:title:{session_id}
        - Summary:       idemp:summary:{session_id}:{msg_count}
    """
    if task_type == "UPDATE_MEMORY":
        user_id = kwargs.get("user_id", "0")
        msg = str(kwargs.get("latest_message", "")).strip()
        msg_hash = hashlib.sha256(msg.encode("utf-8")).hexdigest()[:16]
        return f"idemp:mem:{user_id}:{msg_hash}"
    elif task_type == "UPDATE_TITLE":
        session_id = kwargs.get("session_id", "0")
        return f"idemp:title:{session_id}"
    elif task_type == "SESSION_SUMMARIZE":
        session_id = kwargs.get("session_id", "0")
        msg_count = kwargs.get("msg_count", "0")
        return f"idemp:summary:{session_id}:{msg_count}"
    elif task_type == "EVAL_SAFETY_CHECK":
        msg_id = kwargs.get("message_id", "0")
        return f"idemp:eval_safety:{msg_id}"
    elif task_type == "EVAL_TURN":
        msg_id = kwargs.get("message_id", "0")
        return f"idemp:eval:{msg_id}"
    else:
        raw_str = f"{task_type}:{sorted(kwargs.items())}"
        digest = hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:16]
        return f"idemp:generic:{task_type}:{digest}"


def enqueue_task(
    task_type: str,
    payload: Dict[str, Any],
    idempotency_key: Optional[str] = None,
    ttl: int = 300,
    client_override: Optional[Any] = None
) -> bool:
    """
    Publishes a background task to the Redis Stream with idempotency deduplication.
    
    Args:
        task_type: The task identifier (e.g., 'UPDATE_MEMORY', 'UPDATE_TITLE', 'SESSION_SUMMARIZE')
        payload: Dict of arguments needed by the background worker.
        idempotency_key: Key used to prevent duplicate enqueues. If omitted, one is generated.
        ttl: Time-to-live for the idempotency key in seconds (default 300s).
        client_override: Optional Redis client for dependency injection / testing.
        
    Returns:
        bool: True if task was successfully enqueued, False if deduplicated or broker offline.
    """
    client = client_override if client_override is not None else get_redis_client()

    # 1. Generate idempotency key if not explicitly provided (Phase C)
    if not idempotency_key:
        idempotency_key = generate_idempotency_key(task_type, **payload)

    # 2. Redis available path: Apply Phase C (SET NX) and Phase A (XADD)
    if client is not None:
        try:
            # Phase C: Idempotency filter using Redis SET NX
            set_success = client.set(idempotency_key, "PENDING", ex=ttl, nx=True)
            if not set_success:
                print(f"[TaskDispatcher] [IDEMPOTENT_DROP] Duplicate task filtered: {idempotency_key}")
                return False

            # Phase A: Durable persistence via Redis Stream
            stream_entry = {
                "task_type": task_type,
                "payload": json.dumps(payload),
                "idempotency_key": idempotency_key,
                "retry_count": "0",
                "enqueued_at": str(time.time()),
            }
            msg_id = client.xadd(TASK_STREAM, stream_entry)
            print(f"[TaskDispatcher] [STREAM_ENQUEUE] Task {task_type} enqueued as {msg_id} (key: {idempotency_key})")
            return True
        except Exception as exc:
            logger.error(f"[TaskDispatcher] Redis stream enqueue failed: {exc}")
            # Fall through to fallback below

    # 3. Fallback path (Phase A Graceful Degradation)
    # When Redis server is absent (e.g. testing or local dev without Redis instance),
    # log warning and execute inline or background thread fallback so app never crashes.
    print(f"[TaskDispatcher] [FALLBACK] Redis not connected; executing fallback for {task_type}")
    return _execute_inline_fallback(task_type, payload)


def _execute_inline_fallback(task_type: str, payload: Dict[str, Any]) -> bool:
    """
    Executes task synchronously when Redis is not running.
    Ensures developer local experience remains fully functional out-of-the-box.
    """
    try:
        from services.chat_service import (
            process_memory_update,
            process_title_update,
            process_session_summarize
        )
        if task_type == "UPDATE_MEMORY":
            process_memory_update(
                user_id=payload.get("user_id"),
                latest_message=payload.get("latest_message", ""),
                history_text=payload.get("history_text", "")
            )
        elif task_type == "UPDATE_TITLE":
            process_title_update(
                session_id=payload.get("session_id"),
                first_message=payload.get("first_message", "")
            )
        elif task_type == "SESSION_SUMMARIZE":
            process_session_summarize(
                session_id=payload.get("session_id")
            )
        elif task_type == "EVAL_SAFETY_CHECK":
            from services.eval_service import run_safety_eval
            run_safety_eval(payload)
        elif task_type == "EVAL_TURN":
            from services.eval_service import run_turn_eval
            run_turn_eval(payload)
        return True
    except Exception as exc:
        logger.error(f"[TaskDispatcher] Fallback execution failed: {exc}")
        return False


# Convenience dispatchers
def enqueue_memory_update(user_id: int, latest_message: str, history_text: str) -> bool:
    """Convenience helper to enqueue user memory update."""
    payload = {
        "user_id": user_id,
        "latest_message": latest_message,
        "history_text": history_text
    }
    return enqueue_task("UPDATE_MEMORY", payload, ttl=180)


def enqueue_title_update(session_id: int, first_message: str) -> bool:
    """Convenience helper to enqueue session title generation."""
    payload = {
        "session_id": session_id,
        "first_message": first_message
    }
    return enqueue_task("UPDATE_TITLE", payload, ttl=3600)


def enqueue_session_summarize(session_id: int, msg_count: int) -> bool:
    """Convenience helper to enqueue context window summarization."""
    payload = {
        "session_id": session_id,
        "msg_count": msg_count
    }
    return enqueue_task("SESSION_SUMMARIZE", payload, ttl=600)


def enqueue_eval_safety_check(
    message_id: int,
    query: str,
    response: str,
    patient_state: Optional[Dict[str, Any]] = None,
    red_flag_scenario: Optional[str] = None,
    expected_tier: Optional[str] = None
) -> bool:
    """Convenience helper to enqueue deterministic rule-based safety check on every turn."""
    payload = {
        "message_id": message_id,
        "query": query,
        "response": response,
        "patient_state": patient_state,
        "red_flag_scenario": red_flag_scenario,
        "expected_tier": expected_tier
    }
    return enqueue_task("EVAL_SAFETY_CHECK", payload, ttl=120)


def enqueue_eval_turn(
    message_id: int,
    query: str,
    generated_answer: str,
    retrieved_chunks: list,
    patient_state: Optional[Dict[str, Any]] = None
) -> bool:
    """Convenience helper to enqueue sampled online shadow evaluation."""
    payload = {
        "message_id": message_id,
        "query": query,
        "generated_answer": generated_answer,
        "retrieved_chunks": retrieved_chunks,
        "patient_state": patient_state
    }
    return enqueue_task("EVAL_TURN", payload, ttl=120)

