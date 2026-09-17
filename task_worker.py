"""
Decoupled Background Task Worker (Redis Streams + DLQ + Retries)
================================================================
Consumes asynchronous tasks from Redis Stream 'medical-background-tasks':
- Processes tasks within Flask application context (memory, title, summary).
- Phase B: Retries failed tasks up to 3 times with exponential backoff.
- Phase B: Routes unrecoverable tasks (exceeding 3 retries) to Dead-Letter Queue ('medical-tasks-dlq').
- Graceful Shutdown: Listens for SIGINT/SIGTERM and completes active tasks cleanly.
"""

import os
import sys
import time
import json
import signal
import socket
import logging
import traceback
from typing import Optional

try:
    import redis
except ImportError:
    redis = None

from app import app
from services.chat_service import (
    process_memory_update,
    process_title_update,
    process_session_summarize,
)
from services.eval_service import (
    run_safety_eval,
    run_turn_eval,
)
from services.task_dispatcher import (
    REDIS_URL,
    TASK_STREAM,
    DLQ_STREAM,
    CONSUMER_GROUP,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [TaskWorker] %(message)s"
)
logger = logging.getLogger("TaskWorker")

# Named retry policy — max_retries per task type.
# Tune here, not in process_message control flow.
RETRY_POLICY = {
    "UPDATE_MEMORY":     {"max_retries": 3, "reason": "DB write, transient failure likely"},
    "UPDATE_TITLE":      {"max_retries": 3, "reason": "LLM call, transient failure likely"},
    "SESSION_SUMMARIZE": {"max_retries": 3, "reason": "LLM call, transient failure likely"},
    "EVAL_TURN":         {"max_retries": 2, "reason": "LLM eval, moderate cost"},
    "EVAL_SAFETY_CHECK": {"max_retries": 1, "reason": "Deterministic rule-based, no retry value"},
}
DEFAULT_MAX_RETRIES = 3

CONSUMER_NAME = f"worker-{socket.gethostname()}-{os.getpid()}"
_keep_running = True


def _signal_handler(signum, frame):
    global _keep_running
    sig_name = signal.Signals(signum).name
    logger.info(f"Received signal {sig_name}. Initiating graceful shutdown...")
    _keep_running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def get_redis_connection():
    if not REDIS_URL:
        return None
    if redis is None:
        logger.error("'redis' python package is not installed. Worker cannot start.")
        sys.exit(1)
    try:
        client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        client.ping()
        return client
    except Exception as exc:
        logger.error(f"Cannot connect to Redis at {REDIS_URL}: {exc}")
        return None


def init_consumer_group(client):
    """Ensures the consumer group and stream exist in Redis."""
    try:
        # Create group from '$' (only new messages) or '0' (all messages)
        # MKSTREAM automatically creates the stream if it does not exist
        client.xgroup_create(TASK_STREAM, CONSUMER_GROUP, id="0", mkstream=True)
        logger.info(f"Created consumer group '{CONSUMER_GROUP}' on stream '{TASK_STREAM}'.")
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" in str(e):
            logger.info(f"Consumer group '{CONSUMER_GROUP}' already exists.")
        else:
            raise


def execute_task(task_type: str, payload: dict):
    """Dispatches execution to the corresponding medical domain service."""
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
        run_safety_eval(payload)
    elif task_type == "EVAL_TURN":
        run_turn_eval(payload)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def process_message(client, msg_id: str, data: dict):
    """
    Executes a single stream task with Phase B retry logic and DLQ routing.
    """
    task_type = data.get("task_type", "UNKNOWN")
    raw_payload = data.get("payload", "{}")
    idempotency_key = data.get("idempotency_key", "")
    retry_count = int(data.get("retry_count", 0))

    try:
        payload = json.loads(raw_payload) if isinstance(raw_payload, str) else raw_payload
    except Exception:
        payload = {}

    logger.info(f"Processing task {task_type} (msg_id: {msg_id}, retry: {retry_count})")

    # Look up retry limit from named policy
    effective_max_retries = RETRY_POLICY.get(task_type, {}).get("max_retries", DEFAULT_MAX_RETRIES)

    with app.app_context():
        try:
            execute_task(task_type, payload)
            # Success: acknowledge message from stream
            client.xack(TASK_STREAM, CONSUMER_GROUP, msg_id)
            if idempotency_key:
                client.set(idempotency_key, "COMPLETED", ex=60)
            logger.info(f"[SUCCESS] Task {task_type} ({msg_id}) processed and acknowledged.")
        except Exception as exc:
            err_trace = traceback.format_exc()
            retry_count += 1
            logger.warning(f"Task {task_type} ({msg_id}) failed (attempt {retry_count}/{effective_max_retries}): {exc}")

            if retry_count < effective_max_retries:
                # Phase B: Exponential backoff re-enqueue
                backoff_delay = min(60, 2 ** retry_count)
                time.sleep(backoff_delay)
                data["retry_count"] = str(retry_count)
                client.xack(TASK_STREAM, CONSUMER_GROUP, msg_id)
                new_id = client.xadd(TASK_STREAM, data)
                logger.info(f"[RETRY_QUEUED] Re-enqueued {task_type} as {new_id} after {backoff_delay}s backoff.")
            else:
                # Phase B: Exceeded max retries -> Move to Dead-Letter Queue (DLQ)
                client.xack(TASK_STREAM, CONSUMER_GROUP, msg_id)
                dlq_entry = {
                    "original_msg_id": msg_id,
                    "task_type": task_type,
                    "payload": raw_payload if isinstance(raw_payload, str) else json.dumps(raw_payload),
                    "idempotency_key": idempotency_key,
                    "retry_count": str(retry_count),
                    "failed_at": str(time.time()),
                    "error": str(exc),
                    "traceback": err_trace,
                }
                dlq_id = client.xadd(DLQ_STREAM, dlq_entry)
                if idempotency_key:
                    client.set(idempotency_key, "FAILED_DLQ", ex=300)
                logger.error(f"[DLQ_ROUTED] Task {task_type} failed {effective_max_retries} times. Routed to DLQ ({dlq_id}).")


def run_worker():
    """Main worker loop consuming from Redis Stream."""
    if not REDIS_URL:
        logger.info(
            "[TaskWorker] REDIS_URL is not set. Standalone fallback is handled in-process "
            "by ThreadPoolExecutor in TaskDispatcher. Worker exiting cleanly."
        )
        return

    logger.info(f"Starting worker {CONSUMER_NAME} connecting to {REDIS_URL}...")
    client = get_redis_connection()

    attempts = 0
    max_attempts = 3
    while client is None and _keep_running and attempts < max_attempts:
        attempts += 1
        logger.warning(f"Waiting for Redis connection at {REDIS_URL} (attempt {attempts}/{max_attempts})...")
        time.sleep(3)
        client = get_redis_connection()

    if not _keep_running or client is None:
        logger.info(
            f"[TaskWorker] Could not connect to Redis at {REDIS_URL} after {max_attempts} attempts. "
            "Exiting cleanly; background operations will run via ThreadPoolExecutor."
        )
        return

    init_consumer_group(client)
    logger.info(f"Worker {CONSUMER_NAME} is listening for tasks on stream '{TASK_STREAM}'...")

    while _keep_running:
        try:
            # Read new messages for this consumer group
            # block=2000 means wait up to 2 seconds for new messages
            entries = client.xreadgroup(
                groupname=CONSUMER_GROUP,
                consumername=CONSUMER_NAME,
                streams={TASK_STREAM: ">"},
                count=5,
                block=2000
            )

            if not entries:
                continue

            for stream_name, messages in entries:
                for msg_id, data in messages:
                    if not _keep_running:
                        break
                    process_message(client, msg_id, data)

        except redis.exceptions.ConnectionError as ce:
            logger.warning(f"Redis connection dropped: {ce}. Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            logger.error(f"Unexpected worker loop exception: {e}")
            time.sleep(1)

    logger.info(f"Worker {CONSUMER_NAME} shutdown complete.")


if __name__ == "__main__":
    run_worker()
