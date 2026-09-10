"""
Unit Tests for Decoupled Asynchronous Reliability
===================================================
Tests Phase A (Redis Streams Persistence), Phase B (DLQ & 3 Retries),
and Phase C (Idempotency Filtering).
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from services.task_dispatcher import (
    enqueue_task,
    generate_idempotency_key,
    enqueue_memory_update,
    enqueue_title_update,
    enqueue_session_summarize,
    TASK_STREAM,
    DLQ_STREAM
)
from task_worker import process_message, MAX_RETRIES


class TaskQueueTestCase(unittest.TestCase):

    def setUp(self):
        self.mock_redis = MagicMock()

    # ─────────────────────────────────────────────────────────────
    # Phase A: Redis Stream Persistence Tests
    # ─────────────────────────────────────────────────────────────

    def test_enqueue_task_publishes_to_redis_stream(self):
        """Verify Phase A: Tasks are enqueued to the durable Redis Stream via xadd."""
        self.mock_redis.set.return_value = True
        self.mock_redis.xadd.return_value = "1720000000000-0"

        payload = {"user_id": 42, "latest_message": "penicillin allergy", "history_text": "none"}
        result = enqueue_task(
            task_type="UPDATE_MEMORY",
            payload=payload,
            client_override=self.mock_redis
        )

        self.assertTrue(result)
        self.mock_redis.xadd.assert_called_once()
        call_args = self.mock_redis.xadd.call_args
        stream_name = call_args[0][0]
        stream_entry = call_args[0][1]

        self.assertEqual(stream_name, TASK_STREAM)
        self.assertEqual(stream_entry["task_type"], "UPDATE_MEMORY")
        self.assertEqual(stream_entry["retry_count"], "0")
        self.assertIn("enqueued_at", stream_entry)

        unpacked_payload = json.loads(stream_entry["payload"])
        self.assertEqual(unpacked_payload["user_id"], 42)

    def test_convenience_helpers(self):
        """Verify convenience dispatchers produce expected task types."""
        self.mock_redis.set.return_value = True

        with patch("services.task_dispatcher.get_redis_client", return_value=self.mock_redis):
            enqueue_memory_update(10, "Hello", "Hi")
            enqueue_title_update(20, "First msg")
            enqueue_session_summarize(30, 8)

        self.assertEqual(self.mock_redis.xadd.call_count, 3)
        types = [call[0][1]["task_type"] for call in self.mock_redis.xadd.call_args_list]
        self.assertIn("UPDATE_MEMORY", types)
        self.assertIn("UPDATE_TITLE", types)
        self.assertIn("SESSION_SUMMARIZE", types)

    # ─────────────────────────────────────────────────────────────
    # Phase C: Idempotency Key Deduplication Tests
    # ─────────────────────────────────────────────────────────────

    def test_idempotency_key_generation(self):
        """Verify deterministic idempotency key format for different tasks."""
        key_mem = generate_idempotency_key("UPDATE_MEMORY", user_id=1, latest_message="headache")
        key_title = generate_idempotency_key("UPDATE_TITLE", session_id=5)
        key_summary = generate_idempotency_key("SESSION_SUMMARIZE", session_id=5, msg_count=12)

        self.assertTrue(key_mem.startswith("idemp:mem:1:"))
        self.assertEqual(key_title, "idemp:title:5")
        self.assertEqual(key_summary, "idemp:summary:5:12")

    def test_duplicate_task_filtered_by_idempotency_check(self):
        """Verify Phase C: If SET NX fails (already exists), task is dropped and not sent to stream."""
        # Simulate key already existing in Redis
        self.mock_redis.set.return_value = False

        payload = {"session_id": 99, "msg_count": 8}
        result = enqueue_task(
            task_type="SESSION_SUMMARIZE",
            payload=payload,
            client_override=self.mock_redis
        )

        # Enqueue should return False and xadd should NEVER be called
        self.assertFalse(result)
        self.mock_redis.xadd.assert_not_called()

    # ─────────────────────────────────────────────────────────────
    # Phase B: Dead-Letter Queue (DLQ) & 3-Retries Tests
    # ─────────────────────────────────────────────────────────────

    @patch("task_worker.execute_task")
    @patch("time.sleep", return_value=None)  # avoid actual sleep delay in test
    def test_worker_retries_transient_failure_up_to_3_times(self, mock_sleep, mock_exec):
        """Verify Phase B: Transient failures (e.g. rate limit) trigger exponential backoff retry."""
        mock_exec.side_effect = Exception("Groq 429 Too Many Requests")

        msg_data = {
            "task_type": "UPDATE_MEMORY",
            "payload": json.dumps({"user_id": 1, "latest_message": "test", "history_text": ""}),
            "idempotency_key": "idemp:mem:1:abc",
            "retry_count": "1"  # attempt 1
        }

        process_message(self.mock_redis, "msg-101", msg_data)

        # Current message should be acked
        self.mock_redis.xack.assert_called_once()
        # Message should be re-added to stream with retry_count incremented to '2'
        self.mock_redis.xadd.assert_called_once()
        requeued_data = self.mock_redis.xadd.call_args[0][1]
        self.assertEqual(requeued_data["retry_count"], "2")

    @patch("task_worker.execute_task")
    def test_worker_routes_to_dlq_after_3_failures(self, mock_exec):
        """Verify Phase B: After 3 consecutive failures, task is routed to Dead-Letter Queue (DLQ)."""
        mock_exec.side_effect = Exception("Permanent failure / repeated 429 quota exhaustion")

        msg_data = {
            "task_type": "SESSION_SUMMARIZE",
            "payload": json.dumps({"session_id": 10}),
            "idempotency_key": "idemp:summary:10:8",
            "retry_count": "2"  # 3rd attempt about to fail
        }

        process_message(self.mock_redis, "msg-202", msg_data)

        # Main stream message must be acknowledged so it doesn't block workers forever
        self.mock_redis.xack.assert_called_once()

        # Task MUST be written to the Dead-Letter Queue stream
        dlq_calls = [
            call for call in self.mock_redis.xadd.call_args_list
            if call[0][0] == DLQ_STREAM
        ]
        self.assertEqual(len(dlq_calls), 1)

        dlq_entry = dlq_calls[0][0][1]
        self.assertEqual(dlq_entry["task_type"], "SESSION_SUMMARIZE")
        self.assertEqual(dlq_entry["original_msg_id"], "msg-202")
        self.assertEqual(dlq_entry["retry_count"], "3")
        self.assertIn("Permanent failure", dlq_entry["error"])
        self.assertIn("traceback", dlq_entry)


if __name__ == "__main__":
    unittest.main()
