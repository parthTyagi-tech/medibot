import unittest
from unittest.mock import MagicMock, patch
import time
import concurrent.futures

from research.src.helper import PineconeHostedEmbeddings
from services.ai_service import CustomPineconeRetriever, build_system_prompt, build_prompt
from research.src.guardrails import (
    apply_output_guardrails,
    MEDICAL_DISCLAIMER,
    NO_REFERENCE_DISCLAIMER
)


class TestPineconeEmbeddings(unittest.TestCase):
    """
    Unit tests for Pinecone hosted inference embeddings and RAG resilience.
    100% mocked, zero network.
    """

    def setUp(self):
        self.mock_pc = MagicMock()
        # Mock the probe call on init
        probe_embedding = MagicMock()
        probe_embedding.values = [0.01] * 1024
        mock_probe_res = MagicMock()
        mock_probe_res.data = [probe_embedding]
        mock_probe_res.usage = {"total_tokens": 5}
        self.mock_pc.inference.embed.return_value = mock_probe_res

    # 1. Slow call times out (Amendment 3)
    def test_slow_call_times_out(self):
        with patch("research.src.helper.Pinecone", return_value=self.mock_pc):
            embedder = PineconeHostedEmbeddings(api_key="test-key", timeout=0.2)

        embedder.pc.inference.embed.reset_mock()
        def slow_embed(*args, **kwargs):
            time.sleep(0.5)
            return mock_res

        mock_res = MagicMock()
        mock_res.data = [MagicMock(values=[0.1] * 1024)]
        embedder.pc.inference.embed.side_effect = slow_embed

        start = time.time()
        result = embedder.embed_query("slow query")
        elapsed = time.time() - start

        # Embed query should have timed out, retried once (attempt 1: 0.2s, wait: 0.5s, attempt 2: 0.2s), and returned None
        self.assertIsNone(result)
        self.assertLess(elapsed, 1.3)
        self.assertEqual(embedder.pc.inference.embed.call_count, 2)

    # 2. Quota / 4xx error is not retried (Amendment 4)
    def test_quota_4xx_not_retried(self):
        with patch("research.src.helper.Pinecone", return_value=self.mock_pc):
            embedder = PineconeHostedEmbeddings(api_key="test-key")

        # Simulate 429 Quota Exceeded error
        class QuotaExceededException(Exception):
            status = 429
            status_code = 429

        embedder.pc.inference.embed.reset_mock()
        embedder.pc.inference.embed.side_effect = QuotaExceededException("Quota exceeded: 429 Too Many Requests")

        result = embedder.embed_query("test query")
        self.assertIsNone(result)
        # MUST NOT retry on 429 / quota error -> exactly 1 call
        self.assertEqual(embedder.pc.inference.embed.call_count, 1)

    # 2b. 5xx error is retried once
    def test_5xx_error_retried_once(self):
        with patch("research.src.helper.Pinecone", return_value=self.mock_pc):
            embedder = PineconeHostedEmbeddings(api_key="test-key")

        class ServerErrorException(Exception):
            status = 503

        embedder.pc.inference.embed.reset_mock()
        embedder.pc.inference.embed.side_effect = ServerErrorException("503 Service Unavailable")

        result = embedder.embed_query("test query")
        self.assertIsNone(result)
        # Retries once on 5xx -> exactly 2 calls
        self.assertEqual(embedder.pc.inference.embed.call_count, 2)

    # 3. Breaker opens and recovers (Amendment 4)
    def test_circuit_breaker_opens_and_recovers(self):
        with patch("research.src.helper.Pinecone", return_value=self.mock_pc):
            embedder = PineconeHostedEmbeddings(api_key="test-key")

        class QuotaException(Exception):
            status = 429

        embedder.pc.inference.embed.side_effect = QuotaException("429 rate limit")

        # 3 consecutive failures
        self.assertIsNone(embedder.embed_query("fail 1"))
        self.assertIsNone(embedder.embed_query("fail 2"))
        self.assertIsNone(embedder.embed_query("fail 3"))

        # Breaker should now be OPEN
        self.assertGreater(embedder._breaker_open_until, time.time())

        # 4th call should be intercepted immediately by circuit breaker without calling API
        embedder.pc.inference.embed.reset_mock()
        result = embedder.embed_query("fail 4 while open")
        self.assertIsNone(result)
        embedder.pc.inference.embed.assert_not_called()

        # Simulate 60s cooldown elapsing
        embedder._breaker_open_until = time.time() - 1.0

        # Now mock a successful response
        mock_success = MagicMock()
        mock_success.data = [MagicMock(values=[0.5] * 1024)]
        embedder.pc.inference.embed.side_effect = None
        embedder.pc.inference.embed.return_value = mock_success

        res_recovered = embedder.embed_query("recovered query")
        self.assertIsNotNone(res_recovered)
        self.assertEqual(len(res_recovered), 1024)
        # Consecutive failures reset to 0
        self.assertEqual(embedder._consecutive_failures, 0)
        self.assertEqual(embedder._breaker_open_until, 0.0)

    # 4. Startup check failure -> no-reference mode (Amendment 6)
    def test_startup_check_dimension_mismatch(self):
        mock_embeddings = MagicMock()
        mock_embeddings.dimension = 1024

        mock_index = MagicMock()
        mock_index.describe_index_stats.return_value = {
            "dimension": 384,  # Old index dimension (mismatch!)
            "total_vector_count": 5000
        }

        retriever = CustomPineconeRetriever(
            index=mock_index,
            embeddings=mock_embeddings,
            k=4
        )

        self.assertTrue(retriever.no_reference_mode)
        docs = retriever.invoke("hypertension")
        self.assertEqual(docs, [])

    def test_startup_check_empty_index(self):
        mock_embeddings = MagicMock()
        mock_embeddings.dimension = 1024

        mock_index = MagicMock()
        mock_index.describe_index_stats.return_value = {
            "dimension": 1024,
            "total_vector_count": 0  # Empty index!
        }

        retriever = CustomPineconeRetriever(
            index=mock_index,
            embeddings=mock_embeddings,
            k=4
        )

        self.assertTrue(retriever.no_reference_mode)
        docs = retriever.invoke("hypertension")
        self.assertEqual(docs, [])

    def test_startup_check_healthy_index(self):
        mock_embeddings = MagicMock()
        mock_embeddings.dimension = 1024
        mock_embeddings.embed_query.return_value = [0.1] * 1024

        mock_index = MagicMock()
        mock_index.describe_index_stats.return_value = {
            "dimension": 1024,
            "total_vector_count": 9000
        }
        mock_index.query.return_value = {
            "matches": [
                {"metadata": {"text": "Hypertension is high blood pressure.", "source": "Gale", "page": 12}}
            ]
        }

        retriever = CustomPineconeRetriever(
            index=mock_index,
            embeddings=mock_embeddings,
            k=4
        )

        self.assertFalse(retriever.no_reference_mode)
        docs = retriever.invoke("hypertension")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, "Hypertension is high blood pressure.")
        self.assertEqual(docs[0].metadata.get("page"), 12)

    # 5. None is not cached (Amendment 8)
    def test_none_is_never_cached(self):
        with patch("research.src.helper.Pinecone", return_value=self.mock_pc):
            embedder = PineconeHostedEmbeddings(api_key="test-key")

        embedder.pc.inference.embed.side_effect = Exception("Inference network error")

        res_none = embedder.embed_query("failing_query")
        self.assertIsNone(res_none)

        # Cache MUST NOT contain "failing_query"
        self.assertNotIn("failing_query", embedder._cache)

        # Now test successful caching
        mock_success = MagicMock()
        mock_success.data = [MagicMock(values=[0.2] * 1024)]
        embedder.pc.inference.embed.side_effect = None
        embedder.pc.inference.embed.return_value = mock_success

        res_success = embedder.embed_query("working_query")
        self.assertIsNotNone(res_success)
        self.assertIn("working_query", embedder._cache)
        self.assertEqual(embedder._cache["working_query"], [0.2] * 1024)

    # 6. Disclaimer variant in no-reference mode (Amendment 7)
    def test_disclaimer_variant_in_no_reference_mode(self):
        sample_answer = "Hypertension commonly involves elevated blood pressure above 130/80 mmHg."

        # Case A: Normal medical mode (with references)
        ref_output = apply_output_guardrails(
            sample_answer,
            is_medical=True,
            show_disclaimer=True,
            no_reference=False
        )
        self.assertIn("The Gale Encyclopedia of Medicine", ref_output)
        self.assertIn("CDC", ref_output)
        self.assertIn("WHO", ref_output)

        # Case B: No-reference mode
        no_ref_output = apply_output_guardrails(
            sample_answer,
            is_medical=True,
            show_disclaimer=True,
            no_reference=True
        )
        # Must use NO_REFERENCE_DISCLAIMER
        self.assertIn("general medical knowledge", no_ref_output)
        # Must NOT claim it is based on Gale Encyclopedia, CDC, or WHO
        self.assertNotIn("The Gale Encyclopedia of Medicine", no_ref_output)
        self.assertNotIn("CDC", no_ref_output)
        self.assertNotIn("WHO", no_ref_output)

    def test_system_prompt_variant_in_no_reference_mode(self):
        # Case A: With references
        prompt_with_refs = build_system_prompt(
            retrieved_docs=[MagicMock(page_content="Asthma clinical details.")],
            no_reference=False
        )
        self.assertIn("The Gale Encyclopedia of Medicine", prompt_with_refs)
        self.assertIn("<reference_library>\nAsthma clinical details.\n</reference_library>", prompt_with_refs)

        # Case B: No references available (Amendment 7)
        prompt_no_refs = build_system_prompt(
            retrieved_docs=[],
            no_reference=True
        )
        self.assertIn("No reference material available; answer from general knowledge, do not cite sources.", prompt_no_refs)
        self.assertNotIn("Authoritative Clinical References: The Gale Encyclopedia of Medicine", prompt_no_refs)


if __name__ == "__main__":
    unittest.main()
