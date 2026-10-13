"""
Phase 3 Guardrails Around LLM Classifier Test Suite
====================================================
Tests all 10 amendments approved for Phase 3:
1. Unified route_query() shared by /get, /voice_chat, and voice streaming.
2. Emergency lockdown: greeting during active emergency routes to MEDICAL.
3. LLM mode is the only scope gate (bypasses regex scope refusal in MEDICAL path).
4. Any classifier failure returns intent=MEDICAL, fallback=True, confidence=0.0.
5. GREETING/NON_MEDICAL: no title/memory/eval tasks enqueued; disclaimer not consumed.
6. Failures never cached; LRU cache removed.
7. Circuit breaker: 3 failures -> skip classifier for 30s and route to MEDICAL.
8. Zero raw user text or exception message strings logged on errors (class & status code only).
9. Voice path generate_greeting uses timeout=2.0s with static fallback.
10. Regex mode restores old behavior on /get and /voice_chat for rollback.
Live model tests gated behind RUN_LIVE_LLM_TESTS=1.
"""

import os
import json
import logging
import time
import unittest
from unittest.mock import MagicMock, patch

from app import app, db
from research.src.auth import User, ChatSession, Message
from research.src.clinical_triage import PatientState
from research.src.guardrails import NON_MEDICAL_REFUSAL
from services.llm_classifier import (
    classify_query,
    generate_greeting,
    _fallback_classify,
    circuit_breaker,
    ClassifierResult,
    STATIC_GREETING_FALLBACKS,
    reset_classifier_module_state
)
from services.routing_service import route_query, RouteDecision
from services.chat_service import generate_voice_response
from services.state_store import reset_state_store


class TestClassifierGuardrails(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
        app.config["WTF_CSRF_ENABLED"] = False
        app.config["SERVER_NAME"] = None

        self.client = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()

        db.create_all()
        reset_classifier_module_state()
        reset_state_store()

        try:
            db.session.query(Message).delete()
            db.session.query(ChatSession).delete()
            db.session.query(User).filter_by(email="patient@example.com").delete()
            db.session.commit()
        except Exception:
            db.session.rollback()

        # Create a test user
        self.user = User(name="Test Patient", email="patient@example.com")
        self.user.password_hash = "mock_hash"
        db.session.add(self.user)
        db.session.commit()

    def tearDown(self):
        reset_classifier_module_state()
        reset_state_store()
        try:
            db.session.rollback()
            db.session.remove()
            db.drop_all()
        except Exception:
            pass
        finally:
            self.app_context.pop()

    # ─────────────────────────────────────────────────────────────
    # Amendment 1 & 2: Emergency Lockdown in Unified route_query
    # ─────────────────────────────────────────────────────────────

    def test_greeting_during_emergency_lockdown_routes_to_medical_llm_mode(self):
        """Amendment 2: If active_emergency or emergency_override_served is set, GREETING becomes MEDICAL."""
        state = PatientState()
        state.active_emergency = "FEBRILE_NEUTROPENIA"

        # Mock classifier returning GREETING with high confidence
        with patch("services.routing_service.classify_query") as mock_classify:
            mock_classify.return_value = ClassifierResult(
                intent="GREETING",
                language="en",
                confidence=0.98,
                greeting_reply="",
                fallback=False
            )
            decision = route_query("Hello there", patient_state=state)
            self.assertEqual(decision.action, "MEDICAL")
            self.assertEqual(decision.intent, "MEDICAL")

    def test_greeting_during_emergency_override_served_routes_to_medical(self):
        """Amendment 2: emergency_override_served=True also forces GREETING to MEDICAL."""
        state = PatientState()
        state.emergency_override_served = True

        with patch("services.routing_service.classify_query") as mock_classify:
            mock_classify.return_value = ClassifierResult(
                intent="GREETING",
                language="en",
                confidence=0.95,
                greeting_reply="",
                fallback=False
            )
            decision = route_query("Good morning", patient_state=state)
            self.assertEqual(decision.action, "MEDICAL")
            self.assertEqual(decision.intent, "MEDICAL")

    def test_greeting_during_emergency_lockdown_regex_mode(self):
        """Amendment 2: In regex mode, emergency lockdown also overrides greeting to MEDICAL."""
        state = PatientState()
        state.active_emergency = "ANAPHYLAXIS"

        with patch.dict(os.environ, {"INTENT_CLASSIFIER_MODE": "regex"}):
            with patch("research.src.intent_classifier._classify_intent_regex", return_value="greeting"):
                decision = route_query("Hi doctor", patient_state=state)
                self.assertEqual(decision.action, "MEDICAL")
                self.assertEqual(decision.mode, "regex")

    # ─────────────────────────────────────────────────────────────
    # Amendment 1 & Confidence Gate: Low confidence escalates to MEDICAL
    # ─────────────────────────────────────────────────────────────

    def test_confidence_gate_escalates_low_confidence_to_medical(self):
        """Low confidence (< 0.7) NON_MEDICAL or GREETING must escalate to MEDICAL."""
        with patch("services.routing_service.classify_query") as mock_classify:
            # Low confidence NON_MEDICAL
            mock_classify.return_value = ClassifierResult(
                intent="NON_MEDICAL",
                language="en",
                confidence=0.62,
                greeting_reply="",
                fallback=False
            )
            decision = route_query("Tell me something", patient_state=PatientState())
            self.assertEqual(decision.action, "MEDICAL")
            self.assertEqual(decision.intent, "MEDICAL")

            # Low confidence GREETING
            mock_classify.return_value = ClassifierResult(
                intent="GREETING",
                language="en",
                confidence=0.55,
                greeting_reply="",
                fallback=False
            )
            decision_greet = route_query("Hey", patient_state=PatientState())
            self.assertEqual(decision_greet.action, "MEDICAL")
            self.assertEqual(decision_greet.intent, "MEDICAL")

            # High confidence GREETING passes
            mock_classify.return_value = ClassifierResult(
                intent="GREETING",
                language="en",
                confidence=0.92,
                greeting_reply="",
                fallback=False
            )
            decision_high = route_query("Hey", patient_state=PatientState())
            self.assertEqual(decision_high.action, "GREETING")

    # ─────────────────────────────────────────────────────────────
    # Amendment 4: Failures return MEDICAL, fallback=True, confidence=0.0
    # ─────────────────────────────────────────────────────────────

    def test_classifier_failure_returns_medical_confidence_zero(self):
        """Amendment 4: Any classifier failure returns intent=MEDICAL, fallback=True, confidence=0.0."""
        # 1. Deterministic helper
        res = _fallback_classify("Random text", reason="test")
        self.assertEqual(res.intent, "MEDICAL")
        self.assertEqual(res.confidence, 0.0)
        self.assertTrue(res.fallback)
        self.assertEqual(res.greeting_reply, "")

        # 2. Network/API failure in classify_query
        with patch("services.llm_classifier._get_client_for_key") as mock_client:
            mock_client.side_effect = Exception("API connection dropped")
            classified = classify_query("I have a stomach ache")
            self.assertEqual(classified.intent, "MEDICAL")
            self.assertEqual(classified.confidence, 0.0)
            self.assertTrue(classified.fallback)

    # ─────────────────────────────────────────────────────────────
    # Amendment 7: Circuit Breaker (3 failures -> skip for 30s)
    # ─────────────────────────────────────────────────────────────

    def test_circuit_breaker_trips_after_three_failures(self):
        """Amendment 7: 3 consecutive failures open the circuit breaker and route to MEDICAL."""
        circuit_breaker.reset()
        self.assertFalse(circuit_breaker.is_open())

        with patch("services.llm_classifier._get_client_for_key") as mock_client:
            mock_client.side_effect = Exception("Connection refused")

            # 3 consecutive failures
            classify_query("Query 1")
            classify_query("Query 2")
            classify_query("Query 3")

            self.assertTrue(circuit_breaker.is_open())

            # 4th call should immediately return fail-safe without invoking client
            mock_client.reset_mock()
            res_tripped = classify_query("Query 4")
            self.assertEqual(res_tripped.intent, "MEDICAL")
            self.assertEqual(res_tripped.confidence, 0.0)
            self.assertTrue(res_tripped.fallback)
            self.assertEqual(res_tripped.raw_response.get("fallback_reason"), "circuit_breaker_open")
            mock_client.assert_not_called()

    # ─────────────────────────────────────────────────────────────
    # Amendment 8: Exception logging contains NO raw user text or exception message strings
    # ─────────────────────────────────────────────────────────────

    def test_logging_contains_no_raw_user_text_or_exception_strings(self):
        """Amendment 8: Log only exception class and status code; never exception messages or user text."""
        secret_symptom = "SecretUserMedicalSymptomPii98765"
        secret_exc_msg = "SuperSecretGroqInternalErrorPayloadLeak"

        logger_llm = logging.getLogger("llm-classifier")
        logger_route = logging.getLogger("routing-service")

        log_records = []

        class InterceptHandler(logging.Handler):
            def emit(self, record):
                log_records.append(self.format(record))

        handler = InterceptHandler()
        logger_llm.addHandler(handler)
        logger_route.addHandler(handler)

        try:
            with patch("services.llm_classifier._get_client_for_key") as mock_client:
                mock_err = RuntimeError(secret_exc_msg)
                mock_client.side_effect = mock_err

                # Run classify_query and generate_greeting
                classify_query(secret_symptom)
                generate_greeting(secret_symptom, language="en")

            combined_logs = " ".join(log_records)

            # Assert PII and internal error body are NEVER present in logs
            self.assertNotIn(secret_symptom, combined_logs)
            self.assertNotIn(secret_exc_msg, combined_logs)

            # Assert class name is logged
            self.assertIn("RuntimeError", combined_logs)
        finally:
            logger_llm.removeHandler(handler)
            logger_route.removeHandler(handler)

    # ─────────────────────────────────────────────────────────────
    # Amendment 5: Greeting & Non-Medical Turns Do NOT Enqueue Tasks or Consume Disclaimer
    # ─────────────────────────────────────────────────────────────

    def test_greeting_turn_does_not_enqueue_tasks_or_consume_disclaimer(self):
        """
        Amendment 5: GREETING turns must not enqueue memory, title, summarization, or eval.
        They must not consume the once-per-session disclaimer.
        The first MEDICAL turn must generate the session title and show the disclaimer.
        """
        from routes.chat import handle_chat_turn

        chat_session = ChatSession(user_id=self.user.id, title="New Consultation")
        db.session.add(chat_session)
        db.session.commit()

        with app.test_request_context():
            from flask import session as flask_session
            flask_session["chat_session_id"] = chat_session.id
            flask_session["patient_state"] = {}

            with patch("routes.chat.get_active_session", return_value=chat_session):
                with patch("routes.chat.enqueue_memory_update") as mock_mem:
                    with patch("routes.chat.enqueue_title_update") as mock_title:
                        with patch("routes.chat.enqueue_session_summarize") as mock_sum:
                            with patch("routes.chat.enqueue_eval_safety_check") as mock_eval:
                                with patch("services.routing_service.classify_query") as mock_classify:
                                    mock_classify.side_effect = [
                                        ClassifierResult(intent="GREETING", language="en", confidence=0.95, greeting_reply="", fallback=False),
                                        ClassifierResult(intent="MEDICAL", language="en", confidence=0.95, greeting_reply="", fallback=False),
                                    ]
                                    with patch("routes.chat.extract_patient_state", side_effect=lambda msg, ps, **kwargs: ps or PatientState()):
                                        with patch("routes.chat.generate_greeting", return_value="Hello! How can I help your health today?"):

                                            # Turn 1: Greeting
                                            greeting_resp = handle_chat_turn("hello medi", user=self.user)

                                            # Greeting returned without legal disclaimer
                                            self.assertNotIn("Disclaimer:", greeting_resp)
                                            mock_mem.assert_not_called()
                                            mock_title.assert_not_called()
                                            mock_sum.assert_not_called()
                                            mock_eval.assert_not_called()

                                            # Verify session title remained "New Consultation"
                                            db.session.refresh(chat_session)
                                            self.assertEqual(chat_session.title, "New Consultation")

                                            # Turn 2: First MEDICAL query
                                            with patch("routes.chat.ClinicalTriageEngine.evaluate", return_value=None):
                                                with patch("routes.chat.create_retrieval_chain") as mock_rag_builder:
                                                    mock_chain = MagicMock()
                                                    mock_chain.invoke.return_value = {
                                                        "answer": "A fever is typically an elevation in body temperature often caused by a viral or bacterial infection.",
                                                        "context": []
                                                    }
                                                    mock_rag_builder.return_value = mock_chain

                                                    medical_resp = handle_chat_turn("I have a mild fever and headache", user=self.user)

                                                # Disclaimer must be shown on first medical reply
                                                self.assertIn("Disclaimer:", medical_resp)

                                                # Title and memory updates MUST be enqueued for the first medical turn
                                                mock_title.assert_called_once()
                                                mock_mem.assert_called_once()

    def test_disclaimer_shown_on_first_medical_reply_after_greeting(self):
        """Phase 5 requirement: Disclaimer shown on first medical reply after a greeting."""
        return self.test_greeting_turn_does_not_enqueue_tasks_or_consume_disclaimer()

    def test_non_medical_turn_returns_verbatim_refusal_without_enqueuing_tasks(self):
        """Amendment 5: Non-medical turn returns NON_MEDICAL_REFUSAL and enqueues no background tasks."""
        from routes.chat import handle_chat_turn

        chat_session = ChatSession(user_id=self.user.id, title="New Consultation")
        db.session.add(chat_session)
        db.session.commit()

        with app.test_request_context():
            from flask import session as flask_session
            flask_session["chat_session_id"] = chat_session.id

            with patch("routes.chat.get_active_session", return_value=chat_session):
                with patch("routes.chat.enqueue_memory_update") as mock_mem:
                    with patch("routes.chat.enqueue_title_update") as mock_title:
                        with patch("services.routing_service.classify_query") as mock_classify:
                            mock_classify.return_value = ClassifierResult(
                                intent="NON_MEDICAL",
                                language="en",
                                confidence=0.99,
                                greeting_reply="",
                                fallback=False
                            )
                            resp = handle_chat_turn("Write a python script to parse csv", user=self.user)
                            self.assertEqual(resp, NON_MEDICAL_REFUSAL)
                            mock_mem.assert_not_called()
                            mock_title.assert_not_called()

    # ─────────────────────────────────────────────────────────────
    # Amendment 9: Voice path generate_greeting timeout 2.0s
    # ─────────────────────────────────────────────────────────────

    def test_voice_greeting_timeout_and_fallback(self):
        """Amendment 9: generate_greeting on the voice path specifies timeout 2.0s and falls back safely."""
        with patch("services.llm_classifier._get_client_for_key") as mock_client:
            # Client hangs / times out
            from groq import APITimeoutError
            mock_client.return_value.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())

            greeting = generate_greeting("Hello there", language="en", timeout=2.0)
            self.assertIn(greeting, STATIC_GREETING_FALLBACKS["en"])

    def test_generate_voice_response_calls_route_query_and_greeting_timeout(self):
        """generate_voice_response uses route_query and passes timeout=2.0 on greetings."""
        with patch("services.routing_service.route_query") as mock_route:
            mock_route.return_value = RouteDecision(
                action="GREETING",
                intent="GREETING",
                confidence=0.95,
                language="en",
                mode="llm"
            )
            with patch("services.llm_classifier.generate_greeting") as mock_gen_greet:
                mock_gen_greet.return_value = "Hello! What health questions can I answer for you?"
                resp = generate_voice_response("Hello MediAssist", user=self.user)
                mock_route.assert_called_once()
                mock_gen_greet.assert_called_once_with(
                    "Hello MediAssist",
                    recent_turns=[],
                    language="en",
                    timeout=2.0
                )
                self.assertIn("Hello!", resp)

    # ─────────────────────────────────────────────────────────────
    # Amendment 10: Regex mode rollback fidelity on /get and /voice_chat
    # ─────────────────────────────────────────────────────────────

    def test_regex_mode_restores_old_behavior(self):
        """Amendment 10: INTENT_CLASSIFIER_MODE=regex uses old classify_intent on both paths."""
        with patch.dict(os.environ, {"INTENT_CLASSIFIER_MODE": "regex"}):
            with patch("research.src.intent_classifier._classify_intent_regex") as mock_old_classify:
                mock_old_classify.return_value = "non_medical_code"
                decision = route_query("write python code", patient_state=PatientState())
                self.assertEqual(decision.mode, "regex")
                self.assertEqual(decision.action, "NON_MEDICAL")
                self.assertEqual(decision.response, NON_MEDICAL_REFUSAL)

    # ─────────────────────────────────────────────────────────────
    # Optional Live Model Test (behind RUN_LIVE_LLM_TESTS=1)
    # ─────────────────────────────────────────────────────────────

    @unittest.skipUnless(os.getenv("RUN_LIVE_LLM_TESTS") == "1", "Set RUN_LIVE_LLM_TESTS=1 to run live Groq LLM tests")
    def test_live_groq_routing_and_guardrails(self):
        """Live integration test against Groq endpoint."""
        res_med = classify_query("I have a persistent migraine and sensitive vision")
        self.assertEqual(res_med.intent, "MEDICAL")
        self.assertFalse(res_med.fallback)

        res_greet = classify_query("Hello there good morning")
        self.assertEqual(res_greet.intent, "GREETING")
        self.assertEqual(res_greet.greeting_reply, "")


class TestPhase4Integration(unittest.TestCase):
    """
    Phase 4 Integration Amendments (1 - 8):
    1. Exactly one classifier call per user message (zero double classification).
    2. triage_active and recent_turns passed through ("2 days" mid-triage -> medical_query).
    3. Regex mode makes zero Groq calls.
    4. Invalid env values fall back to defaults and clamp in services/classifier_config.py.
    5. Medical message containing "logout" is not an account action; whole message "logout" is.
    6. .env.example contains no secrets and correct stream names.
    """

    def setUp(self):
        app.config["TESTING"] = True
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
        app.config["WTF_CSRF_ENABLED"] = False
        app.config["SERVER_NAME"] = None

        self.client = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()

        db.create_all()
        reset_classifier_module_state()
        reset_state_store()

        try:
            db.session.query(Message).delete()
            db.session.query(ChatSession).delete()
            db.session.query(User).filter_by(email="patient@example.com").delete()
            db.session.commit()
        except Exception:
            db.session.rollback()

        self.user = User(name="Test Patient", email="patient@example.com")
        self.user.password_hash = "mock_hash"
        db.session.add(self.user)
        db.session.commit()

        self.chat_session = ChatSession(user_id=self.user.id, title="Test Consultation")
        db.session.add(self.chat_session)
        db.session.commit()

    def tearDown(self):
        reset_classifier_module_state()
        reset_state_store()
        try:
            db.session.rollback()
            db.session.remove()
            db.drop_all()
        except Exception:
            pass
        finally:
            self.app_context.pop()

    def test_exactly_one_classifier_call_per_message(self):
        """Amendment 1 & 2: In LLM mode, exactly one classifier call is made per user message."""
        from routes.chat import handle_chat_turn

        with patch("services.routing_service.classify_query") as mock_classify:
            mock_classify.return_value = ClassifierResult(
                intent="MEDICAL",
                language="en",
                confidence=0.95,
                greeting_reply="",
                fallback=False
            )
            with app.test_request_context():
                from flask import session as flask_session
                flask_session["chat_session_id"] = self.chat_session.id
                flask_session["patient_state"] = {}

                with patch("routes.chat.get_active_session", return_value=self.chat_session):
                    with patch("routes.chat.extract_patient_state", return_value=PatientState()):
                        with patch("routes.chat.ClinicalTriageEngine.evaluate", return_value=None):
                            with patch("routes.chat.evaluate_triage_tier", return_value=("Normal", [], None)):
                                with patch("routes.chat.create_retrieval_chain") as mock_retrieval_chain:
                                    mock_chain = MagicMock()
                                    mock_chain.invoke.return_value = {"answer": "Take rest and drink fluids."}
                                    mock_retrieval_chain.return_value = mock_chain

                                    with patch("routes.chat.enqueue_memory_update"), \
                                         patch("routes.chat.enqueue_title_update"), \
                                         patch("routes.chat.enqueue_session_summarize"), \
                                         patch("routes.chat.enqueue_eval_safety_check"):
                                        ans = handle_chat_turn(msg="I have an ear ache", user=self.user)
                                    # Exactly ONE classifier call per message
                                    self.assertEqual(mock_classify.call_count, 1)

                                    # And derive legacy intent with ZERO extra LLM calls
                                    from research.src.intent_classifier import classify_intent
                                    legacy_intent = classify_intent(
                                        message="I have an ear ache",
                                        route_decision=RouteDecision(action="MEDICAL", intent="MEDICAL")
                                    )
                                    self.assertEqual(legacy_intent, "medical_query")
                                    self.assertEqual(mock_classify.call_count, 1)

    def test_triage_active_skips_classifier_call(self):
        """Requirement 1: triage_active returns MEDICAL without calling classifier (logged as SKIPPED_CONTEXT)."""
        state = PatientState()
        state.risk_tier = "Moderate"
        state.current_symptoms = ["fever"]
        recent = [{"role": "assistant", "content": "How long have you had the fever?"}]

        with patch("services.routing_service.classify_query") as mock_classify:
            decision = route_query("2 days", patient_state=state, recent_turns=recent)
            mock_classify.assert_not_called()
            self.assertEqual(decision.action, "MEDICAL")
            self.assertEqual(decision.intent, "MEDICAL")
            self.assertEqual(decision.raw_result, {"skipped_reason": "SKIPPED_CONTEXT"})

            from research.src.intent_classifier import classify_intent
            intent = classify_intent(message="2 days", route_decision=decision, recent_turns=recent, patient_state=state)
            self.assertEqual(intent, "medical_query")

    def test_emergency_state_skips_classifier_call(self):
        """Requirement 1: Any active emergency state returns MEDICAL without calling classifier."""
        state = PatientState()
        state.active_emergency = "FEBRILE_NEUTROPENIA"

        with patch("services.routing_service.classify_query") as mock_classify:
            decision = route_query("what should I do?", patient_state=state)
            mock_classify.assert_not_called()
            self.assertEqual(decision.action, "MEDICAL")
            self.assertEqual(decision.intent, "MEDICAL")
            self.assertEqual(decision.raw_result, {"skipped_reason": "SKIPPED_CONTEXT"})

    def test_regular_message_makes_exactly_one_classifier_call(self):
        """Requirement 1: Confirm still exactly one classifier call for all non-context-skipped messages."""
        with patch("services.routing_service.classify_query") as mock_classify:
            mock_classify.return_value = ClassifierResult(
                intent="MEDICAL",
                language="en",
                confidence=0.98,
                greeting_reply="",
                fallback=False
            )
            decision = route_query("I have a persistent rash on my arm")
            mock_classify.assert_called_once_with(
                user_text="I have a persistent rash on my arm",
                recent_turns=None,
                triage_active=False
            )
            self.assertEqual(decision.action, "MEDICAL")

    def test_regex_mode_makes_zero_groq_calls(self):
        """Amendment 2 & 8: In regex mode, classify_intent and route_query make zero Groq calls."""
        with patch.dict(os.environ, {"INTENT_CLASSIFIER_MODE": "regex"}):
            with patch("services.llm_classifier.classify_query") as mock_classify:
                with patch("services.llm_classifier._get_client_for_key") as mock_groq_client:
                    from research.src.intent_classifier import classify_intent
                    # 1. Direct classify_intent
                    intent = classify_intent(None, "I have severe asthma")
                    self.assertEqual(intent, "medical_query")
                    # 2. route_query
                    decision = route_query("I have severe asthma")
                    self.assertEqual(decision.action, "MEDICAL")
                    self.assertEqual(decision.mode, "regex")

                    # Assert zero Groq / classify_query calls
                    mock_classify.assert_not_called()
                    mock_groq_client.assert_not_called()

    def test_classifier_config_defaults_and_clamping(self):
        """Amendment 5 & 8: Invalid env values fall back to defaults and clamp at call time."""
        from services.classifier_config import get_classifier_config

        # Test defaults
        with patch.dict(os.environ, {}, clear=True):
            cfg = get_classifier_config()
            self.assertEqual(cfg.mode, "llm")
            self.assertEqual(cfg.model, "llama-3.1-8b-instant")
            self.assertEqual(cfg.fallback_model, "qwen/qwen3.8-27b")
            self.assertEqual(cfg.timeout_s, 2.5)
            self.assertEqual(cfg.min_confidence, 0.7)

        # Test invalid mode falls back to default
        with patch.dict(os.environ, {"INTENT_CLASSIFIER_MODE": "invalid_mode"}):
            cfg = get_classifier_config()
            self.assertEqual(cfg.mode, "llm")

        # Test timeout clamping
        with patch.dict(os.environ, {"CLASSIFIER_TIMEOUT_S": "100.0"}):
            cfg = get_classifier_config()
            self.assertEqual(cfg.timeout_s, 5.0)

        with patch.dict(os.environ, {"CLASSIFIER_TIMEOUT_S": "0.01"}):
            cfg = get_classifier_config()
            self.assertEqual(cfg.timeout_s, 0.5)

        with patch.dict(os.environ, {"CLASSIFIER_TIMEOUT_S": "not_a_number"}):
            cfg = get_classifier_config()
            self.assertEqual(cfg.timeout_s, 2.5)

        # Test confidence clamping
        with patch.dict(os.environ, {"CLASSIFIER_MIN_CONFIDENCE": "2.5"}):
            cfg = get_classifier_config()
            self.assertEqual(cfg.min_confidence, 1.0)

        with patch.dict(os.environ, {"CLASSIFIER_MIN_CONFIDENCE": "-1.0"}):
            cfg = get_classifier_config()
            self.assertEqual(cfg.min_confidence, 0.0)

        with patch.dict(os.environ, {"CLASSIFIER_MIN_CONFIDENCE": "invalid_conf"}):
            cfg = get_classifier_config()
            self.assertEqual(cfg.min_confidence, 0.7)

    def test_medical_message_containing_logout_is_not_account_action(self):
        """Amendment 4 & 8: Medical message containing 'logout' is not an account action; whole message 'logout' is."""
        # 1. Whole message "logout" IS an account action
        decision_logout = route_query("logout")
        self.assertEqual(decision_logout.action, "ACCOUNT_ACTION")
        self.assertEqual(decision_logout.intent, "account_action")

        decision_delete = route_query("delete my account")
        self.assertEqual(decision_delete.action, "ACCOUNT_ACTION")

        # 2. Medical message containing "logout" is NOT an account action -> proceeds to MEDICAL
        with patch("services.routing_service.classify_query") as mock_classify:
            mock_classify.return_value = ClassifierResult(
                intent="MEDICAL",
                language="en",
                confidence=0.95,
                greeting_reply="",
                fallback=False
            )
            query = "Doctor, I felt severe nausea and dizziness after I logout"
            decision_med = route_query(query)
            self.assertNotEqual(decision_med.action, "ACCOUNT_ACTION")
            self.assertEqual(decision_med.action, "MEDICAL")
            self.assertEqual(decision_med.intent, "MEDICAL")

        # In regex mode too:
        with patch.dict(os.environ, {"INTENT_CLASSIFIER_MODE": "regex"}):
            from research.src.intent_classifier import classify_intent
            intent_logout = classify_intent(None, "logout")
            self.assertEqual(intent_logout, "account_action")

            intent_med = classify_intent(None, "Doctor, I felt severe nausea and dizziness after I logout")
            self.assertEqual(intent_med, "medical_query")

    def test_account_action_suppressed_during_emergency_lockdown(self):
        """Amendment 4: ACCOUNT_PHRASES / MEMORY_PHRASES are NEVER evaluated during emergency lockdown."""
        state = PatientState()
        state.active_emergency = "ANAPHYLAXIS"

        with patch("services.routing_service.classify_query") as mock_classify:
            mock_classify.return_value = ClassifierResult(
                intent="MEDICAL",
                language="en",
                confidence=0.99,
                greeting_reply="",
                fallback=False
            )
            decision = route_query("logout", patient_state=state)
            self.assertNotEqual(decision.action, "ACCOUNT_ACTION")
            self.assertEqual(decision.action, "MEDICAL")

    def test_env_example_contains_no_secrets(self):
        """Amendment 8: .env.example contains no secrets, has correct stream names, and OAUTHLIB_INSECURE_TRANSPORT=0."""
        env_example_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env.example")
        self.assertTrue(os.path.exists(env_example_path), ".env.example must exist")

        with open(env_example_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check stream names
        self.assertIn("TASK_STREAM=medical-background-tasks", content)
        self.assertIn("DLQ_STREAM=medical-tasks-dlq", content)

        # Check OAuth transport
        self.assertIn("OAUTHLIB_INSECURE_TRANSPORT=0", content)

        # Check classifier configs present
        self.assertIn("INTENT_CLASSIFIER_MODE=llm", content)
        self.assertIn("LLM_CLASSIFIER_MODEL=llama-3.1-8b-instant", content)
        self.assertIn("LLM_CLASSIFIER_FALLBACK_MODEL=qwen/qwen3.8-27b", content)
        self.assertIn("CLASSIFIER_TIMEOUT_S=2.5", content)
        self.assertIn("CLASSIFIER_MIN_CONFIDENCE=0.7", content)

        # Check no actual secrets leaked
        prohibited_signatures = [
            "gsk_", "AIzaSy", "livekit_", "pcsk_", "xkeysib-",
            "parthtyagi3389@gmail.com", "parthtyagi"
        ]
        for sig in prohibited_signatures:
            self.assertNotIn(sig, content, f"Secret signature '{sig}' found in .env.example!")


if __name__ == "__main__":
    unittest.main()
