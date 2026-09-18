"""
MediAssist Advanced Clinical Quality & Safety Test Suite
========================================================
Comprehensive automated test suite validating the 5 production-grade clinical hardening dimensions:
1. Negation Blindness (e.g. 'no fever', 'afebrile' prevents emergency triage)
2. Third-Party Entity Attribution (relative cancer/fever alerts caregiver without mutating patient state)
3. Emergency State TTL Expiration (12-hour auto-reset)
4. Emergency Affirmative Stand-Down & Resolution (e.g. 'back from hospital, doctor cleared me')
5. Deterministic Output Guardrails (BRAT diet scrubbing, oncology antipyretic interception)
6. Distributed Session State Store (InMemory & Redis fallback)
"""

import time
import unittest
from research.src.clinical_triage import (
    PatientState,
    ClinicalTriageEngine,
    evaluate_triage_tier
)
from research.src.clinical_parser import ClinicalEntityParser
from services.state_store import (
    InMemoryStateStore,
    RedisStateStore,
    check_and_apply_ttl,
    check_emergency_resolution
)
from services.output_guardrails import ClinicalOutputGuardrail


from services.audit_logger import AuditLogger


class TestAdvancedClinicalQuality(unittest.TestCase):

    def setUp(self):
        self.cancer_patient = PatientState(
            has_cancer_history=True,
            is_active_cancer_chemo=True,
            disclosed_conditions=["Cancer"]
        )

    # ─────────────────────────────────────────────────────────
    # 1. Negation Blindness Tests
    # ─────────────────────────────────────────────────────────

    def test_negation_protection_fever(self):
        """A cancer patient reporting symptoms with explicit fever negation must NOT trigger febrile neutropenia."""
        user_msg = "I have a sore throat and mild cough, but definitely no fever."
        result = ClinicalTriageEngine.evaluate(user_msg, self.cancer_patient)
        self.assertIsNone(result)
        self.assertNotIn("fever", self.cancer_patient.current_symptoms)
        self.assertIsNone(self.cancer_patient.active_emergency)

    def test_afebrile_and_denies_fever(self):
        """Tests medical negation terms 'afebrile' and 'denies fever'."""
        signals_afebrile = ClinicalEntityParser.extract_clinical_signals("Patient is afebrile with normal vital signs.")
        self.assertTrue(signals_afebrile["fever_negated"])
        self.assertFalse(signals_afebrile["has_fever"])

        signals_denies = ClinicalEntityParser.extract_clinical_signals("I have a runny nose, denies any fever or chills.")
        self.assertTrue(signals_denies["fever_negated"])
        self.assertFalse(signals_denies["has_fever"])

    # ─────────────────────────────────────────────────────────
    # 2. Third-Party Entity Attribution Tests
    # ─────────────────────────────────────────────────────────

    def test_caregiver_attribution_isolation(self):
        """
        When a user reports that a relative (e.g. 'my mom') has cancer and a fever:
        1. Emergency caregiver advice is provided immediately
        2. The user's own patient_state is NOT marked with cancer or emergency
        """
        fresh_state = PatientState()
        user_msg = "My mom has lymphoma and she developed a fever of 102 degrees today."

        result = ClinicalTriageEngine.evaluate(user_msg, fresh_state)
        self.assertIsNotNone(result)
        self.assertEqual(result["tier"], "Emergency")
        self.assertEqual(result["emergency_type"], "FEBRILE_NEUTROPENIA_CAREGIVER")
        self.assertIn("mom", result["message"].lower())

        # Crucial: User's personal state must NOT be contaminated
        self.assertFalse(fresh_state.has_cancer_history)
        self.assertFalse(fresh_state.is_active_cancer_chemo)
        self.assertIsNone(fresh_state.active_emergency)

    def test_third_party_cancer_only_no_emergency_no_mutation(self):
        """A user mentioning their father has cancer without fever must not mutate user's state."""
        fresh_state = PatientState()
        user_msg = "My father was diagnosed with lung cancer last week."

        result = ClinicalTriageEngine.evaluate(user_msg, fresh_state)
        self.assertIsNone(result)
        self.assertFalse(fresh_state.has_cancer_history)
        self.assertNotIn("Cancer", fresh_state.disclosed_conditions)

    # ─────────────────────────────────────────────────────────
    # 3. Emergency State TTL Expiration Tests
    # ─────────────────────────────────────────────────────────

    def test_emergency_ttl_expiration(self):
        """Active emergency older than 12 hours (43200s) must automatically reset."""
        state = PatientState(
            has_cancer_history=True,
            active_emergency="FEBRILE_NEUTROPENIA",
            emergency_timestamp=time.time() - 50000,  # > 12 hours ago
            emergency_turn_count=2,
            current_symptoms=["fever"]
        )

        expired = check_and_apply_ttl(state, ttl_seconds=43200)
        self.assertTrue(expired)
        self.assertIsNone(state.active_emergency)
        self.assertEqual(state.emergency_turn_count, 0)
        self.assertEqual(state.resolved_emergency, "EXPIRED_TTL")
        self.assertNotIn("fever", state.current_symptoms)

    def test_emergency_state_within_ttl_remains_active(self):
        """Active emergency younger than 12 hours must remain active."""
        state = PatientState(
            has_cancer_history=True,
            active_emergency="FEBRILE_NEUTROPENIA",
            emergency_timestamp=time.time() - 1800,  # 30 mins ago
            emergency_turn_count=1
        )

        expired = check_and_apply_ttl(state, ttl_seconds=43200)
        self.assertFalse(expired)
        self.assertEqual(state.active_emergency, "FEBRILE_NEUTROPENIA")

    # ─────────────────────────────────────────────────────────
    # 4. Emergency Affirmative Resolution / Stand-Down Tests
    # ─────────────────────────────────────────────────────────

    def test_emergency_resolution_trigger(self):
        """When a user states they have returned from the ER and were cleared, emergency resets."""
        state = PatientState(
            has_cancer_history=True,
            active_emergency="FEBRILE_NEUTROPENIA",
            emergency_turn_count=2
        )

        user_msg = "I'm back from the hospital, the oncologist checked me and gave IV antibiotics. Fever is gone now."
        resolved = check_emergency_resolution(user_msg, state)
        self.assertTrue(resolved)
        self.assertIsNone(state.active_emergency)
        self.assertEqual(state.resolved_emergency, "FEBRILE_NEUTROPENIA")

        # Triage engine should not trigger emergency on this turn
        triage_res = ClinicalTriageEngine.evaluate(user_msg, state)
        self.assertIsNone(triage_res)

    # ─────────────────────────────────────────────────────────
    # 5. Deterministic Output Guardrails Tests
    # ─────────────────────────────────────────────────────────

    def test_output_guardrail_diet_scrub(self):
        """BRAT diet recommendations must be deterministically replaced with modern nutrition guidance."""
        raw_llm_output = (
            "For your upset stomach, you should follow the BRAT diet. "
            "Stick with bananas, rice, applesauce, and toast for the next two days."
        )
        sanitized = ClinicalOutputGuardrail.sanitize_response(raw_llm_output, PatientState())
        self.assertNotIn("BRAT diet", sanitized)
        self.assertNotIn("bananas, rice, applesauce, and toast", sanitized)
        self.assertIn("nutrient-dense", sanitized.lower())

    def test_output_guardrail_antipyretic_scrub(self):
        """Permissive antipyretic suggestions to cancer patients must be intercepted and prohibited."""
        raw_llm_output = (
            "To manage your fever at home, take 500mg of acetaminophen or paracetamol every 6 hours."
        )
        state = PatientState(has_cancer_history=True)
        sanitized = ClinicalOutputGuardrail.sanitize_response(raw_llm_output, state)

        # Must not suggest taking the drug
        self.assertNotIn("take 500mg of acetaminophen", sanitized)
        # Must include explicit contraindication block
        self.assertIn("DO NOT take over-the-counter fever-reducing medications", sanitized)
        self.assertIn("oncologist", sanitized.lower())

    def test_safe_response_untouched_for_normal_patient(self):
        """Legitimate general advice for a non-cancer user is not falsely modified."""
        raw_output = "Be sure to stay hydrated with fluids and rest."
        state = PatientState()
        sanitized = ClinicalOutputGuardrail.sanitize_response(raw_output, state)
        self.assertEqual(sanitized, raw_output)

    # ─────────────────────────────────────────────────────────
    # 6. Distributed State Store Tests
    # ─────────────────────────────────────────────────────────

    def test_in_memory_state_store(self):
        store = InMemoryStateStore()
        ps = PatientState(age=45, has_cancer_history=True)

        store.set("session_123", ps)
        retrieved = store.get("session_123")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.age, 45)
        self.assertTrue(retrieved.has_cancer_history)

        store.delete("session_123")
        self.assertIsNone(store.get("session_123"))

    def test_redis_store_fallback(self):
        """RedisStateStore with unreachable host gracefully falls back to memory without crash."""
        redis_store = RedisStateStore(redis_url="redis://non_existent_host_9999:6379/0")
        ps = PatientState(age=30, is_pregnant=True)

        # Should seamlessly use fallback
        redis_store.set("session_fallback", ps)
        retrieved = redis_store.get("session_fallback")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.age, 30)
        self.assertTrue(retrieved.is_pregnant)

    # ─────────────────────────────────────────────────────────
    # 7. Medico-Legal Audit Telemetry Tests
    # ─────────────────────────────────────────────────────────

    def test_audit_logger_telemetry(self):
        """Verify AuditLogger logs structured JSON events for all 5 audit conditions."""
        test_session = "test_audit_session_999"
        
        # Test all 5 events
        rec1 = AuditLogger.log_emergency_triggered(test_session, "FEBRILE_NEUTROPENIA", "ONCOLOGY_FEVER_INTERSECTION")
        self.assertEqual(rec1["event_type"], "EMERGENCY_TRIGGERED")
        self.assertEqual(rec1["session_id"], test_session)

        rec2 = AuditLogger.log_caregiver_intercept(test_session, "FEBRILE_NEUTROPENIA_CAREGIVER", "THIRD_PARTY_ONCOLOGY_FEVER")
        self.assertEqual(rec2["event_type"], "CAREGIVER_INTERCEPT")

        rec3 = AuditLogger.log_guardrail_override(test_session, "ONCOLOGY_FEVER", "ANTIPYRETIC_CONTRAINDICATION", raw_tokens_intercepted="acetaminophen")
        self.assertEqual(rec3["event_type"], "GUARDRAIL_OVERRIDE")
        self.assertEqual(rec3["raw_tokens_intercepted"], "acetaminophen")

        rec4 = AuditLogger.log_ttl_expired(test_session, "FEBRILE_NEUTROPENIA", "12H_TTL_POLICY")
        self.assertEqual(rec4["event_type"], "TTL_EXPIRED")

        rec5 = AuditLogger.log_emergency_resolved(test_session, "FEBRILE_NEUTROPENIA", "AFFIRMATIVE_CLINICAL_STAND_DOWN")
        self.assertEqual(rec5["event_type"], "EMERGENCY_RESOLVED")

        # Read back from recent events
        recent = AuditLogger.get_recent_audit_events(limit=10)
        session_events = [e for e in recent if e.get("session_id") == test_session]
        self.assertGreaterEqual(len(session_events), 5)
        event_types = {e.get("event_type") for e in session_events}
        self.assertTrue({"EMERGENCY_TRIGGERED", "CAREGIVER_INTERCEPT", "GUARDRAIL_OVERRIDE", "TTL_EXPIRED", "EMERGENCY_RESOLVED"}.issubset(event_types))


if __name__ == "__main__":
    unittest.main()
