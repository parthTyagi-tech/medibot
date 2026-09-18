import unittest
from research.src.clinical_triage import (
    PatientState,
    ClinicalTriageEngine,
    format_context_aware_greeting
)
from research.src.clinical_parser import ClinicalEntityParser
from routes.chat import is_greeting_text
from services.state_store import check_emergency_resolution


class TestTriageLoopRegression(unittest.TestCase):
    """
    Regression test suite for MediBot Grade 9.8 Standard:
    1. Reproduce exact live transcript failure sequence and verify full resolution
    2. Ensure mixed-intent greetings (e.g. 'Hello, I have 103 fever') are NOT hijacked
    3. Ensure negated resolution disclosures (e.g. 'doctor has NOT cleared me') do not de-escalate
    4. Ensure clause-bounded negation works properly ('no fever, but doctor cleared me')
    """

    def test_reproduce_exact_transcript_resolution(self):
        patient_state = PatientState()

        # Turn 1: Cancer and fever report
        turn_1_msg = "I have cancer and running a 102 fever"
        signals_1 = ClinicalEntityParser.extract_clinical_signals(turn_1_msg)
        self.assertTrue(signals_1["has_cancer"])
        self.assertTrue(signals_1["has_fever"])
        self.assertTrue(signals_1["has_clinical_signals"])

        res_1 = ClinicalTriageEngine.evaluate(turn_1_msg, patient_state, signals_1)
        self.assertIsNotNone(res_1)
        self.assertTrue(res_1.get("requires_escalation"))
        self.assertEqual(patient_state.active_emergency, "FEBRILE_NEUTROPENIA")
        self.assertEqual(patient_state.emergency_turn_count, 1)
        self.assertIn("neutrophil", res_1["message"].lower())
        self.assertIn("screening questions", res_1["message"].lower())

        # Turn 2: User asks for medication/remedies for fever
        turn_2_msg = "Can you suggest some medication for fever?"
        signals_2 = ClinicalEntityParser.extract_clinical_signals(turn_2_msg)
        res_2 = ClinicalTriageEngine.evaluate(turn_2_msg, patient_state, signals_2)
        self.assertIsNotNone(res_2)
        self.assertTrue(res_2.get("requires_escalation"))
        self.assertEqual(patient_state.emergency_turn_count, 2)
        # Verify Turn 2 refuses fever reducers and is distinct from Turn 1
        self.assertNotEqual(res_1["message"], res_2["message"])
        self.assertIn("cannot recommend any medications", res_2["message"].lower())
        self.assertIn("fever reducers", res_2["message"].lower())

        # Turn 3: User sends pure greeting ("hello how are you medi ??")
        turn_3_msg = "hello how are you medi ??"
        signals_3 = ClinicalEntityParser.extract_clinical_signals(turn_3_msg)
        self.assertTrue(is_greeting_text(turn_3_msg))
        self.assertFalse(signals_3["has_clinical_signals"])
        is_pure_greeting_3 = is_greeting_text(turn_3_msg) and not signals_3.get("has_clinical_signals", False)
        self.assertTrue(is_pure_greeting_3)

        # Pure greeting must bypass triage monologue
        res_3 = ClinicalTriageEngine.evaluate(turn_3_msg, patient_state, signals_3)
        self.assertIsNone(res_3)

        greeting_resp_3 = format_context_aware_greeting(patient_state, first_name="Alex", msg=turn_3_msg)
        # Verify NO sirens, NO 75-word cold/congestion monologue, and empathetic check-in
        self.assertNotIn("🚨", greeting_resp_3)
        self.assertNotIn("mild cold symptoms", greeting_resp_3.lower())
        self.assertNotIn("runny nose", greeting_resp_3.lower())
        self.assertNotIn("congestion", greeting_resp_3.lower())
        self.assertIn("oncology team", greeting_resp_3.lower())

        # Turn 4: User states doctor cleared them
        turn_4_msg = "My oncologist saw me and cleared me"
        signals_4 = ClinicalEntityParser.extract_clinical_signals(turn_4_msg)
        res_4 = ClinicalTriageEngine.evaluate(turn_4_msg, patient_state, signals_4)
        self.assertIsNotNone(res_4)
        self.assertEqual(res_4.get("tier"), "RESOLVED")
        self.assertFalse(res_4.get("requires_escalation"))
        self.assertIsNone(patient_state.active_emergency)
        self.assertIn("cleared by your doctor", res_4["message"].lower())

        # Turn 5: User sends greeting after clearance
        turn_5_msg = "hello how are you medi ??"
        signals_5 = ClinicalEntityParser.extract_clinical_signals(turn_5_msg)
        self.assertTrue(is_greeting_text(turn_5_msg))
        self.assertFalse(signals_5["has_clinical_signals"])
        res_5 = ClinicalTriageEngine.evaluate(turn_5_msg, patient_state, signals_5)
        self.assertIsNone(res_5)

        greeting_resp_5 = format_context_aware_greeting(patient_state, first_name="Alex", msg=turn_5_msg)
        # Verify calm greeting without emergency alerts
        self.assertNotIn("🚨", greeting_resp_5)
        self.assertNotIn("emergency", greeting_resp_5.lower())
        self.assertNotIn("runny nose", greeting_resp_5.lower())
        self.assertIn("support you", greeting_resp_5.lower())

    def test_mixed_intent_greeting_not_hijacked(self):
        """Mixed intent greeting + emergency symptom must NOT be treated as pure greeting."""
        patient_state = PatientState()
        user_msg = "Hello, I have a 103 fever and cancer"
        signals = ClinicalEntityParser.extract_clinical_signals(user_msg)

        self.assertTrue(is_greeting_text(user_msg))
        self.assertTrue(signals.get("has_clinical_signals"))
        is_pure_greeting = is_greeting_text(user_msg) and not signals.get("has_clinical_signals", False)
        self.assertFalse(is_pure_greeting)

        res = ClinicalTriageEngine.evaluate(user_msg, patient_state, signals)
        self.assertIsNotNone(res)
        self.assertTrue(res.get("requires_escalation"))
        self.assertEqual(res.get("tier"), "Emergency")

    def test_negated_resolution_does_not_deescalate(self):
        """Negated resolution like 'Doctor has NOT cleared me' must not reset emergency state."""
        patient_state = PatientState(
            has_cancer_history=True,
            active_emergency="FEBRILE_NEUTROPENIA",
            emergency_turn_count=2
        )
        user_msg = "The doctor has NOT cleared me yet"
        signals = ClinicalEntityParser.extract_clinical_signals(user_msg)

        # 1. Test check_emergency_resolution helper
        resolved_flag = check_emergency_resolution(user_msg, patient_state)
        self.assertFalse(resolved_flag)
        self.assertEqual(patient_state.active_emergency, "FEBRILE_NEUTROPENIA")

        # 2. Test ClinicalTriageEngine.evaluate
        res = ClinicalTriageEngine.evaluate(user_msg, patient_state, signals)
        # Should not return RESOLVED tier
        if res is not None:
            self.assertNotEqual(res.get("tier"), "RESOLVED")
        self.assertEqual(patient_state.active_emergency, "FEBRILE_NEUTROPENIA")

    def test_clause_bounded_negation(self):
        """Punctuation boundary stops negation from bleeding into affirmative clearance."""
        patient_state = PatientState(
            has_cancer_history=True,
            active_emergency="FEBRILE_NEUTROPENIA",
            emergency_turn_count=2
        )
        user_msg = "I have no fever, but doctor cleared me"
        signals = ClinicalEntityParser.extract_clinical_signals(user_msg)

        # 'no fever' is negated, but 'doctor cleared me' is in a separate clause and valid
        resolved_flag = check_emergency_resolution(user_msg, patient_state)
        self.assertTrue(resolved_flag)
        self.assertIsNone(patient_state.active_emergency)


if __name__ == "__main__":
    unittest.main()
