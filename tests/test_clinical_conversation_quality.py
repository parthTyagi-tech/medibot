import unittest
from research.src.clinical_triage import (
    PatientState,
    ClinicalTriageEngine,
    format_context_aware_greeting,
    evaluate_triage_tier
)
from services.ai_service import build_system_prompt


class TestClinicalConversationQuality(unittest.TestCase):
    """
    Test suite verifying conversational and clinical safety hardening in MediBot:
    1. Alarm fatigue & siren removal
    2. Multi-turn anti-repetition (Turn 1 fever/cancer vs Turn 2 secondary symptoms)
    3. Contextual whiplash prevention on greetings
    4. Prohibition of dietary hallucinations (BRAT diet)
    5. Antipyretic medication restriction for oncology/immunosuppressed patients
    """

    def test_no_sirens_or_alarmist_formatting(self):
        """
        Asserts absence of sirens (🚨) and 'CRITICAL MEDICAL EMERGENCY',
        and verifies calm, nurse-grade explanation containing 'neutrophil' and 'temperature'.
        """
        state = PatientState()
        user_msg = "I have a fever of 101 and I am currently undergoing chemo for cancer"
        
        result = ClinicalTriageEngine.evaluate(user_msg, state)
        self.assertIsNotNone(result)
        response_text = result["message"]

        # Assert absence of sirens and sensational alarms
        self.assertNotIn("🚨", response_text)
        self.assertNotIn("CRITICAL MEDICAL EMERGENCY", response_text)

        # Verify calm explanation with 'neutrophil' and 'temperature'
        self.assertIn("neutrophil", response_text.lower())
        self.assertIn("temperature", response_text.lower())

        # Also test evaluate_triage_tier directly
        state_2 = PatientState()
        tier, flags, guidance = evaluate_triage_tier(state_2, user_msg)
        self.assertEqual(tier, "Emergency")
        self.assertIsNotNone(guidance)
        self.assertNotIn("🚨", guidance)
        self.assertNotIn("CRITICAL MEDICAL EMERGENCY", guidance)
        self.assertIn("neutrophil", guidance.lower())
        self.assertIn("temperature", guidance.lower())

    def test_multi_turn_anti_repetition(self):
        """
        Simulates Turn 1 (fever + cancer) followed by Turn 2 (stomach pain)
        and asserts the two responses are distinct and context-specific.
        """
        patient_state = PatientState()

        # Turn 1: Initial report (fever + cancer)
        turn_1_msg = "I have cancer and developed a fever of 101.5 today"
        res_1 = ClinicalTriageEngine.evaluate(turn_1_msg, patient_state)
        self.assertIsNotNone(res_1)
        turn_1_resp = res_1["message"]

        # Verify Turn 1 is initial emergency triage
        self.assertEqual(patient_state.emergency_turn_count, 1)
        self.assertEqual(patient_state.active_emergency, "FEBRILE_NEUTROPENIA")
        self.assertTrue(patient_state.has_cancer_history)
        self.assertIn("neutrophil", turn_1_resp.lower())
        self.assertIn("screening questions", turn_1_resp.lower())

        # Turn 2: Secondary symptom request (stomach pain)
        turn_2_msg = "My stomach is hurting really bad, what can I take for the stomach pain?"
        res_2 = ClinicalTriageEngine.evaluate(turn_2_msg, patient_state)
        self.assertIsNotNone(res_2)
        turn_2_resp = res_2["message"]

        # Verify Turn 2 has progressed and is distinct from Turn 1
        self.assertEqual(patient_state.emergency_turn_count, 2)
        self.assertNotEqual(turn_1_resp, turn_2_resp)
        self.assertIn("stomach pain", turn_2_resp.lower())
        self.assertIn("secondary symptoms", turn_2_resp.lower())
        self.assertNotIn("screening questions", turn_2_resp.lower())

    def test_greeting_context_preservation(self):
        """
        Simulates a greeting after a fever/cancer report and asserts absence of
        toxic positivity ('feeling great' or 'great too') and presence of oncology check-in.
        """
        # Scenario A: Patient with active febrile neutropenia emergency
        patient_state = PatientState(
            has_cancer_history=True,
            active_emergency="FEBRILE_NEUTROPENIA",
            emergency_turn_count=1
        )
        patient_state.current_symptoms.append("fever")

        greeting_msg = "hello good evening"
        greeting_response = format_context_aware_greeting(patient_state, first_name="Alex", msg=greeting_msg)

        # Asserts absence of toxic positivity
        self.assertNotIn("feeling great", greeting_response.lower())
        self.assertNotIn("great too", greeting_response.lower())

        # Asserts presence of oncology check-in
        self.assertIn("oncology", greeting_response.lower())
        self.assertIn("fever", greeting_response.lower())
        self.assertIn("emergency room", greeting_response.lower())

        # Scenario B: Patient with cancer history without active emergency
        cancer_only_state = PatientState(
            has_cancer_history=True,
            active_emergency=None
        )
        resp_b = format_context_aware_greeting(cancer_only_state, first_name="Alex", msg="hi there")
        self.assertNotIn("feeling great", resp_b.lower())
        self.assertNotIn("great too", resp_b.lower())
        self.assertIn("how are you feeling", resp_b.lower())

    def test_no_brat_diet_hallucination(self):
        """
        Asserts 'brat' and 'bananas' do not appear for a cold/fever query
        unless GI symptoms (diarrhea/vomiting) are present.
        """
        # Case A: Cold and fever, no GI symptoms disclosed
        patient_state = {
            "has_cancer_history": False,
            "active_emergency": None,
            "emergency_turn_count": 0,
            "current_symptoms": ["cold", "fever", "cough"]
        }
        prompt = build_system_prompt(patient_state)

        # Verify negative constraint is injected to forbid BRAT diet hallucination
        self.assertIn("STRICT NEGATIVE CONSTRAINT: DO NOT mention, suggest, or introduce the BRAT diet", prompt)

        # Assert no recommendations of BRAT diet in non-GI triage
        state_obj = PatientState()
        state_obj.current_symptoms = ["cold", "fever"]
        res = ClinicalTriageEngine.evaluate("I have a bad cold and mild fever", state_obj)
        if res:
            self.assertNotIn("brat", res["message"].lower())
            self.assertNotIn("bananas", res["message"].lower())

    def test_antipyretic_prohibition(self):
        """
        Asserts that OTC antipyretics are warned against when cancer history is present.
        """
        # 1. Verify system prompt injection
        cancer_patient_state = {
            "has_cancer_history": True,
            "active_emergency": "FEBRILE_NEUTROPENIA",
            "current_symptoms": ["fever"]
        }
        prompt = build_system_prompt(cancer_patient_state)
        self.assertIn("CRITICAL MEDICATION RESTRICTION", prompt)
        self.assertIn("STRICTLY FORBID recommending over-the-counter fever reducers", prompt)

        # 2. Verify triage engine output explicitly forbids OTC fever reducers
        state = PatientState(has_cancer_history=True)
        res = ClinicalTriageEngine.evaluate("I have a fever of 101, should I take Tylenol or ibuprofen?", state)
        self.assertIsNotNone(res)
        guidance = res["message"].lower()

        self.assertTrue(
            "do not take over-the-counter fever reducers" in guidance
            or "self-treating secondary symptoms" in guidance
            or "do not take" in guidance
        )
        self.assertIn("fever", guidance)


if __name__ == "__main__":
    unittest.main()
