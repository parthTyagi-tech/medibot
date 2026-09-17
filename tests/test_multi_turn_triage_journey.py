import unittest
from research.src.clinical_triage import (
    PatientState,
    extract_patient_state,
    evaluate_triage_tier,
    check_medication_contraindications
)
from research.src.guardrails import (
    apply_output_guardrails,
    suppress_hallucinated_specialties,
    suppress_specific_drug_dosing
)


class TestMultiTurnTriageJourney(unittest.TestCase):
    """
    Simulates the exact 3-turn audit conversation:
    Turn 1: "I have a fever for 3 days"
    Turn 2: "temp - 103, dry cough no medicines"
    Turn 3: "I have cancer" (widened Febrile Neutropenia emergency override)
    Turn 3b: "I have cancer, but I finished treatment 5 years ago and I'm not on anything" (stand-down)
    """

    def test_turn_1_symptom_and_duration_capture(self):
        state = PatientState()
        state = extract_patient_state("I have a fever for 3 days", state)

        # 1. State accumulation
        self.assertIn("fever", state.current_symptoms)
        self.assertIsNotNone(state.duration)
        self.assertIn("3 day", state.duration.lower())

        # 2. No hallucinated conditions
        self.assertFalse(state.is_active_cancer_chemo)
        self.assertNotIn("Cancer", state.disclosed_conditions)
        self.assertNotIn("Asthma", state.disclosed_conditions)

        # 3. Output guardrail validation
        mock_output = "I hear you have had a fever for 3 days. What is your temperature and do you have other symptoms?"
        cleaned, hallucinated = suppress_hallucinated_specialties(mock_output, state)
        self.assertFalse(hallucinated)

    def test_turn_2_vitals_meds_and_anti_repetition(self):
        state = PatientState()
        state = extract_patient_state("I have a fever for 3 days", state)
        state = extract_patient_state("temp - 103, dry cough no medicines", state)

        # 1. State accumulation: temperature, duration, meds, detailed symptoms
        self.assertIsNotNone(state.temperature)
        self.assertIn("103", state.temperature)
        self.assertIsNotNone(state.duration)
        self.assertIn("3 day", state.duration.lower())
        self.assertIsNotNone(state.medication_status)
        self.assertIn("No medicines", state.medication_status)
        self.assertIn("dry cough", state.reported_symptoms_detail)

        # 2. Zero hallucinated specialist/condition
        self.assertFalse(state.is_active_cancer_chemo)
        test_hallucinated_llm = "Because of your cancer and asthma, you should contact your oncologist and take dextromethorphan 10mg."
        cleaned, was_hallucinated = suppress_hallucinated_specialties(test_hallucinated_llm, state)
        self.assertTrue(was_hallucinated)
        self.assertNotIn("your oncologist", cleaned)
        self.assertNotIn("your cancer", cleaned)
        self.assertNotIn("your asthma", cleaned)

        # 3. Dosing suppression
        cleaned_dose, was_dosing = suppress_specific_drug_dosing(cleaned, state)
        self.assertTrue(was_dosing)
        self.assertNotIn("10mg", cleaned_dose)

    def test_turn_3_cancer_disclosure_triggers_widened_emergency_override(self):
        state = PatientState()
        state = extract_patient_state("I have a fever for 3 days", state)
        state = extract_patient_state("temp - 103, dry cough no medicines", state)
        state = extract_patient_state("I have cancer", state)

        # 1. Widened trigger fires even though chemo status is unconfirmed
        self.assertIsNone(state.active_chemo_confirmed)
        self.assertIn("Cancer", state.disclosed_conditions)

        tier, red_flags, guidance = evaluate_triage_tier(state, "I have cancer")
        self.assertEqual(tier, "Emergency")
        self.assertTrue(any("Febrile Neutropenia" in rf for rf in red_flags))
        self.assertIn("FEBRILE NEUTROPENIA", guidance)
        # Check that it avoids absolute antipyretic bans if oncologist gave a plan
        self.assertIn("checking with your oncology team first", guidance)

        # 2. Medication contraindications block any OTC drug naming for cancer patient
        blocked, refusal = check_medication_contraindications(state, "Can I take paracetamol for the fever?")
        self.assertTrue(blocked)
        self.assertIn("MEDICATION SAFETY RESTRICTION", refusal)

    def test_turn_3b_affirmative_stand_down_disclosure(self):
        state = PatientState()
        state = extract_patient_state("I have a fever for 3 days", state)
        state = extract_patient_state("temp - 103, dry cough no medicines", state)
        msg_3b = "I have cancer, but I finished treatment 5 years ago and I'm not on anything"
        state = extract_patient_state(msg_3b, state)

        # 1. Explicit affirmative stand-down recognized
        self.assertFalse(state.active_chemo_confirmed)
        self.assertIsNotNone(state.stand_down_reason)

        # 2. Tier stands down from Emergency to Urgent
        tier, red_flags, guidance = evaluate_triage_tier(state, msg_3b)
        self.assertEqual(tier, "Urgent")
        self.assertIn("URGENT CLINICAL EVALUATION RECOMMENDED", guidance)
        self.assertNotIn("Emergency Department immediately", guidance)


if __name__ == "__main__":
    unittest.main()
