import unittest
import os
import sys

sys.path.insert(0, os.path.abspath("."))

from services.ai_service import build_system_prompt
from research.src.clinical_triage import PatientState
from research.src.guardrails import apply_output_guardrails, CLINICAL_OVERRIDE_TEXT


class TestConversationalContinuityAndMedication(unittest.TestCase):
    """
    Test suite for conversational continuity, anti-looping behavior,
    and clinical medication information boundaries.
    """

    def test_turn_1_initial_turn_prompt_directives(self):
        """Turn 1 prompt should direct assistant to greet once and ask progressive questions."""
        prompt = build_system_prompt(history_text="", user_memory="")
        self.assertIn("CONVERSATIONAL CONTINUITY DIRECTIVES (INITIAL TURN)", prompt)
        self.assertIn("Greet the patient once with professional warmth at the very start", prompt)
        self.assertIn("Context Gathering: Ask essential clarifying questions progressively", prompt)
        self.assertNotIn("CONVERSATIONAL CONTINUITY DIRECTIVES (FOLLOW-UP TURN)", prompt)

    def test_turn_2_followup_turn_prompt_directives(self):
        """Turn 2 prompt should strictly prohibit re-greetings and prevent repeating questions."""
        history = (
            "Patient: what medicines should i take in fever\n"
            "MediAssist: Hi Parth, common options include paracetamol or ibuprofen.\n"
            "Patient: medicines ?"
        )
        prompt = build_system_prompt(history_text=history, user_memory="")
        self.assertIn("CONVERSATIONAL CONTINUITY DIRECTIVES (FOLLOW-UP TURN)", prompt)
        self.assertIn("STRICTLY PROHIBIT RE-GREETINGS & APOLOGIES", prompt)
        self.assertIn("DO NOT REPEAT QUESTIONS", prompt)
        self.assertIn("CONTEXT RETENTION", prompt)
        self.assertNotIn("CONVERSATIONAL CONTINUITY DIRECTIVES (INITIAL TURN)", prompt)

    def test_medication_information_boundaries_in_prompt(self):
        """System prompt must reflect informational naming of OTC classes with contrast and contraindications."""
        prompt = build_system_prompt(history_text="", user_memory="")
        self.assertIn("MEDICATION INFORMATION BOUNDARIES (MANDATORY)", prompt)
        self.assertIn("Informational, Not Prescriptive: You may name standard over-the-counter (OTC) drug classes", prompt)
        self.assertIn("paracetamol/acetaminophen, ibuprofen", prompt)
        self.assertIn("No Dosages: Do NOT prescribe specific milligram dosages, frequencies, or schedules", prompt)
        self.assertIn("Contrast & Contraindications", prompt)

    def test_routine_fever_permits_informational_otc_drug_names(self):
        """Routine adult response naming paracetamol and ibuprofen must NOT be scrubbed by output guardrails."""
        state = PatientState()
        routine_response = (
            "Common over-the-counter options to reduce fever include paracetamol (acetaminophen) and ibuprofen. "
            "Paracetamol is gentle on the stomach for mild fever, whereas ibuprofen reduces inflammation but should be avoided "
            "if you have stomach ulcers, kidney disease, or asthma. Always follow the packaging instructions or consult a pharmacist."
        )
        cleaned = apply_output_guardrails(routine_response, is_medical=True, patient_state=state)
        
        # Drug names must be preserved for routine information
        self.assertIn("paracetamol", cleaned.lower())
        self.assertIn("ibuprofen", cleaned.lower())
        self.assertIn("acetaminophen", cleaned.lower())
        self.assertIn("stomach ulcers", cleaned.lower())

    def test_numeric_dosage_still_strictly_scrubbed_for_routine_patient(self):
        """Leaked numerical dosages (e.g. 650mg every 6 hours) must still be intercepted and scrubbed."""
        state = PatientState()
        leaked_dosing = "You can take 650mg of acetaminophen every 6 hours to manage the fever."
        cleaned = apply_output_guardrails(leaked_dosing, is_medical=True, patient_state=state)
        
        self.assertNotIn("650mg", cleaned)
        self.assertNotIn("every 6 hours", cleaned)
        self.assertIn("[consult pharmacist or physician for dosage]", cleaned)

    def test_high_risk_cancer_patient_strictly_prohibits_all_drug_names(self):
        """For active cancer/chemo patients, antipyretics and drug names remain completely prohibited."""
        state = PatientState()
        state.is_active_cancer_chemo = True
        state.disclosed_conditions.append("Cancer")

        oncology_leaked = "You could take paracetamol or ibuprofen to bring down your temperature."
        cleaned = apply_output_guardrails(oncology_leaked, is_medical=True, patient_state=state)
        
        self.assertNotIn("You could take", cleaned)
        self.assertTrue(
            CLINICAL_OVERRIDE_TEXT in cleaned
            or "discuss any medication choices directly with your doctor or pharmacist" in cleaned.lower()
        )

    def test_pediatric_high_risk_patient_strictly_prohibits_drug_names(self):
        """For young pediatric patients (<12), drug names are suppressed to enforce weight-based pediatrician evaluation."""
        state = PatientState()
        state.age = 4
        state.age_unit = "years"

        pediatric_leaked = "You could give ibuprofen for the fever."
        cleaned = apply_output_guardrails(pediatric_leaked, is_medical=True, patient_state=state)
        
        self.assertNotIn("ibuprofen", cleaned.lower())
        self.assertIn("discuss any medication choices directly with your doctor or pharmacist", cleaned.lower())


if __name__ == "__main__":
    unittest.main()
