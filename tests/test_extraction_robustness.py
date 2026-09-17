import unittest
from research.src.clinical_triage import (
    PatientState,
    extract_patient_state
)


class TestExtractionRobustness(unittest.TestCase):
    """
    Validates extraction robustness against diverse natural language phrasings.
    """

    def test_fever_since_day_of_week(self):
        state = PatientState()
        state = extract_patient_state("I have had a fever since Monday", state)
        self.assertIn("fever", state.current_symptoms)
        self.assertIsNotNone(state.duration)
        self.assertIn("since monday", state.duration.lower())

    def test_temperature_yesterday_phrasing(self):
        state = PatientState()
        state = extract_patient_state("it was 101 yesterday, worse today", state)
        self.assertIsNotNone(state.temperature)
        self.assertIn("101", state.temperature)

    def test_started_taking_specific_medication(self):
        state = PatientState()
        state = extract_patient_state("started taking Dolo this morning for headache", state)
        self.assertIsNotNone(state.medication_status)
        self.assertIn("dolo", state.medication_status.lower())

    def test_no_meds_so_far_phrasing(self):
        state = PatientState()
        state = extract_patient_state("no meds so far, just drinking water", state)
        self.assertIsNotNone(state.medication_status)
        self.assertEqual(state.medication_status, "No medicines taken")

    def test_additive_state_preservation(self):
        state = PatientState()
        # Turn 1
        state = extract_patient_state("fever for 3 days", state)
        # Turn 2
        state = extract_patient_state("my temperature is 102.5 degrees", state)
        # Turn 3
        state = extract_patient_state("also have a dry cough", state)

        # Confirm nothing was lost
        self.assertIn("3 day", state.duration.lower())
        self.assertIn("102.5", state.temperature)
        self.assertIn("fever", state.current_symptoms)
        self.assertIn("dry cough", state.reported_symptoms_detail)


if __name__ == "__main__":
    unittest.main()
