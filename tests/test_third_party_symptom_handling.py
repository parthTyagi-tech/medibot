import unittest
from research.src.clinical_triage import (
    PatientState,
    extract_patient_state,
    evaluate_triage_tier
)
from research.src.intent_classifier import is_third_party_query


class TestThirdPartySymptomHandling(unittest.TestCase):
    """
    Tests Bug 3: Scoping symptoms and conditions to grammatical subject.
    Ensures that mentions of friends, family members, or third parties
    do NOT mutate the patient's own PatientState or trigger emergency triage.
    """

    def test_third_party_query_detection(self):
        tp_queries = [
            ("my friend aman dealing with the depression what can i do ??", "friend"),
            ("my mom was diagnosed with cancer", "mom"),
            ("my brother has a severe fever", "brother"),
            ("my son has an ear infection", "son"),
            ("someone I know is experiencing chest pain", "someone I know"),
            ("for my father who has high blood pressure", "father"),
        ]
        for query, expected_rel in tp_queries:
            with self.subTest(query=query):
                is_tp, rel = is_third_party_query(query)
                self.assertTrue(is_tp, f"Expected {query} to be detected as third-party")
                self.assertIsNotNone(rel)

        first_person_queries = [
            "I have a fever of 103",
            "I was diagnosed with cancer",
            "Can you tell me about hypertension",
        ]
        for query in first_person_queries:
            with self.subTest(query=query):
                is_tp, _ = is_third_party_query(query)
                self.assertFalse(is_tp, f"Expected {query} NOT to be detected as third-party")

    def test_third_party_fever_does_not_mutate_patient_state(self):
        state = PatientState()
        msg = "my friend has a fever of 103"
        state = extract_patient_state(msg, state)

        # Patient's own vitals and symptoms must NOT be polluted
        self.assertIsNone(state.temperature)
        self.assertNotIn("fever", state.reported_symptoms_detail)
        self.assertNotIn("fever", state.current_symptoms)
        self.assertIsNotNone(state.third_party_context)

        # Triage should remain Informational, not Emergency or Urgent
        tier, red_flags, guidance = evaluate_triage_tier(state, msg)
        self.assertNotEqual(tier, "Emergency")
        self.assertIsNone(guidance)

    def test_third_party_cancer_does_not_trigger_personal_oncology_emergency(self):
        state = PatientState()
        # Even if patient already had a mild fever:
        state.current_symptoms.append("fever")
        state.reported_symptoms_detail.append("fever")

        msg = "my mom was diagnosed with cancer"
        state = extract_patient_state(msg, state)

        # Patient's own high-risk flags must NOT be set
        self.assertFalse(state.is_active_cancer_chemo)
        self.assertNotIn("cancer", [c.lower() for c in state.disclosed_conditions])
        self.assertNotIn("active chemotherapy/cancer", [c.lower() for c in state.conditions])

        # Triage evaluation: since the patient themselves does not have cancer,
        # Febrile Neutropenia emergency override MUST NOT fire
        tier, red_flags, guidance = evaluate_triage_tier(state, msg)
        self.assertNotEqual(tier, "Emergency")
        self.assertNotIn("Febrile Neutropenia Risk (Cancer History + Fever)", red_flags)


if __name__ == "__main__":
    unittest.main()
