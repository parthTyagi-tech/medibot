import json
import unittest
from research.src.clinical_triage import PatientState


class TestPatientStateBackwardCompat(unittest.TestCase):
    """
    Tests backward compatibility when deserializing PatientState from older JSON blobs
    or dicts that predate the addition of emergency_override_served, active_emergency_topic,
    last_reminder_turn_index, and third_party_context.
    """

    def test_deserialization_of_legacy_state_blob_without_new_fields(self):
        legacy_data = {
            "age": 45,
            "age_unit": "years",
            "is_infant_under_3mo": False,
            "is_pregnant": False,
            "is_elderly": False,
            "is_immunocompromised": False,
            "is_active_cancer_chemo": True,
            "temperature": "103F",
            "duration": "3 days",
            "medication_status": "none",
            "reported_symptoms_detail": ["fever"],
            "disclosed_conditions": ["Cancer"],
            "active_chemo_confirmed": True,
            "stand_down_reason": None,
            "conditions": ["Cancer"],
            "medications": [],
            "allergies": [],
            "current_symptoms": ["fever"],
            "risk_tier": "Emergency",
            "red_flags": ["Febrile Neutropenia Risk (Cancer History + Fever)"],
            "contraindications": [],
            "disclaimer_shown": True,
            "newly_disclosed_high_risk": None,
            "needs_prior_advice_correction": False
            # Notice: missing emergency_override_served, active_emergency_topic,
            # last_reminder_turn_index, and third_party_context!
        }

        # 1. Test from_dict
        state_from_dict = PatientState.from_dict(legacy_data)
        self.assertFalse(state_from_dict.emergency_override_served)
        self.assertIsNone(state_from_dict.active_emergency_topic)
        self.assertEqual(state_from_dict.last_reminder_turn_index, -1)
        self.assertIsNone(state_from_dict.third_party_context)
        self.assertEqual(state_from_dict.temperature, "103F")
        self.assertEqual(state_from_dict.risk_tier, "Emergency")

        # 2. Test from_json
        json_str = json.dumps(legacy_data)
        state_from_json = PatientState.from_json(json_str)
        self.assertFalse(state_from_json.emergency_override_served)
        self.assertIsNone(state_from_json.active_emergency_topic)
        self.assertEqual(state_from_json.last_reminder_turn_index, -1)
        self.assertIsNone(state_from_json.third_party_context)
        self.assertEqual(state_from_json.temperature, "103F")

    def test_deserialization_of_empty_or_corrupt_json(self):
        state_empty = PatientState.from_json("")
        self.assertIsInstance(state_empty, PatientState)
        self.assertFalse(state_empty.emergency_override_served)

        state_corrupt = PatientState.from_json("{invalid_json: true,")
        self.assertIsInstance(state_corrupt, PatientState)
        self.assertFalse(state_corrupt.emergency_override_served)


if __name__ == "__main__":
    unittest.main()
