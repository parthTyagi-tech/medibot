import unittest
from research.src.clinical_triage import PatientState
from research.src.guardrails import (
    apply_output_guardrails,
    requires_fixed_emergency_response
)


class TestGuardrailFixedFallback(unittest.TestCase):
    """
    Tests the fail-closed circuit breaker:
    When an emergency tier is active and the generated output trips a suppression validator
    (dosing numbers, hallucinated specialists), the system discards the generation and returns
    a pre-approved, clinician-reviewed emergency message.
    """

    def test_emergency_with_dosing_triggers_fail_closed_circuit_breaker(self):
        state = PatientState()
        state.is_active_cancer_chemo = True
        state.disclosed_conditions.append("Cancer")
        state.current_symptoms.append("fever")

        # Flawed LLM generation containing numeric dosing during an active emergency
        flawed_generation = "You have a fever. You can take acetaminophen 650mg every 6 hours and rest."

        response = apply_output_guardrails(
            flawed_generation,
            is_medical=True,
            patient_state=state,
            triage_tier="Emergency"
        )

        # Confirm the response is the fixed, clinician-reviewed emergency message
        self.assertIn("CRITICAL MEDICAL EMERGENCY: IMMEDIATE CLINICAL EVALUATION REQUIRED", response)
        self.assertIn("nearest Emergency Department", response)
        self.assertNotIn("650mg", response)
        self.assertNotIn("take acetaminophen", response)

    def test_emergency_with_hallucinated_specialist_triggers_fail_closed_circuit_breaker(self):
        state = PatientState()
        state.current_symptoms.append("chest_pain")  # Emergency tier

        # Flawed LLM generation hallucinating an unstated oncologist during a cardiac emergency
        flawed_generation = "Your symptoms are severe. Please check with your oncologist immediately."

        response = apply_output_guardrails(
            flawed_generation,
            is_medical=True,
            patient_state=state,
            triage_tier="Emergency"
        )

        self.assertIn("CRITICAL MEDICAL EMERGENCY: IMMEDIATE CLINICAL EVALUATION REQUIRED", response)
        self.assertNotIn("your oncologist", response)

    def test_non_emergency_tripping_validator_does_not_use_emergency_fixed_fallback(self):
        state = PatientState()
        # Non-emergency: patient has mild cough
        routine_generation = "For your cough, take dextromethorphan 10mg every 8 hours."

        response = apply_output_guardrails(
            routine_generation,
            is_medical=True,
            patient_state=state,
            triage_tier="Routine"
        )

        # Routine tier should sanitize in-place without triggering the emergency circuit breaker
        self.assertNotIn("CRITICAL MEDICAL EMERGENCY", response)
        self.assertNotIn("10mg", response)
        self.assertIn("[consult pharmacist or physician for dosage]", response)


if __name__ == "__main__":
    unittest.main()
