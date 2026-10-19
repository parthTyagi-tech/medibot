import unittest
import pytest
from research.src.clinical_triage import (
    PatientState,
    check_medication_contraindications
)
from research.src.guardrails import (
    apply_output_guardrails,
    suppress_specific_drug_dosing,
    is_pediatric,
    CLINICAL_DEFERRAL_TEXT,
    CLINICAL_OVERRIDE_TEXT
)
from services.output_guardrails import (
    ClinicalOutputGuardrail,
    StreamingGuardrailProcessor
)


class MockChunk:
    def __init__(self, content: str):
        self.content = content


class TestDrugDosingSuppressionPhase4(unittest.TestCase):
    """
    Phase 4 Test Suite:
    - Position-aware prohibition detection
    - Dosing regex precision (non-drug vs drug quantities, word numbers, collapsing)
    - Pediatric check with age_unit awareness
    - High-risk descriptive informational sentences vs recommending sentences
    - Streaming vs non-streaming parity
    - Standard-risk indication queries ('what is paracetamol used for?')
    """

    # ─────────────────────────────────────────────────────────────
    # 1. Standard-risk indication queries
    # ─────────────────────────────────────────────────────────────
    def test_standard_risk_paracetamol_indication_query(self):
        """
        'what is paracetamol used for?' must return an indication,
        keep the drug name, contain no dose, and show no refusal, for a standard-risk user.
        """
        state = PatientState()
        user_query = "what is paracetamol used for?"

        # 1. Triage safety check should NOT refuse/block the query
        blocked, refusal = check_medication_contraindications(state, user_query)
        self.assertFalse(blocked)
        self.assertIsNone(refusal)

        # 2. Guardrails on informational response should preserve drug name and indication without refusal
        llm_response = (
            "Paracetamol is commonly used to treat mild-to-moderate pain such as headaches, "
            "muscle aches, toothaches, and backaches, and is also used to reduce fever."
        )
        cleaned = apply_output_guardrails(llm_response, is_medical=True, show_disclaimer=False, patient_state=state)
        sanitized_obj, triggered = ClinicalOutputGuardrail.sanitize_response(cleaned, state)
        final_text = str(sanitized_obj)

        self.assertFalse(triggered)
        self.assertIn("Paracetamol", final_text)
        self.assertIn("mild-to-moderate pain", final_text)
        self.assertIn("reduce fever", final_text)
        self.assertNotIn("[consult pharmacist or physician for dosage]", final_text)
        self.assertNotIn(CLINICAL_OVERRIDE_TEXT, final_text)
        self.assertNotIn(CLINICAL_DEFERRAL_TEXT, final_text)

    # ─────────────────────────────────────────────────────────────
    # 2. Dosing regex precision & collapsing
    # ─────────────────────────────────────────────────────────────
    def test_dosing_regex_preserves_non_drug_quantities(self):
        """Non-drug quantities (water, protein) stay untouched."""
        state = PatientState()
        text = "Please drink 250 ml of water and consume 5 g of protein with your meal."
        cleaned = apply_output_guardrails(text, is_medical=True, show_disclaimer=False, patient_state=state)

        self.assertIn("250 ml of water", cleaned)
        self.assertIn("5 g of protein", cleaned)
        self.assertNotIn("[consult pharmacist or physician for dosage]", cleaned)

    def test_dosing_regex_suppresses_drug_quantities_and_schedules(self):
        """
        'take 2 tablets', 'take one tablet twice daily', 'half a tablet',
        '1-2 pills', 'once a day', 'every 6 hours' are suppressed.
        Consecutive placeholders collapse cleanly.
        """
        state = PatientState()
        dosing_samples = [
            ("take 2 tablets with water", "[consult pharmacist or physician for dosage]"),
            ("take one tablet twice daily", "[consult pharmacist or physician for dosage]"),
            ("take half a tablet if symptoms persist", "[consult pharmacist or physician for dosage]"),
            ("take 1-2 pills as needed", "[consult pharmacist or physician for dosage]"),
            ("take once a day in the morning", "[consult pharmacist or physician for dosage]"),
            ("take every 6 hours", "[consult pharmacist or physician for dosage]"),
            ("take 500 mg every 6 hours", "[consult pharmacist or physician for dosage]"),
        ]

        for raw, expected_token in dosing_samples:
            cleaned, flag = suppress_specific_drug_dosing(raw, state)
            self.assertTrue(flag, f"Failed to detect dosing in: {raw}")
            self.assertIn(expected_token, cleaned, f"Expected placeholder in: {cleaned}")
            # Ensure numbers/frequencies are removed
            self.assertNotIn("every 6 hours", cleaned)
            self.assertNotIn("500 mg", cleaned)
            self.assertNotIn("twice daily", cleaned)

    def test_dosing_placeholder_collapsing(self):
        """Consecutive dosing phrases collapse into a single placeholder."""
        state = PatientState()
        compound_dosing = "take 500 mg every 6 hours twice daily"
        cleaned, flag = suppress_specific_drug_dosing(compound_dosing, state)
        self.assertTrue(flag)
        self.assertEqual(cleaned.count("[consult pharmacist or physician for dosage]"), 1)

    # ─────────────────────────────────────────────────────────────
    # 3. Pediatric check using age_unit
    # ─────────────────────────────────────────────────────────────
    def test_pediatric_check_with_various_age_units(self):
        """
        Any age in days/weeks/months counts as pediatric.
        Only age_unit == 'years' is compared to < 12.
        """
        # Days
        infant_days = PatientState(age=45, age_unit="days")
        self.assertTrue(is_pediatric(infant_days))

        # Weeks
        infant_weeks = PatientState(age=8, age_unit="weeks")
        self.assertTrue(is_pediatric(infant_weeks))

        # Months
        child_months = PatientState(age=18, age_unit="months")
        self.assertTrue(is_pediatric(child_months))

        # Years (< 12 -> True)
        child_years = PatientState(age=7, age_unit="years")
        self.assertTrue(is_pediatric(child_years))

        # Years (>= 12 -> False)
        teen_years = PatientState(age=14, age_unit="years")
        self.assertFalse(is_pediatric(teen_years))

        # Dictionary format support
        self.assertTrue(is_pediatric({"age": 20, "age_unit": "days"}))
        self.assertTrue(is_pediatric({"age": 4, "age_unit": "years"}))
        self.assertFalse(is_pediatric({"age": 25, "age_unit": "years"}))

    def test_infant_with_age_unit_days_takes_pediatric_path(self):
        """Infant with age_unit='days' triggers pediatric medication restrictions and deferral."""
        infant = PatientState(age=45, age_unit="days")

        # Contraindication check
        blocked, refusal = check_medication_contraindications(infant, "Can I give paracetamol?")
        self.assertTrue(blocked)
        self.assertIn("PEDIATRIC MEDICATION RESTRICTION", refusal)

        # Output guardrails intercept permissive dosing for infant
        raw_output = "You can give paracetamol to help with the fever."
        cleaned = apply_output_guardrails(raw_output, is_medical=True, show_disclaimer=False, patient_state=infant)
        self.assertNotIn("give paracetamol", cleaned.lower())
        self.assertIn(CLINICAL_DEFERRAL_TEXT, cleaned)

    # ─────────────────────────────────────────────────────────────
    # 4. Position-aware prohibition & permissive verb resolution
    # ─────────────────────────────────────────────────────────────
    def test_oncology_patient_permissive_with_prohibition_word_replaced(self):
        """
        'You can take ibuprofen, don't worry about it' for an oncology patient is replaced.
        The prohibition word 'don't' appears after the drug and does not negate 'can take'.
        """
        state = PatientState(has_cancer_history=True)
        raw_output = "You can take ibuprofen, don't worry about it."

        # 1. apply_output_guardrails replaces permissive sentence with override
        cleaned = apply_output_guardrails(raw_output, is_medical=True, show_disclaimer=False, patient_state=state)
        self.assertNotIn("You can take ibuprofen", cleaned)
        self.assertIn(ClinicalOutputGuardrail.CLINICAL_OVERRIDE_TEXT, cleaned)

        # 2. ClinicalOutputGuardrail.sanitize_response also triggers and replaces
        sanitized_obj, triggered = ClinicalOutputGuardrail.sanitize_response(raw_output, state)
        self.assertTrue(triggered)
        self.assertNotIn("You can take ibuprofen", str(sanitized_obj))
        self.assertIn(ClinicalOutputGuardrail.CLINICAL_OVERRIDE_TEXT, str(sanitized_obj))

    def test_position_aware_prohibition_and_permissive_adjacent_sentences(self):
        """
        'Do not take ibuprofen. You can take paracetamol.'
        keeps the first sentence intact and replaces the second.
        """
        state = PatientState(has_cancer_history=True)
        raw_output = "Do not take ibuprofen. You can take paracetamol."

        # 1. Test via apply_output_guardrails
        cleaned = apply_output_guardrails(raw_output, is_medical=True, show_disclaimer=False, patient_state=state)
        self.assertIn("Do not take ibuprofen.", cleaned)
        self.assertNotIn("You can take paracetamol", cleaned)
        self.assertIn(ClinicalOutputGuardrail.CLINICAL_OVERRIDE_TEXT, cleaned)

        # 2. Test via ClinicalOutputGuardrail.sanitize_response directly
        sanitized_obj, triggered = ClinicalOutputGuardrail.sanitize_response(raw_output, state)
        self.assertTrue(triggered)
        final_text = str(sanitized_obj)
        self.assertIn("Do not take ibuprofen.", final_text)
        self.assertNotIn("You can take paracetamol", final_text)
        self.assertIn(ClinicalOutputGuardrail.CLINICAL_OVERRIDE_TEXT, final_text)

    def test_avoid_restricted_to_avoid_drug_form(self):
        """'avoiding side effects' does not count as a drug prohibition."""
        state = PatientState(has_cancer_history=True)
        # Here 'avoiding side effects' is mentioned, but the drug 'ibuprofen' is recommended
        raw_output = "While avoiding side effects is important, you can take ibuprofen."

        # 1. Test via apply_output_guardrails
        cleaned = apply_output_guardrails(raw_output, is_medical=True, show_disclaimer=False, patient_state=state)
        self.assertNotIn("you can take ibuprofen", cleaned.lower())
        self.assertIn(ClinicalOutputGuardrail.CLINICAL_OVERRIDE_TEXT, cleaned)

        # 2. Test via ClinicalOutputGuardrail.sanitize_response directly
        sanitized_obj, triggered = ClinicalOutputGuardrail.sanitize_response(raw_output, state)
        self.assertTrue(triggered)
        self.assertNotIn("you can take ibuprofen", str(sanitized_obj).lower())
        self.assertIn(ClinicalOutputGuardrail.CLINICAL_OVERRIDE_TEXT, str(sanitized_obj))

    # ─────────────────────────────────────────────────────────────
    # 5. Descriptive informational vs recommending sentences for high-risk
    # ─────────────────────────────────────────────────────────────
    def test_descriptive_informational_sentence_preserved_for_pregnancy(self):
        """
        'Paracetamol is used to reduce fever.' stays intact for pregnancy.
        Recommending sentences ('You can take paracetamol') are replaced.
        """
        state = PatientState(is_pregnant=True)

        # 1. Purely descriptive sentence
        descriptive = "Paracetamol is used to reduce fever."
        cleaned = apply_output_guardrails(descriptive, is_medical=True, show_disclaimer=False, patient_state=state)
        self.assertEqual(cleaned.strip(), "Paracetamol is used to reduce fever.")

        # 2. Recommending sentence replaced with deferral
        recommending = "You should take paracetamol for your headache."
        cleaned_rec = apply_output_guardrails(recommending, is_medical=True, show_disclaimer=False, patient_state=state)
        self.assertNotIn("take paracetamol", cleaned_rec.lower())
        self.assertIn(CLINICAL_DEFERRAL_TEXT, cleaned_rec)

    def test_descriptive_informational_sentence_preserved_for_pediatrics(self):
        """
        'Paracetamol is used to reduce fever.' stays intact for pediatric cases.
        Recommending sentences ('You can give paracetamol') are replaced.
        """
        state = PatientState(age=5, age_unit="years")

        # 1. Purely descriptive sentence
        descriptive = "Paracetamol is used to reduce fever."
        cleaned = apply_output_guardrails(descriptive, is_medical=True, show_disclaimer=False, patient_state=state)
        self.assertEqual(cleaned.strip(), "Paracetamol is used to reduce fever.")

        # 2. Recommending sentence replaced with deferral
        recommending = "You can give paracetamol for the fever."
        cleaned_rec = apply_output_guardrails(recommending, is_medical=True, show_disclaimer=False, patient_state=state)
        self.assertNotIn("give paracetamol", cleaned_rec.lower())
        self.assertIn(CLINICAL_DEFERRAL_TEXT, cleaned_rec)

    def test_active_cancer_stricter_antipyretic_interception(self):
        """
        Active cancer / immunocompromised maintains strict antipyretic interception:
        even descriptive antipyretic statements trigger the critical safety override
        to prevent masking neutropenic fever.
        """
        state = PatientState(is_active_cancer_chemo=True)
        descriptive = "Paracetamol is used to reduce fever."
        cleaned = apply_output_guardrails(descriptive, is_medical=True, show_disclaimer=False, patient_state=state)
        self.assertIn(ClinicalOutputGuardrail.CLINICAL_OVERRIDE_TEXT, cleaned)

    # ─────────────────────────────────────────────────────────────
    # 6. Streaming vs non-streaming parity
    # ─────────────────────────────────────────────────────────────
    def test_streaming_vs_non_streaming_parity(self):
        """Streaming and non-streaming pipelines produce identical outputs for the same model generation."""
        patient_state = PatientState(has_cancer_history=True)
        raw_model_output = (
            "Do not take ibuprofen for fever. "
            "You can take paracetamol if approved by your oncologist. "
            "Stay well hydrated and report any temp above 100.4."
        )

        # 1. Non-streaming execution
        non_stream_guardrail = apply_output_guardrails(
            raw_model_output,
            is_medical=True,
            show_disclaimer=False,
            patient_state=patient_state
        )
        non_stream_final, _ = ClinicalOutputGuardrail.sanitize_response(non_stream_guardrail, patient_state)

        # 2. Streaming execution
        processor = StreamingGuardrailProcessor(
            patient_state=patient_state,
            risk_tier="Routine",
            is_medical=True
        )
        chunks = [MockChunk(w + " ") for w in raw_model_output.split()]
        stream_yielded = "".join(processor.process_stream(chunks))

        self.assertEqual(stream_yielded, str(non_stream_final))
        self.assertIn("Do not take ibuprofen for fever.", stream_yielded)
        self.assertIn(ClinicalOutputGuardrail.CLINICAL_OVERRIDE_TEXT, stream_yielded)


if __name__ == "__main__":
    unittest.main()
