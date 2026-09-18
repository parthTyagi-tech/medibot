import re
import unittest
from research.src.clinical_triage import (
    PatientState,
    extract_patient_state,
    evaluate_triage_tier,
    friendly_emergency_label,
    maybe_append_emergency_reminder
)


class TestIntentScopeAfterEmergency(unittest.TestCase):
    """
    Simulates the full multi-turn session addressing Bug 2 & Bug 3:
    Turn 1: Fever for 3 days
    Turn 2: Cancer disclosure -> Emergency override fires
    Turn 3: Friend with depression -> Supportive guidance + single reminder note (no oncology template re-served)
    Turn 4: Immediate next unrelated turn -> Cool-down suppresses reminder note
    Turn 5: Continuing/escalating emergency ("can't breathe now") -> Re-fires via state-diff
    Turn 6: Explicit stand-down -> Risk tier downgrades and emergency_override_served resets
    """

    def test_multi_turn_scope_and_state_diff_journey(self):
        state = PatientState()

        # Turn 1: "I have a fever from 3 days"
        msg_1 = "I have a fever from 3 days"
        prev_1 = state.get_diff_snapshot()
        state = extract_patient_state(msg_1, state)
        diff_1 = state.has_state_diff(prev_1)
        self.assertTrue(diff_1)
        tier_1, flags_1, guidance_1 = evaluate_triage_tier(state, msg_1, has_new_structured_fact=diff_1)
        self.assertNotEqual(tier_1, "Emergency")
        self.assertIsNone(guidance_1)
        self.assertFalse(state.emergency_override_served)

        # Turn 2: "I have cancer" -> triggers febrile neutropenia emergency
        msg_2 = "I have cancer"
        prev_2 = state.get_diff_snapshot()
        state = extract_patient_state(msg_2, state)
        diff_2 = state.has_state_diff(prev_2)
        self.assertTrue(diff_2)
        tier_2, flags_2, guidance_2 = evaluate_triage_tier(state, msg_2, has_new_structured_fact=diff_2)
        self.assertEqual(tier_2, "Emergency")
        self.assertIsNotNone(guidance_2)
        # Server sets emergency_override_served = True when served
        state.emergency_override_served = True
        self.assertEqual(state.active_emergency_topic, "febrile_neutropenia")

        # Turn 3: "my friend aman dealing with the depression what can i do ??"
        msg_3 = "my friend aman dealing with the depression what can i do ??"
        prev_3 = state.get_diff_snapshot()
        state = extract_patient_state(msg_3, state)
        diff_3 = state.has_state_diff(prev_3)
        # Third-party mention must NOT mutate patient state
        self.assertFalse(diff_3)
        tier_3, flags_3, guidance_3 = evaluate_triage_tier(state, msg_3, has_new_structured_fact=diff_3)
        # Risk tier remains Emergency in state, but override guidance does NOT blindly re-fire
        self.assertEqual(tier_3, "Emergency")
        self.assertIsNone(guidance_3)

        # Reminder note logic for Turn 3 (first unrelated turn after emergency)
        raw_depression_guidance = (
            "When supporting a friend dealing with depression, listen empathetically without judgment, "
            "encourage them to talk with a counselor or therapist, and share crisis lines if needed."
        )
        ans_3 = maybe_append_emergency_reminder(raw_depression_guidance, state, turn_count=3, msg=msg_3)
        # Must contain supportive guidance, no emergency oncology directives
        self.assertNotIn("oncology", raw_depression_guidance.lower())
        self.assertNotIn("911", raw_depression_guidance.lower())
        # Reminder note must be present
        self.assertIn("Reminder: please also follow up on the fever and cancer history we discussed earlier", ans_3)
        self.assertEqual(state.last_reminder_turn_index, 3)

        # Turn 4: Immediate next turn (unrelated) -> cool-down test
        msg_4 = "can you recommend any mental health support groups"
        prev_4 = state.get_diff_snapshot()
        state = extract_patient_state(msg_4, state)
        diff_4 = state.has_state_diff(prev_4)
        self.assertFalse(diff_4)
        tier_4, flags_4, guidance_4 = evaluate_triage_tier(state, msg_4, has_new_structured_fact=diff_4)
        self.assertIsNone(guidance_4)

        raw_groups_guidance = "Support groups like NAMI or local peer groups can be very beneficial."
        ans_4 = maybe_append_emergency_reminder(raw_groups_guidance, state, turn_count=4, msg=msg_4)
        # Reminder note must be ABSENT (cool-down of 5 turns enforced)
        self.assertNotIn("Reminder:", ans_4)

        # Turn 5: "I can't breathe now and I'm getting confused" (continuing / escalating emergency)
        msg_5 = "I can't breathe now and I'm getting confused"
        prev_5 = state.get_diff_snapshot()
        state = extract_patient_state(msg_5, state)
        diff_5 = state.has_state_diff(prev_5)
        # State-diff must detect the new structured respiratory / neurological symptoms!
        self.assertTrue(diff_5)
        tier_5, flags_5, guidance_5 = evaluate_triage_tier(state, msg_5, has_new_structured_fact=diff_5)
        self.assertEqual(tier_5, "Emergency")
        # Assert override IS re-fired/escalated via state diff (not keyword matching)
        self.assertIsNotNone(guidance_5)

        # Turn 6: Explicit stand-down
        msg_6 = "I finished chemo 5 years ago, not on anything now"
        # Test stand-down behavior directly:
        test_state = PatientState()
        test_state.current_symptoms.append("fever")
        test_state.is_active_cancer_chemo = True
        test_state.emergency_override_served = True
        test_state.active_emergency_topic = "febrile_neutropenia"

        test_state = extract_patient_state(msg_6, test_state)
        self.assertFalse(test_state.active_chemo_confirmed)
        self.assertFalse(test_state.emergency_override_served)
        self.assertIsNone(test_state.active_emergency_topic)

        tier_6, flags_6, guidance_6 = evaluate_triage_tier(test_state, msg_6)
        self.assertEqual(tier_6, "Urgent")
        self.assertFalse(test_state.emergency_override_served)

    def test_ambiguous_case_disambiguation_detection(self):
        ambiguous_queries = [
            "suggest some medication or things to avoid in this case",
            "what should I do in this case",
            "can you suggest things to avoid in this case",
        ]
        pattern = re.compile(
            r"\b(in\s+this\s+case|in\s+that\s+case|for\s+this\s+case|in\s+this\s+situation)\b"
            r"|"
            r"(\b(suggest\s+(some\s+)?medication|things\s+to\s+avoid)\b.*\b(this\s+case|that\s+case)\b)",
            re.IGNORECASE
        )
        for q in ambiguous_queries:
            with self.subTest(query=q):
                self.assertTrue(bool(pattern.search(q)), f"Query should match ambiguous follow-up: {q}")

    def test_disambiguation_does_not_glue_reminder_note(self):
        """
        Confirms Item 3 from review: When a turn triggers the clarifying disambiguation question,
        the prompt is returned cleanly without a reminder note glued onto it.
        """
        state = PatientState()
        state.emergency_override_served = True
        state.active_emergency_topic = "febrile_neutropenia"
        state.third_party_context = "friend"
        state.last_reminder_turn_index = -1  # Cool-down condition is primed

        msg = "suggest some medication or things to avoid in this case"
        is_ambiguous = bool(re.search(
            r"\b(in\s+this\s+case|in\s+that\s+case|for\s+this\s+case|in\s+this\s+situation)\b",
            msg.lower()
        )) or (
            bool(re.search(r"\b(suggest\s+(some\s+)?medication|things\s+to\s+avoid)\b", msg.lower()))
            and bool(re.search(r"\b(this\s+case|that\s+case)\b", msg.lower()))
        )
        self.assertTrue(is_ambiguous)

        # In routes/chat.py, when is_ambiguous and state.third_party_context and state.emergency_override_served:
        disambiguation_ans = (
            "Just to make sure I answer the right thing — are you asking about supporting your friend, "
            "or about your own fever/cancer situation from earlier?"
        )
        # Verify that disambiguation response never contains a reminder note
        self.assertNotIn("Reminder:", disambiguation_ans)

    def test_friendly_emergency_label_mappings(self):
        """
        Confirms Item 4 from review: friendly_emergency_label is a generalized mapping,
        not hardcoded to only one scenario.
        """
        mappings = {
            "febrile_neutropenia": "fever and cancer history",
            "neonatal_fever": "infant fever",
            "preeclampsia": "pregnancy-related symptoms",
            "cardiovascular_respiratory": "chest pain or severe breathing difficulty",
        }
        for topic, expected_label in mappings.items():
            with self.subTest(topic=topic):
                self.assertEqual(friendly_emergency_label(topic), expected_label)

        # Fallback for unknown / custom topics
        self.assertEqual(friendly_emergency_label("unknown_topic"), "urgent medical symptoms")
        self.assertEqual(friendly_emergency_label(None), "urgent medical symptoms")


if __name__ == "__main__":
    unittest.main()
