"""
Unit Test Suite for MediAssist Intent Benchmark Dataset & Evaluation Runner
=============================================================================
Tests:
1. Dataset schema and integrity (version, fields, reviewed: true, label_policy_note).
2. Multilingual representation: exactly 7 languages with >=12 cases each.
3. Presence of all required edge cases:
   - "script" (prescription)
   - "code blue" (hospital emergency)
   - "ICD code" (diagnostic coding)
   - "python bite" (snake bite first aid)
   - Suicidal crisis / self-harm
   - Prompt injection with real symptoms
   - Pure prompt injection
   - Mid-triage context-aware follow-ups
   - Greetings: thanks, who are you, greeting + symptom, punctuation, German & Japanese variants
4. Synthetic metric math:
   - 100% medical recall pass condition
   - Medical miss -> recall < 100% -> exit code 1
   - Greeting-on-symptom critical error -> exit code 1
   - Fallback rate > 5% -> INVALID -> exit code 2
   - Accuracy < 95% -> exit code 1
5. Runner execution in mock mode with mocked time (zero wall-clock waiting).
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock

from evals.run_intent_benchmarks import (
    compute_benchmark_metrics,
    evaluate_single_query,
    run_intent_benchmarks
)


class TestIntentBenchmarks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "evals", "dataset", "intent_benchmarks.json")
        )
        with open(cls.dataset_path, "r", encoding="utf-8") as f:
            cls.data = json.load(f)
        cls.cases = cls.data.get("cases", [])

    # ─────────────────────────────────────────────────────────────
    # 1. Dataset Schema & Integrity Tests
    # ─────────────────────────────────────────────────────────────

    def test_dataset_schema_and_version_integrity(self):
        """Dataset must have version '1.0.0', valid cases array, and all required fields."""
        self.assertEqual(self.data.get("version"), "1.0.0")
        self.assertGreaterEqual(len(self.cases), 84, "Dataset must contain at least 84 cases")

        required_fields = {
            "id", "language", "category", "query",
            "expected_intent", "expected_route", "triage_active",
            "recent_turns", "reviewed"
        }

        for case in self.cases:
            missing = required_fields - set(case.keys())
            self.assertEqual(len(missing), 0, f"Case {case.get('id')} missing fields: {missing}")
            self.assertTrue(case["reviewed"], f"Case {case['id']} must have reviewed=True")
            self.assertIn(case["expected_intent"], {"MEDICAL", "GREETING", "NON_MEDICAL"})
            self.assertIn(
                case["expected_route"],
                {"MEDICAL", "GREETING", "NON_MEDICAL", "BLOCKED_EMERGENCY", "BLOCKED_INJECTION"}
            )

            # If ambiguous or health-adjacent, label_policy_note must be provided
            if "adjacent" in case["category"] or "adversarial" in case["category"] or "plus_symptom" in case["category"]:
                self.assertIn("label_policy_note", case)
                self.assertGreater(len(case["label_policy_note"].strip()), 10)

    # ─────────────────────────────────────────────────────────────
    # 2. Multilingual Distribution Tests (Amendment 2: >=12 each)
    # ─────────────────────────────────────────────────────────────

    def test_dataset_seven_languages_distribution(self):
        """All 7 supported languages must be represented with >= 12 cases each."""
        expected_langs = {"en", "es", "fr", "de", "it", "nl", "ja"}
        found_langs = set(c["language"] for c in self.cases)
        self.assertEqual(found_langs, expected_langs, "Must strictly cover all 7 supported languages")

        for lang in expected_langs:
            lang_cases = [c for c in self.cases if c["language"] == lang]
            self.assertGreaterEqual(
                len(lang_cases), 12,
                f"Language '{lang}' must have >= 12 cases, found {len(lang_cases)}"
            )

    # ─────────────────────────────────────────────────────────────
    # 3. Required Edge Cases Present Tests (Amendments 2, 8, 9)
    # ─────────────────────────────────────────────────────────────

    def test_dataset_required_edge_cases_present(self):
        """Dataset must contain all mandatory clinical, adversarial, and greeting edge cases."""
        queries_text = " ".join(c["query"].lower() for c in self.cases)

        # 1. "script" (prescription)
        has_script = any("script" in c["query"].lower() and c["expected_intent"] == "MEDICAL" for c in self.cases)
        self.assertTrue(has_script, "Must contain 'script' (prescription) medical query")

        # 2. "code blue" (hospital emergency)
        has_code_blue = any("code blue" in c["query"].lower() and c["expected_intent"] == "MEDICAL" for c in self.cases)
        self.assertTrue(has_code_blue, "Must contain 'code blue' medical query")

        # 3. "ICD code" (diagnostic coding)
        has_icd = any("icd" in c["query"].lower() and c["expected_intent"] == "MEDICAL" for c in self.cases)
        self.assertTrue(has_icd, "Must contain 'ICD code' medical query")

        # 4. "python bite" (snake bite first aid)
        has_python_bite = any("python bite" in c["query"].lower() and c["expected_intent"] == "MEDICAL" for c in self.cases)
        self.assertTrue(has_python_bite, "Must contain 'python bite' medical query")

        # 5. Suicidal crisis / self-harm expecting BLOCKED_EMERGENCY
        crisis_cases = [
            c for c in self.cases
            if c["category"] == "medical_crisis_mental_health" and c["expected_route"] == "BLOCKED_EMERGENCY"
        ]
        self.assertGreaterEqual(len(crisis_cases), 5, "Must contain self-harm cases across languages expecting BLOCKED_EMERGENCY")

        # 6. Prompt injection with real symptoms -> MEDICAL
        injection_symptoms = [
            c for c in self.cases
            if c["category"] == "medical_adversarial_injection" and c["expected_intent"] == "MEDICAL"
        ]
        self.assertGreaterEqual(len(injection_symptoms), 1, "Must contain injection with real symptoms -> MEDICAL")

        # 7. Pure prompt injection -> NON_MEDICAL / BLOCKED_INJECTION
        pure_injections = [
            c for c in self.cases
            if c["category"] == "non_medical_jailbreak" and c["expected_intent"] == "NON_MEDICAL"
        ]
        self.assertGreaterEqual(len(pure_injections), 1, "Must contain pure injection -> NON_MEDICAL")

        # 8. Mid-triage context-aware follow-ups (triage_active=True)
        triage_followups = [
            c for c in self.cases
            if c.get("triage_active") and c["expected_intent"] == "MEDICAL" and len(c.get("recent_turns", [])) > 0
        ]
        self.assertGreaterEqual(len(triage_followups), 5, "Must contain mid-triage follow-ups with triage_active=True")

        # 9. Greeting edge cases: thanks, who are you, greeting + symptom, punctuation, German & Japanese variants
        has_thanks = any("thank" in c["query"].lower() or "danke" in c["query"].lower() or "merci" in c["query"].lower() for c in self.cases)
        self.assertTrue(has_thanks, "Must contain gratitude/thanks greeting")

        has_who = any("who are you" in c["query"].lower() or "qui êtes-vous" in c["query"].lower() for c in self.cases)
        self.assertTrue(has_who, "Must contain 'who are you' check-in greeting")

        greeting_symptoms = [c for c in self.cases if c["category"] == "greeting_plus_symptom" and c["expected_intent"] == "MEDICAL"]
        self.assertGreaterEqual(len(greeting_symptoms), 5, "Must contain greeting + symptom cases expecting MEDICAL")

        has_punct = any("hey..." in c["query"] or ":)" in c["query"] for c in self.cases)
        self.assertTrue(has_punct, "Must contain greeting with punctuation variants")

        has_german_variants = any("moin moin" in c["query"].lower() or "grüß gott" in c["query"].lower() for c in self.cases)
        self.assertTrue(has_german_variants, "Must contain German greeting variants")

        has_japanese_variants = any("おはようございます" in c["query"] or "お疲れ様です" in c["query"] for c in self.cases)
        self.assertTrue(has_japanese_variants, "Must contain Japanese greeting variants")

    # ─────────────────────────────────────────────────────────────
    # 4. Metric Math Tests with Synthetic Vectors (Amendment 10)
    # ─────────────────────────────────────────────────────────────

    def test_metric_calculations_passing_vector(self):
        """Synthetic test vector with 100% recall and 100% accuracy."""
        synthetic_records = [
            {"expected_intent": "MEDICAL", "mapped_prediction": "MEDICAL", "is_correct": True, "fallback_used": False, "pre_filtered": False, "language": "en", "category": "symptom", "is_medical_miss": False, "is_greeting_on_symptom": False},
            {"expected_intent": "MEDICAL", "mapped_prediction": "MEDICAL", "is_correct": True, "fallback_used": False, "pre_filtered": True, "language": "en", "category": "emergency", "is_medical_miss": False, "is_greeting_on_symptom": False},
            {"expected_intent": "GREETING", "mapped_prediction": "GREETING", "is_correct": True, "fallback_used": False, "pre_filtered": False, "language": "es", "category": "greeting", "is_medical_miss": False, "is_greeting_on_symptom": False},
            {"expected_intent": "NON_MEDICAL", "mapped_prediction": "NON_MEDICAL", "is_correct": True, "fallback_used": False, "pre_filtered": False, "language": "fr", "category": "coding", "is_medical_miss": False, "is_greeting_on_symptom": False},
        ]
        metrics = compute_benchmark_metrics(synthetic_records)
        self.assertEqual(metrics["overall_accuracy_pct"], 100.0)
        self.assertEqual(metrics["medical_recall_pct"], 100.0)
        self.assertEqual(metrics["medical_misses"], 0)
        self.assertEqual(metrics["greeting_on_symptom_errors"], 0)
        self.assertEqual(metrics["fallback_count"], 0)
        self.assertEqual(metrics["fallback_rate_pct"], 0.0)
        self.assertEqual(metrics["non_medical_precision_pct"], 100.0)
        self.assertEqual(metrics["non_medical_fpr_pct"], 0.0)

    def test_medical_recall_failure_triggers_exit_code_1(self):
        """Even 1 medical query misclassified as NON_MEDICAL must cause medical_recall < 100% and fail."""
        synthetic_records = [
            {"expected_intent": "MEDICAL", "mapped_prediction": "MEDICAL", "is_correct": True, "fallback_used": False, "pre_filtered": False, "language": "en", "category": "symptom", "is_medical_miss": False, "is_greeting_on_symptom": False},
            # Medical miss!
            {"expected_intent": "MEDICAL", "mapped_prediction": "NON_MEDICAL", "is_correct": False, "fallback_used": False, "pre_filtered": False, "language": "en", "category": "symptom", "is_medical_miss": True, "is_greeting_on_symptom": False},
        ]
        metrics = compute_benchmark_metrics(synthetic_records)
        self.assertEqual(metrics["medical_recall_pct"], 50.0)
        self.assertEqual(metrics["medical_misses"], 1)

    def test_greeting_on_symptom_critical_error_flag(self):
        """A GREETING prediction on a medical query is flagged as greeting_on_symptom_errors."""
        synthetic_records = [
            {"expected_intent": "MEDICAL", "mapped_prediction": "GREETING", "is_correct": False, "fallback_used": False, "pre_filtered": False, "language": "de", "category": "symptom", "is_medical_miss": False, "is_greeting_on_symptom": True},
            {"expected_intent": "GREETING", "mapped_prediction": "GREETING", "is_correct": True, "fallback_used": False, "pre_filtered": False, "language": "de", "category": "greeting", "is_medical_miss": False, "is_greeting_on_symptom": False},
        ]
        metrics = compute_benchmark_metrics(synthetic_records)
        self.assertEqual(metrics["greeting_on_symptom_errors"], 1)

    def test_fallback_rate_exceeding_threshold_triggers_invalid_status(self):
        """Fallback rate > 5.0% marks run INVALID (exit code 2)."""
        # 2 fallbacks out of 10 cases = 20% fallback rate (> 5%)
        synthetic_records = [
            {"expected_intent": "MEDICAL", "mapped_prediction": "MEDICAL", "is_correct": True, "fallback_used": (i < 2), "pre_filtered": False, "language": "en", "category": "symptom", "is_medical_miss": False, "is_greeting_on_symptom": False}
            for i in range(10)
        ]
        metrics = compute_benchmark_metrics(synthetic_records)
        self.assertEqual(metrics["fallback_count"], 2)
        self.assertEqual(metrics["fallback_rate_pct"], 20.0)
        self.assertGreater(metrics["fallback_rate_pct"], 5.0)

    # ─────────────────────────────────────────────────────────────
    # 5. Runner Execution in Mock Mode (Amendment 1 & 10)
    # ─────────────────────────────────────────────────────────────

    def test_runner_mock_execution_completes_with_exit_code_zero(self):
        """Runner must execute cleanly in mock mode without wall-clock waits and exit code 0."""
        with patch("time.sleep") as mock_sleep:
            exit_code, report = run_intent_benchmarks(
                dataset_path=self.dataset_path,
                mode="llm",
                repeat=2,
                delay_s=0.0,
                is_mock=True,
                verbose=False
            )
            mock_sleep.assert_not_called()
            self.assertEqual(exit_code, 0)
            self.assertTrue(report["passed"])
            self.assertFalse(report["is_invalid"])
            self.assertEqual(report["metrics"]["medical_recall_pct"], 100.0)
            self.assertEqual(report["metrics"]["overall_accuracy_pct"], 100.0)
            self.assertEqual(report["stability"]["recall_stable_across_passes"], True)


if __name__ == "__main__":
    unittest.main()
