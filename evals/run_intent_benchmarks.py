"""
MediAssist Golden Intent Classification & Scope Routing Benchmark Runner
========================================================================
Executes the multilingual regression gate against MediAssist's routing pipeline.

Enforces:
1. Medical Recall = 100.0% (Zero medical queries classified as non-medical or greeting).
2. Overall Accuracy >= 95.0%.
3. Fallback rate <= 5.0% (timeouts/429s marked as INVALID with exit code 2).
4. Zero Greeting-on-Symptom errors (predicting GREETING on a medical case is critical fail).
5. Cross-run stability check via --repeat N (default 2) requiring identical recall.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
try:
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

from services.classifier_config import get_classifier_config
from services.routing_service import route_query, RouteDecision
from research.src.clinical_triage import PatientState


def evaluate_single_query(
    case: Dict[str, Any],
    mode: str = "llm",
    is_mock: bool = False
) -> Dict[str, Any]:
    """
    Evaluates a single benchmark case against route_query().
    Returns structured prediction metadata.
    """
    query = case["query"]
    triage_active = case.get("triage_active", False)
    recent_turns = case.get("recent_turns", [])

    patient_state = PatientState() if triage_active else None
    if triage_active:
        patient_state.risk_tier = "Moderate"
        patient_state.current_symptoms = ["fever"]

    if is_mock:
        # Deterministic mock for fast CI testing / unit tests
        expected = case["expected_intent"]
        expected_route = case.get("expected_route", expected)
        pred_action = expected_route
        pred_intent = expected
        confidence = 0.98
        fallback = False
        pre_filtered = expected_route in ("BLOCKED_INJECTION", "BLOCKED_EMERGENCY")
    else:
        decision: RouteDecision = route_query(
            msg=query,
            patient_state=patient_state,
            recent_turns=recent_turns
        )
        pred_action = decision.action
        pred_intent = decision.intent
        confidence = decision.confidence
        fallback = decision.fallback
        pre_filtered = pred_action in ("BLOCKED_INJECTION", "BLOCKED_EMERGENCY")

    # Map predicted route to clinical top-level intent
    if pred_action in ("MEDICAL", "BLOCKED_EMERGENCY"):
        mapped_pred = "MEDICAL"
    elif pred_action == "GREETING":
        mapped_pred = "GREETING"
    else:
        mapped_pred = "NON_MEDICAL"

    expected_intent = case["expected_intent"]
    is_correct = (mapped_pred == expected_intent)

    # Critical check: Did a medical/symptom case get predicted as GREETING?
    is_greeting_on_symptom = (expected_intent == "MEDICAL" and mapped_pred == "GREETING")
    # Critical check: Did a medical case get predicted as NON_MEDICAL?
    is_medical_miss = (expected_intent == "MEDICAL" and mapped_pred == "NON_MEDICAL")

    return {
        "id": case["id"],
        "language": case["language"],
        "category": case["category"],
        "query": query,
        "expected_intent": expected_intent,
        "expected_route": case.get("expected_route", expected_intent),
        "predicted_action": pred_action,
        "predicted_intent": pred_intent,
        "mapped_prediction": mapped_pred,
        "confidence": confidence,
        "fallback_used": fallback,
        "pre_filtered": pre_filtered,
        "is_correct": is_correct,
        "is_greeting_on_symptom": is_greeting_on_symptom,
        "is_medical_miss": is_medical_miss
    }


def compute_benchmark_metrics(eval_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes statistical and clinical metrics across a set of evaluation records.
    """
    total = len(eval_records)
    if total == 0:
        return {}

    correct = sum(1 for r in eval_records if r["is_correct"])
    accuracy = (correct / total) * 100.0

    medical_cases = [r for r in eval_records if r["expected_intent"] == "MEDICAL"]
    total_medical = len(medical_cases)
    medical_hits = sum(1 for r in medical_cases if r["mapped_prediction"] == "MEDICAL")
    medical_misses = sum(1 for r in medical_cases if r["is_medical_miss"])
    greeting_on_symptom_errors = sum(1 for r in medical_cases if r["is_greeting_on_symptom"])

    medical_recall = (medical_hits / total_medical * 100.0) if total_medical > 0 else 100.0

    # Non-medical precision and FPR
    non_med_expected = [r for r in eval_records if r["expected_intent"] == "NON_MEDICAL"]
    total_non_med = len(non_med_expected)

    predicted_non_med = [r for r in eval_records if r["mapped_prediction"] == "NON_MEDICAL"]
    true_non_med = sum(1 for r in predicted_non_med if r["expected_intent"] == "NON_MEDICAL")
    non_med_precision = (true_non_med / len(predicted_non_med) * 100.0) if predicted_non_med else 100.0

    # FPR: Non-medical incorrectly predicted as Medical
    non_med_false_positives = sum(1 for r in non_med_expected if r["mapped_prediction"] == "MEDICAL")
    non_med_fpr = (non_med_false_positives / total_non_med * 100.0) if total_non_med > 0 else 0.0

    # Fallback metrics
    fallbacks = sum(1 for r in eval_records if r["fallback_used"])
    fallback_rate = (fallbacks / total) * 100.0

    # Pre-filtered vs classifier metrics
    pre_filtered_count = sum(1 for r in eval_records if r["pre_filtered"])
    classifier_count = total - pre_filtered_count
    classifier_correct = sum(1 for r in eval_records if not r["pre_filtered"] and r["is_correct"])
    classifier_accuracy = (classifier_correct / classifier_count * 100.0) if classifier_count > 0 else 100.0

    # Confusion matrix
    classes = ["MEDICAL", "GREETING", "NON_MEDICAL"]
    confusion_matrix = {
        exp: {pred: 0 for pred in classes}
        for exp in classes
    }
    for r in eval_records:
        exp = r["expected_intent"]
        pred = r["mapped_prediction"]
        if exp in confusion_matrix and pred in confusion_matrix[exp]:
            confusion_matrix[exp][pred] += 1

    # Language breakdown
    languages = sorted(list(set(r["language"] for r in eval_records)))
    language_breakdown = {}
    for lang in languages:
        l_records = [r for r in eval_records if r["language"] == lang]
        l_total = len(l_records)
        l_correct = sum(1 for r in l_records if r["is_correct"])
        l_med = [r for r in l_records if r["expected_intent"] == "MEDICAL"]
        l_med_hits = sum(1 for r in l_med if r["mapped_prediction"] == "MEDICAL")
        l_rec = (l_med_hits / len(l_med) * 100.0) if l_med else 100.0
        language_breakdown[lang] = {
            "total": l_total,
            "correct": l_correct,
            "accuracy_pct": round((l_correct / l_total) * 100.0, 2),
            "medical_recall_pct": round(l_rec, 2),
            "fallbacks": sum(1 for r in l_records if r["fallback_used"])
        }

    # Category breakdown
    categories = sorted(list(set(r["category"] for r in eval_records)))
    category_breakdown = {}
    for cat in categories:
        c_records = [r for r in eval_records if r["category"] == cat]
        c_total = len(c_records)
        c_correct = sum(1 for r in c_records if r["is_correct"])
        category_breakdown[cat] = {
            "total": c_total,
            "correct": c_correct,
            "pass_rate_pct": round((c_correct / c_total) * 100.0, 2)
        }

    return {
        "total_cases": total,
        "correct_cases": correct,
        "overall_accuracy_pct": round(accuracy, 2),
        "medical_cases_total": total_medical,
        "medical_recall_pct": round(medical_recall, 2),
        "medical_misses": medical_misses,
        "greeting_on_symptom_errors": greeting_on_symptom_errors,
        "non_medical_cases_total": total_non_med,
        "non_medical_precision_pct": round(non_med_precision, 2),
        "non_medical_fpr_pct": round(non_med_fpr, 2),
        "fallback_count": fallbacks,
        "fallback_rate_pct": round(fallback_rate, 2),
        "pre_filtered_count": pre_filtered_count,
        "classifier_count": classifier_count,
        "classifier_accuracy_pct": round(classifier_accuracy, 2),
        "confusion_matrix": confusion_matrix,
        "language_breakdown": language_breakdown,
        "category_breakdown": category_breakdown
    }


def run_intent_benchmarks(
    dataset_path: str = "evals/dataset/intent_benchmarks.json",
    mode: str = "llm",
    repeat: int = 2,
    delay_s: float = 0.25,
    is_mock: bool = False,
    verbose: bool = False,
    output_path: Optional[str] = None
) -> Tuple[int, Dict[str, Any]]:
    """
    Executes repeated passes across the benchmark dataset.
    Returns (exit_code, full_report_dict).
    """
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Benchmark dataset not found at: {dataset_path}")
        return 1, {"error": "Dataset not found"}

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_version = data.get("version", "unknown")
    cases = data.get("cases", [])
    total_cases = len(cases)

    os.environ["INTENT_CLASSIFIER_MODE"] = mode
    cfg = get_classifier_config()
    from services.llm_classifier import circuit_breaker
    circuit_breaker.reset()
    model_id = cfg.model if mode == "llm" else "regex-rule-engine"
    timestamp = datetime.now(timezone.utc).isoformat()

    print("=" * 76)
    print(" MediAssist Production Intent Classification & Scope Routing Benchmark ")
    print("=" * 76)
    print(f"Timestamp:       {timestamp}")
    print(f"Model ID:        {model_id}")
    print(f"Dataset Version: {dataset_version}")
    print(f"Total Cases:     {total_cases}")
    print(f"Execution Mode:  {mode} (repeat={repeat}, delay={delay_s}s, mock={is_mock})")
    print("Regression Note: Treat dataset as a regression gate, not proof (sample size is small).")
    print("=" * 76)

    pass_results = []
    case_predictions_across_runs: Dict[str, List[str]] = {c["id"]: [] for c in cases}

    for pass_idx in range(1, repeat + 1):
        print(f"\n--- Running Pass {pass_idx}/{repeat} ---")
        run_records = []

        for idx, case in enumerate(cases, start=1):
            record = evaluate_single_query(case, mode=mode, is_mock=is_mock)
            run_records.append(record)
            case_predictions_across_runs[case["id"]].append(record["mapped_prediction"])

            if verbose:
                status = "PASS" if record["is_correct"] else "FAIL"
                print(
                    f"[{idx:02d}/{total_cases:02d}] {case['id']} ({case['language']}) "
                    f"Exp={record['expected_intent']:<11} Got={record['mapped_prediction']:<11} "
                    f"Action={record['predicted_action']:<17} [{status}]"
                )

            if delay_s > 0 and not is_mock and idx < total_cases:
                time.sleep(delay_s)

        metrics = compute_benchmark_metrics(run_records)
        pass_results.append({
            "pass_number": pass_idx,
            "metrics": metrics,
            "records": run_records
        })

        print(f"Pass {pass_idx} Overall Accuracy: {metrics['overall_accuracy_pct']}%")
        print(f"Pass {pass_idx} Medical Recall:   {metrics['medical_recall_pct']}% (Misses: {metrics['medical_misses']})")
        print(f"Pass {pass_idx} Fallback Rate:    {metrics['fallback_rate_pct']}% (Count: {metrics['fallback_count']})")

        if not is_mock and pass_idx < repeat:
            time.sleep(3.0)

    # Stability Analysis across runs
    flips = []
    for c in cases:
        cid = c["id"]
        preds = case_predictions_across_runs[cid]
        if len(set(preds)) > 1:
            flips.append({
                "id": cid,
                "query": c["query"],
                "predictions_across_passes": preds
            })

    # Validate identical medical recall across runs
    recalls = [p["metrics"]["medical_recall_pct"] for p in pass_results]
    recall_stable = (len(set(recalls)) == 1)

    # Primary pass (last pass) metrics
    primary_metrics = pass_results[-1]["metrics"]
    primary_records = pass_results[-1]["records"]

    # Diagnostics: list failures
    failures = [r for r in primary_records if not r["is_correct"]]

    print("\n" + "=" * 76)
    print(" SUMMARY BENCHMARK REPORT ")
    print("=" * 76)
    print(f"Overall Accuracy:           {primary_metrics['overall_accuracy_pct']}% (Target: >= 95.0%)")
    print(f"Medical Recall:             {primary_metrics['medical_recall_pct']}% (Target: 100.0%)")
    print(f"Medical False Negatives:    {primary_metrics['medical_misses']} (Target: 0)")
    print(f"Greeting-on-Symptom Errors: {primary_metrics['greeting_on_symptom_errors']} (Target: 0)")
    print(f"Non-Medical Precision:      {primary_metrics['non_medical_precision_pct']}%")
    print(f"Non-Medical False Pos Rate: {primary_metrics['non_medical_fpr_pct']}%")
    print(f"Fallback Rate:              {primary_metrics['fallback_rate_pct']}% (Threshold: <= 5.0%)")
    print(f"Pre-Filtered Queries:       {primary_metrics['pre_filtered_count']}/{total_cases}")
    print(f"Classifier Queries:         {primary_metrics['classifier_count']}/{total_cases} (Accuracy: {primary_metrics['classifier_accuracy_pct']}%)")
    print(f"Cross-Run Stability:        {len(flips)} label flips, Recall Stable = {recall_stable}")

    # Print Confusion Matrix
    print("\n--- Confusion Matrix ---")
    cm = primary_metrics["confusion_matrix"]
    print(f"{'Expected \\ Predicted':<22} | {'MEDICAL':<9} | {'GREETING':<9} | {'NON_MEDICAL':<11}")
    print("-" * 62)
    for row in ["MEDICAL", "GREETING", "NON_MEDICAL"]:
        print(f"{row:<22} | {cm[row]['MEDICAL']:<9} | {cm[row]['GREETING']:<9} | {cm[row]['NON_MEDICAL']:<11}")

    # Print Per-Language Breakdown
    print("\n--- Per-Language Performance ---")
    print(f"{'Lang':<6} | {'Cases':<6} | {'Accuracy':<10} | {'Medical Recall':<16} | {'Fallbacks':<10}")
    print("-" * 62)
    for lang, lmetrics in primary_metrics["language_breakdown"].items():
        print(
            f"{lang:<6} | {lmetrics['total']:<6} | {lmetrics['accuracy_pct']:>7}%   | "
            f"{lmetrics['medical_recall_pct']:>13}%   | {lmetrics['fallbacks']:<10}"
        )

    # Failures list
    if failures:
        print("\n--- Misclassified Queries ---")
        for f in failures:
            print(
                f"- [{f['id']}] ({f['language']}) Expected: {f['expected_intent']} | Got: {f['mapped_prediction']} "
                f"(Action: {f['predicted_action']}, Conf: {f['confidence']})\n  Query: '{f['query']}'"
            )

    # Determine Exit Code & Status
    # Amendment 5: If >5% of cases fell back -> INVALID (exit code 2)
    is_invalid = primary_metrics["fallback_rate_pct"] > 5.0
    # Strict Medical Recall requirement: 100.0%, no misses, no greeting on symptoms, accuracy >= 95%
    passed = (
        not is_invalid and
        primary_metrics["medical_recall_pct"] == 100.0 and
        primary_metrics["medical_misses"] == 0 and
        primary_metrics["greeting_on_symptom_errors"] == 0 and
        primary_metrics["overall_accuracy_pct"] >= 95.0 and
        recall_stable
    )

    exit_code = 0 if passed else (2 if is_invalid else 1)
    status_label = "VALID & PASSED" if passed else ("INVALID (High Fallback Rate)" if is_invalid else "FAILED (Accuracy/Recall below threshold)")

    print("\n" + "=" * 76)
    print(f"BENCHMARK STATUS: {status_label} (Exit Code {exit_code})")
    print("=" * 76)

    # Prepare report object
    report = {
        "timestamp": timestamp,
        "model_id": model_id,
        "dataset_version": dataset_version,
        "dataset_path": dataset_path,
        "total_cases": total_cases,
        "status": status_label,
        "exit_code": exit_code,
        "passed": passed,
        "is_invalid": is_invalid,
        "repeat_passes": repeat,
        "stability": {
            "recall_stable_across_passes": recall_stable,
            "label_flips_count": len(flips),
            "label_flips": flips
        },
        "metrics": primary_metrics,
        "failures": failures,
        "pass_history": [
            {"pass_number": p["pass_number"], "metrics": p["metrics"]}
            for p in pass_results
        ]
    }

    # Save report
    if not output_path:
        results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
        os.makedirs(results_dir, exist_ok=True)
        ts_clean = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(results_dir, f"intent_benchmark_{ts_clean}.json")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nFull evaluation report saved to:\n{output_path}\n")
    except Exception as e:
        print(f"[WARNING] Could not save evaluation report: {e}")

    return exit_code, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediAssist Intent Classification & Routing Benchmark Runner")
    parser.add_argument("--dataset", type=str, default="evals/dataset/intent_benchmarks.json", help="Path to benchmarks JSON")
    parser.add_argument("--mode", type=str, choices=["llm", "regex"], default="llm", help="Evaluation routing mode")
    parser.add_argument("--repeat", type=int, default=2, help="Number of evaluation passes for stability check")
    parser.add_argument("--delay", type=float, default=0.25, help="Delay in seconds between live queries")
    parser.add_argument("--mock", action="store_true", help="Run deterministic mock evaluation (offline / CI)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print per-case detailed output")
    parser.add_argument("--output", type=str, default=None, help="Custom output path for report JSON")

    args = parser.parse_args()

    exit_code, _ = run_intent_benchmarks(
        dataset_path=args.dataset,
        mode=args.mode,
        repeat=args.repeat,
        delay_s=args.delay,
        is_mock=args.mock,
        verbose=args.verbose,
        output_path=args.output
    )
    sys.exit(exit_code)
