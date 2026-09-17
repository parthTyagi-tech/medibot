"""
Offline Gold Evaluation Runner (Path A)
=======================================
Executes batch evaluation of the MediBot RAG pipeline against data/eval_gold_set.json.
- Computes reference-based metrics (context_recall, answer_correctness, completeness)
- Computes reference-free metrics (context_relevance, context_rank_correlation, faithfulness, answer_relevance)
- Computes clinical safety metrics (triage_tier_correctness, red_flag_recall, non_diagnostic_compliance, harmfulness_flag)
- Distinguishes trusted (reviewed: True) from provisional (reviewed: False) scores (Change 6)
- Exports detailed per-case results to CSV and outputs executive summary to stdout
"""

import os
import sys
import json
import csv
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GoldEvalRunner")

from research.src.clinical_triage import (
    PatientState,
    extract_patient_state,
    evaluate_triage_tier,
    check_medication_contraindications,
)
from research.src.guardrails import apply_output_guardrails
from research.src.eval_metrics import (
    evaluate_context_relevance,
    evaluate_context_recall,
    calculate_spearman_rank_correlation,
    evaluate_faithfulness,
    evaluate_answer_relevance,
    evaluate_answer_correctness,
    evaluate_completeness,
    evaluate_triage_tier_correctness,
    evaluate_red_flag_recall,
    evaluate_non_diagnostic_compliance,
    evaluate_harmfulness_flag,
)


def run_pipeline_for_query(query: str, patient_state: PatientState) -> Dict[str, Any]:
    """
    Executes the real MediBot retrieval + generation pipeline for a single query.
    Returns: {"answer": str, "retrieved_chunks": List[str], "triage_tier": str}
    """
    from services.ai_service import retriever, chatModel, build_prompt
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

    # 1. Triage & contraindications
    dosing_blocked, dosing_refusal = check_medication_contraindications(patient_state, query)
    risk_tier, red_flags, override_guidance = evaluate_triage_tier(patient_state, query)

    if override_guidance and risk_tier == "Emergency":
        raw_answer = override_guidance
        answer = apply_output_guardrails(raw_answer, is_medical=True, show_disclaimer=False)
        return {
            "answer": answer,
            "retrieved_chunks": [],
            "triage_tier": risk_tier,
            "is_emergency_override": True
        }

    if dosing_blocked:
        raw_answer = dosing_refusal
        answer = apply_output_guardrails(raw_answer, is_medical=True, show_disclaimer=False)
        return {
            "answer": answer,
            "retrieved_chunks": [],
            "triage_tier": risk_tier,
            "is_emergency_override": False
        }

    # 2. RAG retrieval & generation
    dynamic_prompt = build_prompt(history_text="", user_memory="", patient_state=patient_state)
    retrieved_chunks = []

    try:
        qa_chain = create_stuff_documents_chain(chatModel, dynamic_prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)
        response = rag_chain.invoke({"input": query})
        raw_answer = response.get("answer", "")
        context_docs = response.get("context", [])
        if isinstance(context_docs, list):
            retrieved_chunks = [d.page_content if hasattr(d, "page_content") else str(d) for d in context_docs]
        elif isinstance(context_docs, str):
            retrieved_chunks = [context_docs]
    except Exception as exc:
        logger.warning(f"RAG retrieval fallback for '{query[:30]}...': {exc}")
        direct_prompt = dynamic_prompt.format(context="Clinical medicine reference and Gale Encyclopedia principles.", input=query)
        raw_resp = chatModel.invoke(direct_prompt)
        raw_answer = raw_resp.content if hasattr(raw_resp, "content") else str(raw_resp)

    answer = apply_output_guardrails(raw_answer, is_medical=True, show_disclaimer=False)
    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "triage_tier": risk_tier,
        "is_emergency_override": False
    }


def evaluate_single_gold_entry(entry: Dict[str, Any], max_llm_judge: bool = True) -> Dict[str, Any]:
    """Runs pipeline and all applicable metrics on a single gold entry."""
    query = entry["query"]
    gold_answer = entry["gold_answer"]
    expected_tier = entry.get("expected_tier")
    red_flag_scenario = entry.get("red_flag_scenario")
    reviewed = entry.get("reviewed", False)

    # 1. State extraction
    patient_state = extract_patient_state(query, PatientState())

    # 2. Pipeline execution
    pipeline_out = run_pipeline_for_query(query, patient_state)
    answer = pipeline_out["answer"]
    chunks = pipeline_out["retrieved_chunks"]

    # 3. Rule-based evaluation (0 LLM cost)
    triage_eval = evaluate_triage_tier_correctness(query, answer, patient_state, expected_tier)
    red_flag_eval = evaluate_red_flag_recall(answer, query, patient_state, red_flag_scenario)
    non_diag_eval = evaluate_non_diagnostic_compliance(answer)

    # 4. Retrieval evaluation
    chunk_scores = []
    if chunks and max_llm_judge:
        for c in chunks[:4]:
            res = evaluate_context_relevance(query, c)
            chunk_scores.append(res["score"])
        avg_context_relevance = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0.0
        rank_corr = calculate_spearman_rank_correlation(chunk_scores)
    else:
        avg_context_relevance = 0.0
        rank_corr = 0.0

    # 5. Generation & Reference-based evaluation
    if max_llm_judge:
        faith_eval = evaluate_faithfulness(chunks, answer)
        ans_rel_eval = evaluate_answer_relevance(query, answer)
        correctness_eval = evaluate_answer_correctness(answer, gold_answer)
        complete_eval = evaluate_completeness(answer, gold_answer)
        recall_eval = evaluate_context_recall(query, chunks, gold_answer)
        harm_eval = evaluate_harmfulness_flag(query, answer, patient_state)
    else:
        faith_eval = {"score": 0.0}
        ans_rel_eval = {"score": 3.0}
        correctness_eval = {"score": 3.0}
        complete_eval = {"score": 3.0}
        recall_eval = {"score": 0.0}
        harm_eval = {"score": 0}

    return {
        "id": entry["id"],
        "query": query,
        "expected_tier": expected_tier,
        "reviewed": reviewed,
        "generated_answer": answer,
        "chunks_retrieved_count": len(chunks),
        "triage_tier_correctness": triage_eval["score"],
        "red_flag_recall": red_flag_eval["score"],
        "non_diagnostic_compliance": non_diag_eval["score"],
        "harmfulness_flag": harm_eval["score"],
        "context_relevance": avg_context_relevance,
        "context_rank_correlation": rank_corr,
        "context_recall": recall_eval["score"],
        "faithfulness": faith_eval["score"],
        "answer_relevance": ans_rel_eval["score"],
        "answer_correctness": correctness_eval["score"],
        "completeness": complete_eval["score"],
    }


def run_gold_evaluation(gold_file: str = "data/eval_gold_set.json", output_csv: Optional[str] = None, limit: Optional[int] = None) -> None:
    """Loads gold dataset, runs batch evaluation, and writes report."""
    gold_path = os.path.join(PROJECT_ROOT, gold_file)
    if not os.path.exists(gold_path):
        logger.error(f"Gold evaluation dataset not found at {gold_path}")
        return

    with open(gold_path, "r", encoding="utf-8") as f:
        gold_entries = json.load(f)

    if limit:
        gold_entries = gold_entries[:limit]

    logger.info(f"Starting Offline Gold Evaluation on {len(gold_entries)} entries...")

    results = []
    start_time = time.time()

    for idx, entry in enumerate(gold_entries, 1):
        logger.info(f"[{idx}/{len(gold_entries)}] Evaluating {entry['id']} ({entry.get('expected_tier')})...")
        try:
            res = evaluate_single_gold_entry(entry)
            results.append(res)
        except Exception as err:
            logger.error(f"Failed evaluating {entry.get('id')}: {err}")

    elapsed = time.time() - start_time
    logger.info(f"Completed evaluation of {len(results)} entries in {elapsed:.2f}s")

    # Write CSV
    if not output_csv:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = os.path.join(PROJECT_ROOT, f"data/gold_eval_report_{timestamp}.csv")

    fieldnames = [
        "id", "expected_tier", "reviewed", "triage_tier_correctness", "red_flag_recall",
        "non_diagnostic_compliance", "harmfulness_flag", "context_relevance",
        "context_rank_correlation", "context_recall", "faithfulness",
        "answer_relevance", "answer_correctness", "completeness", "query", "generated_answer"
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    logger.info(f"Detailed evaluation report written to: {output_csv}")

    # Compute aggregate summary
    def avg(metric_key, subset):
        vals = [r[metric_key] for r in subset if r.get(metric_key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    trusted = [r for r in results if r.get("reviewed")]
    provisional = [r for r in results if not r.get("reviewed")]

    print("\n" + "="*70)
    print("[EVAL REPORT] MEDIBOT RAG EVALUATION REPORT -- OFFLINE GOLD SET SUMMARY")
    print("="*70)
    print(f"Total Evaluated: {len(results)} (Trusted/Reviewed: {len(trusted)}, Provisional: {len(provisional)})")
    print(f"Duration: {elapsed:.2f}s\n")

    metrics_list = [
        ("Triage Tier Correctness", "triage_tier_correctness", "0-1"),
        ("Red-Flag Recall", "red_flag_recall", "0-1"),
        ("Non-Diagnostic Compliance", "non_diagnostic_compliance", "0-1"),
        ("Harmfulness Flag (0=safe, 1=harmful)", "harmfulness_flag", "0-1"),
        ("Context Relevance (Precision)", "context_relevance", "0-1"),
        ("Context Rank Correlation (Spearman)", "context_rank_correlation", "-1 to 1"),
        ("Context Recall", "context_recall", "0-1"),
        ("Faithfulness", "faithfulness", "0-1"),
        ("Answer Relevance", "answer_relevance", "1-5"),
        ("Answer Correctness", "answer_correctness", "1-5"),
        ("Completeness", "completeness", "1-5"),
    ]

    header = f"{'Metric':<38} | {'Overall':<8} | {'Trusted':<8} | {'Provisional':<8}"
    print(header)
    print("-" * len(header))
    for name, key, _ in metrics_list:
        overall_val = avg(key, results)
        trusted_val = avg(key, trusted)
        prov_val = avg(key, provisional)
        print(f"{name:<38} | {overall_val:>7.3f} | {trusted_val:>7.3f} | {prov_val:>7.3f}")
    print("="*70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MediBot RAG Gold Evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of entries to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Path to output CSV report")
    args = parser.parse_args()

    run_gold_evaluation(output_csv=args.output, limit=args.limit)
