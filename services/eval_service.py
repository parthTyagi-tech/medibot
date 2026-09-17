"""
MediAssist Evaluation Orchestrator Service
==========================================
Coordinates evaluation runs for live traffic (online shadow eval) and deterministic safety checks.
- Path B: Online shadow eval (reference-free metrics only)
- Safety check: Rule-based fast evaluation on all turns (Change 2)
- Dispatches email alerts on critical clinical safety failures (Change 5)
- Persists metric scores to EvalResult database table
- Optionally logs feedback to LangSmith if API key is provided
"""

import os
import logging
from typing import List, Dict, Any, Optional

from research.src.eval_metrics import (
    evaluate_rule_based_safety,
    evaluate_context_relevance,
    calculate_spearman_rank_correlation,
    evaluate_faithfulness,
    evaluate_answer_relevance,
    evaluate_harmfulness_flag,
)
from services.email_service import send_eval_alert

logger = logging.getLogger(__name__)

# Initialize LangSmith client gracefully with lazy lookup
_langsmith_client = None

def _get_langsmith_client():
    global _langsmith_client
    if _langsmith_client is not None:
        return _langsmith_client
    if os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY"):
        try:
            from langsmith import Client
            _langsmith_client = Client()
        except Exception as e:
            logger.debug(f"[EvalService] LangSmith client initialization skipped: {e}")
    return _langsmith_client


def _save_eval_results_to_db(results: List[Dict[str, Any]], message_id: Optional[int] = None) -> None:
    """Persists evaluation results into the EvalResult SQLAlchemy table."""
    try:
        from app import app
        from research.src.auth import db, EvalResult

        with app.app_context():
            for r in results:
                eval_row = EvalResult(
                    message_id=message_id,
                    metric_name=r.get("metric", "unknown"),
                    score=r.get("score"),
                    rationale=r.get("rationale"),
                    is_rule_based=r.get("is_rule_based", False),
                    severity=r.get("severity", "normal"),
                    langsmith_run_id=r.get("langsmith_run_id")
                )
                db.session.add(eval_row)
            db.session.commit()
    except Exception as exc:
        logger.error(f"[EvalService] Failed to save eval results to database: {exc}")


def _log_to_langsmith(results: List[Dict[str, Any]], query: str, answer: str, message_id: Optional[int] = None) -> Optional[str]:
    """Creates a LangSmith run and attaches evaluation metric feedback."""
    client = _get_langsmith_client()
    if client is None:
        return None

    try:
        import uuid
        project_name = os.getenv("LANGCHAIN_PROJECT", "medibot-eval")
        run_id = uuid.uuid4()
        run_name = f"eval_turn_msg_{message_id}" if message_id else "eval_turn"

        client.create_run(
            id=run_id,
            name=run_name,
            run_type="chain",
            inputs={"query": query},
            outputs={"answer": answer},
            project_name=project_name,
            extra={"message_id": message_id}
        )

        for r in results:
            r["langsmith_run_id"] = str(run_id)
            try:
                client.create_feedback(
                    run_id=run_id,
                    key=r.get("metric"),
                    score=r.get("score"),
                    comment=r.get("rationale", "")
                )
            except Exception as fe:
                logger.debug(f"[EvalService] LangSmith feedback notice for {r.get('metric')}: {fe}")

        logger.info(f"[EvalService] Logged eval run {run_id} with {len(results)} metrics to LangSmith project '{project_name}'")
        return str(run_id)
    except Exception as e:
        logger.debug(f"[EvalService] LangSmith logging notice: {e}")
        return None


def run_safety_eval(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Executes lightweight rule-based safety evaluation on every turn (Change 2).
    0 LLM calls, deterministic, fast.
    Dispatches immediate alert email if red_flag_recall fails (Change 5).
    """
    query = payload.get("query", "")
    response = payload.get("response") or payload.get("generated_answer") or ""
    message_id = payload.get("message_id")
    red_flag_scenario = payload.get("red_flag_scenario")
    expected_tier = payload.get("expected_tier")

    patient_state = None
    state_dict = payload.get("patient_state")
    if state_dict:
        try:
            from research.src.clinical_triage import PatientState
            patient_state = PatientState.from_dict(state_dict) if isinstance(state_dict, dict) else state_dict
        except Exception:
            patient_state = None

    results = evaluate_rule_based_safety(
        query=query,
        response=response,
        patient_state=patient_state,
        red_flag_scenario=red_flag_scenario,
        expected_tier=expected_tier
    )

    # Check for critical safety failure
    for r in results:
        if r.get("metric") == "red_flag_recall" and r.get("score") == 0:
            logger.critical(f"[SAFETY ALERT] Critical red-flag recall failure on msg {message_id}: {r.get('rationale')}")
            # Change 5: Synchronous email notification
            html_body = f"""
            <h3>🚨 MediAssist Clinical Safety Alert: Red-Flag Missed</h3>
            <p><strong>Message ID:</strong> {message_id}</p>
            <p><strong>User Query:</strong> {query}</p>
            <p><strong>Bot Response:</strong> {response}</p>
            <p><strong>Failure Rationale:</strong> {r.get('rationale')}</p>
            <p><strong>Applicable Scenario:</strong> {red_flag_scenario or 'Triage Matrix Override'}</p>
            """
            send_eval_alert("Red-Flag Emergency Escalation Missed", html_body)

    _save_eval_results_to_db(results, message_id)
    return results


def run_turn_eval(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Executes full online shadow evaluation on sampled medical query turns (Change 1 Path B).
    Includes reference-free retrieval, generation, and clinical safety metrics.
    """
    query = payload.get("query", "")
    generated_answer = payload.get("generated_answer") or payload.get("response") or ""
    retrieved_chunks = payload.get("retrieved_chunks") or []
    message_id = payload.get("message_id")

    patient_state = None
    state_dict = payload.get("patient_state")
    if state_dict:
        try:
            from research.src.clinical_triage import PatientState
            patient_state = PatientState.from_dict(state_dict) if isinstance(state_dict, dict) else state_dict
        except Exception:
            patient_state = None

    results: List[Dict[str, Any]] = []

    # 1. Retrieval Metrics
    chunk_scores: List[float] = []
    if retrieved_chunks:
        for chunk in retrieved_chunks:
            chunk_text = chunk if isinstance(chunk, str) else str(chunk)
            rel_res = evaluate_context_relevance(query, chunk_text)
            chunk_scores.append(rel_res["score"])

        # Aggregate context relevance
        avg_relevance = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0.0
        results.append({
            "metric": "context_relevance",
            "score": round(avg_relevance, 4),
            "rationale": f"Evaluated {len(chunk_scores)} chunks; {sum(1 for s in chunk_scores if s > 0)}/{len(chunk_scores)} relevant.",
            "is_rule_based": False,
            "severity": "normal"
        })

        # Metric #3: Context Rank Correlation (Change 4: Pure Python Spearman calculation)
        rank_corr = calculate_spearman_rank_correlation(chunk_scores)
        results.append({
            "metric": "context_rank_correlation",
            "score": rank_corr,
            "rationale": f"Spearman rank correlation of retrieved vs relevance order: {rank_corr}",
            "is_rule_based": True,
            "severity": "normal"
        })

    # 2. Generation Metrics (Reference-free)
    results.append(evaluate_faithfulness(retrieved_chunks, generated_answer))
    results.append(evaluate_answer_relevance(query, generated_answer))

    # 3. Clinical Safety Metrics
    safety_results = evaluate_rule_based_safety(
        query=query,
        response=generated_answer,
        patient_state=patient_state
    )
    results.extend(safety_results)

    harm_res = evaluate_harmfulness_flag(query, generated_answer, patient_state)
    results.append(harm_res)

    # Check for critical failures requiring immediate alert (Change 5)
    critical_failures = [r for r in results if r.get("severity") == "critical" or (r.get("metric") == "harmfulness_flag" and r.get("score") == 1)]
    if critical_failures:
        failure_desc = "<br/>".join([f"<strong>{cf['metric']}:</strong> {cf['rationale']}" for cf in critical_failures])
        logger.critical(f"[SAFETY ALERT] Critical evaluation failures on msg {message_id}: {failure_desc}")
        html_body = f"""
        <h3>🚨 MediAssist Clinical Safety Alert: Critical Eval Failure</h3>
        <p><strong>Message ID:</strong> {message_id}</p>
        <p><strong>User Query:</strong> {query}</p>
        <p><strong>Bot Response:</strong> {generated_answer}</p>
        <p><strong>Critical Issues:</strong></p>
        <div>{failure_desc}</div>
        """
        send_eval_alert("Critical Safety Eval Failure Detected", html_body)

    # Persist and optionally log to LangSmith
    _log_to_langsmith(results, query, generated_answer, message_id)
    _save_eval_results_to_db(results, message_id)

    return results
