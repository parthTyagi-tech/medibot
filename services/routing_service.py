"""
Unified Routing Orchestrator for MediAssist.
============================================
Provides ONE shared function `route_query()` called by /get, /voice_chat, and voice streaming.
Enforces the canonical pipeline:
1. Input injection regex check (is_prompt_injection)
2. Emergency regex check (detect_medical_emergency)
3. INTENT_CLASSIFIER_MODE check (llm | regex)
   - In regex mode: uses old classifier for complete rollback fidelity.
   - In llm mode: the LLM classifier is the only scope gate.
4. Confidence gate: low-confidence (<0.7) non-medical or greeting escalates to MEDICAL.
5. Emergency lockdown: if active_emergency or emergency_override_served is set,
   GREETING is treated as MEDICAL (no cheerful greeting during emergency).
6. Non-PII audit telemetry logging.
"""

import os
import re
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from research.src.guardrails import (
    is_prompt_injection,
    detect_medical_emergency,
    NON_MEDICAL_REFUSAL
)
from services.classifier_config import get_classifier_config
from services.llm_classifier import (
    classify_query,
    generate_greeting,
    _is_greeting_sane,
    get_static_greeting_fallback,
    ClassifierResult
)
from services.audit_logger import AuditLogger

logger = logging.getLogger("routing-service")

# Anchored whole-message regexes for account actions and memory recall (Amendment 4 & 8)
# Evaluated strictly after injection & emergency, and NEVER during emergency lockdown.
ANCHORED_ACCOUNT_REGEX = re.compile(
    r"^\s*(delete\s+(my\s+)?account|delete\s+this\s+consultation|delete\s+chat|log\s*out|sign\s*out)\s*[.?!]*$",
    re.IGNORECASE
)
ANCHORED_MEMORY_REGEX = re.compile(
    r"^\s*(what\s+symptoms\s+did\s+i\s+mention|what\s+do\s+you\s+know\s+about\s+me|what\s+did\s+i\s+tell\s+you|"
    r"what\s+was\s+my\s+allergy|what\s+medications?\s+am\s+i\s+taking|my\s+past\s+consultations|"
    r"recall\s+my\s+symptoms|recall\s+my\s+medical\s+history|remember\s+what\s+i\s+said|what\s+is\s+my\s+diagnosis)\s*[.?!]*$",
    re.IGNORECASE
)


@dataclass
class RouteDecision:
    """Canonical decision object returned by route_query."""
    action: str        # "BLOCKED_INJECTION", "BLOCKED_EMERGENCY", "ACCOUNT_ACTION", "MEMORY_RECALL", "NON_MEDICAL", "GREETING", "MEDICAL"
    intent: str        # "INJECTION", "EMERGENCY", "account_action", "memory_recall", "NON_MEDICAL", "GREETING", "MEDICAL"
    response: Optional[str] = None  # Verbatim refusal or guidance if blocked/non-medical/account
    confidence: float = 1.0
    language: str = "en"
    fallback: bool = False
    mode: str = "llm"  # "llm" or "regex"
    raw_result: Optional[Any] = None


def route_query(
    msg: str,
    session: Optional[Any] = None,
    patient_state: Optional[Any] = None,
    recent_turns: Optional[List[Dict[str, str]]] = None,
    session_id: Optional[str] = None,
    triage_active: Optional[bool] = None
) -> RouteDecision:
    """
    Unified query routing function shared by /get, /voice_chat, and voice streaming.
    Guaranteed to never raise an exception.
    """
    clean_msg = (msg or "").strip()
    cfg = get_classifier_config()

    if session and not session_id:
        try:
            if isinstance(session, dict):
                session_id = session.get("chat_session_id") or session.get("session_id")
            else:
                session_id = getattr(session, "id", None) or getattr(session, "session_id", None)
        except Exception:
            session_id = None

    if not clean_msg:
        return RouteDecision(
            action="NON_MEDICAL",
            intent="NON_MEDICAL",
            response="Please enter a message.",
            confidence=1.0,
            language="en",
            fallback=False,
            mode=cfg.mode
        )

    # 1. Input injection regex tripwire
    if is_prompt_injection(clean_msg):
        refusal = (
            "I cannot fulfill this request. I am MediAssist, a medical AI assistant, "
            "and my instructions and safety protocols cannot be overridden."
        )
        return RouteDecision(
            action="BLOCKED_INJECTION",
            intent="INJECTION",
            response=refusal,
            confidence=1.0,
            language="en",
            fallback=False,
            mode="regex"
        )

    # 2. Deterministic emergency red-flag detector
    is_em, em_msg = detect_medical_emergency(clean_msg)
    if is_em:
        return RouteDecision(
            action="BLOCKED_EMERGENCY",
            intent="EMERGENCY",
            response=em_msg,
            confidence=1.0,
            language="en",
            fallback=False,
            mode="regex"
        )

    # Determine if patient is in active emergency state
    is_emergency_state = bool(
        (patient_state and (
            (isinstance(patient_state, dict) and (
                patient_state.get("active_emergency") or
                patient_state.get("emergency_override_served") or
                patient_state.get("active_emergency_topic") or
                (patient_state.get("risk_tier") == "Emergency")
            )) or
            (not isinstance(patient_state, dict) and (
                getattr(patient_state, "active_emergency", None) or
                getattr(patient_state, "emergency_override_served", False) or
                getattr(patient_state, "active_emergency_topic", None) or
                (getattr(patient_state, "risk_tier", None) == "Emergency")
            ))
        )) or
        (session and isinstance(session, dict) and (
            session.get("active_emergency") or
            (session.get("patient_state") and isinstance(session.get("patient_state"), dict) and session["patient_state"].get("active_emergency"))
        ))
    )

    in_emergency_lockdown = bool(
        is_emergency_state or
        (patient_state and (
            getattr(patient_state, "active_emergency", None) or
            getattr(patient_state, "emergency_override_served", False)
        ))
    )

    # Determine if triage is active
    is_active_triage = bool(
        (triage_active is True) or
        is_emergency_state or
        (patient_state and (
            (isinstance(patient_state, dict) and (
                patient_state.get("triage_active") or
                (patient_state.get("risk_tier") and patient_state.get("risk_tier") not in ("Normal", "Informational")) or
                bool(patient_state.get("current_symptoms")) or
                bool(patient_state.get("reported_symptoms_detail"))
            )) or
            (not isinstance(patient_state, dict) and (
                getattr(patient_state, "triage_active", False) or
                (getattr(patient_state, "risk_tier", None) and patient_state.risk_tier not in ("Normal", "Informational")) or
                bool(getattr(patient_state, "current_symptoms", None)) or
                bool(getattr(patient_state, "reported_symptoms_detail", None))
            ))
        ))
    )

    # Follow-up Optimization: If triage_active or emergency state is set, return MEDICAL
    # without calling the classifier (logged as SKIPPED_CONTEXT).
    if is_active_triage or is_emergency_state:
        logger.info("[RoutingService] SKIPPED_CONTEXT: triage_active or emergency state is set; returning MEDICAL without calling classifier")
        AuditLogger.log_classifier_routing(
            session_id=str(session_id or "unknown"),
            intent="MEDICAL",
            confidence=1.0,
            latency_ms=0.0,
            fallback_used=False,
            mode="SKIPPED_CONTEXT"
        )
        return RouteDecision(
            action="MEDICAL",
            intent="MEDICAL",
            response=None,
            confidence=1.0,
            language="en",
            fallback=False,
            mode=cfg.mode,
            raw_result={"skipped_reason": "SKIPPED_CONTEXT"}
        )

    # 3. Anchored ACCOUNT_PHRASES / MEMORY_PHRASES (Amendment 4 & 8)
    # Strictly after injection and emergency checks, and NEVER during emergency lockdown.
    if not in_emergency_lockdown:
        if ANCHORED_ACCOUNT_REGEX.match(clean_msg):
            return RouteDecision(
                action="ACCOUNT_ACTION",
                intent="account_action",
                response="Please use the account controls available in the navigation bar to manage your account or consultation history.",
                confidence=1.0,
                language="en",
                fallback=False,
                mode=cfg.mode
            )
        if ANCHORED_MEMORY_REGEX.match(clean_msg):
            return RouteDecision(
                action="MEMORY_RECALL",
                intent="memory_recall",
                response=None,
                confidence=1.0,
                language="en",
                fallback=False,
                mode=cfg.mode
            )

    # 4. Check INTENT_CLASSIFIER_MODE (llm | regex)
    classifier_mode = cfg.mode

    if classifier_mode == "regex":
        # Rollback path: use old classifier
        try:
            from research.src.intent_classifier import _classify_intent_regex
            old_intent = _classify_intent_regex(None, clean_msg)
        except Exception as e:
            exc_cls = e.__class__.__name__
            status = getattr(e, "status_code", "none")
            logger.warning(f"[RoutingService] Regex classify error class={exc_cls} status={status}")
            old_intent = "medical_query"

        if old_intent in ("general_chat", "non_medical_code"):
            resolved_intent = "NON_MEDICAL"
        elif old_intent == "greeting":
            if in_emergency_lockdown:
                logger.info("[RoutingService] Emergency lockdown: overriding regex GREETING to MEDICAL")
                resolved_intent = "MEDICAL"
            else:
                resolved_intent = "GREETING"
        elif old_intent == "account_action":
            resolved_intent = "account_action"
        elif old_intent == "memory_recall":
            resolved_intent = "memory_recall"
        else:
            resolved_intent = "MEDICAL"

        AuditLogger.log_classifier_routing(
            session_id=str(session_id or "unknown"),
            intent=resolved_intent,
            confidence=1.0,
            latency_ms=0.0,
            fallback_used=False,
            mode="regex"
        )

        resp = None
        if resolved_intent == "NON_MEDICAL":
            resp = NON_MEDICAL_REFUSAL
        elif resolved_intent == "account_action":
            resp = "Please use the account controls available in the navigation bar to manage your account or consultation history."

        action_name = "ACCOUNT_ACTION" if resolved_intent == "account_action" else (
            "MEMORY_RECALL" if resolved_intent == "memory_recall" else resolved_intent
        )

        return RouteDecision(
            action=action_name,
            intent=resolved_intent,
            response=resp,
            confidence=1.0,
            language="en",
            fallback=False,
            mode="regex"
        )

    # 5. LLM Mode: LLM classifier is the ONLY scope gate
    triage_active = bool(
        patient_state and (
            in_emergency_lockdown or
            (getattr(patient_state, "risk_tier", None) and patient_state.risk_tier != "Normal") or
            bool(getattr(patient_state, "current_symptoms", None))
        )
    )

    try:
        classifier_res: ClassifierResult = classify_query(
            user_text=clean_msg,
            recent_turns=recent_turns,
            triage_active=triage_active
        )
    except Exception as e:
        exc_cls = e.__class__.__name__
        status = getattr(e, "status_code", "none")
        logger.warning(f"[RoutingService] classify_query error class={exc_cls} status={status}")
        classifier_res = ClassifierResult(
            intent="MEDICAL",
            language="en",
            confidence=0.0,
            greeting_reply="",
            raw_response={"fail_safe": True},
            fallback=True,
            latency_ms=0.0
        )

    # Confidence gate: applied in ONE place only (services/routing_service.py) via cfg.min_confidence (Amendment 5)
    resolved_intent = classifier_res.intent
    if resolved_intent in ("NON_MEDICAL", "GREETING") and classifier_res.confidence < cfg.min_confidence:
        logger.info(
            f"[RoutingService] Low confidence {classifier_res.confidence} (< {cfg.min_confidence}) for {resolved_intent}. Escalating to MEDICAL."
        )
        resolved_intent = "MEDICAL"

    # Emergency lockdown check: No cheerful greeting during emergency
    if in_emergency_lockdown and resolved_intent == "GREETING":
        logger.info("[RoutingService] Emergency lockdown: overriding GREETING to MEDICAL")
        resolved_intent = "MEDICAL"

    # Log telemetry
    AuditLogger.log_classifier_routing(
        session_id=str(session_id or "unknown"),
        intent=resolved_intent,
        confidence=classifier_res.confidence,
        latency_ms=classifier_res.latency_ms,
        fallback_used=classifier_res.fallback,
        mode="llm"
    )

    return RouteDecision(
        action=resolved_intent,
        intent=resolved_intent,
        response=NON_MEDICAL_REFUSAL if resolved_intent == "NON_MEDICAL" else None,
        confidence=classifier_res.confidence,
        language=classifier_res.language,
        fallback=classifier_res.fallback,
        mode="llm",
        raw_result=classifier_res
    )
