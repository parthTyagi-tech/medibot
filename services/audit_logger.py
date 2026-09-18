"""
MediAssist Medico-Legal Audit Telemetry Logger
==============================================
Structured JSON audit logger for clinical decision-support compliance,
risk management, and auditable event tracking.
Records:
1. EMERGENCY_TRIGGERED: Acute clinical triage escalations
2. CAREGIVER_INTERCEPT: Third-party family triage routing
3. GUARDRAIL_OVERRIDE: Post-LLM deterministic interceptor firings
4. TTL_EXPIRED: Automatic 12h emergency lifecycle resets
5. EMERGENCY_RESOLVED: Affirmative physician/ER stand-down disclosures
"""

import os
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

logger = logging.getLogger("medico-legal-audit")

AUDIT_LOG_DIR = os.getenv("AUDIT_LOG_DIR", "logs")
AUDIT_LOG_FILE = os.path.join(AUDIT_LOG_DIR, "audit.log")

_audit_lock = threading.Lock()


class AuditLogger:
    """
    Thread-safe structured JSON audit logger writing to disk for regulatory & audit compliance.
    """

    LOG_PATH = AUDIT_LOG_FILE

    @classmethod
    def _ensure_dir(cls) -> None:
        os.makedirs(os.path.dirname(cls.LOG_PATH), exist_ok=True)

    @classmethod
    def log_event(
        cls,
        session_id: str,
        event_type: str,
        condition: str,
        rule_triggered: str,
        raw_tokens_intercepted: Optional[str] = None,
        client_ip_hash: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Logs a structured audit telemetry record adhering to standard schema.
        """
        cls._ensure_dir()

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": str(session_id or "unknown"),
            "event_type": str(event_type),
            "condition": str(condition),
            "rule_triggered": str(rule_triggered)
        }

        if raw_tokens_intercepted is not None:
            record["raw_tokens_intercepted"] = str(raw_tokens_intercepted)
        if client_ip_hash is not None:
            record["client_ip_hash"] = str(client_ip_hash)

        json_line = json.dumps(record)

        try:
            with _audit_lock:
                with open(cls.LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json_line + "\n")
            logger.info(f"[AuditTelemetry] {event_type} - {condition} - {rule_triggered}")
        except Exception as e:
            logger.error(f"[AuditTelemetry] Failed to write audit record: {e}")

        return record

    @classmethod
    def log_emergency_triggered(
        cls,
        session_id: str,
        condition: str = "FEBRILE_NEUTROPENIA",
        rule_triggered: str = "ONCOLOGY_FEVER_INTERSECTION",
        client_ip_hash: Optional[str] = None
    ) -> Dict[str, Any]:
        return cls.log_event(
            session_id=session_id,
            event_type="EMERGENCY_TRIGGERED",
            condition=condition,
            rule_triggered=rule_triggered,
            client_ip_hash=client_ip_hash
        )

    @classmethod
    def log_caregiver_intercept(
        cls,
        session_id: str,
        condition: str = "FEBRILE_NEUTROPENIA_CAREGIVER",
        rule_triggered: str = "THIRD_PARTY_ONCOLOGY_FEVER",
        client_ip_hash: Optional[str] = None
    ) -> Dict[str, Any]:
        return cls.log_event(
            session_id=session_id,
            event_type="CAREGIVER_INTERCEPT",
            condition=condition,
            rule_triggered=rule_triggered,
            client_ip_hash=client_ip_hash
        )

    @classmethod
    def log_guardrail_override(
        cls,
        session_id: str,
        condition: str,
        rule_triggered: str,
        raw_tokens_intercepted: Optional[str] = None,
        client_ip_hash: Optional[str] = None
    ) -> Dict[str, Any]:
        return cls.log_event(
            session_id=session_id,
            event_type="GUARDRAIL_OVERRIDE",
            condition=condition,
            rule_triggered=rule_triggered,
            raw_tokens_intercepted=raw_tokens_intercepted,
            client_ip_hash=client_ip_hash
        )

    @classmethod
    def log_ttl_expired(
        cls,
        session_id: str,
        condition: str = "FEBRILE_NEUTROPENIA",
        rule_triggered: str = "12H_TTL_POLICY",
        client_ip_hash: Optional[str] = None
    ) -> Dict[str, Any]:
        return cls.log_event(
            session_id=session_id,
            event_type="TTL_EXPIRED",
            condition=condition,
            rule_triggered=rule_triggered,
            client_ip_hash=client_ip_hash
        )

    @classmethod
    def log_emergency_resolved(
        cls,
        session_id: str,
        condition: str = "FEBRILE_NEUTROPENIA",
        rule_triggered: str = "AFFIRMATIVE_CLINICAL_STAND_DOWN",
        client_ip_hash: Optional[str] = None
    ) -> Dict[str, Any]:
        return cls.log_event(
            session_id=session_id,
            event_type="EMERGENCY_RESOLVED",
            condition=condition,
            rule_triggered=rule_triggered,
            client_ip_hash=client_ip_hash
        )

    @classmethod
    def get_recent_audit_events(cls, limit: int = 50) -> List[Dict[str, Any]]:
        """Reads recent audit telemetry records from the audit log file."""
        if not os.path.exists(cls.LOG_PATH):
            return []

        events = []
        try:
            with _audit_lock:
                with open(cls.LOG_PATH, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines[-limit:]:
                        if line.strip():
                            events.append(json.loads(line.strip()))
        except Exception as e:
            logger.error(f"[AuditTelemetry] Error reading audit records: {e}")

        return events
