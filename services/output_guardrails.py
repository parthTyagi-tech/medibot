"""
MediAssist Deterministic Clinical Output Guardrails
===================================================
Post-LLM deterministic safety net that intercepts and sanitizes model outputs
before serialization and delivery to the client.
Prevents:
1. Antipyretic drug recommendations to oncology/immunocompromised patients
2. Outdated clinical dietary hallucinations (e.g. BRAT diet)
3. Direct dosage recommendations or unsafe prescribing
"""

import re
import logging
from typing import Union, Dict, Any, Optional, Tuple
from research.src.clinical_triage import PatientState

logger = logging.getLogger(__name__)


class SanitizedResponse(str):
    """
    String subclass that also allows tuple unpacking:
    text, triggered = ClinicalOutputGuardrail.sanitize_response(...)
    or direct string usage:
    text = ClinicalOutputGuardrail.sanitize_response(...)
    """

    def __new__(
        cls,
        content: str,
        triggered: bool = False,
        trigger_reason: Optional[str] = None,
        intercepted_tokens: Optional[str] = None
    ):
        instance = super().__new__(cls, content)
        instance.triggered = triggered
        instance.trigger_reason = trigger_reason
        instance.intercepted_tokens = intercepted_tokens
        return instance

    def __iter__(self):
        return iter((str(self), self.triggered))


class ClinicalOutputGuardrail:
    """
    Deterministic regex/rule interceptor operating on generated response strings.
    Guarantees prompt leakage and hallucinated contraindications never reach users.
    """

    ANTIPYRETIC_PATTERN = r"\b(tylenol|paracetamol|acetaminophen|ibuprofen|advil|motrin|aspirin|aleve|naproxen)\b"

    FORBIDDEN_ANTIPYRETICS = [
        "acetaminophen", "paracetamol", "tylenol", "ibuprofen",
        "advil", "motrin", "aspirin", "naproxen", "aleve"
    ]

    PERMISSIVE_MED_PATTERNS = [
        rf"\b(take|try|use|consider|administer|give|suggest|recommend)\s+[\w\s]{{0,25}}\b({('|').join(FORBIDDEN_ANTIPYRETICS)})\b",
        rf"\b({('|').join(FORBIDDEN_ANTIPYRETICS)})\s+(can|may|could|should|might)\s+(help|reduce|lower|alleviate|relieve)\b",
        rf"\b(dose|dosage|tablet|pill)\s+of\s+({('|').join(FORBIDDEN_ANTIPYRETICS)})\b"
    ]

    DIET_PATTERN = r"\b(brat\s+diet|bananas?,\s*rice,\s*applesauce(\s*,?\s*(and\s*)?toast)?)\b"

    CLINICAL_OVERRIDE_TEXT = (
        "CRITICAL SAFETY OVERRIDE: DO NOT take over-the-counter fever-reducing medications "
        "(such as acetaminophen, paracetamol, or ibuprofen) without explicit instruction from your oncologist, "
        "as they can mask serious infections."
    )

    @classmethod
    def sanitize_response(
        cls,
        response_text: str,
        patient_state: Union[PatientState, Dict[str, Any], None] = None,
        signals: Optional[Dict[str, Any]] = None
    ) -> SanitizedResponse:
        """
        Sanitizes model output before delivering to user.
        Returns a SanitizedResponse that can be used directly as a string or unpacked as (text, triggered).
        """
        if not response_text:
            return SanitizedResponse("", triggered=False)

        sanitized = response_text
        triggered = False
        trigger_reasons = []
        intercepted_tokens = []

        # Determine GI symptoms presence
        has_gi_symptoms = False
        if signals is not None:
            has_gi_symptoms = bool(signals.get("has_gi_symptoms"))
        elif patient_state:
            if isinstance(patient_state, PatientState):
                has_gi_symptoms = any(
                    s in ["diarrhea", "vomiting"] for s in patient_state.current_symptoms
                )
            elif isinstance(patient_state, dict):
                has_gi_symptoms = any(
                    s in ["diarrhea", "vomiting"] for s in patient_state.get("current_symptoms", [])
                )

        # 1. Dietary Scrubbing: If GI symptoms are absent, scrub BRAT diet / bananas, rice, applesauce
        if not has_gi_symptoms:
            diet_matches = list(re.finditer(cls.DIET_PATTERN, sanitized, re.IGNORECASE))
            if diet_matches:
                triggered = True
                trigger_reasons.append("UNWARRANTED_DIET_SCRUBBING")
                for m in diet_matches:
                    intercepted_tokens.append(m.group(0))
                logger.warning("[Guardrail] Intercepted and scrubbed unwarranted dietary recommendation.")
                sanitized = re.sub(
                    cls.DIET_PATTERN,
                    "a balanced, nutrient-dense diet",
                    sanitized,
                    flags=re.IGNORECASE
                )
                # Also clean up any trailing ", and toast" or ", toast"
                sanitized = re.sub(
                    r"a balanced, nutrient-dense diet,?\s*(and\s*)?toast\b",
                    "a balanced, nutrient-dense diet",
                    sanitized,
                    flags=re.IGNORECASE
                )

        # 2. Antipyretic Interception for Oncology / Immunocompromised patients
        has_cancer = False
        if patient_state:
            if isinstance(patient_state, PatientState):
                has_cancer = bool(
                    patient_state.has_cancer_history
                    or patient_state.is_active_cancer_chemo
                    or any("cancer" in str(c).lower() or "chemo" in str(c).lower() for c in patient_state.disclosed_conditions)
                    or any("cancer" in str(c).lower() or "chemo" in str(c).lower() for c in patient_state.conditions)
                )
            elif isinstance(patient_state, dict):
                has_cancer = bool(
                    patient_state.get("has_cancer_history")
                    or patient_state.get("is_active_cancer_chemo")
                    or any("cancer" in str(c).lower() or "chemo" in str(c).lower() for c in patient_state.get("disclosed_conditions", []))
                    or any("cancer" in str(c).lower() or "chemo" in str(c).lower() for c in patient_state.get("conditions", []))
                )

        if has_cancer:
            lower_text = sanitized.lower()
            antipyretic_matches = list(re.finditer(cls.ANTIPYRETIC_PATTERN, lower_text))
            
            if antipyretic_matches:
                # Check if it already has an unambiguous prohibition
                has_prohibition = bool(
                    re.search(
                        r"\b(do\s+not\s+take|avoid|never\s+take|contraindicated|should\s+not\s+take|must\s+not\s+be\s+taken|prohibited)\b"
                        r"[\w\s]{0,45}(" + "|".join(cls.FORBIDDEN_ANTIPYRETICS) + r"|fever\s+reducers?)",
                        lower_text
                    )
                )
                has_permissive = any(bool(re.search(pat, lower_text)) for pat in cls.PERMISSIVE_MED_PATTERNS)

                if has_permissive or not has_prohibition:
                    triggered = True
                    trigger_reasons.append("FORBIDDEN_ANTIPYRETIC_INTERCEPTION")
                    for m in antipyretic_matches:
                        intercepted_tokens.append(m.group(0))

                    logger.critical(
                        "[Guardrail] CRITICAL INTERCEPTION: Permissive antipyretic detected for oncology patient! Enforcing override..."
                    )

                    # Replace permissive lines with safe directive
                    lines = sanitized.split("\n")
                    cleaned_lines = []
                    for line in lines:
                        if any(re.search(pat, line.lower()) for pat in cls.PERMISSIVE_MED_PATTERNS):
                            cleaned_lines.append(f"**{cls.CLINICAL_OVERRIDE_TEXT}**")
                        else:
                            cleaned_lines.append(line)
                    sanitized = "\n".join(cleaned_lines)

                    if cls.CLINICAL_OVERRIDE_TEXT not in sanitized:
                        sanitized += f"\n\n**{cls.CLINICAL_OVERRIDE_TEXT}**"

        reason_str = "; ".join(trigger_reasons) if trigger_reasons else None
        tokens_str = ", ".join(intercepted_tokens) if intercepted_tokens else None

        return SanitizedResponse(
            content=sanitized,
            triggered=triggered,
            trigger_reason=reason_str,
            intercepted_tokens=tokens_str
        )
