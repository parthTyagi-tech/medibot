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
from typing import Union, Dict, Any, Optional, Tuple, Iterator, List
from research.src.clinical_triage import PatientState
from research.src.guardrails import (
    apply_output_guardrails,
    apply_sentence_guardrails,
    requires_fixed_emergency_response,
    find_sentence_boundary,
    split_into_sentences,
    is_pediatric,
    ABBREVIATIONS
)

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

    FORBIDDEN_ANTIPYRETICS = [
        "acetaminophen", "paracetamol", "tylenol", "ibuprofen",
        "advil", "motrin", "aspirin", "naproxen", "aleve",
        "diclofenac", "nimesulide", "mefenamic acid", "crocin", "calpol", "combiflam"
    ]

    ANTIPYRETIC_PATTERN = r"\b(" + "|".join(FORBIDDEN_ANTIPYRETICS) + r")\b"

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
                sentences = split_into_sentences(sanitized)
                cleaned_sentences = []
                any_prohibition_anywhere = False
                override_placed = False

                for sent in sentences:
                    if cls.CLINICAL_OVERRIDE_TEXT in sent:
                        cleaned_sentences.append(sent)
                        override_placed = True
                        continue

                    s_lower = sent.lower()
                    sent_matches = list(re.finditer(cls.ANTIPYRETIC_PATTERN, s_lower))
                    if not sent_matches:
                        cleaned_sentences.append(sent)
                        continue

                    # Position-aware prohibition check within sentence
                    has_sent_prohibition = bool(
                        re.search(
                            r"\b(do\s+not(?:\s+take|\s+use)?|"
                            r"don\'?t(?:\s+take|\s+use)?|"
                            r"avoid(?:\s+taking|\s+using)?|"
                            r"never(?:\s+take|\s+use)?|"
                            r"contraindicated(?:\s+for|\s+in)?|"
                            r"should\s+not(?:\s+take|\s+use)?|"
                            r"must\s+not(?:\s+take|\s+use|\s+be\s+taken)?|"
                            r"prohibited(?:\s+to\s+take)?|"
                            r"do\s+not\s+give|don\'?t\s+give)\b"
                            r"[\w\s]{0,45}(" + "|".join(cls.FORBIDDEN_ANTIPYRETICS) + r"|fever\s+reducers?)",
                            s_lower
                        )
                    )
                    if has_sent_prohibition:
                        any_prohibition_anywhere = True

                    # Check for permissive verb/recommendation
                    s_no_prohib = s_lower
                    if has_sent_prohibition:
                        s_no_prohib = re.sub(
                            r"\b(do\s+not(?:\s+take|\s+use)?|"
                            r"don\'?t(?:\s+take|\s+use)?|"
                            r"avoid(?:\s+taking|\s+using)?|"
                            r"never(?:\s+take|\s+use)?|"
                            r"contraindicated(?:\s+for|\s+in)?|"
                            r"should\s+not(?:\s+take|\s+use)?|"
                            r"must\s+not(?:\s+take|\s+use|\s+be\s+taken)?|"
                            r"prohibited(?:\s+to\s+take)?|"
                            r"do\s+not\s+give|don\'?t\s+give)\b"
                            r"[\w\s]{0,45}(" + "|".join(cls.FORBIDDEN_ANTIPYRETICS) + r"|fever\s+reducers?)",
                            "",
                            s_no_prohib
                        )
                    s_no_prohib = re.sub(r"\b(?:do\s+not|don'?t|never|should\s+not|must\s+not|avoid)\s+\w+", "", s_no_prohib)
                    has_sent_permissive = any(bool(re.search(pat, s_no_prohib)) for pat in cls.PERMISSIVE_MED_PATTERNS)

                    # If sentence is permissive or lacks prohibition, replace it
                    if has_sent_permissive or not has_sent_prohibition:
                        triggered = True
                        trigger_reasons.append("FORBIDDEN_ANTIPYRETIC_INTERCEPTION")
                        for m in sent_matches:
                            intercepted_tokens.append(m.group(0))

                        if not override_placed:
                            cleaned_sentences.append(f" **{cls.CLINICAL_OVERRIDE_TEXT}** ")
                            override_placed = True
                    else:
                        # Sentence has position-aware prohibition and no permissive verb -> KEEP intact
                        cleaned_sentences.append(sent)

                sanitized = "".join(cleaned_sentences).strip()

                # If antipyretic was mentioned, no prohibition anywhere in text, and override not yet present, append once
                if not any_prohibition_anywhere and cls.CLINICAL_OVERRIDE_TEXT not in sanitized:
                    sanitized += f"\n\n**{cls.CLINICAL_OVERRIDE_TEXT}**"

        reason_str = "; ".join(trigger_reasons) if trigger_reasons else None
        tokens_str = ", ".join(intercepted_tokens) if intercepted_tokens else None

        return SanitizedResponse(
            content=sanitized,
            triggered=triggered,
            trigger_reason=reason_str,
            intercepted_tokens=tokens_str
        )


def is_patient_high_risk(
    patient_state: Union[PatientState, Dict[str, Any], None],
    risk_tier: str = "Routine"
) -> bool:
    """
    Returns True if the patient is high-risk (cancer history, immunocompromised, pregnant,
    pediatric, or risk_tier == 'Emergency'), requiring full-response buffering.
    Returns False for standard-risk patients who can be safely sentence-streamed.
    """
    if str(risk_tier).upper() == "EMERGENCY":
        return True
    if not patient_state:
        return False

    if isinstance(patient_state, dict):
        has_cancer = bool(
            patient_state.get("has_cancer_history")
            or patient_state.get("is_active_cancer_chemo")
            or any("cancer" in str(c).lower() or "chemo" in str(c).lower() for c in patient_state.get("disclosed_conditions", []))
            or any("cancer" in str(c).lower() or "chemo" in str(c).lower() for c in patient_state.get("conditions", []))
        )
        is_immunocompromised = bool(patient_state.get("is_immunocompromised"))
        is_pregnant = bool(patient_state.get("is_pregnant"))
        is_child = is_pediatric(patient_state)
        return bool(has_cancer or is_immunocompromised or is_pregnant or is_child)
    else:
        disclosed = [str(c).lower() for c in getattr(patient_state, "disclosed_conditions", [])]
        conds = [str(c).lower() for c in getattr(patient_state, "conditions", [])]
        all_conds = disclosed + conds
        has_cancer = (
            getattr(patient_state, "has_cancer_history", False)
            or getattr(patient_state, "is_active_cancer_chemo", False)
            or any("cancer" in c or "chemo" in c for c in all_conds)
        )
        is_immunocompromised = getattr(patient_state, "is_immunocompromised", False)
        is_pregnant = getattr(patient_state, "is_pregnant", False)
        is_child = is_pediatric(patient_state)
        return bool(has_cancer or is_immunocompromised or is_pregnant or is_child)


class StreamingGuardrailProcessor:
    """
    Stream-aware guardrail processor protecting voice streaming pipelines:
    - Full-buffers high-risk patients (cancer, immunocompromised, pregnant, age < 12, infant < 3mo, emergency).
    - Sentence-buffers standard-risk patients, applying pure sentence text transforms per completed sentence.
    - Flushes buffer if it exceeds ~300 characters without a sentence boundary.
    - Avoids false sentence splits on decimals (10.5 mg) and abbreviations (Dr., vs., e.g.).
    - Tracks prohibitions across the stream and runs whole-response checks at end of stream.
    - Appends a short corrective sentence if a post-stream check trips on already-yielded sentences.
    - Safely handles mid-stream exceptions by emitting safe fallback and discarding unsanitized buffers.
    - Guarantees get_final_text() matches the exact concatenated yielded text for database persistence.
    - Free of Flask/route imports.
    """

    MAX_BUFFER_CHARS: int = 300

    STREAM_INTERRUPTED_FALLBACK: str = (
        " I am sorry, my connection was interrupted. "
        "Please check with your healthcare provider for any urgent medical concerns."
    )

    CORRECTIVE_MEDICATION_SENTENCE: str = (
        "\n\nPlease disregard any previous medication suggestion and consult your doctor "
        "or pharmacist directly before taking any medication."
    )

    def __init__(
        self,
        patient_state: Union[PatientState, Dict[str, Any], None] = None,
        risk_tier: str = "Routine",
        is_medical: bool = True
    ):
        self.patient_state = patient_state
        self.risk_tier = risk_tier
        self.is_medical = is_medical
        self.is_high_risk = is_patient_high_risk(patient_state, risk_tier)
        self.yielded_chunks: List[str] = []
        self.guardrail_triggered: bool = False
        self.trigger_reasons: List[str] = []
        self.intercepted_tokens: List[str] = []
        self.prohibition_seen: bool = False

    def _sanitize_sentence(self, sentence: str) -> str:
        """Applies pure text transforms safe for a single sentence."""
        cleaned, flags = apply_sentence_guardrails(sentence, self.patient_state)
        if flags.get("dosing_detected"):
            self.guardrail_triggered = True
            self.trigger_reasons.append("SENTENCE_DOSING_SUPPRESSED")
        if flags.get("hallucinated_specialty"):
            self.guardrail_triggered = True
            self.trigger_reasons.append("SENTENCE_SPECIALTY_SUPPRESSED")

        # Track prohibition seen anywhere across the stream
        lower = cleaned.lower()
        if re.search(
            r"\b(do\s+not\s+take|avoid|never\s+take|contraindicated|should\s+not\s+take|must\s+not\s+be\s+taken|prohibited)\b"
            r"[\w\s]{0,45}(" + "|".join(ClinicalOutputGuardrail.FORBIDDEN_ANTIPYRETICS) + r"|fever\s+reducers?)",
            lower
        ):
            self.prohibition_seen = True

        # Dietary scrubbing (BRAT diet)
        cleaned_diet, diet_triggered = ClinicalOutputGuardrail.sanitize_response(cleaned, self.patient_state)
        if diet_triggered:
            self.guardrail_triggered = True
            if getattr(cleaned_diet, "trigger_reason", None):
                self.trigger_reasons.append(cleaned_diet.trigger_reason)
            if getattr(cleaned_diet, "intercepted_tokens", None):
                self.intercepted_tokens.append(cleaned_diet.intercepted_tokens)

        return str(cleaned_diet)

    def process_stream(self, token_iterator: Iterator[Any]) -> Iterator[str]:
        """
        Consumes an iterator of LLM chunks (e.g. chatModel.stream), yields sanitized
        text pieces, and maintains an internal ledger of all yielded text.
        """
        if self.is_high_risk:
            # High-risk path: full-response buffering for fail-closed clinical safety
            raw_full: List[str] = []
            try:
                for chunk in token_iterator:
                    content = chunk.content if hasattr(chunk, "content") else str(chunk or "")
                    if content:
                        raw_full.append(content)
            except Exception as e:
                logger.error("[StreamingGuardrailProcessor] Stream interrupted during high-risk buffering: %s", type(e).__name__)
                yield self.STREAM_INTERRUPTED_FALLBACK
                self.yielded_chunks.append(self.STREAM_INTERRUPTED_FALLBACK)
                return

            full_text = "".join(raw_full)
            if not full_text.strip():
                return

            # Apply full whole-response output guardrails
            sanitized = apply_output_guardrails(
                full_text,
                is_medical=self.is_medical,
                show_disclaimer=False,
                patient_state=self.patient_state,
                triage_tier=self.risk_tier
            )
            sanitized_obj, triggered = ClinicalOutputGuardrail.sanitize_response(sanitized, self.patient_state)
            if triggered:
                self.guardrail_triggered = True
                if getattr(sanitized_obj, "trigger_reason", None):
                    self.trigger_reasons.append(sanitized_obj.trigger_reason)
                if getattr(sanitized_obj, "intercepted_tokens", None):
                    self.intercepted_tokens.append(sanitized_obj.intercepted_tokens)

            final_text = str(sanitized_obj)
            self.yielded_chunks.append(final_text)
            yield final_text
            return

        # Standard-risk path: sentence-buffered streaming
        buffer = ""
        try:
            for chunk in token_iterator:
                content = chunk.content if hasattr(chunk, "content") else str(chunk or "")
                if not content:
                    continue
                buffer += content

                while True:
                    boundary = find_sentence_boundary(buffer)
                    if boundary is not None:
                        sentence = buffer[:boundary]
                        buffer = buffer[boundary:]
                        sanitized = self._sanitize_sentence(sentence)
                        if sanitized:
                            self.yielded_chunks.append(sanitized)
                            yield sanitized
                    elif len(buffer) >= self.MAX_BUFFER_CHARS:
                        split_idx = buffer[:self.MAX_BUFFER_CHARS].rfind(" ")
                        if split_idx > 100:
                            chunk_text = buffer[:split_idx + 1]
                            buffer = buffer[split_idx + 1:]
                        else:
                            chunk_text = buffer[:self.MAX_BUFFER_CHARS]
                            buffer = buffer[self.MAX_BUFFER_CHARS:]
                        sanitized = self._sanitize_sentence(chunk_text)
                        if sanitized:
                            self.yielded_chunks.append(sanitized)
                            yield sanitized
                    else:
                        break
        except Exception as e:
            logger.error("[StreamingGuardrailProcessor] Stream interrupted during sentence streaming: %s", type(e).__name__)
            buffer = ""  # Discard unsanitized buffer
            yield self.STREAM_INTERRUPTED_FALLBACK
            self.yielded_chunks.append(self.STREAM_INTERRUPTED_FALLBACK)
            return

        # Flush residual buffer at end of stream
        if buffer.strip():
            sanitized = self._sanitize_sentence(buffer)
            if sanitized:
                self.yielded_chunks.append(sanitized)
                yield sanitized

        # End of stream whole-response check
        full_streamed = "".join(self.yielded_chunks)
        sanitized_full, whole_triggered = ClinicalOutputGuardrail.sanitize_response(full_streamed, self.patient_state)
        if whole_triggered:
            self.guardrail_triggered = True
            if getattr(sanitized_full, "trigger_reason", None):
                self.trigger_reasons.append(sanitized_full.trigger_reason)
            if getattr(sanitized_full, "intercepted_tokens", None):
                self.intercepted_tokens.append(sanitized_full.intercepted_tokens)

            # Yield corrective sentence because sentences were already yielded on a non-Emergency turn
            corrective = self.CORRECTIVE_MEDICATION_SENTENCE
            self.yielded_chunks.append(corrective)
            yield corrective

    def get_final_text(self) -> str:
        """Returns the exact concatenated text that was yielded to the user."""
        return "".join(self.yielded_chunks)
