"""
Production LLM Query Classifier & Dynamic Greeter for MediAssist.
=================================================================
Provides high-speed query intent classification and dynamic, non-repetitive
greeting generation via Groq inference with multi-key failover.

Guarantees:
1. Reuses existing Groq key pool from services.ai_service.
2. Strict JSON response schema in classify_query():
   {"intent": "MEDICAL" | "GREETING" | "NON_MEDICAL", "language": "<ISO>",
    "confidence": 0.0-1.0, "greeting_reply": "<string>"}
3. classify_query() runs at temperature=0.0, max_tokens<=200, timeout=2.5s.
4. generate_greeting() runs at temperature=0.8, top_p=0.9, max_tokens=80, timeout=2.5s.
5. Dynamic greeter incorporates recent_turns history, applies strict safety sanity
   checks (<=300 chars, no "mg", no drug names, no "diagnos"), and falls back to
   a non-repeating static multilingual pool.
6. Never raises: gracefully falls back on any exception or timeout.
"""

import os
import re
import json
import time
import random
import logging
import threading
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

from dotenv import load_dotenv
load_dotenv()

from groq import Groq, RateLimitError, APIStatusError, APITimeoutError

def get_groq_api_keys() -> List[str]:
    keys: List[str] = []
    raw_keys = os.getenv("GROQ_API_KEYS", "")
    for k in raw_keys.split(","):
        k = k.strip().strip('"').strip("'")
        if k and k not in keys:
            keys.append(k)
    raw_key = os.getenv("GROQ_API_KEY", "")
    for k in raw_key.split(","):
        k = k.strip().strip('"').strip("'")
        if k and k not in keys:
            keys.append(k)
    return keys

from research.src.guardrails import (
    is_code_or_programming_request,
    apply_output_guardrails,
    UNIVERSALLY_PROHIBITED_DRUGS
)
from research.src.intent_classifier import MEDICAL_KEYWORDS, NON_MEDICAL_KEYWORDS
from research.src.prompt import (
    build_greeter_messages,
    build_routing_classifier_messages,
    ROUTING_CLASSIFIER_SYSTEM_PROMPT
)
from services.classifier_config import get_classifier_config

logger = logging.getLogger("llm-classifier")

VALID_INTENTS = {"MEDICAL", "GREETING", "NON_MEDICAL"}
ALLOWED_LANGUAGES = {"en", "es", "fr", "de", "it", "nl", "ja"}
CLASSIFY_MAX_TOKENS = 400
GREETING_MAX_TOKENS = 80

# ─────────────────────────────────────────────────────────────
# Static Fallback Pools (4-5 variants per language)
# ─────────────────────────────────────────────────────────────
STATIC_GREETING_FALLBACKS: Dict[str, List[str]] = {
    "en": [
        "Hello! How can I assist you with your health today?",
        "Hi there! What health or wellness questions can I help you with?",
        "Good to see you! How are you feeling today?",
        "Hello! I'm here to support you with any medical or health questions.",
        "Hi! Feel free to share any symptoms or health concerns on your mind."
    ],
    "es": [
        "¡Hola! ¿Cómo puedo ayudarte con tu salud hoy?",
        "¡Hola! ¿En qué puedo orientarte hoy sobre tu bienestar?",
        "¡Qué gusto saludarte! ¿Tienes alguna consulta sobre tu salud?",
        "¡Hola! Cuéntame qué dudas de salud o síntomas tienes hoy.",
        "¡Bienvenido! Estoy aquí para responder a tus preguntas médicas."
    ],
    "fr": [
        "Bonjour ! Comment puis-je vous aider concernant votre santé aujourd'hui ?",
        "Bonjour ! Quelles questions de santé puis-je aborder avec vous ?",
        "Ravi de vous accueillir ! Comment vous sentez-vous aujourd'hui ?",
        "Bonjour ! N'hésitez pas à me poser vos questions médicales.",
        "Bonjour ! Je suis à votre écoute pour toute question sur votre bien-être."
    ],
    "de": [
        "Hallo! Wie kann ich Ihnen heute bei Ihren Gesundheitsfragen helfen?",
        "Guten Tag! Welche gesundheitlichen Fragen haben Sie heute?",
        "Schön, Sie zu sehen! Wie fühlen Sie sich heute?",
        "Hallo! Ich bin hier, um Ihre medizinischen Fragen zu beantworten.",
        "Willkommen! Erzählen Sie mir gerne, welche Symptome oder Fragen Sie haben."
    ],
    "it": [
        "Ciao! Come posso aiutarti con la tua salute oggi?",
        "Buongiorno! Quali dubbi o domande mediche posso chiarire oggi?",
        "Piacere di sentirti! Come ti senti oggi?",
        "Ciao! Sono qui per aiutarti con qualsiasi domanda sulla tua salute.",
        "Benvenuto! Dimmi pure quali sintomi o preoccupazioni hai."
    ],
    "nl": [
        "Hallo! Hoe kan ik je vandaag helpen met je gezondheid?",
        "Hoi! Welke medische vragen of gezondheidskwesties heb je vandaag?",
        "Fijn je weer te spreken! Hoe voel je je vandaag?",
        "Hallo! Ik sta voor je klaar om je gezondheidsvragen te beantwoorden.",
        "Welkom! Deel gerust je symptomen of vragen met me."
    ],
    "ja": [
        "こんにちは！本日はどのような健康のお悩みがありますか？",
        "こんにちは！気になる症状や医療に関するご質問はありますか？",
        "お話しできて光栄です！本日のお体の調子はいかがですか？",
        "こんにちは！健康や医療に関するご相談をお気軽にどうぞ。",
        "いらっしゃいませ！気になる体調の変化などがあれば教えてくださいね。"
    ],
}

# Session tracker avoiding identical consecutive fallback variants per language
_LAST_STATIC_FALLBACK: Dict[str, str] = {}
_FALLBACK_LOCK = threading.Lock()


def get_static_greeting_fallback(language: str = "en", last_used: Optional[str] = None) -> str:
    """
    Returns a static greeting from the language pool.
    Guaranteed to never return the same variant twice in a row.
    """
    lang = (language or "en").strip().lower()[:2]
    pool = STATIC_GREETING_FALLBACKS.get(lang, STATIC_GREETING_FALLBACKS["en"])

    with _FALLBACK_LOCK:
        disallowed = set()
        if last_used and last_used.strip():
            disallowed.add(last_used.strip())
        if lang in _LAST_STATIC_FALLBACK:
            disallowed.add(_LAST_STATIC_FALLBACK[lang])

        available = [v for v in pool if v not in disallowed]
        if not available:
            available = [v for v in pool if v != last_used] or pool

        chosen = random.choice(available)
        _LAST_STATIC_FALLBACK[lang] = chosen
        return chosen


@dataclass
class ClassifierResult:
    """Structured response object returned by classify_query."""
    intent: str              # Strictly "MEDICAL" | "GREETING" | "NON_MEDICAL"
    language: str            # ISO 639-1 language code (e.g., "en", "es", "hi")
    confidence: float        # Bounded between 0.0 and 1.0
    greeting_reply: str      # Friendly greeting reply or empty string ""
    raw_response: Optional[Dict[str, Any]] = None
    fallback: bool = False   # True if produced by deterministic rule fallback
    latency_ms: float = 0.0


_COUNTER_LOCK = threading.Lock()


def _get_daily_counter_path(day_str: str) -> str:
    log_dir = os.getenv("AUDIT_LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"classifier_quota_{day_str}.json")


def _persist_counter_increment(fallback: bool) -> Tuple[int, int]:
    """Persists request counts to Redis if reachable, else to daily log JSON file."""
    import datetime
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

    # 1. Try Redis
    try:
        from services.task_dispatcher import get_redis_client
        r = get_redis_client()
        if r:
            k_total = f"medibot:classifier:{today}:total"
            k_fallback = f"medibot:classifier:{today}:fallback"
            tot = r.incr(k_total)
            fb = r.incr(k_fallback) if fallback else int(r.get(k_fallback) or 0)
            r.expire(k_total, 86400 * 7)
            r.expire(k_fallback, 86400 * 7)
            return int(tot), int(fb)
    except Exception as exc:
        logger.debug(f"[ClassifierQuotaMonitor] Redis counter unavailable: {exc}")

    # 2. File fallback
    with _COUNTER_LOCK:
        path = _get_daily_counter_path(today)
        tot = 0
        fb = 0
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    tot = int(data.get("total_requests", 0))
                    fb = int(data.get("fallback_requests", 0))
            except Exception:
                tot, fb = 0, 0
        tot += 1
        if fallback:
            fb += 1
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"date": today, "total_requests": tot, "fallback_requests": fb}, f)
        except Exception as exc:
            logger.debug(f"[ClassifierQuotaMonitor] File counter write error: {exc}")
        return tot, fb


def _read_persisted_counter() -> Tuple[int, int]:
    """Reads current daily count from Redis or file."""
    import datetime
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    try:
        from services.task_dispatcher import get_redis_client
        r = get_redis_client()
        if r:
            k_total = f"medibot:classifier:{today}:total"
            k_fallback = f"medibot:classifier:{today}:fallback"
            vals = r.mget([k_total, k_fallback])
            tot = int(vals[0] or 0) if vals and vals[0] else 0
            fb = int(vals[1] or 0) if vals and vals[1] else 0
            return tot, fb
    except Exception:
        pass

    with _COUNTER_LOCK:
        path = _get_daily_counter_path(today)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return int(data.get("total_requests", 0)), int(data.get("fallback_requests", 0))
            except Exception:
                pass
    return 0, 0


class ClassifierCircuitBreaker:
    """
    Circuit breaker for Groq LLM Classifier:
    3 consecutive failures -> open circuit for 30 seconds -> route to MEDICAL.
    Maintains persistent fallback metrics across Gunicorn worker recycles.
    """
    def __init__(self, failure_threshold: int = 3, reset_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.total_requests = 0
        self.fallback_requests = 0
        self.lock = threading.Lock()

    def record_request(self, fallback: bool = False) -> None:
        with self.lock:
            self.total_requests += 1
            if fallback:
                self.fallback_requests += 1

        # Persist across worker recycles (Redis or daily file)
        try:
            persisted_tot, persisted_fb = _persist_counter_increment(fallback)
            with self.lock:
                self.total_requests = max(self.total_requests, persisted_tot)
                self.fallback_requests = max(self.fallback_requests, persisted_fb)
        except Exception:
            pass

    def get_fallback_summary(self) -> Dict[str, Any]:
        with self.lock:
            # Sync with persisted counts if in-memory is zero (freshly recycled Gunicorn worker)
            if self.total_requests == 0:
                p_tot, p_fb = _read_persisted_counter()
                self.total_requests = p_tot
                self.fallback_requests = p_fb

            rate = (self.fallback_requests / self.total_requests * 100.0) if self.total_requests > 0 else 0.0
            return {
                "total_requests": self.total_requests,
                "fallback_requests": self.fallback_requests,
                "fallback_rate_pct": round(rate, 2),
                "is_breaker_open": (self.failure_count >= self.failure_threshold and (time.time() - self.last_failure_time) < self.reset_timeout)
            }

    def record_success(self) -> None:
        with self.lock:
            self.failure_count = 0

    def record_failure(self) -> None:
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

    def is_open(self) -> bool:
        with self.lock:
            if self.failure_count >= self.failure_threshold:
                if (time.time() - self.last_failure_time) < self.reset_timeout:
                    return True
                # Timeout elapsed, allow trial call
                return False
            return False

    def reset(self) -> None:
        with self.lock:
            self.failure_count = 0
            self.last_failure_time = 0.0
            self.total_requests = 0
            self.fallback_requests = 0


circuit_breaker = ClassifierCircuitBreaker(failure_threshold=3, reset_timeout=30.0)


# Thread-safe client cache and round-robin pointer
_CLIENT_CACHE: Dict[str, Groq] = {}
_CACHE_LOCK = threading.Lock()
_KEY_ROTATION_INDEX = 0


def reset_classifier_module_state() -> None:
    """Resets all module-level state for test isolation."""
    global _KEY_ROTATION_INDEX
    circuit_breaker.reset()
    with _CACHE_LOCK:
        _CLIENT_CACHE.clear()
        _KEY_ROTATION_INDEX = 0
    with _FALLBACK_LOCK:
        _LAST_STATIC_FALLBACK.clear()


def _get_client_for_key(api_key: str) -> Groq:
    """Retrieves or instantiates a cached Groq client with strict timeout."""
    with _CACHE_LOCK:
        if api_key not in _CLIENT_CACHE:
            cfg = get_classifier_config()
            _CLIENT_CACHE[api_key] = Groq(api_key=api_key, timeout=cfg.timeout_s, max_retries=0)
        return _CLIENT_CACHE[api_key]


def _build_classification_prompt(
    user_text: str,
    recent_turns: Optional[List[Dict[str, str]]] = None,
    triage_active: bool = False
) -> List[Dict[str, str]]:
    """Builds the routing classifier messages array via research.src.prompt."""
    return build_routing_classifier_messages(
        user_text=user_text,
        recent_turns=recent_turns,
        triage_active=triage_active
    )


def _validate_payload(data: Any) -> Optional[ClassifierResult]:
    """
    Strictly validates the parsed JSON payload.
    Rejects any payload that does not conform to required types and intent values.
    Validates language against allowed set {'en', 'es', 'fr', 'de', 'it', 'nl', 'ja'}; defaults to 'en'.
    Enforces that greeting_reply is ALWAYS empty string "".
    """
    if not isinstance(data, dict):
        return None

    raw_intent = data.get("intent")
    if not isinstance(raw_intent, str):
        return None
    intent_norm = raw_intent.strip().upper()
    if intent_norm not in VALID_INTENTS:
        return None

    raw_lang = data.get("language")
    if isinstance(raw_lang, str) and raw_lang.strip().lower() in ALLOWED_LANGUAGES:
        language = raw_lang.strip().lower()
    else:
        language = "en"

    raw_conf = data.get("confidence")
    try:
        confidence = float(raw_conf)
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.85

    # Classifier must ALWAYS return "" for greeting_reply.
    # Greeting text is generated exclusively by generate_greeting().
    greeting_reply = ""

    return ClassifierResult(
        intent=intent_norm,
        language=language,
        confidence=confidence,
        greeting_reply=greeting_reply,
        raw_response=data,
        fallback=False
    )


def _fallback_classify(
    user_text: str = "",
    triage_active: bool = False,
    reason: str = "fail_safe"
) -> ClassifierResult:
    """
    Deterministic fail-safe when LLM fails, times out, or returns malformed JSON.
    Under Phase 3 Amendment 4, ANY classifier failure returns:
    intent="MEDICAL", fallback=True, confidence=0.0, greeting_reply="".
    Never refuses because the classifier failed.
    """
    circuit_breaker.record_request(fallback=True)
    summary = circuit_breaker.get_fallback_summary()
    logger.warning(
        f"[ClassifierQuotaMonitor] Fallback triggered ({reason}) | "
        f"Fallback rate: {summary['fallback_rate_pct']}% ({summary['fallback_requests']}/{summary['total_requests']}) | "
        f"Breaker open: {summary['is_breaker_open']}"
    )
    return ClassifierResult(
        intent="MEDICAL",
        language="en",
        confidence=0.0,
        greeting_reply="",
        raw_response={"fallback_reason": reason},
        fallback=True
    )


def _is_greeting_sane(text: str) -> bool:
    """
    Validates greeting text against strict clinical sanity checks:
    - length <= 300 characters
    - no "mg" dosage units
    - no prohibited pharmaceutical drug names
    - no diagnostic terms ("diagnos")
    - no structured list formatting
    """
    if not text or not isinstance(text, str):
        return False

    cleaned = text.strip()
    if not cleaned or len(cleaned) > 300:
        return False

    # 1. Reject dosage references like "500mg", "50 mg", or standalone "mg"
    if re.search(r"\b\d*\s*mg\b", cleaned, re.IGNORECASE):
        return False

    # 2. Reject diagnostic language
    if "diagnos" in cleaned.lower():
        return False

    # 3. Reject prohibited drug names
    for pat in UNIVERSALLY_PROHIBITED_DRUGS:
        if re.search(pat, cleaned, re.IGNORECASE):
            return False

    # 4. Reject list formatting
    if re.search(r"^\s*[-*•\d+.]\s+", cleaned, re.MULTILINE):
        return False

    return True


def generate_greeting(
    user_text: str,
    recent_turns: Optional[List[Dict[str, str]]] = None,
    language: str = "en",
    timeout: Optional[float] = None
) -> str:
    """
    Generates a dynamic, context-aware, non-repetitive greeting reply using Groq LLM.

    Args:
        user_text: The user's input greeting.
        recent_turns: Recent conversation turns (last 3 user and assistant messages).
        language: User's ISO language code (e.g., 'en', 'es', 'fr', 'de', 'it', 'nl', 'ja').
        timeout: Timeout in seconds (default derived from CLASSIFIER_TIMEOUT_S; voice path uses 2.0s).

    Returns:
        A warm, non-repetitive greeting reply in the user's language.
        Guaranteed to never raise an exception.
    """
    global _KEY_ROTATION_INDEX

    cfg = get_classifier_config()
    actual_timeout = timeout if timeout is not None else cfg.timeout_s

    lang = (language or "en").strip().lower()[:2]
    last_assistant_msg = None
    if recent_turns:
        for t in reversed(recent_turns):
            role = t.get("role") or ("assistant" if "assistant" in t else None)
            content = t.get("content") or t.get("assistant")
            if (role == "assistant" or "assistant" in t) and content:
                last_assistant_msg = str(content).strip()
                break

    try:
        keys = get_groq_api_keys()
        if not keys:
            return get_static_greeting_fallback(lang, last_assistant_msg)

        # Build prompt using research/src/prompt.py greeter template
        messages = build_greeter_messages(
            user_text=user_text,
            recent_turns=recent_turns,
            language=lang
        )

        # Determine pooled key order
        with _CACHE_LOCK:
            primary_idx = _KEY_ROTATION_INDEX % len(keys)
            _KEY_ROTATION_INDEX = (_KEY_ROTATION_INDEX + 1) % len(keys)

        attempt_keys = [keys[primary_idx]]
        if len(keys) > 1:
            next_idx = (primary_idx + 1) % len(keys)
            attempt_keys.append(keys[next_idx])

        model_to_use = cfg.model

        for attempt, key in enumerate(attempt_keys):
            try:
                client = _get_client_for_key(key)
                try:
                    response = client.chat.completions.create(
                        model=model_to_use,
                        messages=messages,
                        temperature=0.8,
                        top_p=0.9,
                        max_tokens=GREETING_MAX_TOKENS,
                        timeout=actual_timeout
                    )
                except APIStatusError as api_err:
                    if getattr(api_err, "status_code", None) == 404 or "model_not_found" in str(api_err):
                        model_to_use = cfg.fallback_model
                        response = client.chat.completions.create(
                            model=model_to_use,
                            messages=messages,
                            temperature=0.8,
                            top_p=0.9,
                            max_tokens=GREETING_MAX_TOKENS,
                            timeout=actual_timeout
                        )
                    else:
                        raise

                candidate = response.choices[0].message.content
                if not candidate:
                    continue

                candidate = candidate.strip()

                # Sanity check: length <= 300, no mg, no drug names, no diagnos
                if not _is_greeting_sane(candidate):
                    logger.warning("[Greeter] Sanity check failed for generated greeting")
                    return get_static_greeting_fallback(lang, last_assistant_msg)

                # Run output guardrails (non-medical, no legal disclaimer)
                guarded = apply_output_guardrails(candidate, is_medical=False, show_disclaimer=False)
                if not _is_greeting_sane(guarded):
                    return get_static_greeting_fallback(lang, last_assistant_msg)

                return guarded

            except RateLimitError as rle:
                status = getattr(rle, "status_code", 429)
                logger.warning(f"[Greeter] RateLimitError status={status}")
                continue
            except APITimeoutError as te:
                status = getattr(te, "status_code", "timeout")
                logger.warning(f"[Greeter] APITimeoutError status={status}")
                break
            except Exception as ex:
                exc_cls = ex.__class__.__name__
                status = getattr(ex, "status_code", "none")
                logger.warning(f"[Greeter] Exception class={exc_cls} status={status}")
                break

    except Exception as e:
        exc_cls = e.__class__.__name__
        status = getattr(e, "status_code", "none")
        logger.warning(f"[Greeter] Error generating greeting class={exc_cls} status={status}")

    # Fallback to non-repeating static pool
    return get_static_greeting_fallback(lang, last_assistant_msg)


def classify_query(
    user_text: str,
    recent_turns: Optional[List[Dict[str, str]]] = None,
    triage_active: bool = False
) -> ClassifierResult:
    """
    Classifies a user's query into MEDICAL, GREETING, or NON_MEDICAL.
    Runs at temperature=0.0 in JSON mode.

    Args:
        user_text: The user's input text for the current turn.
        recent_turns: Optional list of recent turns (e.g., [{"role": "user", "content": "..."}]).
        triage_active: True if the patient is currently in an ongoing triage state.

    Returns:
        ClassifierResult containing intent, language, confidence, and greeting_reply.
        Guaranteed to never raise an exception.
    """
    global _KEY_ROTATION_INDEX

    # Check circuit breaker
    if circuit_breaker.is_open():
        logger.warning("[LLMClassifier] Circuit breaker OPEN (3 consecutive failures). Skipping classifier for 30s and routing to MEDICAL.")
        return _fallback_classify(user_text, triage_active=triage_active, reason="circuit_breaker_open")

    t0 = time.perf_counter()

    if not user_text or not user_text.strip():
        return ClassifierResult(
            intent="NON_MEDICAL",
            language="en",
            confidence=1.0,
            greeting_reply="",
            raw_response={"empty_input": True},
            fallback=True
        )

    messages = build_routing_classifier_messages(
        user_text=user_text.strip(),
        recent_turns=recent_turns,
        triage_active=triage_active
    )

    keys = get_groq_api_keys()
    if not keys:
        logger.warning("[LLMClassifier] No Groq API keys available; using fail-safe to MEDICAL.")
        circuit_breaker.record_failure()
        return _fallback_classify(user_text, triage_active=triage_active, reason="no_keys")

    # Determine key order with rotation
    with _CACHE_LOCK:
        primary_idx = _KEY_ROTATION_INDEX % len(keys)
        _KEY_ROTATION_INDEX = (_KEY_ROTATION_INDEX + 1) % len(keys)

    attempt_keys = [keys[(primary_idx + i) % len(keys)] for i in range(len(keys))]

    cfg = get_classifier_config()
    model_to_use = cfg.model

    for attempt_round in range(2):
        for attempt, key in enumerate(attempt_keys):
            try:
                client = _get_client_for_key(key)
                try:
                    response = client.chat.completions.create(
                        model=model_to_use,
                        messages=messages,
                        response_format={"type": "json_object"},
                        temperature=0.0,
                        max_tokens=CLASSIFY_MAX_TOKENS,
                        timeout=cfg.timeout_s
                    )
                except APIStatusError as api_err:
                    if getattr(api_err, "status_code", None) == 404 or "model_not_found" in str(api_err):
                        fallback_models = [cfg.fallback_model]
                        for m in ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]:
                            if m not in fallback_models:
                                fallback_models.append(m)
                        response = None
                        for fb_model in fallback_models:
                            logger.info(f"[LLMClassifier] Model '{model_to_use}' not found. Trying fallback '{fb_model}'.")
                            try:
                                response = client.chat.completions.create(
                                    model=fb_model,
                                    messages=messages,
                                    response_format={"type": "json_object"},
                                    temperature=0.0,
                                    max_tokens=CLASSIFY_MAX_TOKENS,
                                    timeout=cfg.timeout_s
                                )
                                model_to_use = fb_model
                                break
                            except Exception as fb_err:
                                logger.warning(f"[LLMClassifier] Fallback model '{fb_model}' failed: {fb_err}")
                                continue
                        if response is None:
                            continue
                    else:
                        raise

                content = response.choices[0].message.content
                if not content:
                    logger.warning("[LLMClassifier] Empty content from LLM choice.")
                    continue

                parsed_data = json.loads(content)
                result = _validate_payload(parsed_data)
                if result is not None:
                    circuit_breaker.record_success()
                    circuit_breaker.record_request(fallback=False)
                    result.latency_ms = (time.perf_counter() - t0) * 1000.0
                    return result
                else:
                    logger.warning("[LLMClassifier] Schema validation failed on LLM output.")
                    circuit_breaker.record_failure()
                    return _fallback_classify(user_text, triage_active=triage_active, reason="schema_validation_failed")

            except RateLimitError as rle:
                status = getattr(rle, "status_code", 429)
                logger.warning(f"[LLMClassifier] RateLimitError status={status}: {rle}")
                # If primary model exhausted quota, failover to fallback model with its separate TPD bucket
                if cfg.fallback_model and model_to_use != cfg.fallback_model:
                    logger.info(f"[LLMClassifier] Model '{model_to_use}' rate-limited. Failing over to '{cfg.fallback_model}'.")
                    model_to_use = cfg.fallback_model
                    continue

                wait_s = 1.0
                if hasattr(rle, "response") and rle.response is not None:
                    h = getattr(rle.response, "headers", {})
                    if "retry-after" in h:
                        try:
                            wait_s = float(h["retry-after"])
                        except Exception:
                            pass
                    elif "x-ratelimit-reset-tokens" in h:
                        try:
                            wait_s = float(str(h["x-ratelimit-reset-tokens"]).replace("s", "").strip())
                        except Exception:
                            pass
                time.sleep(min(wait_s, 2.0))
                continue

            except APITimeoutError as te:
                status = getattr(te, "status_code", "timeout")
                logger.warning(f"[LLMClassifier] APITimeoutError status={status}")
                circuit_breaker.record_failure()
                res = _fallback_classify(user_text, triage_active=triage_active, reason="timeout")
                res.latency_ms = (time.perf_counter() - t0) * 1000.0
                return res

            except APIStatusError as api_err:
                status = getattr(api_err, "status_code", "unknown")
                err_msg = str(api_err)
                if "rate_limit" in err_msg.lower() or status == 429:
                    logger.warning(f"[LLMClassifier] Rate limit in APIStatusError ({status}): {api_err}. Rotating key.")
                    wait_s = 1.0
                    if hasattr(api_err, "response") and api_err.response is not None:
                        h = getattr(api_err.response, "headers", {})
                        if "retry-after" in h:
                            try:
                                wait_s = float(h["retry-after"])
                            except Exception:
                                pass
                        elif "x-ratelimit-reset-tokens" in h:
                            try:
                                wait_s = float(str(h["x-ratelimit-reset-tokens"]).replace("s", "").strip())
                            except Exception:
                                pass
                    time.sleep(min(wait_s, 3.0))
                    continue
                logger.warning(f"[LLMClassifier] APIStatusError status={status}: {api_err}")
                circuit_breaker.record_failure()
                res = _fallback_classify(user_text, triage_active=triage_active, reason="api_status_error")
                res.latency_ms = (time.perf_counter() - t0) * 1000.0
                return res

            except Exception as ex:
                exc_cls = ex.__class__.__name__
                status = getattr(ex, "status_code", "none")
                logger.warning(f"[LLMClassifier] Exception class={exc_cls} status={status}")
                circuit_breaker.record_failure()
                res = _fallback_classify(user_text, triage_active=triage_active, reason="exception")
                res.latency_ms = (time.perf_counter() - t0) * 1000.0
                return res

    circuit_breaker.record_failure()
    res = _fallback_classify(user_text, triage_active=triage_active, reason="api_error")
    res.latency_ms = (time.perf_counter() - t0) * 1000.0
    return res
