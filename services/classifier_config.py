"""
Classifier Configuration Helper
=================================
Dynamically reads and validates environment variables at call time:
- INTENT_CLASSIFIER_MODE ("llm" | "regex", default: "llm")
- LLM_CLASSIFIER_MODEL (default: "llama-3.1-8b-instant")
- LLM_CLASSIFIER_FALLBACK_MODEL (default: "qwen/qwen3.8-27b")
- CLASSIFIER_TIMEOUT_S (clamped 0.5 to 5.0 seconds, default: 2.5)
- CLASSIFIER_MIN_CONFIDENCE (clamped 0.0 to 1.0, default: 0.7)

Guarantees:
- Invalid or unparseable values fall back to defaults and log a warning.
- Clamping prevents unstable timeouts and invalid probability ranges.
- Evaluated at call time to support dynamic runtime adjustments and testing.
"""

import os
import logging
from dataclasses import dataclass

logger = logging.getLogger("classifier-config")

VALID_MODES = {"llm", "regex"}
DEFAULT_MODE = "llm"
DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_FALLBACK_MODEL = "qwen/qwen3.8-27b"
DEFAULT_TIMEOUT_S = 2.5
DEFAULT_MIN_CONFIDENCE = 0.7

MIN_TIMEOUT_S = 0.5
MAX_TIMEOUT_S = 5.0
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0


@dataclass(frozen=True)
class ClassifierConfig:
    """Immutable snapshot of classifier configuration."""
    mode: str
    model: str
    fallback_model: str
    timeout_s: float
    min_confidence: float


def get_classifier_config() -> ClassifierConfig:
    """
    Reads and validates classifier settings dynamically at call time.
    Invalid or out-of-range values fall back to defaults with a warning.
    """
    # 1. Mode (llm | regex)
    raw_mode = os.getenv("INTENT_CLASSIFIER_MODE", DEFAULT_MODE).strip().lower()
    if raw_mode not in VALID_MODES:
        logger.warning(
            f"[ClassifierConfig] Invalid INTENT_CLASSIFIER_MODE '{raw_mode}'. "
            f"Expected one of {VALID_MODES}. Falling back to '{DEFAULT_MODE}'."
        )
        mode = DEFAULT_MODE
    else:
        mode = raw_mode

    # 2. Model & Fallback Model
    model = os.getenv("LLM_CLASSIFIER_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    fallback_model = (
        os.getenv("LLM_CLASSIFIER_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL).strip()
        or DEFAULT_FALLBACK_MODEL
    )

    # 3. Timeout in seconds (clamped 0.5 - 5.0)
    raw_timeout = os.getenv("CLASSIFIER_TIMEOUT_S", str(DEFAULT_TIMEOUT_S)).strip()
    try:
        val_timeout = float(raw_timeout)
        if val_timeout < MIN_TIMEOUT_S or val_timeout > MAX_TIMEOUT_S:
            clamped_timeout = max(MIN_TIMEOUT_S, min(MAX_TIMEOUT_S, val_timeout))
            logger.warning(
                f"[ClassifierConfig] CLASSIFIER_TIMEOUT_S {val_timeout} outside allowed range "
                f"[{MIN_TIMEOUT_S}, {MAX_TIMEOUT_S}]. Clamping to {clamped_timeout}."
            )
            timeout_s = clamped_timeout
        else:
            timeout_s = val_timeout
    except (ValueError, TypeError):
        logger.warning(
            f"[ClassifierConfig] Invalid CLASSIFIER_TIMEOUT_S '{raw_timeout}'. "
            f"Falling back to default {DEFAULT_TIMEOUT_S}s."
        )
        timeout_s = DEFAULT_TIMEOUT_S

    # 4. Minimum Confidence (clamped 0.0 - 1.0)
    raw_conf = os.getenv("CLASSIFIER_MIN_CONFIDENCE", str(DEFAULT_MIN_CONFIDENCE)).strip()
    try:
        val_conf = float(raw_conf)
        if val_conf < MIN_CONFIDENCE or val_conf > MAX_CONFIDENCE:
            clamped_conf = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, val_conf))
            logger.warning(
                f"[ClassifierConfig] CLASSIFIER_MIN_CONFIDENCE {val_conf} outside allowed range "
                f"[{MIN_CONFIDENCE}, {MAX_CONFIDENCE}]. Clamping to {clamped_conf}."
            )
            min_confidence = clamped_conf
        else:
            min_confidence = val_conf
    except (ValueError, TypeError):
        logger.warning(
            f"[ClassifierConfig] Invalid CLASSIFIER_MIN_CONFIDENCE '{raw_conf}'. "
            f"Falling back to default {DEFAULT_MIN_CONFIDENCE}."
        )
        min_confidence = DEFAULT_MIN_CONFIDENCE

    return ClassifierConfig(
        mode=mode,
        model=model,
        fallback_model=fallback_model,
        timeout_s=timeout_s,
        min_confidence=min_confidence
    )
