"""
MediAssist Distributed Patient State Store & Lifecycle Management
==================================================================
Abstract state store interface with InMemory and Redis backends.
Enforces:
1. Session isolation across multi-worker/serverless environments
2. State Time-To-Live (TTL) auto-expiration (12-hour default for acute emergencies)
3. Affirmative resolution tracking (emergency stand-down)
"""

import os
import time
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union

from research.src.clinical_triage import PatientState

logger = logging.getLogger(__name__)

DEFAULT_EMERGENCY_TTL_SECONDS = 12 * 3600  # 12 hours

RESOLVE_PATTERNS = [
    r"back from (the )?(hospital|er|emergency room|clinic|doctor)",
    r"\b(doctor|oncologist)\s+(has\s+)?(cleared|checked|saw|discharged|treated)\s+me\b",
    r"\b(doctor|oncologist)\s+cleared\b",
    r"\b(saw|seen)\s+(the\s+|my\s+)?(doctor|oncologist)\b",
    r"received (iv )?antibiotics",
    r"fever is (gone|resolved|down|normal)",
    r"i('m| am) fine now",
    r"feeling much better now",
    r"temperature is back to normal"
]


def check_and_apply_ttl(
    state_or_session_id: Union[PatientState, str],
    ttl_seconds: float = DEFAULT_EMERGENCY_TTL_SECONDS
) -> bool:
    """
    Checks if active emergency has exceeded the TTL threshold.
    Accepts either a PatientState object or a session_id string.
    If expired, resets active_emergency and marks resolved_emergency='EXPIRED_TTL'.
    Returns True if an expiration occurred, False otherwise.
    """
    if isinstance(state_or_session_id, str):
        return get_state_store().check_and_apply_ttl(state_or_session_id, ttl_seconds)

    state = state_or_session_id
    if not state.active_emergency:
        return False

    now = time.time()
    if state.emergency_timestamp and (now - state.emergency_timestamp > ttl_seconds):
        logger.info(
            f"[StateStore] Emergency {state.active_emergency} exceeded TTL of {ttl_seconds}s. "
            f"Elapsed: {now - state.emergency_timestamp:.1f}s. Resetting emergency state."
        )
        state.reset_emergency(resolution_reason="EXPIRED_TTL")
        return True

    return False


def check_emergency_resolution(user_text: str, state: PatientState) -> bool:
    """
    Detects affirmative resolution disclosures from the user (e.g., 'back from hospital',
    'doctor cleared me'). If detected, transitions active_emergency to resolved_emergency.
    Returns True if resolution occurred, False otherwise.
    """
    if not state.active_emergency:
        return False

    text = (user_text or "").lower()

    # Clause-bounded negation check for clearance/resolution phrases
    if re.search(r"\b(not|never|hasn\'?t|haven\'?t|cannot|can\'?t|didn\'?t|won\'?t|no)\b[^.,;!\n]{0,25}\b(cleared|seen|saw|checked|back|discharged|resolved|better|normal)\b", text):
        return False

    for pat in RESOLVE_PATTERNS:
        if re.search(pat, text):
            logger.info(f"[StateStore] Affirmative emergency resolution detected via pattern: '{pat}'")
            state.reset_emergency(resolution_reason=state.active_emergency)
            return True

    return False


class BaseStateStore(ABC):
    """Abstract interface for session patient state storage."""

    @abstractmethod
    def get(self, session_id: str) -> Optional[PatientState]:
        pass

    @abstractmethod
    def set(self, session_id: str, state: PatientState, ttl_seconds: Optional[int] = None) -> None:
        pass

    @abstractmethod
    def delete(self, session_id: str) -> None:
        pass

    def get_patient_state(self, session_id: str) -> PatientState:
        state = self.get(session_id)
        if state is None:
            state = PatientState()
            self.set(session_id, state)
        return state

    def save_patient_state(self, session_id: str, state: PatientState) -> None:
        self.set(session_id, state)

    def check_and_apply_ttl(self, session_id: str, ttl_seconds: float = DEFAULT_EMERGENCY_TTL_SECONDS) -> bool:
        state = self.get(session_id)
        if state and state.active_emergency and state.emergency_timestamp:
            if (time.time() - state.emergency_timestamp) > ttl_seconds:
                state.reset_emergency(resolution_reason="TTL_EXPIRED")
                self.save_patient_state(session_id, state)
                return True
        return False


import threading


class InMemoryStateStore(BaseStateStore):
    """
    Thread-safe in-memory state store with TTL tracking per session.
    Suitable for multi-worker threads, local testing, and development.
    """

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> Optional[PatientState]:
        with self._lock:
            entry = self._store.get(session_id)
            if not entry:
                return None

            # Check session TTL if set
            expires_at = entry.get("expires_at")
            if expires_at and time.time() > expires_at:
                del self._store[session_id]
                return None

            state = entry.get("state")
            if state:
                # Check internal emergency TTL
                check_and_apply_ttl(state)
            return state

    def set(self, session_id: str, state: PatientState, ttl_seconds: Optional[int] = None) -> None:
        expires_at = (time.time() + ttl_seconds) if ttl_seconds else None
        with self._lock:
            self._store[session_id] = {
                "state": state,
                "expires_at": expires_at
            }

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


class RedisStateStore(BaseStateStore):
    """
    Distributed Redis state store with JSON serialization and key expiration.
    Gracefully falls back to InMemoryStateStore if Redis is unavailable.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0", key_prefix: str = "medibot:session:"):
        self.key_prefix = key_prefix
        self.fallback = InMemoryStateStore()
        self._client = None

        try:
            import redis
            self._client = redis.Redis.from_url(redis_url, decode_responses=True)
            self._client.ping()
            logger.info(f"[RedisStateStore] Successfully connected to Redis at {redis_url}")
        except Exception as e:
            logger.warning(f"[RedisStateStore] Redis unavailable ({e}). Falling back to InMemoryStateStore.")
            self._client = None

    def _get_key(self, session_id: str) -> str:
        return f"{self.key_prefix}{session_id}"

    def get(self, session_id: str) -> Optional[PatientState]:
        if not self._client:
            return self.fallback.get(session_id)

        try:
            raw_data = self._client.get(self._get_key(session_id))
            if not raw_data:
                return None
            state = PatientState.from_json(raw_data)
            check_and_apply_ttl(state)
            return state
        except Exception as e:
            logger.error(f"[RedisStateStore] Error retrieving session {session_id}: {e}")
            return self.fallback.get(session_id)

    def set(self, session_id: str, state: PatientState, ttl_seconds: Optional[int] = None) -> None:
        if not self._client:
            self.fallback.set(session_id, state, ttl_seconds)
            return

        try:
            key = self._get_key(session_id)
            json_str = state.to_json()
            if ttl_seconds:
                self._client.setex(key, ttl_seconds, json_str)
            else:
                self._client.set(key, json_str)
        except Exception as e:
            logger.error(f"[RedisStateStore] Error storing session {session_id}: {e}")
            self.fallback.set(session_id, state, ttl_seconds)

    def delete(self, session_id: str) -> None:
        if not self._client:
            self.fallback.delete(session_id)
            return

        try:
            self._client.delete(self._get_key(session_id))
        except Exception as e:
            logger.error(f"[RedisStateStore] Error deleting session {session_id}: {e}")
            self.fallback.delete(session_id)


# Global singleton instance
_GLOBAL_STATE_STORE: Optional[BaseStateStore] = None


def get_state_store() -> BaseStateStore:
    """
    Returns global configured state store singleton.
    Reads REDIS_URL environment variable if set.
    """
    global _GLOBAL_STATE_STORE
    if _GLOBAL_STATE_STORE is None:
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            _GLOBAL_STATE_STORE = RedisStateStore(redis_url=redis_url)
        else:
            _GLOBAL_STATE_STORE = InMemoryStateStore()
    return _GLOBAL_STATE_STORE


def reset_state_store() -> None:
    """Resets global state store (primarily for unit tests)."""
    global _GLOBAL_STATE_STORE
    _GLOBAL_STATE_STORE = None


state_store: BaseStateStore = get_state_store()

