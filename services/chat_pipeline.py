import os
import re
import sys
import random
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from flask import current_app, session
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from research.src.auth import db, ChatSession, Message, User
from research.src.memory import get_user_memory
from research.src.clinical_triage import (
    PatientState,
    ClinicalTriageEngine,
    format_context_aware_greeting,
    extract_patient_state,
    evaluate_triage_tier,
    check_medication_contraindications,
    check_mid_conversation_correction,
    maybe_append_emergency_reminder,
)
from research.src.clinical_parser import ClinicalEntityParser
from research.src.intent_classifier import classify_intent, is_third_party_query
from research.src.guardrails import apply_output_guardrails, NON_MEDICAL_REFUSAL
from services.ai_service import (
    chatModel,
    classifierModel,
    retriever,
    build_prompt,
)
from services.routing_service import route_query, RouteDecision
from services.llm_classifier import (
    generate_greeting,
    _is_greeting_sane,
    get_static_greeting_fallback,
)
from services.chat_service import (
    get_active_session,
    get_active_session_for_user,
    build_history_text,
    load_patient_state,
    save_patient_state,
)
from services.task_dispatcher import (
    enqueue_memory_update,
    enqueue_title_update,
    enqueue_session_summarize,
    enqueue_eval_safety_check,
    enqueue_eval_turn,
)
from services.state_store import (
    get_state_store,
    check_and_apply_ttl,
    check_emergency_resolution,
)
from services.output_guardrails import (
    ClinicalOutputGuardrail,
    StreamingGuardrailProcessor,
)
from services.audit_logger import AuditLogger

logger = logging.getLogger("chat-pipeline")


def _get_dep(name: str, default: Any, source_module: Optional[str] = None) -> Any:
    """
    Retrieves dependency by name, honoring test mocks/patches across modules.
    Priority order:
    1. If routes.chat has an active mock/patch (hasattr _mock_name / assert_called), return it.
    2. If source_module is specified and that module's attribute has an active mock or differs from default, return it.
    3. If routes.chat defines the attribute, return it.
    4. If source_module defines the attribute, return it.
    5. Return default.
    """
    chat_mod = sys.modules.get("routes.chat")
    if chat_mod is not None:
        chat_val = getattr(chat_mod, name, None)
        if chat_val is not None and (hasattr(chat_val, "_mock_name") or hasattr(chat_val, "assert_called")):
            return chat_val

    if source_module and source_module in sys.modules:
        src_mod = sys.modules[source_module]
        src_val = getattr(src_mod, name, None)
        if src_val is not None and (hasattr(src_val, "_mock_name") or hasattr(src_val, "assert_called") or src_val is not default):
            return src_val

    if chat_mod is not None:
        chat_val = getattr(chat_mod, name, None)
        if chat_val is not None:
            return chat_val

    if source_module and source_module in sys.modules:
        src_mod = sys.modules[source_module]
        src_val = getattr(src_mod, name, None)
        if src_val is not None:
            return src_val

    return default


def _resolve_session(
    user: Optional[User] = None,
    chat_session: Optional[ChatSession] = None,
    channel: str = "web"
) -> Optional[ChatSession]:
    """Resolves or initializes ChatSession based on channel and user context."""
    if chat_session is not None:
        return chat_session

    if channel == "web":
        get_active_session_fn = _get_dep("get_active_session", get_active_session)
        try:
            return get_active_session_fn()
        except Exception as e:
            logger.warning("Failed to get active web chat session: %s", type(e).__name__)
            return None
    else:
        if user and hasattr(user, "id"):
            try:
                return get_active_session_for_user(user.id)
            except Exception as e:
                logger.warning("Failed to get active voice chat session: %s", type(e).__name__)
                return None
    return None


def _load_and_sync_patient_state(chat_session: Optional[ChatSession]) -> Tuple[PatientState, Any]:
    """Loads patient state from state store or DB, syncing with Flask session if present."""
    state_store = get_state_store()
    patient_state = (state_store.get(str(chat_session.id)) if chat_session else None) or load_patient_state(chat_session)

    try:
        sess_ps = session.get("patient_state", {}) if session else {}
        if sess_ps:
            if sess_ps.get("has_cancer_history"):
                patient_state.has_cancer_history = True
            if sess_ps.get("active_emergency"):
                patient_state.active_emergency = sess_ps.get("active_emergency")
            if sess_ps.get("emergency_turn_count", 0) > patient_state.emergency_turn_count:
                patient_state.emergency_turn_count = sess_ps["emergency_turn_count"]
    except Exception:
        pass

    return patient_state, state_store


def _fetch_recent_turns(chat_session: Optional[ChatSession]) -> List[Dict[str, str]]:
    """Retrieves up to 3 prior messages from chat_session."""
    recent_turns: List[Dict[str, str]] = []
    if chat_session:
        try:
            prior_messages = Message.query.filter_by(
                session_id=chat_session.id
            ).order_by(Message.id.desc()).limit(3).all()
            for m in reversed(prior_messages):
                recent_turns.append({"role": m.role, "content": m.content})
        except Exception as e:
            logger.warning("Failed to fetch prior messages: %s", type(e).__name__)
            recent_turns = []
    return recent_turns


def run_turn(
    msg: str,
    user: Optional[User] = None,
    chat_session: Optional[ChatSession] = None,
    channel: str = "web",
    **kwargs
) -> Union[str, Iterator[str]]:
    """
    Unified entry point for conversational turns across web, voice, and voice_stream channels.
    Executes identical clinical triage, state extraction, guardrails, and persistence.
    Channel only governs delivery format (str vs Iterator[str]).
    """
    if channel == "voice_stream":
        return _run_turn_stream(msg, user=user, chat_session=chat_session, channel=channel, **kwargs)
    else:
        return _run_turn_sync(msg, user=user, chat_session=chat_session, channel=channel, **kwargs)


def _run_turn_sync(
    msg: str,
    user: Optional[User] = None,
    chat_session: Optional[ChatSession] = None,
    channel: str = "web",
    **kwargs
) -> str:
    """Synchronous pipeline execution returning the complete sanitized answer as a string."""
    chat_session = _resolve_session(user=user, chat_session=chat_session, channel=channel)
    patient_state, state_store = _load_and_sync_patient_state(chat_session)
    recent_turns = _fetch_recent_turns(chat_session)

    session_dict = None
    try:
        session_dict = session if session else None
    except Exception:
        session_dict = None

    route_query_fn = _get_dep("route_query", route_query, "services.routing_service")
    route_decision = route_query_fn(
        msg=msg,
        session=session_dict,
        patient_state=patient_state,
        recent_turns=recent_turns,
        session_id=str(chat_session.id if chat_session else "none")
    )

    # 1. BLOCKED_INJECTION
    if route_decision.action == "BLOCKED_INJECTION":
        return route_decision.response or "Request blocked."

    # 2. BLOCKED_EMERGENCY
    if route_decision.action == "BLOCKED_EMERGENCY":
        resp = route_decision.response or "Medical emergency detected."
        if chat_session:
            try:
                user_msg = Message(session_id=chat_session.id, role="user", content=msg)
                bot_msg = Message(session_id=chat_session.id, role="assistant", content=resp)
                db.session.add_all([user_msg, bot_msg])
                db.session.commit()
            except Exception as e:
                logger.warning("Failed to save emergency message: %s", type(e).__name__)
        return resp

    # 3. NON_MEDICAL (Amendment 5: No title, memory, summarization, or eval enqueued)
    if route_decision.action == "NON_MEDICAL":
        ans_refusal = route_decision.response or NON_MEDICAL_REFUSAL
        if chat_session:
            try:
                user_msg = Message(session_id=chat_session.id, role="user", content=msg)
                bot_msg = Message(session_id=chat_session.id, role="assistant", content=ans_refusal)
                db.session.add_all([user_msg, bot_msg])
                db.session.commit()
            except Exception as e:
                logger.warning("Failed to save non-medical message: %s", type(e).__name__)
        return ans_refusal

    # 4. GREETING (Amendment 5 & 9: timeout 2.0s on voice, no disclaimer consumed, no title, no memory)
    if route_decision.action == "GREETING":
        gen_greet_fn = _get_dep("generate_greeting", generate_greeting, "services.llm_classifier")
        timeout_val = 2.0 if channel == "voice" else None
        if timeout_val:
            raw_greeting = gen_greet_fn(msg, recent_turns=recent_turns, language=route_decision.language, timeout=timeout_val)
        else:
            raw_greeting = gen_greet_fn(msg, recent_turns=recent_turns, language=route_decision.language)

        if not _is_greeting_sane(raw_greeting):
            raw_greeting = get_static_greeting_fallback(route_decision.language)
        answer = apply_output_guardrails(raw_greeting, is_medical=False, show_disclaimer=False)
        if chat_session:
            try:
                user_msg = Message(session_id=chat_session.id, role="user", content=msg)
                bot_msg = Message(session_id=chat_session.id, role="assistant", content=str(answer))
                db.session.add_all([user_msg, bot_msg])
                chat_session.updated_at = datetime.now(timezone.utc)
                db.session.commit()
            except Exception as e:
                logger.warning("Failed to save greeting message: %s", type(e).__name__)
        return str(answer)

    # 5. ACCOUNT_ACTION
    if route_decision.action == "ACCOUNT_ACTION":
        ans_account = route_decision.response or "Please use the account controls available in the navigation bar to manage your account or consultation history."
        if chat_session:
            try:
                user_msg = Message(session_id=chat_session.id, role="user", content=msg)
                bot_msg = Message(session_id=chat_session.id, role="assistant", content=ans_account)
                db.session.add_all([user_msg, bot_msg])
                chat_session.updated_at = datetime.now(timezone.utc)
                db.session.commit()
            except Exception as e:
                logger.warning("Failed to save account action message: %s", type(e).__name__)
        return ans_account

    # ─────────────────────────────────────────────────────────────
    # 6. MEDICAL Turn: Clinical Triage, RAG, and Output Guardrails
    # ─────────────────────────────────────────────────────────────
    if chat_session:
        user_msg = Message(session_id=chat_session.id, role="user", content=msg)
        db.session.add(user_msg)
        db.session.commit()

    history_text = build_history_text(chat_session) if chat_session else ""
    user_memory = get_user_memory(user) if user else ("Voice Session" if channel == "voice" else "No previous records.")

    enqueue_mem_fn = _get_dep("enqueue_memory_update", enqueue_memory_update)
    enqueue_title_fn = _get_dep("enqueue_title_update", enqueue_title_update)
    enqueue_sum_fn = _get_dep("enqueue_session_summarize", enqueue_session_summarize)
    enqueue_eval_safety_fn = _get_dep("enqueue_eval_safety_check", enqueue_eval_safety_check)
    enqueue_eval_turn_fn = _get_dep("enqueue_eval_turn", enqueue_eval_turn)

    # Enqueue memory update
    if user and hasattr(user, "id"):
        enqueue_mem_fn(user.id, msg, history_text)

    # Enqueue title update on first MEDICAL message
    if chat_session and chat_session.title == "New Consultation":
        enqueue_title_fn(chat_session.id, msg)

    # Enqueue session summarization if history threshold reached
    msg_count = Message.query.filter_by(session_id=chat_session.id).count() if chat_session else 0
    if chat_session and msg_count >= 6 and msg_count % 4 == 0:
        enqueue_sum_fn(chat_session.id, msg_count)

    # Apply TTL auto-expiration and check emergency resolution
    if check_and_apply_ttl(patient_state) and chat_session:
        AuditLogger.log_ttl_expired(session_id=str(chat_session.id))
    if check_emergency_resolution(msg, patient_state) and chat_session:
        AuditLogger.log_emergency_resolved(session_id=str(chat_session.id))

    # Extract per-turn clinical signals
    signals = ClinicalEntityParser.extract_clinical_signals(msg)
    patient_state.transient_symptoms = signals.get("active_symptoms", [])

    prev_snapshot = patient_state.get_diff_snapshot()
    extract_ps_fn = _get_dep("extract_patient_state", extract_patient_state)
    # UNIFIED: Both web and voice use structured LLM extraction via classifierModel
    patient_state = extract_ps_fn(msg, patient_state, llm=classifierModel)
    has_new_structured_fact = patient_state.has_state_diff(prev_snapshot)

    # Clinical Triage Engine Evaluation
    triage_cls = _get_dep("ClinicalTriageEngine", ClinicalTriageEngine)
    triage_eval = triage_cls.evaluate(msg, patient_state, signals)

    try:
        if session:
            session["patient_state"] = patient_state.to_dict()
    except Exception:
        pass

    # Case A: Emergency Resolved
    if triage_eval and triage_eval.get("tier") == "RESOLVED":
        if chat_session:
            AuditLogger.log_emergency_resolved(session_id=str(chat_session.id))
        raw_answer = triage_eval["message"]
        answer = apply_output_guardrails(
            raw_answer,
            is_medical=True,
            show_disclaimer=False,
            patient_state=patient_state
        )
        if chat_session:
            bot_msg = Message(session_id=chat_session.id, role="assistant", content=str(answer))
            db.session.add(bot_msg)
            save_patient_state(chat_session, patient_state)
            state_store.set(str(chat_session.id), patient_state)
            chat_session.updated_at = datetime.now(timezone.utc)
            db.session.commit()
            try:
                if session:
                    session["patient_state"] = patient_state.to_dict()
            except Exception:
                pass
        return str(answer)

    # Case B: Emergency Escalation
    if triage_eval and triage_eval.get("requires_escalation"):
        patient_state.emergency_override_served = True
        patient_state.risk_tier = "Emergency"
        em_type = triage_eval.get("emergency_type", "FEBRILE_NEUTROPENIA")
        patient_state.active_emergency = em_type
        raw_answer = triage_eval["message"]
        answer = apply_output_guardrails(
            raw_answer,
            is_medical=True,
            show_disclaimer=False,
            patient_state=patient_state,
            triage_tier="Emergency"
        )
        sanitized_obj = ClinicalOutputGuardrail.sanitize_response(answer, patient_state)
        answer = str(sanitized_obj)
        if chat_session:
            if sanitized_obj.triggered:
                AuditLogger.log_guardrail_override(
                    session_id=str(chat_session.id),
                    condition="EMERGENCY_OUTPUT_GUARDRAIL",
                    rule_triggered=sanitized_obj.trigger_reason or "GUARDRAIL_INTERCEPTION",
                    raw_tokens_intercepted=sanitized_obj.intercepted_tokens
                )
            if em_type == "FEBRILE_NEUTROPENIA_CAREGIVER":
                AuditLogger.log_caregiver_intercept(
                    session_id=str(chat_session.id),
                    condition="FEBRILE_NEUTROPENIA_CAREGIVER",
                    rule_triggered="CAREGIVER_TRIAGE_INTERCEPT"
                )
            else:
                AuditLogger.log_emergency_triggered(
                    session_id=str(chat_session.id),
                    condition=em_type,
                    rule_triggered="CLINICAL_TRIAGE_ESCALATION"
                )

            bot_msg = Message(session_id=chat_session.id, role="assistant", content=str(answer))
            db.session.add(bot_msg)
            save_patient_state(chat_session, patient_state)
            state_store.set(str(chat_session.id), patient_state)
            chat_session.updated_at = datetime.now(timezone.utc)
            db.session.commit()
            try:
                if session:
                    session["patient_state"] = patient_state.to_dict()
            except Exception:
                pass

            try:
                enqueue_eval_safety_fn(
                    message_id=bot_msg.id,
                    query=msg,
                    response=str(answer),
                    patient_state=patient_state.to_dict()
                )
            except Exception as e:
                logger.warning("Failed to enqueue eval safety check: %s", type(e).__name__)
        return str(answer)

    # Post-triage Intent Determination
    classify_intent_fn = _get_dep("classify_intent", classify_intent)
    intent = classify_intent_fn(
        classifierModel,
        msg,
        route_decision=route_decision,
        recent_turns=recent_turns,
        patient_state=patient_state
    )

    correction_alert = check_mid_conversation_correction(patient_state, history_text)
    dosing_blocked, dosing_refusal = check_medication_contraindications(patient_state, msg)

    eval_triage_fn = _get_dep("evaluate_triage_tier", evaluate_triage_tier)
    risk_tier, red_flags, override_guidance = eval_triage_fn(
        patient_state, msg, has_new_structured_fact=has_new_structured_fact
    )
    patient_state.risk_tier = risk_tier
    patient_state.red_flags = red_flags
    try:
        if session:
            session["patient_state"] = patient_state.to_dict()
    except Exception:
        pass

    retrieved_chunks: List[str] = []

    # Ambiguous pronoun check
    is_ambiguous_case_query = bool(re.search(
        r"\b(in\s+this\s+case|in\s+that\s+case|for\s+this\s+case|in\s+this\s+situation)\b",
        msg.lower()
    )) or (
        bool(re.search(r"\b(suggest\s+(some\s+)?medication|things\s+to\s+avoid)\b", msg.lower()))
        and bool(re.search(r"\b(this\s+case|that\s+case)\b", msg.lower()))
    )

    if is_ambiguous_case_query and patient_state.third_party_context and patient_state.emergency_override_served:
        answer = (
            "Just to make sure I answer the right thing — are you asking about supporting your friend, "
            "or about your own fever/cancer situation from earlier?"
        )
        if chat_session:
            bot_msg = Message(session_id=chat_session.id, role="assistant", content=answer)
            db.session.add(bot_msg)
            save_patient_state(chat_session, patient_state)
            state_store.set(str(chat_session.id), patient_state)
            chat_session.updated_at = datetime.now(timezone.utc)
            db.session.commit()
        return answer

    # Handle Red-Flag Overrides
    if override_guidance and risk_tier == "Emergency":
        patient_state.emergency_override_served = True
        raw_answer = override_guidance
        if correction_alert:
            raw_answer = f"{correction_alert}\n\n{raw_answer}"
        answer = apply_output_guardrails(
            raw_answer,
            is_medical=True,
            show_disclaimer=False,
            patient_state=patient_state,
            triage_tier=risk_tier
        )

    elif dosing_blocked:
        raw_answer = dosing_refusal
        if correction_alert:
            raw_answer = f"{correction_alert}\n\n{raw_answer}"
        answer = apply_output_guardrails(
            raw_answer,
            is_medical=True,
            show_disclaimer=False,
            patient_state=patient_state,
            triage_tier=risk_tier
        )

    elif intent == "third_party_query" or is_third_party_query(msg)[0]:
        _, rel = is_third_party_query(msg)
        patient_state.third_party_context = rel or "third party"
        third_party_prompt = (
            f"You are MediAssist, an empathetic medical AI assistant. "
            f"The user is asking how to help or support someone else:\n"
            f"User message: '{msg}'\n\n"
            f"Provide compassionate, practical guidance on how to support this person. "
            f"If the concern involves mental health or depression, suggest empathetic listening, being there for them, "
            f"encouraging them to speak with a healthcare professional or counselor, and sharing available crisis resources (like 988 Suicide & Crisis Lifeline). "
            f"Do not address the user as if they are the patient. Do not discuss emergency oncology or hospital directives."
        )
        try:
            raw_resp = chatModel.invoke(third_party_prompt)
            raw_answer = raw_resp.content if hasattr(raw_resp, "content") else str(raw_resp)
        except Exception:
            raw_answer = (
                "When supporting a friend or loved one dealing with depression or emotional distress, "
                "one of the most helpful things you can do is listen without judgment, let them know they are not alone, "
                "and encourage them to reach out to a doctor, counselor, or mental health professional. "
                "If they are in crisis, they can call or text the Suicide & Crisis Lifeline at 988."
            )
        raw_answer = maybe_append_emergency_reminder(raw_answer, patient_state, msg_count, msg)
        answer = apply_output_guardrails(raw_answer, is_medical=False, show_disclaimer=False)

    elif intent == "medical_query":
        build_prompt_fn = _get_dep("build_prompt", build_prompt)
        dynamic_prompt = build_prompt_fn(history_text, user_memory, user=user, patient_state=patient_state)
        retriever_obj = _get_dep("retriever", retriever)
        create_stuff_chain_fn = _get_dep("create_stuff_documents_chain", create_stuff_documents_chain)
        create_retrieval_chain_fn = _get_dep("create_retrieval_chain", create_retrieval_chain)

        context_docs = []
        try:
            question_answer_chain = create_stuff_chain_fn(chatModel, dynamic_prompt)
            rag_chain = create_retrieval_chain_fn(retriever_obj, question_answer_chain)
            response = rag_chain.invoke({"input": msg})
            raw_answer = response.get("answer", "")
            context_docs = response.get("context", [])
            if isinstance(context_docs, list):
                retrieved_chunks = [d.page_content if hasattr(d, "page_content") else str(d) for d in context_docs]
            elif isinstance(context_docs, str):
                retrieved_chunks = [context_docs]
            if not raw_answer or not raw_answer.strip():
                raise ValueError("Empty retrieval response")
        except Exception as rag_err:
            logger.info("[RAG] Retrieval fallback to direct clinical consultation: %s", rag_err)
            direct_prompt = dynamic_prompt.format(
                context="No reference material available; answer from general knowledge, do not cite sources.",
                input=msg
            )
            raw_resp = chatModel.invoke(direct_prompt)
            raw_answer = raw_resp.content if hasattr(raw_resp, "content") else str(raw_resp)

        if correction_alert:
            raw_answer = f"{correction_alert}\n\n{raw_answer}"

        is_med_inquiry = any(kw in msg.lower() for kw in ["medication", "medicine", "drug", "pill", "tablet", "dose", "dosing", "syrup"])
        show_disc = (not patient_state.disclaimer_shown) and not is_med_inquiry
        raw_answer = maybe_append_emergency_reminder(raw_answer, patient_state, msg_count, msg)

        is_no_ref = (not context_docs) or getattr(retriever_obj, "no_reference_mode", False)
        answer = apply_output_guardrails(
            raw_answer,
            is_medical=True,
            show_disclaimer=show_disc,
            patient_state=patient_state,
            triage_tier=risk_tier,
            no_reference=is_no_ref
        )
        patient_state.disclaimer_shown = True

    elif intent == "greeting":
        first_name = user.name.split()[0] if user and hasattr(user, "name") and user.name else "there"
        raw_answer = format_context_aware_greeting(patient_state, first_name=first_name, msg=msg)
        answer = apply_output_guardrails(raw_answer, is_medical=False, show_disclaimer=False)

    elif intent == "memory_recall":
        recall_prompt = (
            f"You are MediAssist. Answer the user's question about their medical history or previous discussion.\n"
            f"User Profile & Known Medical Memory:\n{user_memory}\n\n"
            f"Recent Conversation:\n{history_text}\n\n"
            f"User Query: {msg}"
        )
        raw_resp = chatModel.invoke(recall_prompt)
        raw_answer = raw_resp.content if hasattr(raw_resp, "content") else str(raw_resp)
        answer = apply_output_guardrails(raw_answer, is_medical=False, show_disclaimer=False)

    elif intent == "account_action":
        answer = "Please use the account controls available in the navigation bar to manage your account or consultation history."

    else:
        answer = NON_MEDICAL_REFUSAL

    # Post-generation deterministic guardrail
    sanitized_obj = ClinicalOutputGuardrail.sanitize_response(answer, patient_state)
    answer = str(sanitized_obj)
    if sanitized_obj.triggered and chat_session:
        AuditLogger.log_guardrail_override(
            session_id=str(chat_session.id),
            condition="POST_LLM_OUTPUT_GUARDRAIL",
            rule_triggered=sanitized_obj.trigger_reason or "GUARDRAIL_INTERCEPTION",
            raw_tokens_intercepted=sanitized_obj.intercepted_tokens
        )

    if chat_session:
        bot_msg = Message(session_id=chat_session.id, role="assistant", content=str(answer))
        db.session.add(bot_msg)
        save_patient_state(chat_session, patient_state)
        state_store.set(str(chat_session.id), patient_state)
        chat_session.updated_at = datetime.now(timezone.utc)
        db.session.commit()
        try:
            if session:
                session["patient_state"] = patient_state.to_dict()
        except Exception:
            pass

        # Rule-based safety check
        try:
            enqueue_eval_safety_fn(
                message_id=bot_msg.id,
                query=msg,
                response=answer,
                patient_state=patient_state.to_dict() if patient_state else None
            )
        except Exception as eval_err:
            logger.warning("[EvalQueue] Failed to enqueue safety check: %s", eval_err)

        # Shadow evaluation sampled on medical_query
        try:
            sample_rate = float(os.getenv("EVAL_SAMPLE_RATE", "0.2"))
            if intent == "medical_query" and random.random() < sample_rate:
                enqueue_eval_turn_fn(
                    message_id=bot_msg.id,
                    query=msg,
                    generated_answer=answer,
                    retrieved_chunks=retrieved_chunks,
                    patient_state=patient_state.to_dict() if patient_state else None
                )
        except Exception as eval_err:
            logger.warning("[EvalQueue] Failed to enqueue turn eval: %s", eval_err)

    return answer


def _run_turn_stream(
    msg: str,
    user: Optional[User] = None,
    chat_session: Optional[ChatSession] = None,
    channel: str = "voice_stream",
    **kwargs
) -> Iterator[str]:
    """Streaming pipeline execution yielding sanitized chunks through StreamingGuardrailProcessor."""
    try:
        chat_session = _resolve_session(user=user, chat_session=chat_session, channel=channel)
        patient_state, state_store = _load_and_sync_patient_state(chat_session)
        recent_turns = _fetch_recent_turns(chat_session)

        session_dict = None
        try:
            session_dict = session if session else None
        except Exception:
            session_dict = None

        route_query_fn = _get_dep("route_query", route_query, "services.routing_service")
        route_decision = route_query_fn(
            msg=msg,
            session=session_dict,
            patient_state=patient_state,
            recent_turns=recent_turns,
            session_id=str(chat_session.id if chat_session else "none")
        )

        # 1. BLOCKED_INJECTION
        if route_decision.action == "BLOCKED_INJECTION":
            yield route_decision.response or "Request blocked."
            return

        # 2. BLOCKED_EMERGENCY
        if route_decision.action == "BLOCKED_EMERGENCY":
            resp = route_decision.response or "Medical emergency detected."
            if chat_session:
                try:
                    user_msg = Message(session_id=chat_session.id, role="user", content=msg)
                    bot_msg = Message(session_id=chat_session.id, role="assistant", content=resp)
                    db.session.add_all([user_msg, bot_msg])
                    db.session.commit()
                except Exception as e:
                    logger.warning("Failed to persist voice emergency message: %s", type(e).__name__)
            yield resp
            return

        # 3. NON_MEDICAL
        if route_decision.action == "NON_MEDICAL":
            ans_refusal = route_decision.response or NON_MEDICAL_REFUSAL
            if chat_session:
                try:
                    user_msg = Message(session_id=chat_session.id, role="user", content=msg)
                    bot_msg = Message(session_id=chat_session.id, role="assistant", content=ans_refusal)
                    db.session.add_all([user_msg, bot_msg])
                    db.session.commit()
                except Exception as e:
                    logger.warning("Failed to persist voice non-medical message: %s", type(e).__name__)
            yield ans_refusal
            return

        # 4. GREETING
        if route_decision.action == "GREETING":
            gen_greet_fn = _get_dep("generate_greeting", generate_greeting, "services.llm_classifier")
            raw_greeting = gen_greet_fn(msg, recent_turns=recent_turns, language=route_decision.language, timeout=2.0)
            if not _is_greeting_sane(raw_greeting):
                raw_greeting = get_static_greeting_fallback(route_decision.language)
            answer = apply_output_guardrails(raw_greeting, is_medical=False, show_disclaimer=False)
            if chat_session:
                try:
                    user_msg = Message(session_id=chat_session.id, role="user", content=msg)
                    bot_msg = Message(session_id=chat_session.id, role="assistant", content=str(answer))
                    db.session.add_all([user_msg, bot_msg])
                    chat_session.updated_at = datetime.now(timezone.utc)
                    db.session.commit()
                except Exception as e:
                    logger.warning("Failed to persist voice greeting message: %s", type(e).__name__)
            yield str(answer)
            return

        # 5. ACCOUNT_ACTION
        if route_decision.action == "ACCOUNT_ACTION":
            ans_account = route_decision.response or "Please use the account controls available in the application."
            if chat_session:
                try:
                    user_msg = Message(session_id=chat_session.id, role="user", content=msg)
                    bot_msg = Message(session_id=chat_session.id, role="assistant", content=ans_account)
                    db.session.add_all([user_msg, bot_msg])
                    chat_session.updated_at = datetime.now(timezone.utc)
                    db.session.commit()
                except Exception as e:
                    logger.warning("Failed to persist voice account action message: %s", type(e).__name__)
            yield ans_account
            return

        # 6. MEDICAL Path
        if chat_session:
            user_msg = Message(session_id=chat_session.id, role="user", content=msg)
            db.session.add(user_msg)
            db.session.commit()

        history_text = build_history_text(chat_session) if chat_session else ""
        user_memory = get_user_memory(user) if user else "Voice Session"

        enqueue_mem_fn = _get_dep("enqueue_memory_update", enqueue_memory_update)
        enqueue_title_fn = _get_dep("enqueue_title_update", enqueue_title_update)
        enqueue_eval_safety_fn = _get_dep("enqueue_eval_safety_check", enqueue_eval_safety_check)

        if user and hasattr(user, "id"):
            enqueue_mem_fn(user.id, msg, history_text)

        if chat_session and chat_session.title == "New Consultation":
            enqueue_title_fn(chat_session.id, msg)

        if check_and_apply_ttl(patient_state) and chat_session:
            AuditLogger.log_ttl_expired(session_id=str(chat_session.id))
        if check_emergency_resolution(msg, patient_state) and chat_session:
            AuditLogger.log_emergency_resolved(session_id=str(chat_session.id))

        signals = ClinicalEntityParser.extract_clinical_signals(msg)
        patient_state.transient_symptoms = signals.get("active_symptoms", [])

        prev_snapshot = patient_state.get_diff_snapshot()
        extract_ps_fn = _get_dep("extract_patient_state", extract_patient_state)
        # UNIFIED: Both web and voice use structured LLM extraction via classifierModel
        patient_state = extract_ps_fn(msg, patient_state, llm=classifierModel)
        has_new_structured_fact = patient_state.has_state_diff(prev_snapshot)

        triage_cls = _get_dep("ClinicalTriageEngine", ClinicalTriageEngine)
        triage_eval = triage_cls.evaluate(msg, patient_state, signals)

        if triage_eval and triage_eval.get("tier") == "RESOLVED":
            if chat_session:
                AuditLogger.log_emergency_resolved(session_id=str(chat_session.id))
            raw_answer = triage_eval["message"]
            answer = apply_output_guardrails(
                raw_answer,
                is_medical=True,
                show_disclaimer=False,
                patient_state=patient_state
            )
            if chat_session:
                bot_msg = Message(session_id=chat_session.id, role="assistant", content=str(answer))
                db.session.add(bot_msg)
                save_patient_state(chat_session, patient_state)
                state_store.set(str(chat_session.id), patient_state)
                chat_session.updated_at = datetime.now(timezone.utc)
                db.session.commit()
            yield str(answer)
            return

        if triage_eval and triage_eval.get("requires_escalation"):
            patient_state.emergency_override_served = True
            patient_state.risk_tier = "Emergency"
            em_type = triage_eval.get("emergency_type", "FEBRILE_NEUTROPENIA")
            patient_state.active_emergency = em_type
            raw_answer = triage_eval["message"]
            answer = apply_output_guardrails(
                raw_answer,
                is_medical=True,
                show_disclaimer=False,
                patient_state=patient_state,
                triage_tier="Emergency"
            )
            sanitized_obj = ClinicalOutputGuardrail.sanitize_response(answer, patient_state)
            answer = str(sanitized_obj)
            if chat_session:
                if sanitized_obj.triggered:
                    AuditLogger.log_guardrail_override(
                        session_id=str(chat_session.id),
                        condition="EMERGENCY_OUTPUT_GUARDRAIL",
                        rule_triggered=sanitized_obj.trigger_reason or "GUARDRAIL_INTERCEPTION",
                        raw_tokens_intercepted=sanitized_obj.intercepted_tokens
                    )
                if em_type == "FEBRILE_NEUTROPENIA_CAREGIVER":
                    AuditLogger.log_caregiver_intercept(
                        session_id=str(chat_session.id),
                        condition="FEBRILE_NEUTROPENIA_CAREGIVER",
                        rule_triggered="CAREGIVER_TRIAGE_INTERCEPT"
                    )
                else:
                    AuditLogger.log_emergency_triggered(
                        session_id=str(chat_session.id),
                        condition=em_type,
                        rule_triggered="CLINICAL_TRIAGE_ESCALATION"
                    )

            save_patient_state(chat_session, patient_state)
            if chat_session:
                state_store.set(str(chat_session.id), patient_state)
                bot_msg = Message(session_id=chat_session.id, role="assistant", content=str(answer))
                db.session.add(bot_msg)
                chat_session.updated_at = datetime.now(timezone.utc)
                db.session.commit()
            yield str(answer)
            return

        classify_intent_fn = _get_dep("classify_intent", classify_intent)
        intent = classify_intent_fn(
            classifierModel,
            msg,
            route_decision=route_decision,
            recent_turns=recent_turns,
            patient_state=patient_state
        )

        correction_alert = check_mid_conversation_correction(patient_state, history_text)
        dosing_blocked, dosing_refusal = check_medication_contraindications(patient_state, msg)
        eval_triage_fn = _get_dep("evaluate_triage_tier", evaluate_triage_tier)
        risk_tier, red_flags, override_guidance = eval_triage_fn(
            patient_state, msg, has_new_structured_fact=has_new_structured_fact
        )
        patient_state.risk_tier = risk_tier
        patient_state.red_flags = red_flags

        build_prompt_fn = _get_dep("build_prompt", build_prompt)
        dynamic_prompt = build_prompt_fn(history_text, user_memory, user=user, patient_state=patient_state)

        full_response: List[str] = []

        if override_guidance and risk_tier == "Emergency":
            text = override_guidance
            if correction_alert:
                text = f"{correction_alert}\n\n{text}"
            answer = apply_output_guardrails(
                text,
                is_medical=True,
                show_disclaimer=False,
                patient_state=patient_state,
                triage_tier=risk_tier
            )
            sanitized, triggered = ClinicalOutputGuardrail.sanitize_response(answer, patient_state)
            final_override = str(sanitized)
            yield final_override
            full_response.append(final_override)

        elif dosing_blocked:
            text = dosing_refusal
            if correction_alert:
                text = f"{correction_alert}\n\n{text}"
            answer = apply_output_guardrails(
                text,
                is_medical=True,
                show_disclaimer=False,
                patient_state=patient_state,
                triage_tier=risk_tier
            )
            sanitized, triggered = ClinicalOutputGuardrail.sanitize_response(answer, patient_state)
            final_dosing = str(sanitized)
            yield final_dosing
            full_response.append(final_dosing)

        elif intent == "medical_query":
            retriever_obj = _get_dep("retriever", retriever)
            try:
                docs = retriever_obj.invoke(msg)
                context = "\n\n".join([doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in docs])
            except Exception:
                context = "No reference material available; answer from general knowledge, do not cite sources."
            if not context.strip():
                context = "No reference material available; answer from general knowledge, do not cite sources."
            formatted_prompt = dynamic_prompt.format(context=context, input=msg)

            if correction_alert:
                yield correction_alert + "\n\n"
                full_response.append(correction_alert + "\n\n")

            processor = StreamingGuardrailProcessor(
                patient_state=patient_state,
                risk_tier=risk_tier,
                is_medical=True
            )
            try:
                for chunk_text in processor.process_stream(chatModel.stream(formatted_prompt)):
                    yield chunk_text
                    full_response.append(chunk_text)
            except Exception as stream_err:
                logger.error("LLM streaming interrupted: %s", type(stream_err).__name__)
                fallback_msg = StreamingGuardrailProcessor.STREAM_INTERRUPTED_FALLBACK
                yield fallback_msg
                full_response.append(fallback_msg)

            if processor.guardrail_triggered and chat_session:
                AuditLogger.log_guardrail_override(
                    session_id=str(chat_session.id),
                    condition="VOICE_STREAM_GUARDRAIL",
                    rule_triggered="; ".join(processor.trigger_reasons) or "STREAM_INTERCEPTION",
                    raw_tokens_intercepted=", ".join(processor.intercepted_tokens) if processor.intercepted_tokens else None
                )
            patient_state.disclaimer_shown = True

        else:
            prompt_val = None
            if intent == "memory_recall":
                prompt_val = f"User Memory:\n{user_memory}\n\nConversation History:\n{history_text}\n\nUser:\n{msg}"
            elif intent == "greeting":
                prompt_val = f"Reply naturally to: {msg}"
            elif intent == "account_action":
                prompt_val = "Please use the account controls available in the application."
                yield prompt_val
                full_response.append(prompt_val)
            else:
                prompt_val = NON_MEDICAL_REFUSAL
                yield prompt_val
                full_response.append(prompt_val)

            if prompt_val and intent in ("memory_recall", "greeting"):
                processor = StreamingGuardrailProcessor(
                    patient_state=patient_state,
                    risk_tier=risk_tier,
                    is_medical=False
                )
                try:
                    for chunk_text in processor.process_stream(chatModel.stream(prompt_val)):
                        yield chunk_text
                        full_response.append(chunk_text)
                except Exception as stream_err:
                    logger.error("LLM streaming interrupted: %s", type(stream_err).__name__)
                    fallback_msg = StreamingGuardrailProcessor.STREAM_INTERRUPTED_FALLBACK
                    yield fallback_msg
                    full_response.append(fallback_msg)

                if processor.guardrail_triggered and chat_session:
                    AuditLogger.log_guardrail_override(
                        session_id=str(chat_session.id),
                        condition="VOICE_STREAM_GUARDRAIL",
                        rule_triggered="; ".join(processor.trigger_reasons) or "STREAM_INTERCEPTION",
                        raw_tokens_intercepted=", ".join(processor.intercepted_tokens) if processor.intercepted_tokens else None
                    )

        save_patient_state(chat_session, patient_state)
        if chat_session:
            state_store.set(str(chat_session.id), patient_state)
        if chat_session and full_response:
            bot_answer = "".join(full_response)
            bot_msg = Message(session_id=chat_session.id, role="assistant", content=bot_answer)
            db.session.add(bot_msg)
            chat_session.updated_at = datetime.now(timezone.utc)
            db.session.commit()

            try:
                enqueue_eval_safety_fn(
                    message_id=bot_msg.id,
                    query=msg,
                    response=bot_answer,
                    patient_state=patient_state.to_dict() if patient_state else None
                )
            except Exception as eval_err:
                logger.warning("[EvalQueue] Failed to enqueue safety check: %s", eval_err)

    except Exception as e:
        logger.error(f"[run_turn_stream] Streaming error: {e}", exc_info=True)
        yield "Sorry, something went wrong on my end."
    finally:
        db.session.remove()
