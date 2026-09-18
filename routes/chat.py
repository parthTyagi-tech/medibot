import os
import re
import random
import threading
import traceback
from datetime import datetime, timezone

from flask import (
    Blueprint,
    render_template,
    request,
    jsonify,
    session,
    current_app
)
from flask_login import login_required, current_user

chat_bp = Blueprint("chat", __name__)

GREETING_PATTERNS = [
    r"^(hi|hello|hey|greetings|howdy|good\s+(morning|afternoon|evening|day)|salutations)\b",
    r"^how\s+(are\s+you|is\s+it\s+going|do\s+you\s+do|good\s+are\s+you)\b",
    r"^(what\'?s\s+up|sup|yo)\b",
]


def is_greeting_text(text: str) -> bool:
    cleaned = (text or "").strip().lower()
    return any(bool(re.search(pat, cleaned)) for pat in GREETING_PATTERNS)


from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from research.src.auth import db, ChatSession, Message
from research.src.memory import get_user_memory, clear_user_memory
from deepgram_tts import text_to_speech
from research.src.clinical_triage import (
    PatientState,
    ClinicalTriageEngine,
    format_context_aware_greeting,
    extract_patient_state,
    evaluate_triage_tier,
    check_medication_contraindications,
    check_mid_conversation_correction,
    friendly_emergency_label,
    maybe_append_emergency_reminder
)
from research.src.intent_classifier import is_third_party_query
from services.chat_service import (
    get_active_session,
    build_history_text,
    update_memory_in_background,
    update_title_in_background,
    summarize_session,
    summarize_session_in_background,
    load_patient_state,
    save_patient_state,
)
from services.task_dispatcher import (
    enqueue_memory_update,
    enqueue_title_update,
    enqueue_session_summarize,
    enqueue_eval_safety_check,
    enqueue_eval_turn
)
from services.state_store import (
    get_state_store,
    check_and_apply_ttl,
    check_emergency_resolution
)
from services.output_guardrails import ClinicalOutputGuardrail
from services.audit_logger import AuditLogger
from research.src.clinical_parser import ClinicalEntityParser


@chat_bp.route("/health", methods=["GET"])
@chat_bp.route("/ping", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "MediAssist AI",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), 200


@chat_bp.route("/", endpoint="index")
@login_required
def index():
    try:
        past_sessions = ChatSession.query.filter_by(
            user_id=current_user.id
        ).order_by(ChatSession.updated_at.desc()).limit(10).all()
    except Exception as e:
        logger.error(f"Error fetching past_sessions on index route: {e}")
        db.session.rollback()
        past_sessions = []

    return render_template(
        "chat.html",
        user=current_user,
        past_sessions=past_sessions,
        active_session_id=session.get("chat_session_id")
    )


@chat_bp.route("/new_chat", methods=["POST"], endpoint="new_chat")
@login_required
def new_chat():
    session_id = session.get("chat_session_id")
    if session_id:
        old_session = ChatSession.query.get(session_id)
        if old_session:
            summarize_session(old_session)

    session.pop("chat_session_id", None)
    return jsonify({"status": "ok"})


@chat_bp.route("/load_session/<int:session_id>", methods=["GET"], endpoint="load_session")
@login_required
def load_session(session_id):
    chat_session = ChatSession.query.filter_by(
        id=session_id, user_id=current_user.id
    ).first()

    if not chat_session:
        return jsonify({"error": "Session not found"}), 404

    session["chat_session_id"] = session_id
    print("LOADED SESSION:", session_id)

    messages = Message.query.filter_by(
        session_id=session_id
    ).order_by(Message.created_at.asc(), Message.id.asc()).all()

    return jsonify({
        "session_id": session_id,
        "title":      chat_session.title,
        "messages":   [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.created_at.strftime("%I:%M %p") if m.created_at else ""
            }
            for m in messages
        ]
    })


def handle_chat_turn(msg: str, user=None) -> str:
    """
    Request lifecycle handler for chat messages.
    - Ensures session['patient_state'] persists across turns with default keys
    - Runs Pre-intent ClinicalTriageEngine evaluation
    - Intercepts greetings with context-aware check-in
    - Handles medical queries and safety guardrails
    """
    if not msg or not msg.strip():
        return "Please enter a message."
    msg = msg.strip()

    if user is None:
        user = current_user if current_user and current_user.is_authenticated else None

    # Ensure session["patient_state"] persists across turns with default keys
    try:
        if "patient_state" not in session or not isinstance(session.get("patient_state"), dict):
            session["patient_state"] = {
                "has_cancer_history": False,
                "active_emergency": None,
                "emergency_turn_count": 0,
                "current_symptoms": []
            }
        else:
            session["patient_state"].setdefault("has_cancer_history", False)
            session["patient_state"].setdefault("active_emergency", None)
            session["patient_state"].setdefault("emergency_turn_count", 0)
            session["patient_state"].setdefault("current_symptoms", [])
    except Exception:
        pass

    # 1. Run Input Guardrails (Prompt injection, Content safety, Medical emergency)
    is_blocked, category, guard_msg = app_module.apply_input_guardrails(msg)
    if is_blocked:
        if category == "medical_emergency":
            try:
                chat_session = get_active_session()
                user_msg = Message(session_id=chat_session.id, role="user", content=msg)
                bot_msg = Message(session_id=chat_session.id, role="assistant", content=guard_msg)
                db.session.add_all([user_msg, bot_msg])
                db.session.commit()
            except Exception:
                pass
        return guard_msg

    try:
        chat_session = get_active_session()

        user_msg = Message(session_id=chat_session.id, role="user", content=msg)
        db.session.add(user_msg)
        db.session.commit()

        history_text = build_history_text(chat_session)
        user_memory = get_user_memory(user) if user else "No previous records."

        # Trigger decoupled background memory update asynchronously
        if user and hasattr(user, "id"):
            enqueue_memory_update(user.id, msg, history_text)

        # Trigger decoupled background title update asynchronously
        if chat_session.title == "New Consultation":
            enqueue_title_update(chat_session.id, msg)

        # Trigger decoupled background context window summarization when history reaches 6+ messages
        msg_count = Message.query.filter_by(session_id=chat_session.id).count()
        if msg_count >= 6 and msg_count % 4 == 0:
            enqueue_session_summarize(chat_session.id, msg_count)

        # Load or initialize structured patient state (State Store + DB backed)
        state_store = get_state_store()
        patient_state = state_store.get(str(chat_session.id)) or load_patient_state(chat_session)

        # Apply TTL auto-expiration and check affirmative resolution
        if check_and_apply_ttl(patient_state):
            AuditLogger.log_ttl_expired(session_id=str(chat_session.id))
        if check_emergency_resolution(msg, patient_state):
            AuditLogger.log_emergency_resolved(session_id=str(chat_session.id))

        # Sync with session["patient_state"] (persistent medical history only, NOT accumulator symptoms)
        try:
            sess_ps = session.get("patient_state", {})
            if sess_ps:
                if sess_ps.get("has_cancer_history"):
                    patient_state.has_cancer_history = True
                if sess_ps.get("active_emergency"):
                    patient_state.active_emergency = sess_ps.get("active_emergency")
                if sess_ps.get("emergency_turn_count", 0) > patient_state.emergency_turn_count:
                    patient_state.emergency_turn_count = sess_ps["emergency_turn_count"]
                # Stale current_symptoms are not synced across turns to prevent accumulator poisoning
        except Exception:
            pass

        # Extract per-turn clinical signals
        signals = ClinicalEntityParser.extract_clinical_signals(msg)
        patient_state.transient_symptoms = signals.get("active_symptoms", [])

        # Pure greeting bypass: Return nurse-grade context-aware greeting without clinical lecture
        is_pure_greeting = is_greeting_text(msg) and not signals.get("has_clinical_signals", False)
        if is_pure_greeting:
            first_name = user.name.split()[0] if user and hasattr(user, "name") and user.name else "there"
            raw_answer = format_context_aware_greeting(patient_state, first_name=first_name, msg=msg)
            answer = app_module.apply_output_guardrails(raw_answer, is_medical=False, show_disclaimer=False)
            bot_msg = Message(session_id=chat_session.id, role="assistant", content=str(answer))
            db.session.add(bot_msg)
            save_patient_state(chat_session, patient_state)
            state_store.set(str(chat_session.id), patient_state)
            chat_session.updated_at = datetime.now(timezone.utc)
            db.session.commit()
            try:
                session["patient_state"] = patient_state.to_dict()
            except Exception:
                pass
            return str(answer)

        prev_snapshot = patient_state.get_diff_snapshot()
        patient_state = extract_patient_state(msg, patient_state, llm=app_module.classifierModel)
        has_new_structured_fact = patient_state.has_state_diff(prev_snapshot)

        # PRE-INTENT EVALUATION: ClinicalTriageEngine with extracted signals
        triage_eval = ClinicalTriageEngine.evaluate(msg, patient_state, signals)
        try:
            session["patient_state"] = patient_state.to_dict()
        except Exception:
            pass

        if triage_eval and triage_eval.get("tier") == "RESOLVED":
            AuditLogger.log_emergency_resolved(session_id=str(chat_session.id))
            raw_answer = triage_eval["message"]
            answer = app_module.apply_output_guardrails(
                raw_answer,
                is_medical=True,
                show_disclaimer=False,
                patient_state=patient_state
            )
            bot_msg = Message(session_id=chat_session.id, role="assistant", content=str(answer))
            db.session.add(bot_msg)
            save_patient_state(chat_session, patient_state)
            state_store.set(str(chat_session.id), patient_state)
            chat_session.updated_at = datetime.now(timezone.utc)
            db.session.commit()
            try:
                session["patient_state"] = patient_state.to_dict()
            except Exception:
                pass
            return str(answer)

        if triage_eval and triage_eval.get("requires_escalation"):
            patient_state.emergency_override_served = True
            patient_state.risk_tier = "Emergency"
            em_type = triage_eval.get("emergency_type", "FEBRILE_NEUTROPENIA")
            patient_state.active_emergency = em_type
            raw_answer = triage_eval["message"]
            answer = app_module.apply_output_guardrails(
                raw_answer,
                is_medical=True,
                show_disclaimer=False,
                patient_state=patient_state,
                triage_tier="Emergency"
            )
            answer, guardrail_triggered = ClinicalOutputGuardrail.sanitize_response(answer, patient_state)
            if guardrail_triggered:
                AuditLogger.log_guardrail_override(
                    session_id=str(chat_session.id),
                    condition="EMERGENCY_OUTPUT_GUARDRAIL",
                    rule_triggered=answer.trigger_reason or "GUARDRAIL_INTERCEPTION",
                    raw_tokens_intercepted=answer.intercepted_tokens
                )

            # Audit logging for emergency / caregiver
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
                enqueue_eval_safety_check(
                    message_id=bot_msg.id,
                    query=msg,
                    response=str(answer),
                    patient_state=patient_state.to_dict()
                )
            except Exception:
                pass
            return str(answer)

        # Generic Intent Classification (only if no pre-intent emergency triggered)
        intent = app_module.classify_intent(app_module.classifierModel, msg)
        print("=" * 50)
        print("USER:", msg)
        print("INTENT:", intent)
        print("=" * 50)

        # Check for mid-conversation high-risk disclosure correction
        correction_alert = check_mid_conversation_correction(patient_state, history_text)

        # Check medication & dosing contraindications
        dosing_blocked, dosing_refusal = check_medication_contraindications(patient_state, msg)

        # Evaluate clinical triage risk tier & red-flag overrides with state-diff awareness
        risk_tier, red_flags, override_guidance = evaluate_triage_tier(
            patient_state, msg, has_new_structured_fact=has_new_structured_fact
        )
        patient_state.risk_tier = risk_tier
        patient_state.red_flags = red_flags
        try:
            session["patient_state"] = patient_state.to_dict()
        except Exception:
            pass

        retrieved_chunks = []

        # Ambiguous pronoun/follow-up check ("in this case", "suggest medication in this case")
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
            bot_msg = Message(session_id=chat_session.id, role="assistant", content=answer)
            db.session.add(bot_msg)
            save_patient_state(chat_session, patient_state)
            state_store.set(str(chat_session.id), patient_state)
            chat_session.updated_at = datetime.now(timezone.utc)
            db.session.commit()
            return answer

        # Handle Immediate Red-Flag Overrides (e.g. Febrile Neutropenia / Neonatal Fever / Emergency)
        if override_guidance and risk_tier == "Emergency":
            patient_state.emergency_override_served = True
            raw_answer = override_guidance
            if correction_alert:
                raw_answer = f"{correction_alert}\n\n{raw_answer}"
            answer = app_module.apply_output_guardrails(
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
            answer = app_module.apply_output_guardrails(
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
                raw_resp = app_module.chatModel.invoke(third_party_prompt)
                raw_answer = raw_resp.content if hasattr(raw_resp, "content") else str(raw_resp)
            except Exception:
                raw_answer = (
                    "When supporting a friend or loved one dealing with depression or emotional distress, "
                    "one of the most helpful things you can do is listen without judgment, let them know they are not alone, "
                    "and encourage them to reach out to a doctor, counselor, or mental health professional. "
                    "If they are in crisis, they can call or text the Suicide & Crisis Lifeline at 988."
                )

            raw_answer = maybe_append_emergency_reminder(raw_answer, patient_state, msg_count, msg)
            answer = app_module.apply_output_guardrails(raw_answer, is_medical=False, show_disclaimer=False)

        elif intent == "medical_query":
            dynamic_prompt = app_module.build_prompt(history_text, user_memory, patient_state=patient_state)
            try:
                question_answer_chain = create_stuff_documents_chain(app_module.chatModel, dynamic_prompt)
                rag_chain = create_retrieval_chain(app_module.retriever, question_answer_chain)
                response  = rag_chain.invoke({"input": msg})
                raw_answer = response.get("answer", "")
                context_docs = response.get("context", [])
                if isinstance(context_docs, list):
                    retrieved_chunks = [d.page_content if hasattr(d, "page_content") else str(d) for d in context_docs]
                elif isinstance(context_docs, str):
                    retrieved_chunks = [context_docs]
                if not raw_answer or not raw_answer.strip():
                    raise ValueError("Empty retrieval response")
            except Exception as rag_err:
                print(f"[RAG] Retrieval fallback to direct clinical consultation: {rag_err}")
                direct_prompt = dynamic_prompt.format(context="Clinical medicine reference and Gale Encyclopedia principles.", input=msg)
                raw_resp = app_module.chatModel.invoke(direct_prompt)
                raw_answer = raw_resp.content if hasattr(raw_resp, "content") else str(raw_resp)

            if correction_alert:
                raw_answer = f"{correction_alert}\n\n{raw_answer}"

            # Only show legal disclaimer on the initial medical turn; suppress on follow-up and medication responses
            is_med_inquiry = any(kw in msg.lower() for kw in ["medication", "medicine", "drug", "pill", "tablet", "dose", "dosing", "syrup"])
            show_disc = (not patient_state.disclaimer_shown) and not is_med_inquiry

            raw_answer = maybe_append_emergency_reminder(raw_answer, patient_state, msg_count, msg)

            answer = app_module.apply_output_guardrails(
                raw_answer,
                is_medical=True,
                show_disclaimer=show_disc,
                patient_state=patient_state,
                triage_tier=risk_tier
            )
            patient_state.disclaimer_shown = True

        elif intent == "greeting":
            first_name = user.name.split()[0] if user and hasattr(user, "name") and user.name else "there"
            # Context-Aware Greeting Interception
            raw_answer = format_context_aware_greeting(patient_state, first_name=first_name, msg=msg)
            answer = app_module.apply_output_guardrails(raw_answer, is_medical=False, show_disclaimer=False)

        elif intent == "memory_recall":
            recall_prompt = (
                f"You are MediAssist. Answer the user's question about their medical history or previous discussion.\n"
                f"User Profile & Known Medical Memory:\n{user_memory}\n\n"
                f"Recent Conversation:\n{history_text}\n\n"
                f"User Query: {msg}"
            )
            raw_resp = app_module.chatModel.invoke(recall_prompt)
            raw_answer = raw_resp.content if hasattr(raw_resp, "content") else str(raw_resp)
            answer = app_module.apply_output_guardrails(raw_answer, is_medical=False, show_disclaimer=False)

        elif intent == "account_action":
            answer = "Please use the account controls available in the navigation bar to manage your account or consultation history."

        else:
            # Non-medical query: enforce strict medical specialization
            answer = app_module.NON_MEDICAL_REFUSAL

        # Post-generation deterministic guardrail on all outgoing answers
        answer, guardrail_triggered = ClinicalOutputGuardrail.sanitize_response(answer, patient_state)
        if guardrail_triggered:
            AuditLogger.log_guardrail_override(
                session_id=str(chat_session.id),
                condition="POST_LLM_OUTPUT_GUARDRAIL",
                rule_triggered=answer.trigger_reason or "GUARDRAIL_INTERCEPTION",
                raw_tokens_intercepted=answer.intercepted_tokens
            )

        bot_msg = Message(session_id=chat_session.id, role="assistant", content=str(answer))
        db.session.add(bot_msg)

        # Persist patient state to DB & distributed state store
        save_patient_state(chat_session, patient_state)
        state_store.set(str(chat_session.id), patient_state)
        try:
            session["patient_state"] = patient_state.to_dict()
        except Exception:
            pass
        chat_session.updated_at = datetime.now(timezone.utc)
        db.session.commit()

        # Change 2: Deterministic rule-based safety evaluation on EVERY turn (0 LLM cost)
        try:
            enqueue_eval_safety_check(
                message_id=bot_msg.id,
                query=msg,
                response=answer,
                patient_state=patient_state.to_dict() if patient_state else None
            )
        except Exception as eval_err:
            print(f"[EvalQueue] Failed to enqueue safety check: {eval_err}")

        # Change 1B + Change 3: Sampled online shadow evaluation on medical_query turns
        try:
            sample_rate = float(os.getenv("EVAL_SAMPLE_RATE", "0.2"))
            if intent == "medical_query" and random.random() < sample_rate:
                enqueue_eval_turn(
                    message_id=bot_msg.id,
                    query=msg,
                    generated_answer=answer,
                    retrieved_chunks=retrieved_chunks,
                    patient_state=patient_state.to_dict() if patient_state else None
                )
        except Exception as eval_err:
            print(f"[EvalQueue] Failed to enqueue turn eval: {eval_err}")

        return answer

    except Exception:
        traceback.print_exc()
        return "Something went wrong."


handle_chat_message = handle_chat_turn


@chat_bp.route("/get", methods=["POST"], endpoint="chat")
@login_required
def chat():
    msg = request.form.get("msg", "").strip()
    return handle_chat_turn(msg, current_user)


@chat_bp.route("/delete_session/<int:session_id>", methods=["POST"], endpoint="delete_session")
@login_required
def delete_session(session_id):
    chat_session = ChatSession.query.filter_by(
        id=session_id, user_id=current_user.id
    ).first()

    if not chat_session:
        return jsonify({"success": False}), 404

    db.session.delete(chat_session)
    db.session.commit()

    if session.get("chat_session_id") == session_id:
        session.pop("chat_session_id", None)

    return jsonify({"success": True})


@chat_bp.route("/get_memory", methods=["GET"], endpoint="get_user_memory_route")
@login_required
def get_user_memory_route():
    return jsonify({"memory": get_user_memory(current_user)})


@chat_bp.route("/clear_memory", methods=["POST"], endpoint="clear_user_memory_route")
@login_required
def clear_user_memory_route():
    try:
        clear_user_memory(current_user)
        db.session.commit()
        return jsonify({"status": "ok"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


@chat_bp.route("/tts", methods=["POST"], endpoint="tts")
@login_required
def tts():
    text = request.form.get("text", "")
    filename = text_to_speech(text)
    return jsonify({"audio_url": f"/{filename}"})


def __getattr__(name):
    if name == "app_module":
        import app
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


try:
    import app as app_module
except Exception:
    pass

