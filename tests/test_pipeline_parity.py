"""
Pipeline Parity Test Suite
==========================
Verifies that services.chat_pipeline.run_turn maintains 100% parity across
web, voice, and voice_stream channels:
1. Identical triage tier evaluation.
2. Identical clinical guardrail outcomes (dosing suppression, forbidden antipyretics).
3. Identical persisted PatientState in state_store and database.
4. Channel parameter only controls delivery format (str vs generator), NEVER safety checks.
5. Verification of drift fix: extract_patient_state is passed classifierModel on all channels.
"""

import os
import json
import time
from unittest.mock import patch, MagicMock
import pytest

from app import app, db
from research.src.auth import User, ChatSession, Message
from research.src.clinical_triage import PatientState
from research.src.guardrails import NON_MEDICAL_REFUSAL
from services.chat_pipeline import run_turn
from services.chat_service import load_patient_state
from services.state_store import reset_state_store, get_state_store
from services.output_guardrails import ClinicalOutputGuardrail
from services.ai_service import GroqChatModel, classifierModel


class MockChunk:
    def __init__(self, content: str):
        self.content = content


@pytest.fixture(autouse=True)
def setup_test_db():
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["WTF_CSRF_ENABLED"] = False
    app.config["SERVER_NAME"] = None

    reset_state_store()
    with app.app_context():
        db.create_all()
        yield
        db.session.remove()
        db.drop_all()
    reset_state_store()


def _create_user_and_sessions():
    user = User(email="parity.patient@example.com", name="Parity Patient", password_hash="hash")
    db.session.add(user)
    db.session.commit()

    session_web = ChatSession(user_id=user.id, title="Consultation Web")
    session_voice = ChatSession(user_id=user.id, title="Consultation Voice")
    session_stream = ChatSession(user_id=user.id, title="Consultation Stream")

    db.session.add_all([session_web, session_voice, session_stream])
    db.session.commit()

    return user, session_web, session_voice, session_stream


def test_parity_routine_medical_query():
    """Verify routine medical query produces identical triage tier and state across all 3 channels."""
    user, s_web, s_voice, s_stream = _create_user_and_sessions()
    s_web_id, s_voice_id, s_stream_id = s_web.id, s_voice.id, s_stream.id
    query = "I have a mild tension headache. What lifestyle tips can help?"

    answer_text = "Rest in a quiet, dark room, stay well-hydrated, and practice gentle stretching."
    mock_docs = [MagicMock(page_content="Tension headaches respond well to rest and hydration.")]

    mock_chunks = [
        MockChunk("Rest in a quiet, "),
        MockChunk("dark room, "),
        MockChunk("stay well-hydrated, "),
        MockChunk("and practice gentle stretching.")
    ]

    mock_rag_chain = MagicMock()
    mock_rag_chain.invoke.return_value = {"answer": answer_text, "context": mock_docs}

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = mock_docs
    mock_retriever.no_reference_mode = False

    with patch("services.chat_pipeline.retriever", mock_retriever):
        with patch("services.chat_pipeline.create_retrieval_chain", return_value=mock_rag_chain):
            with patch.object(GroqChatModel, "invoke", return_value=MagicMock(content=answer_text)):
                with patch.object(GroqChatModel, "stream", return_value=iter(mock_chunks)):
                    # Web channel
                    out_web = run_turn(query, user=user, chat_session=s_web, channel="web")
                    # Voice channel (non-stream)
                    out_voice = run_turn(query, user=user, chat_session=s_voice, channel="voice")
                    # Voice channel (streaming)
                    stream_gen = run_turn(query, user=user, chat_session=s_stream, channel="voice_stream")
                    out_stream = "".join(list(stream_gen))

    # All channels deliver text
    assert isinstance(out_web, str) and len(out_web) > 0
    assert isinstance(out_voice, str) and len(out_voice) > 0
    assert isinstance(out_stream, str) and len(out_stream) > 0

    # Core advice is present across all channels
    assert "rest in a quiet" in out_web.lower()
    assert "rest in a quiet" in out_voice.lower()
    assert "rest in a quiet" in out_stream.lower()

    # Load fresh persisted patient states
    ps_web = load_patient_state(db.session.get(ChatSession, s_web_id))
    ps_voice = load_patient_state(db.session.get(ChatSession, s_voice_id))
    ps_stream = load_patient_state(db.session.get(ChatSession, s_stream_id))

    # Identical triage tier
    assert ps_web.risk_tier == ps_voice.risk_tier == ps_stream.risk_tier == "Routine"
    # Identical disclaimer consumption
    assert ps_web.disclaimer_shown is True
    assert ps_voice.disclaimer_shown is True
    assert ps_stream.disclaimer_shown is True

    # Check database persistence equals delivered content
    msg_web = Message.query.filter_by(session_id=s_web_id, role="assistant").order_by(Message.id.desc()).first()
    msg_voice = Message.query.filter_by(session_id=s_voice_id, role="assistant").order_by(Message.id.desc()).first()
    msg_stream = Message.query.filter_by(session_id=s_stream_id, role="assistant").order_by(Message.id.desc()).first()

    assert msg_web.content == out_web
    assert msg_voice.content == out_voice
    assert msg_stream.content == out_stream


def test_parity_emergency_oncology_fever_triage():
    """Verify emergency fever for oncology patient triggers identical Emergency tier & escalation across channels."""
    user, s_web, s_voice, s_stream = _create_user_and_sessions()
    s_web_id, s_voice_id, s_stream_id = s_web.id, s_voice.id, s_stream.id

    # Setup cancer history on all sessions
    for sess in (s_web, s_voice, s_stream):
        st = PatientState(has_cancer_history=True)
        sess.patient_state_json = json.dumps(st.to_dict())
    db.session.commit()

    query = "I am on chemotherapy and my temperature is 101.5 F"

    out_web = run_turn(query, user=user, chat_session=s_web, channel="web")
    out_voice = run_turn(query, user=user, chat_session=s_voice, channel="voice")
    out_stream = "".join(list(run_turn(query, user=user, chat_session=s_stream, channel="voice_stream")))

    # Emergency escalation language must be present across all channels
    for out in (out_web, out_voice, out_stream):
        lower = out.lower()
        assert "emergency" in lower or "oncology" in lower or "urgent" in lower or "immediately" in lower

    # Parity check on persisted PatientState using re-fetched sessions
    ps_web = load_patient_state(db.session.get(ChatSession, s_web_id))
    ps_voice = load_patient_state(db.session.get(ChatSession, s_voice_id))
    ps_stream = load_patient_state(db.session.get(ChatSession, s_stream_id))

    assert ps_web.risk_tier == "Emergency"
    assert ps_voice.risk_tier == "Emergency"
    assert ps_stream.risk_tier == "Emergency"

    assert ps_web.emergency_override_served is True
    assert ps_voice.emergency_override_served is True
    assert ps_stream.emergency_override_served is True

    # Persisted messages match delivered output
    msg_web = Message.query.filter_by(session_id=s_web_id, role="assistant").order_by(Message.id.desc()).first()
    msg_voice = Message.query.filter_by(session_id=s_voice_id, role="assistant").order_by(Message.id.desc()).first()
    msg_stream = Message.query.filter_by(session_id=s_stream_id, role="assistant").order_by(Message.id.desc()).first()

    assert msg_web.content == out_web
    assert msg_voice.content == out_voice
    assert msg_stream.content == out_stream


def test_parity_high_risk_antipyretic_suppression():
    """Verify permissive antipyretic for oncology patient is sanitized identically across all 3 channels."""
    user, s_web, s_voice, s_stream = _create_user_and_sessions()
    s_web_id, s_voice_id, s_stream_id = s_web.id, s_voice.id, s_stream.id

    for sess in (s_web, s_voice, s_stream):
        st = PatientState(has_cancer_history=True)
        sess.patient_state_json = json.dumps(st.to_dict())
    db.session.commit()

    query = "Can I take ibuprofen for my fever?"

    mock_llm_text = "You can take ibuprofen for the fever. Rest well."
    mock_chunks = [
        MockChunk("You can "),
        MockChunk("take ibuprofen "),
        MockChunk("for the fever. "),
        MockChunk("Rest well.")
    ]

    mock_rag_chain = MagicMock()
    mock_rag_chain.invoke.return_value = {"answer": mock_llm_text, "context": []}

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    mock_retriever.no_reference_mode = False

    with patch("services.chat_pipeline.retriever", mock_retriever):
        with patch("services.chat_pipeline.create_retrieval_chain", return_value=mock_rag_chain):
            with patch.object(GroqChatModel, "invoke", return_value=MagicMock(content=mock_llm_text)):
                with patch.object(GroqChatModel, "stream", return_value=iter(mock_chunks)):
                    out_web = run_turn(query, user=user, chat_session=s_web, channel="web")
                    out_voice = run_turn(query, user=user, chat_session=s_voice, channel="voice")
                    out_stream = "".join(list(run_turn(query, user=user, chat_session=s_stream, channel="voice_stream")))

    # Guardrail must suppress permissive antipyretic in all 3 channels
    for out in (out_web, out_voice, out_stream):
        assert "take ibuprofen" not in out.lower()
        assert ClinicalOutputGuardrail.CLINICAL_OVERRIDE_TEXT in out

    # Persisted message parity
    msg_web = Message.query.filter_by(session_id=s_web_id, role="assistant").order_by(Message.id.desc()).first()
    msg_voice = Message.query.filter_by(session_id=s_voice_id, role="assistant").order_by(Message.id.desc()).first()
    msg_stream = Message.query.filter_by(session_id=s_stream_id, role="assistant").order_by(Message.id.desc()).first()

    assert msg_web.content == out_web
    assert msg_voice.content == out_voice
    assert msg_stream.content == out_stream


def test_parity_numeric_dosing_suppression():
    """Verify numeric dosage is suppressed and replaced by standard disclaimer across all 3 channels."""
    user, s_web, s_voice, s_stream = _create_user_and_sessions()
    query = "What is the dosage for paracetamol?"

    mock_llm_text = "You should take 500mg or 1000mg every 6 hours with a glass of water."
    mock_chunks = [
        MockChunk("You should take "),
        MockChunk("500mg or 1000mg "),
        MockChunk("every 6 hours "),
        MockChunk("with a glass of water.")
    ]

    mock_rag_chain = MagicMock()
    mock_rag_chain.invoke.return_value = {"answer": mock_llm_text, "context": []}

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    mock_retriever.no_reference_mode = False

    with patch("services.chat_pipeline.retriever", mock_retriever):
        with patch("services.chat_pipeline.create_retrieval_chain", return_value=mock_rag_chain):
            with patch.object(GroqChatModel, "invoke", return_value=MagicMock(content=mock_llm_text)):
                with patch.object(GroqChatModel, "stream", return_value=iter(mock_chunks)):
                    out_web = run_turn(query, user=user, chat_session=s_web, channel="web")
                    out_voice = run_turn(query, user=user, chat_session=s_voice, channel="voice")
                    out_stream = "".join(list(run_turn(query, user=user, chat_session=s_stream, channel="voice_stream")))

    for out in (out_web, out_voice, out_stream):
        assert "500mg" not in out
        assert "1000mg" not in out
        assert "[consult pharmacist or physician for dosage]" in out


def test_parity_non_medical_refusal():
    """Verify non-medical query yields NON_MEDICAL_REFUSAL and no memory tasks across all 3 channels."""
    user, s_web, s_voice, s_stream = _create_user_and_sessions()
    query = "Write a python script to calculate fibonacci numbers."

    with patch("services.task_dispatcher.enqueue_memory_update") as mock_mem:
        with patch("services.task_dispatcher.enqueue_title_update") as mock_title:
            with patch("services.task_dispatcher.enqueue_eval_safety_check") as mock_eval:
                out_web = run_turn(query, user=user, chat_session=s_web, channel="web")
                out_voice = run_turn(query, user=user, chat_session=s_voice, channel="voice")
                out_stream = "".join(list(run_turn(query, user=user, chat_session=s_stream, channel="voice_stream")))

                mock_mem.assert_not_called()
                mock_title.assert_not_called()
                mock_eval.assert_not_called()

    assert out_web == NON_MEDICAL_REFUSAL
    assert out_voice == NON_MEDICAL_REFUSAL
    assert out_stream == NON_MEDICAL_REFUSAL


def test_drift_fix_extract_patient_state_receives_classifier_model_consistently():
    """Verify drift fix: extract_patient_state is passed llm=classifierModel on web, voice, and stream."""
    user, s_web, s_voice, s_stream = _create_user_and_sessions()
    query = "I have a mild tension headache."

    calls = []

    def mock_extract(msg, patient_state, **kwargs):
        calls.append({"msg": msg, "llm": kwargs.get("llm")})
        return patient_state or PatientState()

    mock_llm_response = MagicMock(content="Rest and hydrate.")
    mock_chunks = [MockChunk("Rest and hydrate.")]

    mock_rag_chain = MagicMock()
    mock_rag_chain.invoke.return_value = {"answer": "Rest and hydrate.", "context": []}

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    mock_retriever.no_reference_mode = False

    with patch("research.src.clinical_triage.extract_patient_state", side_effect=mock_extract):
        with patch("services.chat_pipeline.retriever", mock_retriever):
            with patch("services.chat_pipeline.create_retrieval_chain", return_value=mock_rag_chain):
                with patch.object(GroqChatModel, "invoke", return_value=mock_llm_response):
                    with patch.object(GroqChatModel, "stream", return_value=iter(mock_chunks)):
                        run_turn(query, user=user, chat_session=s_web, channel="web")
                        run_turn(query, user=user, chat_session=s_voice, channel="voice")
                        list(run_turn(query, user=user, chat_session=s_stream, channel="voice_stream"))

    assert len(calls) == 3
    # Check that in all 3 channels, llm is classifierModel (non-None and consistent)
    for call in calls:
        assert call["llm"] is not None
        assert call["llm"] == classifierModel
