import os
import time
import json
import hmac
import hashlib
from unittest.mock import patch, MagicMock
import pytest

from app import app, db
from research.src.auth import User, ChatSession, Message
from research.src.clinical_triage import PatientState
from services.output_guardrails import (
    StreamingGuardrailProcessor,
    ClinicalOutputGuardrail,
    find_sentence_boundary,
    is_patient_high_risk
)
from services.ai_service import GroqChatModel
from research.src.guardrails import apply_output_guardrails


from services.state_store import reset_state_store


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


def test_sentence_splitter_ignores_decimals_and_abbreviations():
    """Verify sentence splitter does not split on decimal numbers or abbreviations."""
    # Decimals: 10.5 mg
    text_with_decimal = "The dosage is 10.5 mg in the morning. Please take it with food."
    b1 = find_sentence_boundary(text_with_decimal)
    assert b1 is not None
    first_sentence = text_with_decimal[:b1]
    assert first_sentence == "The dosage is 10.5 mg in the morning. "

    # Abbreviations: Dr., vs., e.g.
    text_with_abbrev = "Please contact Dr. Smith vs. your local clinic e.g. urgent care. Rest well."
    b2 = find_sentence_boundary(text_with_abbrev)
    assert b2 is not None
    full_sentence = text_with_abbrev[:b2]
    assert full_sentence == "Please contact Dr. Smith vs. your local clinic e.g. urgent care. "


def test_dose_split_across_chunks():
    """Verify numeric dosage is suppressed even when digits and units are split across stream chunks."""
    patient_state = PatientState(has_cancer_history=False)
    processor = StreamingGuardrailProcessor(
        patient_state=patient_state,
        risk_tier="Routine",
        is_medical=True
    )

    chunks = [
        MockChunk("You "),
        MockChunk("can "),
        MockChunk("take "),
        MockChunk("500"),
        MockChunk("mg "),
        MockChunk("of "),
        MockChunk("paracetamol "),
        MockChunk("for "),
        MockChunk("pain. "),
        MockChunk("Rest "),
        MockChunk("well.")
    ]

    yielded = list(processor.process_stream(chunks))
    full_yielded = "".join(yielded)

    # 500mg must be suppressed
    assert "500mg" not in full_yielded
    assert "[consult pharmacist or physician for dosage]" in full_yielded
    assert processor.get_final_text() == full_yielded


def test_prohibition_in_sentence1_followed_by_neutral_sentence2():
    """Verify CLINICAL_OVERRIDE_TEXT is NOT appended when prohibition is present in stream."""
    patient_state = PatientState(has_cancer_history=True)
    processor = StreamingGuardrailProcessor(
        patient_state=patient_state,
        risk_tier="Routine",
        is_medical=True
    )

    chunks = [
        MockChunk("Do not take ibuprofen for fever. "),
        MockChunk("Drink plenty of water and rest.")
    ]

    yielded = list(processor.process_stream(chunks))
    full_yielded = "".join(yielded)

    # Valid prohibition must be preserved intact without redundant override append
    assert "Do not take ibuprofen for fever." in full_yielded
    assert "Drink plenty of water and rest." in full_yielded
    # Because prohibition was explicit, the critical override warning is not duplicated
    assert full_yielded.count(ClinicalOutputGuardrail.CLINICAL_OVERRIDE_TEXT) == 0


def test_prohibition_sentence_for_oncology_patient_preserved_intact():
    """Verify that a clinically sound prohibition for an oncology patient remains intact."""
    patient_state = PatientState(is_active_cancer_chemo=True)
    processor = StreamingGuardrailProcessor(
        patient_state=patient_state,
        risk_tier="Routine",
        is_medical=True
    )

    chunks = [
        MockChunk("You should avoid ibuprofen and acetaminophen as they can mask serious infections. "),
        MockChunk("Please contact your oncologist immediately.")
    ]

    yielded = list(processor.process_stream(chunks))
    full_yielded = "".join(yielded)

    assert "avoid ibuprofen and acetaminophen" in full_yielded
    assert "contact your oncologist immediately" in full_yielded


def test_stream_with_no_terminator_flushes_at_cap():
    """Verify that a long stream without punctuation flushes at ~300 characters."""
    patient_state = PatientState(has_cancer_history=False)
    processor = StreamingGuardrailProcessor(
        patient_state=patient_state,
        risk_tier="Routine",
        is_medical=True
    )

    # 35 words of 10 chars each = ~350 characters with no periods
    chunks = [MockChunk(f"word{i} ") for i in range(70)]

    yielded = list(processor.process_stream(chunks))

    # Must have flushed mid-stream before stream termination
    assert len(yielded) >= 2
    full_yielded = "".join(yielded)
    assert processor.get_final_text() == full_yielded


def test_mid_stream_exception_handling():
    """Verify processor emits safe fallback, discards unverified buffer, and tracks yielded text."""
    patient_state = PatientState(has_cancer_history=False)
    processor = StreamingGuardrailProcessor(
        patient_state=patient_state,
        risk_tier="Routine",
        is_medical=True
    )

    def failing_stream():
        yield MockChunk("Please stay hydrated and get plenty of rest. ")
        raise RuntimeError("Simulated connection dropped mid-stream")

    yielded = list(processor.process_stream(failing_stream()))
    full_yielded = "".join(yielded)

    assert "Please stay hydrated and get plenty of rest." in full_yielded
    assert StreamingGuardrailProcessor.STREAM_INTERRUPTED_FALLBACK in full_yielded
    assert processor.get_final_text() == full_yielded


def test_streaming_parity_with_non_streaming_sanitization():
    """Verify streaming result produces identical clinical sanitization to non-streaming output."""
    patient_state = PatientState(has_cancer_history=True)

    test_raw_text = (
        "You can take 650mg of acetaminophen every 6 hours for your fever. "
        "Also follow the BRAT diet to help with digestion. "
        "Keep in touch with your oncologist."
    )

    # Non-streaming path
    non_stream_sanitized = apply_output_guardrails(
        test_raw_text,
        is_medical=True,
        show_disclaimer=False,
        patient_state=patient_state
    )
    non_stream_final, _ = ClinicalOutputGuardrail.sanitize_response(non_stream_sanitized, patient_state)

    # Streaming path (oncology is high-risk -> full-response buffered)
    processor = StreamingGuardrailProcessor(
        patient_state=patient_state,
        risk_tier="Routine",
        is_medical=True
    )
    chunks = [MockChunk(w + " ") for w in test_raw_text.split()]
    stream_yielded = "".join(processor.process_stream(chunks))

    # Parity assertions:
    # 1. 650mg and every 6 hours suppressed
    assert "[consult pharmacist or physician for dosage]" in stream_yielded
    assert "[consult pharmacist or physician for dosage]" in non_stream_final
    # 2. BRAT diet scrubbed
    assert "a balanced, nutrient-dense diet" in stream_yielded
    assert "a balanced, nutrient-dense diet" in non_stream_final
    # 3. Critical antipyretic override enforced
    assert ClinicalOutputGuardrail.CLINICAL_OVERRIDE_TEXT in stream_yielded
    assert ClinicalOutputGuardrail.CLINICAL_OVERRIDE_TEXT in non_stream_final


def test_voice_chat_endpoint_streaming_antipyretic_sanitization_and_persistence():
    """End-to-end endpoint test: /voice_chat streams sanitized text and persists exact match in DB."""
    client = app.test_client()
    user = User(email="onco.patient@example.com", name="Onco Patient", password_hash="hash")
    db.session.add(user)
    db.session.commit()

    chat_session = ChatSession(user_id=user.id, title="Consultation")
    db.session.add(chat_session)
    db.session.commit()

    # Create patient state with cancer history
    patient_state = PatientState(has_cancer_history=True)
    chat_session.patient_state_json = json.dumps(patient_state.to_dict())
    db.session.commit()
    session_id = chat_session.id

    secret = os.getenv("VOICE_INTERNAL_SECRET", "test-voice-secret-key-for-test-suites-32chars")
    ts = str(int(time.time()))
    payload = {"message": "I have a fever, what medicine can I take?", "user_id": user.id, "stream": True}
    body_bytes = json.dumps(payload, separators=(',', ':')).encode("utf-8")
    canonical = f"{ts}:/voice_chat:{user.id}:".encode("utf-8") + body_bytes
    sig = hmac.new(secret.encode("utf-8"), canonical, hashlib.sha256).hexdigest()

    mock_llm_chunks = [
        MockChunk("You "),
        MockChunk("can "),
        MockChunk("take "),
        MockChunk("ibuprofen "),
        MockChunk("for "),
        MockChunk("the "),
        MockChunk("fever. "),
        MockChunk("Rest "),
        MockChunk("well.")
    ]

    with patch.object(GroqChatModel, "stream", return_value=iter(mock_llm_chunks)):
        resp = client.post(
            "/voice_chat",
            data=body_bytes,
            content_type="application/json",
            headers={
                "X-Voice-Timestamp": ts,
                "X-Voice-Signature": sig
            }
        )

        assert resp.status_code == 200
        streamed_content = resp.data.decode("utf-8")

        # 1. Permissive antipyretic must NOT reach user
        assert "take ibuprofen" not in streamed_content.lower()
        # 2. Critical oncology override MUST be present
        assert ClinicalOutputGuardrail.CLINICAL_OVERRIDE_TEXT in streamed_content

        # 3. DB Message.content must match the exact streamed content
        saved_msg = Message.query.filter_by(session_id=session_id, role="assistant").order_by(Message.id.desc()).first()
        assert saved_msg is not None
        assert saved_msg.content == streamed_content


def test_voice_chat_endpoint_streaming_dosing_suppression_and_persistence():
    """End-to-end endpoint test: /voice_chat suppresses numeric dosage and persists exact match in DB."""
    client = app.test_client()
    user = User(email="standard.patient@example.com", name="Standard Patient", password_hash="hash")
    db.session.add(user)
    db.session.commit()

    chat_session = ChatSession(user_id=user.id, title="Consultation")
    patient_state = PatientState(has_cancer_history=False)
    chat_session.patient_state_json = json.dumps(patient_state.to_dict())
    db.session.add(chat_session)
    db.session.commit()
    session_id = chat_session.id

    secret = os.getenv("VOICE_INTERNAL_SECRET", "test-voice-secret-key-for-test-suites-32chars")
    ts = str(int(time.time()))
    payload = {"message": "How much medicine should I take?", "user_id": user.id, "stream": True}
    body_bytes = json.dumps(payload, separators=(',', ':')).encode("utf-8")
    canonical = f"{ts}:/voice_chat:{user.id}:".encode("utf-8") + body_bytes
    sig = hmac.new(secret.encode("utf-8"), canonical, hashlib.sha256).hexdigest()

    mock_llm_chunks = [
        MockChunk("Take "),
        MockChunk("650"),
        MockChunk("mg "),
        MockChunk("every "),
        MockChunk("4 to 6 hours. "),
        MockChunk("Drink "),
        MockChunk("water.")
    ]

    with patch.object(GroqChatModel, "stream", return_value=iter(mock_llm_chunks)):
        resp = client.post(
            "/voice_chat",
            data=body_bytes,
            content_type="application/json",
            headers={
                "X-Voice-Timestamp": ts,
                "X-Voice-Signature": sig
            }
        )

        assert resp.status_code == 200
        streamed_content = resp.data.decode("utf-8")

        # Dosage numbers must be suppressed
        assert "650mg" not in streamed_content
        assert "[consult pharmacist or physician for dosage]" in streamed_content

        # DB Message.content must match exact streamed content
        saved_msg = Message.query.filter_by(session_id=session_id, role="assistant").order_by(Message.id.desc()).first()
        assert saved_msg is not None
        assert saved_msg.content == streamed_content
