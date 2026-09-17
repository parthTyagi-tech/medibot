import unittest
import json
from app import app
from research.src.auth import db, User, ChatSession
from research.src.clinical_triage import (
    PatientState,
    extract_patient_state
)
from services.chat_service import (
    load_patient_state,
    save_patient_state
)


class TestStatePersistenceMultiWorker(unittest.TestCase):
    """
    Simulates multi-process/multi-worker state persistence via database.
    Confirms that turn 1 state saved to DB is cleanly loaded in an isolated worker context on turn 2.
    """

    def setUp(self):
        app.config["TESTING"] = True
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
        self.app_context = app.app_context()
        self.app_context.push()
        db.create_all()

        # Create test user
        self.user = User(email="worker_test@example.com", name="Worker Test")
        db.session.add(self.user)
        db.session.commit()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def test_multiworker_database_state_persistence(self):
        # ── Worker A (Turn 1) ──────────────────────────────────
        session_a = ChatSession(user_id=self.user.id)
        db.session.add(session_a)
        db.session.commit()
        session_id = session_a.id

        # Worker A extracts Turn 1 symptoms & duration
        state_worker_a = load_patient_state(session_a)
        state_worker_a = extract_patient_state("I have a fever for 3 days", state_worker_a)
        save_patient_state(session_a, state_worker_a)
        db.session.commit()

        # Worker A process ends / memory cleared
        del state_worker_a
        del session_a
        db.session.expunge_all()

        # ── Worker B (Turn 2) ──────────────────────────────────
        # Worker B starts fresh, loads session from DB using session_id
        session_b = db.session.get(ChatSession, session_id)
        self.assertIsNotNone(session_b)
        self.assertTrue(len(session_b.patient_state_json) > 0)

        # Worker B loads patient state from DB
        state_worker_b = load_patient_state(session_b)
        self.assertIn("fever", state_worker_b.current_symptoms)
        self.assertIn("3 day", state_worker_b.duration.lower())

        # Worker B extracts Turn 2 details (temperature, dry cough, no medicines)
        state_worker_b = extract_patient_state("temp - 103, dry cough no medicines", state_worker_b)
        save_patient_state(session_b, state_worker_b)
        db.session.commit()

        # Confirm that both Turn 1 and Turn 2 facts are present in DB
        db.session.expunge_all()
        reloaded_session = db.session.get(ChatSession, session_id)
        final_state = load_patient_state(reloaded_session)

        self.assertIn("103", final_state.temperature)
        self.assertIn("3 day", final_state.duration.lower())
        self.assertEqual(final_state.medication_status, "No medicines taken")
        self.assertIn("dry cough", final_state.reported_symptoms_detail)


if __name__ == "__main__":
    unittest.main()
