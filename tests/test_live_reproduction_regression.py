import unittest
import os
import sys
from dotenv import load_dotenv

load_dotenv(os.path.abspath(".env"))
sys.path.insert(0, ".")

from app import app
from research.src.auth import db, User, ChatSession, Message
from services.chat_service import load_patient_state, save_patient_state, build_history_text
from research.src.clinical_triage import extract_patient_state, evaluate_triage_tier
from services.ai_service import build_prompt, chatModel, classifierModel, retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from research.src.guardrails import apply_output_guardrails, UNIVERSALLY_PROHIBITED_DRUGS


class TestLiveReproductionRegression(unittest.TestCase):
    """
    Directly asserts expected behaviors on the user's reproduction case:
    Turn 1: "hello I have a fever from 3 days"
    Turn 2: "can you suggest any medications"
    """

    def setUp(self):
        self.app_context = app.app_context()
        self.app_context.push()
        db.create_all()
        self.user = User.query.first()
        if not self.user:
            self.user = User(email="test_repro@example.com", name="Alex Rivera")
            db.session.add(self.user)
            db.session.commit()

        self.session = ChatSession(user_id=self.user.id, title="Repro Audit Session")
        db.session.add(self.session)
        db.session.commit()

    def tearDown(self):
        if hasattr(self, "session") and self.session.id:
            try:
                Message.query.filter_by(session_id=self.session.id).delete()
                ChatSession.query.filter_by(id=self.session.id).delete()
                db.session.commit()
            except Exception:
                pass
        self.app_context.pop()

    def test_live_reproduction_turns_1_and_2(self):
        # ─────────────────────────────────────────────────────────────
        # TURN 1: "hello I have a fever from 3 days"
        # ─────────────────────────────────────────────────────────────
        msg_1 = "hello I have a fever from 3 days"
        db.session.add(Message(session_id=self.session.id, role="user", content=msg_1))
        db.session.commit()

        p1 = load_patient_state(self.session)
        p1 = extract_patient_state(msg_1, p1, llm=classifierModel)
        tier_1, flags_1, _ = evaluate_triage_tier(p1, msg_1)
        p1.risk_tier = tier_1

        prompt_1 = build_prompt(build_history_text(self.session), "", user=self.user, patient_state=p1)
        qa_1 = create_stuff_documents_chain(chatModel, prompt_1)
        res_1 = create_retrieval_chain(retriever, qa_1).invoke({"input": msg_1})
        raw_1 = res_1.get("answer", "")
        ans_1 = apply_output_guardrails(raw_1, is_medical=True, show_disclaimer=True, patient_state=p1, triage_tier=tier_1)
        p1.disclaimer_shown = True

        db.session.add(Message(session_id=self.session.id, role="assistant", content=ans_1))
        save_patient_state(self.session, p1)
        db.session.commit()

        # Assertions for Turn 1:
        ans_1_lower = ans_1.lower()
        # Bug 1: No unprompted condition/specialist hallucination
        self.assertNotIn("oncolog", ans_1_lower)
        self.assertNotIn("asthma", ans_1_lower)
        self.assertNotIn("cancer", ans_1_lower)

        # Bug 2: No specific drug names
        self.assertNotIn("acetaminophen", ans_1_lower)
        self.assertNotIn("ibuprofen", ans_1_lower)
        self.assertNotIn("aspirin", ans_1_lower)
        self.assertNotIn("paracetamol", ans_1_lower)

        # Bug 3: Triage questions asked
        has_triage_questions = any(q in ans_1_lower for q in ["temperature", "highest", "other symptom", "cough", "taken", "medication"])
        self.assertTrue(has_triage_questions, f"Turn 1 should ask clarifying triage questions: {ans_1}")

        # Bug 5: No dangling markdown truncation artifacts
        self.assertFalse(ans_1.endswith("*") and not ans_1.endswith("**") and not ans_1.endswith(".*"))

        # Bug 6: Disclaimer footer is present
        self.assertIn("Disclaimer: MediAssist provides informational", ans_1)

        # ─────────────────────────────────────────────────────────────
        # TURN 2: "can you suggest any medications"
        # ─────────────────────────────────────────────────────────────
        msg_2 = "can you suggest any medications"
        db.session.add(Message(session_id=self.session.id, role="user", content=msg_2))
        db.session.commit()

        p2 = load_patient_state(self.session)
        p2 = extract_patient_state(msg_2, p2, llm=classifierModel)
        tier_2, flags_2, _ = evaluate_triage_tier(p2, msg_2)
        p2.risk_tier = tier_2

        prompt_2 = build_prompt(build_history_text(self.session), "", user=self.user, patient_state=p2)
        qa_2 = create_stuff_documents_chain(chatModel, prompt_2)
        res_2 = create_retrieval_chain(retriever, qa_2).invoke({"input": msg_2})
        raw_2 = res_2.get("answer", "")
        is_med_inquiry = any(kw in msg_2.lower() for kw in ["medication", "medicine", "drug", "pill", "tablet", "dose", "dosing", "syrup"])
        show_disc_2 = (not p2.disclaimer_shown) and not is_med_inquiry
        ans_2 = apply_output_guardrails(raw_2, is_medical=True, show_disclaimer=show_disc_2, patient_state=p2, triage_tier=tier_2)

        print("\n--- TEST TURN 1 OUTPUT ---\n", ans_1)
        print("\n--- TEST TURN 2 OUTPUT ---\n", ans_2)
        ans_2_lower = ans_2.lower()
        # Bug 1 & 2: Zero hallucinated specialists or prohibited drugs
        self.assertNotIn("oncolog", ans_2_lower)
        self.assertNotIn("asthma", ans_2_lower)
        self.assertNotIn("acetaminophen", ans_2_lower)
        self.assertNotIn("ibuprofen", ans_2_lower)

        # Bug 4: Continuity — directly addresses medication inquiry without copying Turn 1's question block
        self.assertTrue("fever reducer" in ans_2_lower or "pharmacist" in ans_2_lower or "over-the-counter" in ans_2_lower)

        # Bug 5: No mid-sentence truncation
        self.assertTrue(ans_2.strip().endswith((".", "!", ")", "*")), f"Turn 2 ended mid-sentence: {ans_2}")

        # Bug 6: Disclaimer omitted on follow-up medication responses to prevent repetitive annoyance
        self.assertNotIn("Disclaimer: MediAssist provides informational", ans_2)


if __name__ == "__main__":
    unittest.main()
