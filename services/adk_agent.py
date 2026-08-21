"""
Google Agent Development Kit (ADK) Medical Agent & Clinical Triage Pipeline.
Follows the /google_adk_integration skill directives.
Provides:
- Clinical Doctor Triage Protocol (collects essential details before diagnosing, avoids 1000-word dumps)
- Concise, precise medical advice grounded in The Gale Encyclopedia of Medicine
- Rolling context window summarization & essential memory management
"""

import os
import logging
from typing import Optional, List, Dict, Any

try:
    from google.adk import Agent
except Exception:
    Agent = None

try:
    from google import genai
except Exception:
    genai = None

from research.src.guardrails import (
    apply_input_guardrails,
    apply_output_guardrails,
    NON_MEDICAL_REFUSAL
)

logger = logging.getLogger("adk-agent")


# ─────────────────────────────────────────────────────────────
# 1. Clinical Doctor Instruction for Google ADK Agent
# ─────────────────────────────────────────────────────────────

CLINICAL_DOCTOR_INSTRUCTION = """You are MediAssist, an expert, empathetic, and highly precise clinical doctor AI.

CLINICAL CONSULTATION PROTOCOL:
1. HOW REAL DOCTORS OPERATE (TRIAGE FIRST):
   - When a patient presents with an initial symptom without details (e.g., "I have a fever", "I have a headache", "My stomach hurts", "I have asthma"):
     * Do NOT dump a long 10-paragraph essay or full encyclopedia article!
     * Express warm, brief empathy (1 sentence).
     * Ask 2-3 focused, essential clinical triage questions to collect genuine details:
       a. Onset & Duration (When did it start? Sudden or gradual?)
       b. Severity & Measurable signs (e.g. Temperature if fever, scale of 1-10 if pain)
       c. Associated symptoms (e.g. Chills, rash, cough, breathing issues)
       d. Current medications or existing health conditions
     * Keep your initial response under 100 words.

2. PRECISE & TARGETED ASSESSMENT (ONCE DETAILS ARE PROVIDED):
   - When the user gives the details or asks a specific medical question:
     * Provide a direct, concise, and structured assessment (100 to 200 words max).
     * Use clear, bulleted sections:
       - **Likely Causes / Clinical Insight**: Concise explanation based on medical facts.
       - **Self-Care & Immediate Measures**: Practical, safe home care steps.
       - **When to See a Doctor (Red Flags)**: Specific warning signs requiring in-person evaluation.
     * Never hallucinate or invent unverified medical facts.

3. STRICT MEDICAL SPECIALIZATION:
   - You ONLY answer health, medical, wellness, symptom, and clinical questions.
   - If a user asks for non-medical topics (coding, math, general trivia, essays), politely refuse and reiterate your medical focus.

4. SAFETY & RED FLAGS:
   - For acute emergencies (severe chest pain, stroke symptoms, difficulty breathing), immediately direct them to emergency services (911/112/999).
"""

# ─────────────────────────────────────────────────────────────
# 2. Google ADK Medical Agent Instance
# ─────────────────────────────────────────────────────────────

def create_adk_medical_agent(api_key: Optional[str] = None):
    """
    Creates and configures the Google ADK Medical Agent using gemini-flash-latest / gemini-2.5-flash.
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if key:
        os.environ["GEMINI_API_KEY"] = key

    if Agent is not None:
        try:
            return Agent(
                name="MediAssistClinicalAgent",
                model="gemini-2.5-flash",
                instruction=CLINICAL_DOCTOR_INSTRUCTION,
                description="Clinical Medical Assistant trained for doctor-patient consultation and precise medical guidance."
            )
        except Exception as e:
            logger.warning(f"[ADK Agent] Initialization notice: {e}")

    # Fallback lightweight proxy
    class FallbackADKAgent:
        name = "MediAssistClinicalAgent"
        model = "gemini-2.5-flash"
        instruction = CLINICAL_DOCTOR_INSTRUCTION

    return FallbackADKAgent()



# Global ADK Agent instance
adk_agent = create_adk_medical_agent()



# ─────────────────────────────────────────────────────────────
# 3. Context Window Summarization & Memory Manager
# ─────────────────────────────────────────────────────────────

def summarize_context_window(client: genai.Client, history_messages: List[Dict[str, str]]) -> str:
    """
    Summarizes older conversation turns to maintain a clean, high-density context window.
    """
    if len(history_messages) < 4:
        return ""

    conversation_text = "\n".join([
        f"{m.get('role', 'User')}: {m.get('content', '')}"
        for m in history_messages
    ])

    prompt = f"""Summarize this patient consultation in 2-3 concise bullet points.
Focus ONLY on:
- Primary symptoms & timeline
- Relevant user medical history mentioned
- Guidance or recommendations given so far

Conversation:
{conversation_text}

Summary:"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return (response.text or "").strip()
    except Exception as e:
        logger.warning(f"[ADK Context Window] Summarization notice: {e}")
        return ""
