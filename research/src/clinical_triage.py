"""
MediAssist Clinical Triage & Safety Architecture
=================================================
Auditable, evidence-based triage decision support engine.
Implements:
1. PatientState structured state management across conversation turns
2. Auditable Triage Risk Tiering (Emergency / Urgent / Routine / Informational)
3. Red-flag override protocols (e.g. Febrile Neutropenia, Neonatal Sepsis, Preeclampsia)
4. Medication & Dosing Safety Guardrails (Contraindication blocks)
5. Mid-conversation condition disclosure re-evaluation & correction alerts
6. Authoritative clinical citations (WHO, CDC, UpToDate, NIH, Gale Encyclopedia of Medicine)
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

# ─────────────────────────────────────────────────────────────
# 1. Structured Patient State
# ─────────────────────────────────────────────────────────────

@dataclass
class PatientState:
    """
    Maintains structured patient facts across conversation turns.
    Strictly records user-disclosed attributes to prevent hallucinated assumptions.
    """
    age: Optional[int] = None
    age_unit: str = "years"  # "years", "months", "days"
    is_infant_under_3mo: bool = False
    is_pregnant: bool = False
    is_elderly: bool = False
    is_immunocompromised: bool = False
    is_active_cancer_chemo: bool = False

    # Detailed vitals, duration & medication state (v2)
    temperature: Optional[str] = None
    duration: Optional[str] = None
    medication_status: Optional[str] = None
    reported_symptoms_detail: List[str] = field(default_factory=list)
    disclosed_conditions: List[str] = field(default_factory=list)
    active_chemo_confirmed: Optional[bool] = None  # None=unconfirmed, True=confirmed active, False=confirmed finished/none
    stand_down_reason: Optional[str] = None
    
    conditions: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    current_symptoms: List[str] = field(default_factory=list)
    
    # Triage classification
    risk_tier: str = "Informational"  # "Emergency", "Urgent", "Routine", "Informational"
    red_flags: List[str] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)
    
    # State tracking
    disclaimer_shown: bool = False
    prior_advice_history: List[Dict[str, Any]] = field(default_factory=list)
    newly_disclosed_high_risk: Optional[str] = None
    needs_prior_advice_correction: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "age": self.age,
            "age_unit": self.age_unit,
            "is_infant_under_3mo": self.is_infant_under_3mo,
            "is_pregnant": self.is_pregnant,
            "is_elderly": self.is_elderly,
            "is_immunocompromised": self.is_immunocompromised,
            "is_active_cancer_chemo": self.is_active_cancer_chemo,
            "temperature": self.temperature,
            "duration": self.duration,
            "medication_status": self.medication_status,
            "reported_symptoms_detail": self.reported_symptoms_detail,
            "disclosed_conditions": self.disclosed_conditions,
            "active_chemo_confirmed": self.active_chemo_confirmed,
            "stand_down_reason": self.stand_down_reason,
            "conditions": self.conditions,
            "medications": self.medications,
            "allergies": self.allergies,
            "current_symptoms": self.current_symptoms,
            "risk_tier": self.risk_tier,
            "red_flags": self.red_flags,
            "contraindications": self.contraindications,
            "disclaimer_shown": self.disclaimer_shown,
            "newly_disclosed_high_risk": self.newly_disclosed_high_risk,
            "needs_prior_advice_correction": self.needs_prior_advice_correction
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatientState":
        if not data:
            return cls()
        return cls(
            age=data.get("age"),
            age_unit=data.get("age_unit", "years"),
            is_infant_under_3mo=data.get("is_infant_under_3mo", False),
            is_pregnant=data.get("is_pregnant", False),
            is_elderly=data.get("is_elderly", False),
            is_immunocompromised=data.get("is_immunocompromised", False),
            is_active_cancer_chemo=data.get("is_active_cancer_chemo", False),
            temperature=data.get("temperature"),
            duration=data.get("duration"),
            medication_status=data.get("medication_status"),
            reported_symptoms_detail=data.get("reported_symptoms_detail", []),
            disclosed_conditions=data.get("disclosed_conditions", []),
            active_chemo_confirmed=data.get("active_chemo_confirmed"),
            stand_down_reason=data.get("stand_down_reason"),
            conditions=data.get("conditions", []),
            medications=data.get("medications", []),
            allergies=data.get("allergies", []),
            current_symptoms=data.get("current_symptoms", []),
            risk_tier=data.get("risk_tier", "Informational"),
            red_flags=data.get("red_flags", []),
            contraindications=data.get("contraindications", []),
            disclaimer_shown=data.get("disclaimer_shown", False),
            newly_disclosed_high_risk=data.get("newly_disclosed_high_risk"),
            needs_prior_advice_correction=data.get("needs_prior_advice_correction", False)
        )


# ─────────────────────────────────────────────────────────────
# 2. Auditable Clinical Triage Matrix (Version 1.0)
# ─────────────────────────────────────────────────────────────

AUDITABLE_TRIAGE_MATRIX = {
    "VERSION": "1.0.4",
    "LAST_REVIEWED": "2026-08",
    "RULES": [
        {
            "id": "EMERG-001",
            "name": "Febrile Neutropenia in Oncology/Immunosuppression",
            "trigger_conditions": ["active_cancer", "chemotherapy", "immunocompromised", "bone_marrow_transplant"],
            "trigger_symptoms": ["fever", "chills", "temperature >= 38.0C (100.4F)", "shivering"],
            "risk_tier": "Emergency",
            "action": "Immediate Emergency Department evaluation or 24/7 Oncology Hotline.",
            "strict_prohibitions": ["No home remedies", "No antipyretics without oncologist approval"],
            "citation": "Infectious Diseases Society of America (IDSA) / ASCO Clinical Practice Guidelines for Febrile Neutropenia"
        },
        {
            "id": "EMERG-002",
            "name": "Neonatal Fever / Sepsis Risk (<3 months)",
            "trigger_conditions": ["infant_under_3mo", "age < 90 days"],
            "trigger_symptoms": ["fever >= 38.0C (100.4F)", "lethargy", "poor feeding", "irritability"],
            "risk_tier": "Emergency",
            "action": "Immediate Emergency Department evaluation for full neonatal sepsis workup.",
            "strict_prohibitions": ["No OTC antipyretics", "No delay", "No home observation"],
            "citation": "American Academy of Pediatrics (AAP) Clinical Practice Guideline: Evaluation and Management of Well-Appearing Febrile Infants 8 to 60 Days Old"
        },
        {
            "id": "EMERG-003",
            "name": "Preeclampsia / Maternal Emergency in Pregnancy",
            "trigger_conditions": ["pregnancy", "postpartum"],
            "trigger_symptoms": ["severe headache", "visual changes", "upper abdominal pain", "facial swelling", "shortness of breath", "fever"],
            "risk_tier": "Emergency",
            "action": "Immediate Obstetric / Emergency Department evaluation.",
            "strict_prohibitions": ["No home pain relief without obstetrical triage"],
            "citation": "American College of Obstetricians and Gynecologists (ACOG) Guidelines on Gestational Hypertension and Preeclampsia"
        },
        {
            "id": "EMERG-004",
            "name": "Acute Coronary Syndrome / Stroke / Anaphylaxis",
            "trigger_conditions": ["any"],
            "trigger_symptoms": ["crushing chest pain", "slurred speech", "facial droop", "arm weakness", "stridor", "throat closing"],
            "risk_tier": "Emergency",
            "action": "Call 911/112/999 immediately. Do not drive to hospital.",
            "strict_prohibitions": ["No waiting", "No oral fluids/meds in acute airway compromise"],
            "citation": "AHA/ACC Emergency Cardiovascular Care Guidelines & CDC Stroke Protocol"
        },
        {
            "id": "URGENT-001",
            "name": "High Prolonged Fever or Respiratory Distress in Chronic Illness",
            "trigger_conditions": ["asthma", "copd", "diabetes", "heart_failure", "elderly"],
            "trigger_symptoms": ["fever > 39.4C (103F)", "fever > 3 days", "wheezing", "productive cough with dyspnea"],
            "risk_tier": "Urgent",
            "action": "Urgent Care / Same-Day Primary Care evaluation within hours.",
            "strict_prohibitions": ["No unmonitored escalation of bronchodilators without clinical advice"],
            "citation": "CDC Influenza & Respiratory Illness Clinical Guidance"
        },
        {
            "id": "ROUTINE-001",
            "name": "Uncomplicated Acute Symptoms in Standard Risk Patient",
            "trigger_conditions": ["none_high_risk"],
            "trigger_symptoms": ["mild fever < 48h", "sore throat", "runny nose", "mild headache", "minor cough"],
            "risk_tier": "Routine",
            "action": "Structured clinical triage questioning (onset, duration, severity, red-flag screening) and supportive hydration/rest.",
            "strict_prohibitions": ["No specific prescription drug dosing"],
            "citation": "WHO Clinical Guidelines & The Gale Encyclopedia of Medicine"
        },
        {
            "id": "INFO-001",
            "name": "General Health, Wellness & Educational Inquiries",
            "trigger_conditions": ["none"],
            "trigger_symptoms": ["general question", "how does insulin work", "what is hypertension"],
            "risk_tier": "Informational",
            "action": "Educational clinical explanations citing authoritative sources.",
            "strict_prohibitions": ["No definitive personalized diagnosis"],
            "citation": "NIH / CDC Health Topics & The Gale Encyclopedia of Medicine"
        }
    ]
}


# ─────────────────────────────────────────────────────────────
# 3. Patient State Extraction (Rule & Regex Based)
# ─────────────────────────────────────────────────────────────

PATIENT_CONDITION_PATTERNS = {
    "active_cancer_chemo": [
        r"\b(chemotherapy|chemo|cancer\s+treatment|active\s+cancer|oncologist|radiation\s+therapy|leukemia|lymphoma|carcinoma|on\s+chemo)\b"
    ],
    "immunocompromised": [
        r"\b(immunocompromised|immunosuppressed|organ\s+transplant|kidney\s+transplant|liver\s+transplant|hiv|aids|on\s+steroids|prednisone|methotrexate|biologics|tacrolimus)\b"
    ],
    "pregnancy": [
        r"\b(pregnant|pregnancy|expecting|trimester|weeks\s+pregnant|postpartum|nursing)\b"
    ],
    "infant": [
        r"\b(\b\d+\s*(month|months|mo|day|days|week|weeks)\s*old\b|infant|newborn|baby)\b"
    ],
    "chronic_disease": [
        r"\b(diabetes|type\s+1|type\s+2|hypertension|high\s+blood\s+pressure|asthma|copd|kidney\s+disease|chronic\s+kidney|liver\s+disease|cirrhosis|heart\s+failure|congestive\s+heart\s+failure)\b"
    ],
    "elderly": [
        r"\b(\b(7[0-9]|8[0-9]|9[0-9]|10[0-9])\s*years?\s*old\b|elderly|geriatric|senior\s+citizen)\b"
    ]
}

SYMPTOM_PATTERNS = {
    "fever": r"\b(fever|temperature|high\s+temp|feverish|chills|shivering|burning\s+up|\d+(\.\d+)?\s*(f|c|degrees))\b",
    "respiratory": r"\b(shortness\s+of\s+breath|difficulty\s+breathing|wheezing|cough|coughing|sore\s+throat|stridor)\b",
    "cardiac": r"\b(chest\s+pain|chest\s+pressure|palpitations|irregular\s+heartbeat|tightness\s+in\s+chest)\b",
    "neurological": r"\b(severe\s+headache|headache|dizziness|fainting|syncope|confusion|slurred\s+speech|vision\s+loss|blurred\s+vision|blurry\s+vision|seizure)\b",
    "gastrointestinal": r"\b(vomiting|diarrhea|abdominal\s+pain|stomach\s+cramps|unable\s+to\s+keep\s+fluids\s+down|dehydration)\b",
    "preeclampsia_signs": r"\b(headache|blurr(y|ed)\s+vision|visual\s+(changes|disturbances)|swelling\s+in\s+(hands|face|feet)|upper\s+right\s+(belly|abdominal)\s+pain|sudden\s+swelling)\b"
}

DOSING_REQUEST_PATTERNS = [
    r"\b(how\s+much\s+(can|should|do|to)\s+i\s+take|what\s+dose|dosage\s+of|how\s+many\s+mg|how\s+many\s+pills|how\s+many\s+tablets|prescribe\s+me|medication\s+amount|give\s+me\s+(the\s+)?(exact\s+)?dose|tell\s+me\s+the\s+(mg|dose|number)|exact\s+dose)\b",
    r"\b(paracetamol|ibuprofen|acetaminophen|amoxicillin|aspirin|tylenol|advil|dolo)\s*(dose|dosage|amount|mg)\b",
    r"\b(dose|dosage)\s+of\s+(paracetamol|ibuprofen|acetaminophen|amoxicillin|aspirin|tylenol|advil|dolo)\b",
    r"\b(tell\s+me|give\s+me)\s+(the\s+)?(mg|milligrams|number|dose|dosage)\b",
    r"\b(tell\s+me|give\s+me)\s+the\s+number\b"
]


def extract_patient_state_llm(user_text: str, llm=None) -> Optional[Dict[str, Any]]:
    """Primary: Structured LLM extraction for natural medical language."""
    if not llm:
        return None
    import json
    prompt = (
        "You are a clinical NLP information extraction system. "
        "Extract the following clinical attributes from the patient's message. "
        "Return ONLY a valid JSON object with these exact keys (use null or [] if absent):\n"
        "{\n"
        "  \"temperature\": string or null,\n"
        "  \"duration\": string or null,\n"
        "  \"medications_taken\": string or null,\n"
        "  \"symptoms\": [string],\n"
        "  \"conditions_disclosed\": [string],\n"
        "  \"active_chemo_confirmed\": boolean or null,\n"
        "  \"treatment_ended_long_ago\": boolean or null\n"
        "}\n\n"
        f"Patient Message: \"{user_text}\"\n\n"
        "JSON:"
    )
    try:
        resp = llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        m = re.search(r"\{[\s\S]*\}", content)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    return None


def extract_patient_state(user_text: str, current_state: Optional[PatientState] = None, llm=None) -> PatientState:
    """
    Extracts and updates structured patient attributes from user input.
    Guarantees that state only reflects explicit disclosures and accumulates additively across turns.
    Uses structured LLM extraction when available, backed by comprehensive regex fallbacks.
    """
    state = current_state or PatientState()
    text = user_text.lower().strip()

    # 0. Primary: Structured LLM extraction if LLM is provided
    llm_data = extract_patient_state_llm(user_text, llm)
    if llm_data and isinstance(llm_data, dict):
        if llm_data.get("temperature") and not state.temperature:
            state.temperature = str(llm_data["temperature"])
        if llm_data.get("duration") and not state.duration:
            state.duration = str(llm_data["duration"])
        if llm_data.get("medications_taken") and not state.medication_status:
            state.medication_status = str(llm_data["medications_taken"])
        if llm_data.get("symptoms"):
            for s in llm_data["symptoms"]:
                s_clean = str(s).strip().lower()
                if s_clean and s_clean not in [x.lower() for x in state.reported_symptoms_detail]:
                    state.reported_symptoms_detail.append(s_clean)
        if llm_data.get("conditions_disclosed"):
            for c in llm_data["conditions_disclosed"]:
                c_clean = str(c).strip().title()
                if c_clean and c_clean not in state.disclosed_conditions:
                    state.disclosed_conditions.append(c_clean)
                if c_clean.lower() in ("cancer", "active cancer", "chemo", "chemotherapy"):
                    state.is_active_cancer_chemo = True
                    if "Active Chemotherapy/Cancer" not in state.conditions:
                        state.conditions.append("Active Chemotherapy/Cancer")
        if llm_data.get("active_chemo_confirmed") is True:
            state.active_chemo_confirmed = True
        elif llm_data.get("treatment_ended_long_ago") is True:
            state.active_chemo_confirmed = False
            state.stand_down_reason = "Affirmative disclosure: treatment ended >12 months ago with no active therapy"

    # 1. Detect Age
    age_match = re.search(r"\b(\d+)\s*[- ]*(months?|mo|days?|weeks?|years?|yrs?|yo)\s*[- ]*old\b", text)
    if not age_match:
        age_match = re.search(r"\b(i\s*am|patient\s*is|my\s*age\s*is)\s*(\d+)\b", text)
        if age_match:
            try:
                state.age = int(age_match.group(2))
                state.age_unit = "years"
            except Exception:
                pass
    else:
        try:
            val = int(age_match.group(1))
            unit = age_match.group(2)
            state.age = val
            if "mo" in unit:
                state.age_unit = "months"
                if val < 3:
                    state.is_infant_under_3mo = True
            elif "day" in unit:
                state.age_unit = "days"
                if val < 90:
                    state.is_infant_under_3mo = True
            elif "week" in unit:
                state.age_unit = "weeks"
                if val < 13:
                    state.is_infant_under_3mo = True
            else:
                state.age_unit = "years"
                if val >= 65:
                    state.is_elderly = True
        except Exception:
            pass

    # 2. Extract Quantitative Vitals (Temperature)
    if not state.temperature:
        temp_match = re.search(
            r"\b(temp(erature)?\s*[-:=]?\s*(10[0-6](\.\d+)?|9[6-9](\.\d+)?|3[6-9](\.\d+)?|4[0-2](\.\d+)?)\s*(°?\s*[fc]|degrees)?|"
            r"(10[0-6](\.\d+)?|9[6-9](\.\d+)?)\s*(°?\s*f|degrees\s*f|degrees\b|f\b)|"
            r"(3[7-9](\.\d+)?|4[0-2](\.\d+)?)\s*(°?\s*c|degrees\s*c|c\b))\b",
            text
        )
        if temp_match:
            state.temperature = temp_match.group(0).strip()
        elif re.search(r"\b(10[0-6](\.\d+)?)\b", text):
            num_m = re.search(r"\b(10[0-6](\.\d+)?)\b", text)
            state.temperature = f"{num_m.group(0)}°F"

    # 3. Extract Duration
    if not state.duration:
        dur_match = re.search(
            r"\b(fever\s*(for|since)?\s*\d+\s*(days?|weeks?|hours?|hrs?)|"
            r"\d+\s*(days?|weeks?|hours?|hrs?)\s*(of\s+fever|fever)?|"
            r"(fever\s+)?since\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|yesterday))\b",
            text
        )
        if dur_match:
            state.duration = dur_match.group(0).strip()

    # 4. Extract Medication Status
    if not state.medication_status or state.medication_status in ("None", ""):
        no_med_match = re.search(
            r"\b(no\s+(medicines?|meds|medications?)|not\s+taking\s+(any\s+)?(medicines?|meds|medications?|anything)|"
            r"no\s+meds\s+so\s+far|haven't\s+taken\s+any|unmedicated)\b",
            text
        )
        if no_med_match:
            state.medication_status = "No medicines taken"
        else:
            taking_match = re.search(
                r"\b((started\s+taking|taking|took|on)\s+([a-zA-Z0-9\s]+?)(?=\s+(since|this|yesterday|for|,|\.|$)))\b",
                text
            )
            if taking_match and not any(k in taking_match.group(0) for k in ["no meds", "nothing", "not taking"]):
                state.medication_status = taking_match.group(0).strip()

    # 5. Detect High-Risk Conditions & Disclosures
    if re.search(r"\b(cancer|chemo|chemotherapy|leukemia|lymphoma|oncology|oncologist|carcinoma|tumor)\b", text):
        if "Cancer" not in state.disclosed_conditions:
            state.disclosed_conditions.append("Cancer")
        if not state.is_active_cancer_chemo:
            state.newly_disclosed_high_risk = "Active Cancer / Chemotherapy"
        state.is_active_cancer_chemo = True
        if "Active Chemotherapy/Cancer" not in state.conditions:
            state.conditions.append("Active Chemotherapy/Cancer")

    # Check for affirmative stand-down disclosure:
    stand_down_match = re.search(
        r"\b(finished\s+treatment\s+(\d+|several)\s+years?\s+ago|"
        r"in\s+remission\s+(for\s+)?\d+\s+years?|"
        r"no\s+(chemo|treatment)\s+(for\s+)?\d+\s+years?|"
        r"not\s+on\s+anything|cancer\s+free\s+for\s+\d+)\b",
        text
    )
    if stand_down_match:
        state.active_chemo_confirmed = False
        state.stand_down_reason = "Affirmative disclosure: treatment ended >12 months ago with no active therapy"

    # Detect other high-risk conditions
    for cond_type, patterns in PATIENT_CONDITION_PATTERNS.items():
        for p in patterns:
            if re.search(p, text):
                if cond_type == "immunocompromised":
                    if not state.is_immunocompromised:
                        state.newly_disclosed_high_risk = "Immunosuppression"
                    state.is_immunocompromised = True
                    if "Immunocompromised" not in state.conditions:
                        state.conditions.append("Immunocompromised")
                    if "Immunosuppression" not in state.disclosed_conditions:
                        state.disclosed_conditions.append("Immunosuppression")
                elif cond_type == "pregnancy":
                    if not state.is_pregnant:
                        state.newly_disclosed_high_risk = "Pregnancy"
                    state.is_pregnant = True
                    if "Pregnancy" not in state.conditions:
                        state.conditions.append("Pregnancy")
                    if "Pregnancy" not in state.disclosed_conditions:
                        state.disclosed_conditions.append("Pregnancy")
                elif cond_type == "infant":
                    if state.age is not None and state.age_unit in ("months", "days", "weeks"):
                        if (state.age_unit == "months" and state.age < 3) or (state.age_unit == "days" and state.age < 90) or (state.age_unit == "weeks" and state.age < 13):
                            state.is_infant_under_3mo = True
                    if "Infant" not in state.conditions:
                        state.conditions.append("Infant")
                elif cond_type == "chronic_disease":
                    match_obj = re.search(p, text)
                    if match_obj:
                        cond_name = match_obj.group(0).title()
                        if cond_name not in state.conditions:
                            state.conditions.append(cond_name)
                        if cond_name not in state.disclosed_conditions:
                            state.disclosed_conditions.append(cond_name)

    # 6. Detect Symptoms
    for symp_type, pattern in SYMPTOM_PATTERNS.items():
        if re.search(pattern, text):
            if symp_type not in state.current_symptoms:
                state.current_symptoms.append(symp_type)
            if symp_type == "fever" and "fever" not in [s.lower() for s in state.reported_symptoms_detail]:
                state.reported_symptoms_detail.append("fever")
            elif symp_type == "respiratory" and "cough" not in [s.lower() for s in state.reported_symptoms_detail]:
                if "dry cough" in text:
                    state.reported_symptoms_detail.append("dry cough")
                elif "cough" in text:
                    state.reported_symptoms_detail.append("cough")

    return state


# ─────────────────────────────────────────────────────────────
# 4. Triage Evaluation & Red-Flag Override Engine
# ─────────────────────────────────────────────────────────────

def evaluate_triage_tier(state: PatientState, latest_user_msg: str) -> Tuple[str, List[str], Optional[str]]:
    """
    Evaluates patient state against AUDITABLE_TRIAGE_MATRIX with widened safety margins.
    Returns: (RiskTier, RedFlags, PrimaryActionGuidance)
    """
    text = latest_user_msg.lower()
    red_flags = []
    
    # 1. Critical Red-Flag Intersections (Immediate Emergency Tier)
    
    # A. Chemo / Cancer / Immunocompromised + Fever/Infection (WIDENED TRIGGER)
    has_cancer = (
        state.is_active_cancer_chemo
        or any(c.lower() in ("cancer", "active chemotherapy/cancer", "chemo", "chemotherapy") for c in state.disclosed_conditions)
        or any("cancer" in c.lower() or "chemo" in c.lower() for c in state.conditions)
        or bool(re.search(r"\b(cancer|chemotherapy|chemo|leukemia|lymphoma|oncology|oncologist)\b", text))
    )
    has_fever = (
        "fever" in state.current_symptoms
        or any("fever" in s.lower() for s in state.reported_symptoms_detail)
        or bool(state.temperature and any(t in state.temperature for t in ["100", "101", "102", "103", "104", "105", "38", "39", "40"]))
        or bool(re.search(SYMPTOM_PATTERNS["fever"], text))
    )

    if (has_cancer or state.is_immunocompromised) and has_fever:
        # Check for explicit affirmative stand-down disclosure
        if state.active_chemo_confirmed is False:
            state.risk_tier = "Urgent"
            red_flags.append("High Prolonged Fever with Completed Cancer Treatment")
            return "Urgent", red_flags, (
                "⚠️ **URGENT CLINICAL EVALUATION RECOMMENDED**\n\n"
                "You noted a history of cancer with completed treatment and no active therapy. While this reduces the risk of acute febrile neutropenia, "
                "a high fever (103°F) lasting 3 days still requires prompt in-person medical evaluation by a primary care doctor or urgent care clinic.\n\n"
                "**RECOMMENDED ACTIONS:**\n"
                "1. Seek same-day urgent care evaluation to identify the source of infection.\n"
                "2. Maintain adequate hydration and rest.\n"
                "3. If you develop sudden chills, shortness of breath, or confusion, seek emergency care immediately."
            )

        red_flags.append("Febrile Neutropenia Risk (Cancer History + Fever)")
        state.risk_tier = "Emergency"
        return "Emergency", red_flags, (
            "🚨 **CRITICAL MEDICAL EMERGENCY: FEBRILE NEUTROPENIA RISK**\n\n"
            "Because you have mentioned cancer along with an active fever, please **do not self-treat the fever without checking with your oncology team first** — "
            "if you already have a specific, documented plan from them for this situation, follow that; otherwise this combination requires **prompt emergency evaluation** "
            "at the nearest Emergency Department or via your oncologist's 24/7 emergency hotline.\n\n"
            "**SAFETY DIRECTIVES:**\n"
            "1. Contact your oncology emergency hotline or proceed to the nearest Emergency Department immediately. Upon arrival, inform staff immediately that you have cancer and a fever.\n"
            "2. **Do not take over-the-counter antipyretics** (acetaminophen, paracetamol, ibuprofen) or home remedies without specialist clearance, as suppressing the fever can mask life-threatening infection progression.\n\n"
            "*(Guideline Reference: ASCO/IDSA Clinical Practice Guidelines for Antimicrobial Prophylaxis and Outpatient Management of Fever and Neutropenia in Adults Treated for Malignancy)*"
        )

    # B. Infant < 3 months + Fever
    if state.is_infant_under_3mo and ("fever" in state.current_symptoms or re.search(SYMPTOM_PATTERNS["fever"], text)):
        red_flags.append("Neonatal Sepsis Risk (Infant <3 months + Fever >=38.0C/100.4F)")
        state.risk_tier = "Emergency"
        return "Emergency", red_flags, (
            "🚨 **CRITICAL PEDIATRIC EMERGENCY: NEONATAL FEVER EVALUATION REQUIRED**\n"
            "In infants younger than 3 months (<= 90 days), a fever of **38.0°C (100.4°F) or higher requires immediate in-person emergency hospital evaluation**.\n"
            "**ACTION REQUIRED NOW:**\n"
            "1. Take the infant to the nearest Pediatric Emergency Department immediately.\n"
            "2. **DO NOT administer over-the-counter fever medicines (paracetamol/ibuprofen)** before clinical examination, as an urgent medical workup (blood/urine/CSF) is required.\n"
            "*(Guideline Reference: American Academy of Pediatrics (AAP) Clinical Practice Guideline on the Febrile Infant)*"
        )

    # C. Pregnancy + Preeclampsia or Severe Symptoms
    if state.is_pregnant and (
        re.search(SYMPTOM_PATTERNS["preeclampsia_signs"], text)
        or ("neurological" in state.current_symptoms and re.search(r"\b(headache|vision|blurred|swelling)\b", text))
        or ("fever" in state.current_symptoms and re.search(r"\b(high\s+fever|chills|pain)\b", text))
    ):
        red_flags.append("Obstetric High-Risk / Preeclampsia Alert")
        state.risk_tier = "Emergency"
        return "Emergency", red_flags, (
            "🚨 **URGENT OBSTETRIC ALERT: IMMEDIATE CLINICAL EVALUATION REQUIRED**\n"
            "In pregnancy, severe headaches, visual disturbances, or high fever require immediate evaluation to rule out preeclampsia and maternal-fetal complications.\n"
            "**ACTION REQUIRED NOW:** Contact your obstetrician or proceed to Labor & Delivery Triage / Emergency immediately.\n"
            "*(Guideline Reference: ACOG Practice Bulletin on Gestational Hypertension and Preeclampsia)*"
        )

    # D. Acute General Red-Flags (Chest pain, stroke, severe respiratory distress)
    if "cardiac" in state.current_symptoms or re.search(r"\b(crushing\s+chest\s+pain|chest\s+pressure|radiating\s+to\s+arm|stroke|slurred\s+speech|cannot\s+breathe)\b", text):
        red_flags.append("Acute Cardiovascular / Neurological / Airway Emergency")
        state.risk_tier = "Emergency"
        return "Emergency", red_flags, (
            "🚨 **LIFE-THREATENING EMERGENCY: CALL 911 / 112 / 999 IMMEDIATELY**\n"
            "Your symptoms indicate potential acute cardiac, neurological, or severe respiratory distress.\n"
            "**ACTION REQUIRED NOW:** Call emergency services immediately. Do not drive yourself to the hospital."
        )

    # 2. Urgent Tier (High prolonged fever, chronic disease exacerbation)
    if ("asthma" in str(state.conditions).lower() or "copd" in str(state.conditions).lower() or state.is_elderly) and "respiratory" in state.current_symptoms:
        state.risk_tier = "Urgent"
        return "Urgent", red_flags, "Urgent same-day clinical assessment required for chronic respiratory vulnerability."

    if "fever" in state.current_symptoms and (
        re.search(r"\b(3\s+days|4\s+days|5\s+days|103|104|39\.5|40)\b", text)
        or (state.duration and any(d in state.duration for d in ["3 day", "4 day", "5 day"]))
        or (state.temperature and any(t in state.temperature for t in ["103", "104", "105"]))
    ):
        state.risk_tier = "Urgent"
        return "Urgent", red_flags, "Urgent primary care / urgent care visit indicated due to fever duration or high elevation."

    # 3. Routine Tier (Standard acute symptoms)
    if state.current_symptoms:
        state.risk_tier = "Routine"
        return "Routine", red_flags, None

    # 4. Informational Tier (Educational/General)
    state.risk_tier = "Informational"
    return "Informational", red_flags, None


# ─────────────────────────────────────────────────────────────
# 5. Medication & Dosing Safety Filter
# ─────────────────────────────────────────────────────────────

def check_medication_contraindications(state: PatientState, user_msg: str) -> Tuple[bool, Optional[str]]:
    """
    Blocks exact drug dosing and dangerous medication recommendations:
    - Automatically blocks specific drug recommendations if user has active cancer / chemo / immunosuppression / pregnancy
    - Blocks specific numerical drug dosing for all users
    """
    text = user_msg.lower()
    is_dosing_request = any(re.search(p, text) for p in DOSING_REQUEST_PATTERNS)
    is_drug_inquiry = bool(re.search(r"\b(can\s+i\s+take|what\s+medicine|give\s+me|prescribe|which\s+(medicine|drug|pill)|take\s+(paracetamol|tylenol|ibuprofen|advil|aspirin|dolo|cough\s+syrup))\b", text))

    is_high_risk = (
        state.is_active_cancer_chemo
        or any("cancer" in c.lower() or "chemo" in c.lower() for c in state.disclosed_conditions)
        or state.is_immunocompromised
        or state.is_pregnant
        or (state.age is not None and state.age < 12)
        or state.is_infant_under_3mo
    )

    # Automatic block for high-risk populations whenever medication is raised or requested
    if is_high_risk and (is_dosing_request or is_drug_inquiry or "medicine" in text or "medication" in text or "drug" in text):
        if state.is_active_cancer_chemo or any("cancer" in c.lower() for c in state.disclosed_conditions) or state.is_immunocompromised:
            return True, (
                "⚠️ **MEDICATION SAFETY RESTRICTION**: Because you have disclosed cancer or immunosuppression, "
                "taking medications (including over-the-counter fever reducers or cough suppressants) can mask serious infection or interact with oncology therapies. "
                "Specific drugs and dosages cannot be recommended by AI. Please consult your oncology team, nurse hotline, or pharmacist before taking any medication."
            )
        if state.is_pregnant:
            return True, (
                "⚠️ **MEDICATION SAFETY RESTRICTION (PREGNANCY)**: Many medications carry risks during pregnancy. "
                "Specific medications and dosages cannot be recommended by AI. Please consult your obstetrician or pharmacist."
            )
        if (state.age is not None and state.age < 12) or state.is_infant_under_3mo:
            return True, (
                "⚠️ **PEDIATRIC MEDICATION RESTRICTION**: Pediatric medication dosing is strictly weight-based (mg/kg) and must be "
                "prescribed by a pediatrician. Exact drug dosages cannot be calculated safely here."
            )

    # General block on numerical dosing for all users
    if is_dosing_request:
        return True, (
            "⚠️ **CLINICAL DOSING RESTRICTION**: As a safety protocol, MediAssist does not provide specific drug dosages or numerical dosing schedules. "
            "Safe medication dosing requires an evaluation of your full health history, body weight, kidney and liver function, and existing medications. "
            "Please follow the manufacturer packaging instructions or speak with a licensed pharmacist or physician."
        )

    return False, None


# ─────────────────────────────────────────────────────────────
# 6. Mid-Conversation Disclosure Re-Evaluation
# ─────────────────────────────────────────────────────────────

def check_mid_conversation_correction(state: PatientState, history_text: str) -> Optional[str]:
    """
    If user revealed a critical high-risk factor mid-conversation (e.g. chemotherapy or pregnancy),
    generate a clear correction alert to override any previous routine advice.
    """
    if not state.newly_disclosed_high_risk:
        return None

    factor = state.newly_disclosed_high_risk
    # Clear flag after generating alert
    state.newly_disclosed_high_risk = None
    state.needs_prior_advice_correction = False

    return (
        f"🚨 **CLINICAL RE-EVALUATION ALERT ({factor.upper()} DISCLOSED)** 🚨\n\n"
        f"You have noted a critical medical factor: **{factor}**.\n"
        f"**IMPORTANT CORRECTION**: Any prior routine self-care or home monitoring advice given earlier in this chat is now **SUPERSEDED**.\n"
        f"In {factor}, standard symptoms carry significantly elevated clinical risks (such as infection escalation or medication contraindications). "
        f"Please prioritize immediate evaluation by your specialist or emergency provider."
    )
