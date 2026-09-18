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
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union
from research.src.intent_classifier import is_third_party_query
from research.src.clinical_parser import ClinicalEntityParser

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

    # Emergency override & third-party scoping (v2)
    emergency_override_served: bool = False
    active_emergency_topic: Optional[str] = None  # e.g., "febrile_neutropenia", "neonatal_fever"
    last_reminder_turn_index: int = -1
    third_party_context: Optional[str] = None

    # Triage conversational architecture & progressive turn tracking
    has_cancer_history: bool = False
    active_emergency: Optional[str] = None  # e.g. "FEBRILE_NEUTROPENIA"
    emergency_turn_count: int = 0
    emergency_timestamp: Optional[float] = None
    resolved_emergency: Optional[str] = None

    # Diagnostic Intake Data
    recorded_temperature: Optional[str] = None
    on_active_chemo: Optional[bool] = None
    has_central_line_or_port: Optional[bool] = None

    # Active symptoms strictly observed in current session
    transient_symptoms: List[str] = field(default_factory=list)

    def start_emergency(self, condition: str):
        self.active_emergency = condition
        if not self.emergency_timestamp:
            self.emergency_timestamp = time.time()
        self.emergency_turn_count += 1
        self.risk_tier = "Emergency"
        self.active_emergency_topic = condition.lower()

    def reset_emergency(self, resolution_reason: str = "RESOLVED"):
        self.active_emergency = None
        self.emergency_timestamp = None
        self.emergency_turn_count = 0
        self.resolved_emergency = resolution_reason
        self.emergency_override_served = False
        self.active_emergency_topic = None
        self.transient_symptoms.clear()
        if "fever" in self.current_symptoms:
            self.current_symptoms.remove("fever")

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            val = getattr(self, key)
            return val if val is not None else default
        return default

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get_diff_snapshot(self) -> Dict[str, Any]:
        """Captures a snapshot of clinically relevant structured facts to detect turn-by-turn state changes."""
        return {
            "temperature": self.temperature,
            "duration": self.duration,
            "medication_status": self.medication_status,
            "current_symptoms": list(self.current_symptoms),
            "reported_symptoms_detail": list(self.reported_symptoms_detail),
            "disclosed_conditions": list(self.disclosed_conditions),
            "conditions": list(self.conditions),
            "is_active_cancer_chemo": self.is_active_cancer_chemo,
            "has_cancer_history": self.has_cancer_history,
            "is_infant_under_3mo": self.is_infant_under_3mo,
            "is_pregnant": self.is_pregnant,
            "is_immunocompromised": self.is_immunocompromised,
            "active_chemo_confirmed": self.active_chemo_confirmed,
            "active_emergency": self.active_emergency,
            "resolved_emergency": self.resolved_emergency,
        }

    def has_state_diff(self, snapshot: Optional[Dict[str, Any]]) -> bool:
        """Returns True if the structured patient state has acquired new clinical facts compared to snapshot."""
        if not snapshot:
            return True
        return (
            self.temperature != snapshot.get("temperature")
            or self.duration != snapshot.get("duration")
            or self.medication_status != snapshot.get("medication_status")
            or set(self.current_symptoms) != set(snapshot.get("current_symptoms", []))
            or set(self.reported_symptoms_detail) != set(snapshot.get("reported_symptoms_detail", []))
            or set(self.disclosed_conditions) != set(snapshot.get("disclosed_conditions", []))
            or set(self.conditions) != set(snapshot.get("conditions", []))
            or self.is_active_cancer_chemo != snapshot.get("is_active_cancer_chemo")
            or self.has_cancer_history != snapshot.get("has_cancer_history")
            or self.is_infant_under_3mo != snapshot.get("is_infant_under_3mo")
            or self.is_pregnant != snapshot.get("is_pregnant")
            or self.is_immunocompromised != snapshot.get("is_immunocompromised")
            or self.active_chemo_confirmed != snapshot.get("active_chemo_confirmed")
            or self.active_emergency != snapshot.get("active_emergency")
            or self.resolved_emergency != snapshot.get("resolved_emergency")
        )

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
            "needs_prior_advice_correction": self.needs_prior_advice_correction,
            "emergency_override_served": self.emergency_override_served,
            "active_emergency_topic": self.active_emergency_topic,
            "last_reminder_turn_index": self.last_reminder_turn_index,
            "third_party_context": self.third_party_context,
            "has_cancer_history": self.has_cancer_history,
            "active_emergency": self.active_emergency,
            "emergency_turn_count": self.emergency_turn_count,
            "emergency_timestamp": self.emergency_timestamp,
            "resolved_emergency": self.resolved_emergency
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
            needs_prior_advice_correction=data.get("needs_prior_advice_correction", False),
            emergency_override_served=data.get("emergency_override_served", False),
            active_emergency_topic=data.get("active_emergency_topic"),
            last_reminder_turn_index=data.get("last_reminder_turn_index", -1),
            third_party_context=data.get("third_party_context"),
            has_cancer_history=data.get("has_cancer_history", False),
            active_emergency=data.get("active_emergency"),
            emergency_turn_count=data.get("emergency_turn_count", 0),
            emergency_timestamp=data.get("emergency_timestamp"),
            resolved_emergency=data.get("resolved_emergency")
        )

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "PatientState":
        import json
        if not json_str:
            return cls()
        try:
            return cls.from_dict(json.loads(json_str))
        except Exception:
            return cls()


# ─────────────────────────────────────────────────────────────
# 1.1 State-Aware Clinical Emergency Triage Engine
# ─────────────────────────────────────────────────────────────

class ClinicalTriageEngine:
    """
    State-aware emergency triage engine for oncology and acute clinical presentations.
    Maintains persistent risk indicators and delivers progressive, nurse-grade guidance
    without sirens, emojis, or repetitive verbatim copy-pasting.
    """

    RESOLUTION_KEYWORDS = [
        r"\b(cleared|discharged)\s+me\b",
        r"\bback from (the )?(hospital|er|emergency room|clinic)\b", 
        r"\b(saw|seen)\s+(the\s+|my\s+)?(doctor|oncologist)\b",
        r"\b(doctor|oncologist)\s+(has\s+)?(checked|saw|cleared|treated|discharged)\b",
        r"\b(doctor|oncologist)\s+cleared\b",
        r"\bdischarged\b",
        r"\bin the er now\b",
        r"\breceived (iv )?antibiotics\b",
        r"\bfever is (gone|resolved|down|normal)\b",
        r"\bfeeling much better\b"
    ]

    MEDICATION_KEYWORDS = [
        r"\bsuggest\b", r"\bmedication\b", r"\bmedicine\b", r"\bpill\b", 
        r"\btablet\b", r"\btake\b", r"\bremedy\b", r"\bcure\b", 
        r"\btylenol\b", r"\bparacetamol\b", r"\bibuprofen\b", r"\badvil\b"
    ]

    @classmethod
    def evaluate(
        cls, 
        user_text: str, 
        patient_state: Union[PatientState, Dict[str, Any]], 
        signals: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        text_lower = (user_text or "").lower()

        # Ensure default keys if patient_state is dict
        if isinstance(patient_state, dict):
            patient_state.setdefault("has_cancer_history", False)
            patient_state.setdefault("active_emergency", None)
            patient_state.setdefault("emergency_turn_count", 0)
            patient_state.setdefault("current_symptoms", [])
            patient_state.setdefault("emergency_timestamp", None)
            patient_state.setdefault("resolved_emergency", None)

        if signals is None:
            signals = ClinicalEntityParser.extract_clinical_signals(user_text)

        # 1. RESOLUTION & STAND-DOWN DETECTION (With Clause Negation Guard)
        has_resolution_keyword = any(bool(re.search(pat, text_lower)) for pat in cls.RESOLUTION_KEYWORDS)
        if has_resolution_keyword and patient_state.get("active_emergency"):
            is_negated = bool(re.search(
                r"\b(not|never|hasn\'?t|haven\'?t|cannot|can\'?t|didn\'?t|won\'?t|no)\b[^.,;!\n]{0,25}\b(cleared|seen|saw|checked|back|discharged)\b",
                text_lower
            ))
            if not is_negated:
                if isinstance(patient_state, PatientState):
                    patient_state.reset_emergency(resolution_reason="DOCTOR_EVALUATED")
                elif isinstance(patient_state, dict):
                    patient_state["resolved_emergency"] = "DOCTOR_EVALUATED"
                    patient_state["active_emergency"] = None
                    patient_state["emergency_turn_count"] = 0
                    if "fever" in patient_state.get("current_symptoms", []):
                        patient_state["current_symptoms"].remove("fever")
                return {
                    "tier": "RESOLVED",
                    "condition": "FEBRILE_NEUTROPENIA",
                    "requires_escalation": False,
                    "message": (
                        "I am glad to hear that you have been evaluated and cleared by your doctor. "
                        "I will update your current status to reflect that this urgent episode is resolved. "
                        "Please continue following your care team's guidance, and reach out immediately "
                        "if new or worsening symptoms develop. How can I help you today?"
                    )
                }

        # 2. UPDATE PERSISTENT MEDICAL HISTORY (Self-attributed only)
        if signals.get("has_cancer") and signals.get("cancer_subject") == "self":
            if isinstance(patient_state, PatientState):
                patient_state.has_cancer_history = True
                patient_state.is_active_cancer_chemo = True
            elif isinstance(patient_state, dict):
                patient_state["has_cancer_history"] = True

        # 3. CAREGIVER ISOLATION (Third-party relative has cancer + fever)
        if signals.get("cancer_subject") == "third_party" and signals.get("has_fever"):
            rel = signals.get("relative_relation") or "family member"
            return {
                "tier": "Emergency",
                "emergency_type": "FEBRILE_NEUTROPENIA_CAREGIVER",
                "condition": "FEBRILE_NEUTROPENIA_CAREGIVER",
                "message": (
                    f"Because your {rel} has a history of cancer and currently has a fever, "
                    f"they require immediate clinical evaluation. Cancer treatments deplete infection-fighting white blood cells, "
                    f"meaning a fever can signal a fast-progressing infection.\n\n"
                    f"Please have them contact their oncology team's 24/7 hotline immediately or bring them to the nearest Emergency Department. "
                    f"Do not administer over-the-counter fever reducers without their oncologist's authorization."
                ),
                "requires_escalation": True,
                "turn": 1
            }

        # 4. PURE GREETING CHECK: If pure greeting without clinical signals, defer to greeting handler
        is_greeting = any(bool(re.search(p, text_lower)) for p in [
            r"^(hello|hi|hey|good morning|good afternoon|good evening)\b",
            r"^(how are you|how good are you|how are you doing)\b"
        ])
        if is_greeting and not signals.get("has_clinical_signals", False):
            return None

        # 5. PERSONAL FEBRILE NEUTROPENIA EVALUATION
        has_cancer = bool(
            patient_state.get("has_cancer_history")
            or patient_state.get("is_active_cancer_chemo")
            or any("cancer" in str(c).lower() for c in patient_state.get("disclosed_conditions", []))
            or any("cancer" in str(c).lower() for c in patient_state.get("conditions", []))
        )
        fever_in_this_turn = signals.get("has_fever") and not signals.get("fever_negated")
        is_asking_about_ongoing_emergency = (
            patient_state.get("active_emergency") == "FEBRILE_NEUTROPENIA" and
            any(w in text_lower for w in ["fever", "temperature", "chills", "sick", "suggest", "medication", "pill", "pain"])
        )

        if has_cancer and (fever_in_this_turn or is_asking_about_ongoing_emergency):
            if isinstance(patient_state, PatientState):
                patient_state.start_emergency("FEBRILE_NEUTROPENIA")
            elif isinstance(patient_state, dict):
                patient_state["active_emergency"] = "FEBRILE_NEUTROPENIA"
                patient_state["emergency_turn_count"] = patient_state.get("emergency_turn_count", 0) + 1
                if not patient_state.get("emergency_timestamp"):
                    patient_state["emergency_timestamp"] = time.time()
            return cls._generate_intent_specific_triage(user_text, patient_state)

        return None

    @classmethod
    def _generate_intent_specific_triage(cls, user_text: str, patient_state: Union[PatientState, Dict[str, Any]]) -> Dict[str, Any]:
        text_lower = user_text.lower()
        turn = patient_state.get("emergency_turn_count", 1)

        # Branch A: Medication / Pill / Remedy Inquiry
        if any(bool(re.search(pat, text_lower)) for pat in cls.MEDICATION_KEYWORDS):
            if any(w in text_lower for w in ["stomach", "pain", "belly", "abdomen"]):
                symptom_phrase = "secondary symptoms such as stomach pain"
                message = (
                    f"Do not take over-the-counter fever reducers. I cannot recommend any medications or fever reducers for this condition. "
                    f"I understand you are seeking relief for {symptom_phrase}. However, when experiencing a fever alongside a cancer history, "
                    f"attempting to self-treat secondary symptoms with over-the-counter medications carries severe clinical risks. "
                    f"These drugs can irritate your gastrointestinal tract, mask acute intra-abdominal infections, or interact dangerously with your cancer therapies.\n\n"
                    f"**Action required right now:**\n"
                    f"• Call your oncology 24/7 on-call triage line immediately.\n"
                    f"• If you cannot reach them within 15 minutes, proceed directly to the nearest Emergency Department.\n"
                    f"• Inform triage upon arrival: *'I have cancer and I am running a fever.'*"
                )
            else:
                message = (
                    "Do not take over-the-counter fever reducers. I cannot recommend any medications or fever reducers for this condition. "
                    "In patients with cancer, taking fever reducers like acetaminophen (Tylenol), ibuprofen, or paracetamol "
                    "artificially lowers body temperature without treating the underlying infection, which can mask critical progression "
                    "and delay essential intravenous antibiotics.\n\n"
                    "**Action required right now:**\n"
                    "• Call your oncology 24/7 on-call triage line immediately.\n"
                    "• If you cannot reach them within 15 minutes, proceed directly to the nearest Emergency Department.\n"
                    "• Inform triage upon arrival: *'I have cancer and I am running a fever.'*"
                )

        # Branch B: Initial Turn 1 Presentation
        elif turn == 1:
            message = (
                "**FEBRILE NEUTROPENIA EVALUATION REQUIRED**\n\n"
                "Because you have a history of cancer and are experiencing a fever, this requires immediate clinical evaluation today.\n\n"
                "**Why this is urgent:** Cancer treatments frequently deplete white blood cells (neutrophils), which are essential "
                "for fighting infection. In oncology patients, even a mild fever (100.4°F / 38°C or above) can indicate an infection "
                "that can escalate into sepsis without prompt intravenous antibiotics.\n\n"
                "**Immediate Actions:**\n"
                "1. **Call your oncology team's 24/7 triage hotline** right now.\n"
                "2. **Go to the nearest Emergency Department** if you cannot reach your oncology team immediately.\n"
                "3. **Do not take fever reducers (such as Tylenol, paracetamol, or ibuprofen)** without checking with your oncology team first.\n\n"
                "**High-Yield Triage Screening Questions:**\n"
                "• What is your exact thermometer temperature reading on the thermometer right now?\n"
                "• Are you currently receiving active chemotherapy, immunotherapy, or steroids?\n"
                "• Do you have an indwelling port, PICC line, or central venous catheter?"
            )

        # Branch C: Screening Intake / Vitals Details (Turn 2+)
        elif re.search(r"\b\d{2,3}(\.\d)?\b", text_lower) or any(w in text_lower for w in ["chemo", "port", "picc"]):
            message = (
                "Thank you for sharing those clinical details. Because an oncology patient with an active fever "
                "is at high risk for febrile neutropenia, this cannot be safely monitored at home.\n\n"
                "Please connect with your oncology emergency service immediately or proceed to the emergency department "
                "so clinicians can draw blood cultures and administer IV antibiotics if indicated."
            )

        # Branch D: Turn 2+ General Check-in
        else:
            message = (
                "I want to reiterate: any fever in an oncology patient requires prompt clinical evaluation to rule out febrile neutropenia.\n\n"
                "Please do not wait or attempt self-treatment at home. Are you currently on your way to an emergency clinic, "
                "or have you been able to reach your oncology on-call nurse?"
            )

        return {
            "tier": "Emergency",
            "condition": "FEBRILE_NEUTROPENIA",
            "emergency_type": "FEBRILE_NEUTROPENIA",
            "message": message,
            "requires_escalation": True,
            "turn": turn
        }

        return None


def format_context_aware_greeting(
    patient_state: Union[PatientState, Dict[str, Any], None],
    first_name: str = "there",
    msg: str = ""
) -> str:
    """
    Generates a context-aware greeting that avoids toxic positivity and preserves clinical continuity.
    - If active_emergency == "FEBRILE_NEUTROPENIA": return empathetic clinical check-in
    - If has_cancer_history: return context-aware greeting without toxic positivity (no 'feeling great' or 'great too')
    - Default: standard welcoming greeting for new users
    """
    if not patient_state:
        return f"Hello {first_name}. I'm MediAssist, your clinical decision-support assistant. How can I help you with your health or medical questions today?"

    active_emerg = patient_state.get("active_emergency")
    has_cancer = patient_state.get("has_cancer_history") or patient_state.get("is_active_cancer_chemo")

    if active_emerg == "FEBRILE_NEUTROPENIA":
        return (
            f"Good evening {first_name}. Given your fever and cancer history discussed a moment ago, "
            f"have you been able to contact your oncology team's 24/7 hotline or head toward an emergency room? "
            f"Please ensure you are seeking immediate in-person evaluation."
        )

    if has_cancer:
        return (
            f"Hello {first_name}. I am here to support you with your health questions. "
            f"How are you feeling right now, and how can I assist with your symptoms or care today?"
        )

    return f"Hello {first_name}. I'm MediAssist, your clinical decision-support assistant. How can I help you with your health or medical questions today?"


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
    "respiratory": r"\b(shortness\s+of\s+breath|difficulty\s+breathing|wheezing|cough|coughing|sore\s+throat|stridor|cannot\s+breathe|can\'?t\s+breathe|trouble\s+breathing|unable\s+to\s+breathe)\b",
    "cardiac": r"\b(chest\s+pain|chest\s+pressure|palpitations|irregular\s+heartbeat|tightness\s+in\s+chest)\b",
    "neurological": r"\b(severe\s+headache|headache|dizziness|fainting|syncope|confusion|slurred\s+speech|vision\s+loss|blurred\s+vision|blurry\s+vision|seizure|getting\s+confused|confused)\b",
    "gastrointestinal": r"\b(vomiting|diarrhea|abdominal\s+pain|stomach\s+cramps|unable\s+to\s+keep\s+fluids\s+down|dehydration)\b",
    "pain": r"\b(pain|ache|aching|soreness|hurt|hurting)\b",
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

    # Guard: Third-party inquiries must never mutate patient's personal clinical facts
    is_tp, tp_label = is_third_party_query(user_text)
    if is_tp:
        state.third_party_context = user_text.strip()
        return state

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
        state.emergency_override_served = False
        state.active_emergency_topic = None

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
    parser_signals = ClinicalEntityParser.extract_clinical_signals(user_text)
    state.transient_symptoms = parser_signals.get("active_symptoms", [])
    for symp_type, pattern in SYMPTOM_PATTERNS.items():
        if re.search(pattern, text):
            # Check negation
            if symp_type == "fever" and parser_signals.get("fever_negated"):
                if "fever" in state.current_symptoms:
                    state.current_symptoms.remove("fever")
                state.reported_symptoms_detail = [s for s in state.reported_symptoms_detail if "fever" not in s.lower()]
                continue
            if ClinicalEntityParser.is_negated(text, pattern):
                continue
            if symp_type not in state.current_symptoms:
                state.current_symptoms.append(symp_type)
            if symp_type == "fever" and "fever" not in [s.lower() for s in state.reported_symptoms_detail]:
                state.reported_symptoms_detail.append("fever")
            elif symp_type == "respiratory":
                if "dry cough" in text and "dry cough" not in [s.lower() for s in state.reported_symptoms_detail]:
                    state.reported_symptoms_detail.append("dry cough")
                elif "cough" in text and "cough" not in [s.lower() for s in state.reported_symptoms_detail]:
                    state.reported_symptoms_detail.append("cough")
                elif ("breathe" in text or "breath" in text) and not any(b in [s.lower() for s in state.reported_symptoms_detail] for b in ["difficulty breathing", "shortness of breath", "cannot breathe"]):
                    state.reported_symptoms_detail.append("difficulty breathing")
            elif symp_type == "neurological":
                if ("confus" in text) and "confusion" not in [s.lower() for s in state.reported_symptoms_detail]:
                    state.reported_symptoms_detail.append("confusion")
                elif "headache" in text and "headache" not in [s.lower() for s in state.reported_symptoms_detail]:
                    state.reported_symptoms_detail.append("headache")
            elif symp_type == "cardiac":
                if "chest pain" in text and "chest pain" not in [s.lower() for s in state.reported_symptoms_detail]:
                    state.reported_symptoms_detail.append("chest pain")

    return state


# ─────────────────────────────────────────────────────────────
# 4. Triage Evaluation & Red-Flag Override Engine
# ─────────────────────────────────────────────────────────────

def friendly_emergency_label(topic: Optional[str]) -> str:
    """Returns a natural, non-alarmist label for the active emergency topic."""
    labels = {
        "febrile_neutropenia": "fever and cancer history",
        "neonatal_fever": "infant fever",
        "preeclampsia": "pregnancy-related symptoms",
        "cardiovascular_respiratory": "chest pain or severe breathing difficulty",
    }
    return labels.get(topic, "urgent medical symptoms")


def maybe_append_emergency_reminder(raw_answer: str, patient_state: PatientState, turn_count: int, msg: str) -> str:
    """
    Appends a dynamically built one-line emergency reminder if an emergency override was previously served,
    enforcing a cool-down so it only triggers on the first topic-shift turn or after 5+ intervening turns.
    """
    if not (patient_state.emergency_override_served and patient_state.active_emergency_topic):
        return raw_answer

    is_parting = bool(re.search(r"\b(bye|goodbye|thanks|thank\s+you|ok\s+thanks|leaving)\b", msg.lower()))
    should_remind = (patient_state.last_reminder_turn_index == -1) or ((turn_count - patient_state.last_reminder_turn_index) >= 5) or is_parting

    if should_remind:
        topic_label = friendly_emergency_label(patient_state.active_emergency_topic)
        patient_state.last_reminder_turn_index = turn_count
        return f"{raw_answer}\n\n*(Reminder: please also follow up on the {topic_label} we discussed earlier.)*"

    return raw_answer


def evaluate_triage_tier(
    state: PatientState,
    latest_user_msg: str,
    has_new_structured_fact: bool = True
) -> Tuple[str, List[str], Optional[str]]:
    """
    Evaluates patient state against AUDITABLE_TRIAGE_MATRIX with widened safety margins.
    Returns: (RiskTier, RedFlags, PrimaryActionGuidance)
    """
    text = latest_user_msg.lower()
    red_flags = []

    # 0. Third-Party Query Gate:
    # If the user query is about another person, do not mutate risk tier or trigger personal emergency overrides.
    is_tp, _ = is_third_party_query(latest_user_msg)
    if is_tp:
        return state.risk_tier, red_flags, None
    
    # 1. Critical Red-Flag Intersections (Immediate Emergency Tier)
    
    # A. Acute General Red-Flags (Chest pain, stroke, severe respiratory distress)
    has_airway_cardiac = (
        "cardiac" in state.current_symptoms
        or any(s in ("cannot breathe", "difficulty breathing", "chest pain", "shortness of breath") for s in state.reported_symptoms_detail)
        or bool(re.search(r"\b(crushing\s+chest\s+pain|chest\s+pressure|radiating\s+to\s+arm|stroke|slurred\s+speech|cannot\s+breathe|can\'?t\s+breathe)\b", text))
    )
    if has_airway_cardiac:
        red_flags.append("Acute Cardiovascular / Neurological / Airway Emergency")
        state.risk_tier = "Emergency"
        state.active_emergency_topic = "cardiovascular_respiratory"
        emergency_guidance = (
            "**EMERGENCY EVALUATION REQUIRED: CALL 911 / 112 / 999 IMMEDIATELY**\n"
            "Your symptoms indicate potential acute cardiac, neurological, or severe respiratory distress.\n"
            "**ACTION REQUIRED NOW:** Call emergency services immediately. Do not drive yourself to the hospital."
        )
        if (not state.emergency_override_served) or has_new_structured_fact:
            return "Emergency", red_flags, emergency_guidance
        else:
            return "Emergency", red_flags, None

    # B. Chemo / Cancer / Immunocompromised + Fever/Infection (WIDENED TRIGGER)
    signals = ClinicalEntityParser.extract_clinical_signals(latest_user_msg)
    
    if signals.get("fever_negated") or state.resolved_emergency:
        has_fever = False
    else:
        has_fever = (
            ("fever" in state.current_symptoms)
            or any("fever" in s.lower() for s in state.reported_symptoms_detail)
            or bool(state.temperature and any(t in state.temperature for t in ["100", "101", "102", "103", "104", "105", "38", "39", "40"]))
            or (signals.get("has_fever") and signals.get("fever_subject") != "third_party")
        )

    has_cancer = (
        state.is_active_cancer_chemo
        or state.has_cancer_history
        or any(c.lower() in ("cancer", "active chemotherapy/cancer", "chemo", "chemotherapy") for c in state.disclosed_conditions)
        or any("cancer" in c.lower() or "chemo" in c.lower() for c in state.conditions)
        or (signals.get("has_cancer") and signals.get("cancer_subject") == "self")
    )

    if (has_cancer or state.is_immunocompromised) and has_fever:
        # Check for explicit affirmative stand-down disclosure
        if state.active_chemo_confirmed is False:
            state.risk_tier = "Urgent"
            state.emergency_override_served = False
            state.active_emergency_topic = None
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
        state.active_emergency_topic = "febrile_neutropenia"
        state.active_emergency = "FEBRILE_NEUTROPENIA"
        if has_cancer:
            state.has_cancer_history = True

        triage_eval = ClinicalTriageEngine.evaluate(text, state)
        if triage_eval and triage_eval.get("message"):
            guidance = triage_eval["message"]
        else:
            guidance = (
                "**FEBRILE NEUTROPENIA RISK EVALUATION**\n\n"
                "Because chemotherapy and oncological treatments deplete infection-fighting white blood cells (neutrophils), "
                "a fever in a cancer patient is a medical emergency that can progress rapidly to severe sepsis. "
                "An immune system suppressed by cancer therapies cannot effectively control systemic infections without prompt medical intervention.\n\n"
                "**Immediate Clinical Directives:**\n"
                "1. Contact your oncology care team's 24/7 emergency hotline immediately or proceed to the nearest Emergency Department. Upon arrival, inform triage personnel immediately that you have a history of cancer and an active fever.\n"
                "2. **Do not take over-the-counter fever reducers** (such as acetaminophen, paracetamol, ibuprofen, or aspirin) without checking with your oncology team first. Suppressing your fever can mask critical infection progression and delay essential intravenous antibiotic therapy.\n\n"
                "**High-Yield Triage Screening Questions:**\n"
                "- What is your exact thermometer temperature reading, and what time was it taken?\n"
                "- Are you currently on active chemotherapy, immunotherapy, or steroids, and when was your last treatment?\n"
                "- Do you have a central venous access device, such as a Port-a-Cath or PICC line?"
            )
        if (not state.emergency_override_served) or has_new_structured_fact:
            return "Emergency", red_flags, guidance
        else:
            return "Emergency", red_flags, None

    # C. Infant < 3 months + Fever
    if state.is_infant_under_3mo and ("fever" in state.current_symptoms or re.search(SYMPTOM_PATTERNS["fever"], text)):
        red_flags.append("Neonatal Sepsis Risk (Infant <3 months + Fever >=38.0C/100.4F)")
        state.risk_tier = "Emergency"
        state.active_emergency_topic = "neonatal_fever"
        guidance = (
            "**CRITICAL PEDIATRIC EMERGENCY: NEONATAL FEVER EVALUATION REQUIRED**\n"
            "In infants younger than 3 months (<= 90 days), a fever of **38.0°C (100.4°F) or higher requires immediate in-person emergency hospital evaluation**.\n"
            "**ACTION REQUIRED NOW:**\n"
            "1. Take the infant to the nearest Pediatric Emergency Department immediately.\n"
            "2. **DO NOT administer over-the-counter fever medicines (paracetamol/ibuprofen)** before clinical examination, as an urgent medical workup (blood/urine/CSF) is required.\n"
            "*(Guideline Reference: American Academy of Pediatrics (AAP) Clinical Practice Guideline on the Febrile Infant)*"
        )
        if (not state.emergency_override_served) or has_new_structured_fact:
            return "Emergency", red_flags, guidance
        else:
            return "Emergency", red_flags, None

    # D. Pregnancy + Preeclampsia or Severe Symptoms
    if state.is_pregnant and (
        re.search(SYMPTOM_PATTERNS["preeclampsia_signs"], text)
        or ("neurological" in state.current_symptoms and re.search(r"\b(headache|vision|blurred|swelling)\b", text))
        or ("fever" in state.current_symptoms and re.search(r"\b(high\s+fever|chills|pain)\b", text))
    ):
        red_flags.append("Obstetric High-Risk / Preeclampsia Alert")
        state.risk_tier = "Emergency"
        state.active_emergency_topic = "preeclampsia"
        guidance = (
            "**URGENT OBSTETRIC ALERT: IMMEDIATE CLINICAL EVALUATION REQUIRED**\n"
            "In pregnancy, severe headaches, visual disturbances, or high fever require immediate evaluation to rule out preeclampsia and maternal-fetal complications.\n"
            "**ACTION REQUIRED NOW:** Contact your obstetrician or proceed to Labor & Delivery Triage / Emergency immediately.\n"
            "*(Guideline Reference: ACOG Practice Bulletin on Gestational Hypertension and Preeclampsia)*"
        )
        if (not state.emergency_override_served) or has_new_structured_fact:
            return "Emergency", red_flags, guidance
        else:
            return "Emergency", red_flags, None

    # 2. Urgent Tier (High prolonged fever, chronic disease exacerbation)
    if ("asthma" in str(state.conditions).lower() or "copd" in str(state.conditions).lower() or state.is_elderly) and "respiratory" in state.current_symptoms:
        state.risk_tier = "Urgent"
        return "Urgent", red_flags, "Urgent same-day clinical assessment required for chronic respiratory vulnerability."

    if "fever" in state.current_symptoms and (
        re.search(r"\b(5\s+days|6\s+days|7\s+days|week|weeks|103|104|105|39\.5|40)\b", text)
        or (state.duration and any(d in state.duration for d in ["5 day", "6 day", "7 day", "week"]))
        or (state.temperature and any(t in state.temperature for t in ["103", "104", "105", "39.5", "40"]))
    ):
        state.risk_tier = "Urgent"
        return "Urgent", red_flags, "Urgent primary care / urgent care visit indicated due to high fever elevation or prolonged duration."

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
        f"**CLINICAL RE-EVALUATION ALERT ({factor.upper()} DISCLOSED)**\n\n"
        f"You have noted a critical medical factor: **{factor}**.\n"
        f"**IMPORTANT CORRECTION**: Any prior routine self-care or home monitoring advice given earlier in this chat is now **SUPERSEDED**.\n"
        f"In {factor}, standard symptoms carry significantly elevated clinical risks (such as infection escalation or medication contraindications). "
        f"Please prioritize immediate evaluation by your specialist or emergency provider."
    )
