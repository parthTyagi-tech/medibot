"""
Medical Chatbot Guardrails & Safety Pipeline.
Includes:
- Prompt Injection & Jailbreak detection
- Medical Emergency detection & immediate life-saving protocol
- Content Safety & dangerous request interception
- Response validation & medical disclaimers
"""

import re
from typing import Tuple, Optional, Any, Dict

# ─────────────────────────────────────────────────────────────
# 1. Prompt Injection & Jailbreak Defense
# ─────────────────────────────────────────────────────────────

PROMPT_INJECTION_PATTERNS = [
    # Direct instruction overrides
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules|commands)",
    r"disregard\s+(all\s+)?(previous|prior|above|existing)\s+(instructions|prompts|rules)",
    r"forget\s+(all\s+)?(previous|prior|everything|instructions)",
    r"system\s+(prompt|override|command|message)",
    r"new\s+(instruction|persona|system|directive)",
    r"bypass\s+(safety|rules|restrictions|filters|guidelines)",
    r"do\s+not\s+follow\s+(your\s+)?(rules|instructions|guidelines)",
    r"you\s+are\s+no\s+longer\s+(a\s+)?(medical|assistant|mediassist)",
    r"you\s+are\s+now\s+(in\s+)?(unrestricted|dan|developer|god|evil)\s+mode",
    
    # DAN / Jailbreak signatures
    r"\bdan\s+mode\b",
    r"\bdo\s+anything\s+now\b",
    r"\bdeveloper\s+mode\s+(v\d+|enabled|active|on)\b",
    r"\bjailbreak\b",
    r"\bopposite\s+mode\b",
    r"\bchaos\s+mode\b",
    
    # Information disclosure / System prompt extraction
    r"(reveal|show|print|display|tell\s+me|output)\s+(your\s+)?(system\s+prompt|initial\s+prompt|hidden\s+instructions|base\s+prompt)",
    r"what\s+are\s+your\s+(exact\s+)?(instructions|rules|system\s+directives)",
    
    # Delimiter / Tag injections
    r"<\s*/?\s*(system_prompt|user_query|context|memory|instruction|prompt)\s*>",
    r"\[\s*system\s*\]",
    r"```\s*system",
]

COMPILED_INJECTION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in PROMPT_INJECTION_PATTERNS]


def is_prompt_injection(text: str) -> bool:
    """
    Returns True if the text contains known prompt injection or jailbreak patterns.
    """
    if not text or not isinstance(text, str):
        return False

    cleaned = text.strip()
    
    # Check regex patterns
    for pattern in COMPILED_INJECTION_PATTERNS:
        if pattern.search(cleaned):
            return True

    # Check for excessive delimiter obfuscation
    lowered = cleaned.lower()
    if "human:" in lowered and "assistant:" in lowered:
        return True
        
    return False


# ─────────────────────────────────────────────────────────────
# 2. Medical Emergency Detection
# ─────────────────────────────────────────────────────────────

EMERGENCY_PATTERNS = [
    # Heart / Cardiac
    r"\b(crushing\s+(chest\s+pain|chest\s+pressure)|chest\s+pressure\s+radiating|heart\s+attack)\b",
    r"\b(chest\s+pain\s+(and|with|\+)\s+(shortness\s+of\s+breath|left\s+arm|sweating|nausea))\b",
    
    # Stroke FAST signs (Face drooping, Arm weakness, Slurred speech, Time)
    r"\bface\s+(is\s+|feels\s+)?(droop|drooping|numb)\b",
    r"\b(slurred\s+speech|speech\s+(is\s+|feels\s+)?slurred|difficulty\s+speaking)\b",
    r"\b(arm\s+weakness|arm\s+(is\s+|feels\s+)?(weak|numb|paralyzed))\b",
    r"\b(sudden\s+numbness\s+on\s+one\s+side|stroke\s+symptoms|having\s+a\s+stroke)\b",
    
    # Severe Respiratory Distress
    r"\b(can't\s+breathe|cannot\s+breathe|severe\s+shortness\s+of\s+breath|gasping\s+for\s+air|suffocating)\b",
    r"\b(throat\s+(is\s+)?closing(\s+up)?|anaphylaxis|anaphylactic\s+shock)\b",
    
    # Severe Bleeding / Trauma
    r"\b(uncontrolled\s+bleeding|arterial\s+bleeding|coughing\s+up\s+(large\s+amounts\s+of\s+)?blood|vomiting\s+blood)\b",
    
    # Poisoning / Overdose
    r"\b(swallowed\s+poison|drank\s+bleach|overdosed\s+on\s+pills|carbon\s+monoxide\s+poisoning)\b",
    
    # Loss of consciousness / Seizure
    r"\b(unconscious\s+person|unresponsive\s+and\s+not\s+breathing|active\s+seizure\s+lasting\s+over\s+5\s+minutes)\b",
    
    # Self-harm / Suicide Crisis
    r"\b(want\s+to\s+(kill\s+myself|end\s+my\s+life|commit\s+suicide)|suicidal\s+thoughts)\b",
]

COMPILED_EMERGENCY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in EMERGENCY_PATTERNS]

EMERGENCY_RESPONSE = (
    "🚨 **CRITICAL MEDICAL ALERT: IMMEDIATE ACTION REQUIRED** 🚨\n\n"
    "Based on the symptoms you described, this may be a **life-threatening medical emergency**.\n\n"
    "**Please take the following steps IMMEDIATELY:**\n"
    "1. **Call Emergency Services right now**: Dial **911** (US/Canada), **112** (Europe/India), or **999** (UK), or your local emergency number.\n"
    "2. **If you are experiencing chest pain or stroke symptoms**: Do not drive yourself to the hospital; wait for paramedics.\n"
    "3. **If you are feeling suicidal or in emotional crisis**: Please call/text **988** (Suicide & Crisis Lifeline) or contact local emergency services immediately.\n"
    "4. **If poison was ingested**: Call Poison Control at **1-800-222-1222** (US) or your local poison center.\n\n"
    "*Do not rely on an AI chatbot for acute emergency situations. Medical personnel are equipped to save your life.*"
)


def detect_medical_emergency(text: str) -> Tuple[bool, Optional[str]]:
    """
    Detects if user input describes an acute emergency.
    Returns (True, emergency_message) if emergency detected, else (False, None).
    """
    if not text:
        return False, None

    for pattern in COMPILED_EMERGENCY_PATTERNS:
        if pattern.search(text):
            return True, EMERGENCY_RESPONSE

    return False, None


# ─────────────────────────────────────────────────────────────
# 3. Content Safety & Dangerous Request Filter
# ─────────────────────────────────────────────────────────────

HARMFUL_REQUEST_PATTERNS = [
    r"\b(how\s+to\s+(make|synthesize|cook|manufacture)\s+(meth|cocaine|fentanyl|heroin|lsd|explosives|poison|ricin|anthrax))\b",
    r"\b(how\s+to\s+(harm|kill|poison|overdose)\s+(someone|myself|a\s+person))\b",
    r"\b(lethal\s+dose\s+of\s+.*to\s+die)\b",
]

COMPILED_HARMFUL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in HARMFUL_REQUEST_PATTERNS]


def check_content_safety(text: str) -> Tuple[bool, Optional[str]]:
    """
    Checks for illegal, harmful, or dangerous non-medical requests.
    Returns (is_safe, refusal_reason)
    """
    if not text:
        return True, None

    for pattern in COMPILED_HARMFUL_PATTERNS:
        if pattern.search(text):
            return False, (
                "I cannot assist with requests involving harmful substances, illegal drug synthesis, "
                "or self-harm. As MediAssist, my goal is to provide safe, responsible health and medical information."
            )

    return True, None


# ─────────────────────────────────────────────────────────────
# 4. Master Input Guardrail Pipeline
# ─────────────────────────────────────────────────────────────

def apply_input_guardrails(user_input: str) -> Tuple[bool, str, Optional[str]]:
    """
    Runs all input guardrails in priority order:
    1. Prompt Injection
    2. Harmful Content
    3. Medical Emergency
    
    Returns:
    - (is_blocked, guardrail_category, response_message)
    - If is_blocked is False, response_message is None and normal chat processing continues.
    """
    if not user_input or not user_input.strip():
        return True, "empty_input", "Please provide a medical question or message."

    # 1. Check prompt injection
    if is_prompt_injection(user_input):
        return True, "prompt_injection", (
            "I cannot fulfill this request. I am MediAssist, a medical AI assistant, "
            "and my instructions and safety protocols cannot be overridden."
        )

    # 2. Check content safety
    is_safe, safety_msg = check_content_safety(user_input)
    if not is_safe:
        return True, "content_safety", safety_msg

    # 3. Check acute medical emergency
    is_emergency, emergency_msg = detect_medical_emergency(user_input)
    if is_emergency:
        return True, "medical_emergency", emergency_msg

    return False, "passed", None


# ─────────────────────────────────────────────────────────────
# 5. Output Safety, Decision Support Language & Medical Disclaimers
# ─────────────────────────────────────────────────────────────

NON_MEDICAL_REFUSAL = (
    "I am MediAssist, a specialized medical AI assistant. "
    "I am designed exclusively to assist with health, symptoms, wellness, and medical questions. "
    "I cannot assist with non-medical topics (such as writing code, homework, or general trivia). "
    "Please feel free to ask me any health or medical questions!"
)

MEDICAL_DISCLAIMER = (
    "\n\n---\n*Disclaimer: MediAssist provides informational clinical decision-support based on authoritative medical literature "
    "(such as The Gale Encyclopedia of Medicine, CDC, and WHO). It does not provide definitive medical diagnoses, prescriptions, "
    "or individualized treatment plans and is not a substitute for evaluation by a qualified healthcare professional.*"
)

# Common OTC and prescription drug names to suppress for high-risk patients
HIGH_RISK_PROHIBITED_DRUGS = [
    r"\bdextromethorphan\b",
    r"\bloperamide\b",
    r"\bacetaminophen\b",
    r"\bparacetamol\b",
    r"\bibuprofen\b",
    r"\badvil\b",
    r"\btylenol\b",
    r"\baspirin\b",
    r"\bnaproxen\b",
    r"\baleve\b",
    r"\bguaifenesin\b",
    r"\bpseudoephedrine\b",
    r"\bcodeine\b",
    r"\bamoxicillin\b",
    r"\bdolo\b"
]

# Decision-support language conversions (replace diagnostic phrases with decision-support phrasing)
DIAGNOSTIC_LANGUAGE_REPLACEMENTS = [
    (r"\byou\s+have\s+(a\s+)?(fever|infection|pneumonia|bronchitis|strep|covid|flu|migraine|asthma)\b", r"these symptoms are commonly associated with \2"),
    (r"\bi\s+diagnose\s+you\s+with\b", "this clinical pattern warrants evaluation for"),
    (r"\byou\s+are\s+suffering\s+from\b", "your symptoms may suggest"),
]


def enforce_decision_support_language(text: str) -> str:
    """
    Ensures language remains decision-support focused rather than declaring definitive diagnoses.
    """
    result = text
    for pattern, replacement in DIAGNOSTIC_LANGUAGE_REPLACEMENTS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def suppress_hallucinated_specialties(text: str, patient_state: Optional[Any] = None) -> Tuple[str, bool]:
    """
    Validates output against patient state disclosures.
    Detects and neutralizes hallucinated specialist or condition references not disclosed by user.
    Returns: (cleaned_text, was_hallucinated)
    """
    cleaned = text
    hallucinated = False

    has_cancer = False
    has_asthma = False
    has_pregnancy = False

    if patient_state:
        disclosed = [str(c).lower() for c in getattr(patient_state, "disclosed_conditions", [])]
        conds = [str(c).lower() for c in getattr(patient_state, "conditions", [])]
        all_conds = disclosed + conds
        has_cancer = getattr(patient_state, "is_active_cancer_chemo", False) or any("cancer" in c or "chemo" in c for c in all_conds)
        has_asthma = any("asthma" in c for c in all_conds)
        has_pregnancy = getattr(patient_state, "is_pregnant", False) or any("pregnan" in c for c in all_conds)

    # If cancer NOT disclosed, neutralize oncology hallucination
    if not has_cancer:
        if re.search(r"\b(your\s+oncologist|your\s+oncology\s+team|your\s+cancer|your\s+chemotherapy|your\s+chemo)\b", cleaned, re.IGNORECASE):
            hallucinated = True
            cleaned = re.sub(r"\byour\s+oncologist\b", "a physician", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\byour\s+oncology\s+team\b", "a medical professional", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\b(because\s+of\s+your\s+cancer(\s+therapies)?|given\s+your\s+cancer)\b.*?[,.]", "", cleaned, flags=re.IGNORECASE)

    # If asthma NOT disclosed, neutralize asthma hallucination
    if not has_asthma:
        if re.search(r"\b(your\s+asthma|your\s+pulmonologist)\b", cleaned, re.IGNORECASE):
            hallucinated = True
            cleaned = re.sub(r"\byour\s+asthma\b", "respiratory symptoms", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\byour\s+pulmonologist\b", "a doctor", cleaned, flags=re.IGNORECASE)

    # If pregnancy NOT disclosed, neutralize obstetric hallucination
    if not has_pregnancy:
        if re.search(r"\b(your\s+pregnancy|your\s+obstetrician|your\s+ob[- ]gyn)\b", cleaned, re.IGNORECASE):
            hallucinated = True
            cleaned = re.sub(r"\byour\s+pregnancy\b", "your health status", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\b(your\s+obstetrician|your\s+ob[- ]gyn)\b", "a healthcare provider", cleaned, flags=re.IGNORECASE)

    return cleaned, hallucinated


def suppress_specific_drug_dosing(text: str, patient_state: Optional[Any] = None) -> Tuple[str, bool]:
    """
    Suppresses numeric drug dosing and specific drug names in high-risk patients.
    Returns: (cleaned_text, was_dosing_suppressed)
    """
    cleaned = text
    dosing_suppressed = False

    # 1. Regex scan for numeric dosages (e.g. 10mg, 650mg, 2 tablets, q8h, every 6 hours)
    dosing_num_patterns = [
        r"\b\d+(\.\d+)?\s*(mg|milligrams?|mcg|ml|tablets?|pills?|capsules?)\b",
        r"\bq\d+h\b",
        r"\bevery\s+\d+\s*(to\s+\d+\s*)?(hours?|hrs?)\b"
    ]
    for pattern in dosing_num_patterns:
        if re.search(pattern, cleaned, re.IGNORECASE):
            dosing_suppressed = True
            cleaned = re.sub(pattern, "[consult pharmacist or physician for dosage]", cleaned, flags=re.IGNORECASE)

    # 2. For high-risk patients: drop specific drug names and enforce category-level redirect
    is_high_risk = False
    if patient_state:
        disclosed = [str(c).lower() for c in getattr(patient_state, "disclosed_conditions", [])]
        conds = [str(c).lower() for c in getattr(patient_state, "conditions", [])]
        all_conds = disclosed + conds
        is_high_risk = (
            getattr(patient_state, "is_active_cancer_chemo", False)
            or any("cancer" in c or "chemo" in c for c in all_conds)
            or getattr(patient_state, "is_immunocompromised", False)
            or getattr(patient_state, "is_pregnant", False)
            or (getattr(patient_state, "age", None) is not None and getattr(patient_state, "age") < 12)
            or getattr(patient_state, "is_infant_under_3mo", False)
        )

    if is_high_risk:
        for drug_pat in HIGH_RISK_PROHIBITED_DRUGS:
            if re.search(drug_pat, cleaned, re.IGNORECASE):
                dosing_suppressed = True
                cleaned = re.sub(
                    r"([^.\n]*?" + drug_pat[2:-2] + r"[^.\n]*?\.)",
                    " Please discuss any medication choices directly with your doctor or pharmacist.",
                    cleaned,
                    flags=re.IGNORECASE
                )

    return cleaned, dosing_suppressed


def requires_fixed_emergency_response(patient_state: Optional[Any], triage_tier: str, validator_flags: Dict[str, bool]) -> Optional[str]:
    """
    Fail-Closed Circuit Breaker:
    If triage tier is Emergency and output tripped any suppression validator,
    discard flawed generation and serve pre-approved clinician-reviewed emergency message.
    """
    if triage_tier.upper() == "EMERGENCY" and any(validator_flags.values()):
        return (
            "🚨 **CRITICAL MEDICAL EMERGENCY: IMMEDIATE CLINICAL EVALUATION REQUIRED**\n\n"
            "Based on the combination of symptoms and health factors you have reported, this situation requires "
            "**immediate in-person medical evaluation** at the nearest Emergency Department or via an emergency oncology/medical hotline.\n\n"
            "**SAFETY DIRECTIVES:**\n"
            "1. Please proceed to the nearest Emergency Department immediately. If you have an oncology team or specialist with a 24/7 hotline, contact them now.\n"
            "2. **Do not self-treat with over-the-counter medications or fever reducers** without specialist authorization, as suppressing symptoms can mask critical infection progression.\n"
            "3. If experiencing difficulty breathing, chest pain, or confusion, call 911 / 112 / 999 immediately."
        )
    return None


def apply_output_guardrails(
    response_text: str,
    is_medical: bool = False,
    show_disclaimer: bool = True,
    patient_state: Optional[Any] = None,
    triage_tier: str = "Routine"
) -> str:
    """
    Validates and enriches output with clinical safety guardrails, dosing suppression,
    grounding verification, and fail-closed emergency circuit breakers.
    """
    if not response_text:
        return "I am ready to assist with your medical questions."

    # Strip any leaked prompt tags if any
    cleaned = re.sub(r"<\s*/?\s*(system_prompt|user_query|context|memory)\s*>", "", response_text)
    
    # Strip any raw thinking blocks if model outputted them
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned)
    
    # Strip any reasoning scratchpad lines like '* User says:' if leaked
    if cleaned.startswith("*   User says:") or cleaned.startswith("* User says:"):
        option_match = re.search(r"\*Option \d+.*?\*:\s*(.*)", cleaned)
        if option_match:
            cleaned = option_match.group(1).strip()
        else:
            lines = [l for l in cleaned.split("\n") if not l.strip().startswith("*")]
            cleaned = "\n".join(lines).strip() or cleaned

    # Enforce non-diagnostic decision-support wording
    cleaned = enforce_decision_support_language(cleaned)

    # Run Grounding Validation (Neutralize unstated oncology/asthma references)
    cleaned, flag_hallucinated = suppress_hallucinated_specialties(cleaned, patient_state)

    # Run Dosing & Drug-Identity Suppression
    cleaned, flag_dosing = suppress_specific_drug_dosing(cleaned, patient_state)

    # Fail-Closed Circuit Breaker on Emergency Tier
    circuit_breaker_resp = requires_fixed_emergency_response(
        patient_state,
        triage_tier,
        {"hallucinated_specialty": flag_hallucinated, "dosing_detected": flag_dosing}
    )
    if circuit_breaker_resp:
        return circuit_breaker_resp

    cleaned = cleaned.strip()

    # Append disclaimer once if requested and not already present
    if is_medical and show_disclaimer and "Disclaimer:" not in cleaned and len(cleaned) > 120:
        cleaned += MEDICAL_DISCLAIMER

    return cleaned

