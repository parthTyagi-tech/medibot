"""
Medical Chatbot Guardrails & Safety Pipeline.
Includes:
- Prompt Injection & Jailbreak detection
- Medical Emergency detection & immediate life-saving protocol
- Content Safety & dangerous request interception
- Response validation & medical disclaimers
"""

import re
from typing import Tuple, Optional

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
# 5. Output Safety & Medical Disclaimers
# ─────────────────────────────────────────────────────────────

MEDICAL_DISCLAIMER = (
    "\n\n---\n*Disclaimer: MediAssist provides informational medical guidance based on clinical literature "
    "(such as The Gale Encyclopedia of Medicine) and is not a substitute for professional medical diagnosis, "
    "treatment, or advice from a qualified healthcare provider.*"
)


def apply_output_guardrails(response_text: str, is_medical: bool = False) -> str:
    """
    Validates and enriches output with necessary medical guardrails.
    """
    if not response_text:
        return "I am ready to assist with your medical questions."

    # Strip any leaked prompt tags if any
    cleaned = re.sub(r"<\s*/?\s*(system_prompt|user_query|context|memory)\s*>", "", response_text)
    
    # Strip any raw thinking blocks if model outputted them
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned).strip()

    # If medical query and disclaimer not already present, append a concise disclaimer
    if is_medical and "Disclaimer:" not in cleaned and len(cleaned) > 120:
        cleaned += MEDICAL_DISCLAIMER

    return cleaned
