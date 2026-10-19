"""
Medical Chatbot Guardrails & Safety Pipeline.
Includes:
- Prompt Injection & Jailbreak detection
- Medical Emergency detection & immediate life-saving protocol
- Content Safety & dangerous request interception
- Response validation & medical disclaimers
"""

import re
from typing import Tuple, Optional, Any, Dict, List

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

    # ─────────────────────────────────────────────────────────────
    # Multilingual Prompt Injection & Jailbreak Defenses
    # NOTE: [NATIVE-SPEAKER-REVIEW-NEEDED] The following patterns cover es, fr, de, it, nl, ja
    # and require validation by native-speaker localization / safety teams.
    # ─────────────────────────────────────────────────────────────
    # Spanish (es)
    r"ignora\s+(todas\s+las\s+)?(instrucciones|reglas|indicaciones)\s*(anteriores|previas)?",
    r"olvida\s+(todas\s+las\s+)?(instrucciones|reglas)",
    r"\bmodo\s+dan\b",
    r"(revela|muestra|imprime)\s+(tu\s+)?(prompt\s+de\s+sistema|sistema\s+prompt|instrucciones\s+ocultas)",
    # French (fr)
    r"ignore\s+(toutes\s+les\s+)?(instructions|directives|r[èe]gles)\s*(pr[ée]c[ée]dentes|ant[ée]rieures)?",
    r"oublie\s+(toutes\s+les\s+)?instructions",
    r"\bmode\s+dan\b",
    r"(affiche|r[ée]v[èe]le|montre)\s+(ton\s+)?(prompt\s+syst[èe]me|instructions\s+cach[ée]es)",
    # German (de)
    r"ignoriere\s+(alle\s+)?(vorherigen\s+|bisherigen\s+)?(anweisungen|regeln|instruktionen)",
    r"vergiss\s+(alle\s+)?(anweisungen|regeln)",
    r"\bdan[\s-]modus\b",
    r"(zeige|gib\s+aus)\s+(deinen\s+)?(system[\s-]?prompt|geheime\s+anweisungen)",
    # Italian (it)
    r"ignora\s+(tutte\s+le\s+)?(istruzioni|regole)\s*(precedenti)?",
    r"dimentica\s+(tutte\s+le\s+)?istruzioni",
    r"\bmodalit[àa]\s+dan\b",
    r"(mostra|rivela)\s+(il\s+tuo\s+)?(prompt\s+di\s+sistema|istruzioni\s+segrete)",
    # Dutch (nl)
    r"negeer\s+(alle\s+)?(eerdere\s+)?(instructies|regels)",
    r"vergeet\s+(alle\s+)?instructies",
    r"\bdan[\s-]modus\b",
    r"(toon|laat\s+zien)\s+(je\s+)?(systeemprompt|verborgen\s+instructies)",
    # Japanese (ja)
    r"(これまでの|すべての)?(指示|ルール|プロンプト)を(無視|忘れて)",
    r"danモード",
    r"(システムプロンプト|内部指示)を(表示|出力|教えて)",
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
    r"\b(chest\s+pain|chest\s+pressure)\b.*?\b(radiating|left\s+arm|left\s+shoulder|shortness\s+of\s+breath|sweating|nausea)\b",
    
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

    # ─────────────────────────────────────────────────────────────
    # Multilingual Medical Emergency & Acute Crisis Defenses
    # NOTE: [NATIVE-SPEAKER-REVIEW-NEEDED] The following patterns cover es, fr, de, it, nl, ja
    # and require validation by native-speaker localization / clinical safety teams.
    # ─────────────────────────────────────────────────────────────
    # Spanish (es)
    r"\b(dolor\s+(punzante\s+|opresivo\s+)?en\s+el\s+pecho|ataque\s+al\s+coraz[oó]n|infarto)\b",
    r"\b(falta\s+el\s+aire|no\s+puedo\s+respirar|dificultad\s+(grave\s+)?para\s+respirar|asfixi[aá]ndome)\b",
    r"\b(accidente\s+cerebrovascular|derrame\s+cerebral|cara\s+ca[ií]da)\b",
    r"\b(sangrado\s+incontrolable|vomitando\s+sangre|tosiendo\s+sangre)\b",
    r"\b(quiero\s+(morir|suicidarme|quitarme\s+la\s+vida|terminar\s+con\s+mi\s+vida)|pensamientos\s+suicidas)\b",

    # French (fr)
    r"\b(douleur\s+thoracique(\s+[ée]crasante)?|crise\s+cardiaque|infarctus)\b",
    r"\b(bras\s+gauche\s+est\s+engourdi|je\s+ne\s+peux\s+pas\s+respirer|difficult[ée]\s+respiratoire|[ée]touffement)\b",
    r"\b(accident\s+vasculaire\s+c[ée]r[ée]bral|avc|visage\s+affaiss[ée])\b",
    r"\b(saignement\s+incontr[ôo]lable|vomir\s+du\s+sang|cracher\s+du\s+sang)\b",
    r"\b(veux\s+mettre\s+fin\s+[àa]\s+mes\s+jours|envie\s+de\s+(mourir|me\s+suicider)|id[ée]es\s+suicides)\b",

    # German (de)
    r"\b(vernichtende\s+brustschmerzen|brustschmerz|herzinfarkt|schmerzen\s+in\s+der\s+brust)\b",
    r"\b(kann\s+nicht\s+atmen|keine\s+luft|schwere\s+atemnot|ersticke)\b",
    r"\b(schlaganfall|gesichtsl[äa]hmung|herabh[äa]ngender\s+mundwinkel)\b",
    r"\b(unkontrollierbare\s+blutung|blut\s+erbrechen|blut\s+husten)\b",
    r"\b(suizidgedanken|selbstmord|nicht\s+mehr\s+weiterleben|mein\s+leben\s+beenden|will\s+sterben)\b",

    # Italian (it)
    r"\b(dolore\s+oppressivo\s+al\s+torace|dolore\s+al\s+petto|attacco\s+di\s+cuore|infarto)\b",
    r"\b(non\s+riesco\s+a\s+respirare|grave\s+difficolt[àa]\s+respiratoria|sto\s+soffocando)\b",
    r"\b(ictus|paresi\s+facciale|faccia\s+cadente)\b",
    r"\b(emorragia\s+incontrollabile|vomitare\s+sangue|sputare\s+sangue)\b",
    r"\b(voglio\s+(togliermi\s+la\s+vita|morire|suicidarmi|farla\s+finita)|pensieri\s+suicidi)\b",

    # Dutch (nl)
    r"\b(beklemmende\s+pijn\s+op\s+de\s+borst|pijn\s+op\s+de\s+borst|hartaanval|hartinfarct)\b",
    r"\b(kan\s+niet\s+ademen|ernstige\s+benauwdheid|stik|geen\s+lucht)\b",
    r"\b(beroerte|scheve\s+mond|verlamming\s+in\s+gezicht)\b",
    r"\b(onbeheersbare\s+bloeding|bloed\s+overgeven|bloed\s+spugen)\b",
    r"\b(wil\s+zelfmoord\s+plegen|een\s+einde\s+aan\s+mijn\s+leven\s+maken|zie\s+het\s+leven\s+niet\s+meer\s+zitten|wil\s+dood)\b",

    # Japanese (ja)
    r"(胸が締め付けられるような激痛|胸の激しい痛み|心筋梗塞)",
    r"(息ができない|呼吸困難|窒息しそう)",
    r"(脳卒中|顔の麻痺|ろれつが回らない)",
    r"(止まらない大量出血|血を吐いた|吐血)",
    r"(死にたい|自殺したい|命を絶ちたい|自死)",
]

COMPILED_EMERGENCY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in EMERGENCY_PATTERNS]

EMERGENCY_RESPONSE = (
    "**URGENT MEDICAL ALERT: IMMEDIATE ACTION REQUIRED**\n\n"
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
# 3b. Code & Programming Generation Filter
# ─────────────────────────────────────────────────────────────

CODE_REQUEST_PATTERNS = [
    r"\b(write|create|generate|give\s+me|show\s+me|build|make|provide|need|print)\s+.*(code|script|program|class|function|sql|query|algorithm|app|bot|snippet)\b",
    r"\b(python|javascript|typescript|java|c\+\+|c#|ruby|golang|rust|html|css|sql|bash|powershell)\s+(code|script|program|class|function|snippet|query|example)\b",
    r"\b(write|create|code|script)\s+.*(in|using|with)\s+(python|javascript|typescript|java|c\+\+|c#|sql|bash)\b",
    r"\bwrite\s+(me\s+)?(a\s+)?(python|script|code|program|sql|function|class)\b",
    r"\b(how\s+to\s+(code|program|script))\b",
    r"\b(sql\s+query|database\s+query)\b",
    r"\b(script\s+to\s+(track|scrape|run|calculate|automate|manage))\b",
]

COMPILED_CODE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in CODE_REQUEST_PATTERNS]


def is_code_or_programming_request(text: str) -> bool:
    """
    Canonical single source of truth for detecting code, script, programming, or developer requests.
    Evaluates independently of medical keywords.
    """
    if not text or not isinstance(text, str):
        return False
    cleaned = text.strip()
    for pattern in COMPILED_CODE_PATTERNS:
        if pattern.search(cleaned):
            return True
    return False


# ─────────────────────────────────────────────────────────────
# 4. Master Input Guardrail Pipeline
# ─────────────────────────────────────────────────────────────

def apply_input_guardrails(user_input: str) -> Tuple[bool, str, Optional[str]]:
    """
    Runs all input guardrails in priority order:
    1. Prompt Injection
    2. Harmful Content
    3. Non-medical Code / Script Request
    4. Medical Emergency
    
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

    # 3. Check non-medical code / script generation request
    if is_code_or_programming_request(user_input):
        return True, "non_medical_code", NON_MEDICAL_REFUSAL

    # 4. Check acute medical emergency
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

NO_REFERENCE_DISCLAIMER = (
    "\n\n---\n*Disclaimer: MediAssist provides informational clinical decision-support based on general medical knowledge. "
    "It does not provide definitive medical diagnoses, prescriptions, "
    "or individualized treatment plans and is not a substitute for evaluation by a qualified healthcare professional.*"
)

# ─────────────────────────────────────────────────────────────
# Sentence Boundary & Splitting Primitives
# ─────────────────────────────────────────────────────────────

ABBREVIATIONS = frozenset({
    "dr", "mr", "mrs", "ms", "prof", "vs", "eg", "ie", "etc", "approx", "no", "dept", "vol", "fig"
})


def find_sentence_boundary(buffer: str) -> Optional[int]:
    """
    Finds the end index of the first completed sentence in buffer,
    avoiding decimal numbers (10.5 mg) and abbreviations (Dr., vs., e.g.).
    Returns index to split at (inclusive of delimiter and trailing space), or None.
    """
    pattern = re.compile(r"([.!?]+(?:\s+|\n+)|[\n]{2,})")
    pos = 0
    while True:
        match = pattern.search(buffer, pos)
        if not match:
            break
        punct_start = match.start()
        punct_char = buffer[punct_start]

        if punct_char == ".":
            # 1. Decimal check: digits immediately before and after
            if punct_start > 0 and punct_start + 1 < len(buffer):
                if buffer[punct_start - 1].isdigit() and buffer[punct_start + 1].isdigit():
                    pos = match.end()
                    continue

            # 2. Abbreviation check: examine preceding word
            prefix = buffer[:punct_start]
            words = re.findall(r"[A-Za-z]+", prefix)
            if words:
                last_word = words[-1].lower()
                if last_word in ABBREVIATIONS:
                    pos = match.end()
                    continue
                # Also check multi-part abbreviations like e.g. or i.e.
                if len(words) >= 2 and f"{words[-2].lower()}.{words[-1].lower()}" in {"e.g", "i.e"}:
                    pos = match.end()
                    continue

        return match.end()

    return None


def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences using find_sentence_boundary, preserving
    abbreviations (Dr., e.g., vs.) and decimal numbers (10.5 mg).
    Trailing non-delimited text is returned as the final sentence segment.
    """
    sentences: List[str] = []
    rem = text
    while rem:
        idx = find_sentence_boundary(rem)
        if idx is not None and idx > 0:
            sentences.append(rem[:idx])
            rem = rem[idx:]
        else:
            sentences.append(rem)
            break
    return sentences


def is_pediatric(patient_state: Optional[Any]) -> bool:
    """
    Evaluates whether patient is pediatric:
    - Any age reported in days, weeks, or months counts as pediatric.
    - Only age with unit 'years' (or unstated) is compared against < 12.
    - is_infant_under_3mo flag also counts as pediatric.
    """
    if not patient_state:
        return False
    if isinstance(patient_state, dict):
        if patient_state.get("is_infant_under_3mo"):
            return True
        age = patient_state.get("age")
        unit = str(patient_state.get("age_unit", "years") or "years").lower().strip()
    else:
        if getattr(patient_state, "is_infant_under_3mo", False):
            return True
        age = getattr(patient_state, "age", None)
        unit = str(getattr(patient_state, "age_unit", "years") or "years").lower().strip()

    if age is None:
        return False
    try:
        age_val = float(age)
    except (ValueError, TypeError):
        return False

    if any(u in unit for u in ["day", "week", "month"]):
        return True
    return age_val < 12.0


# ─────────────────────────────────────────────────────────────
# Drug Lists & Interception Patterns
# ─────────────────────────────────────────────────────────────

# GREETING_PROHIBITED_DRUGS: Used exclusively by _is_greeting_sane() to ensure
# conversational greetings do not hallucinate unsolicited medication advice.
GREETING_PROHIBITED_DRUGS = [
    r"\bdextromethorphan\b",
    r"\bloperamide\b",
    r"\bacetaminophen\b",
    r"\bparacetamol\b",
    r"\bibuprofen\b",
    r"\badvil\b",
    r"\btylenol\b",
    r"\bmotrin\b",
    r"\baspirin\b",
    r"\bnaproxen\b",
    r"\baleve\b",
    r"\bguaifenesin\b",
    r"\bpseudoephedrine\b",
    r"\bcodeine\b",
    r"\bamoxicillin\b",
    r"\bdolo\b"
]
# Backward-compatibility alias
UNIVERSALLY_PROHIBITED_DRUGS = GREETING_PROHIBITED_DRUGS

# HIGH_RISK_DRUG_PATTERNS: Drug names and brands restricted for high-risk patients
# (cancer/immunosuppressed, pregnancy, pediatrics).
# Stored as a separate list from GREETING_PROHIBITED_DRUGS because high-risk clinical
# interception has different clinical scope and coverage requirements.
HIGH_RISK_DRUG_PATTERNS = [
    r"\bdextromethorphan\b",
    r"\bloperamide\b",
    r"\bacetaminophen\b",
    r"\bparacetamol\b",
    r"\bibuprofen\b",
    r"\badvil\b",
    r"\btylenol\b",
    r"\bmotrin\b",
    r"\baspirin\b",
    r"\bnaproxen\b",
    r"\baleve\b",
    r"\bguaifenesin\b",
    r"\bpseudoephedrine\b",
    r"\bcodeine\b",
    r"\bamoxicillin\b",
    r"\bdolo\b",
    # Regional brands and NSAID/antipyretic additions (needs pharmacist review)
    r"\bdiclofenac\b",       # needs pharmacist review
    r"\bnimesulide\b",       # needs pharmacist review
    r"\bmefenamic\s+acid\b", # needs pharmacist review
    r"\bcrocin\b",           # needs pharmacist review
    r"\bcalpol\b",           # needs pharmacist review
    r"\bcombiflam\b"         # needs pharmacist review
]
HIGH_RISK_PROHIBITED_DRUGS = HIGH_RISK_DRUG_PATTERNS

CLINICAL_DEFERRAL_TEXT = "Please discuss any medication choices directly with your doctor or pharmacist."

# Universal dosing patterns: catches exact dosages, frequencies, and schedules
# while carefully preserving non-drug quantities (e.g. 250 ml of water, 5 g of protein).
DOSING_NUM_PATTERNS = [
    # 1. Exact drug quantities: mg, mcg, etc.
    re.compile(r"\b(?:half|quarter|one|two|three|four|five|\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?)\s*(?:mg|milligrams?|mcg|micrograms?)\b", re.IGNORECASE),
    # 2. Pill/tablet/capsule counts with word numbers, digit ranges, or fractions
    re.compile(r"\b(?:take\s+)?(?:half|quarter|one|two|three|four|five|\d+(?:\s*-\s*\d+)?)\s*(?:a\s+)?(?:tablets?|pills?|capsules?|drops?|caplets?|lozenges?)\b", re.IGNORECASE),
    # 3. Milliliters when NOT followed by water/fluids/juice/tea/soup/broth/milk/solution/electrolyte
    re.compile(r"\b(?:take\s+)?(?:\d+(?:\.\d+)?)\s*(?:ml|milliliters?)(?!\s+of\s+(?:water|fluids?|juice|liquids?|tea|soup|broth|milk|solution|electrolyte))\b", re.IGNORECASE),
    # 4. Frequencies and schedules: every N hours, qNh, once/twice daily, x times a day
    re.compile(r"\bq\d+h\b", re.IGNORECASE),
    re.compile(r"\bevery\s+(?:\d+|one|two|three|four|five|six|eight|twelve)(?:\s*(?:-|to)\s*\d+)?\s*(?:hours?|hrs?)\b", re.IGNORECASE),
    re.compile(r"\b(?:once|twice|three\s+times|\d+\s*times)\s+(?:a\s+day|daily|per\s+day)\b", re.IGNORECASE),
    re.compile(r"\btwice\s+daily\b", re.IGNORECASE),
    re.compile(r"\bonce\s+daily\b", re.IGNORECASE)
]

# Position-aware prohibition patterns: prohibition word must appear BEFORE the drug within ~40 characters.
# Restricts 'avoid' to 'avoid <drug>' forms so 'avoiding side effects' does not count.
PROHIBITION_BEFORE_DRUG_PATTERNS = [
    re.compile(
        rf"\b(?:do\s+not(?:\s+take|\s+use|\s+administer|\s+give)?|"
        rf"don'?t(?:\s+take|\s+use|\s+administer|\s+give)?|"
        rf"never(?:\s+take|\s+use|\s+administer|\s+give)?|"
        rf"should\s+not(?:\s+take|\s+use)?|"
        rf"must\s+not(?:\s+take|\s+use|\s+be\s+taken)?|"
        rf"avoid(?:\s+taking|\s+using)?|"
        rf"contraindicated(?:\s+for|\s+in)?|"
        rf"prohibited(?:\s+to\s+take)?)\b[\w\s]{{0,40}}{drug_pat}",
        re.IGNORECASE
    )
    for drug_pat in HIGH_RISK_DRUG_PATTERNS
]

# Drug followed directly by is/are contraindicated/prohibited
DRUG_IS_CONTRAINDICATED_PATTERNS = [
    re.compile(
        rf"{drug_pat}[\w\s]{{0,25}}\b(?:is|are)\s+(?:contraindicated|strictly\s+prohibited|prohibited)\b",
        re.IGNORECASE
    )
    for drug_pat in HIGH_RISK_DRUG_PATTERNS
]

# Permissive or recommending verbs directed at patient/treatment
PERMISSIVE_VERBS_PAT = re.compile(
    r"\b(?:you\s+)?(?:can|could|may|might|should|to)?\s*(?:take|try|use|administer|give|recommend|suggest|prescribe)\b|"
    r"\b(?:recommended|suggested)\s+to\s+take\b",
    re.IGNORECASE
)

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

    # If cancer NOT disclosed, neutralize oncology/cancer hallucination (including compound phrases)
    if not has_cancer:
        cancer_patterns = [
            r"\b(?:keep\s+)?your\s+[\w\s]{0,35}?(?:oncolog\w*|cancer|chemo\w*|leukemia|lymphoma)\s+(?:and\s+[\w\s]+\s+)?(?:care\s+providers?|team|doctor|physician|specialist|treatment|therap\w*)[^.\n]*[.?]?",
            r"\b(your\s+oncologist|your\s+oncology\s+team|your\s+cancer|your\s+chemotherapy|your\s+chemo)\b",
            r"\b(?:oncology|cancer|chemo)\s+care\s+providers?\b",
            r"\b(because\s+of\s+your\s+cancer(\s+therapies)?|given\s+your\s+cancer)\b.*?[,.]"
        ]
        for pat in cancer_patterns:
            if re.search(pat, cleaned, re.IGNORECASE):
                hallucinated = True
                cleaned = re.sub(pat, " your healthcare provider ", cleaned, flags=re.IGNORECASE)

    # If asthma NOT disclosed, neutralize asthma/pulmonology hallucination
    if not has_asthma:
        asthma_patterns = [
            r"\b(?:keep\s+)?your\s+[\w\s]{0,35}?(?:asthma|pulmonolog\w*)\s+(?:and\s+[\w\s]+\s+)?(?:care\s+providers?|team|doctor|physician|specialist|treatment|inhaler)[^.\n]*[.?]?",
            r"\b(your\s+asthma|your\s+pulmonologist)\b",
            r"\basthma\s+care\s+providers?\b"
        ]
        for pat in asthma_patterns:
            if re.search(pat, cleaned, re.IGNORECASE):
                hallucinated = True
                cleaned = re.sub(pat, " your healthcare provider ", cleaned, flags=re.IGNORECASE)

    # If pregnancy NOT disclosed, neutralize obstetric hallucination
    if not has_pregnancy:
        preg_patterns = [
            r"\b(your\s+pregnancy|your\s+obstetrician|your\s+ob[- ]gyn)\b",
            r"\bobstetric\s+care\s+providers?\b"
        ]
        for pat in preg_patterns:
            if re.search(pat, cleaned, re.IGNORECASE):
                hallucinated = True
                cleaned = re.sub(pat, " your healthcare provider ", cleaned, flags=re.IGNORECASE)

    # Clean up any duplicate spacing created by redactions
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, hallucinated


def suppress_specific_drug_dosing(text: str, patient_state: Optional[Any] = None) -> Tuple[str, bool]:
    """
    Suppresses numeric drug dosing universally for all users, and intercepts specific drug
    names for high-risk patients (cancer/immunosuppressed, pregnancy, pediatrics):
    - Universal: numeric dosages and dosing schedules (tablets, mg, q8h, once/twice daily)
      are suppressed without mangling non-drug quantities (e.g. 250 ml of water, 5 g of protein).
    - High-risk (cancer/chemo/immunosuppressed): strict antipyretic interception replacing
      recommending/permissive sentences with deferral text while preserving explicit prohibitions intact.
    - High-risk (pregnancy/pediatrics): replaces recommending/instructing sentences with deferral text;
      preserves purely descriptive informational sentences (e.g. 'Paracetamol is used to reduce fever').
    - Standard risk: allows drug names in informational context ('what is paracetamol used for?').
    Returns: (cleaned_text, was_dosing_suppressed)
    """
    cleaned = text
    dosing_suppressed = False

    # 1. Regex scan for numeric dosages and administration schedules (Universal)
    for pattern in DOSING_NUM_PATTERNS:
        if pattern.search(cleaned):
            dosing_suppressed = True
            cleaned = pattern.sub("[consult pharmacist or physician for dosage]", cleaned)

    # Collapse consecutive placeholder repetitions
    cleaned = re.sub(
        r"(\[consult pharmacist or physician for dosage\]\s*)+",
        "[consult pharmacist or physician for dosage] ",
        cleaned
    ).strip()
    cleaned = re.sub(r"\s+([.!?])", r"\1", cleaned)

    # 2. Check patient risk profile
    is_high_risk = False
    is_cancer_or_immuno = False
    is_preg_or_peds = False

    if patient_state:
        if isinstance(patient_state, dict):
            disclosed = [str(c).lower() for c in patient_state.get("disclosed_conditions", [])]
            conds = [str(c).lower() for c in patient_state.get("conditions", [])]
            all_conds = disclosed + conds
            has_cancer = bool(
                patient_state.get("has_cancer_history")
                or patient_state.get("is_active_cancer_chemo")
                or any("cancer" in c or "chemo" in c for c in all_conds)
            )
            has_immuno = bool(patient_state.get("is_immunocompromised") or any("immuno" in c for c in all_conds))
            has_preg = bool(patient_state.get("is_pregnant") or any("pregnan" in c for c in all_conds))
            has_peds = is_pediatric(patient_state)
        else:
            disclosed = [str(c).lower() for c in getattr(patient_state, "disclosed_conditions", [])]
            conds = [str(c).lower() for c in getattr(patient_state, "conditions", [])]
            all_conds = disclosed + conds
            has_cancer = bool(
                getattr(patient_state, "has_cancer_history", False)
                or getattr(patient_state, "is_active_cancer_chemo", False)
                or any("cancer" in c or "chemo" in c for c in all_conds)
            )
            has_immuno = bool(getattr(patient_state, "is_immunocompromised", False) or any("immuno" in c for c in all_conds))
            has_preg = bool(getattr(patient_state, "is_pregnant", False) or any("pregnan" in c for c in all_conds))
            has_peds = is_pediatric(patient_state)

        is_cancer_or_immuno = has_cancer or has_immuno
        is_preg_or_peds = has_preg or has_peds
        is_high_risk = is_cancer_or_immuno or is_preg_or_peds

    if not is_high_risk:
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleaned, dosing_suppressed

    # 3. Evaluate drug mentions sentence-by-sentence using shared splitter
    sentences = split_into_sentences(cleaned)
    processed_sentences: List[str] = []
    deferral_added = False

    for sent in sentences:
        sent_lower = sent.lower()
        # Does this sentence mention any restricted high-risk drug?
        matched_drug_indices = [
            i for i, drug_pat in enumerate(HIGH_RISK_DRUG_PATTERNS)
            if re.search(drug_pat, sent_lower)
        ]

        if not matched_drug_indices:
            processed_sentences.append(sent)
            continue

        dosing_suppressed = True

        # Position-aware prohibition check: prohibition must appear BEFORE drug within ~40 chars
        has_prohibition = any(
            PROHIBITION_BEFORE_DRUG_PATTERNS[idx].search(sent_lower)
            or DRUG_IS_CONTRAINDICATED_PATTERNS[idx].search(sent_lower)
            for idx in matched_drug_indices
        )

        # Check for permissive / recommending verbs
        # Strip prohibition clauses first so 'take' in 'do not take' does not trigger permissive
        s_no_prohib = sent_lower
        for idx in matched_drug_indices:
            s_no_prohib = PROHIBITION_BEFORE_DRUG_PATTERNS[idx].sub("", s_no_prohib)
            s_no_prohib = DRUG_IS_CONTRAINDICATED_PATTERNS[idx].sub("", s_no_prohib)
        s_no_prohib = re.sub(r"\b(?:do\s+not|don'?t|never|should\s+not|must\s+not|avoid)\s+\w+", "", s_no_prohib)
        has_permissive = bool(PERMISSIVE_VERBS_PAT.search(s_no_prohib))

        # Rule 1: If sentence has both prohibition and permissive verb, treat as permissive
        # Rule 2: If position-aware prohibition holds with NO permissive verb -> KEEP intact
        if has_prohibition and not has_permissive:
            processed_sentences.append(sent)
            continue

        # Rule 3: For pregnancy and pediatric cases, descriptive informational sentences
        # (e.g. "Paracetamol is used to reduce fever") stay intact without instruction.
        if is_preg_or_peds and not is_cancer_or_immuno and not has_permissive:
            processed_sentences.append(sent)
            continue

        # Rule 4: Recommending / permissive sentence to high-risk patient -> replace with deferral
        if not deferral_added:
            processed_sentences.append(f" {CLINICAL_DEFERRAL_TEXT}")
            deferral_added = True

    cleaned = "".join(processed_sentences)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, dosing_suppressed


def requires_fixed_emergency_response(patient_state: Optional[Any], triage_tier: str, validator_flags: Dict[str, bool]) -> Optional[str]:
    """
    Fail-Closed Circuit Breaker:
    If triage tier is Emergency and output tripped any suppression validator,
    discard flawed generation and serve pre-approved clinician-reviewed emergency message.
    """
    if triage_tier.upper() == "EMERGENCY" and any(validator_flags.values()):
        return (
            "**CRITICAL MEDICAL EMERGENCY: IMMEDIATE CLINICAL EVALUATION REQUIRED**\n\n"
            "Based on the combination of symptoms and health factors you have reported, this situation requires "
            "**immediate in-person medical evaluation** at the nearest Emergency Department or via an emergency oncology/medical hotline.\n\n"
            "**SAFETY DIRECTIVES:**\n"
            "1. Please proceed to the nearest Emergency Department immediately. If you have an oncology team or specialist with a 24/7 hotline, contact them now.\n"
            "2. **Do not self-treat with over-the-counter medications or fever reducers** without specialist authorization, as suppressing symptoms can mask critical infection progression.\n"
            "3. If experiencing difficulty breathing, chest pain, or confusion, call 911 / 112 / 999 immediately."
        )
    return None


def normalize_unicode_punctuation(text: str) -> str:
    """Normalizes unicode hyphens, spaces, and quotes to standard ASCII equivalents."""
    return (
        text.replace("\u2011", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u202f", " ")
        .replace("\u00a0", " ")
        .replace("\u2248", "~")
    )


def apply_sentence_guardrails(
    text: str,
    patient_state: Optional[Any] = None
) -> Tuple[str, Dict[str, bool]]:
    """
    Pure text transforms safe to run on an individual sentence during streaming:
    1. Strip prompt tags / thinking blocks / scratchpads
    2. Enforce non-diagnostic decision-support wording
    3. Grounding specialty hallucination suppression
    4. Dosing and drug-identity suppression
    5. Unicode punctuation normalization
    Returns: (cleaned_sentence, flags_dict)
    """
    if not text:
        return "", {"hallucinated_specialty": False, "dosing_detected": False}

    cleaned = re.sub(r"<\s*/?\s*(system_prompt|user_query|context|memory)\s*>", "", text)
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned)
    if cleaned.startswith("*   User says:") or cleaned.startswith("* User says:"):
        option_match = re.search(r"\*Option \d+.*?\*:\s*(.*)", cleaned)
        if option_match:
            cleaned = option_match.group(1).strip()
        else:
            lines = [l for l in cleaned.split("\n") if not l.strip().startswith("*")]
            cleaned = "\n".join(lines).strip() or cleaned

    cleaned = enforce_decision_support_language(cleaned)
    cleaned, flag_hallucinated = suppress_hallucinated_specialties(cleaned, patient_state)
    cleaned, flag_dosing = suppress_specific_drug_dosing(cleaned, patient_state)
    cleaned = normalize_unicode_punctuation(cleaned)

    return cleaned, {"hallucinated_specialty": flag_hallucinated, "dosing_detected": flag_dosing}


def apply_output_guardrails(
    response_text: str,
    is_medical: bool = False,
    show_disclaimer: bool = True,
    patient_state: Optional[Any] = None,
    triage_tier: str = "Routine",
    no_reference: bool = False
) -> str:
    """
    Validates and enriches output with clinical safety guardrails, dosing suppression,
    grounding verification, and fail-closed emergency circuit breakers.
    """
    if not response_text:
        return "I am ready to assist with your medical questions."

    cleaned, flags = apply_sentence_guardrails(response_text, patient_state)
    flag_hallucinated = flags["hallucinated_specialty"]
    flag_dosing = flags["dosing_detected"]

    # Fail-Closed Circuit Breaker on Emergency Tier
    circuit_breaker_resp = requires_fixed_emergency_response(
        patient_state,
        triage_tier,
        {"hallucinated_specialty": flag_hallucinated, "dosing_detected": flag_dosing}
    )
    if circuit_breaker_resp:
        return circuit_breaker_resp

    if no_reference:
        # Amendment 7: never claim output is based on Gale Encyclopedia/CDC/WHO in no-reference mode
        cleaned = re.sub(
            r"\b(?:based\s+on|according\s+to|citing|referenced\s+in)\s+(?:the\s+)?(?:gale\s+encyclopedia(?:\s+of\s+medicine)?|cdc|who)\b[^.,;\n]*[.,;]?",
            "",
            cleaned,
            flags=re.IGNORECASE
        )
        cleaned = re.sub(r"\bThe Gale Encyclopedia of Medicine\b", "general clinical literature", cleaned, flags=re.IGNORECASE)

    # Append or strip disclaimer
    active_disclaimer = NO_REFERENCE_DISCLAIMER if no_reference else MEDICAL_DISCLAIMER
    if not show_disclaimer:
        # Strip any redundant disclaimer if generated by LLM or retrieved context
        cleaned = re.sub(r'(?:\n|\s)*---\s*\*?Disclaimer:.*$', '', cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
        cleaned = re.sub(r'\*?Disclaimer:\s*MediAssist.*$', '', cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    elif is_medical and "Disclaimer:" not in cleaned and len(cleaned) > 40:
        cleaned += active_disclaimer
    elif is_medical and no_reference and "Disclaimer:" in cleaned:
        cleaned = re.sub(r'(?:\n|\s)*---\s*\*?Disclaimer:.*$', '', cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
        cleaned += NO_REFERENCE_DISCLAIMER

    # Clean up trailing incomplete sentence / dangling bullet point if cut off mid-generation
    body = cleaned
    disc = ""
    if "---" in body and "Disclaimer:" in body:
        parts = body.split("---", 1)
        body = parts[0]
        disc = "\n\n---" + parts[1]

    if body and not body.strip().endswith((".", "!", ")", "*", '"', "'")):
        body_stripped = re.sub(r'[\r\n]+[-*•]\s*[^\r\n.!?)]*$', '', body).strip()
        if body_stripped and body_stripped.endswith((".", "!", ")", "*", '"', "'")):
            body = body_stripped
        else:
            last_punct = max(body.rfind("."), body.rfind("!"), body.rfind("?"))
            if last_punct > 40:
                body = body[:last_punct + 1].strip()
    cleaned = body + disc

    return cleaned

