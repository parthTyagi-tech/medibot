import re
from typing import Tuple, Optional, List, Dict, Any
from research.src.guardrails import is_code_or_programming_request
from services.classifier_config import get_classifier_config

# Anchored regexes for whole-message matching (Amendment 4 & 8)
# Evaluated strictly as whole-message/anchored, preventing medical queries containing "logout" from misclassifying.
ANCHORED_ACCOUNT_REGEX = re.compile(
    r"^\s*(delete\s+(my\s+)?account|delete\s+this\s+consultation|delete\s+chat|log\s*out|sign\s*out)\s*[.?!]*$",
    re.IGNORECASE
)
ANCHORED_MEMORY_REGEX = re.compile(
    r"^\s*(what\s+symptoms\s+did\s+i\s+mention|what\s+do\s+you\s+know\s+about\s+me|what\s+did\s+i\s+tell\s+you|"
    r"what\s+was\s+my\s+allergy|what\s+medications?\s+am\s+i\s+taking|my\s+past\s+consultations|"
    r"recall\s+my\s+symptoms|recall\s+my\s+medical\s+history|remember\s+what\s+i\s+said|what\s+is\s+my\s+diagnosis)\s*[.?!]*$",
    re.IGNORECASE
)

# Comprehensive list of medical keyword roots for instant 0ms classification
MEDICAL_KEYWORDS = {
    # Common symptoms & conditions
    "asthma", "asthama", "fever", "cough", "cold", "flu", "headache", "migraine", 
    "pain", "ache", "sore", "infection", "rash", "allergy", "allergic", "swelling",
    "nausea", "vomiting", "diarrhea", "constipation", "fatigue", "dizzy", "dizziness",
    "hypertension", "hypotension", "blood pressure", "diabetes", "diabetic", "cancer",
    "heart", "cardiac", "stroke", "seizure", "epilepsy", "arthritis", "pneumonia",
    "bronchitis", "covid", "virus", "bacterial", "sprain", "fracture", "bruise",
    "cramp", "insomnia", "anxiety", "depression", "panic", "eczema", "acne",
    "ulcer", "acid reflux", "gerd", "cholesterol", "thyroid", "anemia", "kidney",
    "liver", "lung", "stomach", "throat", "ear", "eye", "skin", "bone", "muscle",
    
    # Medical terms & queries
    "symptom", "symptoms", "diagnosis", "diagnose", "treatment", "cure", "therapy",
    "medication", "medicine", "drug", "dosage", "pill", "tablet", "antibiotic",
    "prescription", "vaccine", "vaccination", "doctor", "hospital", "clinic",
    "physician", "nurse", "surgery", "operation", "dose", "side effect", "syndrome",
    "disease", "disorder", "condition", "illness", "pregnant", "pregnancy"
}

NON_MEDICAL_KEYWORDS = {
    "python", "javascript", "code", "coding", "program", "programming", "software",
    "script", "html", "css", "java", "c++", "sql", "algorithm", "debug", "compile",
    "essay", "poem", "story", "joke", "weather", "crypto", "bitcoin", "stock market",
    "homework", "movie", "song", "lyrics", "game", "recipe"
}

THIRD_PARTY_PATTERNS = [
    (r"\bmy\s+(friend|buddy|pal|roommate|colleague|coworker)(\s+[a-zA-Z]+)?\b", "friend"),
    (r"\bmy\s+(mom|mother|dad|father|parent|parents)\b", "parent"),
    (r"\bmy\s+(brother|sister|sibling)\b", "sibling"),
    (r"\bmy\s+(son|daughter|child|kid)\b", "child"),
    (r"\bmy\s+(wife|husband|partner|spouse)\b", "partner"),
    (r"\bmy\s+(cousin|uncle|aunt|nephew|niece|grandma|grandmother|grandpa|grandfather)\b", "relative"),
    (r"\b(someone\s+i\s+know|for\s+a\s+friend|asking\s+for\s+a\s+friend)\b", "friend"),
    (r"\babout\s+my\s+(friend|mom|mother|dad|father|brother|sister|son|daughter|wife|husband|partner)\b", "third_party"),
]

COMPILED_THIRD_PARTY_PATTERNS = [(re.compile(p, re.IGNORECASE), label) for p, label in THIRD_PARTY_PATTERNS]


def is_third_party_query(text: str) -> Tuple[bool, Optional[str]]:
    """
    Detects if the query is asked on behalf of another person (third party).
    Returns (is_third_party, relationship_label).
    """
    if not text or not isinstance(text, str):
        return False, None
    for pattern, label in COMPILED_THIRD_PARTY_PATTERNS:
        if pattern.search(text):
            return True, label
    return False, None


def _classify_intent_regex(chatModel, message: str) -> str:
    """
    Legacy regex and keyword-based intent classification.
    Used exclusively in regex mode (INTENT_CLASSIFIER_MODE=regex).
    """
    if not message or not isinstance(message, str):
        return "general_chat"

    msg_clean = message.strip().lower()
    msg_words = set(re.findall(r"\b[a-zA-Z]+\b", msg_clean))

    # 0. Canonical code/script/programming request check (Single Source of Truth)
    if is_code_or_programming_request(message):
        return "general_chat"

    # 1. Anchored Account action detection (Amendment 4 & 8)
    if ANCHORED_ACCOUNT_REGEX.match(msg_clean):
        return "account_action"

    # 2. Greetings (when short)
    greetings = {
        "hi", "hello", "hey", "greetings", "good morning", "good afternoon", 
        "good evening", "howdy", "hola", "yo", "hi there", "hello there"
    }
    if (msg_clean.rstrip("?!.") in greetings or any(msg_clean.startswith(g + " ") for g in greetings)) and len(msg_words) <= 4:
        return "greeting"

    # 3. Anchored Memory recall detection (Amendment 4 & 8)
    if ANCHORED_MEMORY_REGEX.match(msg_clean):
        return "memory_recall"

    # 4. Third-party health / support query detection
    is_tp, _ = is_third_party_query(message)
    if is_tp:
        return "third_party_query"

    # 5. Instant Non-Medical heuristic (evaluated before general medical keywords)
    if any(w in NON_MEDICAL_KEYWORDS for w in msg_words):
        return "general_chat"

    # 6. Instant Medical Query heuristic
    if any(keyword in msg_clean for keyword in MEDICAL_KEYWORDS) or any(w in MEDICAL_KEYWORDS for w in msg_words):
        return "medical_query"

    # 7. LLM Classification fallback for edge cases
    if chatModel:
        prompt = f"""You are a strict intent classifier for MediAssist, a specialized medical assistant.
Classify the user's message into exactly ONE of the following:
- greeting
- medical_query (any health, symptom, body, medicine, or wellness question)
- memory_recall (asking what the assistant remembers about the user)
- account_action (logout, delete account/chat)
- general_chat (any non-medical question, e.g. coding, math, general trivia)

Return ONLY the classification word.

User: {message}"""
        try:
            response = chatModel.invoke(prompt)
            content = (response.content if hasattr(response, "content") else str(response)).strip().lower()
            allowed_intents = ["medical_query", "greeting", "memory_recall", "account_action", "third_party_query", "general_chat"]
            for intent in allowed_intents:
                if intent in content:
                    return intent
            return "general_chat"
        except Exception as e:
            print("[Intent Classifier] Fallback error:", e)

    return "medical_query" if any(w in msg_clean for w in ["help", "feel", "body", "hurt"]) else "general_chat"


def classify_intent(
    chatModel=None,
    message: str = "",
    route_decision: Optional[Any] = None,
    recent_turns: Optional[List[Dict[str, str]]] = None,
    patient_state: Optional[Any] = None
) -> str:
    """
    Classifies or maps user message into legacy intent labels:
    - 'medical_query'
    - 'third_party_query'
    - 'greeting'
    - 'memory_recall'
    - 'account_action'
    - 'general_chat'

    In LLM mode:
    - If route_decision is already computed by route_query(), derives the legacy label
      directly from route_decision with ZERO additional LLM calls.
    - If route_decision is not provided, invokes route_query() preserving context
      (recent_turns, patient_state) and maps the result.
      NEVER calls classify_query(user_text=message) without context!

    In regex mode:
    - Runs _classify_intent_regex only. ZERO Groq calls.
    """
    cfg = get_classifier_config()

    # If route_decision was precomputed, map it directly (Amendment 2: ZERO double LLM calls)
    if route_decision is not None:
        action = getattr(route_decision, "action", "")
        intent = getattr(route_decision, "intent", "")

        if action == "BLOCKED_INJECTION":
            return "general_chat"
        if action == "BLOCKED_EMERGENCY":
            return "medical_query"
        if action == "ACCOUNT_ACTION" or intent == "account_action":
            return "account_action"
        if action == "MEMORY_RECALL" or intent == "memory_recall":
            return "memory_recall"
        if intent == "NON_MEDICAL" or action == "NON_MEDICAL":
            return "general_chat"
        if intent == "GREETING" or action == "GREETING":
            return "greeting"

        # Medical intent: check for third-party sub-route
        is_tp, _ = is_third_party_query(message)
        if is_tp:
            return "third_party_query"
        return "medical_query"

    # When route_decision is not precomputed (e.g. legacy caller / test):
    if cfg.mode == "regex":
        return _classify_intent_regex(chatModel, message)

    # In LLM mode without precomputed route_decision:
    # Delegate to route_query preserving all available context
    from services.routing_service import route_query
    decision = route_query(
        msg=message,
        patient_state=patient_state,
        recent_turns=recent_turns
    )
    return classify_intent(
        chatModel=chatModel,
        message=message,
        route_decision=decision,
        recent_turns=recent_turns,
        patient_state=patient_state
    )


