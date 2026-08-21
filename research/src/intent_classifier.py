import re

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


def classify_intent(chatModel, message: str) -> str:
    if not message or not isinstance(message, str):
        return "general_chat"

    msg_clean = message.strip().lower()
    msg_words = set(re.findall(r"\b[a-zA-Z]+\b", msg_clean))

    # 1. Account action detection
    account_phrases = [
        "delete my account", "delete this consultation", "logout", "log out", 
        "sign out", "signout", "delete account", "delete chat"
    ]
    if any(phrase in msg_clean for phrase in account_phrases):
        return "account_action"

    # 2. Greetings (when short)
    greetings = {
        "hi", "hello", "hey", "greetings", "good morning", "good afternoon", 
        "good evening", "howdy", "hola", "yo", "hi there", "hello there"
    }
    if (msg_clean.rstrip("?!.") in greetings or any(msg_clean.startswith(g + " ") for g in greetings)) and len(msg_words) <= 4:
        return "greeting"

    # 3. Memory recall detection
    memory_phrases = [
        "what symptoms did i mention", "what do you know about me", "my previous", 
        "what did i tell you", "my diagnosis", "remember what i said", "recall my",
        "what was my allergy", "what medications am i taking", "my past consultations"
    ]
    if any(phrase in msg_clean for phrase in memory_phrases):
        return "memory_recall"

    # 4. Instant Medical Query heuristic
    if any(keyword in msg_clean for keyword in MEDICAL_KEYWORDS) or any(w in MEDICAL_KEYWORDS for w in msg_words):
        return "medical_query"

    # 5. Instant Non-Medical heuristic
    if any(w in NON_MEDICAL_KEYWORDS for w in msg_words):
        return "general_chat"

    # 6. LLM Classification fallback for edge cases
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
        
        allowed_intents = ["medical_query", "greeting", "memory_recall", "account_action", "general_chat"]
        for intent in allowed_intents:
            if intent in content:
                return intent
                
        return "general_chat"
    except Exception as e:
        print("[Intent Classifier] Fallback error:", e)
        # Default to medical_query if any doubt, or general_chat
        return "medical_query" if any(w in msg_clean for w in ["help", "feel", "body", "hurt"]) else "general_chat"

