def classify_intent(chatModel, message):
    # 1. Local heuristics for fast execution (0ms latency)
    msg_clean = message.strip().lower().rstrip("?!.")
    
    # Simple greeting detection
    greetings = {
        "hi", "hello", "hey", "greetings", "good morning", "good afternoon", 
        "good evening", "howdy", "hola", "yo", "hi there", "hello there"
    }
    if msg_clean in greetings or any(msg_clean.startswith(g + " ") for g in greetings):
        if len(msg_clean.split()) <= 3:
            return "greeting"

    # Simple account action detection
    account_keywords = {
        "delete my account", "delete this consultation", "logout", "log out", 
        "sign out", "signout", "delete account", "delete chat"
    }
    if any(keyword in msg_clean for keyword in account_keywords):
        return "account_action"

    # 2. LLM Classification fallback
    prompt = f"""
Classify the user's intent.

Choose ONLY one:
- greeting
- medical_query
- memory_recall
- account_action
- general_chat

Examples:

Hi
Hello
Good morning
-> greeting

I have fever
I have cough
My stomach hurts
-> medical_query

What symptoms did I mention before?
What do you know about me?
What was my previous diagnosis?
-> memory_recall

Delete my account
Delete this consultation
Logout
-> account_action

Tell me a joke
Who are you?
What is AI?
-> general_chat

Return ONLY the intent word, nothing else.

User:
{message}
"""

    try:
        response = chatModel.invoke(prompt)
        content = response.content.strip().lower()
        
        # Robust substring matching to find the exact intent
        allowed_intents = ["greeting", "medical_query", "memory_recall", "account_action", "general_chat"]
        for intent in allowed_intents:
            if intent in content:
                return intent
                
        return "general_chat" # fallback
    except Exception as e:
        print("Intent classification error:", e)
        return "general_chat"
    
# speech to text where the model is appleciable or not ? else what we have to say ??
