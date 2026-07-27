def get_user_memory(user):
    """
    Return stored memory for the user.
    """

    if not user:
        return ""

    return user.memory or ""


def update_user_memory(user, chatModel, latest_message, history_text=None):
    """
    Update long-term user memory using the latest message and recent history for context.
    """

    if not user or not latest_message:
        return

    current_memory = user.memory or ""

    context_str = ""
    if history_text:
        context_str = f"Recent Conversation Context:\n{history_text}\n\n"

    memory_prompt = f"""
You are a medical memory manager.

Current User Memory:
{current_memory}

{context_str}Latest User Message:
{latest_message}

Your job is to update the user's long-term memory.

Store ONLY:
- The user's name and personal details
- Symptoms mentioned
- Medical conditions
- Allergies
- Medications
- Health concerns
- Age, weight, blood type if mentioned
- Doctor visits or diagnoses
- Important preferences or context

Do NOT store:
- Greetings or small talk
- Jokes or casual conversation
- Transient details that don't matter long-term

Rules:
- Keep memory concise.
- Remove duplicates.
- Merge related facts.
- Keep under 150 words.
- Write in bullet points.
- If nothing medically relevant, return the current memory unchanged.

Return ONLY the updated memory, nothing else.
"""

    try:

        response = chatModel.invoke(memory_prompt)

        updated_memory = response.content.strip()

        if updated_memory:
            user.memory = updated_memory

    except Exception as e:
        print("Memory Update Error:", e)


def clear_user_memory(user):
    """
    Clear all memory for a user.
    """

    if user:
        user.memory = ""


def memory_exists(user):
    """
    Check whether user already has memory.
    """

    return bool(user and user.memory and user.memory.strip())


def format_memory_for_prompt(user):
    """
    Format memory before injecting into prompts.
    """

    memory = get_user_memory(user)

    if not memory:
        return "No known information about the user."

    return f"""
Known Information About User:

{memory}
"""