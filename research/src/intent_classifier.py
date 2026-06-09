def classify_intent(chatModel, message):

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

    response = chatModel.invoke(prompt)

    return response.content.strip().lower()