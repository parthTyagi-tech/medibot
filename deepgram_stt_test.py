import os
from dotenv import load_dotenv
from deepgram import DeepgramClient

load_dotenv()

deepgram = DeepgramClient(
    api_key=os.getenv("DEEPGRAM_API_KEY")
)

print("Deepgram STT Ready")