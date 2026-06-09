from deepgram import DeepgramClient
from dotenv import load_dotenv
import os

load_dotenv()

deepgram = DeepgramClient(
    api_key=os.getenv("DEEPGRAM_API_KEY")
)

def text_to_speech(text):

    filename = "static/output.mp3"

    audio_generator = deepgram.speak.v1.audio.generate(
        text=text,
        model="aura-2-thalia-en",
        encoding="mp3"
    )

    with open(filename, "wb") as f:
        for chunk in audio_generator:
            f.write(chunk)

    return filename