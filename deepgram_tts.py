import uuid
import glob
import time
from deepgram import DeepgramClient
from dotenv import load_dotenv
import os

load_dotenv()

deepgram = DeepgramClient(
    api_key=os.getenv("DEEPGRAM_API_KEY")
)

def text_to_speech(text):
    os.makedirs("static/audio", exist_ok=True)

    # Clean up audio files older than 5 minutes to prevent disk bloat
    now = time.time()
    for f in glob.glob("static/audio/*.mp3"):
        if os.stat(f).st_mtime < now - 300:
            try:
                os.remove(f)
            except OSError:
                pass

    filename = f"static/audio/{uuid.uuid4()}.mp3"

    audio_generator = deepgram.speak.v1.audio.generate(
        text=text,
        model="aura-2-thalia-en",
        encoding="mp3"
    )

    with open(filename, "wb") as f:
        for chunk in audio_generator:
            f.write(chunk)

    return filename