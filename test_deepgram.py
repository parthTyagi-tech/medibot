from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("DEEPGRAM_API_KEY")

print("API Key Found:", bool(api_key))

if api_key:
    print("First 8 chars:", api_key[:8])