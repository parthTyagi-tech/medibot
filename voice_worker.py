from __future__ import annotations

import os
import sys
import json
import asyncio
import logging
import aiohttp
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv

from livekit import rtc

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    get_job_context,
)
from livekit.agents import llm, room_io, ModelSettings, TurnHandlingOptions
from livekit.plugins import deepgram, groq, silero

load_dotenv()

_port = os.getenv("PORT", "5050").strip()
BACKEND_URL = os.getenv("VOICE_BACKEND_URL", f"http://127.0.0.1:{_port}").rstrip("/")

_REQUIRED_ENV = (
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "DEEPGRAM_API_KEY",
    "GROQ_API_KEY",
)

LANGUAGE_CONFIGS = {
    "en": {
        "stt_lang": "en-US",
        "stt_model": "nova-3",
        "tts_model": "aura-2-thalia-en",
        "system_instruction": "You are MediAssist, a medical assistant. Keep answers concise — they will be spoken aloud. Answer in English.",
        "greeting": "Hello, I am MediAssist. How can I help you today?",
    },
    "es": {
        "stt_lang": "es",
        "stt_model": "nova-2",
        "tts_model": "aura-talia-es",
        "system_instruction": "Eres MediAssist, un asistente médico. Mantén las respuestas concisas: se leerán en voz alta. Responde en español.",
        "greeting": "Hola, soy MediAssist. ¿Cómo puedo ayudarte hoy?",
    },
    "fr": {
        "stt_lang": "fr",
        "stt_model": "nova-2",
        "tts_model": "aura-talia-fr",
        "system_instruction": "Vous êtes MediAssist, un assistant médical. Restez concis, les réponses seront lues à haute voix. Répondez en français.",
        "greeting": "Bonjour, je suis MediAssist. Comment puis-je vous aider aujourd'hui?",
    },
    "de": {
        "stt_lang": "de",
        "stt_model": "nova-2",
        "tts_model": "aura-talia-de",
        "system_instruction": "Sie sind MediAssist, ein medizinischer Assistent. Fassen Sie sich kurz – die Antworten werden laut vorgelesen. Antworten Sie auf Deutsch.",
        "greeting": "Hallo, ich bin MediAssist. Wie kann ich Ihnen heute helfen?",
    },
    "it": {
        "stt_lang": "it",
        "stt_model": "nova-2",
        "tts_model": "aura-talia-it",
        "system_instruction": "Sei MediAssist, un assistente medico. Mantieni le risposte concise: verranno lette ad alta voce. Rispondi in italiano.",
        "greeting": "Ciao, sono MediAssist. Come posso aiutarti oggi?",
    },
    "pt": {
        "stt_lang": "pt",
        "stt_model": "nova-2",
        "tts_model": "aura-talia-pt",
        "system_instruction": "Você é o MediAssist, um assistente médico. Seja conciso — as respostas serão lidas em voz alta. Resposta em português.",
        "greeting": "Olá, eu sou o MediAssist. Como posso ajudar você hoje?",
    },
    "hi": {
        "stt_lang": "hi",
        "stt_model": "nova-2",
        "tts_model": "aura-talia-hi",
        "system_instruction": "आप MediAssist हैं, एक चिकित्सा सहायक। अपने उत्तरों को संक्षिप्त रखें - उन्हें जोर से बोला जाएगा। हिंदी में उत्तर दें।",
        "greeting": "नमस्ते, मैं MediAssist हूँ। आज मैं आपकी क्या मदद कर सकता हूँ?",
    },
    "ja": {
        "stt_lang": "ja",
        "stt_model": "nova-2",
        "tts_model": "aura-talia-ja",
        "system_instruction": "あなたは医療アシスタントのMediAssistです。回答は簡潔にしてください。音声で読み上げられます。日本語で回答してください。",
        "greeting": "こんにちは、MediAssistです。本日はどのようなご用件でしょうか？",
    }
}


def _validate_env() -> None:
    missing = [name for name in _REQUIRED_ENV if not os.getenv(name, "").strip()]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Add them to your .env file."
        )


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


def _install_room_stream_handlers(room: rtc.Room) -> None:
    """Prevent 'no callback attached' noise; SessionHost may replace these later."""
    orig_byte = room.register_byte_stream_handler
    orig_text = room.register_text_stream_handler

    def _byte(topic: str, handler: Callable[[rtc.ByteStreamReader, str], Any]) -> None:
        byte_handlers = getattr(room, "_byte_stream_handlers", None)
        if byte_handlers is not None and byte_handlers.get(topic) is not None:
            byte_handlers[topic] = handler
            return
        orig_byte(topic, handler)

    def _text(topic: str, handler: Callable[[rtc.TextStreamReader, str], Any]) -> None:
        text_handlers = getattr(room, "_text_stream_handlers", None)
        if text_handlers is not None and text_handlers.get(topic) is not None:
            text_handlers[topic] = handler
            return
        orig_text(topic, handler)

    room.register_byte_stream_handler = _byte  # type: ignore[method-assign]
    room.register_text_stream_handler = _text  # type: ignore[method-assign]

    async def _drain_byte(reader: rtc.ByteStreamReader, _identity: str) -> None:
        try:
            async for _ in reader:
                pass
        except Exception:
            pass

    async def _drain_text(reader: rtc.TextStreamReader, _identity: str) -> None:
        try:
            async for _ in reader:
                pass
        except Exception:
            pass

    room.register_byte_stream_handler(
        "lk.agent.session",
        lambda reader, identity: asyncio.create_task(_drain_byte(reader, identity)),
    )
    room.register_text_stream_handler(
        "lk.transcription",
        lambda reader, identity: asyncio.create_task(_drain_text(reader, identity)),
    )


logging.getLogger("charset_normalizer").setLevel(logging.WARNING)
logging.getLogger("livekit.plugins.deepgram").setLevel(logging.INFO)


class MedicalAgent(Agent):

    def __init__(self, instructions: str, greeting: str):
        super().__init__(instructions=instructions)
        self.greeting = greeting
        self._user_identity: str | None = None

    def _room(self) -> rtc.Room:
        return self.session.room_io.room

    def _user_id(self) -> str | None:
        if self._user_identity:
            return self._user_identity
        room = self._room()
        for identity in room.remote_participants.keys():
            self._user_identity = identity
            return identity
        return None

    async def ask_backend(self, message: str, user_id: str | None = None) -> str:
        try:
            async with aiohttp.ClientSession() as http:
                payload = {"message": message}
                if user_id:
                    payload["user_id"] = user_id
                async with http.post(
                    f"{BACKEND_URL}/voice_chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=90),
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        print(f"Backend HTTP {resp.status}: {text[:200]}")
                        return "Sorry, the medical backend returned an error."
                    data = await resp.json()
                    return data.get(
                        "response",
                        "Sorry, I couldn't get an answer.",
                    )
        except Exception as e:
            print(f"Backend error: {e}")
            return (
                "Sorry, I couldn't reach the medical backend. "
                "Please make sure the Flask app is running on port 5050."
            )

    async def stream_backend(self, message: str, user_id: str | None = None):
        try:
            async with aiohttp.ClientSession() as http:
                payload = {"message": message, "stream": True}
                if user_id:
                    payload["user_id"] = user_id
                async with http.post(
                    f"{BACKEND_URL}/voice_chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=90),
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        print(f"Backend HTTP {resp.status}: {text[:200]}")
                        yield "Sorry, the medical backend returned an error."
                        return

                    async for chunk, _ in resp.content.iter_chunks():
                        if chunk:
                            yield chunk.decode("utf-8")
        except Exception as e:
            print(f"Backend stream error: {e}")
            yield "Sorry, I couldn't reach the medical backend."

    async def send_text_to_room(
        self,
        text: str,
        role: str = "assistant",
    ) -> None:
        payload = json.dumps({
            "type": "transcript",
            "role": role,
            "text": text,
        }).encode("utf-8")

        dest = []
        user_id = self._user_id()
        if user_id:
            dest = [user_id]

        await self._room().local_participant.publish_data(
            payload,
            reliable=True,
            topic="mediassist.transcript",
            destination_identities=dest,
        )

    async def on_enter(self) -> None:
        print("AGENT ENTERED")
        ctx = get_job_context()
        try:
            participant = await ctx.wait_for_participant()
            self._user_identity = participant.identity
            print(f"USER JOINED: {participant.identity}")
        except Exception as e:
            print(f"wait_for_participant warning: {e}")

        # ✅ FIX: Call directly instead of registering as async .on() callback.
        # LiveKit's .on() does not accept async functions — on_enter is already
        # async so we can await on_session_start() here directly.
        await self.on_session_start()

    async def on_session_start(self) -> None:
        print("SESSION STARTED")
        await self.send_text_to_room(self.greeting, role="assistant")
        await self.session.say(self.greeting, allow_interruptions=True)

    async def on_user_turn_completed(
        self,
        turn_ctx: llm.ChatContext,
        new_message: llm.ChatMessage,
    ) -> None:
        user_text = new_message.text_content or ""
        if user_text:
            print(f"USER SAID: {user_text}")
            await self.send_text_to_room(user_text, role="user")

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ):
        user_text = ""
        for item in reversed(chat_ctx.items):
            if isinstance(item, llm.ChatMessage) and item.role == "user":
                user_text = item.text_content or ""
                break

        if not user_text:
            print("WARNING: empty user_text in llm_node")
            return

        print(f"Streaming voice reply for: '{user_text}'")
        full_reply = []
        async for chunk in self.stream_backend(user_text, user_id=self._user_id()):
            yield chunk
            full_reply.append(chunk)

        bot_response = "".join(full_reply)
        print(f"BOT REPLY (full): {bot_response[:120]}...")
        await self.send_text_to_room(bot_response, role="assistant")


async def entrypoint(ctx: JobContext):
    print("=" * 50)
    print("ENTRYPOINT STARTED")
    print("ROOM:", ctx.room.name)
    print("=" * 50)

    ctx.log_context_fields = {"room": ctx.room.name}

    _install_room_stream_handlers(ctx.room)

    vad = ctx.proc.userdata.get("vad")
    if vad is None:
        print("Loading VAD in job (prewarm miss)...")
        vad = silero.VAD.load()

    # Parse language from job metadata (default to 'en')
    lang = (ctx.job.metadata or "en").strip().lower()
    if lang not in LANGUAGE_CONFIGS:
        lang = "en"
    
    config = LANGUAGE_CONFIGS[lang]
    print(f"Configuring voice channel for language: {lang}")

    session = AgentSession(
        stt=deepgram.STT(model=config["stt_model"], language=config["stt_lang"]),
        vad=vad,
        llm=groq.LLM(model="llama-3.3-70b-versatile"),
        tts=deepgram.TTS(model=config["tts_model"]),
        aec_warmup_duration=2.0,
        turn_handling=TurnHandlingOptions(
            endpointing={"min_delay": 0.5, "max_delay": 5.0},
        ),
    )

    @session.on("user_input_transcribed")
    def _on_transcribed(ev) -> None:
        suffix = " (final)" if ev.is_final else ""
        print(f"STT{suffix}: {ev.transcript}")

    @session.on("agent_state_changed")
    def _on_agent_state(ev) -> None:
        print(f"AGENT STATE: {ev.old_state} -> {ev.new_state}")

    await session.start(
        room=ctx.room,
        agent=MedicalAgent(
            instructions=config["system_instruction"],
            greeting=config["greeting"]
        ),
        record=False,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
            audio_output=room_io.AudioOutputOptions(),
            text_output=False,
        ),
    )


if __name__ == "__main__":
    _validate_env()

    if (
        sys.platform == "win32"
        and len(sys.argv) > 1
        and sys.argv[1] == "dev"
        and "--no-reload" not in sys.argv
    ):
        sys.argv.append("--no-reload")

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="medical-agent",
            initialize_process_timeout=120.0,
        )
    )