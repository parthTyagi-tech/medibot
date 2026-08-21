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
    StopResponse,
)
from livekit.agents import llm, room_io, ModelSettings, TurnHandlingOptions
from livekit.plugins import deepgram, groq, silero

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("medical-agent")
logger.setLevel(logging.INFO)

_port = os.getenv("PORT", "5050").strip()
BACKEND_URL = os.getenv("VOICE_BACKEND_URL", f"http://127.0.0.1:{_port}").rstrip("/")

_REQUIRED_ENV = (
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "DEEPGRAM_API_KEY",
    "GROQ_API_KEY",
)

def is_prompt_injection(user_input: str) -> bool:
    if not user_input:
        return False
    text = user_input.lower()
    blocked_phrases = [
        "ignore previous", "ignore all previous", "system override",
        "developer mode", "you are no longer", "disregard instructions",
        "new persona", "system prompt", "forget everything",
        "bypass rules", "do not follow", "dan (", "dan mode"
    ]
    for phrase in blocked_phrases:
        if phrase in text:
            return True
    return False

# ==============================================================================

LANGUAGE_CONFIGS = {
    "en": {
        "stt_lang": "en-US",
        "stt_model": "nova-2",
        "tts_model": "aura-2-thalia-en",
        "system_instruction": "You are MediAssist, a medical assistant. Keep answers concise — they will be spoken aloud. Answer in English.",
        "greeting": "Hello, I am MediAssist. How can I help you today?",
    },
    "es": {
        "stt_lang": "es",
        "stt_model": "nova-2",
        "tts_model": "aura-2-celeste-es",
        "system_instruction": "Eres MediAssist, un asistente médico. Mantén las respuestas concisas: se leerán en voz alta. Responde en español.",
        "greeting": "Hola, soy MediAssist. ¿Cómo puedo ayudarte hoy?",
    },
    "fr": {
        "stt_lang": "fr",
        "stt_model": "nova-2",
        "tts_model": "aura-2-agathe-fr",
        "system_instruction": "Vous êtes MediAssist, un assistant médical. Restez concis, les réponses seront lues à haute voix. Répondez en français.",
        "greeting": "Bonjour, je suis MediAssist. Comment puis-je vous aider aujourd'hui?",
    },
    "de": {
        "stt_lang": "de",
        "stt_model": "nova-2",
        "tts_model": "aura-2-aurelia-de",
        "system_instruction": "Sie sind MediAssist, ein medizinischer Assistent. Fassen Sie sich kurz – die Antworten werden laut vorgelesen. Antworten Sie auf Deutsch.",
        "greeting": "Hallo, ich bin MediAssist. Wie kann ich Ihnen heute helfen?",
    },
    "it": {
        "stt_lang": "it",
        "stt_model": "nova-2",
        "tts_model": "aura-2-livia-it",
        "system_instruction": "Sei MediAssist, un assistente medico. Mantieni le risposte concise: verranno lette ad alta voce. Rispondi in italiano.",
        "greeting": "Ciao, sono MediAssist. Come posso aiutarti oggi?",
    },
    "nl": {
        "stt_lang": "nl",
        "stt_model": "nova-2",
        "tts_model": "aura-2-rhea-nl",
        "system_instruction": "Je bent MediAssist, een medische assistent. Houd antwoorden beknopt - ze worden hardop voorgelezen. Antwoord in het Nederlands.",
        "greeting": "Hallo, ik ben MediAssist. Hoe kan ik je vandaag helpen?",
    },
    "ja": {
        "stt_lang": "ja",
        "stt_model": "nova-2",
        "tts_model": "aura-2-izanami-ja",
        "system_instruction": "あなたは医療アシスタントのMediAssistです。回答は簡潔にしてください。音声で読み上げられます。日本語で回答してください。",
        "greeting": "こんにちは、MediAssistです。本日はどのようなご用件でしょうか？",
    }
}


def _normalize_livekit_url() -> None:
    url = os.getenv("LIVEKIT_URL", "").strip()
    if url.startswith("https://"):
        os.environ["LIVEKIT_URL"] = "wss://" + url[8:]
    elif url.startswith("http://"):
        os.environ["LIVEKIT_URL"] = "ws://" + url[7:]


def _validate_env() -> None:
    _normalize_livekit_url()
    missing = [name for name in _REQUIRED_ENV if not os.getenv(name, "").strip()]
    if missing:
        msg = (
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Please set them in Render Environment Variables or your .env file."
        )
        logger.error(f"[voice_worker] ERROR: {msg}")
        raise RuntimeError(msg)


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load(
        min_speech_duration=0.25,
        min_silence_duration=0.5,
        prefix_padding_duration=0.2,
    )


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
        urls_to_try = [BACKEND_URL]
        _port = os.getenv("PORT", "5050").strip()
        fallback_url = f"http://127.0.0.1:{_port}"
        if fallback_url not in urls_to_try:
            urls_to_try.append(fallback_url)

        last_error = None
        for url in urls_to_try:
            try:
                logger.info(f"[ask_backend] Calling {url}/voice_chat with message='{message[:60]}...'")
                async with aiohttp.ClientSession() as http:
                    payload = {"message": message}
                    if user_id:
                        payload["user_id"] = user_id
                    async with http.post(
                        f"{url}/voice_chat",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            logger.error(f"[ask_backend] {url}/voice_chat HTTP {resp.status}: {text[:200]}")
                            continue
                        data = await resp.json()
                        answer = data.get("response", "Sorry, I couldn't get an answer.")
                        logger.info(f"[ask_backend] Got response ({len(answer)} chars): '{answer[:80]}...'")
                        return answer
            except asyncio.TimeoutError:
                last_error = "HTTP request timed out after 30s"
                logger.error(f"[ask_backend] Timeout calling {url}/voice_chat (30s)")
            except Exception as e:
                last_error = e
                logger.error(f"[ask_backend] Exception calling {url}/voice_chat: {e}")

        logger.error(f"[ask_backend] All backend URLs exhausted. Last error: {last_error}")
        return "Sorry, I couldn't reach the medical backend."

    async def stream_backend(self, message: str, user_id: str | None = None):
        urls_to_try = [BACKEND_URL]
        _port = os.getenv("PORT", "5050").strip()
        fallback_url = f"http://127.0.0.1:{_port}"
        if fallback_url not in urls_to_try:
            urls_to_try.append(fallback_url)

        for url in urls_to_try:
            try:
                async with aiohttp.ClientSession() as http:
                    payload = {"message": message, "stream": True}
                    if user_id:
                        payload["user_id"] = user_id
                    async with http.post(
                        f"{url}/voice_chat",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            logger.error(f"[stream_backend] HTTP {resp.status}: {text[:200]}")
                            continue

                        async for chunk, _ in resp.content.iter_chunks():
                            if chunk:
                                yield chunk.decode("utf-8")
                        return
            except Exception as e:
                logger.error(f"[stream_backend] Request to {url} failed: {e}")
        yield "Sorry, I couldn't reach the medical backend."

    async def send_text_to_room(
        self,
        text: str,
        role: str = "assistant",
    ) -> None:
        try:
            payload = json.dumps({
                "type": "transcript",
                "role": role,
                "text": text,
            }).encode("utf-8")

            user_id = self._user_id()
            dest = [user_id] if user_id else []

            await self._room().local_participant.publish_data(
                payload,
                reliable=True,
                topic="mediassist.transcript",
                destination_identities=dest,
            )
        except Exception as e:
            logger.warning(f"Failed to publish transcript data to room: {e}")

    async def process_user_prompt(self, user_text: str) -> None:
        user_text = (user_text or "").strip()
        if not user_text:
            return

        logger.info(f"[process_user_prompt] START — '{user_text}'")
        try:
            logger.info(f"[process_user_prompt] Step 1: Publishing user transcript to room")
            await self.send_text_to_room(user_text, role="user")

            if is_prompt_injection(user_text):
                bot_response = "I cannot fulfill this request. I am a medical AI assistant, and my instructions cannot be overridden."
                logger.info(f"[process_user_prompt] Step 2: Prompt injection detected, using refusal")
            else:
                logger.info(f"[process_user_prompt] Step 2: Calling ask_backend...")
                bot_response = await self.ask_backend(user_text, user_id=self._user_id())
                logger.info(f"[process_user_prompt] Step 2: Backend returned ({len(bot_response)} chars)")

            logger.info(f"[process_user_prompt] Step 3: Publishing bot transcript to room")
            await self.send_text_to_room(bot_response, role="assistant")

            logger.info(f"[process_user_prompt] Step 4: Calling session.say() for TTS...")
            await self.session.say(bot_response, allow_interruptions=True)
            logger.info(f"[process_user_prompt] Step 5: session.say() COMPLETED — turn finished")

        except Exception as e:
            logger.error(f"[process_user_prompt] EXCEPTION for '{user_text}': {e}", exc_info=True)
            try:
                fallback_msg = "Sorry, I encountered an issue generating a response."
                await self.session.say(fallback_msg, allow_interruptions=True)
            except Exception as inner_e:
                logger.error(f"[process_user_prompt] Fallback TTS also failed: {inner_e}")

    async def on_enter(self) -> None:
        logger.info("AGENT ENTERED ROOM")
        room = self._room()

        # Try to find user identity from remote participants immediately, or wait up to 3s
        participant = None
        for p in room.remote_participants.values():
            participant = p
            break

        if not participant:
            try:
                for _ in range(30):
                    await asyncio.sleep(0.1)
                    for p in room.remote_participants.values():
                        participant = p
                        break
                    if participant:
                        break
            except Exception:
                pass

        if participant:
            self._user_identity = participant.identity
            logger.info(f"USER JOINED ROOM: {participant.identity}")
        else:
            logger.warning("No remote participant detected on enter")

        # Listen for client data packets (typed text or prompt clicks in voice mode)
        @self._room().on("data_received")
        def _on_data(data_packet):
            try:
                payload = json.loads(data_packet.data.decode("utf-8"))
                if payload.get("type") == "user_text" and payload.get("text"):
                    asyncio.create_task(self.process_user_prompt(payload["text"]))
            except Exception:
                pass

        await self.on_session_start()

    async def on_session_start(self) -> None:
        logger.info("VOICE SESSION STARTED")
        await self.send_text_to_room(self.greeting, role="assistant")
        await self.session.say(self.greeting, allow_interruptions=True)

    async def on_user_turn_completed(
        self,
        turn_ctx: llm.ChatContext,
        new_message: llm.ChatMessage,
    ) -> None:
        user_text = new_message.text_content or ""
        logger.info(f"[on_user_turn_completed] Received turn: '{user_text}'")
        if user_text:
            # CRITICAL: Await process_user_prompt so session.say() completes
            # BEFORE the turn system moves to the next turn. Fire-and-forget
            # (create_task) causes the framework to accept the next turn while
            # TTS is still playing, silently dropping all subsequent messages.
            await self.process_user_prompt(user_text)
        raise StopResponse()


async def entrypoint(ctx: JobContext):
    logger.info(f"ENTRYPOINT STARTED - ROOM: {ctx.room.name}")

    ctx.log_context_fields = {"room": ctx.room.name}

    _install_room_stream_handlers(ctx.room)

    vad = ctx.proc.userdata.get("vad")
    if vad is None:
        logger.info("Loading VAD in job (prewarm miss)...")
        vad = silero.VAD.load(
            min_speech_duration=0.25,
            min_silence_duration=0.5,
            prefix_padding_duration=0.2,
        )

    # Parse language from job metadata (default to 'en')
    lang = (ctx.job.metadata or "en").strip().lower()
    if lang not in LANGUAGE_CONFIGS:
        lang = "en"
    
    config = LANGUAGE_CONFIGS[lang]
    logger.info(f"Configuring voice channel for language: {lang}")

    session = AgentSession(
        stt=deepgram.STT(model=config["stt_model"], language=config["stt_lang"]),
        vad=vad,
        tts=deepgram.TTS(model=config["tts_model"]),
        aec_warmup_duration=2.0,
        turn_handling=TurnHandlingOptions(
            endpointing={"min_delay": 0.5, "max_delay": 5.0},
        ),
    )

    @session.on("user_input_transcribed")
    def _on_transcribed(ev) -> None:
        suffix = " (final)" if ev.is_final else ""
        logger.info(f"STT{suffix}: {ev.transcript}")

    @session.on("agent_state_changed")
    def _on_agent_state(ev) -> None:
        logger.info(f"AGENT STATE: {ev.old_state} -> {ev.new_state}")

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