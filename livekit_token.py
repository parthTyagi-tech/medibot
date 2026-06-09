import os
from typing import Optional

from dotenv import load_dotenv
from livekit.api import AccessToken, VideoGrants
from livekit.protocol.agent_dispatch import RoomAgentDispatch
from livekit.protocol.room import RoomConfiguration

load_dotenv()

AGENT_NAME = "medical-agent"


def voice_room_name(user_id: int | str) -> str:
    return f"mediassist-voice-{user_id}"


def generate_livekit_token(
    room_name: str,
    participant_identity: str,
    participant_name: str = "User",
    *,
    dispatch_agent: bool = True,
    agent_metadata: str = "",
) -> str:
    api_key = os.getenv("LIVEKIT_API_KEY", "").strip()
    api_secret = os.getenv("LIVEKIT_API_SECRET", "").strip()

    if not api_key:
        raise ValueError("LIVEKIT_API_KEY is missing from .env")
    if not api_secret:
        raise ValueError("LIVEKIT_API_SECRET is missing from .env")

    token = (
        AccessToken(api_key, api_secret)
        .with_identity(str(participant_identity))
        .with_name(str(participant_name))
        .with_grants(
            VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
    )

    if dispatch_agent:
        token = token.with_room_config(
            RoomConfiguration(
                agents=[
                    RoomAgentDispatch(
                        agent_name=AGENT_NAME,
                        metadata=agent_metadata or str(participant_identity),
                    )
                ]
            )
        )

    return token.to_jwt()
