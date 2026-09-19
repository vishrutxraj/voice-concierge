"""
AudioTransport -- one physical connection to a caller, direction-agnostic.

call_loop.py is written entirely against this ABC. Swapping the browser-mic
WebSocket implementation for a Twilio media-stream one (README's stated next
step once a live number exists) means writing a new subclass, not touching the
orchestration logic that already calls ASRClient/TTSClient/run_turn correctly.

Each inbound "utterance" is a complete audio blob, not a raw byte stream --
this matches ASRClient.transcribe's contract exactly (it takes one full
recording, real Sarvam or mock) and pushes any real VAD/chunking decision to
the transport implementation, where the wire format actually lives. Phase 4's
WebSocketAudioTransport treats one inbound binary frame as one utterance,
which is the simplest contract a browser client can implement (push-to-talk or
client-side silence detection); it is not a statement that server-side VAD is
solved.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TransportClosed(Exception):
    """Raised by any AudioTransport method once the caller has disconnected."""


AUDIO_PREFERENCES = ("native", "english", "both")


class AudioTransport(ABC):
    # Which reply audio the caller wants to HEAR: their own language, English,
    # or both back to back. Text always carries both (reply event's `text` and
    # `text_en`); this only decides which TTS renderings get synthesized and
    # streamed, so an unwanted one costs no TTS call and no bandwidth. Read by
    # call_loop at delivery time, so a mid-call change applies to the next reply.
    audio_preference: str = "native"

    @abstractmethod
    async def receive_utterance(self) -> bytes | None:
        """
        Block until one complete inbound utterance is available.

        Returns None on a clean hangup. Raises TransportClosed on an abrupt
        disconnect -- callers should treat both as "end the call loop", the
        distinction exists only for logging.
        """

    @abstractmethod
    async def send_event(self, event: dict[str, Any]) -> None:
        """Send one JSON control/status message (transcript, reply text,
        awaiting_confirmation, barge_in, ...)."""

    @abstractmethod
    async def send_audio_chunk(self, chunk: bytes) -> None:
        """Send one chunk of outbound (synthesized) audio."""
