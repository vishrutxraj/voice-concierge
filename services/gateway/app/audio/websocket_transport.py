"""
Browser-mic AudioTransport, over FastAPI's native WebSocket.

Wire protocol (deliberately minimal -- see transport.py for why one binary
frame is one whole utterance rather than a raw stream):

  client -> server
    binary frame   one complete utterance's audio bytes
    {"type": "hangup"}   end the call cleanly

  server -> client
    {"type": ...}        control/status events, see call_loop.py for the set
    binary frame         one chunk of outbound synthesized audio

Any other text frame from the client is ignored rather than treated as an
error -- a browser client evolving its own keepalive/ping messages shouldn't
require a server change here.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from app.audio.transport import AudioTransport

logger = logging.getLogger("concierge.audio")


class WebSocketAudioTransport(AudioTransport):
    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket

    async def receive_utterance(self) -> bytes | None:
        while True:
            try:
                message = await self._ws.receive()
            except WebSocketDisconnect:
                return None

            if message.get("type") == "websocket.disconnect":
                return None

            if (data := message.get("bytes")) is not None:
                return data

            if (text := message.get("text")) is not None:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if payload.get("type") == "hangup":
                    return None
                # Anything else (keepalive, unrecognized control message) is
                # ignored -- keep listening for the next real utterance.
                continue

    async def send_event(self, event: dict[str, Any]) -> None:
        if self._ws.client_state != WebSocketState.CONNECTED:
            return
        try:
            await self._ws.send_json(event)
        except (WebSocketDisconnect, RuntimeError):
            # Caller hung up between our last receive and this send -- the
            # call loop's next receive_utterance() will observe the same
            # disconnect and unwind; nothing to log as an error here.
            pass

    async def send_audio_chunk(self, chunk: bytes) -> None:
        if self._ws.client_state != WebSocketState.CONNECTED:
            return
        try:
            await self._ws.send_bytes(chunk)
        except (WebSocketDisconnect, RuntimeError):
            pass
