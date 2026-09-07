"""
/ws/call wire-protocol tests, end-to-end through FastAPI's real WebSocket
stack (starlette.testclient.TestClient) rather than calling call_loop
directly -- these exist to catch handshake/routing/wiring bugs that a unit
test of app.audio.call_loop can't see (wrong endpoint path, a bad handshake
response, main.py wiring the wrong transport class, etc).

Each test always fully drains one turn's audio (reads to "audio_end") before
sending the next utterance, so there is no barge-in race to reason about here
-- that concurrency behaviour is covered deterministically in
test_audio_call_loop.py instead.
"""

from __future__ import annotations

import json

import pytest
from app.main import app
from app.providers.asr_client import reset_asr_client
from app.providers.cache import reset_caches
from app.providers.llm_client import reset_llm_client
from app.providers.order_client import InProcessOrderClient, reset_order_client
from app.providers.tts_client import reset_tts_client
from fastapi.testclient import TestClient
from order_api.kv import reset_backend
from order_api.seed import build_fixtures
from order_api.store import get_store
from starlette.websockets import WebSocketDisconnect

RAVI = "9990000002"  # one open order: DLV1004 (FAILED_ATTEMPT)


@pytest.fixture(autouse=True)
def fresh_env(tmp_path, monkeypatch):
    reset_backend()
    reset_order_client()
    reset_llm_client()
    reset_asr_client()
    reset_tts_client()
    reset_caches()
    import order_api.store as store_mod

    store_mod._store = None
    store = get_store()
    store.reset()
    store.load(build_fixtures())

    monkeypatch.setenv("CHECKPOINT_DSN", f"sqlite:///{tmp_path / 'checkpoints.db'}")
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    from app.config import get_settings

    get_settings.cache_clear()

    yield store

    get_settings.cache_clear()
    reset_caches()


def _read_until_control(ws) -> tuple[list[bytes], dict]:
    """Read raw frames until a text (JSON control) frame arrives, returning
    any binary audio chunks seen first alongside the parsed control event."""
    chunks: list[bytes] = []
    while True:
        message = ws.receive()
        if "text" in message:
            return chunks, json.loads(message["text"])
        chunks.append(message["bytes"])


def test_order_lookup_round_trip():
    with TestClient(app) as client, client.websocket_connect("/ws/call") as ws:
        ws.send_json({"type": "start", "session_id": "ws1", "caller_phone": RAVI})
        ready = ws.receive_json()
        assert ready == {"type": "ready", "session_id": "ws1"}

        ws.send_bytes(b"where is my order")

        _, transcript = _read_until_control(ws)
        assert transcript == {
            "type": "transcript", "text": "where is my order", "language": "en-IN",
        }

        _, reply = _read_until_control(ws)
        assert reply["type"] == "reply"
        assert reply["intent"] == "order_lookup"
        assert "delivery attempt" in reply["text"].lower()
        assert reply["awaiting_confirmation"] is None

        chunks, end = _read_until_control(ws)
        assert end == {"type": "audio_end"}
        assert len(chunks) > 1, "reply audio must be streamed, not sent as one blob"
        audio = b"".join(chunks)
        assert audio[:4] == b"RIFF"

        ws.send_json({"type": "hangup"})


def test_reschedule_confirm_and_commit_round_trip():
    with TestClient(app) as client, client.websocket_connect("/ws/call") as ws:
        ws.send_json({"type": "start", "session_id": "ws2", "caller_phone": RAVI})
        ws.receive_json()

        ws.send_bytes(b"reschedule to in 3 days evening")
        _read_until_control(ws)  # transcript
        _, proposal = _read_until_control(ws)
        assert proposal["awaiting_confirmation"]["kind"] == "confirm_reschedule"
        _read_until_control(ws)  # audio_end for the read-back

        ws.send_bytes(b"yes")
        _read_until_control(ws)  # transcript "yes"
        _, confirmation = _read_until_control(ws)
        assert confirmation["awaiting_confirmation"] is None
        _read_until_control(ws)  # audio_end

        ws.send_json({"type": "hangup"})

    order = InProcessOrderClient().get_order("DLV1004")
    assert order.reschedule_count == 1


def test_reschedule_decline_round_trip_does_not_commit():
    with TestClient(app) as client, client.websocket_connect("/ws/call") as ws:
        ws.send_json({"type": "start", "session_id": "ws3", "caller_phone": RAVI})
        ws.receive_json()

        ws.send_bytes(b"reschedule to in 3 days evening")
        _read_until_control(ws)
        _read_until_control(ws)
        _read_until_control(ws)  # audio_end

        ws.send_bytes(b"no")
        _read_until_control(ws)  # transcript
        _, declined = _read_until_control(ws)
        assert declined["awaiting_confirmation"] is None
        assert "better" in declined["text"].lower() or "date" in declined["text"].lower()

        ws.send_json({"type": "hangup"})

    order = InProcessOrderClient().get_order("DLV1004")
    assert order.reschedule_count == 0


def test_unclear_confirmation_reply_reasks():
    with TestClient(app) as client, client.websocket_connect("/ws/call") as ws:
        ws.send_json({"type": "start", "session_id": "ws4", "caller_phone": RAVI})
        ws.receive_json()

        ws.send_bytes(b"reschedule to in 3 days evening")
        _read_until_control(ws)
        _read_until_control(ws)
        _read_until_control(ws)

        ws.send_bytes(b"maybe later")
        _read_until_control(ws)  # transcript
        _, reask = _read_until_control(ws)
        assert "yes or a no" in reask["text"].lower()
        assert reask["awaiting_confirmation"]["kind"] == "confirm_reschedule"

        ws.send_json({"type": "hangup"})

    order = InProcessOrderClient().get_order("DLV1004")
    assert order.reschedule_count == 0


def test_missing_start_handshake_closes_connection():
    with TestClient(app) as client, client.websocket_connect("/ws/call") as ws:
        ws.send_json({"type": "not-a-start"})
        with pytest.raises(WebSocketDisconnect) as excinfo:
            ws.receive_json()
        assert excinfo.value.code == 1008


def test_session_id_is_generated_when_omitted():
    with TestClient(app) as client, client.websocket_connect("/ws/call") as ws:
        ws.send_json({"type": "start", "caller_phone": RAVI})
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert isinstance(ready["session_id"], str) and ready["session_id"]
        ws.send_json({"type": "hangup"})
