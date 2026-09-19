"""
Dual-language output: every reply is available as text in BOTH English and the
caller's language, and the caller picks which one to HEAR (native / english /
both) -- per call over the WebSocket, per turn in the Gradio harness.

Only the audio is a choice. Text always carries both, because it costs
nothing; audio doesn't (each rendering is a TTS call), so an unwanted one is
never synthesized.
"""

from __future__ import annotations

import json

import pytest
from app.audio.call_loop import _CallSession, _deliver
from app.audio.localize import audio_tracks, localize_reply
from app.main import app
from app.providers.asr_client import reset_asr_client
from app.providers.cache import reset_caches
from app.providers.llm_client import reset_llm_client
from app.providers.order_client import reset_order_client
from app.providers.translate_client import reset_translate_client
from app.providers.tts_client import get_tts_client, reset_tts_client
from app.ui.blocks import _format_reply
from app.ui.harness import run_audio_turn, run_text_turn
from fastapi.testclient import TestClient
from order_api.kv import reset_backend
from order_api.seed import build_fixtures
from order_api.store import get_store

from tests.gateway.test_audio_call_loop import RAVI, FakeAudioTransport


@pytest.fixture(autouse=True)
def fresh_env(tmp_path, monkeypatch):
    reset_backend()
    reset_order_client()
    reset_llm_client()
    reset_asr_client()
    reset_tts_client()
    reset_translate_client()
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
    yield
    get_settings.cache_clear()
    reset_caches()
    reset_translate_client()


# ---- audio_tracks: which renderings get spoken ---------------------------------


def _hi_reply():
    return localize_reply("Your order is on its way.", "hi-IN")


def test_tracks_native_only_by_default():
    tracks = audio_tracks(_hi_reply(), "native")
    assert [(t.variant, t.language) for t in tracks] == [("native", "hi-IN")]
    assert tracks[0].text.startswith("[hi-IN]")


def test_tracks_english_only_speaks_the_original_in_en_IN():
    tracks = audio_tracks(_hi_reply(), "english")
    assert [(t.variant, t.language, t.text) for t in tracks] == [
        ("english", "en-IN", "Your order is on its way.")
    ]


def test_tracks_both_plays_native_then_english():
    assert [t.variant for t in audio_tracks(_hi_reply(), "both")] == ["native", "english"]


@pytest.mark.parametrize("pref", ["native", "english", "both"])
def test_untranslated_reply_has_one_track_even_for_both(pref):
    # Must not speak the same English sentence twice.
    tracks = audio_tracks(localize_reply("Hello.", "en-IN"), pref)
    assert [t.variant for t in tracks] == ["english"]


# ---- call loop: audio preference ---------------------------------------------------


async def _deliver_with(pref: str, language: str = "hi-IN") -> FakeAudioTransport:
    transport = FakeAudioTransport()
    transport.audio_preference = pref
    session = _CallSession(
        session_id="pref1", caller_phone=RAVI, transport_kind="text", pinned_language=language
    )
    await _deliver(transport, session, lambda: True, text="Your order is on its way.")
    return transport


async def test_reply_event_carries_both_texts_whatever_the_audio_preference():
    for pref in ("native", "english", "both"):
        reply = (await _deliver_with(pref)).events("reply")[0]
        assert reply["text"] == "[hi-IN] Your order is on its way."
        assert reply["text_en"] == "Your order is on its way."


async def test_native_preference_streams_one_track_with_no_track_marker():
    t = await _deliver_with("native")
    assert t.events("reply")[0]["audio"] == ["native"]
    assert not t.events("audio_track")
    assert t.events("audio_end")


async def test_english_preference_synthesizes_english_not_the_translation():
    t = await _deliver_with("english")
    audio = b"".join(v for k, v in t.log if k == "chunk")
    expected = get_tts_client().synthesize("Your order is on its way.", "en-IN").audio_bytes
    assert t.events("reply")[0]["audio"] == ["english"]
    assert audio == expected


async def test_both_streams_native_then_english_with_markers_and_one_audio_end():
    t = await _deliver_with("both")

    assert t.events("reply")[0]["audio"] == ["native", "english"]
    markers = t.events("audio_track")
    assert [(m["variant"], m["language"]) for m in markers] == [
        ("native", "hi-IN"), ("english", "en-IN"),
    ]
    order = [("chunk" if k == "chunk" else v["type"]) for k, v in t.log]
    collapsed = [x for i, x in enumerate(order) if i == 0 or x != order[i - 1]]
    assert collapsed == ["reply", "audio_track", "chunk", "audio_track", "chunk", "audio_end"]
    assert len(t.events("audio_end")) == 1


async def test_english_caller_with_both_hears_one_track():
    t = await _deliver_with("both", language="en-IN")
    assert t.events("reply")[0]["audio"] == ["english"]
    assert not t.events("audio_track")


# ---- WebSocket wire protocol ----------------------------------------------------------


def _read_until_control(ws):
    chunks = []
    while True:
        message = ws.receive()
        if "text" in message:
            return chunks, json.loads(message["text"])
        chunks.append(message["bytes"])


def _next_of_type(ws, wanted: str) -> dict:
    while True:
        _, event = _read_until_control(ws)
        if event["type"] == wanted:
            return event


def test_ws_start_message_sets_audio_preference_and_reply_carries_both_texts():
    with TestClient(app) as client, client.websocket_connect("/ws/call") as ws:
        ws.send_json({"type": "start", "session_id": "wsl1", "caller_phone": RAVI, "audio": "both"})
        ws.receive_json()
        ws.send_bytes(b"hi-IN::where is my order")

        reply = _next_of_type(ws, "reply")
        assert reply["audio"] == ["native", "english"]
        assert reply["text"].startswith("[hi-IN] ")
        assert "delivery attempt" in reply["text_en"].lower()
        ws.send_json({"type": "hangup"})


def test_ws_audio_preference_can_change_mid_call():
    with TestClient(app) as client, client.websocket_connect("/ws/call") as ws:
        ws.send_json({"type": "start", "session_id": "wsl2", "caller_phone": RAVI})
        ws.receive_json()

        ws.send_bytes(b"hi-IN::where is my order")
        assert _next_of_type(ws, "reply")["audio"] == ["native"]  # default
        _next_of_type(ws, "audio_end")

        ws.send_json({"type": "set_audio", "audio": "english"})
        ws.send_bytes(b"where is my order")
        assert _next_of_type(ws, "reply")["audio"] == ["english"]
        ws.send_json({"type": "hangup"})


def test_ws_invalid_audio_preference_is_ignored_not_fatal():
    with TestClient(app) as client, client.websocket_connect("/ws/call") as ws:
        ws.send_json({
            "type": "start", "session_id": "wsl3", "caller_phone": RAVI, "audio": "klingon",
        })
        ws.receive_json()
        ws.send_json({"type": "set_audio", "audio": "also-bad"})
        ws.send_bytes(b"hi-IN::where is my order")
        assert _next_of_type(ws, "reply")["audio"] == ["native"]
        ws.send_json({"type": "hangup"})


# ---- /call/turn -------------------------------------------------------------------------


def test_call_turn_returns_both_english_and_localized_text():
    with TestClient(app) as client:
        body = client.post("/call/turn", json={
            "session_id": "ct1", "text": "where is my order",
            "caller_phone": RAVI, "language": "hi-IN",
        }).json()
    assert "delivery attempt" in body["reply_text"].lower()
    assert body["reply_text_localized"] == f"[hi-IN] {body['reply_text']}"
    assert body["reply_language"] == "hi-IN"


def test_call_turn_english_has_no_localized_text():
    with TestClient(app) as client:
        body = client.post("/call/turn", json={
            "session_id": "ct2", "text": "where is my order", "caller_phone": RAVI,
        }).json()
    assert body["reply_text_localized"] is None and body["reply_language"] is None


def test_call_turn_localizes_the_confirmation_prompt_when_paused():
    with TestClient(app) as client:
        body = client.post("/call/turn", json={
            "session_id": "ct3", "text": "reschedule to in 3 days evening",
            "caller_phone": RAVI, "language": "ta-IN",
        }).json()
    assert body["reply_text"] == ""  # composer never ran; interrupt() paused the turn
    assert body["reply_text_localized"] == f"[ta-IN] {body['awaiting_confirmation']['prompt']}"


# ---- Gradio harness + chat formatting --------------------------------------------------------


def test_text_turn_in_hindi_returns_both_texts():
    out = run_text_turn("h1", RAVI, "where is my order", language="hi-IN")
    assert out.localized_text == f"[hi-IN] {out.reply_text}"
    assert out.reply_language == "hi-IN"


def test_text_turn_english_has_no_localized_text():
    assert run_text_turn("h2", RAVI, "where is my order").localized_text is None


def test_unclear_confirmation_reask_is_localized_too():
    first = run_text_turn("h3", RAVI, "reschedule to in 3 days evening", language="hi-IN")
    second = run_text_turn(
        "h3", RAVI, "maybe later", pending_confirmation=first.awaiting_confirmation,
        language="hi-IN",
    )
    assert second.localized_text.startswith("[hi-IN] Sorry, was that a yes or a no?")


def test_audio_turn_hear_native_only_synthesizes_native():
    out = run_audio_turn("h4", RAVI, b"hi-IN::where is my order", hear="native")
    assert out.audio_native and out.audio_english is None


def test_audio_turn_hear_english_only_synthesizes_english():
    out = run_audio_turn("h5", RAVI, b"hi-IN::where is my order", hear="english")
    assert out.audio_english and out.audio_native is None
    assert out.turn.localized_text  # text is still available in both languages


def test_audio_turn_hear_both_synthesizes_two_different_audios():
    out = run_audio_turn("h6", RAVI, b"hi-IN::where is my order", hear="both")
    assert out.audio_native and out.audio_english
    assert out.audio_native != out.audio_english  # different text -> different audio


def test_audio_turn_english_caller_fills_both_slots_with_the_same_audio():
    out = run_audio_turn("h7", RAVI, b"where is my order", hear="both")
    assert out.audio_native == out.audio_english and out.audio_english


def test_chat_formatting_shows_english_then_the_named_language():
    out = run_text_turn("h8", RAVI, "where is my order", language="ta-IN")
    formatted = _format_reply(out)
    assert formatted.startswith("**English**\n")
    assert "**Tamil**\n[ta-IN] " in formatted


def test_chat_formatting_english_only_has_no_headers():
    out = run_text_turn("h9", RAVI, "where is my order")
    assert "**English**" not in _format_reply(out)
