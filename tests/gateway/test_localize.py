"""
Output-edge translation tests.

The bug these guard: the graph's reply is always English, and TTS was told
target_language_code=hi-IN but handed English text -- an English sentence in a
Hindi voice. Translation now happens at the edge (app/audio/localize.py) and
must (a) actually run for supported non-English languages, (b) never run for
English, and (c) degrade to plain English speech -- not an error, not a
mismatched voice -- when it can't.

Sarvam's translate endpoint is exercised through httpx.MockTransport asserting
the request shape verified against docs.sarvam.ai (2026-09); no live calls.
"""

from __future__ import annotations

import json

import httpx
import pytest
from app.audio.call_loop import _CallSession, _handle_utterance
from app.audio.localize import localize_reply
from app.providers.asr_client import reset_asr_client
from app.providers.cache import reset_caches
from app.providers.llm_client import reset_llm_client
from app.providers.order_client import reset_order_client
from app.providers.translate_client import (
    MockTranslateClient,
    SarvamTranslateClient,
    TranslateUnavailable,
    reset_translate_client,
    split_for_translation,
)
from app.providers.tts_client import reset_tts_client
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


def _sarvam(handler) -> SarvamTranslateClient:
    client = SarvamTranslateClient(
        api_key="test-key", model="sarvam-translate:v1", mode="formal",
        base_url="https://api.sarvam.ai",
    )
    client._client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.sarvam.ai",
        headers={"api-subscription-key": "test-key"},
    )
    return client


# ---- Sarvam client: verified request/response contract ---------------------


def test_translate_sends_verified_request_shape():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["key"] = request.headers.get("api-subscription-key")
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"request_id": "r", "translated_text": "नमस्ते"})

    result = _sarvam(handler).translate("Hello", "hi-IN")

    assert seen["path"] == "/translate"
    assert seen["key"] == "test-key" and seen["auth"] is None
    assert seen["body"] == {
        "input": "Hello", "source_language_code": "en-IN", "target_language_code": "hi-IN",
        "model": "sarvam-translate:v1", "mode": "formal", "numerals_format": "international",
    }
    assert result.text == "नमस्ते" and not result.cached


def test_translate_result_is_cached_so_a_replayed_reply_costs_nothing():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json={"request_id": "r", "translated_text": "नमस्ते"})

    client = _sarvam(handler)
    first = client.translate("Hello", "hi-IN")
    second = client.translate("Hello", "hi-IN")
    other_lang = client.translate("Hello", "ta-IN")

    assert (first.cached, second.cached, other_lang.cached) == (False, True, False)
    assert len(calls) == 2  # ta-IN is a different key, not a false cache hit


def test_translate_http_error_surfaces_response_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad language pair"}})

    with pytest.raises(TranslateUnavailable, match="bad language pair"):
        _sarvam(handler).translate("Hello", "hi-IN")


def test_translate_empty_response_raises_rather_than_speaking_silence():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"request_id": "r", "translated_text": ""})

    with pytest.raises(TranslateUnavailable):
        _sarvam(handler).translate("Hello", "hi-IN")


def test_long_reply_is_split_under_the_input_limit_and_rejoined():
    sentence = "Your order is out for delivery today."
    text = " ".join([sentence] * 100)  # ~3.7k chars, over sarvam-translate's 2000 cap
    seen_lengths = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)["input"]
        seen_lengths.append(len(body))
        return httpx.Response(200, json={"request_id": "r", "translated_text": "X"})

    result = _sarvam(handler).translate(text, "hi-IN")

    assert len(seen_lengths) > 1 and max(seen_lengths) <= 2000
    assert result.text == " ".join(["X"] * len(seen_lengths))


def test_split_hard_splits_a_single_oversized_sentence():
    chunks = split_for_translation("a" * 50, max_chars=20)
    assert [len(c) for c in chunks] == [20, 20, 10]


# ---- localize_reply: the decision logic -------------------------------------


@pytest.mark.parametrize("lang", ["en-IN", "en", "en-US", None])
def test_english_is_never_translated(lang):
    out = localize_reply("Your order is on its way.", lang)
    assert out.text == "Your order is on its way."
    assert not out.translated and out.language == "en-IN"


def test_supported_language_is_translated_and_keeps_the_english_original():
    out = localize_reply("Your order is on its way.", "hi-IN")
    assert out.translated and out.language == "hi-IN"
    assert out.text == "[hi-IN] Your order is on its way."  # mock's visible tag
    assert out.text_en == "Your order is on its way."


def test_language_tts_cannot_speak_falls_back_to_english_not_a_400():
    # ur-IN is translatable but not in bulbul:v3's roster; sending it to TTS
    # would be rejected, so the caller must get English speech instead.
    out = localize_reply("Your order is on its way.", "ur-IN")
    assert not out.translated and out.language == "en-IN"
    assert out.fallback_reason == "tts_language_unsupported"
    assert out.text == "Your order is on its way."


def test_translation_outage_falls_back_to_english_speech(monkeypatch):
    class Down(MockTranslateClient):
        def translate(self, text, target_language):
            raise TranslateUnavailable("boom")

    monkeypatch.setattr("app.audio.localize.get_translate_client", lambda: Down())
    out = localize_reply("Your order is on its way.", "hi-IN")
    assert not out.translated and out.language == "en-IN"
    assert out.fallback_reason == "translate_unavailable"


# ---- end to end through the call loop ---------------------------------------


async def _run_one(audio: bytes) -> FakeAudioTransport:
    session = _CallSession(session_id="loc1", caller_phone=RAVI, transport_kind="text")
    transport = FakeAudioTransport()
    await _handle_utterance(transport, session, audio, 0)
    return transport


async def test_hindi_caller_gets_translated_reply_event_and_audio():
    transport = await _run_one(b"hi-IN::where is my order")

    reply = transport.events("reply")[0]
    assert reply["language"] == "hi-IN"
    assert reply["text"].startswith("[hi-IN] ")  # translated text is what's spoken
    assert "delivery attempt" in reply["text_en"].lower()  # English original preserved
    assert reply["text_en"] == reply["text"].removeprefix("[hi-IN] ")
    assert transport.events("audio_end")


async def test_english_caller_is_untouched_by_translation():
    transport = await _run_one(b"where is my order")

    reply = transport.events("reply")[0]
    assert reply["language"] == "en-IN"
    assert reply["text"] == reply["text_en"]
    assert "delivery attempt" in reply["text"].lower()


async def test_fixed_fallback_lines_are_translated_too():
    """The hardcoded 'trouble hearing you' / 'yes or a no' lines are English
    literals that bypass the graph -- they must go through the same edge."""
    from app.audio.call_loop import _deliver

    session = _CallSession(
        session_id="loc2", caller_phone=RAVI, transport_kind="text", pinned_language="ta-IN"
    )
    transport = FakeAudioTransport()
    await _deliver(transport, session, lambda: True, text="Sorry, was that a yes or a no?")

    reply = transport.events("reply")[0]
    assert reply["text"] == "[ta-IN] Sorry, was that a yes or a no?"


def test_mock_translate_client_is_deterministic():
    assert MockTranslateClient().translate("hi", "te-IN").text == "[te-IN] hi"
