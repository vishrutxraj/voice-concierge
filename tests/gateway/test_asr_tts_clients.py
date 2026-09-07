"""
Sarvam ASR/TTS client tests.

Real network calls to api.sarvam.ai are never made here -- httpx.MockTransport
substitutes a handler asserting on the exact request shape and returning
responses matching the CURRENT verified API contract (docs.sarvam.ai, checked
directly against live docs before writing the client, not assumed from
training data or the original project brief, which referenced a since-
deprecated endpoint and model). This is the same pattern used for HttpOrderClient
in test_order_client.py: substitute the transport, keep the real client code.
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest
from app.providers.asr_client import (
    ASRUnavailable,
    MockASRClient,
    SarvamASRClient,
)
from app.providers.cache import reset_caches
from app.providers.tts_client import (
    MockTTSClient,
    SarvamTTSClient,
    TTSUnavailable,
    _silent_wav,
)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    from app.config import get_settings

    get_settings.cache_clear()
    reset_caches()
    yield
    reset_caches()
    get_settings.cache_clear()


def _mock_client(client_cls, handler, **kwargs):
    """Build a real client instance, then swap in a mocked transport."""
    instance = client_cls(**kwargs)
    instance._client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.sarvam.ai",
        headers={"api-subscription-key": "test-key"},
    )
    return instance


# ---- ASR: verified request/response contract -----------------------------


def test_asr_sends_correct_auth_header_and_endpoint():
    """
    api-subscription-key, NOT Authorization: Bearer -- Groq's auth scheme is
    used elsewhere in this codebase and copying it here by habit would be a
    real, silent bug (every request would 403).
    """
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth_header"] = request.headers.get("api-subscription-key")
        seen["has_bearer"] = "Authorization" in request.headers
        return httpx.Response(200, json={
            "request_id": "r1", "transcript": "hello", "language_code": "en-IN",
        })

    client = _mock_client(
        SarvamASRClient, handler,
        api_key="test-key", model="saaras:v3", mode="translate", base_url="https://api.sarvam.ai",
    )
    client.transcribe(b"fake-audio-bytes")

    assert seen["path"] == "/speech-to-text"
    assert seen["auth_header"] == "test-key"
    assert seen["has_bearer"] is False


def test_asr_sends_multipart_file_and_mode():
    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode("latin-1")
        assert 'name="file"' in body
        assert 'name="model"' in body
        assert "translate" in body
        return httpx.Response(200, json={
            "request_id": "r2", "transcript": "test", "language_code": "hi-IN",
        })

    client = _mock_client(
        SarvamASRClient, handler,
        api_key="k", model="saaras:v3", mode="translate", base_url="https://api.sarvam.ai",
    )
    client.transcribe(b"\x00\x01audio-payload")


def test_asr_translate_mode_returns_english_text_and_detected_language():
    """
    The core architectural bet: mode=translate returns English in `transcript`
    regardless of what was spoken, while `language_code` reports what was
    ACTUALLY spoken. Confusing the two would break the whole
    reason-in-English/speak-in-detected-language design.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "request_id": "r3",
            "transcript": "Where is my order",  # English, per mode=translate
            "language_code": "te-IN",  # but the caller spoke Telugu
        })

    client = _mock_client(
        SarvamASRClient, handler,
        api_key="k", model="saaras:v3", mode="translate", base_url="https://api.sarvam.ai",
    )
    result = client.transcribe(b"telugu-audio")

    assert result.text == "Where is my order"
    assert result.detected_language == "te-IN"
    assert result.provider == "sarvam"
    assert result.cached is False


def test_asr_http_error_raises_unavailable_not_generic_exception():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "server error"}})

    client = _mock_client(
        SarvamASRClient, handler,
        api_key="k", model="saaras:v3", mode="translate", base_url="https://api.sarvam.ai",
    )
    with pytest.raises(ASRUnavailable) as excinfo:
        client.transcribe(b"audio")
    # Regression: str(httpx.HTTPStatusError) alone drops the response BODY,
    # which is where a provider actually explains a 4xx/5xx (e.g. Sarvam's
    # "Model 'bulbul:v2' has been deprecated..."). A real TTS failure was
    # opaque in every log until someone made the same request by hand just
    # to read it -- the message here must include the body, not just the
    # status code.
    assert "server error" in str(excinfo.value)


def test_asr_caches_identical_audio_and_skips_network_on_replay():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json={
            "request_id": "r4", "transcript": "cached test", "language_code": "en-IN",
        })

    client = _mock_client(
        SarvamASRClient, handler,
        api_key="k", model="saaras:v3", mode="translate", base_url="https://api.sarvam.ai",
    )
    same_audio = b"identical-bytes"

    first = client.transcribe(same_audio)
    second = client.transcribe(same_audio)

    assert first.cached is False
    assert second.cached is True
    assert second.text == first.text
    assert calls["count"] == 1, "second call must not hit the network at all"


def test_asr_different_audio_is_not_conflated_in_cache():
    def handler(request: httpx.Request) -> httpx.Response:
        # Distinguish response by request body length so we can prove the
        # cache key actually varies with audio content.
        n = len(request.content)
        return httpx.Response(200, json={
            "request_id": "r5", "transcript": f"len-{n}", "language_code": "en-IN",
        })

    client = _mock_client(
        SarvamASRClient, handler,
        api_key="k", model="saaras:v3", mode="translate", base_url="https://api.sarvam.ai",
    )
    r1 = client.transcribe(b"short")
    r2 = client.transcribe(b"a much longer audio payload than the first one")
    assert r1.text != r2.text


# ---- TTS: verified request/response contract ------------------------------


def test_tts_sends_correct_auth_header_and_endpoint():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/text-to-speech"
        assert request.headers.get("api-subscription-key") == "test-key"
        assert "Authorization" not in request.headers
        wav = base64.b64encode(_silent_wav(200)).decode()
        return httpx.Response(200, json={"request_id": "t1", "audios": [wav]})

    client = _mock_client(
        SarvamTTSClient, handler,
        api_key="test-key", model="bulbul:v2", speaker="anushka", base_url="https://api.sarvam.ai",
    )
    client.synthesize("hello")


def test_tts_sends_lowercase_speaker_regardless_of_input_casing():
    """Speaker names are case-sensitive per the API; passing 'Anushka' must not
    silently fail -- the client normalizes to lowercase before it ever sends
    a request."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen["speaker"] = body["speaker"]
        wav = base64.b64encode(_silent_wav(200)).decode()
        return httpx.Response(200, json={"request_id": "t2", "audios": [wav]})

    client = _mock_client(
        SarvamTTSClient, handler,
        api_key="k", model="bulbul:v2", speaker="Anushka", base_url="https://api.sarvam.ai",
    )
    client.synthesize("hello", language="en-IN")
    assert seen["speaker"] == "anushka"


def test_tts_decodes_base64_audio_correctly():
    """
    The exact bug every Sarvam migration guide flags: the REST response is
    JSON with base64 audio, not a raw audio stream. This proves the client
    decodes it -- result.audio_bytes must be the real WAV bytes, not the
    base64 text and not the raw JSON response body.
    """
    real_wav = _silent_wav(300)
    encoded = base64.b64encode(real_wav).decode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"request_id": "t3", "audios": [encoded]})

    client = _mock_client(
        SarvamTTSClient, handler,
        api_key="k", model="bulbul:v2", speaker="anushka", base_url="https://api.sarvam.ai",
    )
    result = client.synthesize("test text")

    assert result.audio_bytes == real_wav
    assert result.audio_bytes[:4] == b"RIFF"  # a real WAV header, not base64 text
    assert b"audios" not in result.audio_bytes  # not the raw JSON response


def test_tts_no_audio_in_response_raises_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"request_id": "t4", "audios": []})

    client = _mock_client(
        SarvamTTSClient, handler,
        api_key="k", model="bulbul:v2", speaker="anushka", base_url="https://api.sarvam.ai",
    )
    with pytest.raises(TTSUnavailable):
        client.synthesize("hello")


def test_tts_http_error_raises_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    client = _mock_client(
        SarvamTTSClient, handler,
        api_key="k", model="bulbul:v2", speaker="anushka", base_url="https://api.sarvam.ai",
    )
    with pytest.raises(TTSUnavailable) as excinfo:
        client.synthesize("hello")
    # Same regression as the ASR client -- see that test's comment. This is
    # exactly the class of bug that shipped a broken bulbul:v2 to production
    # silently; the message must carry the body, not just the status line.
    assert "rate limited" in str(excinfo.value)


def test_tts_caches_identical_text_voice_language_and_skips_network():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        wav = base64.b64encode(_silent_wav(200)).decode()
        return httpx.Response(200, json={"request_id": "t5", "audios": [wav]})

    client = _mock_client(
        SarvamTTSClient, handler,
        api_key="k", model="bulbul:v2", speaker="anushka", base_url="https://api.sarvam.ai",
    )
    first = client.synthesize("Your order is out for delivery", language="en-IN")
    second = client.synthesize("Your order is out for delivery", language="en-IN")

    assert first.cached is False
    assert second.cached is True
    assert second.audio_bytes == first.audio_bytes
    assert calls["count"] == 1


def test_tts_same_text_different_language_is_not_a_cache_hit():
    def handler(request: httpx.Request) -> httpx.Response:
        wav = base64.b64encode(_silent_wav(200)).decode()
        return httpx.Response(200, json={"request_id": "t6", "audios": [wav]})

    client = _mock_client(
        SarvamTTSClient, handler,
        api_key="k", model="bulbul:v2", speaker="anushka", base_url="https://api.sarvam.ai",
    )
    en = client.synthesize("hello", language="en-IN")
    hi = client.synthesize("hello", language="hi-IN")
    assert en.cached is False
    assert hi.cached is False  # different cache key -- language is part of it


# ---- Mock clients: usable with zero credentials ---------------------------


def test_mock_asr_round_trips_language_tag():
    result = MockASRClient().transcribe(b"hi-IN::main order kahan hai")
    assert result.detected_language == "hi-IN"
    assert result.text == "main order kahan hai"
    assert result.provider == "mock"


def test_mock_asr_defaults_to_english_with_no_tag():
    result = MockASRClient().transcribe(b"where is my order")
    assert result.detected_language == "en-IN"
    assert result.text == "where is my order"


def test_mock_asr_uses_cache_too():
    client = MockASRClient()
    audio = b"repeat-this-exact-audio"
    first = client.transcribe(audio)
    second = client.transcribe(audio)
    assert first.cached is False
    assert second.cached is True


def test_mock_tts_produces_valid_wav():
    import io
    import wave

    result = MockTTSClient().synthesize("hello world")
    f = wave.open(io.BytesIO(result.audio_bytes), "rb")
    assert f.getnchannels() == 1
    assert f.getframerate() == 16000


def test_mock_tts_uses_cache_too():
    client = MockTTSClient()
    first = client.synthesize("same text", language="en-IN")
    second = client.synthesize("same text", language="en-IN")
    assert first.cached is False
    assert second.cached is True
    assert second.audio_bytes == first.audio_bytes
