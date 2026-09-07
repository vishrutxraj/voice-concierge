"""
TTS client.

Endpoint contract verified against docs.sarvam.ai (Aug 2026). The detail every
migration guide calls out as the single most common bug: Sarvam's REST
response is JSON with base64-encoded audio in an `audios` array, NOT a raw
audio stream. Code that writes response.content directly to a .wav file
produces a file full of JSON text. SarvamTTSClient decodes base64 before ever
returning bytes, so that mistake isn't possible to make downstream of this
module.

Speaker names are case-sensitive and must be lowercase. Defaults differ by
model version (config.sarvam_tts_speaker resolves this) -- passing "Anushka"
where the API expects "anushka" fails silently in some SDKs; here it's a
hardcoded lowercase default so the common path can't hit that either.
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.config import get_settings
from app.providers.cache import get_cache


class TTSUnavailable(RuntimeError):
    pass


@dataclass
class TTSResult:
    audio_bytes: bytes  # decoded WAV
    format: str
    cached: bool
    provider: str
    voice: str
    language: str


class TTSClient(ABC):
    @abstractmethod
    def synthesize(self, text: str, language: str = "en-IN") -> TTSResult: ...


class SarvamTTSClient(TTSClient):
    def __init__(self, api_key: str, model: str, speaker: str, base_url: str) -> None:
        import httpx

        self._client = httpx.Client(
            base_url=base_url,
            headers={"api-subscription-key": api_key},
            timeout=15.0,
        )
        self._model = model
        self._speaker = speaker.lower()  # case-sensitive API; never trust caller casing
        self._cache = get_cache("tts")

    def synthesize(self, text: str, language: str = "en-IN") -> TTSResult:
        import httpx

        key = self._cache.hash_key(text, self._model, self._speaker, language)
        if (hit := self._cache.get(key)) is not None:
            return TTSResult(
                audio_bytes=hit.data, format="wav", cached=True,
                provider="sarvam", voice=self._speaker, language=language,
            )

        try:
            resp = self._client.post(
                "/text-to-speech",
                json={
                    "text": text,
                    "target_language_code": language,
                    "speaker": self._speaker,
                    "model": self._model,
                },
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise TTSUnavailable(str(exc)) from exc

        data = resp.json()
        audios = data.get("audios") or []
        if not audios:
            raise TTSUnavailable(f"Sarvam TTS returned no audio for request_id={data.get('request_id')}")

        # THE decode step -- see module docstring. Never return audios[0] raw.
        audio_bytes = base64.b64decode(audios[0])
        self._cache.set(key, audio_bytes, {"text_len": len(text), "voice": self._speaker})

        return TTSResult(
            audio_bytes=audio_bytes, format="wav", cached=False,
            provider="sarvam", voice=self._speaker, language=language,
        )


class MockTTSClient(TTSClient):
    """
    Emits a minimal valid (silent) WAV file, deterministic per text/voice/
    language via the same cache path as the real client -- so cache hit/miss
    behavior is testable without any Sarvam credentials.
    """

    def synthesize(self, text: str, language: str = "en-IN") -> TTSResult:
        cache = get_cache("tts")
        key = cache.hash_key(text, "mock", "mock-voice", language)
        if (hit := cache.get(key)) is not None:
            return TTSResult(
                audio_bytes=hit.data, format="wav", cached=True,
                provider="mock", voice="mock-voice", language=language,
            )
        audio_bytes = _silent_wav(duration_ms=max(200, len(text) * 40))
        cache.set(key, audio_bytes, {"text_len": len(text)})
        return TTSResult(
            audio_bytes=audio_bytes, format="wav", cached=False,
            provider="mock", voice="mock-voice", language=language,
        )


def _silent_wav(duration_ms: int, sample_rate: int = 16000) -> bytes:
    """Smallest valid PCM16 mono WAV -- real header, silent payload."""
    import struct

    n_samples = int(sample_rate * duration_ms / 1000)
    data = b"\x00\x00" * n_samples
    byte_rate = sample_rate * 2
    header = (
        b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, byte_rate, 2, 16)
        + b"data" + struct.pack("<I", len(data))
    )
    return header + data


_client: TTSClient | None = None


def get_tts_client() -> TTSClient:
    global _client
    if _client is not None:
        return _client
    settings = get_settings()
    if settings.tts_provider == "sarvam" and settings.sarvam_api_key:
        _client = SarvamTTSClient(
            settings.sarvam_api_key, settings.sarvam_tts_model,
            settings.sarvam_tts_speaker, settings.sarvam_base_url,
        )
    else:
        _client = MockTTSClient()
    return _client


def reset_tts_client() -> None:
    global _client
    _client = None
