"""
ASR client.

Endpoint contract verified against docs.sarvam.ai (Aug 2026) rather than
assumed from training data or the earlier project brief -- the brief's
/speech-to-text-translate is now the LEGACY endpoint. Current integrations use
POST /speech-to-text with mode="translate", which is what get_asr_client wires
by default per config.sarvam_asr_mode. Two details that are easy to get wrong
and are handled explicitly here:

  - Auth header is `api-subscription-key: <key>`, NOT `Authorization: Bearer`
    (that's Groq's scheme, used elsewhere in this codebase -- don't copy it
    here by habit).
  - mode="translate" always returns English in `transcript`, regardless of
    what language was spoken. `language_code` in the response tells you what
    was actually spoken -- that's the value pinned to CallState.language,
    never the mode.

Mirrors the OrderClient / LLMClient pattern: one ABC, a real Sarvam
implementation, and a mock that needs no credentials or audio codec support --
so the graph and cache logic are fully testable for zero cost.
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.config import get_settings
from app.providers.cache import get_cache


class ASRUnavailable(RuntimeError):
    pass


@dataclass
class ASRResult:
    text: str  # English transcript (mode="translate")
    detected_language: str | None  # BCP-47, e.g. "hi-IN" -- what was SPOKEN
    language_confidence: float | None
    raw_transcript: str  # exactly what Sarvam's `transcript` field returned
    cached: bool
    provider: str


class ASRClient(ABC):
    @abstractmethod
    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> ASRResult: ...


class SarvamASRClient(ASRClient):
    def __init__(self, api_key: str, model: str, mode: str, base_url: str) -> None:
        import httpx

        self._client = httpx.Client(
            base_url=base_url,
            headers={"api-subscription-key": api_key},
            timeout=15.0,  # STT can legitimately take a few seconds for 30s audio
        )
        self._model = model
        self._mode = mode
        self._cache = get_cache("asr")

    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> ASRResult:
        import httpx

        key = self._cache.hash_key(audio_bytes, self._model, self._mode)
        if (hit := self._cache.get(key)) is not None:
            meta = hit.meta
            return ASRResult(
                text=meta["text"],
                detected_language=meta.get("detected_language"),
                language_confidence=meta.get("language_confidence"),
                raw_transcript=meta.get("raw_transcript", meta["text"]),
                cached=True,
                provider="sarvam",
            )

        try:
            resp = self._client.post(
                "/speech-to-text",
                data={"model": self._model, "mode": self._mode, "language_code": "unknown"},
                files={"file": (filename, audio_bytes, "audio/wav")},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ASRUnavailable(str(exc)) from exc

        data = resp.json()
        text = data.get("transcript", "")
        meta = {
            "text": text,
            "raw_transcript": text,
            "detected_language": data.get("language_code"),
            "language_confidence": data.get("language_probability"),
        }
        # Cache the metadata, not audio bytes -- there's nothing binary worth
        # storing separately from the JSON here (unlike TTS's actual audio).
        self._cache.set(key, b"1", meta)

        return ASRResult(
            text=text,
            detected_language=meta["detected_language"],
            language_confidence=meta["language_confidence"],
            raw_transcript=text,
            cached=False,
            provider="sarvam",
        )


class MockASRClient(ASRClient):
    """
    No network, no credentials, no real audio codec support.

    Test fixtures pass UTF-8-encoded text as "audio_bytes" -- optionally
    prefixed with a BCP-47 tag like "hi-IN::<utterance>" to simulate a
    detected source language. This is enough to exercise the cache, the
    language-pinning logic, and the graph's downstream handling without any
    audio infrastructure, which doesn't exist until phase 4's transport layer.
    """

    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> ASRResult:
        cache = get_cache("asr")
        key = cache.hash_key(audio_bytes, "mock")
        if (hit := cache.get(key)) is not None:
            return ASRResult(
                text=hit.meta["text"],
                detected_language=hit.meta.get("detected_language"),
                language_confidence=1.0,
                raw_transcript=hit.meta["text"],
                cached=True,
                provider="mock",
            )

        raw = audio_bytes.decode("utf-8", errors="replace")
        if "::" in raw:
            lang, text = raw.split("::", 1)
        else:
            lang, text = "en-IN", raw

        cache.set(key, b"1", {"text": text, "detected_language": lang})
        return ASRResult(
            text=text, detected_language=lang, language_confidence=1.0,
            raw_transcript=text, cached=False, provider="mock",
        )


_client: ASRClient | None = None


def get_asr_client() -> ASRClient:
    global _client
    if _client is not None:
        return _client
    settings = get_settings()
    if settings.asr_provider == "sarvam" and settings.sarvam_api_key:
        _client = SarvamASRClient(
            settings.sarvam_api_key, settings.sarvam_asr_model,
            settings.sarvam_asr_mode, settings.sarvam_base_url,
        )
    else:
        _client = MockASRClient()
    return _client


def reset_asr_client() -> None:
    global _client
    _client = None


def decode_base64_audio(b64: str) -> bytes:
    """Shared helper -- Sarvam's TTS response uses the same base64 WAV framing."""
    return base64.b64decode(b64)
