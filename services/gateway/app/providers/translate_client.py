"""
Translate client -- English reply text -> the caller's language.

The graph reasons in English (ASR runs in mode="translate"), so every reply it
composes is English. Without this step the caller's *voice* was Hindi/Tamil/...
(TTS was told target_language_code=hi-IN) but the *words* were still English.

Endpoint contract verified against docs.sarvam.ai (2026-09), not assumed:
  POST /translate, auth header `api-subscription-key`, JSON body
  {input, source_language_code, target_language_code, model, mode}, response
  {request_id, translated_text, source_language_code}.
  - model `sarvam-translate:v1` covers all 22 languages (mayura:v1 only 11)
    but supports only mode="formal" -- fine for customer support.
  - Input limit is 2000 chars for sarvam-translate:v1; replies are split on
    sentence boundaries well under that rather than trusting they're short.

Same shape as the other providers: ABC, real client, credential-free mock,
get_x_client()/reset_x_client() singleton pair.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.config import get_settings
from app.providers.cache import get_cache
from app.providers.http_errors import describe_http_error

# Stay well under sarvam-translate:v1's 2000-char cap.
_MAX_CHUNK_CHARS = 1500
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class TranslateUnavailable(RuntimeError):
    pass


@dataclass
class TranslateResult:
    text: str
    target_language: str
    cached: bool
    provider: str


def split_for_translation(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Greedy sentence packing. A single over-long sentence is hard-split
    rather than sent whole and rejected."""
    chunks: list[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT.split(text.strip()):
        while len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


class TranslateClient(ABC):
    @abstractmethod
    def translate(self, text: str, target_language: str) -> TranslateResult: ...


class SarvamTranslateClient(TranslateClient):
    def __init__(self, api_key: str, model: str, mode: str, base_url: str) -> None:
        import httpx

        self._client = httpx.Client(
            base_url=base_url,
            headers={"api-subscription-key": api_key},
            timeout=10.0,
        )
        self._model = model
        self._mode = mode
        self._cache = get_cache("translate")

    def translate(self, text: str, target_language: str) -> TranslateResult:
        import httpx

        key = self._cache.hash_key(text, self._model, self._mode, target_language)
        if (hit := self._cache.get(key)) is not None:
            return TranslateResult(
                text=hit.data.decode("utf-8"), target_language=target_language,
                cached=True, provider="sarvam",
            )

        parts: list[str] = []
        for chunk in split_for_translation(text):
            try:
                resp = self._client.post(
                    "/translate",
                    json={
                        "input": chunk,
                        "source_language_code": "en-IN",
                        "target_language_code": target_language,
                        "model": self._model,
                        "mode": self._mode,
                        # Keep digits as digits: order IDs, dates, and pincodes
                        # must not be turned into native-script numerals that a
                        # later reader (or TTS) could misread.
                        "numerals_format": "international",
                    },
                )
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise TranslateUnavailable(describe_http_error(exc)) from exc
            translated = resp.json().get("translated_text")
            if not translated:
                raise TranslateUnavailable(
                    f"Sarvam translate returned no text for request_id={resp.json().get('request_id')}"
                )
            parts.append(translated)

        result_text = " ".join(parts)
        self._cache.set(key, result_text.encode("utf-8"), {"target": target_language})
        return TranslateResult(
            text=result_text, target_language=target_language,
            cached=False, provider="sarvam",
        )


class MockTranslateClient(TranslateClient):
    """
    No network. Tags the text with its target language ("[hi-IN] ...") instead
    of returning it unchanged, so a test can tell "translation ran" apart from
    "translation was skipped" -- an identity mock could not.
    """

    def translate(self, text: str, target_language: str) -> TranslateResult:
        return TranslateResult(
            text=f"[{target_language}] {text}", target_language=target_language,
            cached=False, provider="mock",
        )


_client: TranslateClient | None = None


def get_translate_client() -> TranslateClient:
    global _client
    if _client is not None:
        return _client
    settings = get_settings()
    if settings.translate_provider == "sarvam" and settings.sarvam_api_key:
        _client = SarvamTranslateClient(
            settings.sarvam_api_key, settings.sarvam_translate_model,
            settings.sarvam_translate_mode, settings.sarvam_base_url,
        )
    else:
        _client = MockTranslateClient()
    return _client


def reset_translate_client() -> None:
    global _client
    _client = None
