"""
Output-edge localization: English reply -> what the caller actually hears.

The graph composes every reply in English (README: "translate at the edge,
reason in English"). This is the edge. Both audio callers -- app/audio/
call_loop.py and app/ui/harness.py -- run replies through localize_reply()
right before TTS, so the spoken words and the spoken language always agree.

Fallback rules, in order (each one preferable to the alternative of speaking
English text in a Hindi/Tamil/... voice, which is the bug this replaces):
  - English or unknown language      -> no translation, en-IN voice
  - language TTS cannot speak        -> English text, en-IN voice
  - translation provider unavailable -> English text, en-IN voice
"""

from __future__ import annotations

from dataclasses import dataclass

from app.observability.trace import EventKind, trace
from app.providers.translate_client import TranslateUnavailable, get_translate_client

# bulbul:v3's target_language_code roster, verified against docs.sarvam.ai
# (2026-09). Translate supports more languages than TTS can speak; a reply
# translated into e.g. Urdu would then be sent to TTS and rejected with a 400.
TTS_LANGUAGES = frozenset({
    "bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN",
    "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN",
})
DEFAULT_LANGUAGE = "en-IN"

# Display names for UI labels ("Hindi", not "hi-IN").
LANGUAGE_NAMES = {
    "en-IN": "English", "hi-IN": "Hindi", "bn-IN": "Bengali", "gu-IN": "Gujarati",
    "kn-IN": "Kannada", "ml-IN": "Malayalam", "mr-IN": "Marathi", "od-IN": "Odia",
    "pa-IN": "Punjabi", "ta-IN": "Tamil", "te-IN": "Telugu",
}


def language_name(code: str | None) -> str:
    return LANGUAGE_NAMES.get(code or "", code or "Detected language")


@dataclass
class LocalizedReply:
    text: str  # what the caller hears / reads
    text_en: str  # the original English, kept for logs and the UI
    language: str  # TTS language code actually used
    translated: bool
    fallback_reason: str | None = None


def localize_reply(text: str, language: str | None, session_id: str = "") -> LocalizedReply:
    """Blocking (network call on a cache miss) -- call via asyncio.to_thread
    from async code."""
    target = language or DEFAULT_LANGUAGE
    if target.lower().startswith("en"):
        return LocalizedReply(text, text, DEFAULT_LANGUAGE, translated=False)

    if target not in TTS_LANGUAGES:
        return _fallback(text, session_id, target, "tts_language_unsupported")

    try:
        result = get_translate_client().translate(text, target)
    except TranslateUnavailable as exc:
        return _fallback(text, session_id, target, "translate_unavailable", str(exc))

    trace(
        session_id, EventKind.DECISION, "localize", "reply_translated",
        data={"target": target, "provider": result.provider, "cached": result.cached},
    )
    return LocalizedReply(result.text, text, target, translated=True)


@dataclass
class AudioTrack:
    variant: str  # "native" | "english"
    text: str
    language: str  # TTS language code


def audio_tracks(localized: LocalizedReply, preference: str) -> list[AudioTrack]:
    """
    Which renderings of this reply to synthesize, in playback order.

    Untranslated replies (English caller, or a fallback) have exactly one
    possible rendering, so the preference is moot -- and a "both" request must
    not speak the same English sentence twice.
    """
    english = AudioTrack("english", localized.text_en, DEFAULT_LANGUAGE)
    if not localized.translated:
        return [english]
    native = AudioTrack("native", localized.text, localized.language)
    if preference == "english":
        return [english]
    if preference == "both":
        return [native, english]
    return [native]


def _fallback(
    text: str, session_id: str, target: str, reason: str, error: str | None = None
) -> LocalizedReply:
    trace(
        session_id, EventKind.ERROR, "localize", reason,
        data={"target": target, "error": error, "note": "speaking English instead"},
    )
    return LocalizedReply(text, text, DEFAULT_LANGUAGE, translated=False, fallback_reason=reason)
