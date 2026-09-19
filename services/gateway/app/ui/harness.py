"""
Gradio test harness -- pure logic layer.

Deliberately split from blocks.py: Gradio event handlers are hard to unit
test meaningfully (they're wired to UI components, not called with plain
arguments), so every real decision this harness makes lives in plain,
dataclass-returning functions here, tested exactly like every other
provider-facing function in this codebase. blocks.py is thin glue -- UI
components call these functions and format the results for display, nothing
more.

Both turn functions mirror app/audio/call_loop.py's per-turn logic rather
than inventing a third way to drive run_turn(): when a confirmation is
pending, the reply is interpreted via the SAME interpret_confirmation() a
real voice call uses (not the more permissive explicit `resume: bool` the
raw /call/turn API accepts), so testing "yes"/"no" here exercises what a
caller actually has to get right. They're written as one-shot, synchronous
calls rather than reusing call_loop.py's async/streaming/barge-in machinery,
because a UI test harness only ever has one utterance in flight at a time --
none of that machinery has anything to do here.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.audio.confirmation import interpret_confirmation
from app.audio.localize import LocalizedReply, audio_tracks, localize_reply
from app.graph.runner import run_turn
from app.observability.trace import get_sink
from app.providers.asr_client import ASRUnavailable, get_asr_client
from app.providers.tts_client import TTSUnavailable, get_tts_client

TRANSPORT_KIND = "test_harness"


def new_session_id() -> str:
    return f"ui-{uuid.uuid4().hex[:10]}"


@dataclass
class TurnOutcome:
    reply_text: str  # English -- what the graph composed
    intent: str | None
    escalated: bool
    awaiting_confirmation: dict | None
    error: str | None = None
    # The same reply in the caller's language; None when no translation
    # applied (English caller, unsupported language, or translation down).
    localized_text: str | None = None
    reply_language: str | None = None  # TTS language code of localized_text
    localized: LocalizedReply | None = None  # full result, reused for TTS


def _dispatch(
    session_id: str,
    caller_phone: str,
    utterance: str,
    language: str,
    pending_confirmation: dict | None,
) -> TurnOutcome:
    """Shared tail end of both run_text_turn and run_audio_turn, once each has
    its own utterance text (typed, or transcribed). Adds the caller's-language
    rendering of the reply at the very end, so it also covers the fixed
    "yes or a no" re-ask below."""
    outcome = _run(session_id, caller_phone, utterance, language, pending_confirmation)
    if outcome.reply_text:
        localized = localize_reply(outcome.reply_text, language, session_id)
        outcome.localized = localized
        if localized.translated:
            outcome.localized_text = localized.text
            outcome.reply_language = localized.language
    return outcome


def _run(
    session_id: str,
    caller_phone: str,
    utterance: str,
    language: str,
    pending_confirmation: dict | None,
) -> TurnOutcome:
    resume_value = None
    if pending_confirmation is not None:
        resume_value = interpret_confirmation(utterance)
        if resume_value is None:
            prompt = pending_confirmation.get("prompt", "")
            return TurnOutcome(
                reply_text=f"Sorry, was that a yes or a no? {prompt}",
                intent=None, escalated=False,
                awaiting_confirmation=pending_confirmation,
            )

    result = run_turn(
        session_id=session_id, caller_utterance=utterance, caller_phone=caller_phone,
        language=language, transport_kind=TRANSPORT_KIND, resume_value=resume_value,
    )
    router = result.state.get("router") or {}
    prompt = (result.awaiting_confirmation or {}).get("prompt", "")
    return TurnOutcome(
        reply_text=result.reply_text or prompt,
        intent=router.get("intent"),
        escalated=result.state.get("escalated", False),
        awaiting_confirmation=result.awaiting_confirmation,
    )


def run_text_turn(
    session_id: str,
    caller_phone: str,
    text: str,
    pending_confirmation: dict | None = None,
    language: str = "en-IN",
) -> TurnOutcome:
    """`language` is the caller's language for THIS typed chat -- text has no
    ASR to detect it, so the UI lets the tester pick one to see translation."""
    return _dispatch(session_id, caller_phone, text, language, pending_confirmation)


@dataclass
class AudioTurnOutcome:
    transcript: str
    detected_language: str | None
    turn: TurnOutcome
    audio_english: bytes | None = None
    audio_native: bytes | None = None  # same bytes as english when untranslated

    @property
    def reply_audio_bytes(self) -> bytes | None:
        """The caller's-language audio if there is one, else English."""
        return self.audio_native or self.audio_english


HEAR_CHOICES = ("native", "english", "both")


def run_audio_turn(
    session_id: str,
    caller_phone: str,
    audio_bytes: bytes,
    pending_confirmation: dict | None = None,
    hear: str = "native",
) -> AudioTurnOutcome:
    """
    One-shot ASR -> run_turn -> TTS -- the single-turn shape of
    app/audio/call_loop.py's orchestration, so a developer can upload or
    record one WAV, see the transcript and reply, hear the synthesized
    audio, and inspect the explain trail without standing up a WebSocket
    client. `hear` picks which renderings get synthesized ("native",
    "english", "both") -- same semantics as the WebSocket's audio preference.
    """
    try:
        asr_result = get_asr_client().transcribe(audio_bytes)
    except ASRUnavailable as exc:
        empty_turn = TurnOutcome(
            reply_text="", intent=None, escalated=False, awaiting_confirmation=None,
            error=f"ASR unavailable: {exc}",
        )
        return AudioTurnOutcome(transcript="", detected_language=None, turn=empty_turn)

    language = asr_result.detected_language or "en-IN"
    turn = _dispatch(session_id, caller_phone, asr_result.text, language, pending_confirmation)
    outcome = AudioTurnOutcome(
        transcript=asr_result.text, detected_language=asr_result.detected_language, turn=turn,
    )

    if turn.localized is not None and turn.error is None:
        localized = turn.localized
        try:
            for track in audio_tracks(localized, hear):
                audio = get_tts_client().synthesize(track.text, track.language).audio_bytes
                if track.variant == "english":
                    outcome.audio_english = audio
                else:
                    outcome.audio_native = audio
        except TTSUnavailable as exc:
            turn.error = f"TTS unavailable: {exc}"
        if not localized.translated:
            outcome.audio_native = outcome.audio_english

    return outcome


def get_explain(session_id: str) -> dict:
    return get_sink().explain(session_id)
