"""
Call loop: ASRClient -> run_turn -> TTSClient, wrapped around an
AudioTransport, with streaming playback and barge-in.

The graph never sees any of this. run_turn's signature (session_id,
caller_utterance, ..., resume_value) is exactly what phase 2/3 already built;
this module is just a new, audio-flavoured caller of it -- CLAUDE.md's phase 4
note is explicit that the graph itself doesn't change, and it doesn't.

Concurrency model
------------------
One coroutine (`_receiver`) does nothing but pull inbound utterances off the
transport and drop them on an asyncio.Queue as fast as they arrive. The main
loop below takes one at a time and runs a full turn: ASR, then run_turn, then
TTS, then streams the resulting audio out in chunks.

Barge-in is "generation fencing" rather than true preemption. Every utterance
bumps a generation counter; the turn processing it captures that number and,
after every blocking step (ASR, graph, TTS, each outbound chunk), checks
whether it is still current. A superseded turn's output is discarded rather
than sent. This is deliberate, not a shortcut: run_turn is a synchronous call
into LangGraph + SQLite running on a worker thread via asyncio.to_thread, and
Python cannot forcibly cancel a running thread. Fencing the OUTPUT is correct
and race-free; trying to cancel the WORK is not something CPython lets us do
safely here.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field

from app.audio.confirmation import interpret_confirmation
from app.audio.transport import AudioTransport
from app.config import get_settings
from app.graph.runner import run_turn
from app.observability.trace import EventKind, trace
from app.providers.asr_client import ASRUnavailable, get_asr_client
from app.providers.tts_client import TTSUnavailable, get_tts_client

logger = logging.getLogger("concierge.audio")

_UNCLEAR_CONFIRMATION_REPLY = "Sorry, was that a yes or a no? {prompt}"
_ASR_FAILED_REPLY = "I'm having trouble hearing you clearly right now -- could you say that again?"
_TTS_FAILED_NOTE = "voice reply unavailable; text reply already sent"


@dataclass
class _CallSession:
    session_id: str
    caller_phone: str
    transport_kind: str
    pinned_language: str | None = None
    awaiting_confirmation: dict | None = None
    generation: int = 0
    queue: asyncio.Queue[bytes] = field(default_factory=asyncio.Queue)


async def run_call(
    transport: AudioTransport,
    session_id: str,
    caller_phone: str = "",
    transport_kind: str = "browser",
) -> None:
    """Drive one call end-to-end. Returns when the caller hangs up or the
    transport reports a disconnect."""
    session = _CallSession(
        session_id=session_id, caller_phone=caller_phone, transport_kind=transport_kind
    )
    trace(
        session_id, EventKind.NODE_ENTER, "call_loop", "call_start",
        data={"transport_kind": transport_kind},
    )

    receiver_task = asyncio.create_task(_receiver(transport, session))
    try:
        await _process_turns(transport, session)
    finally:
        receiver_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await receiver_task
        trace(session_id, EventKind.NODE_EXIT, "call_loop", "call_end")


async def _receiver(transport: AudioTransport, session: _CallSession) -> None:
    """Only job: keep pulling inbound audio and stamping each with a fresh
    generation number. Runs concurrently with _process_turns so a new
    utterance can be observed while a previous turn is still mid-flight."""
    while True:
        utterance = await transport.receive_utterance()
        if utterance is None:
            await session.queue.put(b"")  # wakes the main loop to end the call
            return
        session.generation += 1
        await session.queue.put(utterance)


async def _process_turns(transport: AudioTransport, session: _CallSession) -> None:
    while True:
        audio = await session.queue.get()
        # Fast-forward through any additional utterances that piled up while
        # this one was queued -- only the most recent is still relevant. An
        # empty marker (hangup) always wins even over queued audio behind it.
        while not session.queue.empty():
            audio = session.queue.get_nowait()
        if not audio:
            return

        my_generation = session.generation
        await _handle_utterance(transport, session, audio, my_generation)


async def _handle_utterance(
    transport: AudioTransport, session: _CallSession, audio: bytes, my_generation: int
) -> None:
    def still_current() -> bool:
        return session.generation == my_generation

    try:
        asr_result = await asyncio.to_thread(get_asr_client().transcribe, audio)
    except ASRUnavailable as exc:
        trace(session.session_id, EventKind.ERROR, "call_loop", "asr_unavailable",
              data={"error": str(exc)})
        if still_current():
            # A pending confirmation, if any, survives an ASR hiccup -- the
            # client must not conclude it was silently dropped.
            await _deliver(
                transport, session, still_current,
                text=_ASR_FAILED_REPLY,
                awaiting_confirmation=session.awaiting_confirmation,
            )
        return

    if session.pinned_language is None:
        # Detected ONCE, at first ASR, and pinned for the rest of the call --
        # see README's "translate at the edge, reason in English" note. Later
        # turns reuse this even if a later utterance's detected_language
        # differs, exactly like the text endpoint's caller-supplied language
        # is expected to stay fixed for a session.
        session.pinned_language = asr_result.detected_language or "en-IN"

    await transport.send_event({
        "type": "transcript", "text": asr_result.text,
        "language": asr_result.detected_language,
    })
    if not still_current():
        return

    resume_value = None
    if session.awaiting_confirmation is not None:
        resume_value = interpret_confirmation(asr_result.text)
        if resume_value is None:
            prompt = session.awaiting_confirmation.get("prompt", "")
            await _deliver(
                transport, session, still_current,
                text=_UNCLEAR_CONFIRMATION_REPLY.format(prompt=prompt),
                awaiting_confirmation=session.awaiting_confirmation,
            )
            return

    result = await asyncio.to_thread(
        run_turn,
        session_id=session.session_id,
        caller_utterance=asr_result.text,
        caller_phone=session.caller_phone,
        language=session.pinned_language,
        transport_kind=session.transport_kind,
        resume_value=resume_value,
    )

    if not still_current():
        # Superseded while ASR/graph was running -- the caller has already
        # moved on. Drop this turn's output rather than talking over them.
        return

    session.awaiting_confirmation = result.awaiting_confirmation
    router = result.state.get("router") or {}
    # composer -- the only node that writes reply_text -- never runs on a
    # turn that pauses at interrupt() (reschedule/address_change return
    # straight to run_turn from inside interrupt()). reply_text is "" on
    # exactly that turn; what the caller needs to actually hear is the
    # read-back line already carried on the interrupt payload itself. The
    # text endpoint gets away with an empty reply_text here because its
    # caller reads awaiting_confirmation.prompt directly -- audio has no such
    # side channel, so call_loop has to do that fallback itself.
    prompt = (result.awaiting_confirmation or {}).get("prompt", "")
    await _deliver(
        transport, session, still_current,
        text=result.reply_text or prompt,
        intent=router.get("intent"),
        awaiting_confirmation=result.awaiting_confirmation,
        escalated=result.state.get("escalated", False),
    )


async def _deliver(
    transport: AudioTransport,
    session: _CallSession,
    still_current,
    *,
    text: str,
    **event_fields,
) -> None:
    """Send the text reply event, then stream its TTS audio -- checking for
    barge-in (still_current) between every outbound chunk."""
    await transport.send_event({"type": "reply", "text": text, **event_fields})
    if not still_current():
        return
    if not text:
        # No speakable text at all (shouldn't normally happen -- see the
        # reply_text-or-prompt fallback above -- but a client waiting on a
        # definite end-of-turn signal must never be left hanging).
        await transport.send_event({"type": "audio_end"})
        return

    try:
        tts_result = await asyncio.to_thread(
            get_tts_client().synthesize, text, session.pinned_language or "en-IN"
        )
    except TTSUnavailable as exc:
        trace(session.session_id, EventKind.ERROR, "call_loop", "tts_unavailable",
              data={"error": str(exc), "note": _TTS_FAILED_NOTE})
        if still_current():
            await transport.send_event({"type": "error", "detail": "tts_unavailable"})
        return

    if not still_current():
        return

    chunk_size = get_settings().ws_audio_chunk_bytes
    audio = tts_result.audio_bytes
    for offset in range(0, len(audio), chunk_size):
        if not still_current():
            trace(session.session_id, EventKind.DECISION, "call_loop", "barge_in",
                  reasoning="new caller audio arrived while streaming a reply")
            await transport.send_event({"type": "barge_in"})
            return
        await transport.send_audio_chunk(audio[offset:offset + chunk_size])

    await transport.send_event({"type": "audio_end"})
