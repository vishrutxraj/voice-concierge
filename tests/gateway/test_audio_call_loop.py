"""
app.audio.call_loop tests -- streaming and barge-in, at the unit level.

Deliberately NOT driven through a real WebSocket here: a FakeAudioTransport
implementing the AudioTransport ABC lets a test control exactly when the
"caller" speaks relative to how much of the previous reply's audio has been
sent, with zero wall-clock races. tests/gateway/test_ws_endpoint.py covers the
real /ws/call wire protocol end-to-end; this file covers the concurrency
behaviour that a real-timing integration test could only assert on flakily.

Same fixture shape as test_graph.py (isolated store/checkpointer) plus
test_asr_tts_clients.py's isolated cache dir, combined, because call_loop
exercises the full stack: ASRClient -> run_turn -> TTSClient.
"""

from __future__ import annotations

import asyncio

import pytest
from app.audio.call_loop import run_call
from app.audio.transport import AudioTransport
from app.providers.asr_client import reset_asr_client
from app.providers.cache import reset_caches
from app.providers.llm_client import reset_llm_client
from app.providers.order_client import InProcessOrderClient, reset_order_client
from app.providers.tts_client import reset_tts_client
from order_api.kv import reset_backend
from order_api.seed import build_fixtures
from order_api.store import get_store

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


class FakeAudioTransport(AudioTransport):
    """
    Records every outbound event/chunk into one ordered log so a test can
    assert on exact interleaving. on_chunk, if set, fires synchronously (from
    inside send_audio_chunk) after each chunk is recorded -- tests use it to
    inject a new inbound utterance at a precise point in the outbound stream.
    """

    def __init__(self) -> None:
        self.inbound: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.log: list[tuple[str, object]] = []
        self.on_chunk = None

    async def receive_utterance(self) -> bytes | None:
        return await self.inbound.get()

    async def send_event(self, event: dict) -> None:
        self.log.append(("event", event))

    async def send_audio_chunk(self, chunk: bytes) -> None:
        self.log.append(("chunk", chunk))
        if self.on_chunk is not None:
            n = sum(1 for kind, _ in self.log if kind == "chunk")
            await self.on_chunk(n)

    def events(self, event_type: str | None = None) -> list[dict]:
        out = [v for k, v in self.log if k == "event"]
        return out if event_type is None else [e for e in out if e["type"] == event_type]


async def _wait_until(predicate, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            pytest.fail("condition never became true")
        await asyncio.sleep(0.01)


# ---- basic round trip -------------------------------------------------


async def test_full_turn_produces_transcript_reply_and_streamed_audio():
    transport = FakeAudioTransport()
    await transport.inbound.put(b"where is my order")

    task = asyncio.create_task(
        run_call(transport, "call1", caller_phone=RAVI, transport_kind="text")
    )
    await _wait_until(lambda: transport.events("audio_end"))
    await transport.inbound.put(None)
    await asyncio.wait_for(task, timeout=5)

    kinds = [e["type"] for e in transport.events()]
    assert kinds == ["transcript", "reply", "audio_end"]

    reply = transport.events("reply")[0]
    assert reply["intent"] == "order_lookup"
    assert "delivery attempt" in reply["text"].lower()

    chunks = [v for k, v in transport.log if k == "chunk"]
    assert len(chunks) > 1, "a real reply must be streamed as more than one chunk"
    audio = b"".join(chunks)
    assert audio[:4] == b"RIFF"  # real WAV, not the raw JSON/base64 (MockTTSClient)


async def test_language_is_pinned_from_first_utterance_only():
    """
    hi-IN on turn 1, te-IN on turn 2 -- TTS must still be asked to speak Hindi
    for both turns, matching the architecture's "detected once, pinned to the
    session" rule (see README). We can't observe the TTS call's language
    argument directly through the transport, but the mock TTS client's cache
    key includes it, so two different pinned languages would double the
    cache's entry count instead of hitting the same key on the second turn's
    identical reply text pattern -- easiest direct check is via the ASR
    transcript events, which report what was actually SPOKEN each turn
    (unaffected by pinning), while call_loop's pinned_language stays fixed
    internally. We assert on that field name directly on the session object.
    """
    from app.audio.call_loop import _CallSession, _handle_utterance

    session = _CallSession(session_id="pin1", caller_phone=RAVI, transport_kind="text")
    transport = FakeAudioTransport()

    await _handle_utterance(transport, session, b"hi-IN::where is my order", 0)
    assert session.pinned_language == "hi-IN"

    await _handle_utterance(transport, session, b"te-IN::where is my order", 0)
    assert session.pinned_language == "hi-IN", "must not repin on a later utterance"


# ---- barge-in -----------------------------------------------------------


async def test_barge_in_cuts_audio_short_and_processes_the_new_utterance():
    transport = FakeAudioTransport()

    async def on_chunk(n: int) -> None:
        if n == 3:
            await transport.inbound.put(b"reschedule to in 3 days evening")
            await asyncio.sleep(0.02)  # let the receiver task observe it

    transport.on_chunk = on_chunk
    await transport.inbound.put(b"where is my order")

    task = asyncio.create_task(
        run_call(transport, "call2", caller_phone=RAVI, transport_kind="text")
    )
    await _wait_until(lambda: len(transport.events("reply")) >= 2)
    await transport.inbound.put(None)
    await asyncio.wait_for(task, timeout=5)

    kinds = [k for k, _ in transport.log]
    event_kinds = [v["type"] for k, v in transport.log if k == "event"]
    assert "barge_in" in event_kinds

    barge_in_pos = next(
        i for i, (k, v) in enumerate(transport.log) if k == "event" and v["type"] == "barge_in"
    )
    chunks_before_barge_in = sum(1 for k, _ in transport.log[:barge_in_pos] if k == "chunk")
    assert chunks_before_barge_in == 3, "must stop exactly when generation was superseded"

    # No audio_end for the interrupted turn -- barge_in replaces it.
    audio_ends_before_barge_in = sum(
        1 for k, v in transport.log[:barge_in_pos] if k == "event" and v["type"] == "audio_end"
    )
    assert audio_ends_before_barge_in == 0

    replies = transport.events("reply")
    assert replies[0]["intent"] == "order_lookup"
    assert replies[1]["awaiting_confirmation"]["kind"] == "confirm_reschedule"

    transcripts = transport.events("transcript")
    assert transcripts[1]["text"] == "reschedule to in 3 days evening"

    # The second turn's own audio streamed and completed normally.
    assert kinds.count("event") >= 1
    assert transport.events("audio_end"), "the NEW turn's audio must still complete"


async def test_barge_in_before_first_turn_starts_drops_the_stale_reply():
    """
    Two utterances queued back-to-back before the loop even starts processing
    -- the first can be superseded before (or shortly after) its own turn
    begins. Regardless of exactly when that's noticed, its REPLY must never
    reach the transport, and the caller ends up talking to the agent about
    what they actually said last (the fast-forward drain in _process_turns
    handles the common case; the generation check in _handle_utterance is the
    backstop either way).
    """
    transport = FakeAudioTransport()
    await transport.inbound.put(b"where is my order")
    await transport.inbound.put(b"reschedule to in 3 days evening")

    task = asyncio.create_task(
        run_call(transport, "call3", caller_phone=RAVI, transport_kind="text")
    )
    await _wait_until(lambda: transport.events("reply"))
    await asyncio.sleep(0.05)  # let a stray stale reply land too, if it ever could
    await transport.inbound.put(None)
    await asyncio.wait_for(task, timeout=5)

    replies = transport.events("reply")
    assert len(replies) == 1, "the superseded utterance must never produce its own reply"
    assert replies[0]["awaiting_confirmation"]["kind"] == "confirm_reschedule"


# ---- confirmation interpretation ----------------------------------------


async def test_unclear_confirmation_reply_reasks_without_resuming():
    transport = FakeAudioTransport()
    await transport.inbound.put(b"reschedule to in 3 days evening")

    task = asyncio.create_task(
        run_call(transport, "call4", caller_phone=RAVI, transport_kind="text")
    )
    await _wait_until(lambda: len(transport.events("reply")) >= 1)
    first_reply = transport.events("reply")[0]
    assert first_reply["awaiting_confirmation"]["kind"] == "confirm_reschedule"

    await transport.inbound.put(b"maybe later")
    await _wait_until(lambda: len(transport.events("reply")) >= 2)
    await transport.inbound.put(None)
    await asyncio.wait_for(task, timeout=5)

    second_reply = transport.events("reply")[1]
    assert "yes or a no" in second_reply["text"].lower()
    assert second_reply["awaiting_confirmation"]["kind"] == "confirm_reschedule"

    order = InProcessOrderClient().get_order("DLV1004")
    assert order.reschedule_count == 0, "an ambiguous reply must never commit a write"


async def test_clear_yes_resumes_and_commits():
    transport = FakeAudioTransport()
    await transport.inbound.put(b"reschedule to in 3 days evening")

    task = asyncio.create_task(
        run_call(transport, "call5", caller_phone=RAVI, transport_kind="text")
    )
    await _wait_until(lambda: len(transport.events("reply")) >= 1)

    await transport.inbound.put(b"yes please")
    await _wait_until(lambda: len(transport.events("reply")) >= 2)
    await transport.inbound.put(None)
    await asyncio.wait_for(task, timeout=5)

    second_reply = transport.events("reply")[1]
    assert second_reply["awaiting_confirmation"] is None
    order = InProcessOrderClient().get_order("DLV1004")
    assert order.reschedule_count == 1


async def test_clear_no_declines_without_committing():
    transport = FakeAudioTransport()
    await transport.inbound.put(b"reschedule to in 3 days evening")

    task = asyncio.create_task(
        run_call(transport, "call6", caller_phone=RAVI, transport_kind="text")
    )
    await _wait_until(lambda: len(transport.events("reply")) >= 1)

    await transport.inbound.put(b"no")
    await _wait_until(lambda: len(transport.events("reply")) >= 2)
    await transport.inbound.put(None)
    await asyncio.wait_for(task, timeout=5)

    second_reply = transport.events("reply")[1]
    assert second_reply["awaiting_confirmation"] is None
    order = InProcessOrderClient().get_order("DLV1004")
    assert order.reschedule_count == 0
