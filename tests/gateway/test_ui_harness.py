"""
Gradio test harness tests.

Pure logic (app/ui/harness.py) gets real coverage, same fresh_env shape as
test_graph.py -- these functions ARE the harness, in every sense that
matters for correctness. app/ui/blocks.py (the Gradio wiring itself) gets
only a smoke test: that it builds without error and that mounting it doesn't
break any existing route, plus the analytics-disabled regression below. Gradio
event-handler bodies are thin enough (a few lines each, no branching logic of
their own beyond what harness.py already returns) that testing them via a
real browser/Gradio client would mostly be testing Gradio itself.
"""

from __future__ import annotations

import re

import pytest
from app.providers.llm_client import reset_llm_client
from app.providers.order_client import InProcessOrderClient, reset_order_client
from app.ui.harness import (
    get_explain,
    new_session_id,
    run_audio_turn,
    run_text_turn,
)
from order_api.kv import reset_backend
from order_api.seed import build_fixtures
from order_api.store import get_store

RAVI = "9990000002"  # one open order: DLV1004 (FAILED_ATTEMPT)


@pytest.fixture(autouse=True)
def fresh_env(tmp_path, monkeypatch):
    reset_backend()
    reset_order_client()
    reset_llm_client()
    import order_api.store as store_mod

    store_mod._store = None
    store = get_store()
    store.reset()
    store.load(build_fixtures())

    monkeypatch.setenv("CHECKPOINT_DSN", f"sqlite:///{tmp_path / 'checkpoints.db'}")
    from app.config import get_settings

    get_settings.cache_clear()

    yield store

    get_settings.cache_clear()


# ---- new_session_id ---------------------------------------------------


def test_new_session_id_is_unique_each_call():
    assert new_session_id() != new_session_id()


# ---- run_text_turn ------------------------------------------------------


def test_run_text_turn_order_lookup():
    outcome = run_text_turn(new_session_id(), RAVI, "where is my order")
    assert outcome.intent == "order_lookup"
    assert "delivery attempt" in outcome.reply_text.lower()
    assert outcome.awaiting_confirmation is None
    assert outcome.error is None


def test_run_text_turn_reschedule_pauses_for_confirmation():
    outcome = run_text_turn(new_session_id(), RAVI, "reschedule to in 3 days evening")
    assert outcome.awaiting_confirmation["kind"] == "confirm_reschedule"
    # reply_text falls back to the interrupt's prompt -- composer never ran.
    assert outcome.reply_text == outcome.awaiting_confirmation["prompt"]


def test_run_text_turn_confirms_with_a_clear_yes():
    session_id = new_session_id()
    first = run_text_turn(session_id, RAVI, "reschedule to in 3 days evening")
    second = run_text_turn(session_id, RAVI, "yes", pending_confirmation=first.awaiting_confirmation)
    assert second.awaiting_confirmation is None
    order = InProcessOrderClient().get_order("DLV1004")
    assert order.reschedule_count == 1


def test_run_text_turn_declines_with_a_clear_no():
    session_id = new_session_id()
    first = run_text_turn(session_id, RAVI, "reschedule to in 3 days evening")
    second = run_text_turn(session_id, RAVI, "no", pending_confirmation=first.awaiting_confirmation)
    assert second.awaiting_confirmation is None
    order = InProcessOrderClient().get_order("DLV1004")
    assert order.reschedule_count == 0


def test_run_text_turn_reasks_on_an_unclear_reply_without_resuming():
    session_id = new_session_id()
    first = run_text_turn(session_id, RAVI, "reschedule to in 3 days evening")
    second = run_text_turn(session_id, RAVI, "maybe later", pending_confirmation=first.awaiting_confirmation)
    assert "yes or a no" in second.reply_text.lower()
    assert second.awaiting_confirmation == first.awaiting_confirmation
    order = InProcessOrderClient().get_order("DLV1004")
    assert order.reschedule_count == 0


def test_run_text_turn_self_harm_input_is_blocked_by_the_graph_guardrail():
    """The harness doesn't reimplement guardrails -- it just calls run_turn(),
    so phase 5's protection applies here for free."""
    outcome = run_text_turn(new_session_id(), RAVI, "I want to kill myself")
    assert outcome.escalated is True


# ---- run_audio_turn -------------------------------------------------------


def test_run_audio_turn_transcribes_and_replies():
    outcome = run_audio_turn(new_session_id(), RAVI, b"hi-IN::where is my order")
    assert outcome.transcript == "where is my order"
    assert outcome.detected_language == "hi-IN"
    assert outcome.turn.intent == "order_lookup"
    assert outcome.reply_audio_bytes is not None
    assert outcome.reply_audio_bytes[:4] == b"RIFF"


def test_run_audio_turn_confirmation_round_trip():
    session_id = new_session_id()
    first = run_audio_turn(session_id, RAVI, b"reschedule to in 3 days evening")
    assert first.turn.awaiting_confirmation["kind"] == "confirm_reschedule"
    second = run_audio_turn(
        session_id, RAVI, b"yes", pending_confirmation=first.turn.awaiting_confirmation,
    )
    assert second.turn.awaiting_confirmation is None
    order = InProcessOrderClient().get_order("DLV1004")
    assert order.reschedule_count == 1


def test_run_audio_turn_pins_detected_language_for_tts():
    """Not directly observable from reply_audio_bytes (the mock TTS client's
    audio doesn't encode language), but the call must not raise, and the
    turn's own language field should reflect what ASR actually detected."""
    outcome = run_audio_turn(new_session_id(), RAVI, b"te-IN::where is my order")
    assert outcome.detected_language == "te-IN"
    assert outcome.reply_audio_bytes is not None


# ---- get_explain ------------------------------------------------------


def test_get_explain_reflects_the_turn_just_run():
    session_id = new_session_id()
    run_text_turn(session_id, RAVI, "where is my order")
    payload = get_explain(session_id)
    assert payload["event_count"] > 0
    assert payload["session_id"] == session_id


def test_get_explain_empty_for_unknown_session():
    payload = get_explain("no-such-session-ever")
    assert payload["event_count"] == 0


# ---- blocks.py: build + mount smoke tests ----------------------------------


def test_build_demo_disables_analytics_and_never_calls_gradios_telemetry(monkeypatch):
    """
    Regression test: gr.Blocks() WITHOUT analytics_enabled=False spins up a
    background thread that calls out to api.gradio.app and huggingface.co --
    a real network call from what must be an offline test suite (confirmed
    manually while building this harness). Patch the two functions Gradio's
    own Blocks.__init__ calls when analytics is on, so this fails loudly if
    analytics_enabled=False is ever dropped, regardless of which HTTP client
    is used underneath.
    """
    import gradio.analytics as gr_analytics

    def _explode(*args, **kwargs):
        raise AssertionError("Gradio must never phone home from this test suite")

    monkeypatch.setattr(gr_analytics, "version_check", _explode)
    monkeypatch.setattr(gr_analytics, "initiated_analytics", _explode)

    from app.ui.blocks import build_demo

    demo = build_demo()
    assert demo.analytics_enabled is False


def test_ui_is_mounted_and_existing_routes_still_work():
    from app.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ui/").status_code == 200
        resp = client.post(
            "/call/turn",
            json={"session_id": "uismoke1", "text": "where is my order", "caller_phone": RAVI},
        )
        assert resp.status_code == 200
        assert resp.json()["intent"] == "order_lookup"


def test_ui_declares_its_own_mount_path_as_root():
    """
    Regression test for a real bug found in an actual browser against the
    live deployment, not by any automated check -- a TestClient GET on
    "/ui/" only checks the initial HTML's status code and never exercises
    this at all. Without root_path="/ui" on gr.mount_gradio_app, Gradio's own
    frontend JS calls its API (queue/join, upload, ...) at the SITE ROOT
    instead of under the mount, 404ing on every real interaction while the
    page itself still loaded fine. Gradio embeds the root it thinks it's
    served from directly in the page as JSON; assert it's actually "/ui",
    not the bare origin.
    """
    from app.main import app
    from fastapi.testclient import TestClient

    with TestClient(app, base_url="http://testserver") as client:
        html = client.get("/ui/").text

    match = re.search(r'"root"\s*:\s*"([^"]*)"', html)
    assert match is not None, "Gradio's embedded config should declare a root path"
    assert match.group(1).endswith("/ui"), (
        f"Gradio thinks its root is {match.group(1)!r} -- API calls (queue/join, "
        "upload) will 404 against the real mount at /ui"
    )
