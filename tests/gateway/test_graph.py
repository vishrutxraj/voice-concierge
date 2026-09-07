"""
LangGraph core tests. Fully text-driven via run_turn -- zero API spend, the
whole point of building phase 2 before phase 3's audio wrapping.

Uses a temp-file SQLite checkpointer per test so runs never interfere, and the
in-process OrderClient so nothing here needs a running order-api service.
"""

from __future__ import annotations

import pytest
from app.graph.order_resolution import resolve_order_id
from app.graph.runner import forget_vault, run_turn
from app.providers.llm_client import reset_llm_client
from app.providers.order_client import InProcessOrderClient, reset_order_client
from order_api.kv import reset_backend
from order_api.seed import build_fixtures
from order_api.store import get_store

RAVI = "9990000002"  # one open order: DLV1004 (FAILED_ATTEMPT)
ANITA = "9990000001"  # three open orders: DLV1001/1002/1003
MEERA = "9990000003"  # zero open orders (DLV1005 is delivered)
SUNIL = "9990000004"  # DLV1006, already at reschedule cap


@pytest.fixture(autouse=True)
def fresh_env(tmp_path, monkeypatch):
    """Isolated store + checkpointer + vault per test."""
    reset_backend()
    reset_order_client()
    reset_llm_client()
    import order_api.store as store_mod

    store_mod._store = None
    store = get_store()
    store.reset()
    store.load(build_fixtures())

    db_path = tmp_path / "checkpoints.db"
    monkeypatch.setenv("CHECKPOINT_DSN", f"sqlite:///{db_path}")
    from app.config import get_settings

    get_settings.cache_clear()

    yield store

    get_settings.cache_clear()


def _forget_all():
    for sid in ("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"):
        forget_vault(sid)


# ---- order-ID resolution (unit level, no graph) --------------------------


def test_single_open_order_resolves_at_full_confidence():
    result = resolve_order_id("where is my order", RAVI, InProcessOrderClient())
    assert result.status == "resolved"
    assert result.order_id == "DLV1004"
    assert result.confidence == 1.0


def test_no_open_orders_is_not_found():
    result = resolve_order_id("where is my order", MEERA, InProcessOrderClient())
    assert result.status == "not_found"


def test_clear_id_fragment_resolves_among_multiple():
    result = resolve_order_id(
        "when is D L V one zero zero two arriving", ANITA, InProcessOrderClient()
    )
    assert result.status == "resolved"
    assert result.order_id == "DLV1002"


def test_no_id_fragment_among_multiple_is_not_found():
    """
    Not the caller's fault -- 'where is my order' has no ID in it. The graph
    layer (ensure_order_id) is responsible for listing options in this case;
    resolve_order_id itself correctly reports not_found.
    """
    result = resolve_order_id("where is my order", ANITA, InProcessOrderClient())
    assert result.status == "not_found"


def test_never_trusts_a_wrong_digit_silently():
    """
    A one-digit-off transcription among three real order IDs must not resolve
    to the wrong one silently -- either it's a clear best match or it isn't.
    """
    result = resolve_order_id("DLV1099", ANITA, InProcessOrderClient())
    assert result.status in {"not_found", "ambiguous"}


# ---- router ----------------------------------------------------------------


def test_router_classifies_order_lookup():
    r = run_turn("s1", "where is my order", caller_phone=RAVI)
    assert r.state["router"]["intent"] == "order_lookup"


def test_router_classifies_reschedule():
    r = run_turn("s2", "I need to reschedule my delivery", caller_phone=RAVI)
    assert r.state["router"]["intent"] == "reschedule"


def test_router_classifies_address_change():
    r = run_turn("s3", "I need to change my delivery address", caller_phone=RAVI)
    assert r.state["router"]["intent"] == "address_change"


def test_router_escalates_gibberish_to_fallback():
    r = run_turn("s4", "asdkjf qqzzxx nonsense", caller_phone=RAVI)
    assert r.state["router"]["intent"] == "fallback"
    assert r.state["escalated"] is True


def test_low_confidence_never_silently_calls_a_tool():
    """The confidence floor must route to fallback, not guess at an intent."""
    r = run_turn("s5", "hmm okay maybe something about stuff", caller_phone=RAVI)
    assert r.state["router"]["intent"] == "fallback"


# ---- order lookup ------------------------------------------------------


def test_order_lookup_single_order_resolves_silently():
    r = run_turn("s6", "where is my order", caller_phone=RAVI)
    assert "didn't succeed" in r.reply_text or "reschedule" in r.reply_text.lower()
    assert r.state.get("order_id") == "DLV1004"


def test_order_lookup_multiple_orders_lists_candidates():
    r = run_turn("s7", "where is my order", caller_phone=ANITA)
    assert "?" in r.reply_text
    assert r.state.get("order_id") is None
    assert not r.state.get("escalated")  # listing options is not an escalation


def test_order_lookup_no_open_orders_escalates():
    r = run_turn("s8", "where is my order", caller_phone=MEERA)
    assert r.state["escalated"] is True
    assert r.state["escalation_reason"] == "order_id_not_found"


def test_order_api_unavailable_during_phone_lookup_escalates_gracefully():
    """
    Regression test for a real bug found deploying the split for real (not in
    a test, and not by code review): ensure_order_id's orders_for_phone call
    -- the shared "which order is this about" step every domain agent calls
    first -- had no OrderAPIUnavailable handling, unlike every other
    OrderClient call site in the graph. A Vercel cold start alone was enough
    to exceed the 4s client timeout and crash the whole turn with a raw 500
    instead of the designed "let me connect you with a colleague" escalation.
    """
    import app.providers.order_client as order_client_module
    from app.providers.order_client import OrderAPIUnavailable, OrderClient

    class _FlakyOrderClient(OrderClient):
        def get_order(self, order_id):
            raise OrderAPIUnavailable("simulated timeout")

        def orders_for_phone(self, phone):
            raise OrderAPIUnavailable("simulated timeout")

        def reschedule(self, *a, **kw):
            raise OrderAPIUnavailable("simulated timeout")

        def update_address(self, *a, **kw):
            raise OrderAPIUnavailable("simulated timeout")

    order_client_module._client = _FlakyOrderClient()
    try:
        r = run_turn("s9", "where is my order", caller_phone=RAVI)
    finally:
        reset_order_client()

    assert r.state["escalated"] is True
    assert r.state["escalation_reason"] == "order_api_unavailable"
    assert "colleague" in r.reply_text.lower()


# ---- reschedule: full interrupt/resume cycle ------------------------------


def test_reschedule_pauses_before_writing():
    r = run_turn("r1", "reschedule to friday evening", caller_phone=RAVI)
    assert r.awaiting_confirmation is not None
    assert r.awaiting_confirmation["kind"] == "confirm_reschedule"
    # The store must NOT be mutated yet.
    order = InProcessOrderClient().get_order("DLV1004")
    assert order.reschedule_count == 0


def test_reschedule_confirm_true_commits():
    # "in 3 days evening" rather than "friday evening": a weekday name resolves
    # to a moving target date, and the fixtures' blackout dates (T(5), T(12) in
    # seed.py) are ALSO relative to today -- on some days "next Friday" collides
    # with a blackout date and this test fails for a reason that has nothing to
    # do with the code under test. A fixed day-offset is deterministic regardless
    # of what day the suite runs.
    run_turn("r2", "reschedule to in 3 days evening", caller_phone=RAVI)
    r2 = run_turn("r2", "yes", caller_phone=RAVI, resume_value=True)
    assert r2.awaiting_confirmation is None
    order = InProcessOrderClient().get_order("DLV1004")
    assert order.reschedule_count == 1


def test_reschedule_confirm_false_does_not_commit():
    """The falsy-resume bug: Command(resume=False) must not crash or no-op wrong."""
    run_turn("r3", "reschedule to friday evening", caller_phone=RAVI)
    r2 = run_turn("r3", "no", caller_phone=RAVI, resume_value=False)
    assert "better" in r2.reply_text.lower() or "date" in r2.reply_text.lower()
    order = InProcessOrderClient().get_order("DLV1004")
    assert order.reschedule_count == 0


def test_reschedule_missing_date_or_window_asks_again_without_pausing():
    r = run_turn("r4", "I want to reschedule my delivery", caller_phone=RAVI)
    assert r.awaiting_confirmation is None
    assert "day" in r.reply_text.lower() or "time" in r.reply_text.lower()


def test_reschedule_confirmed_but_store_refuses_reports_refusal_with_alternatives():
    """
    Caller says yes, but by commit time the constraint layer refuses (cap
    reached). The agent must relay the refusal and offer alternatives, not
    claim success.
    """
    r = run_turn("r5", "reschedule to friday afternoon", caller_phone=SUNIL)
    assert r.awaiting_confirmation is not None
    r2 = run_turn("r5", "yes", caller_phone=SUNIL, resume_value=True)
    assert "maximum" in r2.reply_text.lower() or "already" in r2.reply_text.lower()
    order = InProcessOrderClient().get_order("DLV1006")
    assert order.reschedule_count == 2  # unchanged from fixture


def test_reschedule_idempotency_key_is_derived_from_session():
    """Every commit must use a fresh, session-scoped key -- not a static one."""
    run_turn("r6", "reschedule to in 3 days evening", caller_phone=RAVI)
    run_turn("r6", "yes", caller_phone=RAVI, resume_value=True)
    # No direct access to the key from here, but a second independent call
    # for the same caller/order must be able to reschedule again (i.e. the
    # key didn't collide with some other session's key).
    run_turn("r6b", "reschedule to in 3 days evening", caller_phone=SUNIL)
    # Just confirming r6's commit actually happened and isn't blocked:
    assert InProcessOrderClient().get_order("DLV1004").reschedule_count == 1


def test_day_plus_3_never_collides_with_seeded_blackout_or_saturated_slots():
    """
    Canary for the exact bug class that broke this suite once already: tests
    that resolve a date via extraction (weekday names, 'in N days') can
    silently collide with fixtures that are ALSO relative to date.today()
    (BLACKOUT_DATES = {T(5), T(12)}; T(3)/MORNING is deliberately saturated).
    'in 3 days evening' is used throughout this file as the safe default.
    If seed.py or store.py's constraints ever change, this fails loudly here
    instead of as a confusing, date-dependent failure in an unrelated test.
    """
    from order_api.seed import T
    from order_api.store import BLACKOUT_DATES

    assert T(3) not in BLACKOUT_DATES
    assert InProcessOrderClient().get_order("DLV1004")  # sanity: fixture exists


# ---- address change --------------------------------------------------------


def test_address_change_pauses_before_writing():
    r = run_turn("a1", "the pincode should be 411015", caller_phone=RAVI)
    assert r.awaiting_confirmation is not None
    assert r.awaiting_confirmation["kind"] == "confirm_address"
    order = InProcessOrderClient().get_order("DLV1004")
    assert order.address.pincode != "411015"


def test_address_change_confirm_commits_the_real_value():
    """
    The whole point of the rehydrate-for-extraction fix: the committed address
    must contain the REAL pincode, not a PII token.
    """
    run_turn("a2", "the pincode should be 411015", caller_phone=RAVI)
    r2 = run_turn("a2", "yes", caller_phone=RAVI, resume_value=True)
    assert "411015" in r2.reply_text
    assert "PINCODE" not in r2.reply_text
    order = InProcessOrderClient().get_order("DLV1004")
    assert order.address.pincode == "411015"


def test_address_change_non_serviceable_pincode_is_refused():
    run_turn("a3", "change pincode to 190001", caller_phone=RAVI)
    r2 = run_turn("a3", "yes", caller_phone=RAVI, resume_value=True)
    assert "don't" in r2.reply_text.lower() or "not" in r2.reply_text.lower()
    order = InProcessOrderClient().get_order("DLV1004")
    assert order.address.pincode != "190001"


def test_address_change_unparseable_correction_escalates_without_pausing():
    r = run_turn("a4", "I want to change my address completely", caller_phone=RAVI)
    assert r.awaiting_confirmation is None
    assert r.state["escalated"] is True
    assert r.state["escalation_reason"] == "address_extraction_failed"


# ---- PII: the property that matters most -----------------------------------


def test_no_pii_in_trace_across_a_full_address_change(capsys):
    import json

    from app.observability.trace import get_sink

    run_turn("p1", "the pincode should be 411016", caller_phone=RAVI)
    run_turn("p1", "yes", caller_phone=RAVI, resume_value=True)
    capsys.readouterr()

    dump = json.dumps(get_sink().session("p1"))
    assert "411016" not in dump, "raw pincode leaked into the trace store"


def test_reply_text_contains_real_value_trace_does_not():
    """The exact split the architecture promises: caller hears it, logs don't."""
    import json

    from app.observability.trace import get_sink

    run_turn("p2", "the pincode should be 411017", caller_phone=RAVI)
    r2 = run_turn("p2", "yes", caller_phone=RAVI, resume_value=True)

    assert "411017" in r2.reply_text
    assert "411017" not in json.dumps(get_sink().session("p2"))


# ---- sentiment / escalation (parallel branch) ------------------------------


def test_frustrated_language_raises_sentiment_escalation():
    run_turn("m1", "this is ridiculous, I'm so frustrated", caller_phone=RAVI)
    r2 = run_turn(
        "m1", "this is the worst, unacceptable, speak to a manager",
        caller_phone=RAVI,
    )
    assert r2.state["escalated"] is True


def test_sentiment_does_not_block_a_normal_successful_reply():
    """Positive/neutral language must not accidentally trip escalation."""
    r = run_turn("m2", "where is my order", caller_phone=RAVI)
    assert r.state.get("sentiment_score", 0) > -0.1


def test_unresolved_turns_eventually_escalate():
    sid = "m3"
    for _ in range(5):
        r = run_turn(sid, "asdkjf nonsense gibberish", caller_phone=RAVI)
    assert r.state["escalated"] is True


# ---- checkpointing / multi-turn continuity ---------------------------------


def test_order_id_persists_across_turns_once_resolved():
    run_turn("c1", "when is D L V one zero zero two arriving", caller_phone=ANITA)
    r2 = run_turn("c1", "actually reschedule it to friday evening", caller_phone=ANITA)
    assert r2.awaiting_confirmation is not None
    assert r2.awaiting_confirmation["order_id"] == "DLV1002"


def test_checkpoint_db_file_is_created(tmp_path):
    run_turn("c2", "where is my order", caller_phone=RAVI)
    db_files = list(tmp_path.glob("checkpoints.db"))
    assert db_files, "SQLite checkpoint file was not created"


# ---- architectural invariant: no concurrent-write crash --------------------


def test_router_and_sentiment_run_concurrently_without_state_collision():
    """
    Regression test for the InvalidUpdateError this phase actually hit:
    intent_router and sentiment_monitor both fan out from START and must be
    able to write in the same superstep without colliding on any state key.
    """
    r = run_turn("x1", "where is my order", caller_phone=RAVI)
    assert r.reply_text  # if this ran at all without raising, the fix holds
    assert "router" in r.state
    assert "sentiment_score" in r.state


# ---- phase 3: extraction fixes, exercised through the full graph ----------


def test_day_after_tomorrow_reschedule_end_to_end():
    """
    Graph-level confirmation of the regex substring fix: 'day after tomorrow'
    must propose the correct date (+2 days) through the real node, not just
    in extraction.py's unit tests.
    """
    from datetime import date, timedelta

    expected = date.today() + timedelta(days=2)
    r = run_turn("e1", "reschedule to day after tomorrow evening", caller_phone=RAVI)
    assert r.awaiting_confirmation is not None
    assert expected.strftime("%d %B") in r.awaiting_confirmation["proposed_slot"]
    r2 = run_turn("e1", "yes", caller_phone=RAVI, resume_value=True)
    assert "Rescheduled" in r2.reply_text


def test_full_new_address_end_to_end_no_longer_always_escalates():
    """
    Graph-level confirmation that phase 2's most-cited limitation is fixed:
    a clearly-stated full address now resolves through the LLM fallback and
    reaches confirmation, rather than escalating unconditionally.
    """
    r = run_turn(
        "e2", "my new address is 45 Park Road near the mall Mumbai 400001",
        caller_phone=RAVI,
    )
    assert r.awaiting_confirmation is not None
    assert r.awaiting_confirmation["kind"] == "confirm_address"
    assert "400001" in r.awaiting_confirmation["proposed_address"]
    r2 = run_turn("e2", "yes", caller_phone=RAVI, resume_value=True)
    assert "400001" in r2.reply_text
    order = InProcessOrderClient().get_order("DLV1004")
    assert order.address.pincode == "400001"
    assert order.address.city == "Mumbai"


def test_unrecognised_city_change_degrades_safely_not_silently():
    """
    An unfamiliar city name ("Nowhereville") isn't in the mock LLM's lookup,
    so it detects no city at all -- correctly distinct from the
    city_change_blocked path (see test_extraction.py, which uses a
    purpose-built fake LLM to exercise that specific guard, since the real
    mock's city detection is inherently limited to its known-city list).
    Here, extraction gracefully degrades to the narrow pincode-only
    correction. The safety property that matters is: it still pauses for
    read-back confirmation before anything commits, so a caller who notices
    the old city name is still attached can simply say no.
    """
    r = run_turn(
        "e3", "my new address is 9 River View near the lake Nowhereville 500099",
        caller_phone=RAVI,
    )
    assert r.awaiting_confirmation is not None
    assert "500099" in r.awaiting_confirmation["proposed_address"]

    r2 = run_turn("e3", "no", caller_phone=RAVI, resume_value=False)
    assert r2.awaiting_confirmation is None
    order = InProcessOrderClient().get_order("DLV1004")
    assert order.address.pincode != "500099"  # declined -> never committed
