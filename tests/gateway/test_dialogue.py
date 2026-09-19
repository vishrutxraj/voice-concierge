"""
Multi-turn dialogue: answering a question the agent just asked.

Regression tests for a real session (Telugu caller, three open orders): the
agent asked "which order?", the caller answered "yoga mat", and was handed to
a human colleague. The router classified the bare answer cold (0.1 confidence),
and the order matcher only understood order IDs, not item names.

The LLM here is SCRIPTED to misbehave exactly like the real one did in that
session -- the mock's keyword classifier would happily route some of these
phrases correctly and hide the bug. Overrides are keyed by utterance.
"""

from __future__ import annotations

import json
import uuid

import pytest
from app.graph.nodes.order_identify import _ask
from app.graph.order_reference import OrderRef
from app.graph.order_resolution import resolve_order_id
from app.graph.runner import run_turn
from app.observability.trace import get_sink
from app.providers.llm_client import (
    LLMClient,
    LLMResponse,
    MockLLMClient,
    reset_llm_client,
)
from app.providers.order_client import InProcessOrderClient, reset_order_client
from order_api.kv import reset_backend
from order_api.seed import build_fixtures
from order_api.store import get_store

RAVI = "9990000002"  # one open order: DLV1004 (FAILED_ATTEMPT)
ANITA = "9990000001"  # three open orders: DLV1001 speaker (out for delivery),
#                       DLV1002 bedsheets+pillow covers (in transit), DLV1003 yoga mat (pending)


class ScriptedLLM(LLMClient):
    """Mock LLM, except classification of specific utterances is forced to what
    the real model actually returned."""

    def __init__(self, overrides: dict[str, tuple[str, float]] | None = None) -> None:
        self._mock = MockLLMClient()
        self._overrides = overrides or {}
        self.classified: list[str] = []

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> LLMResponse:
        if "classify" in system.lower():
            self.classified.append(user)
            if user in self._overrides:
                intent, confidence = self._overrides[user]
                payload = {"intent": intent, "confidence": confidence,
                           "reasoning": "scripted", "alternatives": []}
                return LLMResponse(text=json.dumps(payload), raw=payload)
        return self._mock.complete(system, user, json_mode=json_mode)


# What the live Groq model returned in the reported session and in repro.
REAL_GROQ = {
    "yoga mat": ("fallback", 0.1),
    "hmm the blue one": ("fallback", 0.1),
    "the third one": ("address_change", 0.9),  # confidently wrong
    "what about the bluetooth speaker order": ("order_lookup", 0.9),
}


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
    yield
    get_settings.cache_clear()
    reset_llm_client()
    reset_order_client()


@pytest.fixture
def llm():
    import app.providers.llm_client as m

    client = ScriptedLLM(REAL_GROQ)
    m._client = client
    return client


def sid() -> str:
    return f"d-{uuid.uuid4().hex[:8]}"


def turn(session, text, phone=ANITA):
    return run_turn(session, text, caller_phone=phone)


# ---- the reported bug ------------------------------------------------------------------------


def test_naming_the_item_on_the_first_turn_needs_no_follow_up_question(llm):
    r = turn(sid(), "Tell me where my yoga mat order is.")
    assert r.state["order_id"] == "DLV1003"
    assert "hasn't shipped" in r.reply_text
    assert not r.state.get("escalated")
    assert not r.state.get("pending_disambiguation")


def test_answering_the_which_order_question_with_an_item_name_is_not_escalated(llm):
    s = sid()
    first = turn(s, "where is my order")
    assert first.state["pending_disambiguation"]["intent"] == "order_lookup"
    assert "?" in first.reply_text

    # The live model called this "fallback, 0.1" -- and the caller got a human.
    second = turn(s, "yoga mat")

    assert second.state["order_id"] == "DLV1003"
    assert "hasn't shipped" in second.reply_text
    assert not second.state.get("escalated")
    assert second.state["router"]["intent"] == "order_lookup"
    # ...and it never needed the LLM to understand a plain answer.
    assert llm.classified == ["where is my order"]


@pytest.mark.parametrize(
    "answer, order_id, expected_reply",
    [
        ("the third one", "DLV1003", "hasn't shipped"),  # LLM said address_change 0.9
        ("the last one", "DLV1003", "hasn't shipped"),
        ("the pending one", "DLV1003", "hasn't shipped"),
        ("DLV1003", "DLV1003", "hasn't shipped"),  # an exact ID used to loop forever
        ("D L V one zero zero three", "DLV1003", "hasn't shipped"),
        ("the first one", "DLV1001", "out for delivery"),
        ("the bluetooth speaker", "DLV1001", "out for delivery"),
        ("the one out for delivery", "DLV1001", "out for delivery"),
        ("the second one", "DLV1002", "on its way"),
        ("the pillow covers", "DLV1002", "on its way"),
    ],
)
def test_every_natural_way_of_answering_resolves_the_order(llm, answer, order_id, expected_reply):
    s = sid()
    turn(s, "where is my order")
    r = turn(s, answer)
    assert r.state["order_id"] == order_id
    assert expected_reply in r.reply_text
    assert r.state["router"]["intent"] == "order_lookup"
    assert not r.state.get("escalated")
    assert not r.state.get("pending_disambiguation")


def test_the_explain_trail_shows_why_it_treated_the_reply_as_an_answer(llm):
    s = sid()
    turn(s, "where is my order")
    turn(s, "yoga mat")

    decisions = [
        e for e in get_sink().session(s)
        if e["kind"] == "decision" and e["node"] == "router"
    ]
    last = decisions[-1]
    assert last["data"]["source"] == "pending_answer"
    assert "which-order question" in last["reasoning"]
    resolutions = [e for e in get_sink().session(s) if "order_id_resolution:reference" in e["message"]]
    assert resolutions and "answered the pending which-order question" in resolutions[-1]["reasoning"]


def test_an_exact_order_id_resolves_even_though_neighbouring_ids_score_close():
    # DLV1003 vs DLV1001/1002 fuzzy-score ~0.92; that used to read as "ambiguous".
    result = resolve_order_id("DLV1003", ANITA, InProcessOrderClient())
    assert result.status == "resolved" and result.order_id == "DLV1003"


# ---- unclear answers: re-ask once, then escalate -------------------------------------------------


def test_an_unclear_answer_gets_one_polite_re_ask_not_a_handoff(llm):
    s = sid()
    turn(s, "where is my order")
    second = turn(s, "hmm the blue one")  # nothing matches; live LLM: fallback 0.1

    assert not second.state.get("escalated")
    assert second.reply_text.startswith("Sorry, I didn't quite catch which one.")
    assert "Yoga mat" in second.reply_text  # options are read out again
    assert second.state["pending_disambiguation"]["retries"] == 1

    third = turn(s, "yoga mat")  # and answering properly still works afterwards
    assert third.state["order_id"] == "DLV1003" and not third.state.get("escalated")


def test_still_unclear_after_the_re_ask_escalates_with_a_specific_reason(llm):
    s = sid()
    turn(s, "where is my order")
    turn(s, "hmm the blue one")
    third = turn(s, "hmm the blue one")

    assert third.state["escalated"] is True
    assert third.state["escalation_reason"] == "order_disambiguation_failed"
    assert "colleague" in third.reply_text.lower()


def test_narrowing_the_options_is_progress_not_a_failed_answer():
    ref = lambda i, n, st: OrderRef(f"O{i}", n, st)  # noqa: E731
    three = [ref(1, "Lamp", "pending"), ref(2, "Desk", "pending"), ref(3, "Chair", "in_transit")]
    state = {
        "session_id": "x",
        "turns": [1, 2],
        "pending_disambiguation": {
            "intent": "order_lookup", "candidates": [r.to_dict() for r in three],
            "retries": 1, "at_turn": 1,
        },
    }
    live = state["pending_disambiguation"]

    narrowed = _ask(state, "order_lookup", three[:2], {}, live)
    assert not narrowed.state_patch.get("escalated")  # even though retries are used up
    assert narrowed.state_patch["agent_reply"].startswith("Got it.")

    stuck = _ask(state, "order_lookup", three, {}, live)
    assert stuck.state_patch["escalated"] is True


# ---- the order stays put, but the caller can switch ---------------------------------------------------


def test_a_follow_up_about_the_same_order_does_not_refetch_or_re_ask(llm):
    import app.providers.order_client as oc

    calls = []

    class Spy(InProcessOrderClient):
        def orders_for_phone(self, phone):
            calls.append(phone)
            return super().orders_for_phone(phone)

    oc._client = Spy()
    s = sid()
    turn(s, "Tell me where my yoga mat order is.")
    assert len(calls) == 1  # once per turn, not once per helper (it used to be 2-3)

    calls.clear()
    r = turn(s, "when will it arrive")  # no order named -> keep DLV1003
    assert r.state["order_id"] == "DLV1003"
    assert calls == []  # identification came from the session snapshot


def test_the_caller_can_switch_to_a_different_order_mid_call(llm):
    s = sid()
    turn(s, "Tell me where my yoga mat order is.")
    r = turn(s, "what about the bluetooth speaker order")

    assert r.state["order_id"] == "DLV1001"
    assert "out for delivery" in r.reply_text
    switched = [e for e in get_sink().session(s) if "switched from the order" in (e.get("reasoning") or "")]
    assert switched


# ---- the same mechanism for other questions we ask ----------------------------------------------------------


def test_which_order_question_from_reschedule_continues_as_reschedule(llm):
    s = sid()
    first = turn(s, "I need to reschedule my delivery")
    assert first.state["pending_disambiguation"]["intent"] == "reschedule"

    second = turn(s, "yoga mat")  # live model: fallback 0.1
    assert second.state["router"]["intent"] == "reschedule"
    assert second.state["order_id"] == "DLV1003"
    assert "what day works" in second.reply_text.lower()  # carried on to the date question

    third = turn(s, "in 3 days evening")  # the mock cannot classify this at all
    assert third.state["router"]["intent"] == "reschedule"
    assert third.awaiting_confirmation["kind"] == "confirm_reschedule"
    # A turn paused at interrupt() has no composed reply -- it must not still
    # hold turn 2's "what day works?" (which the voice path would speak instead
    # of the read-back). Callers fall back to awaiting_confirmation["prompt"].
    assert third.reply_text == ""


def test_yes_to_the_reschedule_offer_routes_to_reschedule_not_a_human(llm):
    s = sid()
    first = turn(s, "where is my order", phone=RAVI)
    assert "Would you like to reschedule?" in first.reply_text
    assert first.state["pending_followup"]["kind"] == "offer"

    second = turn(s, "yes please", phone=RAVI)  # cold, this classifies as gibberish
    assert second.state["router"]["intent"] == "reschedule"
    assert not second.state.get("escalated")
    assert "what day works" in second.reply_text.lower()


def test_a_no_to_the_offer_is_not_treated_as_a_yes(llm):
    s = sid()
    turn(s, "where is my order", phone=RAVI)
    r = turn(s, "no thanks", phone=RAVI)
    assert r.state["router"]["intent"] != "reschedule"


def test_a_pending_question_expires_if_the_caller_moves_on(llm):
    """An offer the caller ignored must not capture a later, unrelated "yes".
    Built by hand (rather than via run_turn) because the agent re-makes the
    offer every turn while the order is still a failed attempt."""
    from app.graph.nodes.router import route

    offer = {"intent": "reschedule", "kind": "offer", "at_turn": 1}

    def state(turns):
        return {
            "session_id": sid(), "last_utterance": "yes",
            "turns": list(range(turns)), "pending_followup": offer,
        }

    assert route(state(2))["router"]["intent"] == "reschedule"  # the very next turn
    assert route(state(3))["router"]["intent"] == "fallback"  # a turn later: expired


def test_a_real_new_request_is_not_swallowed_by_a_pending_question(llm):
    s = sid()
    turn(s, "where is my order")  # asks which order
    r = turn(s, "I need to change my delivery address")  # a different, clear request
    assert r.state["router"]["intent"] == "address_change"
    # and the which-order question is asked again on behalf of the new task
    assert r.state["pending_disambiguation"]["intent"] == "address_change"


def test_a_paused_turn_does_not_repeat_the_previous_turns_reply(llm):
    s = sid()
    first = turn(s, "I need to reschedule my delivery", phone=RAVI)
    assert "what day works" in first.reply_text.lower()

    second = turn(s, "in 3 days evening", phone=RAVI)

    assert second.awaiting_confirmation["kind"] == "confirm_reschedule"
    assert second.reply_text == ""
    assert "Shall I confirm" in second.awaiting_confirmation["prompt"]


# ---- narrowing down (Divya has two pending orders) ---------------------------------------------

DIVYA = "9990000020"  # DLV1023 Ceramic vase (out for delivery), DLV1024 Bookshelf + DLV1025
#                      Cushion covers -- both PENDING, so "the pending one" fits two orders
KARAN = "9990000019"  # DLV1021 Laptop sleeve (in transit), DLV1022 Wireless mouse (pending)


def test_an_ambiguous_answer_narrows_the_options_instead_of_failing(llm):
    s = sid()
    turn(s, "where is my order", phone=DIVYA)
    narrowed = turn(s, "the pending one", phone=DIVYA)

    assert not narrowed.state.get("escalated")
    assert narrowed.state.get("order_id") is None
    assert narrowed.reply_text.startswith("Got it.")
    assert "Bookshelf" in narrowed.reply_text and "Cushion covers" in narrowed.reply_text
    assert "Ceramic vase" not in narrowed.reply_text  # the vase is not pending
    assert len(narrowed.state["pending_disambiguation"]["candidates"]) == 2
    assert narrowed.state["pending_disambiguation"]["retries"] == 0  # progress, not a miss

    # "the second one" now means the second of the two just read out.
    final = turn(s, "the second one", phone=DIVYA)
    assert final.state["order_id"] == "DLV1025"
    assert "hasn't shipped" in final.reply_text


def test_narrowing_can_be_finished_by_naming_the_item(llm):
    s = sid()
    turn(s, "where is my order", phone=DIVYA)
    turn(s, "the pending one", phone=DIVYA)
    final = turn(s, "the bookshelf", phone=DIVYA)
    assert final.state["order_id"] == "DLV1024"


def test_a_two_order_caller_can_answer_by_item(llm):
    s = sid()
    first = turn(s, "where is my order", phone=KARAN)
    assert "Laptop sleeve" in first.reply_text and "Wireless mouse" in first.reply_text
    second = turn(s, "the wireless mouse", phone=KARAN)
    assert second.state["order_id"] == "DLV1022"
    assert "hasn't shipped" in second.reply_text
