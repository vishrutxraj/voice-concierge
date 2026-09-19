"""
Router node.

Classifies caller intent from the (already PII-tokenized) utterance. Below
config.min_intent_confidence it does not guess — it routes to fallback, which
is a human handoff, not a wrong tool call. A misrouted reschedule that silently
fires is worse than an honest "let me connect you with a colleague."
"""

from __future__ import annotations

import json
import time

from app.audio.confirmation import interpret_confirmation
from app.graph.dialogue import live_pending
from app.graph.order_reference import OrderRef, match_order_reference
from app.graph.regex_extractors import extract_date_regex, extract_window_regex
from app.graph.state import CallState, RouterDecision
from app.observability.pii import redact
from app.observability.trace import EventKind, trace
from app.providers.llm_client import get_llm_client

_SYSTEM_PROMPT = """\
You classify a delivery-company caller's intent. Respond ONLY with JSON:
{"intent": "order_lookup"|"reschedule"|"address_change"|"fallback",
 "confidence": 0.0-1.0,
 "reasoning": "short reason",
 "alternatives": [{"intent": "...", "score": 0.0-1.0}]}

order_lookup: caller wants status/tracking/ETA of an existing order.
reschedule: caller wants to move a delivery to a different date/time.
address_change: caller wants to correct or change the delivery address.
fallback: anything else, or if you are not reasonably confident.
"""


def route(state: CallState) -> dict:
    session_id = state["session_id"]
    utterance = state.get("last_utterance", "")
    settings_confidence_floor = _confidence_floor()

    trace(session_id, EventKind.NODE_ENTER, "router", "classifying intent")
    t0 = time.time()

    # -- A question of ours is waiting for an answer -----------------------------
    # "Which order -- the speaker, the bedsheets, or the yoga mat?" ... "yoga
    # mat". Classified on its own that scores as gibberish (0.1 in the live
    # repro) and the caller was handed to a human for answering our question.
    # If the reply is recognisably an answer, continue the same task,
    # deterministically, without spending an LLM call on it. This has to happen
    # HERE, before classification -- an LLM asked about "the third one"
    # confidently said address_change (0.9). See graph/dialogue.py.
    pending = live_pending(state, "pending_disambiguation")
    followup = live_pending(state, "pending_followup")

    if pending:
        snapshot = state.get("order_snapshot") or pending["candidates"]
        presented = [OrderRef.from_dict(d) for d in pending["candidates"]]
        answer = match_order_reference(
            utterance,
            [OrderRef.from_dict(d) for d in snapshot],
            answering=True,
            ordinal_basis=presented,
        )
        if answer.status != "none":
            return _continue(
                session_id, pending["intent"], 1.0 if answer.status == "unique" else 0.9,
                f"caller is answering the which-order question ({answer.reason})",
                t0, source="pending_answer",
            )

    if followup and _answers_followup(followup, utterance):
        return _continue(
            session_id, followup["intent"], 0.95,
            "caller is replying to the question we just asked "
            + ("(yes to our offer)" if followup["kind"] == "offer" else "(a date/time)"),
            t0, source="pending_followup",
        )

    llm = get_llm_client()
    response = llm.complete(_SYSTEM_PROMPT, utterance, json_mode=True)

    try:
        parsed = json.loads(response.text)
        intent = parsed.get("intent", "unclear")
        confidence = float(parsed.get("confidence", 0.0))
        reasoning = parsed.get("reasoning", "")
        alternatives = parsed.get("alternatives", [])
    except (json.JSONDecodeError, TypeError, ValueError):
        intent, confidence, reasoning, alternatives = "unclear", 0.0, "parse failure", []

    escalate_low_confidence = confidence < settings_confidence_floor
    final_intent = "fallback" if escalate_low_confidence or intent == "unclear" else intent

    # Couldn't classify it, but we DID just ask them something: their reply was
    # probably an unclear answer to our question, not a request we can't
    # handle. Continue the task instead of escalating on the first miss (for a
    # which-order question, ensure_order_id does the re-asking and the counting;
    # for a slot question, the agent simply asks again -- the sentiment
    # monitor's unresolved-turn count still bounds a caller who never answers).
    if final_intent == "fallback":
        if pending:
            # The agent that asked decides between re-asking and escalating
            # (ensure_order_id), so a persistent non-answer ends in a specific
            # "couldn't tell which order" handoff, not a generic one.
            return _continue(
                session_id, pending["intent"], confidence,
                "unclear reply to a pending which-order question; handing back to the agent that asked",
                t0, source="pending_unclear", raw_intent=intent,
            )
        if followup and followup["kind"] == "slot":
            return _continue(
                session_id, followup["intent"], confidence,
                "unclear reply to a pending question; continuing the task instead of escalating",
                t0, source="pending_unclear", raw_intent=intent,
            )

    decision: RouterDecision = {
        "intent": final_intent,
        "confidence": confidence,
        "reasoning": reasoning,
        "alternatives": alternatives,
    }

    trace(
        session_id,
        EventKind.DECISION,
        "router",
        final_intent,
        confidence=confidence,
        reasoning=redact(reasoning).text,
        alternatives=alternatives,
        duration_ms=(time.time() - t0) * 1000,
        data={"below_confidence_floor": escalate_low_confidence, "raw_intent": intent},
    )

    if escalate_low_confidence:
        trace(
            session_id,
            EventKind.ESCALATION,
            "router",
            "low_confidence_routing",
            confidence=confidence,
            data={"floor": settings_confidence_floor},
        )

    return {"router": decision}


def _answers_followup(followup: dict, utterance: str) -> bool:
    """Is this recognisably a reply to the follow-up question we asked?
    Deterministic on purpose -- the whole point is not to rely on classifying a
    bare "yes please" or "friday evening" cold."""
    if followup["kind"] == "offer":
        return interpret_confirmation(utterance) is True
    if followup["intent"] == "reschedule":
        return bool(extract_date_regex(utterance) or extract_window_regex(utterance))
    return False


def _continue(
    session_id: str,
    intent: str,
    confidence: float,
    reasoning: str,
    t0: float,
    *,
    source: str,
    raw_intent: str | None = None,
) -> dict:
    """Route back to the task that asked the question."""
    decision: RouterDecision = {
        "intent": intent,
        "confidence": confidence,
        "reasoning": reasoning,
        "alternatives": [],
    }
    trace(
        session_id, EventKind.DECISION, "router", intent,
        confidence=confidence, reasoning=reasoning,
        duration_ms=(time.time() - t0) * 1000,
        data={"source": source, "raw_intent": raw_intent, "below_confidence_floor": False},
    )
    return {"router": decision}


def _confidence_floor() -> float:
    from app.config import get_settings

    return get_settings().min_intent_confidence


def route_edge(state: CallState) -> str:
    """
    Conditional-edge function: send the graph to the right agent node.

    Deliberately does NOT also check input_guardrail_verdict here, even
    though input_guardrail runs concurrently with intent_router (see
    nodes/input_guardrail.py and builder.py). A LangGraph conditional edge is
    evaluated off its own source node's completion, not the full merged
    superstep -- empirically, this edge fires without reliably seeing a
    sibling branch's same-superstep write, which would make a guardrail
    check here silently racy rather than a real gate. The actual gate is
    nodes/input_guardrail.py::guardrail_block_patch(), called at the top of
    every node this edge can land on (order_lookup, reschedule,
    address_change, fallback) -- by the time ANY of those nodes runs, it's a
    later superstep and the full merged state (including the guardrail's
    verdict) is guaranteed present as ordinary node input, not edge input.
    """
    decision = state.get("router")
    if not decision:
        return "fallback"
    return decision["intent"] if decision["intent"] in {
        "order_lookup", "reschedule", "address_change"
    } else "fallback"
