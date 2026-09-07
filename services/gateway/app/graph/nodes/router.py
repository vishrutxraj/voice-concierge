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
