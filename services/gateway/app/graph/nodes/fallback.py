"""
Fallback — anything unclassified, low-confidence, guardrail-blocked, or
explicitly escalated lands here.
"""

from __future__ import annotations

from app.graph.nodes.input_guardrail import guardrail_block_patch
from app.graph.state import CallState
from app.observability.trace import EventKind, trace


def fallback(state: CallState) -> dict:
    session_id = state["session_id"]

    if (patch := guardrail_block_patch(state)) is not None:
        trace(session_id, EventKind.NODE_ENTER, "fallback", "guardrail_blocked")
        return patch

    already_escalated = state.get("escalated", False)

    trace(
        session_id, EventKind.NODE_ENTER, "fallback",
        "already_escalated" if already_escalated else "new_escalation",
    )

    if already_escalated:
        # A handoff already happened earlier in this session. Say so plainly
        # rather than repeating the generic "let me connect you" line -- this
        # path used to return {} and lean on the previous turn's leftover
        # agent_reply, which no longer survives across turns (see runner.py).
        return {
            "agent_reply": "I've already asked a colleague to help with this — "
                           "they'll follow up with you shortly.",
        }

    trace(session_id, EventKind.ESCALATION, "fallback", "unclassified_or_out_of_scope")
    return {
        "agent_reply": "I want to make sure you get the right help — let me "
                      "connect you with a colleague who can take it from here.",
        "escalated": True,
        "escalation_reason": "unclassified_or_out_of_scope",
        # A caller handed to a human is no longer answering our question.
        "pending_disambiguation": None,
        "pending_followup": None,
    }
