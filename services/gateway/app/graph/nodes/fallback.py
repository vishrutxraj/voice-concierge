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
        # An upstream node (order_identify, reschedule, address) already set a
        # specific reply_text and escalation_reason — don't overwrite it with
        # a generic message.
        return {}

    trace(session_id, EventKind.ESCALATION, "fallback", "unclassified_or_out_of_scope")
    return {
        "agent_reply": "I want to make sure you get the right help — let me "
                      "connect you with a colleague who can take it from here.",
        "escalated": True,
        "escalation_reason": "unclassified_or_out_of_scope",
    }
