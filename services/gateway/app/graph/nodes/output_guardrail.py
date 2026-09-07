"""
Output guardrail -- defense-in-depth screening of the composed reply before
it ever reaches ASR/TTS or the caller.

Sits between composer and END, a plain linear edge (see builder.py) -- no
concurrency to reason about here, unlike the input side. Every domain agent's
reply is templated today (see nodes/order_lookup.py, reschedule.py,
address_change.py), so this should never actually fire; it exists to
complete the "both boundaries" half of the guardrails requirement in
README.md's Responsible AI table, and to catch the day a reply path stops
being purely templated.
"""

from __future__ import annotations

from app.graph.state import CallState
from app.guardrails.moderation import check_output
from app.observability.trace import EventKind, trace

_SAFE_FALLBACK_REPLY = (
    "I'm not able to share that here. I'm connecting you with a colleague who can help."
)


def output_guardrail(state: CallState) -> dict:
    session_id = state["session_id"]
    reply = state.get("reply_text", "")
    verdict = check_output(reply)

    if verdict.allowed:
        trace(session_id, EventKind.GUARDRAIL, "output_guardrail", "allowed")
        return {}

    trace(
        session_id, EventKind.GUARDRAIL, "output_guardrail", "blocked",
        data={"category": verdict.category, "matched_phrase": verdict.matched_phrase},
    )
    return {
        "reply_text": _SAFE_FALLBACK_REPLY,
        "escalated": True,
        "escalation_reason": f"guardrail_blocked_output:{verdict.category}",
    }
