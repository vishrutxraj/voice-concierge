"""
Input guardrail -- safety/injection screening, independent of intent
classification.

Runs as a THIRD concurrent branch off START, alongside intent_router and
sentiment_monitor (see CLAUDE.md rule 4: any new concurrent branch gets its
own state key -- this one writes only `input_guardrail_verdict`, never
agent_reply or reply_text, so there's no collision).

Enforcement is NOT done via a routing override: route_edge (attached to
intent_router's own conditional edge) cannot reliably see this node's write --
proven empirically, a LangGraph conditional edge is evaluated off its source
node's own completion, not the fully-merged superstep, so checking a sibling
branch's write there is silently racy rather than a real gate. Instead,
guardrail_block_patch() below is called at the top of every node route_edge
can land on (order_lookup, reschedule, address_change, fallback). By the time
any of those runs, it's a LATER superstep than this one, so the merged state
(including input_guardrail_verdict) is guaranteed present as ordinary node
input -- this is the same guarantee composer relies on to see both its
concurrent predecessors (see nodes/composer.py). Concretely: intent_router's
own classification call still runs on unsafe/injection text (an unavoidable
side effect of true concurrency, not a security hole -- it only produces an
intent LABEL), but no domain agent ever acts on it, because each one's very
first line is this check.
"""

from __future__ import annotations

from app.graph.state import CallState
from app.guardrails.moderation import check_input
from app.observability.trace import EventKind, trace

# Per-category replies for a guardrail-blocked input. self_harm gets a
# distinct, gentler tone -- the right move for a delivery-support bot is
# immediate, uncomplicated human handoff, not an attempt at counsel it isn't
# equipped to give. The others share a firmer, generic decline.
_GUARDRAIL_REPLIES = {
    "self_harm": "I hear you, and I want to make sure you get real support "
                "right now — I'm connecting you with a person immediately.",
}
_GUARDRAIL_DEFAULT_REPLY = (
    "I'm not able to help with that here. I'm connecting you with a colleague."
)


def input_guardrail(state: CallState) -> dict:
    session_id = state["session_id"]
    utterance = state.get("last_utterance", "")
    verdict = check_input(utterance)

    trace(
        session_id, EventKind.GUARDRAIL, "input_guardrail",
        "allowed" if verdict.allowed else "blocked",
        data=(
            {}
            if verdict.allowed
            else {"category": verdict.category, "matched_phrase": verdict.matched_phrase}
        ),
    )

    return {
        "input_guardrail_verdict": {"allowed": verdict.allowed, "category": verdict.category}
    }


def guardrail_block_patch(state: CallState) -> dict | None:
    """
    None if this turn's input was allowed (the overwhelmingly common case --
    every caller of this checks that and falls through to its normal logic).
    Otherwise a ready-to-return state patch: the guardrail-appropriate reply,
    plus escalation, so the caller ends up talking to a human rather than
    having order_lookup/reschedule/address_change process unsafe text.

    Called at the top of order_lookup, reschedule, address_change, and
    fallback -- see this module's docstring for why the check lives here
    rather than in route_edge.
    """
    verdict = state.get("input_guardrail_verdict")
    if verdict is None or verdict.get("allowed", True):
        return None

    category = verdict.get("category")
    trace(
        state["session_id"], EventKind.ESCALATION, "input_guardrail",
        "blocked_input_routed_around_agent", data={"category": category},
    )
    return {
        "agent_reply": _GUARDRAIL_REPLIES.get(category, _GUARDRAIL_DEFAULT_REPLY),
        "escalated": True,
        "escalation_reason": f"guardrail_blocked_input:{category}",
    }
