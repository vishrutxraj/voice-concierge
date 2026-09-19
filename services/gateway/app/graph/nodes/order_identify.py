"""
Shared "which order is this about" step.

Every domain agent (lookup, reschedule, address) needs an order_id before it
can do anything. Rather than duplicate the resolve-or-ask logic three times,
each agent calls ensure_order_id first and returns early if it isn't resolved
yet -- the state update naturally carries the question back to the caller.

Resolution order, cheapest and most certain first:
  1. exactly one open order            -> just use it
  2. a reference in the utterance      -> app/graph/order_reference.py: order ID,
     ("yoga mat", "the pending one",      item name, status, or (only when
     "the third one", "DLV1003")          answering our question) an ordinal
  3. a garbled order ID                -> fuzzy match (order_resolution.py), the
                                          ASR-error safety net (rule 1)
  4. none of the above                 -> ASK, and remember that we asked

"Remember that we asked" is the important part: the question is stored as
`pending_disambiguation` (see state.py) so the NEXT turn's reply is understood
as the answer, by the router, instead of being classified cold. An unclear
answer gets one polite re-ask before escalating -- a human handoff is for
"I can't help", not for "you said something short".

The caller's orders are fetched at most once per session (order_snapshot) and
never twice in one turn; the fuzzy matcher and the prompt builder used to each
re-query on their own, unguarded against OrderAPIUnavailable.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.graph.dialogue import live_pending, new_pending
from app.graph.order_reference import (
    MAX_DISAMBIGUATION_RETRIES,
    OrderRef,
    ReferenceMatch,
    match_order_reference,
)
from app.graph.order_resolution import format_options, resolve_order_id
from app.graph.state import CallState
from app.observability.trace import EventKind, trace
from app.providers.order_client import OrderAPIUnavailable, get_order_client

# How many options are read aloud. More than this is a wall of speech, and the
# caller can still name any order by item/status/ID -- only "the fourth one"
# is unavailable, so the list we store is exactly the list we read.
_MAX_READ_OUT = 3


@dataclass
class OrderIdOutcome:
    resolved: bool
    order_id: str | None
    state_patch: dict


def ensure_order_id(state: CallState, node_name: str) -> OrderIdOutcome:
    session_id = state["session_id"]
    utterance = state.get("last_utterance", "")
    phone = state.get("caller_phone", "")
    pending = live_pending(state, "pending_disambiguation")
    current = state.get("order_id")
    snapshot = state.get("order_snapshot")
    client = get_order_client()

    summaries = None  # OrderSummary list; only present when we fetched this turn
    if current and snapshot is not None:
        refs = [OrderRef.from_dict(d) for d in snapshot]
        fetched = False
    else:
        try:
            summaries = client.orders_for_phone(phone)
        except OrderAPIUnavailable as exc:
            # Found live, not in a test: a Vercel cold start alone can exceed the
            # 4s client timeout on this exact call, and every OrderClient call
            # site must escalate gracefully instead of crashing the turn.
            trace(session_id, EventKind.ERROR, node_name, "order_api_unavailable",
                  data={"error": str(exc)})
            return OrderIdOutcome(
                False, None,
                {
                    "agent_reply": "I'm having trouble reaching our order system right now — "
                                  "let me get a colleague to help.",
                    "escalated": True,
                    "escalation_reason": "order_api_unavailable",
                },
            )
        refs = [OrderRef.from_summary(o) for o in summaries]
        fetched = True

    base_patch: dict = {"order_snapshot": [r.to_dict() for r in refs]} if fetched else {}

    def resolved(order_id: str, confidence: float) -> OrderIdOutcome:
        return OrderIdOutcome(
            True, order_id,
            {**base_patch, "order_id": order_id, "order_id_confidence": confidence,
             "pending_disambiguation": None, "candidate_orders": []},
        )

    # -- 1. no orders at all -------------------------------------------------
    if not refs:
        trace(session_id, EventKind.DECISION, node_name, "order_id_resolution:not_found",
              confidence=0.0, reasoning="caller has no open orders")
        return _escalate_not_found(session_id, node_name, base_patch)

    # -- 2. exactly one order: nothing to disambiguate ----------------------------
    # The common case, and it means recognising an order rarely matters.
    if len(refs) == 1:
        only = refs[0]
        if current == only.order_id:
            return OrderIdOutcome(True, current, {})
        trace(
            session_id, EventKind.DECISION, node_name, "order_id_resolution:single_open_order",
            confidence=1.0,
            reasoning="caller has exactly one open order; used it without requiring an ID",
        )
        return resolved(only.order_id, 1.0)

    # -- 3. a reference in what they said -------------------------------------------
    presented = [OrderRef.from_dict(d) for d in pending["candidates"]] if pending else refs
    match = match_order_reference(
        utterance, refs, answering=pending is not None, ordinal_basis=presented
    )

    if match.status == "unique":
        if current == match.order_id and pending is None:
            return OrderIdOutcome(True, current, base_patch)  # still the same order
        trace(
            session_id, EventKind.DECISION, node_name,
            f"order_id_resolution:reference:{'+'.join(match.signals)}",
            confidence=1.0 if match.signals == ["id"] else 0.95,
            reasoning=(
                f"{match.reason}; "
                + ("switched from the order discussed earlier" if current and current != match.order_id
                   else "answered the pending which-order question" if pending
                   else "named the order directly")
            ),
        )
        return resolved(match.order_id, 1.0 if match.signals == ["id"] else 0.95)

    # Something was referenced but doesn't pin down one order. If we're already
    # talking about an order and it is among the possibilities, keep it.
    if current and (match.status == "none" or current in match.candidates):
        return OrderIdOutcome(True, current, base_patch)

    # -- 4. garbled order ID: the ASR-error safety net ------------------------------
    if summaries is not None and match.status == "none":
        fuzzy = resolve_order_id(utterance, phone, client, open_orders=summaries)
        trace(
            session_id, EventKind.DECISION, node_name,
            f"order_id_resolution:{fuzzy.status}",
            confidence=fuzzy.confidence, alternatives=fuzzy.candidates,
            reasoning=f"fuzzy-matched caller utterance against open orders for "
                      f"{phone[-4:] if phone else 'unknown'}",
        )
        if fuzzy.status == "resolved":
            return resolved(fuzzy.order_id, fuzzy.confidence)
        if fuzzy.status == "ambiguous":
            by_id = {r.order_id: r for r in refs}
            options = [by_id[c["order_id"]] for c in fuzzy.candidates if c["order_id"] in by_id]
            return _ask(state, node_name, options or refs, base_patch, pending)

    # -- 5. ask ----------------------------------------------------------------------
    narrowed = [r for r in refs if r.order_id in match.candidates] if match.candidates else refs
    return _ask(state, node_name, narrowed, base_patch, pending, match)


def _ask(
    state: CallState,
    node_name: str,
    options: list[OrderRef],
    base_patch: dict,
    pending: dict | None,
    match: ReferenceMatch | None = None,
) -> OrderIdOutcome:
    """Read out the options and remember that we did. On a repeat, escalate
    after MAX_DISAMBIGUATION_RETRIES unclear answers -- unless the list just got
    shorter, which means the caller IS narrowing it down."""
    session_id = state["session_id"]
    read_out = options[:_MAX_READ_OUT]

    retries = 0
    prefix = ""
    if pending is not None:
        previous = len(pending["candidates"])
        retries = pending.get("retries", 0)
        if len(read_out) >= previous:  # no progress: an unclear answer
            if retries >= MAX_DISAMBIGUATION_RETRIES:
                trace(session_id, EventKind.ESCALATION, node_name,
                      "order_disambiguation_failed",
                      reasoning="asked which order twice and still could not tell")
                return OrderIdOutcome(
                    False, None,
                    {
                        **base_patch,
                        "agent_reply": "I'm having trouble working out which order you mean — "
                                       "let me connect you with a colleague.",
                        "escalated": True,
                        "escalation_reason": "order_disambiguation_failed",
                        "pending_disambiguation": None,
                    },
                )
            retries += 1
            prefix = "Sorry, I didn't quite catch which one. "
        prefix = prefix or "Got it. "

    trace(
        session_id, EventKind.DECISION, node_name, "order_id_resolution:asking",
        reasoning=(match.reason if match and match.status != "none" else
                   "no order was identified in what the caller said")
                  + f"; read out {len(read_out)} option(s)",
    )
    return OrderIdOutcome(
        False, None,
        {
            **base_patch,
            "candidate_orders": [{"order_id": r.order_id} for r in read_out],
            "agent_reply": prefix + format_options(read_out),
            "pending_disambiguation": new_pending(
                state,
                intent=node_name,
                candidates=[r.to_dict() for r in read_out],
                retries=retries,
            ),
        },
    )


def _escalate_not_found(session_id: str, node_name: str, base_patch: dict) -> OrderIdOutcome:
    trace(session_id, EventKind.ESCALATION, node_name, "order_id_not_found", confidence=0.0)
    return OrderIdOutcome(
        False, None,
        {
            **base_patch,
            "agent_reply": (
                "I couldn't find that order on your account. Could you read out "
                "the order number, or I can connect you with a colleague."
            ),
            "escalated": True,
            "escalation_reason": "order_id_not_found",
        },
    )
