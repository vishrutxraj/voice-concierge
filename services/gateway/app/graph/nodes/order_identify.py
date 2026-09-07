"""
Shared "which order is this about" step.

Every domain agent (lookup, reschedule, address) needs an order_id before it
can do anything. Rather than duplicate the resolve-or-ask logic three times,
each agent calls ensure_order_id first and returns early if it isn't resolved
yet — the state update naturally carries candidates back to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.graph.order_resolution import format_disambiguation_prompt, resolve_order_id
from app.graph.state import CallState
from app.observability.trace import EventKind, trace
from app.providers.order_client import get_order_client


@dataclass
class OrderIdOutcome:
    resolved: bool
    order_id: str | None
    state_patch: dict


def ensure_order_id(state: CallState, node_name: str) -> OrderIdOutcome:
    session_id = state["session_id"]

    # Already resolved earlier in the call — reuse it rather than re-asking.
    if state.get("order_id"):
        return OrderIdOutcome(True, state["order_id"], {})

    phone = state.get("caller_phone", "")
    utterance = state.get("last_utterance", "")
    client = get_order_client()

    open_orders = client.orders_for_phone(phone)

    # The common case: a generic "where's my order" carries no order-ID
    # fragment at all, so fuzzy-matching against DLV1001/1002/1003 correctly
    # scores everything low -- that's not the caller's fault, they were never
    # going to volunteer an ID for a status check. One open order -> just use
    # it. Several -> list them and ask which, rather than demanding an ID.
    if len(open_orders) == 1:
        only = open_orders[0]
        trace(
            session_id, EventKind.DECISION, node_name, "order_id_resolution:single_open_order",
            confidence=1.0,
            reasoning="caller has exactly one open order; used it without requiring an ID",
        )
        return OrderIdOutcome(
            True, only.order_id,
            {"order_id": only.order_id, "order_id_confidence": 1.0},
        )

    result = resolve_order_id(utterance, phone, client)

    if result.status == "not_found" and len(open_orders) > 1:
        # Caller didn't reference any specific order -- list what they have
        # rather than escalating over a missing ID they never intended to give.
        prompt = format_disambiguation_prompt(
            [{"order_id": o.order_id, "score": 1.0} for o in open_orders],
            client, phone,
        )
        return OrderIdOutcome(False, None, {"candidate_orders": [
            {"order_id": o.order_id} for o in open_orders
        ], "agent_reply": prompt})

    trace(
        session_id,
        EventKind.DECISION,
        node_name,
        f"order_id_resolution:{result.status}",
        confidence=result.confidence,
        alternatives=result.candidates,
        reasoning=f"fuzzy-matched caller utterance against open orders for {phone[-4:] if phone else 'unknown'}",
    )

    if result.status == "resolved":
        return OrderIdOutcome(
            True,
            result.order_id,
            {"order_id": result.order_id, "order_id_confidence": result.confidence},
        )

    if result.status == "ambiguous":
        prompt = format_disambiguation_prompt(result.candidates, client, phone)
        return OrderIdOutcome(
            False,
            None,
            {
                "candidate_orders": result.candidates,
                "agent_reply": prompt,
            },
        )

    # not_found — either no open orders, or nothing cleared the fuzzy threshold.
    trace(
        session_id, EventKind.ESCALATION, node_name, "order_id_not_found",
        confidence=result.confidence,
    )
    return OrderIdOutcome(
        False,
        None,
        {
            "agent_reply": (
                "I couldn't find that order on your account. Could you read out "
                "the order number, or I can connect you with a colleague."
            ),
            "escalated": True,
            "escalation_reason": "order_id_not_found",
        },
    )
