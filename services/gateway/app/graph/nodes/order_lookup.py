"""Order lookup agent — read-only, no interrupt() needed."""

from __future__ import annotations

from app.graph.nodes.input_guardrail import guardrail_block_patch
from app.graph.nodes.order_identify import ensure_order_id
from app.graph.state import CallState
from app.observability.trace import EventKind, trace
from app.providers.order_client import OrderAPIUnavailable, get_order_client


def order_lookup(state: CallState) -> dict:
    session_id = state["session_id"]
    trace(session_id, EventKind.NODE_ENTER, "order_lookup", "start")

    if (patch := guardrail_block_patch(state)) is not None:
        return patch

    outcome = ensure_order_id(state, "order_lookup")
    if not outcome.resolved:
        return outcome.state_patch

    client = get_order_client()
    try:
        order = client.get_order(outcome.order_id)
    except OrderAPIUnavailable as exc:
        trace(session_id, EventKind.ERROR, "order_lookup", "order_api_unavailable",
              data={"error": str(exc)})
        return {
            "agent_reply": "I'm having trouble reaching our order system right now — "
                          "let me get a colleague to help.",
            "escalated": True,
            "escalation_reason": "order_api_unavailable",
        }

    if order is None:
        return {"agent_reply": "I couldn't find that order. Could you confirm the order number?"}

    trace(session_id, EventKind.TOOL_CALL, "order_lookup", "get_order",
          data={"order_id": order.order_id, "status": order.status.value})

    reply = _compose_reply(order)
    return {**outcome.state_patch, "agent_reply": reply}


def _compose_reply(order) -> str:
    status_phrases = {
        "pending": "hasn't shipped yet",
        "in_transit": "is on its way",
        "out_for_delivery": "is out for delivery today",
        "delivered": "was delivered",
        "failed_attempt": "had a delivery attempt that didn't succeed",
        "rto_initiated": "is being returned to our warehouse after repeated failed attempts",
        "cancelled": "was cancelled",
    }
    phrase = status_phrases.get(order.status.value, "is being processed")
    base = f"Your order {phrase}."
    if order.status.value in {"pending", "in_transit", "out_for_delivery"}:
        base += f" It's expected {order.promised_slot.human()}."
    if order.status.value == "failed_attempt":
        base += " Would you like to reschedule?"
    return base
