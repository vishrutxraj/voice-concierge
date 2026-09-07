"""
Reschedule agent.

Two-phase: propose, then commit only after interrupt() confirmation. LangGraph
pauses the graph at interrupt() and returns control to the caller; the next
turn resumes via Command(resume=...) rather than re-entering the node from
scratch, so state like the pending slot survives the pause.

Slot extraction is regex-first (see app/graph/regex_extractors.py), falling
back to an LLM pass (app/graph/extraction.py) only for phrasing the regex
can't parse -- "day after tomorrow", "next Monday" as distinct from "Monday".
What matters architecturally is unchanged from phase 2: nothing gets written
until the caller has heard the slot read back and said yes.
"""

from __future__ import annotations

from langgraph.types import Command, interrupt

from app.graph.extraction import extract_slot
from app.graph.nodes.input_guardrail import guardrail_block_patch
from app.graph.nodes.order_identify import ensure_order_id
from app.graph.state import CallState
from app.observability.trace import EventKind, trace
from app.providers.llm_client import get_llm_client
from app.providers.order_client import OrderAPIUnavailable, OrderClient, get_order_client


def reschedule(state: CallState) -> dict:
    session_id = state["session_id"]
    trace(session_id, EventKind.NODE_ENTER, "reschedule", "start")

    if (patch := guardrail_block_patch(state)) is not None:
        return patch

    outcome = ensure_order_id(state, "reschedule")
    if not outcome.resolved:
        return outcome.state_patch

    # Dates/times are never PII, so the tokenized utterance (see runner.py --
    # every caller utterance is tokenized before entering the graph) is
    # identical to the raw one here; no vault rehydration needed, unlike
    # address_change.py's pincode/flat-number extraction.
    utterance = state.get("last_utterance", "")
    extraction = extract_slot(utterance, get_llm_client())
    new_date, window = extraction.new_date, extraction.window

    trace(
        session_id, EventKind.DECISION, "reschedule", f"slot_extraction:{extraction.method}",
        reasoning=f"date={'found' if new_date else 'missing'}, window={'found' if window else 'missing'}",
    )

    if not (new_date and window):
        missing = "a date" if not new_date else "a time of day"
        return {
            **outcome.state_patch,
            "agent_reply": f"Sure — what day works, and morning, afternoon, or evening? "
                          f"I still need {missing}.",
        }

    client = get_order_client()
    try:
        order = client.get_order(outcome.order_id)
    except OrderAPIUnavailable as exc:
        trace(session_id, EventKind.ERROR, "reschedule", "order_api_unavailable",
              data={"error": str(exc)})
        return {
            "agent_reply": "I can't reach our scheduling system right now — "
                          "let me connect you with a colleague.",
            "escalated": True,
            "escalation_reason": "order_api_unavailable",
        }
    if order is None:
        return {"agent_reply": "I couldn't find that order to reschedule."}

    # --- Read-back confirmation. The graph pauses here; nothing is written
    # until the caller has heard the exact slot and confirmed it. -----------
    from order_contracts.schemas import DeliverySlot

    proposed = DeliverySlot(date=new_date, window=window)
    trace(
        session_id, EventKind.DECISION, "reschedule", "propose_slot",
        reasoning="parsed date/time from caller utterance, awaiting confirmation",
        data={"order_id": outcome.order_id, "proposed": proposed.human()},
    )

    confirmed_raw = interrupt(
        {
            "kind": "confirm_reschedule",
            "order_id": outcome.order_id,
            "proposed_slot": proposed.human(),
            "prompt": f"I can move this to {proposed.human()}. Shall I confirm that?",
        }
    )
    # resume_value is wrapped as {"confirmed": bool} to survive
    # Command(resume=False) -- see runner.py for why.
    confirmed = confirmed_raw["confirmed"] if isinstance(confirmed_raw, dict) else confirmed_raw

    if not confirmed:
        trace(session_id, EventKind.DECISION, "reschedule", "confirmation_declined")
        return {
            **outcome.state_patch,
            "agent_reply": "No problem — what date and time would work better?",
            "pending_confirmation": None,
        }

    idem_key = OrderClient.new_idempotency_key(session_id, "reschedule")
    try:
        result = client.reschedule(outcome.order_id, new_date, window, idem_key)
    except OrderAPIUnavailable as exc:
        trace(session_id, EventKind.ERROR, "reschedule", "order_api_unavailable",
              data={"error": str(exc)})
        return {
            "agent_reply": "The confirmation went through on my end but I couldn't "
                          "reach the scheduling system to save it — a colleague "
                          "will follow up to confirm.",
            "escalated": True,
            "escalation_reason": "order_api_unavailable_post_confirm",
        }

    trace(
        session_id, EventKind.TOOL_RESULT, "reschedule", "commit",
        data={"ok": result.ok, "refusal_code": result.refusal_code},
    )

    if not result.ok:
        reply = result.message
        if result.alternatives:
            options = ", or ".join(a.human() if hasattr(a, "human") else str(a)
                                    for a in result.alternatives[:2])
            reply += f" I could offer {options} instead — would either of those work?"
        return {**outcome.state_patch, "agent_reply": reply, "pending_confirmation": None}

    return {
        **outcome.state_patch,
        "agent_reply": result.message,
        "pending_confirmation": None,
    }


def resume_reschedule(confirmed: bool) -> Command:
    """Built by the turn handler when resuming a paused reschedule."""
    return Command(resume=confirmed)
