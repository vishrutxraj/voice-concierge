"""
Address correction agent.

Same interrupt() shape as reschedule: propose, read back, confirm, then write.
Extraction is regex-first (pincode or flat/house line -- the overwhelming
majority of real correction calls), falling back to an LLM pass for a caller
dictating a genuinely new address (app/graph/extraction.py). This resolves
what was phase 2's most-cited limitation: a full address previously always
escalated to a human regardless of how clearly the caller stated it.

PII note: last_utterance arrives already tokenized (see runner.py -- every
caller utterance is tokenized before it enters the graph, precisely so nodes
never need to touch raw PII). But THIS node's whole job is to read a real
pincode or flat number out of what the caller said. So it's the one place in
the graph that deliberately reverses that: rehydrate from the vault for
extraction only, then discard the raw value immediately. The vault itself
never leaves process memory and is never traced (see pii.py) -- this is a
bounded, audited exception to "the graph only ever sees tokens," not a bypass
of it. The LLM fallback path uses the same rehydrated text -- it needs the
real pincode/city to extract correctly, and (see extraction.py) any value the
LLM returns is re-validated against the Address schema before it can reach
the order store, so an untrusted LLM output can't smuggle a malformed pincode
through unchecked.
"""

from __future__ import annotations

from langgraph.types import interrupt

from app.graph.extraction import extract_address
from app.graph.nodes.input_guardrail import guardrail_block_patch
from app.graph.nodes.order_identify import ensure_order_id
from app.graph.state import CallState
from app.observability.trace import EventKind, trace
from app.providers.llm_client import get_llm_client
from app.providers.order_client import OrderAPIUnavailable, OrderClient, get_order_client


def address_change(state: CallState) -> dict:
    session_id = state["session_id"]
    trace(session_id, EventKind.NODE_ENTER, "address_change", "start")

    if (patch := guardrail_block_patch(state)) is not None:
        return patch

    outcome = ensure_order_id(state, "address_change")
    if not outcome.resolved:
        return outcome.state_patch

    client = get_order_client()
    try:
        order = client.get_order(outcome.order_id)
    except OrderAPIUnavailable as exc:
        trace(session_id, EventKind.ERROR, "address_change", "order_api_unavailable",
              data={"error": str(exc)})
        return {
            "agent_reply": "I can't reach our system to update the address right now — "
                          "let me connect you with a colleague.",
            "escalated": True,
            "escalation_reason": "order_api_unavailable",
        }
    if order is None:
        return {"agent_reply": "I couldn't find that order to update."}

    tokenized_utterance = state.get("last_utterance", "")
    vault = state.get("pii_vault")
    # Rehydrate for extraction ONLY -- this local variable never reaches trace,
    # logging, or agent_reply. See module docstring for why this is safe, and
    # why it's also correct for the LLM fallback path.
    extraction_source = (
        vault.rehydrate(tokenized_utterance) if vault else tokenized_utterance
    )
    extraction = extract_address(extraction_source, order.address, get_llm_client())

    trace(
        session_id, EventKind.DECISION, "address_change", f"address_extraction:{extraction.method}",
        reasoning="regex-first, LLM fallback for a caller stating a full new address",
    )

    if extraction.address is None:
        # method=="llm" here means the full-address signal fired and the LLM
        # was actually invoked but couldn't produce a complete address (missing
        # line1 or pincode) -- worth distinguishing from method=="none", where
        # nothing matched at all and the LLM was never even called, for the
        # explainability trail.
        reason = (
            "address_llm_extraction_incomplete" if extraction.method == "llm"
            else "address_extraction_failed"
        )
        trace(session_id, EventKind.ESCALATION, "address_change", reason)
        return {
            **outcome.state_patch,
            "agent_reply": "For a full address change I'll connect you with a "
                          "colleague who can take that down carefully.",
            "escalated": True,
            "escalation_reason": reason,
        }
    proposed = extraction.address

    trace(
        session_id, EventKind.DECISION, "address_change", "propose_address",
        reasoning="parsed correction from caller utterance, awaiting confirmation",
        data={"order_id": outcome.order_id, "proposed": proposed.one_line()},
    )

    confirmed_raw = interrupt(
        {
            "kind": "confirm_address",
            "order_id": outcome.order_id,
            "proposed_address": proposed.one_line(),
            "prompt": f"I have the address as {proposed.one_line()}. Is that correct?",
        }
    )
    # resume_value is wrapped as {"confirmed": bool} to survive
    # Command(resume=False) -- see runner.py for why.
    confirmed = confirmed_raw["confirmed"] if isinstance(confirmed_raw, dict) else confirmed_raw

    if not confirmed:
        trace(session_id, EventKind.DECISION, "address_change", "confirmation_declined")
        return {
            **outcome.state_patch,
            "agent_reply": "No problem — could you say the correct address again?",
            "pending_confirmation": None,
        }

    idem_key = OrderClient.new_idempotency_key(session_id, "address_change")
    try:
        result = client.update_address(outcome.order_id, proposed, idem_key)
    except OrderAPIUnavailable as exc:
        trace(session_id, EventKind.ERROR, "address_change", "order_api_unavailable",
              data={"error": str(exc)})
        return {
            "agent_reply": "The confirmation went through on my end but I couldn't "
                          "save it to our system — a colleague will follow up.",
            "escalated": True,
            "escalation_reason": "order_api_unavailable_post_confirm",
        }

    trace(
        session_id, EventKind.TOOL_RESULT, "address_change", "commit",
        data={"ok": result.ok, "refusal_code": result.refusal_code},
    )

    return {
        **outcome.state_patch,
        "agent_reply": result.message,
        "pending_confirmation": None,
    }
