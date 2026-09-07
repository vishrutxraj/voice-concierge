"""
Graph state.

One TypedDict, threaded through every node. Two fields need special handling:

  pii_vault   — deliberately typed as PIIVault, not a dict. LangGraph's SQLite
                checkpointer serializes state between turns, and PIIVault has no
                __getstate__: an attempt to persist it raises rather than
                silently writing customer PII to disk. That failure is
                intentional — see app/observability/pii.py.

  session_id  — used as the checkpoint thread_id AND the trace session_id, so
                the decision chain in /api/sessions/{id}/explain lines up
                exactly with the LangGraph checkpoint history for the same call.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from app.observability.pii import PIIVault

Intent = Literal[
    "order_lookup", "reschedule", "address_change", "fallback", "unclear"
]


class Turn(TypedDict):
    role: Literal["caller", "agent"]
    text: str  # PII-tokenized if from the caller, safe to trace as-is
    ts: float


class RouterDecision(TypedDict):
    intent: Intent
    confidence: float
    reasoning: str
    alternatives: list[dict]  # [{"intent": ..., "score": ...}, ...]


class CallState(TypedDict, total=False):
    # Identity
    session_id: str
    caller_phone: str
    language: str  # ISO code pinned at first ASR turn; 'en' until then
    transport_kind: str  # 'browser' | 'twilio' | 'text'

    # Conversation
    turns: Annotated[list[Turn], operator.add]
    last_utterance: str  # raw-ish, PII-tokenized caller text for this turn

    # Routing
    router: RouterDecision | None

    # Guardrails (phase 5) -- input_guardrail is a concurrent branch off
    # START alongside intent_router and sentiment_monitor (see module
    # docstring above and CLAUDE.md rule 4). route_edge overrides routing to
    # fallback when this is not allowed; fallback reads it for the reply.
    # Named distinctly from the "input_guardrail" NODE -- LangGraph forbids a
    # node and a state key sharing a name.
    input_guardrail_verdict: dict | None

    # Order context — set once an order is identified, reused across turns
    order_id: str | None
    order_id_confidence: float | None
    candidate_orders: list[dict]  # ambiguous matches the caller must disambiguate

    # Mutation-in-progress (reschedule / address) awaiting interrupt() confirmation
    pending_confirmation: dict | None

    # Sentiment (parallel node — never gates the response path)
    sentiment_score: float  # -1.0 furious .. +1.0 delighted
    sentiment_history: Annotated[list[float], operator.add]
    unresolved_turns: int

    # Escalation
    escalated: bool
    escalation_reason: str | None

    # Output
    # agent_reply is written by exactly one of the four domain-agent nodes per
    # turn (whichever the router chose) -- never concurrently, so a plain str
    # key is safe. sentiment_monitor runs in the same superstep but writes
    # different keys (sentiment_*, escalated), so there is no collision.
    # composer is the ONLY node that writes reply_text, using agent_reply as
    # its input -- this keeps reply_text single-writer even though multiple
    # branches fan in to composer concurrently.
    agent_reply: str
    reply_text: str  # English; TTS node rehydrates PII and translates voice only

    # Non-serialized — excluded from checkpointing, see pii.py for why.
    pii_vault: PIIVault
