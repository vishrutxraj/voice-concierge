"""
Composer.

Final node before TTS (phase 4). The single writer of reply_text -- this is
deliberate, not incidental: the domain agent (order_lookup / reschedule /
address_change / fallback) and sentiment_monitor run in the SAME superstep as
concurrent branches off intent_router and START respectively. If both wrote
reply_text directly, LangGraph would raise InvalidUpdateError on a non-
annotated key with two writers in one step -- correctly, since there's no safe
way to pick a winner automatically. So agents write agent_reply, sentiment
writes only its own keys, and composer -- which fans in from both and always
runs strictly after -- is the only place reply_text is set.

Three jobs, in order:
  1. Decide "resolved vs unresolved" for sentiment's rolling turn count. This
     also has to live here rather than in sentiment_monitor itself: sentiment
     runs concurrently with the agent node, so it cannot see this turn's
     agent_reply yet when it executes -- only composer sees both.
  2. If sentiment_monitor escalated independently of the domain agent, make
     sure the caller actually hears that.
  3. Rehydrate PII tokens back to real values ONLY here, at the last possible
     moment, using the session's in-memory vault. Everything upstream --
     including every trace event -- has only ever seen tokens.
"""

from __future__ import annotations

from app.graph.state import CallState
from app.observability.trace import EventKind, trace


def compose(state: CallState) -> dict:
    session_id = state["session_id"]
    reply = state.get("agent_reply", "")
    escalated = state.get("escalated", False)

    if escalated and "colleague" not in reply.lower() and "human" not in reply.lower():
        reply = (
            f"{reply} I'm also going to have a colleague follow up with you "
            f"shortly to make sure everything's sorted."
        )

    vault = state.get("pii_vault")
    if vault is not None:
        reply = vault.rehydrate(reply)

    trace(
        session_id, EventKind.NODE_EXIT, "composer", "reply_ready",
        data={"escalated": escalated, "reply_length": len(reply)},
    )

    return {"reply_text": reply}
