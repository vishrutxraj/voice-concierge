"""
Dialogue memory: questions the agent has asked and is waiting on.

The router classifies each utterance on its own. That is right for a fresh
request and wrong for an ANSWER: "yoga mat", "the third one", "yes please",
"in 3 days evening" mean nothing without the question they respond to, score
as gibberish, and used to trigger a handoff to a human -- for the crime of
answering us. The fix is to remember what we just asked.

Two kinds of pending question live in graph state:

  pending_disambiguation   "which order?"  -- see nodes/order_identify.py
  pending_followup         "what day works?" / "would you like to reschedule?"
                           -- {"intent": <task to continue>, "kind": "slot"|"offer"}

Both are written by domain-agent nodes (exactly one runs per turn -- rule 4)
and READ by the router. Both carry `at_turn`, and are only LIVE on the very
next turn. That expiry is the safety property: a pending question that the
caller ignored ("actually, never mind") must not silently capture some later,
unrelated reply. It means nothing ever needs to remember to clear them.

`turns` holds one entry per caller utterance (agent replies aren't stored, and a
Command(resume) confirmation turn adds none), so len(turns) is the turn number.
"""

from __future__ import annotations

from app.graph.state import CallState


def turn_number(state: CallState) -> int:
    return len(state.get("turns", []))


def new_pending(state: CallState, **fields) -> dict:
    """A pending-question record stamped with the turn that asked it."""
    return {**fields, "at_turn": turn_number(state)}


def live_pending(state: CallState, key: str) -> dict | None:
    """The pending question under `key`, only if it was asked on the previous
    turn -- i.e. the current utterance is a reply to it."""
    pending = state.get(key)
    if pending and pending.get("at_turn") == turn_number(state) - 1:
        return pending
    return None
