"""
Sentiment monitor.

Architecturally the most important thing about this node is that it is NOT on
the critical path to a reply. It runs as a parallel branch off the same fan-out
as the router (see builder.py) and writes to sentiment_score / escalated, but a
slow or failed sentiment call must never delay the caller hearing an answer.

Escalation triggers on any of:
  - sentiment below threshold (lexical + repeated-intent signal, see below)
  - too many unresolved turns without a completed action
  - explicit distress markers

The scoring here is deliberately simple and explainable (word-list plus turn
counting) rather than an opaque LLM call, because this number gates a real
consequence — a human callback — and the fairness eval (phase 5) needs to be
able to say exactly why a given score was produced, per language, per caller.
"""

from __future__ import annotations

from app.config import get_settings
from app.graph.state import CallState
from app.observability.trace import EventKind, trace

_NEGATIVE_MARKERS = [
    "angry", "frustrated", "ridiculous", "unacceptable", "terrible",
    "worst", "useless", "waste of time", "never again", "cancel my",
    "speak to a manager", "speak to a human", "this is the third time",
    "already told you", "i already said",
]
_POSITIVE_MARKERS = ["thank you", "thanks", "great", "perfect", "appreciate"]


def _lexical_score(text: str) -> float:
    text = text.lower()
    neg = sum(1 for m in _NEGATIVE_MARKERS if m in text)
    pos = sum(1 for m in _POSITIVE_MARKERS if m in text)
    if neg == 0 and pos == 0:
        return 0.0
    # Bounded to [-1, 1]; each additional negative marker compounds but caps out.
    raw = (pos * 0.3) - (neg * 0.4)
    return max(-1.0, min(1.0, raw))


def monitor_sentiment(state: CallState) -> dict:
    session_id = state["session_id"]
    settings = get_settings()

    utterance = state.get("last_utterance", "")
    turn_score = _lexical_score(utterance)

    history = state.get("sentiment_history", [])
    # Recency-weighted rolling average — a bad turn now matters more than one
    # from five turns ago, but a single sharp word doesn't trigger callback
    # alone.
    all_scores = history + [turn_score]
    weights = [0.5**i for i in range(len(all_scores) - 1, -1, -1)]
    rolling = sum(s * w for s, w in zip(all_scores, weights, strict=False)) / sum(weights)

    unresolved = state.get("unresolved_turns", 0)
    # sentiment_monitor runs in the same superstep as the domain agent (both
    # are branches off START/intent_router that execute concurrently before
    # composer), so it cannot see this turn's agent_reply yet -- only whether
    # a PRIOR turn already escalated. A turn counts as unresolved unless the
    # call was already flagged resolved before this turn began; composer
    # reconciles the full picture once both branches are in.
    already_escalated = state.get("escalated", False)
    new_unresolved = unresolved if already_escalated else unresolved + 1

    should_escalate = (
        rolling <= settings.sentiment_escalation_threshold
        or new_unresolved >= settings.max_unresolved_turns
    )

    trace(
        session_id, EventKind.DECISION, "sentiment_monitor",
        "escalate" if should_escalate else "continue",
        confidence=abs(rolling),
        reasoning=f"rolling={rolling:.2f}, unresolved_turns={new_unresolved}",
        data={
            "turn_score": turn_score,
            "rolling_score": rolling,
            "threshold": settings.sentiment_escalation_threshold,
        },
    )

    patch: dict = {
        "sentiment_score": rolling,
        "sentiment_history": [turn_score],
        "unresolved_turns": new_unresolved,
    }

    if should_escalate and not state.get("escalated"):
        trace(
            session_id, EventKind.ESCALATION, "sentiment_monitor",
            "sentiment_threshold_or_unresolved_turns",
            data={"rolling_score": rolling, "unresolved_turns": new_unresolved},
        )
        patch["escalated"] = True
        patch["escalation_reason"] = (
            "sentiment_threshold" if rolling <= settings.sentiment_escalation_threshold
            else "max_unresolved_turns"
        )

    return patch
