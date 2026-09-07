"""
Order-ID resolution.

The core bet of this whole project: Whisper-class ASR runs 25-35%+ WER on
Indic telephony speech, and an alphanumeric order ID (letters AND digits, no
grammar to lean on) is close to the worst thing you can ask it to transcribe
correctly. So this module never tries.

Instead: identify the caller by phone number (which the transport layer already
has — ANI on a real phone line, or the session's registered number for the
browser demo), pull their 1-4 open orders via OrderClient.orders_for_phone, and
fuzzy-match whatever garbled fragment the caller said against that short list.
Collapsing the search space from "any string" to "3 known strings" turns an
unsolved ASR problem into an essentially solved one.

Three outcomes, not two:
  - one clear winner above threshold      -> resolved silently
  - multiple candidates, none dominant    -> ask the caller to disambiguate
  - nothing clears the threshold          -> fall back to DTMF / spelled-out
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from order_contracts.schemas import OrderSummary
from rapidfuzz import fuzz, process

from app.config import get_settings
from app.providers.order_client import OrderClient

# Loose separation-tolerance: callers say "D L V one zero zero two",
# "DLV-1002", "delivery ten oh two" — normalize before scoring, not after.
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")

_WORD_DIGITS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "to": "2", "too": "2",
    "three": "3", "four": "4", "for": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9",
}


def _normalize(text: str) -> str:
    words = text.lower().split()
    digits_spoken = " ".join(_WORD_DIGITS.get(w, w) for w in words)
    return _NON_ALNUM.sub("", digits_spoken).upper()


@dataclass
class ResolutionResult:
    status: str  # 'resolved' | 'ambiguous' | 'not_found'
    order_id: str | None
    confidence: float
    candidates: list[dict]  # [{"order_id","score","status","items_preview"}, ...]


def resolve_order_id(
    utterance: str,
    caller_phone: str,
    order_client: OrderClient,
    threshold: float | None = None,
) -> ResolutionResult:
    settings = get_settings()
    threshold = threshold if threshold is not None else settings.order_id_fuzzy_threshold

    open_orders: list[OrderSummary] = order_client.orders_for_phone(caller_phone)
    if not open_orders:
        return ResolutionResult("not_found", None, 0.0, [])

    # Single open order: nothing to disambiguate. This is the common case and
    # it means ASR accuracy on the ID barely matters most of the time.
    if len(open_orders) == 1:
        only = open_orders[0]
        return ResolutionResult(
            "resolved", only.order_id, 1.0,
            [{"order_id": only.order_id, "score": 1.0}],
        )

    normalized_utterance = _normalize(utterance)
    choices = {o.order_id: _normalize(o.order_id) for o in open_orders}

    matches = process.extract(
        normalized_utterance,
        choices,
        scorer=fuzz.partial_ratio,
        limit=len(choices),
    )
    # matches: list of (normalized_value, score, order_id_key)
    scored = sorted(
        (
            {
                "order_id": key,
                "score": round(score / 100.0, 3),
            }
            for _, score, key in matches
        ),
        key=lambda m: m["score"],
        reverse=True,
    )

    best = scored[0]
    second = scored[1] if len(scored) > 1 else {"score": 0.0}

    if best["score"] < threshold:
        return ResolutionResult("not_found", None, best["score"], scored)

    # A clear winner needs both to clear the floor AND to meaningfully beat
    # the runner-up — two similar order IDs at 0.74 and 0.72 is not a
    # confident match, it is a coin flip the agent should not make silently.
    if best["score"] - second["score"] < 0.08 and second["score"] >= threshold:
        return ResolutionResult("ambiguous", None, best["score"], scored[:3])

    return ResolutionResult("resolved", best["order_id"], best["score"], scored[:3])


def format_disambiguation_prompt(candidates: list[dict], order_client: OrderClient,
                                  caller_phone: str) -> str:
    """What the agent says when it can't silently pick one."""
    summaries = {o.order_id: o for o in order_client.orders_for_phone(caller_phone)}
    lines = []
    for c in candidates[:3]:
        o = summaries.get(c["order_id"])
        if o:
            lines.append(f"your order for {o.items_preview} ({o.status.value})")
    if not lines:
        return "Which order are you calling about?"
    if len(lines) == 1:
        return f"Just to confirm, is this about {lines[0]}?"
    joined = "; ".join(lines[:-1]) + f"; or {lines[-1]}"
    return f"I see a couple of open orders — is this about {joined}?"
