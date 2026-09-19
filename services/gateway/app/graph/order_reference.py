"""
Order references: "which of this caller's orders did they mean?"

A caller almost never says an order ID. They say "my yoga mat order", "the
pending one", "the third one", or -- when they do say an ID -- ASR hands us
"D L V one zero zero three". This module turns any of those into an order,
deterministically: no LLM, no network, pure functions over a short list of
the caller's own open orders. That makes it cheap to call from anywhere (the
router uses it to recognise an answer to a pending question without paying for
an LLM call) and easy to test exhaustively.

Signals, each producing the SET of orders it points at:
  id       full ID, or its trailing digits ("DLV1003", "1003", spoken digits)
  item     words from an item name ("yoga mat", "the speaker")
  status   "the pending one", "the one that's out for delivery"
  ordinal  "the third one", "the last one" -- only while ANSWERING a question
           we just asked, because "third" is only meaningful relative to the
           list we read out, and only in a short reply (see _MAX_ORDINAL_WORDS)

Combination rule: an explicit ID wins. Otherwise every signal that fired must
agree -- their intersection is the answer if it is exactly one order; signals
that contradict each other ("the third one" + "the speaker") are reported as
ambiguous, never silently resolved to one of them. Never guessing between
orders is the point (CLAUDE.md rule 1's spirit, applied to more than IDs).

Weak evidence: a single word out of a multi-word item name ("running" for
"Running shoes") is only trusted while answering our question. Volunteered
cold, it could be an accident of phrasing ("I'm running late").
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from rapidfuzz import fuzz

# One re-ask of a which-order question before giving up and escalating. Lives
# here (not in the router or order_identify) because both need it and this
# module has no dependencies on either.
MAX_DISAMBIGUATION_RETRIES = 1

# An answer to "which order?" is short. Longer utterances that happen to
# contain "first" or "last" ("reschedule to the first of next month") are
# about something else.
_MAX_ORDINAL_WORDS = 6
# Partial (single-word) item matches only trusted in short replies too.
_MAX_WEAK_ITEM_WORDS = 8

# --- spoken-ID normalisation -------------------------------------------------
# Callers say "D L V one zero zero two", "DLV-1002", "delivery ten oh two" --
# normalise before comparing, not after.
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")
_WORD_DIGITS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "to": "2", "too": "2",
    "three": "3", "four": "4", "for": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9",
}


def normalize_spoken_id(text: str) -> str:
    words = text.lower().split()
    digits_spoken = " ".join(_WORD_DIGITS.get(w, w) for w in words)
    return _NON_ALNUM.sub("", digits_spoken).upper()


# --- data ----------------------------------------------------------------------


@dataclass(frozen=True)
class OrderRef:
    """The minimum needed to recognise an order in speech. Plain strings only,
    so it serialises into checkpointed graph state without ceremony."""

    order_id: str
    items_preview: str
    status: str  # OrderStatus value, e.g. "pending"

    @classmethod
    def from_summary(cls, summary) -> OrderRef:
        return cls(summary.order_id, summary.items_preview, summary.status.value)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> OrderRef:
        return cls(d["order_id"], d["items_preview"], d["status"])


@dataclass
class ReferenceMatch:
    status: str  # 'unique' | 'ambiguous' | 'none'
    order_id: str | None = None
    candidates: list[str] = field(default_factory=list)  # ambiguous: who it might be
    signals: list[str] = field(default_factory=list)  # which signals fired
    reason: str = ""


_NONE = ReferenceMatch("none", reason="no reference to any order found")


# --- signals -------------------------------------------------------------------


def _id_signal(normalized: str, orders: Sequence[OrderRef]) -> set[str]:
    full = {o.order_id for o in orders if normalize_spoken_id(o.order_id) in normalized}
    if full:
        return full
    # Trailing digits alone ("order 1003"). Four+ digits so a stray "2" or
    # "10" in a sentence can never select an order.
    hits: set[str] = set()
    for o in orders:
        m = re.search(r"(\d{4,})$", o.order_id)
        if m and m.group(1) in normalized:
            hits.add(o.order_id)
    return hits


_GENERIC_ITEM_WORDS = {"set", "pack", "of", "and", "with", "for", "the", "a", "an", "kit"}


def _tokens(text: str) -> list[str]:
    out = []
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]  # crude plural: "speakers" ~ "speaker", "covers" ~ "cover"
        out.append(t)
    return out


def _token_present(token: str, utterance_tokens: list[str]) -> bool:
    for u in utterance_tokens:
        if u == token:
            return True
        # ASR/translation wobble ("yoga matt", "bluetooth speeker"). The 85
        # cutoff keeps 3-letter words safe: mat~matt (86) matches, mat~man (67)
        # and pen~pan (67) don't.
        if len(token) >= 3 and len(u) >= 3 and fuzz.ratio(token, u) >= 85:
            return True
    return False


def _item_signal(
    utterance_tokens: list[str], orders: Sequence[OrderRef]
) -> tuple[set[str], bool]:
    """Orders whose item names the caller mentioned, and whether the evidence
    is only partial (some but not all significant words of an item)."""
    best = 0.0
    scores: dict[str, float] = {}
    for o in orders:
        top = 0.0
        for item in o.items_preview.split(","):
            sig = [t for t in _tokens(item) if t not in _GENERIC_ITEM_WORDS]
            if not sig:
                continue
            covered = sum(1 for t in sig if _token_present(t, utterance_tokens))
            top = max(top, covered / len(sig))
        scores[o.order_id] = top
        best = max(best, top)
    if best < 0.5:
        return set(), False
    winners = {oid for oid, s in scores.items() if s == best}
    return winners, best < 1.0


_STATUS_PATTERNS: dict[str, re.Pattern[str]] = {
    "pending": re.compile(r"\bpending\b|\b(?:not|hasn'?t|haven'?t)\s+(?:yet\s+)?(?:been\s+)?shipped\b"),
    "in_transit": re.compile(r"\bin[- ]transit\b|\bon (?:its|the) way\b|(?<!not )(?<!yet )\bshipped\b"),
    "out_for_delivery": re.compile(r"\bout for delivery\b|\bdelivery today\b|\barriving today\b"),
    "failed_attempt": re.compile(r"\bfailed\b|\bmissed\b|\bunsuccessful\b"),
    "rto_initiated": re.compile(r"\breturn(?:ed|ing)?\b|\brto\b"),
    "cancelled": re.compile(r"\bcancel+ed\b"),
}


_NOT_SHIPPED = re.compile(r"\b(?:not|hasn'?t|haven'?t)\s+(?:yet\s+)?(?:been\s+)?shipped\b")


def _status_signal(text: str, orders: Sequence[OrderRef]) -> set[str]:
    wanted = {s for s, pat in _STATUS_PATTERNS.items() if pat.search(text)}
    if _NOT_SHIPPED.search(text):
        # "hasn't shipped" also contains the word "shipped"; it means pending,
        # not in transit.
        wanted.discard("in_transit")
    return {o.order_id for o in orders if o.status in wanted}


_ORDINALS: dict[str, int] = {
    "first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2,
    "fourth": 3, "4th": 3,
}
_LAST_WORDS = {"last", "final", "latest"}


def _ordinal_signal(words: list[str], basis: Sequence[OrderRef]) -> set[str]:
    if not basis or len(words) > _MAX_ORDINAL_WORDS:
        return set()
    hits: set[str] = set()
    for w in words:
        if w in _ORDINALS and _ORDINALS[w] < len(basis):
            hits.add(basis[_ORDINALS[w]].order_id)
        elif w in _LAST_WORDS:
            hits.add(basis[-1].order_id)
    return hits


# --- public API ------------------------------------------------------------------


def match_order_reference(
    utterance: str,
    orders: Sequence[OrderRef],
    *,
    answering: bool = False,
    ordinal_basis: Sequence[OrderRef] | None = None,
) -> ReferenceMatch:
    """
    Which of `orders` is `utterance` talking about?

    answering      True when the caller is replying to a which-order question
                   we just asked. Unlocks ordinals and partial item matches.
    ordinal_basis  the list, in the order it was READ OUT, that "the third
                   one" refers to. Defaults to `orders`.
    """
    if not orders or not utterance.strip():
        return _NONE

    text = utterance.lower()
    words = re.findall(r"[a-z0-9']+", text)
    utt_tokens = _tokens(text)

    signals: dict[str, set[str]] = {}

    if hits := _id_signal(normalize_spoken_id(utterance), orders):
        signals["id"] = hits

    item_hits, partial = _item_signal(utt_tokens, orders)
    if item_hits and (not partial or (answering and len(words) <= _MAX_WEAK_ITEM_WORDS)):
        signals["item"] = item_hits

    if hits := _status_signal(text, orders):
        signals["status"] = hits

    if answering:
        if hits := _ordinal_signal(words, ordinal_basis if ordinal_basis is not None else orders):
            signals["ordinal"] = hits

    if not signals:
        return _NONE

    fired = sorted(signals)
    # An explicit ID is decisive when it names exactly one order.
    if len(signals.get("id", ())) == 1:
        (oid,) = signals["id"]
        return ReferenceMatch("unique", oid, [oid], fired, "caller gave the order ID")

    agreed = set.intersection(*signals.values())
    if len(agreed) == 1:
        (oid,) = agreed
        return ReferenceMatch("unique", oid, [oid], fired, f"matched by {' + '.join(fired)}")

    everyone = sorted(set.union(*signals.values()))
    candidates = sorted(agreed) if agreed else everyone
    why = "signals disagree" if not agreed else "more than one order fits"
    return ReferenceMatch("ambiguous", None, candidates, fired, f"{why} ({' + '.join(fired)})")
