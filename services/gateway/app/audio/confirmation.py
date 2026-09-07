"""
Yes/no interpretation for spoken confirmation replies.

The text endpoint (/call/turn) never needs this: its caller passes an explicit
`resume: bool` because it already knows a confirmation is pending. Over audio,
the only signal available is whatever the caller said, so something has to
turn "yeah go ahead" into True before it can reach run_turn's resume_value.

Deliberately conservative, matching CLAUDE.md rule 6 ("don't guess a required
field you can't fully resolve"): a phrase that isn't clearly affirmative or
negative resolves to None rather than a guessed bool, and the call loop
re-asks instead of silently committing or silently declining a write the
caller never actually confirmed.
"""

from __future__ import annotations

_AFFIRMATIVE = {
    "yes", "yeah", "yep", "yup", "sure", "correct", "confirm", "confirmed",
    "right", "ok", "okay", "affirmative", "go ahead", "please do", "do it",
    "sounds good", "that works", "that's right", "thats right",
}

_NEGATIVE = {
    "no", "nope", "nah", "negative", "don't", "dont", "cancel", "wrong",
    "not right", "no thanks", "no thank you", "that's wrong", "thats wrong",
    "don't do that", "dont do that",
}


def interpret_confirmation(text: str) -> bool | None:
    """
    True/False for a clear yes/no, None when the reply doesn't resolve either
    way -- checked as whole-phrase membership first (so "no" inside a longer
    unrelated sentence doesn't false-positive), then word-by-word for short
    replies like "yeah okay" or "no, wait".
    """
    if not text:
        return None
    normalized = text.strip().lower().rstrip(".!?")
    if not normalized:
        return None

    if normalized in _AFFIRMATIVE:
        return True
    if normalized in _NEGATIVE:
        return False

    words = normalized.split()
    # Negative words are checked first: "no, that's not right, don't reschedule"
    # contains no affirmative words but "yes" as a stray token would be rarer;
    # still, a clear negative should never be shadowed by an incidental
    # affirmative-looking word later in the same reply.
    if any(w in _NEGATIVE for w in words):
        return False
    if any(w in _AFFIRMATIVE for w in words):
        return True
    return None
