"""
Keyword/phrase detectors for the safety guardrail.

Deterministic and explainable by design -- see moderation.py's module
docstring for why this project does not use an LLM-based safety classifier.
Same shape as the rest of the codebase's fast-path pattern modules
(regex_extractors.py, observability/pii.py's PATTERNS): a small ordered table,
checked cheaply, with the highest-severity category checked first so a
coincidental lower-priority match can never shadow a real safety signal.
"""

from __future__ import annotations

# Ordered by severity, most severe first -- find_category() checks in this
# order, so a message matching both self_harm and a lower category (e.g. it
# also happens to contain a prompt-injection-shaped phrase) is reported as
# self_harm, not the other way round.
_SELF_HARM_PHRASES = [
    "kill myself", "end my life", "want to die", "wanted to die",
    "want to kill myself", "suicide", "hurt myself", "hurting myself",
    "self harm", "self-harm", "not worth living", "no reason to live",
]

_VIOLENCE_PHRASES = [
    "kill you", "shoot you", "shoot up", "i'll hurt you", "i will hurt you",
    "beat you up", "burn down", "bomb the", "planting a bomb",
]

_ILLEGAL_ACTIVITY_PHRASES = [
    "how do i hack", "how to hack into", "make a bomb", "buy drugs online",
    "sell drugs", "launder money", "how to steal a", "counterfeit money",
]

_PROMPT_INJECTION_PHRASES = [
    "ignore previous instructions", "ignore your instructions",
    "ignore all previous instructions", "disregard your instructions",
    "disregard previous instructions", "you are now dan", "dan mode",
    "reveal your system prompt", "reveal your prompt", "what is your system prompt",
    "your instructions are", "new instructions:", "act as if you have no rules",
    "pretend you have no restrictions",
]

# category -> phrase list, in the priority order find_category() walks.
CATEGORY_PHRASES: dict[str, list[str]] = {
    "self_harm": _SELF_HARM_PHRASES,
    "violence": _VIOLENCE_PHRASES,
    "illegal_activity": _ILLEGAL_ACTIVITY_PHRASES,
    "prompt_injection": _PROMPT_INJECTION_PHRASES,
}


def find_category(text: str) -> tuple[str, str] | None:
    """
    First matching (category, phrase) in priority order, or None.

    Substring matching on a lowercased copy -- deliberately simple, like
    pii.py's PATTERNS. It catches the phrasing it catches and openly does not
    catch paraphrase, code-switched, or obfuscated attempts; see README.md's
    Limitations section for why that's stated rather than hidden.
    """
    if not text:
        return None
    lowered = text.lower()
    for category, phrases in CATEGORY_PHRASES.items():
        for phrase in phrases:
            if phrase in lowered:
                return category, phrase
    return None
