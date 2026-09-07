"""
Guardrails: deterministic keyword/phrase moderation on both boundaries.

Why keyword-based rather than an LLM safety classifier: this project's hard
rule (CLAUDE.md) is that the test suite runs offline with zero credentials. A
safety-critical check that only works when GROQ_API_KEY is set would mean the
guardrail is silently absent in exactly the configuration -- local dev, CI --
where it's easiest to forget to re-verify it. A fixed phrase list is honest
about what it catches (exact/near-exact phrasing) and, just as importantly,
honest about what it doesn't (paraphrase, code-switched, or obfuscated
attempts) -- see README.md's Limitations section. This is the same fast-path-
first design already used everywhere else in the graph (regex_extractors.py,
pii.py's PATTERNS), not a new pattern invented for this module alone.

Out-of-scope requests (weather, general chit-chat, coding help) are NOT this
module's job. The router already buckets anything it isn't confident about
into intent="fallback" (see nodes/router.py's system prompt) -- that's been
true since phase 2. This module exists for what the router was never designed
to catch: safety-relevant input, and attempts to steer the agent itself
(prompt injection), screened independently of and prior to whatever the
intent classifier decides, so safety enforcement never depends on an LLM
"happening" to classify an adversarial message correctly.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.guardrails.patterns import find_category


@dataclass
class GuardrailVerdict:
    allowed: bool
    category: str | None = None
    matched_phrase: str | None = None


def _check(text: str) -> GuardrailVerdict:
    found = find_category(text)
    if found is None:
        return GuardrailVerdict(allowed=True)
    category, phrase = found
    return GuardrailVerdict(allowed=False, category=category, matched_phrase=phrase)


def check_input(text: str) -> GuardrailVerdict:
    """Screen a caller utterance before it reaches the router or any agent."""
    return _check(text)


def check_output(text: str) -> GuardrailVerdict:
    """
    Screen the composed reply before it reaches TTS/the caller -- defense in
    depth. In practice every domain agent's reply is templated (see
    nodes/order_lookup.py, reschedule.py, address_change.py), so this should
    never fire; it exists for the day a reply path stops being purely
    templated, and to complete the "both boundaries" half of the guardrails
    requirement in README.md's Responsible AI table.
    """
    return _check(text)
