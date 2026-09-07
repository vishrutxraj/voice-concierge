"""
PII handling.

Design note — why tokenization and not just masking:

A delivery agent MUST be able to read an address back to the caller to confirm
it. So we cannot simply destroy PII on the way in. Instead we do two different
things for two different destinations:

  * Toward logs / traces / persistence  -> irreversible redaction.
  * Toward the LLM                      -> reversible tokenization. The model
    sees <ADDR_1>, reasons about it structurally, and the composer rehydrates
    the real value at the last moment, in-process, never persisted.

The vault lives only in the session object in memory. It is never written to
the checkpointer, never traced, and dies with the call. That means an LLM
provider breach or a leaked log file exposes tokens, not customers.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Patterns. Ordered — longer/more specific first so they win the overlap.
# --------------------------------------------------------------------------

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b", re.I)),
    # Indian mobile: optional +91/0 prefix, then 6-9 followed by 9 digits.
    ("PHONE", re.compile(r"(?:(?:\+?91[\s-]?)|\b0)?[6-9]\d{9}\b")),
    # Aadhaar-shaped 12 digits in 4-4-4 grouping.
    ("NATIONAL_ID", re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("PINCODE", re.compile(r"\b[1-9]\d{5}\b")),
]

# Address detection is heuristic by necessity. We look for a house/flat marker
# followed by free text, which catches the common Indian address opener without
# trying to be a full parser.
ADDRESS_HINT = re.compile(
    r"\b(?:flat|plot|house|h\.?no|door|d\.?no|apartment|apt|block|tower)\b"
    r"[\s.:#-]*[\w/\-, ]{4,80}",
    re.I,
)

REDACTION_TEMPLATE = "[{kind} REDACTED]"


@dataclass
class PIIVault:
    """
    In-memory, session-scoped token store.

    Deliberately not serializable: no __getstate__, and the graph state stores
    a reference by session id rather than the object, so an accidental
    checkpoint write cannot carry it to disk.
    """

    _forward: dict[str, str] = field(default_factory=dict)  # token -> real
    _reverse: dict[str, str] = field(default_factory=dict)  # real  -> token
    _counters: dict[str, int] = field(default_factory=dict)

    def tokenize(self, kind: str, value: str) -> str:
        if value in self._reverse:
            return self._reverse[value]
        self._counters[kind] = self._counters.get(kind, 0) + 1
        token = f"<{kind}_{self._counters[kind]}>"
        self._forward[token] = value
        self._reverse[value] = token
        return token

    def rehydrate(self, text: str) -> str:
        """Swap tokens back to real values. Used only at the TTS boundary."""
        for token, value in self._forward.items():
            text = text.replace(token, value)
        return text

    def clear(self) -> None:
        self._forward.clear()
        self._reverse.clear()
        self._counters.clear()

    def __len__(self) -> int:
        return len(self._forward)


@dataclass
class ScanResult:
    text: str
    kinds_found: list[str]

    @property
    def had_pii(self) -> bool:
        return bool(self.kinds_found)


MIN_ADDRESS_FRAGMENT = 4


def _iter_matches(text: str) -> Iterable[tuple[int, int, str, str]]:
    """
    Yield (start, end, kind, matched_text), non-overlapping.

    Structured patterns (phone, pincode, email) run first and claim precise
    spans. The address heuristic then runs over what remains and is *trimmed*
    around claimed regions rather than discarded.

    That trimming matters: an Indian address usually ends in a pincode, so a
    naive overlap check throws away the whole address match because six digits
    inside it were already claimed — leaving the street name in the clear. This
    is the difference between redacting "[ADDRESS REDACTED], [PINCODE REDACTED]"
    and leaking "Flat 12 Nehru Nagar" into a log file.
    """
    claimed: list[tuple[int, int]] = []

    def overlaps(s: int, e: int) -> bool:
        return any(s < ce and e > cs for cs, ce in claimed)

    for kind, pattern in PATTERNS:
        for m in pattern.finditer(text):
            if not overlaps(m.start(), m.end()):
                claimed.append((m.start(), m.end()))
                yield m.start(), m.end(), kind, m.group()

    structured = sorted(claimed)
    for m in ADDRESS_HINT.finditer(text):
        cursor = m.start()
        for cs, ce in structured:
            if ce <= cursor or cs >= m.end():
                continue
            if cs - cursor >= MIN_ADDRESS_FRAGMENT:
                yield cursor, cs, "ADDRESS", text[cursor:cs]
                claimed.append((cursor, cs))
            cursor = max(cursor, ce)
        if m.end() - cursor >= MIN_ADDRESS_FRAGMENT and not overlaps(cursor, m.end()):
            yield cursor, m.end(), "ADDRESS", text[cursor : m.end()]
            claimed.append((cursor, m.end()))


def redact(text: str) -> ScanResult:
    """
    Irreversible. Use on ANYTHING heading for a log, trace, metric, or disk.
    """
    if not text:
        return ScanResult(text=text, kinds_found=[])

    matches = sorted(_iter_matches(text), key=lambda t: t[0], reverse=True)
    kinds: list[str] = []
    out = text
    for start, end, kind, _ in matches:
        out = out[:start] + REDACTION_TEMPLATE.format(kind=kind) + out[end:]
        kinds.append(kind)
    return ScanResult(text=out, kinds_found=sorted(set(kinds)))


def tokenize(text: str, vault: PIIVault) -> ScanResult:
    """
    Reversible. Use on anything heading for the LLM.
    """
    if not text:
        return ScanResult(text=text, kinds_found=[])

    matches = sorted(_iter_matches(text), key=lambda t: t[0], reverse=True)
    kinds: list[str] = []
    out = text
    for start, end, kind, raw in matches:
        out = out[:start] + vault.tokenize(kind, raw) + out[end:]
        kinds.append(kind)
    return ScanResult(text=out, kinds_found=sorted(set(kinds)))


def scrub_mapping(payload: dict, keys_to_drop: frozenset[str] = frozenset()) -> dict:
    """
    Recursively redact a dict destined for structured logging.

    keys_to_drop are removed outright (use for raw audio, api keys, vault refs).
    """
    dropped = keys_to_drop | frozenset(
        {"audio", "audio_bytes", "api_key", "authorization", "vault", "pii_vault"}
    )
    out: dict = {}
    for key, value in payload.items():
        if key.lower() in dropped:
            out[key] = "[DROPPED]"
        elif isinstance(value, str):
            out[key] = redact(value).text
        elif isinstance(value, dict):
            out[key] = scrub_mapping(value, keys_to_drop)
        elif isinstance(value, list):
            out[key] = [
                redact(v).text
                if isinstance(v, str)
                else scrub_mapping(v, keys_to_drop)
                if isinstance(v, dict)
                else v
                for v in value
            ]
        else:
            out[key] = value
    return out
