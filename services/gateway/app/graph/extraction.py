"""
Slot and address extraction: regex first, LLM fallback.

Phase 2's regex extractors (weekday names, "tomorrow", "in N days"; a
pincode-or-flat-line address correction) are kept as the PRIMARY path --
they're free, instant, deterministic, and every phase-2 test depends on their
exact behavior. This module adds a SECOND pass through the already-wired
LLMClient (Groq in production, keyword-mock offline) that only runs when the
regex pass finds nothing, to catch phrasing regex was never going to handle:
"day after tomorrow", "the 15th", "next Monday" (as distinct from "Monday"),
a caller dictating a brand-new address from scratch.

Every result carries a `method` field ("regex" | "llm" | "none") that gets
traced -- this is what turns "why did the agent understand that" into an
answerable question for the explainability endpoint, and what let this file's
own tests catch a real ordering bug during development (see test_extraction.py).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date

from order_contracts.schemas import Address, SlotWindow
from pydantic import ValidationError

from app.graph.regex_extractors import extract_date_regex, extract_window_regex
from app.providers.llm_client import LLMClient

_SLOT_EXTRACTION_SYSTEM = """\
You extract a delivery date and time-of-day window from a caller's utterance.
Today's date is {today} ({weekday}).

Respond ONLY with JSON: {{"date": "YYYY-MM-DD" or null, "window": "morning"|"afternoon"|"evening" or null}}

morning = 9am-1pm, afternoon = 1pm-5pm, evening = 5pm-9pm.
If the caller names a date but no time-of-day, or a time-of-day but no date,
use null for the missing field -- do not guess.
Handle relative phrases precisely: "next Monday" means the Monday of NEXT
week, not this coming Monday if today is not Monday. "Day after tomorrow" is
today + 2 days.
"""

_ADDRESS_EXTRACTION_SYSTEM = """\
A caller is correcting or providing a full delivery address. Extract what you
can into JSON: {"line1": "...", "line2": "..."|null, "city": "..."|null,
"state": "..."|null, "pincode": "..."|null, "landmark": "..."|null}

Use null for any field the caller did not state. line1 should be the
house/flat/plot number and street. Only extract a 6-digit Indian pincode into
"pincode" -- never guess one.
"""


@dataclass
class SlotExtraction:
    new_date: date | None
    window: SlotWindow | None
    method: str  # "regex" | "llm" | "regex+llm" | "none"


@dataclass
class AddressExtraction:
    address: Address | None
    method: str  # "regex" | "llm" | "none"
    missing_required: bool  # True if line1 or pincode couldn't be determined
    # True only when the LLM identified the caller is moving to a DIFFERENT
    # city than the order's current one but couldn't complete the address
    # (see extract_address's ordering fix below). Distinguishes "the LLM
    # found nothing useful, a narrow single-field regex fallback is safe" from
    # "the LLM found clear evidence of a city change it can't safely finish" --
    # the latter must escalate outright rather than silently degrade to a
    # pincode-only patch that would graft a new pincode onto the OLD city
    # name, which is a milder version of the exact corruption bug this
    # module's ordering fix exists to prevent.
    city_change_blocked: bool = False


def extract_slot(
    utterance: str, llm: LLMClient, today: date | None = None,
) -> SlotExtraction:
    today = today or date.today()

    regex_date = extract_date_regex(utterance, today)
    regex_window = extract_window_regex(utterance)
    if regex_date and regex_window:
        return SlotExtraction(regex_date, regex_window, "regex")

    # Partial or empty regex hit: fall back to the LLM only for whichever
    # piece is still missing, rather than discarding a match regex already
    # found in favour of a fresh (possibly worse) LLM guess.
    llm_date, llm_window = _llm_extract_slot(utterance, llm, today)

    final_date = regex_date or llm_date
    final_window = regex_window or llm_window

    if (regex_date or regex_window) and (llm_date or llm_window):
        method = "regex+llm"
    elif regex_date or regex_window:
        method = "regex"
    elif llm_date or llm_window:
        method = "llm"
    else:
        method = "none"

    return SlotExtraction(final_date, final_window, method)


def _llm_extract_slot(
    utterance: str, llm: LLMClient, today: date,
) -> tuple[date | None, SlotWindow | None]:
    system = _SLOT_EXTRACTION_SYSTEM.format(today=today.isoformat(), weekday=today.strftime("%A"))
    response = llm.complete(system, utterance, json_mode=True)
    try:
        parsed = json.loads(response.text)
        d = date.fromisoformat(parsed["date"]) if parsed.get("date") else None
        w = SlotWindow(parsed["window"]) if parsed.get("window") else None
        return d, w
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None, None


_PINCODE_RE = re.compile(r"\b[1-9]\d{5}\b")
_FLAT_RE = re.compile(r"(flat|plot|house|h\.?no|door)[\s.:#-]*([\w/\- ]{2,40})", re.I)
# A caller dictating a brand-new address usually names a city, road, or
# landmark -- if we see one of those alongside something regex CAN'T
# structure (no clean flat-line, no bare pincode), that's the signal to try
# the LLM rather than fail straight to escalation.
_LIKELY_FULL_ADDRESS = re.compile(
    r"\b(near|opposite|behind|street|road|nagar|colony|sector|city|pincode|pin code)\b",
    re.I,
)


def extract_address(
    utterance: str, current: Address, llm: LLMClient,
) -> AddressExtraction:
    # Order matters here, and getting it backwards is a real data-integrity
    # bug, not just a missed optimization: "my new address is 45 Park Road,
    # Mumbai 400001" contains a 6-digit pincode, so a pincode-first regex
    # check would grab ONLY the new pincode and silently keep the caller's
    # OLD street and city -- producing a self-inconsistent address (old
    # street + new city's pincode) that gets read back and confirmed without
    # the caller noticing anything wrong. So: check for full-address signals
    # FIRST. Only when there's no such signal do we trust the narrow
    # single-field regex, which is safe precisely because it only ever fires
    # on a plain "the pincode is X" / "flat 402" correction with nothing else
    # in the sentence suggesting a wholesale address change.
    if _LIKELY_FULL_ADDRESS.search(utterance):
        llm_result = _llm_extract_address(utterance, current, llm)
        if llm_result.address is not None:
            return llm_result
        if llm_result.city_change_blocked:
            # The LLM found clear evidence of a city change but correctly
            # refused to guess the state -- falling through to the narrow
            # regex here would graft a new pincode onto the OLD city name
            # (e.g. "...Hyderabad, 500099" for a caller who just said they
            # moved elsewhere), a milder repeat of the exact bug this
            # function's ordering already fixed once. Escalate outright.
            return llm_result
        # Otherwise the LLM found nothing usable at all -- fall through to
        # the narrow regex in case this was actually just a single-field
        # correction that happened to mention "near" or a road name.

    pincode_match = _PINCODE_RE.search(utterance)
    if pincode_match:
        addr = current.model_copy(update={"pincode": pincode_match.group()})
        return AddressExtraction(addr, "regex", missing_required=False)

    flat_match = _FLAT_RE.search(utterance)
    if flat_match:
        addr = current.model_copy(update={"line1": flat_match.group(0).strip()})
        return AddressExtraction(addr, "regex", missing_required=False)

    return AddressExtraction(None, "none", missing_required=True)


def _llm_extract_address(
    utterance: str, current: Address, llm: LLMClient,
) -> AddressExtraction:
    response = llm.complete(_ADDRESS_EXTRACTION_SYSTEM, utterance, json_mode=True)
    try:
        parsed = json.loads(response.text)
        line1 = parsed.get("line1")
        pincode = parsed.get("pincode")
        if not line1 or not pincode:
            return AddressExtraction(None, "llm", missing_required=True)

        new_city = parsed.get("city")
        new_state = parsed.get("state")
        # If the LLM detected a DIFFERENT city than the order's current one,
        # the old line2/landmark/state are almost certainly stale (they
        # describe a neighborhood in the OLD city) and must not be silently
        # carried forward -- that's how "45 Park Road, Mumbai" nearly got
        # confirmed back to a caller with "Kondapur" (a Hyderabad locality)
        # still attached. Only inherit those fields when the city is
        # unchanged, or when the LLM itself supplied a value.
        city_changed = bool(new_city) and new_city.lower() != current.city.lower()

        if city_changed and not new_state:
            # state is a required field with no safe default across a city
            # change (Hyderabad->Mumbai crosses Telangana->Maharashtra) --
            # guessing it would be exactly the kind of unstated assumption
            # this project's principles rule out. Ask, don't assume. Marked
            # distinctly from a plain "found nothing" miss so the caller
            # (extract_address) knows NOT to fall through to the narrow
            # regex -- see AddressExtraction.city_change_blocked.
            return AddressExtraction(None, "llm", missing_required=True, city_change_blocked=True)

        merged = {
            **current.model_dump(),
            "line1": line1,
            "line2": parsed.get("line2") if (parsed.get("line2") or not city_changed) else None,
            "city": new_city or current.city,
            "state": new_state or current.state,
            "pincode": pincode,
            "landmark": parsed.get("landmark") if (parsed.get("landmark") or not city_changed) else None,
        }
        # model_validate (not model_copy) deliberately: an LLM-derived pincode
        # is untrusted input and must pass the same schema pattern
        # (^[1-9]\d{5}$) that a directly-constructed Address would. model_copy
        # skips validation entirely, which would let a malformed pincode from
        # the LLM reach the order store unchecked -- a real gap this project's
        # own regex path never had, since a regex match already satisfies the
        # pattern by construction.
        addr = Address.model_validate(merged)
        return AddressExtraction(addr, "llm", missing_required=False)
    except (json.JSONDecodeError, TypeError, ValidationError):
        return AddressExtraction(None, "llm", missing_required=True)
