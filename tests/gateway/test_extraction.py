"""
Extraction tests.

Several of these are direct regression tests for bugs found by manually
exercising this module before trusting it (see the phase-3 build notes):
regex substring matching silently mis-resolving "day after tomorrow" and
"next monday"; the narrow address regex corrupting data when a caller states
a whole new address; a stale locality (old city's landmark) leaking into a
new city's address; and a required schema field (state) nearly being guessed
across a city change. Each has a named test below so none of them can regress
silently.
"""

from __future__ import annotations

from datetime import date

import pytest
from app.graph.extraction import extract_address, extract_slot
from app.providers.llm_client import MockLLMClient
from order_contracts.schemas import Address, SlotWindow
from pydantic import ValidationError

TODAY = date(2026, 8, 29)  # a Saturday, fixed so tests don't depend on wall clock


@pytest.fixture
def llm():
    return MockLLMClient()


@pytest.fixture
def addr():
    return Address(line1="12 Old Street", city="Hyderabad", state="Telangana", pincode="500081")


# ---- slot extraction: regex fast path -------------------------------------


def test_regex_handles_plain_weekday_and_window(llm):
    r = extract_slot("reschedule to friday evening", llm, TODAY)
    assert r.new_date == date(2026, 9, 4)
    assert r.window == SlotWindow.EVENING
    assert r.method == "regex"


def test_regex_handles_tomorrow(llm):
    r = extract_slot("move it to tomorrow morning", llm, TODAY)
    assert r.new_date == date(2026, 8, 30)
    assert r.method == "regex"


def test_regex_handles_in_n_days(llm):
    r = extract_slot("in 3 days evening", llm, TODAY)
    assert r.new_date == date(2026, 9, 1)
    assert r.method == "regex"


# ---- regression: "day after tomorrow" substring bug ------------------------


def test_day_after_tomorrow_resolves_two_days_not_one(llm):
    """
    Regression: 'day after tomorrow' contains the substring 'tomorrow', and a
    naive check order matched that first, silently returning today+1 instead
    of today+2 -- and since a date WAS found, the LLM fallback never even ran
    to correct it. Fixed in regex_extractors.py by checking the longer,
    specific phrase before the substring it contains.
    """
    r = extract_slot("day after tomorrow morning", llm, TODAY)
    assert r.new_date == date(2026, 8, 31)
    assert r.method == "regex"  # now resolved by regex alone, for free


def test_plain_tomorrow_is_unaffected_by_the_fix(llm):
    r = extract_slot("tomorrow morning", llm, TODAY)
    assert r.new_date == date(2026, 8, 30)


# ---- regression: "next <weekday>" substring bug ----------------------------


def test_next_weekday_is_a_week_later_than_bare_weekday(llm):
    """
    Regression: 'next monday' contains 'monday', and the bare-weekday check
    matched it first, treating 'next monday' identically to 'monday' -- the
    coming Monday, not the Monday of next week. Fixed by checking the 'next
    <weekday>' pattern before the bare weekday-name substring check.
    """
    bare = extract_slot("monday afternoon", llm, TODAY)
    nxt = extract_slot("next monday afternoon", llm, TODAY)
    assert bare.new_date == date(2026, 8, 31)
    assert nxt.new_date == date(2026, 9, 7)
    assert nxt.new_date == bare.new_date + __import__("datetime").timedelta(days=7)


# ---- LLM fallback: genuinely regex-unparseable phrasing --------------------


def test_llm_fallback_handles_ordinal_day_of_month(llm):
    r = extract_slot("the 15th evening", llm, TODAY)
    assert r.new_date == date(2026, 9, 15)
    assert r.window == SlotWindow.EVENING
    assert r.method == "regex+llm"  # regex found the window, LLM found the date


def test_completely_unparseable_utterance_returns_none_method(llm):
    r = extract_slot("reschedule it please", llm, TODAY)
    assert r.new_date is None and r.window is None
    assert r.method == "none"


def test_partial_regex_match_still_only_asks_llm_for_the_missing_piece(llm):
    """Window found by regex, date genuinely absent -- method should reflect
    that the date came from nowhere (still 'regex' overall, not 'llm'), since
    the LLM mock also finds nothing for such a vague utterance."""
    r = extract_slot("evening works", llm, TODAY)
    assert r.window == SlotWindow.EVENING
    assert r.new_date is None


# ---- address: regex fast path (single-field correction) -------------------


def test_address_regex_pincode_correction(addr, llm):
    r = extract_address("the pincode should be 500083", addr, llm)
    assert r.method == "regex"
    assert r.address.pincode == "500083"
    assert r.address.line1 == addr.line1  # everything else untouched


def test_address_regex_flat_correction(addr, llm):
    r = extract_address("flat 9 sunrise towers", addr, llm)
    assert r.method == "regex"
    assert "sunrise towers" in r.address.line1.lower()
    assert r.address.pincode == addr.pincode  # untouched


def test_address_no_signal_at_all_is_not_found(addr, llm):
    r = extract_address("I want to change my address", addr, llm)
    assert r.address is None
    assert r.method == "none"
    assert r.missing_required is True


# ---- regression: narrow-regex data corruption on a full new address -------


def test_full_new_address_does_not_silently_grab_only_the_pincode(addr, llm):
    """
    Regression: a caller dictating an entirely new address in a different city
    also contains a 6-digit pincode, so the narrow single-field regex fired
    FIRST and grabbed only the pincode -- silently keeping the caller's OLD
    street ("12 Old Street") under the NEW city's pincode. That inconsistent,
    wrong address would have been read back and confirmed without the caller
    noticing anything was off. Fixed by checking for full-address signals
    (road/near/city name) before the narrow regex gets a chance to fire.
    """
    r = extract_address(
        "my new address is 45 Park Road near the mall Mumbai 400001", addr, llm,
    )
    assert r.method == "llm"
    assert r.address is not None
    assert "old street" not in r.address.line1.lower()
    assert r.address.city == "Mumbai"
    assert r.address.pincode == "400001"


def test_single_field_correction_that_happens_to_mention_near_still_works(addr, llm):
    """
    A full-address SIGNAL word ("near") can appear in an otherwise ordinary
    single-field correction. The LLM attempt runs first per the fix above, but
    when it can't build a complete address (mock has no city/pincode to key
    off here), extraction must still fall through to the narrow regex rather
    than escalating unnecessarily.
    """
    r = extract_address("it's flat 12, near the temple, pincode 500090", addr, llm)
    assert r.address is not None
    assert r.address.pincode == "500090"


# ---- regression: stale locality leaking across a city change --------------


def test_city_change_does_not_inherit_old_landmark(addr, llm):
    """
    Regression: after fixing the ordering bug above, the LLM path itself
    defaulted unset fields (landmark, line2) to the CURRENT address's values.
    For a caller who has moved to Mumbai, that silently attached "Kondapur"
    (the OLD Hyderabad-area landmark encoded in the `addr` fixture's
    surrounding context) to a Mumbai address. Fixed so city-specific fields
    are only inherited when the city is unchanged.
    """
    old_landmark_addr = addr.model_copy(update={"landmark": "Near Kondapur Metro"})
    r = extract_address(
        "my new address is 45 Park Road near the mall Mumbai 400001", old_landmark_addr, llm,
    )
    assert r.address is not None
    assert r.address.landmark != "Near Kondapur Metro"


def test_same_city_correction_still_inherits_unstated_fields(addr, llm):
    """The inheritance fix must not be so aggressive it breaks the common case:
    correcting one field within the SAME city should still keep the rest."""
    r = extract_address("the pincode should be 500083", addr, llm)
    assert r.address.line1 == addr.line1
    assert r.address.city == addr.city


# ---- regression: required field (state) must not be guessed ---------------


def test_city_change_without_a_stated_state_escalates_rather_than_guesses(addr):
    """
    Regression: `state` is a required schema field with no safe cross-city
    default (Hyderabad->Mumbai crosses Telangana->Maharashtra). An earlier fix
    silently kept the OLD state, which is exactly the kind of unstated
    assumption this project's principles rule out.

    Uses a purpose-built fake LLM rather than MockLLMClient: the keyword-mock
    can only recognise cities from its own fixed lookup, so it can never
    actually return "an unrecognised city with no state" -- it just returns
    city=null instead, which exercises a different code path (missing line1/
    city entirely) rather than the specific guard this test targets. A real
    LLM would happily name an unfamiliar city without knowing its state, so
    this fake simulates that directly to isolate extraction.py's own logic
    from the mock's heuristic limits.
    """
    from app.providers.llm_client import LLMResponse

    class CityNoStateLLM:
        def complete(self, system, user, *, json_mode=False):
            import json as _json
            return LLMResponse(text=_json.dumps({
                "line1": "9 River View", "city": "Notarealcity", "state": None,
                "pincode": "500099",
            }))

    r = extract_address(
        "my new address is 9 River View near the lake Notarealcity 500099",
        addr, CityNoStateLLM(),
    )
    assert r.address is None
    assert r.missing_required is True


def test_known_city_change_with_inferable_state_succeeds(addr, llm):
    r = extract_address(
        "my new address is 45 Park Road near the mall Mumbai 400001", addr, llm,
    )
    assert r.address is not None
    assert r.address.state == "Maharashtra"


# ---- validation: LLM output is untrusted input -----------------------------


def test_llm_derived_pincode_is_schema_validated_not_trusted_blindly():
    """
    model_validate (not model_copy) on the LLM path is deliberate: an
    LLM-derived pincode is untrusted input and must pass the same
    ^[1-9]\\d{5}$ pattern a directly-constructed Address would. This directly
    exercises that a malformed value from the LLM cannot reach an Address
    object unchecked.
    """
    from app.graph import extraction as extraction_module

    class BadPincodeLLM:
        def complete(self, system, user, *, json_mode=False):
            import json as _json

            from app.providers.llm_client import LLMResponse
            return LLMResponse(text=_json.dumps({
                "line1": "1 Test Road", "city": "Mumbai", "state": "Maharashtra",
                "pincode": "12345",  # 5 digits -- fails the schema pattern
            }))

    addr = Address(line1="x", city="Pune", state="Maharashtra", pincode="411014")
    result = extraction_module.extract_address(
        "my new address is 1 Test Road near the mall Mumbai 12345", addr, BadPincodeLLM(),
    )
    assert result.address is None
    assert result.missing_required is True


def test_address_model_validate_actually_raises_on_bad_pincode():
    """Sanity check on the schema itself, independent of extraction.py --
    confirms the assumption the test above relies on."""
    with pytest.raises(ValidationError):
        Address.model_validate({
            "line1": "x", "city": "Pune", "state": "MH", "pincode": "12345",
        })
