"""
order_reference: "which of the caller's orders did they mean?"

Pure functions, so these tests are exhaustive and instant. The orders are the
seeded 3-order caller (ANITA, 9990000001) -- the exact case where a real session
went wrong: the caller said "yoga mat" and was handed to a human.
"""

from __future__ import annotations

import pytest
from app.graph.dialogue import live_pending, new_pending, turn_number
from app.graph.order_reference import OrderRef, match_order_reference

A = OrderRef("DLV1001", "Bluetooth speaker", "out_for_delivery")
B = OrderRef("DLV1002", "Cotton bedsheet set, Pillow covers", "in_transit")
C = OrderRef("DLV1003", "Yoga mat", "pending")
ORDERS = [A, B, C]


def resolve(text, orders=ORDERS, **kw):
    return match_order_reference(text, orders, **kw)


# ---- item names -------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Tell me where my yoga mat order is.", "DLV1003"),  # the reported utterance
        ("yoga mat", "DLV1003"),
        ("where is the yoga matt", "DLV1003"),  # ASR/translation typo
        ("the bluetooth speaker", "DLV1001"),
        ("my bluetooth speakers", "DLV1001"),  # plural
        ("the cotton bedsheet", "DLV1002"),  # "set" is filler
        ("the pillow covers", "DLV1002"),  # the SECOND item of a two-item order
    ],
)
def test_item_names_identify_the_order(text, expected):
    m = resolve(text)
    assert m.status == "unique" and m.order_id == expected and "item" in m.signals


def test_one_word_of_a_multi_word_item_is_only_trusted_when_answering():
    # "speaker" alone is half of "Bluetooth speaker". Cold, that could be an
    # accident of phrasing; as the answer to our question, it's clearly the one.
    assert resolve("the speaker").status == "none"
    m = resolve("the speaker", answering=True)
    assert m.status == "unique" and m.order_id == "DLV1001"


def test_a_volunteered_common_word_does_not_pick_an_order():
    shoes = [OrderRef("X1", "Running shoes", "pending"), OrderRef("X2", "Yoga mat", "in_transit")]
    assert resolve("I'm running late", shoes).status == "none"
    # ...but as the answer to "which order?", the same word is trusted.
    assert resolve("running", shoes, answering=True).order_id == "X1"


# ---- order IDs, however ASR renders them -------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("DLV1003", "DLV1003"),
        ("dlv-1001", "DLV1001"),
        ("D L V one zero zero three", "DLV1003"),
        ("delivery D L V one zero zero two please", "DLV1002"),
        ("order 1003", "DLV1003"),  # trailing digits alone
    ],
)
def test_order_ids_are_recognised(text, expected):
    m = resolve(text)
    assert m.status == "unique" and m.order_id == expected and m.signals == ["id"]


def test_an_explicit_id_beats_other_signals():
    assert resolve("DLV1001 not the yoga mat").order_id == "DLV1001"


def test_short_digit_strings_never_select_an_order():
    assert resolve("reschedule to 10 am on the 3rd").status == "none"


# ---- status ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("the pending one", "DLV1003"),
        ("the one that hasn't shipped yet", "DLV1003"),  # contains "shipped", still pending
        ("the one out for delivery", "DLV1001"),
        ("the one in transit", "DLV1002"),
        ("the one that's on its way", "DLV1002"),
        ("the one that already shipped", "DLV1002"),
    ],
)
def test_status_phrases_identify_the_order(text, expected):
    m = resolve(text)
    assert m.status == "unique" and m.order_id == expected


def test_a_status_two_orders_share_is_ambiguous_not_a_guess():
    two_pending = [OrderRef("P1", "Lamp", "pending"), OrderRef("P2", "Desk", "pending"), B]
    m = resolve("the pending one", two_pending)
    assert m.status == "ambiguous" and set(m.candidates) == {"P1", "P2"}


# ---- ordinals: only meaningful as an answer, and only in a short reply ------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("the first one", "DLV1001"),
        ("second", "DLV1002"),
        ("the third one", "DLV1003"),
        ("the last one", "DLV1003"),
        ("3rd", "DLV1003"),
    ],
)
def test_ordinals_refer_to_the_list_we_read_out(text, expected):
    assert resolve(text, answering=True).order_id == expected


def test_ordinals_mean_nothing_unless_we_just_asked():
    assert resolve("the third one").status == "none"


def test_ordinal_position_follows_the_order_the_options_were_read_in():
    read_out = [C, A, B]  # a different order than the caller's account listing
    assert resolve("the first one", answering=True, ordinal_basis=read_out).order_id == "DLV1003"
    assert resolve("the last one", answering=True, ordinal_basis=read_out).order_id == "DLV1002"


def test_a_long_sentence_containing_first_is_not_an_ordinal():
    text = "reschedule it to the first of next month please thanks"
    assert resolve(text, answering=True).status == "none"


# ---- combining signals ---------------------------------------------------------------------


def test_agreeing_signals_resolve():
    m = resolve("the pending yoga mat")
    assert m.status == "unique" and m.order_id == "DLV1003"
    assert m.signals == ["item", "status"]


def test_narrowing_signals_intersect():
    orders = [OrderRef("P1", "Lamp", "pending"), OrderRef("P2", "Desk", "pending")]
    m = resolve("the pending lamp", orders)
    assert m.status == "unique" and m.order_id == "P1"


def test_contradicting_signals_are_ambiguous_never_silently_picked():
    m = resolve("the third one, the speaker", answering=True)
    assert m.status == "ambiguous"
    assert set(m.candidates) == {"DLV1001", "DLV1003"}


# ---- nothing to go on -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["where is my order", "banana", "reschedule for friday to evening", "", "   "],
)
def test_no_reference_means_none(text):
    assert resolve(text, answering=True).status == "none"


def test_no_orders_means_none():
    assert resolve("yoga mat", []).status == "none"


# ---- dialogue memory expiry ---------------------------------------------------------------------


def test_a_pending_question_is_only_live_on_the_next_turn():
    asked_on_turn_2 = new_pending({"turns": [1, 2]}, intent="order_lookup")
    assert asked_on_turn_2["at_turn"] == 2

    def state(turns):
        return {"turns": list(range(turns)), "pending_followup": asked_on_turn_2}

    assert live_pending(state(3), "pending_followup") == asked_on_turn_2  # the reply
    assert live_pending(state(2), "pending_followup") is None  # same turn
    assert live_pending(state(4), "pending_followup") is None  # ignored -> expired
    assert live_pending({"turns": [1, 2, 3]}, "missing") is None
    assert turn_number({}) == 0
