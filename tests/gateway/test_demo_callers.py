"""
The web UI's "Call as" list (services/web/src/lib/demoCallers.ts) is a
hand-written copy of phone numbers that only mean something because the order
API's seed data happens to contain them. Nothing links the two at build time,
so this test does: renumber or drop a seeded caller and the demo option that
pointed at it would silently become "no open orders".
"""

from __future__ import annotations

import pathlib
import re

import pytest
from order_api.seed import build_fixtures
from order_api.store import get_store

DEMO_FILE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "services" / "web" / "src" / "lib" / "demoCallers.ts"
)


def _demo_callers() -> list[dict]:
    text = DEMO_FILE.read_text(encoding="utf-8")
    return [
        {"phone": m.group(1), "label": m.group(2)}
        for m in re.finditer(r'phone:\s*"(\d+)",\s*label:\s*"([^"]+)"', text)
    ]


@pytest.fixture(scope="module")
def store():
    import order_api.store as store_mod

    store_mod._store = None
    s = get_store()
    s.reset()
    s.load(build_fixtures())
    return s


def test_the_demo_list_parses_and_is_reasonably_rich():
    callers = _demo_callers()
    assert len(callers) >= 10
    assert len({c["phone"] for c in callers}) == len(callers)


def test_every_demo_phone_is_a_seeded_caller(store):
    seeded = {o.customer_phone for o in build_fixtures()}
    missing = [c for c in _demo_callers() if c["phone"] not in seeded]
    assert not missing, f"demo callers with no seeded orders: {missing}"


def test_callers_whose_label_promises_open_orders_actually_have_them(store):
    for c in _demo_callers():
        open_orders = store.by_phone(c["phone"])
        if "No open orders" in c["label"]:
            assert open_orders == [], c
        else:
            assert open_orders, f"{c['label']} ({c['phone']}) has no open orders"


@pytest.mark.parametrize(
    "phone, expected_open",
    [("9990000001", 3), ("9990000019", 2), ("9990000020", 3)],
)
def test_the_multi_order_callers_have_the_orders_their_labels_describe(store, phone, expected_open):
    assert len(store.by_phone(phone)) == expected_open


def test_divyas_pending_orders_really_are_two(store):
    """The 'both pending' demo only demonstrates narrowing if it is true."""
    statuses = [o.status.value for o in store.by_phone("9990000020")]
    assert statuses.count("pending") == 2
