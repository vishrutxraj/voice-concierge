from __future__ import annotations

import ast
import pathlib

import pytest
from api.index import app
from fastapi.testclient import TestClient
from order_api.kv import reset_backend
from order_api.seed import T
from order_api.store import get_store
from order_contracts.schemas import SlotWindow

client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh_store():
    """Each test gets a clean store — mutations must not leak between tests."""
    reset_backend()
    import order_api.store as store_mod

    store_mod._store = None
    store = get_store()
    store.reset()
    from order_api.seed import build_fixtures

    store.load(build_fixtures())
    yield store


def _resched(order_id: str, day, window, key="k1"):
    return client.post(
        f"/api/orders/{order_id}/reschedule",
        json={"new_date": day.isoformat(), "window": window.value},
        headers={"Idempotency-Key": key},
    )


# ---- reads ---------------------------------------------------------------


def test_health_reports_backend():
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["orders"] == 20


def test_get_order_returns_full_wms_shape():
    body = client.get("/api/orders/DLV1001").json()
    assert body["order_id"] == "DLV1001"
    assert body["status"] == "out_for_delivery"
    assert len(body["tracking"]) >= 3
    assert body["address"]["pincode"] == "411014"


def test_order_id_is_case_insensitive():
    assert client.get("/api/orders/dlv1001").status_code == 200


def test_unknown_order_404s():
    assert client.get("/api/orders/NOPE").status_code == 404


def test_phone_lookup_returns_only_open_orders():
    """The fuzzy-matching search space must exclude terminal orders."""
    summaries = client.get("/api/orders", params={"phone": "9990000001"}).json()
    assert {s["order_id"] for s in summaries} == {"DLV1001", "DLV1002", "DLV1003"}


def test_phone_lookup_tolerates_country_code():
    a = client.get("/api/orders", params={"phone": "+919990000001"}).json()
    b = client.get("/api/orders", params={"phone": "9990000001"}).json()
    assert a == b


def test_delivered_order_absent_from_open_lookup():
    assert client.get("/api/orders", params={"phone": "9990000003"}).json() == []


# ---- reschedule: the happy path -----------------------------------------


def test_reschedule_succeeds_and_increments_count():
    r = _resched("DLV1002", T(6), SlotWindow.AFTERNOON).json()
    assert r["ok"] is True
    assert r["order"]["reschedule_count"] == 1
    assert "Rescheduled to" in r["message"]


def test_reschedule_appends_tracking_event():
    before = len(client.get("/api/orders/DLV1002").json()["tracking"])
    _resched("DLV1002", T(6), SlotWindow.AFTERNOON)
    after = client.get("/api/orders/DLV1002").json()["tracking"]
    assert len(after) == before + 1
    assert "Rescheduled" in after[-1]["note"]


# ---- reschedule: every refusal branch ------------------------------------


def test_refuses_terminal_state():
    r = _resched("DLV1005", T(3), SlotWindow.EVENING).json()
    assert r["ok"] is False and r["refusal_code"] == "TERMINAL_STATE"


def test_refuses_when_at_reschedule_cap():
    r = _resched("DLV1006", T(6), SlotWindow.AFTERNOON).json()
    assert r["refusal_code"] == "RESCHEDULE_LIMIT"


def test_refuses_past_date():
    r = _resched("DLV1002", T(-1), SlotWindow.MORNING).json()
    assert r["refusal_code"] == "PAST_DATE"


def test_refuses_beyond_horizon():
    r = _resched("DLV1002", T(30), SlotWindow.MORNING).json()
    assert r["refusal_code"] == "BEYOND_HORIZON"


def test_refuses_blackout_date_and_offers_alternatives():
    r = _resched("DLV1002", T(5), SlotWindow.MORNING).json()
    assert r["refusal_code"] == "BLACKOUT"
    assert len(r["alternatives"]) > 0, "agent needs something to offer the caller"


def test_refuses_full_slot_and_offers_alternatives():
    """T(3)/MORNING is saturated by four seeded orders."""
    r = _resched("DLV1002", T(3), SlotWindow.MORNING).json()
    assert r["refusal_code"] == "SLOT_FULL"
    assert r["alternatives"]
    assert all(a["date"] != T(3).isoformat() or a["window"] != "morning"
               for a in r["alternatives"])


def test_refusals_never_mutate_the_order():
    before = client.get("/api/orders/DLV1002").json()
    _resched("DLV1002", T(3), SlotWindow.MORNING)
    assert client.get("/api/orders/DLV1002").json() == before


# ---- idempotency ---------------------------------------------------------


def test_same_idempotency_key_does_not_double_apply():
    first = _resched("DLV1002", T(6), SlotWindow.AFTERNOON, key="abc").json()
    second = _resched("DLV1002", T(6), SlotWindow.AFTERNOON, key="abc").json()
    assert first == second
    assert client.get("/api/orders/DLV1002").json()["reschedule_count"] == 1


def test_missing_idempotency_key_is_rejected():
    r = client.post(
        "/api/orders/DLV1002/reschedule",
        json={"new_date": T(6).isoformat(), "window": "morning"},
    )
    assert r.status_code == 422


def test_capacity_is_released_on_successful_reschedule():
    store = get_store()
    original = store.get("DLV1011").promised_slot  # T(3)/MORNING, saturated
    _resched("DLV1011", T(7), SlotWindow.EVENING, key="move")
    assert store.has_capacity(original.date, original.window), (
        "vacating a full slot must free capacity for the next caller"
    )


# ---- address -------------------------------------------------------------


def _addr_payload(city="Pune", pincode="411014"):
    return {
        "address": {
            "line1": "99 New Street",
            "city": city,
            "state": "Maharashtra",
            "pincode": pincode,
        }
    }


def test_address_update_succeeds():
    r = client.post(
        "/api/orders/DLV1002/address",
        json=_addr_payload(),
        headers={"Idempotency-Key": "a1"},
    ).json()
    assert r["ok"] is True and "99 New Street" in r["message"]


def test_address_refuses_non_serviceable_pincode():
    r = client.post(
        "/api/orders/DLV1002/address",
        json=_addr_payload(city="Srinagar", pincode="190001"),
        headers={"Idempotency-Key": "a2"},
    ).json()
    assert r["refusal_code"] == "NON_SERVICEABLE"


def test_address_refuses_city_change_when_out_for_delivery():
    r = client.post(
        "/api/orders/DLV1001/address",
        json=_addr_payload(city="Mumbai", pincode="400001"),
        headers={"Idempotency-Key": "a3"},
    ).json()
    assert r["refusal_code"] == "OUT_FOR_DELIVERY_CITY_CHANGE"


def test_address_allows_same_city_correction_when_out_for_delivery():
    """Typo fixes must still work on the day of delivery — the common case."""
    r = client.post(
        "/api/orders/DLV1001/address",
        json=_addr_payload(city="Pune", pincode="411015"),
        headers={"Idempotency-Key": "a4"},
    ).json()
    assert r["ok"] is True


def test_invalid_pincode_rejected_by_schema():
    r = client.post(
        "/api/orders/DLV1002/address",
        json=_addr_payload(pincode="abc"),
        headers={"Idempotency-Key": "a5"},
    )
    assert r.status_code == 422


# ---- architectural invariant --------------------------------------------


FORBIDDEN_PREFIXES = ("app.graph", "app.providers", "app.ui", "app.guardrails")


def test_module_is_liftable():
    """
    order_api must import nothing from the gateway.

    This is what makes `git mv app/order_api services/order-api` a one-step
    extraction. Enforced mechanically so it cannot rot.
    """
    pkg = pathlib.Path(__file__).parents[2] / "services" / "order-api" / "order_api"
    offenders: list[str] = []
    for path in pkg.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                if m.startswith(FORBIDDEN_PREFIXES):
                    offenders.append(f"{path.name}: {m}")
    assert not offenders, f"order_api leaked gateway imports: {offenders}"
