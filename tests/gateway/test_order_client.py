"""
Split-architecture tests.

The gateway runs on Railway, the order API on Vercel. That boundary is only
safe if the two OrderClient implementations are genuinely interchangeable — so
these tests run the SAME assertions through both and demand identical results.

Without this, "it worked locally" and "it works deployed" are separate claims.
"""

from __future__ import annotations

import httpx
import pytest
from api.index import app as order_app
from app.providers.order_client import (
    HttpOrderClient,
    InProcessOrderClient,
    OrderAPIUnavailable,
    OrderClient,
)
from fastapi.testclient import TestClient
from order_api.kv import MemoryBackend, UpstashBackend, reset_backend
from order_api.seed import T, build_fixtures
from order_api.store import OrderStore, get_store
from order_contracts.schemas import Address, SlotWindow


@pytest.fixture(autouse=True)
def fresh_store():
    reset_backend()
    import order_api.store as store_mod

    store_mod._store = None
    store = get_store()
    store.reset()
    store.load(build_fixtures())
    yield store


@pytest.fixture
def http_client() -> HttpOrderClient:
    """
    HttpOrderClient driven against the real order app over the real HTTP stack,
    in-memory. Exercises serialization, status codes, and header handling —
    everything that differs from the in-process path — with no network and no
    deployed service, so CI stays free and fast.

    TestClient is a sync httpx.Client subclass, so it substitutes directly.
    """
    client = HttpOrderClient.__new__(HttpOrderClient)
    client._client = TestClient(order_app, base_url="http://order-api.test")
    return client


@pytest.fixture(params=["in_process", "http"])
def client(request, http_client) -> OrderClient:
    return InProcessOrderClient() if request.param == "in_process" else http_client


# ---- parity: both implementations, same assertions ----------------------


def test_get_order(client: OrderClient):
    order = client.get_order("DLV1001")
    assert order is not None
    assert order.status.value == "out_for_delivery"
    assert order.address.city == "Pune"


def test_get_unknown_order_returns_none(client: OrderClient):
    assert client.get_order("NOPE") is None


def test_orders_for_phone(client: OrderClient):
    summaries = client.orders_for_phone("9990000001")
    assert {s.order_id for s in summaries} == {"DLV1001", "DLV1002", "DLV1003"}


def test_reschedule_success(client: OrderClient):
    result = client.reschedule("DLV1002", T(6), SlotWindow.AFTERNOON, "key-1")
    assert result.ok is True
    assert result.order.reschedule_count == 1


def test_reschedule_refusal_carries_code_and_alternatives(client: OrderClient):
    """A refusal must give the agent something to say AND something to offer."""
    result = client.reschedule("DLV1002", T(3), SlotWindow.MORNING, "key-2")
    assert result.ok is False
    assert result.refusal_code == "SLOT_FULL"
    assert result.alternatives


def test_address_refusal_survives_serialization(client: OrderClient):
    result = client.update_address(
        "DLV1002",
        Address(line1="1 X Road", city="Srinagar", state="JK", pincode="190001"),
        "key-3",
    )
    assert result.refusal_code == "NON_SERVICEABLE"


def test_idempotent_replay(client: OrderClient):
    a = client.reschedule("DLV1002", T(6), SlotWindow.AFTERNOON, "same-key")
    b = client.reschedule("DLV1002", T(6), SlotWindow.AFTERNOON, "same-key")
    assert a.model_dump() == b.model_dump()
    assert client.get_order("DLV1002").reschedule_count == 1


# ---- cross-implementation equivalence -----------------------------------


def test_both_clients_return_identical_orders(http_client):
    """The strongest form of the claim: byte-identical payloads."""
    assert (
        InProcessOrderClient().get_order("DLV1001").model_dump()
        == http_client.get_order("DLV1001").model_dump()
    )


def test_both_clients_return_identical_refusals(http_client):
    a = InProcessOrderClient().reschedule("DLV1005", T(3), SlotWindow.EVENING, "x1")
    b = http_client.reschedule("DLV1005", T(3), SlotWindow.EVENING, "x2")
    assert a.refusal_code == b.refusal_code == "TERMINAL_STATE"


# ---- failure handling ----------------------------------------------------


def test_unreachable_order_api_raises_rather_than_hanging():
    """
    A dead order service must fail fast so the agent can apologise and escalate.
    A hung call is worse than an honest 'let me get a colleague'.
    """
    client = HttpOrderClient.__new__(HttpOrderClient)
    client._client = httpx.Client(
        transport=httpx.MockTransport(
            lambda req: (_ for _ in ()).throw(httpx.ConnectError("down"))
        ),
        base_url="http://order-api.test",
    )
    with pytest.raises(OrderAPIUnavailable):
        client.get_order("DLV1001")


def test_idempotency_keys_are_unique_per_attempt():
    a = OrderClient.new_idempotency_key("sess-1", "reschedule")
    b = OrderClient.new_idempotency_key("sess-1", "reschedule")
    assert a != b and a.startswith("sess-1:reschedule:")


# ---- KV backend ----------------------------------------------------------


def test_writes_persist_across_store_instances():
    """
    Simulates a Vercel cold start: new process, same KV. The reschedule the
    caller just made must still be there when they call back.
    """
    backend = MemoryBackend()
    OrderStore(backend).load(build_fixtures())

    first = OrderStore(backend)
    assert first.reschedule("DLV1002", T(6), SlotWindow.AFTERNOON, "k").ok

    second = OrderStore(backend)  # "cold container"
    assert second.get("DLV1002").reschedule_count == 1
    assert second.get("DLV1002").promised_slot.date == T(6)


def test_idempotency_survives_cold_start():
    """The retry that matters most is the one after the container died."""
    backend = MemoryBackend()
    OrderStore(backend).load(build_fixtures())
    OrderStore(backend).reschedule("DLV1002", T(6), SlotWindow.AFTERNOON, "dup")
    OrderStore(backend).reschedule("DLV1002", T(6), SlotWindow.AFTERNOON, "dup")
    assert OrderStore(backend).get("DLV1002").reschedule_count == 1


def test_slot_capacity_persists_across_instances():
    backend = MemoryBackend()
    OrderStore(backend).load(build_fixtures())
    assert not OrderStore(backend).has_capacity(T(3), SlotWindow.MORNING)


def test_backend_selection_defaults_to_memory(monkeypatch):
    monkeypatch.delenv("UPSTASH_REDIS_REST_URL", raising=False)
    monkeypatch.delenv("UPSTASH_REDIS_REST_TOKEN", raising=False)
    reset_backend()
    from order_api.kv import get_backend

    assert isinstance(get_backend(), MemoryBackend)


def test_backend_selection_uses_upstash_when_configured(monkeypatch):
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://x.upstash.io")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "tok")
    reset_backend()
    from order_api.kv import get_backend

    assert isinstance(get_backend(), UpstashBackend)
    reset_backend()
