"""
Gateway -> Order API client.

Two implementations behind one interface:

  HttpOrderClient      talks to the Vercel deployment. Production path.
  InProcessOrderClient imports the store directly. Local dev, CI, and tests.

Why bother with the second: the test suite must run with no network and no
deployed service, and phase 2 (the whole LangGraph build) should cost nothing
and run in milliseconds. Selection is by env — ORDER_API_BASE_URL present means
HTTP, absent means in-process. Nothing in the graph knows the difference.

The retry policy matters more than it looks. A voice call cannot wait 30s for a
backoff chain, so timeouts are tight and failures surface as a spoken apology
plus an escalation, not a hung line.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import date

from order_contracts.schemas import (
    Address,
    MutationResult,
    Order,
    OrderSummary,
    SlotWindow,
)

from app.config import get_settings


class OrderAPIUnavailable(RuntimeError):
    """Raised when the order service cannot be reached. Triggers escalation."""


class OrderClient(ABC):
    @abstractmethod
    def get_order(self, order_id: str) -> Order | None: ...

    @abstractmethod
    def orders_for_phone(self, phone: str) -> list[OrderSummary]: ...

    @abstractmethod
    def reschedule(
        self,
        order_id: str,
        new_date: date,
        window: SlotWindow,
        idempotency_key: str,
        reason: str | None = None,
    ) -> MutationResult: ...

    @abstractmethod
    def update_address(
        self,
        order_id: str,
        address: Address,
        idempotency_key: str,
        reason: str | None = None,
    ) -> MutationResult: ...

    @staticmethod
    def new_idempotency_key(session_id: str, intent: str) -> str:
        """
        Stable per (call, intent, attempt).

        Derived from the session so a retry after a dropped WebSocket reuses the
        same key and cannot double-book a slot.
        """
        return f"{session_id}:{intent}:{uuid.uuid4().hex[:8]}"


class InProcessOrderClient(OrderClient):
    def __init__(self) -> None:
        from order_api.store import get_store

        self._store = get_store()

    def get_order(self, order_id: str) -> Order | None:
        return self._store.get(order_id)

    def orders_for_phone(self, phone: str) -> list[OrderSummary]:
        return self._store.by_phone(phone)

    def reschedule(self, order_id, new_date, window, idempotency_key, reason=None):
        return self._store.reschedule(
            order_id, new_date, window, idempotency_key, reason
        )

    def update_address(self, order_id, address, idempotency_key, reason=None):
        return self._store.update_address(order_id, address, idempotency_key, reason)


class HttpOrderClient(OrderClient):
    def __init__(self, base_url: str, timeout: float = 4.0) -> None:
        import httpx

        # Tight timeout: a caller is on the line. Slow is indistinguishable
        # from broken at conversational latency.
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def _post(self, path: str, payload: dict, idempotency_key: str) -> MutationResult:
        import httpx

        try:
            resp = self._client.post(
                path, json=payload, headers={"Idempotency-Key": idempotency_key}
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OrderAPIUnavailable(str(exc)) from exc
        return MutationResult.model_validate(resp.json())

    def get_order(self, order_id: str) -> Order | None:
        import httpx

        try:
            resp = self._client.get(f"/api/orders/{order_id}")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OrderAPIUnavailable(str(exc)) from exc
        return Order.model_validate(resp.json())

    def orders_for_phone(self, phone: str) -> list[OrderSummary]:
        import httpx

        try:
            resp = self._client.get("/api/orders", params={"phone": phone})
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OrderAPIUnavailable(str(exc)) from exc
        return [OrderSummary.model_validate(o) for o in resp.json()]

    def reschedule(self, order_id, new_date, window, idempotency_key, reason=None):
        return self._post(
            f"/api/orders/{order_id}/reschedule",
            {
                "new_date": new_date.isoformat(),
                "window": window.value if hasattr(window, "value") else window,
                "reason": reason,
            },
            idempotency_key,
        )

    def update_address(self, order_id, address, idempotency_key, reason=None):
        return self._post(
            f"/api/orders/{order_id}/address",
            {"address": address.model_dump(mode="json"), "reason": reason},
            idempotency_key,
        )


_client: OrderClient | None = None


def get_order_client() -> OrderClient:
    global _client
    if _client is None:
        base = get_settings().order_api_base_url
        _client = HttpOrderClient(base) if base else InProcessOrderClient()
    return _client


def reset_order_client() -> None:
    global _client
    _client = None
