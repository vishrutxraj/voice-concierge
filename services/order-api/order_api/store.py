"""
In-memory order store with genuine business constraints.

Swap this module for a real WMS client and nothing else in the project changes
— that is the point of keeping order_api free of gateway imports.
"""

from __future__ import annotations

import threading
from datetime import date, datetime, timedelta

from order_contracts.schemas import (
    Address,
    DeliverySlot,
    MutationResult,
    Order,
    OrderStatus,
    OrderSummary,
    SlotWindow,
    TrackingEvent,
)

from order_api.kv import KVBackend, MemoryBackend, get_backend

# --------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------

# Public holidays / no-delivery days. Reschedule must refuse these.
BLACKOUT_DATES: set[date] = set()

# Pincodes we simply do not serve. Address correction must refuse these.
NON_SERVICEABLE_PINCODES: set[str] = {"190001", "796001", "737101"}

# Per (date, window) capacity ceiling. Reschedule must offer alternatives when full.
SLOT_CAPACITY: int = 4

MAX_RESCHEDULE_HORIZON_DAYS = 14


ORDER_KEY = "order:"
IDEM_KEY = "idem:"
LOAD_KEY = "slotload"


class OrderStore:
    """
    Order store backed by a pluggable KV.

    Every mutation writes through immediately, because on Vercel the process
    may not exist by the time the caller speaks their next sentence.
    """

    def __init__(self, backend: KVBackend | None = None) -> None:
        self._lock = threading.RLock()
        self._kv = backend or get_backend()

    # ---- KV helpers ------------------------------------------------------

    def _read_order(self, order_id: str) -> Order | None:
        raw = self._kv.get(ORDER_KEY + order_id)
        return Order.model_validate(raw) if raw else None

    def _write_order(self, order: Order) -> None:
        self._kv.set(ORDER_KEY + order.order_id, order.model_dump(mode="json"))

    def _read_loads(self) -> dict[str, int]:
        return self._kv.get(LOAD_KEY) or {}

    def _write_loads(self, loads: dict[str, int]) -> None:
        self._kv.set(LOAD_KEY, loads)

    @staticmethod
    def _slot_key(d: date, w: SlotWindow) -> str:
        return f"{d.isoformat()}|{w.value}"

    # ---- reads -----------------------------------------------------------

    def get(self, order_id: str) -> Order | None:
        with self._lock:
            return self._read_order(order_id.upper().strip())

    def by_phone(self, phone: str) -> list[OrderSummary]:
        """Open orders for a caller — the search space for fuzzy ID matching."""
        digits = "".join(c for c in phone if c.isdigit())[-10:]
        with self._lock:
            ids = self._kv.get("phone:" + digits) or []
            return [
                OrderSummary(
                    order_id=o.order_id,
                    status=o.status,
                    promised_slot=o.promised_slot,
                    items_preview=", ".join(o.items[:2]) or "—",
                )
                for oid in ids
                if (o := self._read_order(oid)) and o.is_mutable
            ]

    def all_ids(self) -> list[str]:
        with self._lock:
            return [k.removeprefix(ORDER_KEY) for k in self._kv.keys(ORDER_KEY)]

    # ---- capacity --------------------------------------------------------

    def _load(self, d: date, w: SlotWindow) -> int:
        return self._read_loads().get(self._slot_key(d, w), 0)

    def has_capacity(self, d: date, w: SlotWindow) -> bool:
        return self._load(d, w) < SLOT_CAPACITY

    def suggest_slots(self, near: date, limit: int = 3) -> list[DeliverySlot]:
        """Next available slots — what the agent offers after a refusal."""
        out: list[DeliverySlot] = []
        for offset in range(0, MAX_RESCHEDULE_HORIZON_DAYS + 1):
            d = near + timedelta(days=offset)
            if d < date.today() or d in BLACKOUT_DATES:
                continue
            for w in SlotWindow:
                if self.has_capacity(d, w):
                    out.append(DeliverySlot(date=d, window=w))
                    if len(out) >= limit:
                        return out
        return out

    # ---- mutations -------------------------------------------------------

    def _idem(self, key: str | None) -> MutationResult | None:
        if not key:
            return None
        raw = self._kv.get(IDEM_KEY + key)
        return MutationResult.model_validate(raw) if raw else None

    def _remember(self, key: str | None, result: MutationResult) -> MutationResult:
        if key:
            self._kv.set(IDEM_KEY + key, result.model_dump(mode="json"))
        return result

    def reschedule(
        self,
        order_id: str,
        new_date: date,
        window: SlotWindow,
        idempotency_key: str | None = None,
        reason: str | None = None,
    ) -> MutationResult:
        with self._lock:
            if (cached := self._idem(idempotency_key)) is not None:
                return cached

            order = self.get(order_id)
            if order is None:
                return self._remember(
                    idempotency_key,
                    MutationResult(
                        ok=False,
                        order_id=order_id,
                        message="No such order.",
                        refusal_code="NOT_FOUND",
                    ),
                )

            def refuse(code: str, msg: str, alts: list[DeliverySlot] | None = None):
                return self._remember(
                    idempotency_key,
                    MutationResult(
                        ok=False,
                        order_id=order.order_id,
                        message=msg,
                        refusal_code=code,
                        order=order,
                        alternatives=alts or [],
                    ),
                )

            if not order.is_mutable:
                return refuse(
                    "TERMINAL_STATE",
                    f"This order is already {order.status.value.replace('_', ' ')} "
                    "and can no longer be rescheduled.",
                )
            if order.reschedule_count >= order.max_reschedules:
                return refuse(
                    "RESCHEDULE_LIMIT",
                    f"This order has already been rescheduled "
                    f"{order.reschedule_count} times, which is the maximum. "
                    "I'll need to pass you to a colleague.",
                )
            if new_date < date.today():
                return refuse("PAST_DATE", "That date has already passed.")
            if new_date > date.today() + timedelta(days=MAX_RESCHEDULE_HORIZON_DAYS):
                return refuse(
                    "BEYOND_HORIZON",
                    f"I can only reschedule up to "
                    f"{MAX_RESCHEDULE_HORIZON_DAYS} days ahead.",
                )
            if new_date in BLACKOUT_DATES:
                return refuse(
                    "BLACKOUT",
                    "We don't deliver on that date.",
                    self.suggest_slots(new_date + timedelta(days=1)),
                )
            if not self.has_capacity(new_date, window):
                return refuse(
                    "SLOT_FULL",
                    "That slot is fully booked.",
                    self.suggest_slots(new_date),
                )

            # commit
            old = order.promised_slot
            loads = self._read_loads()
            old_key = self._slot_key(old.date, old.window)
            new_key = self._slot_key(new_date, window)
            loads[old_key] = max(0, loads.get(old_key, 0) - 1)
            loads[new_key] = loads.get(new_key, 0) + 1
            self._write_loads(loads)
            order.promised_slot = DeliverySlot(date=new_date, window=window)
            order.reschedule_count += 1
            order.tracking.append(
                TrackingEvent(
                    ts=datetime.now(),
                    status=order.status,
                    location=order.address.city,
                    note=f"Rescheduled to {order.promised_slot.human()}"
                    + (f" — {reason}" if reason else ""),
                )
            )
            self._write_order(order)
            return self._remember(
                idempotency_key,
                MutationResult(
                    ok=True,
                    order_id=order.order_id,
                    message=f"Rescheduled to {order.promised_slot.human()}.",
                    order=order,
                ),
            )

    def update_address(
        self,
        order_id: str,
        address: Address,
        idempotency_key: str | None = None,
        reason: str | None = None,
    ) -> MutationResult:
        with self._lock:
            if (cached := self._idem(idempotency_key)) is not None:
                return cached

            order = self.get(order_id)
            if order is None:
                return self._remember(
                    idempotency_key,
                    MutationResult(
                        ok=False,
                        order_id=order_id,
                        message="No such order.",
                        refusal_code="NOT_FOUND",
                    ),
                )
            if not order.is_mutable:
                return self._remember(
                    idempotency_key,
                    MutationResult(
                        ok=False,
                        order_id=order.order_id,
                        message=f"This order is already "
                        f"{order.status.value.replace('_', ' ')}.",
                        refusal_code="TERMINAL_STATE",
                        order=order,
                    ),
                )
            if address.pincode in NON_SERVICEABLE_PINCODES:
                return self._remember(
                    idempotency_key,
                    MutationResult(
                        ok=False,
                        order_id=order.order_id,
                        message=f"We don't currently deliver to {address.pincode}.",
                        refusal_code="NON_SERVICEABLE",
                        order=order,
                    ),
                )
            # Changing city after dispatch means a re-route, not an edit.
            if (
                order.status is OrderStatus.OUT_FOR_DELIVERY
                and address.city.lower() != order.address.city.lower()
            ):
                return self._remember(
                    idempotency_key,
                    MutationResult(
                        ok=False,
                        order_id=order.order_id,
                        message="The parcel is already out for delivery in "
                        f"{order.address.city}; a different city needs a human "
                        "to arrange a re-route.",
                        refusal_code="OUT_FOR_DELIVERY_CITY_CHANGE",
                        order=order,
                    ),
                )

            order.address = address
            order.tracking.append(
                TrackingEvent(
                    ts=datetime.now(),
                    status=order.status,
                    location=address.city,
                    note="Address updated" + (f" — {reason}" if reason else ""),
                )
            )
            self._write_order(order)
            return self._remember(
                idempotency_key,
                MutationResult(
                    ok=True,
                    order_id=order.order_id,
                    message=f"Address updated to {address.one_line()}.",
                    order=order,
                ),
            )

    # ---- seeding ---------------------------------------------------------

    def load(self, orders: list[Order]) -> None:
        with self._lock:
            phone_index: dict[str, list[str]] = {}
            loads: dict[str, int] = {}
            for o in orders:
                self._write_order(o)
                digits = "".join(c for c in o.customer_phone if c.isdigit())[-10:]
                phone_index.setdefault(digits, []).append(o.order_id)
                key = self._slot_key(o.promised_slot.date, o.promised_slot.window)
                loads[key] = loads.get(key, 0) + 1
            for digits, ids in phone_index.items():
                self._kv.set("phone:" + digits, ids)
            self._write_loads(loads)

    def is_seeded(self) -> bool:
        """Cold Vercel containers self-seed on first request."""
        return bool(self._kv.keys(ORDER_KEY))

    def reset(self) -> None:
        with self._lock:
            if isinstance(self._kv, MemoryBackend):
                self._kv.clear()
            else:
                for key in self._kv.keys(""):
                    self._kv.delete(key)


_store: OrderStore | None = None


def get_store() -> OrderStore:
    global _store
    if _store is None:
        _store = OrderStore()
        from order_api.seed import build_fixtures

        _store.load(build_fixtures())
    return _store
