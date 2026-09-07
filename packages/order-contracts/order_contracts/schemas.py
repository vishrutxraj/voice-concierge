"""
Order domain models.

Shaped after a real WMS rather than a demo stub, because the reschedule agent's
whole value is *refusing* impossible requests. A store that always says yes
proves nothing. So: attempt limits, blackout dates, per-slot capacity, SLA
windows, and serviceability by pincode all exist and all bite.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field


class OrderStatus(str, Enum):
    PENDING = "pending"
    IN_TRANSIT = "in_transit"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    FAILED_ATTEMPT = "failed_attempt"
    RTO_INITIATED = "rto_initiated"  # return to origin
    CANCELLED = "cancelled"


class SlotWindow(str, Enum):
    MORNING = "morning"  # 09:00-13:00
    AFTERNOON = "afternoon"  # 13:00-17:00
    EVENING = "evening"  # 17:00-21:00


SLOT_HOURS: dict[SlotWindow, tuple[str, str]] = {
    SlotWindow.MORNING: ("09:00", "13:00"),
    SlotWindow.AFTERNOON: ("13:00", "17:00"),
    SlotWindow.EVENING: ("17:00", "21:00"),
}


class TrackingEvent(BaseModel):
    ts: datetime
    status: OrderStatus
    location: str
    note: str | None = None


class Address(BaseModel):
    line1: str
    line2: str | None = None
    city: str
    state: str
    pincode: str = Field(pattern=r"^[1-9]\d{5}$")
    landmark: str | None = None

    def one_line(self) -> str:
        parts = [self.line1, self.line2, self.landmark, self.city, self.pincode]
        return ", ".join(p for p in parts if p)


class DeliverySlot(BaseModel):
    date: date
    window: SlotWindow

    def human(self) -> str:
        start, end = SLOT_HOURS[self.window]
        return f"{self.date.strftime('%A %d %B')} between {start} and {end}"


class Order(BaseModel):
    order_id: str
    customer_phone: str
    customer_name: str
    status: OrderStatus
    address: Address
    promised_slot: DeliverySlot
    sla_deadline: date
    delivery_attempts: int = 0
    max_attempts: int = 3
    reschedule_count: int = 0
    max_reschedules: int = 2
    is_cod: bool = False
    cod_amount: float | None = None
    items: list[str] = Field(default_factory=list)
    tracking: list[TrackingEvent] = Field(default_factory=list)

    @property
    def is_mutable(self) -> bool:
        """Terminal states cannot be rescheduled or re-addressed."""
        return self.status not in {
            OrderStatus.DELIVERED,
            OrderStatus.CANCELLED,
            OrderStatus.RTO_INITIATED,
        }

    def latest_event(self) -> TrackingEvent | None:
        return max(self.tracking, key=lambda e: e.ts) if self.tracking else None


# ---- Request / response payloads -----------------------------------------


class RescheduleRequest(BaseModel):
    new_date: date
    window: SlotWindow
    reason: str | None = None


class AddressUpdateRequest(BaseModel):
    address: Address
    reason: str | None = None


class MutationResult(BaseModel):
    ok: bool
    order_id: str
    message: str
    order: Order | None = None
    # Populated on refusal so the agent can explain itself and offer options.
    refusal_code: str | None = None
    alternatives: list[DeliverySlot] = Field(default_factory=list)


class OrderSummary(BaseModel):
    """Lightweight shape for phone-number lookup / fuzzy ID matching."""

    order_id: str
    status: OrderStatus
    promised_slot: DeliverySlot
    items_preview: str
