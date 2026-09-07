"""
Order API routes.

INVARIANT: this package imports NOTHING from app.graph, app.providers, or
app.ui. It is a standalone service that happens to be mounted in-process. A
`git mv app/order_api services/order-api` plus a thin main.py is the entire
extraction, should call volume ever justify running it separately.

Enforced by tests/test_order_api.py::test_module_is_liftable.
"""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Query, status
from order_contracts.schemas import (
    AddressUpdateRequest,
    MutationResult,
    Order,
    OrderSummary,
    RescheduleRequest,
)

from order_api.store import get_store

router = APIRouter(prefix="/api/orders", tags=["orders"])


@router.get("", response_model=list[OrderSummary])
def list_orders_by_phone(
    phone: str = Query(..., min_length=10, description="Caller's phone number"),
) -> list[OrderSummary]:
    """
    Open orders for a caller.

    This is the search space for fuzzy order-ID resolution — the agent matches
    a garbled ASR transcript against these rather than trusting it verbatim.
    """
    return get_store().by_phone(phone)


@router.get("/{order_id}", response_model=Order)
def get_order(order_id: str) -> Order:
    order = get_store().get(order_id)
    if order is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Order {order_id} not found")
    return order


@router.post("/{order_id}/reschedule", response_model=MutationResult)
def reschedule(
    order_id: str,
    payload: RescheduleRequest,
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
) -> MutationResult:
    """
    Reschedule a delivery.

    Idempotency-Key is mandatory: a voice agent retrying after a dropped
    WebSocket must never double-book a slot.
    """
    return get_store().reschedule(
        order_id=order_id,
        new_date=payload.new_date,
        window=payload.window,
        idempotency_key=idempotency_key,
        reason=payload.reason,
    )


@router.post("/{order_id}/address", response_model=MutationResult)
def update_address(
    order_id: str,
    payload: AddressUpdateRequest,
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
) -> MutationResult:
    return get_store().update_address(
        order_id=order_id,
        address=payload.address,
        idempotency_key=idempotency_key,
        reason=payload.reason,
    )
