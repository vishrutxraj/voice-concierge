"""
Fixture data.

Chosen so that every refusal branch in the store is reachable from the demo:
a delivered order (TERMINAL_STATE), one at its reschedule cap
(RESCHEDULE_LIMIT), a deliberately saturated slot (SLOT_FULL), a blackout date,
an out-for-delivery order in a city that can't change, and a caller with three
open orders so fuzzy ID matching has something to actually disambiguate.

Names and numbers are synthetic. The phone numbers use the 999xxxxxxx range
which is not allocated to subscribers.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from order_contracts.schemas import (
    Address,
    DeliverySlot,
    Order,
    OrderStatus,
    SlotWindow,
    TrackingEvent,
)

from order_api.store import BLACKOUT_DATES

TODAY = date.today()
T = lambda n: TODAY + timedelta(days=n)  # noqa: E731


def _register_blackouts() -> None:
    BLACKOUT_DATES.clear()
    BLACKOUT_DATES.update({T(5), T(12)})


def _addr(line1, city, state, pincode, line2=None, landmark=None) -> Address:
    return Address(
        line1=line1,
        line2=line2,
        city=city,
        state=state,
        pincode=pincode,
        landmark=landmark,
    )


def _track(order_id: str, status: OrderStatus, city: str) -> list[TrackingEvent]:
    """A plausible event chain ending in the order's current status."""
    now = datetime.now()
    chain = [
        TrackingEvent(
            ts=now - timedelta(days=3),
            status=OrderStatus.PENDING,
            location="Bhiwandi Hub",
            note="Shipment manifested",
        ),
        TrackingEvent(
            ts=now - timedelta(days=2),
            status=OrderStatus.IN_TRANSIT,
            location="Nagpur Sort Centre",
            note="Departed facility",
        ),
    ]
    if status in {
        OrderStatus.OUT_FOR_DELIVERY,
        OrderStatus.DELIVERED,
        OrderStatus.FAILED_ATTEMPT,
        OrderStatus.RTO_INITIATED,
    }:
        chain.append(
            TrackingEvent(
                ts=now - timedelta(hours=8),
                status=OrderStatus.OUT_FOR_DELIVERY,
                location=f"{city} Delivery Centre",
                note="Out for delivery",
            )
        )
    if status is OrderStatus.FAILED_ATTEMPT:
        chain.append(
            TrackingEvent(
                ts=now - timedelta(hours=4),
                status=status,
                location=city,
                note="Customer unavailable",
            )
        )
    if status is OrderStatus.DELIVERED:
        chain.append(
            TrackingEvent(
                ts=now - timedelta(hours=3),
                status=status,
                location=city,
                note="Delivered, signed by customer",
            )
        )
    if status is OrderStatus.RTO_INITIATED:
        chain.append(
            TrackingEvent(
                ts=now - timedelta(hours=2),
                status=status,
                location=city,
                note="Three attempts failed, returning to origin",
            )
        )
    return chain


_SPEC = [
    # (id, phone, name, status, addr, slot, attempts, resched, cod, items)
    ("DLV1001", "9990000001", "Anita Rao", OrderStatus.OUT_FOR_DELIVERY,
     ("12 Nehru Nagar", "Pune", "Maharashtra", "411014", None, "Near Symbiosis"),
     (T(0), SlotWindow.EVENING), 0, 0, False, ["Bluetooth speaker"]),
    ("DLV1002", "9990000001", "Anita Rao", OrderStatus.IN_TRANSIT,
     ("12 Nehru Nagar", "Pune", "Maharashtra", "411014", None, None),
     (T(2), SlotWindow.MORNING), 0, 0, True, ["Cotton bedsheet set", "Pillow covers"]),
    ("DLV1003", "9990000001", "Anita Rao", OrderStatus.PENDING,
     ("12 Nehru Nagar", "Pune", "Maharashtra", "411014", None, None),
     (T(4), SlotWindow.AFTERNOON), 0, 0, False, ["Yoga mat"]),
    ("DLV1004", "9990000002", "Ravi Kumar", OrderStatus.FAILED_ATTEMPT,
     ("Flat 402, Sunrise Apts", "Hyderabad", "Telangana", "500081", "Kondapur", None),
     (T(1), SlotWindow.MORNING), 1, 0, False, ["Running shoes"]),
    ("DLV1005", "9990000003", "Meera Nair", OrderStatus.DELIVERED,
     ("7 MG Road", "Kochi", "Kerala", "682016", None, None),
     (T(-1), SlotWindow.AFTERNOON), 1, 0, False, ["Espresso cups"]),
    ("DLV1006", "9990000004", "Sunil Deshmukh", OrderStatus.IN_TRANSIT,
     ("H.No 88, Shivaji Peth", "Kolhapur", "Maharashtra", "416012", None, None),
     (T(3), SlotWindow.EVENING), 0, 2, False, ["Table lamp"]),  # at reschedule cap
    ("DLV1007", "9990000005", "Fatima Sheikh", OrderStatus.RTO_INITIATED,
     ("22 Charminar Road", "Hyderabad", "Telangana", "500002", None, None),
     (T(-2), SlotWindow.MORNING), 3, 1, True, ["Wall clock"]),
    ("DLV1008", "9990000006", "Arjun Patel", OrderStatus.PENDING,
     ("Plot 15, GIDC", "Ahmedabad", "Gujarat", "380015", None, None),
     (T(6), SlotWindow.AFTERNOON), 0, 0, False, ["Cricket bat"]),
    ("DLV1009", "9990000007", "Lakshmi Iyer", OrderStatus.OUT_FOR_DELIVERY,
     ("31 Besant Nagar", "Chennai", "Tamil Nadu", "600090", None, "Beach side"),
     (T(0), SlotWindow.AFTERNOON), 0, 1, True, ["Filter coffee kit"]),
    ("DLV1010", "9990000008", "Deepak Sharma", OrderStatus.IN_TRANSIT,
     ("A-9 Rohini Sector 3", "Delhi", "Delhi", "110085", None, None),
     (T(7), SlotWindow.MORNING), 0, 0, False, ["Winter jacket"]),
    # --- four orders saturating T(3)/MORNING so SLOT_FULL is reachable -----
    ("DLV1011", "9990000009", "Priya Menon", OrderStatus.PENDING,
     ("5 Indiranagar", "Bengaluru", "Karnataka", "560038", None, None),
     (T(3), SlotWindow.MORNING), 0, 0, False, ["Desk organiser"]),
    ("DLV1012", "9990000010", "Imran Qureshi", OrderStatus.PENDING,
     ("18 Frazer Town", "Bengaluru", "Karnataka", "560005", None, None),
     (T(3), SlotWindow.MORNING), 0, 0, False, ["Headphones"]),
    ("DLV1013", "9990000011", "Kavya Reddy", OrderStatus.PENDING,
     ("60 Jubilee Hills", "Hyderabad", "Telangana", "500033", None, None),
     (T(3), SlotWindow.MORNING), 0, 0, False, ["Saree"]),
    ("DLV1014", "9990000012", "Rohit Bose", OrderStatus.PENDING,
     ("9 Salt Lake Sec 5", "Kolkata", "West Bengal", "700091", None, None),
     (T(3), SlotWindow.MORNING), 0, 0, False, ["Mixer grinder"]),
    # ----------------------------------------------------------------------
    ("DLV1015", "9990000013", "Sneha Joshi", OrderStatus.IN_TRANSIT,
     ("14 FC Road", "Pune", "Maharashtra", "411005", None, None),
     (T(2), SlotWindow.EVENING), 0, 0, True, ["Backpack", "Water bottle"]),
    ("DLV1016", "9990000014", "Vikram Singh", OrderStatus.FAILED_ATTEMPT,
     ("Tower B, Palm Grove", "Gurugram", "Haryana", "122018", "Sector 56", None),
     (T(1), SlotWindow.EVENING), 2, 0, False, ["Monitor stand"]),
    ("DLV1017", "9990000015", "Ananya Ghosh", OrderStatus.PENDING,
     ("77 Park Street", "Kolkata", "West Bengal", "700016", None, None),
     (T(8), SlotWindow.AFTERNOON), 0, 0, False, ["Notebook set"]),
    ("DLV1018", "9990000016", "Manoj Pillai", OrderStatus.IN_TRANSIT,
     ("3 Marine Drive", "Kochi", "Kerala", "682031", None, None),
     (T(5), SlotWindow.MORNING), 0, 0, False, ["Rice cooker"]),  # blackout date
    ("DLV1019", "9990000017", "Neha Agarwal", OrderStatus.CANCELLED,
     ("21 Civil Lines", "Jaipur", "Rajasthan", "302006", None, None),
     (T(-3), SlotWindow.MORNING), 0, 0, False, ["Curtains"]),
    ("DLV1020", "9990000018", "Suresh Babu", OrderStatus.OUT_FOR_DELIVERY,
     ("45 RS Puram", "Coimbatore", "Tamil Nadu", "641002", None, None),
     (T(0), SlotWindow.MORNING), 0, 0, True, ["Pressure cooker"]),
]


def build_fixtures() -> list[Order]:
    _register_blackouts()
    orders: list[Order] = []
    for (
        oid, phone, name, status, addr, slot, attempts, resched, cod, items
    ) in _SPEC:
        address = _addr(*addr)
        orders.append(
            Order(
                order_id=oid,
                customer_phone=phone,
                customer_name=name,
                status=status,
                address=address,
                promised_slot=DeliverySlot(date=slot[0], window=slot[1]),
                sla_deadline=slot[0] + timedelta(days=2),
                delivery_attempts=attempts,
                reschedule_count=resched,
                is_cod=cod,
                cod_amount=round(499 + hash(oid) % 2500, 2) if cod else None,
                items=items,
                tracking=_track(oid, status, address.city),
            )
        )
    return orders
