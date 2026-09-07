"""
Regex-based date/window extraction -- the fast, free, deterministic path.

Split into its own module (rather than living in nodes/reschedule.py, where it
originated in phase 2) specifically so app/graph/extraction.py can depend on
it without creating a cycle: extraction.py orchestrates "try regex, then fall
back to the LLM", and nodes/reschedule.py calls extraction.py. If these
functions still lived in nodes/reschedule.py, that node would import
extraction.py which imports nodes/reschedule.py -- a cycle. This module has no
node-specific dependencies, so both sides can import it cleanly.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from order_contracts.schemas import SlotWindow

_WINDOW_WORDS = {
    "morning": SlotWindow.MORNING,
    "afternoon": SlotWindow.AFTERNOON,
    "evening": SlotWindow.EVENING,
    "night": SlotWindow.EVENING,
}

_WEEKDAY_INDEX = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def extract_window_regex(text: str) -> SlotWindow | None:
    text = text.lower()
    for word, window in _WINDOW_WORDS.items():
        if word in text:
            return window
    return None


def extract_date_regex(text: str, today: date | None = None) -> date | None:
    today = today or date.today()
    text = text.lower()

    # Longer, more specific phrases must be checked before their substrings --
    # "day after tomorrow" contains "tomorrow", and "next monday" contains
    # "monday". Checking the specific phrase first is what makes both resolve
    # correctly instead of silently matching the looser, wrong pattern.
    if "day after tomorrow" in text:
        return today + timedelta(days=2)
    if "today" in text:
        return today
    if "tomorrow" in text:
        return today + timedelta(days=1)

    next_weekday = re.search(
        r"\bnext (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", text
    )
    if next_weekday:
        idx = _WEEKDAY_INDEX[next_weekday.group(1)]
        days_ahead = (idx - today.weekday()) % 7 or 7
        return today + timedelta(days=days_ahead + 7)  # the Monday of NEXT week

    for name, idx in _WEEKDAY_INDEX.items():
        if name in text:
            days_ahead = (idx - today.weekday()) % 7
            days_ahead = days_ahead or 7  # "monday" said on a Monday means next one
            return today + timedelta(days=days_ahead)

    m = re.search(r"in (\d+) days?", text)
    if m:
        return today + timedelta(days=int(m.group(1)))

    return None
