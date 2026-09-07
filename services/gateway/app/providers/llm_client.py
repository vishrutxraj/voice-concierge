"""
LLM client.

Mirrors the OrderClient pattern: one interface, a real implementation and a
mock, selected by whether credentials exist. The mock is not a stub that always
returns the same thing — it does real keyword-based intent classification, so
phase 2 (router + agents + interrupt logic) is fully testable and demoable with
zero Groq spend. Swapping in the real model changes accuracy, not behaviour.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta

from app.config import get_settings

# Local copy, not imported from app.graph.regex_extractors: a provider module
# staying independent of graph internals is deliberate -- providers should be
# reusable outside this specific graph shape.
_WEEKDAY_INDEX = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


@dataclass
class LLMResponse:
    text: str
    raw: dict | None = None


class LLMClient(ABC):
    @abstractmethod
    def complete(
        self, system: str, user: str, *, json_mode: bool = False
    ) -> LLMResponse: ...


class GroqLLMClient(LLMClient):
    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        import httpx

        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=8.0,
        )
        self._model = model

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> LLMResponse:
        import httpx

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMUnavailable(str(exc)) from exc
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return LLMResponse(text=text, raw=data)


class LLMUnavailable(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Deterministic mock — keyword-scored, not a fixed string. Good enough for the
# router to make real (if less accurate) decisions with zero external calls.
# --------------------------------------------------------------------------

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "order_lookup": [
        "where", "status", "track", "delivered", "arrive", "eta",
        "when is", "when will",
    ],
    "reschedule": [
        "reschedule", "postpone", "delay", "change the date", "different day",
        "different time", "move my delivery", "not home", "another day",
    ],
    "address_change": [
        "wrong address", "change address", "change my address",
        "change my delivery address", "different address", "move to",
        "update address", "new address", "correct address", "pincode",
        "delivery address",
    ],
}


class MockLLMClient(LLMClient):
    """
    Keyword-scored intent classification plus templated agent replies.

    Deliberately returns REAL confidence scores (not always 0.99) so the
    router's escalation-on-low-confidence path is exercisable without Groq.
    """

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> LLMResponse:
        system_lower = system.lower()
        if "classify" in system_lower:
            return self._classify(user)
        if "time-of-day window" in system_lower:
            return self._extract_slot(system, user)
        if "delivery address" in system_lower and "extract" in system_lower:
            return self._extract_address(user)
        return LLMResponse(text=self._template_reply(system, user))

    def _classify(self, utterance: str) -> LLMResponse:
        text = utterance.lower()
        scores: dict[str, float] = {}
        for intent, keywords in _INTENT_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in text)
            if hits:
                # Diminishing returns per extra keyword hit, capped well below 1.0
                # so a single ambiguous word can't masquerade as high confidence.
                scores[intent] = min(0.55 + 0.15 * hits, 0.93)

        if not scores:
            payload = {
                "intent": "unclear",
                "confidence": 0.2,
                "reasoning": "no matching keywords for any known intent",
                "alternatives": [],
            }
            return LLMResponse(text=json.dumps(payload), raw=payload)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_intent, best_score = ranked[0]
        payload = {
            "intent": best_intent,
            "confidence": best_score,
            "reasoning": f"keyword match for '{best_intent}'",
            "alternatives": [
                {"intent": i, "score": s} for i, s in ranked[1:3]
            ],
        }
        return LLMResponse(text=json.dumps(payload), raw=payload)

    def _extract_slot(self, system: str, utterance: str) -> LLMResponse:
        """
        Handles phrasing the regex fast-path (regex_extractors.py) can't:
        "day after tomorrow", "next <weekday>" (a week later than bare
        "<weekday>"), and "the Nth" (day-of-month). Real, not a fixed stub --
        this is what lets test_extraction.py exercise the LLM-fallback branch
        without needing Groq.
        """
        m = re.search(r"today's date is (\d{4}-\d{2}-\d{2})", system.lower())
        today = date.fromisoformat(m.group(1)) if m else date.today()
        text = utterance.lower()

        found_date: date | None = None
        if "day after tomorrow" in text:
            found_date = today + timedelta(days=2)
        else:
            next_weekday = re.search(
                r"next (monday|tuesday|wednesday|thursday|friday|saturday|sunday)", text
            )
            if next_weekday:
                idx = _WEEKDAY_INDEX[next_weekday.group(1)]
                days_ahead = (idx - today.weekday()) % 7 or 7
                found_date = today + timedelta(days=days_ahead + 7)
            else:
                ordinal = re.search(r"\bthe (\d{1,2})(?:st|nd|rd|th)\b", text)
                if ordinal:
                    day_of_month = int(ordinal.group(1))
                    candidate = today.replace(day=1)
                    for _ in range(13):  # walk forward until day-of-month matches
                        try:
                            candidate = candidate.replace(day=day_of_month)
                            if candidate >= today:
                                found_date = candidate
                                break
                        except ValueError:
                            pass
                        candidate = (candidate.replace(day=28) + timedelta(days=4)).replace(day=1)

        found_window = None
        for word in ("morning", "afternoon", "evening", "night"):
            if word in text:
                found_window = "evening" if word == "night" else word
                break

        payload = {
            "date": found_date.isoformat() if found_date else None,
            "window": found_window,
        }
        return LLMResponse(text=json.dumps(payload), raw=payload)

    _KNOWN_CITIES = {
        "mumbai": "Maharashtra", "delhi": "Delhi", "bengaluru": "Karnataka",
        "bangalore": "Karnataka", "hyderabad": "Telangana", "chennai": "Tamil Nadu",
        "kolkata": "West Bengal", "pune": "Maharashtra", "ahmedabad": "Gujarat",
        "jaipur": "Rajasthan", "kochi": "Kerala", "coimbatore": "Tamil Nadu",
        "gurugram": "Haryana", "gurgaon": "Haryana", "kolhapur": "Maharashtra",
    }

    def _extract_address(self, utterance: str) -> LLMResponse:
        """
        Modest heuristic: pull a 6-digit pincode and a recognised city name out
        of free text, treat everything before the city as line1, and supply
        the matching state -- a real LLM would know India's states without
        being told; this small lookup lets the mock demonstrate both the
        success path and the safe-escalation path (extraction.py refuses to
        guess a state for an unrecognised city change) without needing Groq.
        """
        pincode_m = re.search(r"\b[1-9]\d{5}\b", utterance)
        city_key = next((c for c in self._KNOWN_CITIES if c in utterance.lower()), None)

        line1 = None
        if city_key:
            idx = utterance.lower().find(city_key)
            candidate = utterance[:idx].strip(" ,")
            candidate = re.sub(
                r"^(my |the |new |full |address |is |to )+", "", candidate, flags=re.I
            ).strip(" ,")
            line1 = candidate or None

        payload = {
            "line1": line1,
            "line2": None,
            "city": city_key.title() if city_key else None,
            "state": self._KNOWN_CITIES.get(city_key) if city_key else None,
            "pincode": pincode_m.group() if pincode_m else None,
            "landmark": None,
        }
        return LLMResponse(text=json.dumps(payload), raw=payload)

    def _template_reply(self, system: str, user: str) -> str:
        return "Understood — one moment while I take care of that."


_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is not None:
        return _client
    settings = get_settings()
    if settings.llm_provider == "groq" and settings.groq_api_key:
        _client = GroqLLMClient(
            settings.groq_api_key, settings.groq_llm_model, settings.groq_base_url
        )
    else:
        _client = MockLLMClient()
    return _client


def reset_llm_client() -> None:
    global _client
    _client = None
