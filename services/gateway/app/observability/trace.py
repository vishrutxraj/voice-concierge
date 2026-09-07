"""
Observability, traceability, explainability.

Every node emits a TraceEvent. Together they answer the question the
Responsible AI brief actually asks: *why did the agent do that?*

  GET /api/sessions/{id}/trace  ->  ordered decision chain with, for each step,
  the inputs considered, the alternatives rejected, and the confidence held.

Two invariants:
  1. Everything written here has passed through pii.scrub_mapping first.
  2. It works with zero external dependencies. Langfuse is an optional mirror,
     never a requirement — the service must be fully traceable offline.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from app.config import get_settings
from app.observability.pii import scrub_mapping

logger = logging.getLogger("concierge.trace")


class EventKind(str, Enum):
    NODE_ENTER = "node_enter"
    NODE_EXIT = "node_exit"
    DECISION = "decision"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    LLM_CALL = "llm_call"
    GUARDRAIL = "guardrail"
    PII = "pii"
    ESCALATION = "escalation"
    CONSENT = "consent"
    ERROR = "error"


@dataclass
class TraceEvent:
    session_id: str
    kind: EventKind
    node: str
    message: str
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: float = field(default_factory=time.time)
    # Explainability payload
    confidence: float | None = None
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["kind"] = self.kind.value
        return raw


class TraceSink:
    """
    Ring-buffered in-memory store + JSON-lines stdout, with optional Langfuse.

    Retention is bounded both by count and by config.trace_retention_hours so
    an always-on Space cannot accumulate call records indefinitely — data
    minimisation, not just tidiness.
    """

    def __init__(self, max_sessions: int = 500, max_events_per_session: int = 400):
        self._events: dict[str, deque[TraceEvent]] = defaultdict(
            lambda: deque(maxlen=max_events_per_session)
        )
        self._order: deque[str] = deque(maxlen=max_sessions)
        self._langfuse = self._init_langfuse()

    @staticmethod
    def _init_langfuse():
        s = get_settings()
        if not (s.langfuse_public_key and s.langfuse_secret_key):
            return None
        try:
            from langfuse import Langfuse

            return Langfuse(
                public_key=s.langfuse_public_key,
                secret_key=s.langfuse_secret_key,
                host=s.langfuse_host,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Langfuse unavailable, using local tracing only: %s", exc)
            return None

    def emit(self, event: TraceEvent) -> TraceEvent:
        event.data = scrub_mapping(event.data)
        if event.reasoning:
            from app.observability.pii import redact

            event.reasoning = redact(event.reasoning).text

        if event.session_id not in self._events:
            self._order.append(event.session_id)
        self._events[event.session_id].append(event)

        print(json.dumps(event.to_dict(), default=str), file=sys.stdout, flush=True)

        if self._langfuse:
            try:
                self._langfuse.event(
                    trace_id=event.session_id,
                    name=f"{event.node}.{event.kind.value}",
                    metadata=event.to_dict(),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Langfuse emit failed: %s", exc)
        return event

    def session(self, session_id: str) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._events.get(session_id, [])]

    def explain(self, session_id: str) -> dict[str, Any]:
        """
        Human-readable decision chain — the explainability endpoint's payload.
        """
        events = self._events.get(session_id, [])
        decisions = [
            {
                "at": e.ts,
                "node": e.node,
                "chose": e.message,
                "confidence": e.confidence,
                "because": e.reasoning,
                "rejected": e.alternatives,
            }
            for e in events
            if e.kind in (EventKind.DECISION, EventKind.ESCALATION)
        ]
        tools = [
            {"node": e.node, "call": e.message, "data": e.data}
            for e in events
            if e.kind is EventKind.TOOL_CALL
        ]
        guardrails = [
            {"node": e.node, "verdict": e.message, "data": e.data}
            for e in events
            if e.kind is EventKind.GUARDRAIL
        ]
        return {
            "session_id": session_id,
            "event_count": len(events),
            "decisions": decisions,
            "tool_calls": tools,
            "guardrail_checks": guardrails,
            "escalated": any(e.kind is EventKind.ESCALATION for e in events),
        }

    def prune(self) -> int:
        cutoff = time.time() - get_settings().trace_retention_hours * 3600
        removed = 0
        for sid in list(self._events):
            kept = deque(
                (e for e in self._events[sid] if e.ts >= cutoff),
                maxlen=self._events[sid].maxlen,
            )
            removed += len(self._events[sid]) - len(kept)
            if kept:
                self._events[sid] = kept
            else:
                del self._events[sid]
        return removed

    def forget(self, session_id: str) -> bool:
        """Right-to-erasure hook. Wired to DELETE /api/sessions/{id}."""
        return self._events.pop(session_id, None) is not None


_sink: TraceSink | None = None


def get_sink() -> TraceSink:
    global _sink
    if _sink is None:
        _sink = TraceSink()
    return _sink


def trace(
    session_id: str,
    kind: EventKind,
    node: str,
    message: str,
    **kwargs: Any,
) -> TraceEvent:
    return get_sink().emit(
        TraceEvent(
            session_id=session_id, kind=kind, node=node, message=message, **kwargs
        )
    )


def configure_logging() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format='{"ts":"%(asctime)s","level":"%(levelname)s",'
        '"logger":"%(name)s","msg":"%(message)s"}',
        stream=sys.stdout,
    )
