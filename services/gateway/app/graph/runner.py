"""
Turn execution.

One call to run_turn per conversational turn. Handles three cases:
  1. Fresh turn, no interrupt pending      -> graph.invoke(new input)
  2. Turn arrives while an interrupt IS
     pending from the previous turn        -> graph.invoke(Command(resume=...))
  3. First turn of a brand-new session     -> seeds a fresh PIIVault

The checkpointer connection is opened and closed within this function rather
than held open for the process lifetime -- correct for both a Vercel-style
short-lived process and a long-lived Railway container, and avoids the Windows
"file still in use" problem when --reload restarts the process.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from langgraph.types import Command

from app.graph.builder import build_graph
from app.graph.checkpointer import get_checkpointer, thread_config
from app.graph.state import CallState
from app.observability.pii import PIIVault, tokenize
from app.observability.trace import EventKind, trace

_session_vaults: dict[str, PIIVault] = {}


def _vault_for(session_id: str) -> PIIVault:
    """
    One vault per session, kept in process memory only -- never in CallState
    as persisted by the checkpointer (see checkpointer.py's assert_no_vault_leak
    for the enforcement side of this).
    """
    if session_id not in _session_vaults:
        _session_vaults[session_id] = PIIVault()
    return _session_vaults[session_id]


def forget_vault(session_id: str) -> None:
    _session_vaults.pop(session_id, None)


@dataclass
class TurnResult:
    reply_text: str
    awaiting_confirmation: dict | None
    state: dict


def run_turn(
    session_id: str,
    caller_utterance: str,
    caller_phone: str = "",
    language: str = "en",
    transport_kind: str = "text",
    resume_value: bool | None = None,
) -> TurnResult:
    """
    Execute one turn.

    resume_value is provided when the previous turn left the graph paused at
    interrupt() (a reschedule/address confirmation) -- pass the caller's yes/no
    for that specific pause rather than a fresh utterance.
    """
    vault = _vault_for(session_id)
    tokenized = tokenize(caller_utterance, vault)

    trace(
        session_id, EventKind.NODE_ENTER, "turn_runner", "turn_start",
        data={"transport": transport_kind, "language": language},
    )

    with get_checkpointer() as checkpointer:
        graph = build_graph().compile(checkpointer=checkpointer)
        config = thread_config(session_id)

        if resume_value is not None:
            # LangGraph's Command(resume=...) checks the payload for
            # truthiness on entry; Command(resume=False) is indistinguishable
            # from "no resume value provided" and raises EmptyInputError. Wrap
            # the boolean so a caller declining a confirmation ("no") doesn't
            # crash the turn -- interrupt() on the node side still receives
            # the plain bool via the wrapper's unwrapping.
            invoke_arg = Command(resume={"confirmed": resume_value})
        else:
            existing = graph.get_state(config)
            is_new = not existing.values
            invoke_input: CallState = {
                "session_id": session_id,
                "caller_phone": caller_phone,
                "language": language,
                "transport_kind": transport_kind,
                "last_utterance": tokenized.text,
                "turns": [{"role": "caller", "text": tokenized.text, "ts": time.time()}],
                "pii_vault": vault,
            }
            if is_new:
                invoke_input.setdefault("sentiment_history", [])
                invoke_input.setdefault("unresolved_turns", 0)
                invoke_input.setdefault("escalated", False)
                invoke_input.setdefault("order_id", None)
            invoke_arg = invoke_input

        result_state = graph.invoke(invoke_arg, config)

        state_snapshot = graph.get_state(config)
        pending_interrupt = None
        for task in state_snapshot.tasks:
            if task.interrupts:
                pending_interrupt = task.interrupts[0].value
                break

    reply = result_state.get("reply_text", "")
    trace(
        session_id, EventKind.NODE_EXIT, "turn_runner", "turn_end",
        data={"paused": pending_interrupt is not None, "reply_length": len(reply)},
    )

    return TurnResult(
        reply_text=reply,
        awaiting_confirmation=pending_interrupt,
        state=dict(result_state),
    )
