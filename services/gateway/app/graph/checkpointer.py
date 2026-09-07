"""
LangGraph checkpointing.

SQLite by default — no database service, which is what keeps the gateway inside
Railway's $5 Hobby tier and HF Spaces' free tier. One env var (CHECKPOINT_DSN)
swaps to Postgres with no code change, for whenever call volume justifies it.

The guard here matters more than the backend choice: CallState.pii_vault holds
tokenized-but-reversible customer PII (see app/observability/pii.py). If it ever
reached the checkpoint file, every reschedule and address correction would be
sitting in plaintext-adjacent form on disk. serialize_guard strips it before
LangGraph's checkpointer touches the state and raises loudly if that ever stops
happening — a silent no-op here is a privacy incident, not a bug to shrug at.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import get_settings


class VaultLeakError(RuntimeError):
    """Raised if a PIIVault is ever about to be written to a checkpoint."""


def strip_vault(state: dict) -> dict:
    """Return a checkpoint-safe copy of state with pii_vault removed."""
    if "pii_vault" not in state:
        return state
    clean = dict(state)
    clean.pop("pii_vault")
    return clean


def assert_no_vault_leak(state: dict) -> None:
    if "pii_vault" in state and state["pii_vault"] is not None:
        raise VaultLeakError(
            "CallState.pii_vault reached the checkpoint boundary. "
            "This must be stripped before persistence — see strip_vault()."
        )


@contextmanager
def get_checkpointer() -> Iterator[SqliteSaver]:
    """
    Context-managed so the sqlite3 connection closes cleanly — important on
    Windows, where a lingering handle blocks the next `--reload` from touching
    the same file.
    """
    settings = get_settings()
    dsn = settings.checkpoint_dsn

    if dsn.startswith("postgres"):
        from langgraph.checkpoint.postgres import PostgresSaver

        with PostgresSaver.from_conn_string(dsn) as saver:
            saver.setup()
            yield saver
        return

    path = dsn.split("///")[-1] if "///" in dsn else dsn
    conn = sqlite3.connect(path, check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        saver.setup()
        yield saver
    finally:
        conn.close()


def thread_config(session_id: str) -> dict:
    """The `configurable.thread_id` LangGraph needs to resume a call's state."""
    return {"configurable": {"thread_id": session_id}}
