"""
Storage backend for the order store.

WHY THIS EXISTS
---------------
Vercel functions are stateless. Each invocation may hit a cold container, so an
in-memory dict loses every write between requests: the caller reschedules, hangs
up, calls back, and the old slot is still there. That is worse than not shipping
the feature.

Two backends, chosen by env:

  MemoryBackend   — local dev, CI, and tests. Zero setup, zero network.
  UpstashBackend  — Vercel. Serverless Redis over plain HTTPS, so there is no
                    connection pool to leak across invocations (the usual way
                    Redis and serverless go wrong). Free tier is ample here.

Deliberately not a Redis client library: Upstash's REST API is two HTTP calls,
and adding `redis-py` to a Vercel bundle costs cold-start time for nothing.
"""

from __future__ import annotations

import json
import os
import threading
from abc import ABC, abstractmethod
from typing import Any


class KVBackend(ABC):
    @abstractmethod
    def get(self, key: str) -> Any | None: ...

    @abstractmethod
    def set(self, key: str, value: Any) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def keys(self, prefix: str) -> list[str]: ...

    def get_many(self, keys: list[str]) -> dict[str, Any]:
        return {k: v for k in keys if (v := self.get(k)) is not None}


class MemoryBackend(KVBackend):
    """Process-local. Correct for a single long-lived container (Railway, local)."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def keys(self, prefix: str) -> list[str]:
        with self._lock:
            return [k for k in self._data if k.startswith(prefix)]

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class UpstashBackend(KVBackend):
    """
    Upstash Redis via REST. Stateless — safe across serverless invocations.

    Values are JSON-encoded, so callers get dicts back and the store stays
    agnostic about serialization.
    """

    def __init__(self, url: str, token: str, timeout: float = 5.0) -> None:
        self._url = url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"}
        self._timeout = timeout

    def _cmd(self, *args: str) -> Any:
        import httpx

        resp = httpx.post(
            self._url,
            headers=self._headers,
            json=list(args),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json().get("result")

    def get(self, key: str) -> Any | None:
        raw = self._cmd("GET", key)
        return json.loads(raw) if raw else None

    def set(self, key: str, value: Any) -> None:
        self._cmd("SET", key, json.dumps(value, default=str))

    def delete(self, key: str) -> None:
        self._cmd("DEL", key)

    def keys(self, prefix: str) -> list[str]:
        return self._cmd("KEYS", f"{prefix}*") or []


_backend: KVBackend | None = None


def get_backend() -> KVBackend:
    """
    Resolve once per process.

    Absence of Upstash credentials is not an error — it means local mode. The
    order API must run with no external service at all.
    """
    global _backend
    if _backend is not None:
        return _backend

    url = os.getenv("UPSTASH_REDIS_REST_URL")
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
    _backend = UpstashBackend(url, token) if url and token else MemoryBackend()
    return _backend


def reset_backend() -> None:
    """Test hook."""
    global _backend
    _backend = None
