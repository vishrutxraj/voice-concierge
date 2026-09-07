"""
Content-hash cache for ASR and TTS.

Why this exists: Sarvam's free-tier signup credit is roughly 20 uncached
end-to-end calls. Development replays the same handful of test phrases
hundreds of times. Caching by content hash means the first render of "Your
order is out for delivery" costs money once, ever, per (text, voice, language,
model) combination — every replay after that is a filesystem read.

Two independent caches (ASR keyed on audio bytes, TTS keyed on text+voice
params) sharing one implementation. Deliberately NOT a generic key-value store
with TTL/eviction machinery — this is a dev-cost control, not a production
cache tier, so the simplest thing that actually saves money is correct.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CacheEntry:
    data: bytes
    meta: dict
    cached: bool  # False on the call that just wrote this entry


class ContentCache:
    def __init__(self, cache_dir: str | Path, namespace: str) -> None:
        self._dir = Path(cache_dir) / namespace
        self._dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    @staticmethod
    def hash_key(*parts: str | bytes) -> str:
        """
        Stable content hash across str/bytes parts. NUL-separated so
        ("ab", "c") and ("a", "bc") never collide.
        """
        h = hashlib.sha256()
        for part in parts:
            h.update(part if isinstance(part, bytes) else part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()[:40]

    def _data_path(self, key: str) -> Path:
        return self._dir / f"{key}.bin"

    def _meta_path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def get(self, key: str) -> CacheEntry | None:
        data_path = self._data_path(key)
        if not data_path.exists():
            self._misses += 1
            return None
        self._hits += 1
        meta_path = self._meta_path(key)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return CacheEntry(data=data_path.read_bytes(), meta=meta, cached=True)

    def set(self, key: str, data: bytes, meta: dict | None = None) -> CacheEntry:
        self._data_path(key).write_bytes(data)
        full_meta = {**(meta or {}), "cached_at": time.time()}
        self._meta_path(key).write_text(json.dumps(full_meta))
        return CacheEntry(data=data, meta=full_meta, cached=False)

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
            "entries_on_disk": sum(1 for _ in self._dir.glob("*.bin")),
        }


_caches: dict[str, ContentCache] = {}


def get_cache(namespace: str) -> ContentCache:
    """One cache instance per namespace per process, so stats() accumulates
    meaningfully across a run instead of resetting on every call site."""
    if namespace not in _caches:
        from app.config import get_settings

        _caches[namespace] = ContentCache(get_settings().cache_dir, namespace)
    return _caches[namespace]


def reset_caches() -> None:
    _caches.clear()
