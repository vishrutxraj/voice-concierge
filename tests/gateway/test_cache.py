from __future__ import annotations

import pytest
from app.providers.cache import ContentCache, get_cache, reset_caches


@pytest.fixture(autouse=True)
def _reset():
    reset_caches()
    yield
    reset_caches()


def test_hash_key_is_stable():
    assert ContentCache.hash_key("a", "b") == ContentCache.hash_key("a", "b")


def test_hash_key_distinguishes_boundary_placement():
    """('ab','c') and ('a','bc') must not collide -- NUL-separated hashing."""
    assert ContentCache.hash_key("ab", "c") != ContentCache.hash_key("a", "bc")


def test_hash_key_accepts_mixed_str_and_bytes():
    k = ContentCache.hash_key("text", b"\x00\x01\x02", "voice")
    assert isinstance(k, str) and len(k) == 40


def test_miss_then_hit(tmp_path):
    cache = ContentCache(tmp_path, "test-ns")
    key = cache.hash_key("hello")

    assert cache.get(key) is None  # miss

    written = cache.set(key, b"world", {"note": "first write"})
    assert written.cached is False
    assert written.data == b"world"

    hit = cache.get(key)
    assert hit is not None
    assert hit.cached is True
    assert hit.data == b"world"
    assert hit.meta["note"] == "first write"


def test_stats_track_hits_and_misses(tmp_path):
    cache = ContentCache(tmp_path, "stats-ns")
    key = cache.hash_key("x")

    cache.get(key)  # miss
    cache.set(key, b"data")
    cache.get(key)  # hit
    cache.get(key)  # hit

    stats = cache.stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 2
    assert stats["hit_rate"] == pytest.approx(2 / 3, abs=0.01)
    assert stats["entries_on_disk"] == 1


def test_entries_persist_across_cache_instances(tmp_path):
    """A cache is just files on disk -- a fresh instance must see prior writes."""
    ContentCache(tmp_path, "persist-ns").set("k1", b"payload")
    second = ContentCache(tmp_path, "persist-ns")
    hit = second.get("k1")
    assert hit is not None and hit.data == b"payload"


def test_different_namespaces_do_not_collide(tmp_path):
    asr = ContentCache(tmp_path, "asr")
    tts = ContentCache(tmp_path, "tts")
    asr.set("same-key", b"asr-data")
    tts.set("same-key", b"tts-data")
    assert asr.get("same-key").data == b"asr-data"
    assert tts.get("same-key").data == b"tts-data"


def test_get_cache_returns_same_instance_per_namespace(monkeypatch, tmp_path):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    from app.config import get_settings

    get_settings.cache_clear()
    a = get_cache("asr")
    b = get_cache("asr")
    assert a is b
    get_settings.cache_clear()
