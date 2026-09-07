"""
Monorepo path wiring for pytest.

Both services and the shared contracts package are importable without an install
step, so `pytest` at the repo root just works — no editable installs, no
PYTHONPATH juggling in CI.
"""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).parent
for path in (
    ROOT / "packages" / "order-contracts",
    ROOT / "services" / "order-api",
    ROOT / "services" / "gateway",
):
    sys.path.insert(0, str(path))


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch):
    """
    The suite must behave identically regardless of what's in a developer's
    real .env -- CLAUDE.md's hard rule ("if a test needs a live API key, the
    test is wrong") only holds if a credential sitting in .env for actual
    development use (e.g. running scripts/eval_asr.py for real) can never
    silently leak into a test that never asked for it. Settings.env_file
    means pydantic-settings falls back to reading .env whenever the process
    environment doesn't already have a var; setting these to "" here (a real
    env var override, not a deletion) beats that fallback, so get_x_client()
    singletons reliably pick mocks in every test no matter what's on disk.

    Found the hard way: adding a real SARVAM_API_KEY to .env (to run the
    real ASR eval) made every test that goes through get_asr_client() /
    get_tts_client() start making live, slow, unmocked network calls with
    fake test "audio" bytes -- tests that had run in milliseconds started
    timing out. Tests that DO want real-transport behavior construct their
    own client with an explicit mock transport instead (see
    test_asr_tts_clients.py) rather than relying on ambient credentials.
    """
    for var in ("SARVAM_API_KEY", "GROQ_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.setenv(var, "")

    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
