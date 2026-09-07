"""
Single source of truth for every model ID, endpoint, and tunable.

Rule: no model name string appears anywhere else in the codebase. Providers
rotate their rosters without warning (Groq deprecated llama-3.1-8b-instant in
June 2026); when that happens you change one line here, not fifteen call sites.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Env(str, Enum):
    DEV = "dev"
    DEMO = "demo"
    TEST = "test"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    env: Env = Env.DEV
    log_level: str = "INFO"

    # ---- Credentials (all optional; absence degrades to offline mode) ----
    sarvam_api_key: str | None = None
    groq_api_key: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # ---- Model roster -------------------------------------------------
    # ASR — verified against docs.sarvam.ai (Aug 2026). saaras:v2 is gone from
    # current docs; saaras:v3 is the default REST model, with mode="translate"
    # replacing the old (now-legacy) /speech-to-text-translate endpoint.
    sarvam_asr_model: str = "saaras:v3"
    sarvam_asr_mode: str = "translate"  # -> English regardless of input language
    groq_asr_model: str = "whisper-large-v3-turbo"
    # LLM — verified live against a real account's GET /v1/models (2026-09),
    # not docs: llama-3.3-70b-versatile and llama-3.1-8b-instant (this
    # project's previous defaults) are BOTH gone from the actual roster now,
    # despite still showing in some cached doc pages -- account-scoped
    # /models is ground truth, a docs page can be stale. Confirmed working
    # with this project's exact router.py prompt + json_mode=True (no
    # <think>-token leakage into the JSON, which a plain unconstrained call
    # to these newer reasoning-style models WILL produce -- response_format:
    # json_object is what keeps output clean, already set when json_mode is
    # requested, see llm_client.py).
    groq_llm_model: str = "openai/gpt-oss-20b"
    # Was llama-3.1-8b-instant, also gone; kept distinct from groq_llm_model
    # for whenever something actually reads it, but note it's not wired to
    # any call site today (only groq_llm_model is) -- see CLAUDE.md.
    groq_llm_fast_model: str = "openai/gpt-oss-20b"
    # TTS. bulbul:v2 (originally used here for dev, at half v3's per-character
    # rate) is now REMOVED, not just legacy -- confirmed live: Sarvam rejects
    # it outright with "Model 'bulbul:v2' has been deprecated. Please use
    # 'bulbul:v3' instead." (a 400, not a warning), first caught by an actual
    # WebSocket call against the deployed gateway, not by re-reading docs.
    # Both dev and demo use v3 now; the two settings stay separate in case a
    # cheaper dev-tier model shows up again later.
    sarvam_tts_model_dev: str = "bulbul:v3"
    sarvam_tts_model_demo: str = "bulbul:v3"
    # Default speaker differs per model version — verified per-speaker defaults,
    # not guessed. Speaker names are case-sensitive and must be lowercase.
    # "anushka" was bulbul:v2's default and is NOT a valid bulbul:v3 speaker
    # (confirmed live: Sarvam returns the full valid-speaker list on a 400).
    sarvam_tts_speaker_dev: str = "shubh"  # bulbul:v3 default
    sarvam_tts_speaker_demo: str = "shubh"  # bulbul:v3 default

    # ---- Provider selection -------------------------------------------
    asr_provider: str = "sarvam"  # sarvam | groq | mock
    tts_provider: str = "sarvam"  # sarvam | piper | mock
    llm_provider: str = "groq"  # groq | mock

    # ---- Endpoints -----------------------------------------------------
    # Order API lives on Vercel. Empty means in-process (local dev / CI / tests)
    # so the whole suite runs with no network and no deployed service.
    order_api_base_url: str = ""
    sarvam_base_url: str = "https://api.sarvam.ai"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # ---- Persistence ---------------------------------------------------
    # SQLite by default so the whole thing runs with no database service.
    # Set CHECKPOINT_DSN to a postgres:// URL to switch; nothing else changes.
    checkpoint_dsn: str = "sqlite:///./data/checkpoints.db"
    cache_dir: str = "./data/cache"

    # ---- Responsible AI toggles ---------------------------------------
    pii_redaction_enabled: bool = True
    pii_redact_before_llm: bool = True
    guardrails_enabled: bool = True
    consent_required_for_persistence: bool = True
    trace_retention_hours: int = 24

    # ---- Human-in-the-loop thresholds ---------------------------------
    # Router confidence below this escalates rather than guessing.
    min_intent_confidence: float = 0.55
    # Sentiment runs -1.0 (furious) .. +1.0 (delighted).
    sentiment_escalation_threshold: float = -0.45
    max_unresolved_turns: int = 4
    escalation_webhook_url: str | None = None

    # ---- Order-ID resolution ------------------------------------------
    # Never trust raw ASR for an alphanumeric ID. Fuzzy-match against the
    # caller's own open orders instead; this is the accuracy threshold.
    order_id_fuzzy_threshold: float = 0.72

    # ---- Phase 4: WebSocket audio transport ----------------------------
    # Outbound TTS audio is streamed in chunks of this many bytes rather than
    # sent as one blob -- the granularity at which /ws/call can respond to
    # barge-in (see app/audio/call_loop.py). ~100ms of 16kHz mono PCM16.
    ws_audio_chunk_bytes: int = 3200

    @property
    def sarvam_tts_model(self) -> str:
        return (
            self.sarvam_tts_model_demo
            if self.env is Env.DEMO
            else self.sarvam_tts_model_dev
        )

    @property
    def sarvam_tts_speaker(self) -> str:
        return (
            self.sarvam_tts_speaker_demo
            if self.env is Env.DEMO
            else self.sarvam_tts_speaker_dev
        )

    @property
    def offline(self) -> bool:
        """True when no external credentials exist — everything falls back to mocks."""
        return not (self.sarvam_api_key or self.groq_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()


# --------------------------------------------------------------------------
# Startup validation
# --------------------------------------------------------------------------


class RosterValidationError(RuntimeError):
    pass


def validate_roster(settings: Settings | None = None) -> dict[str, list[str]]:
    """
    Verify configured model IDs still exist on the provider.

    Returns a report rather than raising for missing credentials — the service
    must boot in offline/mock mode. It DOES raise when credentials are present
    but a configured model has been retired, because failing at startup beats
    failing mid-call.
    """
    settings = settings or get_settings()
    report: dict[str, list[str]] = {"ok": [], "skipped": [], "missing": []}

    if not settings.groq_api_key:
        report["skipped"].append("groq (no api key — mock mode)")
    else:
        try:
            import httpx

            resp = httpx.get(
                f"{settings.groq_base_url}/models",
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                timeout=10.0,
            )
            resp.raise_for_status()
            available = {m["id"] for m in resp.json().get("data", [])}
            for name in (settings.groq_llm_model, settings.groq_asr_model):
                (report["ok"] if name in available else report["missing"]).append(name)
        except Exception as exc:  # noqa: BLE001
            report["skipped"].append(f"groq (roster check failed: {exc})")

    if not settings.sarvam_api_key:
        report["skipped"].append("sarvam (no api key — mock mode)")
    else:
        # Sarvam has no public /models listing; model IDs are validated by
        # first use. Recorded as ok so the report shape stays consistent.
        report["ok"].extend([settings.sarvam_asr_model, settings.sarvam_tts_model])

    if report["missing"]:
        raise RosterValidationError(
            "Configured models no longer available: "
            + ", ".join(report["missing"])
            + ". Update app/config.py."
        )
    return report


def ensure_dirs(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    os.makedirs(settings.cache_dir, exist_ok=True)
    if settings.checkpoint_dsn.startswith("sqlite"):
        path = settings.checkpoint_dsn.split("///")[-1]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
