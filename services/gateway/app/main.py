"""
Service entrypoint.

Mounts these on one ASGI app:
  * /ws/call          voice gateway (phase 4)
  * /api/sessions/*   observability, explainability, and data-subject rights
  * /app              React web UI (services/web), when built
  * /ui               Gradio test harness (phase 6) -- the fallback UI

The order API is NOT mounted here — it runs on Vercel as its own serverless
service (services/order-api). The gateway reaches it through OrderClient, which
falls back to in-process when ORDER_API_BASE_URL is unset, so local dev and CI
need no deployment and no network.

Deployed to Railway: persistent WebSockets and no execution ceiling, neither of
which a serverless function can offer a live phone call.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import gradio as gr
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.audio.call_loop import run_call
from app.audio.localize import localize_reply
from app.audio.transport import AUDIO_PREFERENCES
from app.audio.websocket_transport import WebSocketAudioTransport
from app.config import RosterValidationError, ensure_dirs, get_settings, validate_roster
from app.graph.runner import forget_vault, run_turn
from app.observability.trace import configure_logging, get_sink
from app.ui.blocks import build_demo

logger = logging.getLogger("concierge")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    ensure_dirs(settings)

    try:
        report = validate_roster(settings)
        logger.info("Model roster validated: %s", report)
    except RosterValidationError as exc:
        # Fail loudly at boot, not silently mid-call.
        logger.error("Roster validation failed: %s", exc)
        raise

    if settings.offline:
        logger.warning(
            "No provider credentials found — running in OFFLINE/MOCK mode. "
            "Set SARVAM_API_KEY and GROQ_API_KEY for live speech."
        )
    yield
    get_sink().prune()


app = FastAPI(
    title="Logistics & Delivery Voice Concierge",
    version="0.1.0",
    description=(
        "Multilingual voice agent for delivery support. "
        "Order lookup, rescheduling, and address correction, with "
        "sentiment-triggered human escalation."
    ),
    lifespan=lifespan,
)

class TurnRequest(BaseModel):
    session_id: str
    text: str
    caller_phone: str = ""
    language: str = "en"
    resume: bool | None = None  # set when replying to a pending confirmation


class TurnResponse(BaseModel):
    session_id: str
    reply_text: str  # English, always
    # Same reply in the request's `language`; null for English or when
    # translation isn't possible (see app/audio/localize.py's fallbacks).
    reply_text_localized: str | None = None
    reply_language: str | None = None
    awaiting_confirmation: dict | None = None
    intent: str | None = None
    escalated: bool = False


@app.post("/call/turn", tags=["call"])
def call_turn(payload: TurnRequest) -> TurnResponse:
    """
    Text-only turn endpoint -- phase 2's entire surface. No audio yet: this is
    where the router, agents, and interrupt()-confirmation flow are exercised
    for free, before ASR/TTS are wired in during phase 4.
    """
    result = run_turn(
        session_id=payload.session_id,
        caller_utterance=payload.text,
        caller_phone=payload.caller_phone,
        language=payload.language,
        transport_kind="text",
        resume_value=payload.resume,
    )
    router = result.state.get("router") or {}
    # A turn paused at interrupt() has no reply_text; its read-back prompt is
    # what the caller needs to see (same fallback call_loop and the harness use).
    english = result.reply_text or (result.awaiting_confirmation or {}).get("prompt", "")
    localized = localize_reply(english, payload.language, payload.session_id) if english else None
    return TurnResponse(
        session_id=payload.session_id,
        reply_text=result.reply_text,
        reply_text_localized=localized.text if localized and localized.translated else None,
        reply_language=localized.language if localized and localized.translated else None,
        awaiting_confirmation=result.awaiting_confirmation,
        intent=router.get("intent"),
        escalated=result.state.get("escalated", False),
    )


@app.websocket("/ws/call")
async def ws_call(websocket: WebSocket) -> None:
    """
    Audio gateway -- phase 4. Text-only /call/turn above still works exactly
    as it did in phase 2/3; this is a new, parallel entrypoint that wraps the
    same run_turn() in ASR/TTS via app.audio.call_loop, per CLAUDE.md's phase
    4 note that the graph itself never changes.

    Handshake: the first message must be a JSON text frame
      {"type": "start", "session_id"?, "caller_phone"?, "transport_kind"?,
       "audio"?: "native" | "english" | "both"}
    session_id is generated when omitted -- a first-time browser caller has
    no session yet. Everything after that is the call_loop wire protocol
    documented in app/audio/websocket_transport.py.
    """
    await websocket.accept()
    try:
        start = await websocket.receive_json()
    except (WebSocketDisconnect, ValueError):
        return

    if start.get("type") != "start":
        await websocket.close(code=1008)  # policy violation: bad handshake
        return

    session_id = start.get("session_id") or uuid.uuid4().hex
    transport = WebSocketAudioTransport(websocket)
    if start.get("audio") in AUDIO_PREFERENCES:
        transport.audio_preference = start["audio"]
    await transport.send_event({"type": "ready", "session_id": session_id})

    await run_call(
        transport,
        session_id=session_id,
        caller_phone=start.get("caller_phone", ""),
        transport_kind=start.get("transport_kind", "browser"),
    )


@app.get("/health", tags=["ops"])
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "env": s.env.value,
        "offline_mode": s.offline,
        "asr": s.asr_provider,
        "tts": s.tts_provider,
        "llm": s.llm_provider,
        "order_api": s.order_api_base_url or "in-process",
    }


# --------------------------------------------------------------------------
# Responsible AI surface
# --------------------------------------------------------------------------

sessions_router_tags = ["responsible-ai"]


@app.get("/api/sessions/{session_id}/trace", tags=sessions_router_tags)
def session_trace(session_id: str) -> JSONResponse:
    """Raw ordered event log for a call. PII already redacted at write time."""
    events = get_sink().session(session_id)
    if not events:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No trace for that session")
    return JSONResponse({"session_id": session_id, "events": events})


@app.get("/api/sessions/{session_id}/explain", tags=sessions_router_tags)
def session_explain(session_id: str) -> JSONResponse:
    """
    Explainability endpoint.

    Answers 'why did the agent do that' — for each decision: what it chose,
    how confident it was, what it rejected, and on what grounds.
    """
    payload = get_sink().explain(session_id)
    if payload["event_count"] == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No trace for that session")
    return JSONResponse(payload)


@app.delete("/api/sessions/{session_id}", tags=sessions_router_tags)
def forget_session(session_id: str) -> dict:
    """Right to erasure. Drops all retained trace data AND the in-memory PII vault."""
    forget_vault(session_id)
    return {"session_id": session_id, "erased": get_sink().forget(session_id)}


# --------------------------------------------------------------------------
# Web UI (services/web, React) at /app. Served from the gateway itself so the
# browser sees ONE origin: the WebSocket, the explain API and the page all
# share it -- no CORS, no second deploy. The Gradio harness below is kept
# deliberately as a working fallback UI; this mount is skipped (not an error)
# when the frontend hasn't been built, so a Python-only checkout still boots.
# --------------------------------------------------------------------------


_DEFAULT_WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"


def find_web_dist() -> Path | None:
    """The built frontend, or None. Explicit setting wins; else the
    repo-relative build output (services/web/dist) used in local dev."""
    configured = get_settings().web_dist_dir
    candidates = [Path(configured)] if configured else []
    candidates.append(_DEFAULT_WEB_DIST)
    for path in candidates:
        if (path / "index.html").is_file():
            return path
    return None


_web_dist = find_web_dist()
if _web_dist is not None:
    app.mount("/app", StaticFiles(directory=_web_dist, html=True), name="web")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/app/" if _web_dist is not None else "/ui/")


# --------------------------------------------------------------------------
# Gradio test harness (phase 6) -- mounted last, after every FastAPI route
# above is registered. app/ui/harness.py holds the actual logic; blocks.py is
# thin Gradio glue over it -- see both modules' docstrings.
#
# root_path="/ui" is load-bearing, not optional: without it, Gradio's own
# frontend JS calls its API (queue/join, upload, ...) at the SITE ROOT
# (/gradio_api/...) instead of under the mount (/ui/gradio_api/...), 404ing
# on every real interaction while the page itself still loads fine -- static
# assets don't need root_path, only Gradio's own follow-up API calls do. A
# TestClient GET on "/ui/" checks the initial HTML status only and never
# exercises this at all; it took an actual browser hitting the real deployed
# URL to surface it. Same lesson as the Docker order_api bundling bug:
# passing a shallow health-style check is not the same as the real client
# interaction working.
# --------------------------------------------------------------------------

app = gr.mount_gradio_app(app, build_demo(), path="/ui", root_path="/ui")
