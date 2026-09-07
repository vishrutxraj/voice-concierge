"""
Gradio Blocks UI -- thin glue over harness.py.

Every handler here does the minimum: pull values off components, call a
harness.py function, format the result back onto components. No business
logic lives in this file; see harness.py's module docstring for why.

Mounted at /ui by app/main.py via gr.mount_gradio_app -- this module only
builds the Blocks graph, it never starts a server of its own (no
.launch() call), so it's exactly as free to import in a test as any other
module here.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr

from app.config import get_settings
from app.ui.harness import (
    TurnOutcome,
    get_explain,
    new_session_id,
    run_audio_turn,
    run_text_turn,
)

# One of the seeded fixtures' phone numbers (order_api/seed.py) -- a caller
# with a real open order, so a first-time visitor sees the graph actually do
# something instead of an immediate "no open orders" escalation.
DEFAULT_CALLER_PHONE = "9990000002"


def _write_temp_wav(data: bytes) -> str:
    """
    Gradio serves audio OUTPUT components from a file path, not raw bytes --
    delete=False is required here because Gradio reads the file after this
    function returns; Gradio's own temp-directory cleanup takes it from there.
    """
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    f.write(data)
    f.close()
    return f.name


def _append_chat(history: list[dict], user_text: str, outcome: TurnOutcome) -> list[dict]:
    reply = outcome.reply_text or "(no reply -- see Explain panel for why)"
    if outcome.error:
        reply = f"{reply}\n\n⚠️ {outcome.error}"
    return history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": reply},
    ]


def build_demo() -> gr.Blocks:
    settings = get_settings()

    # analytics_enabled=False is not a style choice -- without it, Gradio
    # spins up a background thread that phones home to api.gradio.app and
    # huggingface.co on every build_demo() call. That's a real network call
    # from what is supposed to be an offline, zero-network test suite (see
    # CLAUDE.md's hard rule), and it's exactly the kind of undisclosed
    # third-party call the PII/observability layer exists to avoid elsewhere
    # in this codebase.
    with gr.Blocks(title="Voice Concierge — test harness", analytics_enabled=False) as demo:
        gr.Markdown(
            "# Logistics & Delivery Voice Concierge — test harness\n"
            f"Provider mode: **{'offline / mock' if settings.offline else 'live'}** "
            "(set `SARVAM_API_KEY` / `GROQ_API_KEY` for real ASR/TTS/LLM). "
            "Try caller phone `9990000002` (one open order), `9990000001` "
            "(three open orders), or `9990000003` (none)."
        )

        session_id_state = gr.State(new_session_id())
        pending_confirmation_state = gr.State(None)

        with gr.Row():
            session_box = gr.Textbox(
                label="Session ID", value=session_id_state.value, interactive=False, scale=3,
            )
            phone_box = gr.Textbox(label="Caller phone", value=DEFAULT_CALLER_PHONE, scale=2)
            new_session_btn = gr.Button("New session", scale=1)

        with gr.Tab("Text chat"):
            chatbot = gr.Chatbot(type="messages", label="Conversation", height=360)
            with gr.Row():
                msg_box = gr.Textbox(
                    label="Message", placeholder="where is my order",
                    scale=4, show_label=False,
                )
                send_btn = gr.Button("Send", scale=1, variant="primary")

        with gr.Tab("Audio playground"):
            gr.Markdown(
                "Upload or record one complete utterance per turn -- this harness "
                "is single-shot (ASR → graph → TTS), not the streaming/barge-in "
                "path `/ws/call` uses. See `app/audio/call_loop.py` for that."
            )
            audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Say something")
            send_audio_btn = gr.Button("Send audio", variant="primary")
            with gr.Row():
                transcript_box = gr.Textbox(label="Transcript (what ASR heard)", interactive=False)
                language_box = gr.Textbox(label="Detected language", interactive=False)
            reply_box = gr.Textbox(label="Reply text", interactive=False)
            reply_audio = gr.Audio(label="Reply audio", type="filepath", interactive=False)

        explain_json = gr.JSON(label="Explain trail (GET /api/sessions/{id}/explain)")

        def on_new_session():
            sid = new_session_id()
            return sid, sid, [], None, {}

        new_session_btn.click(
            on_new_session,
            outputs=[session_id_state, session_box, chatbot, pending_confirmation_state, explain_json],
        )

        def on_send(session_id, phone, message, pending, history):
            if not message or not message.strip():
                return history, "", pending, gr.update()
            outcome = run_text_turn(session_id, phone, message, pending)
            history = _append_chat(history, message, outcome)
            return history, "", outcome.awaiting_confirmation, get_explain(session_id)

        send_btn.click(
            on_send,
            inputs=[session_id_state, phone_box, msg_box, pending_confirmation_state, chatbot],
            outputs=[chatbot, msg_box, pending_confirmation_state, explain_json],
        )
        msg_box.submit(
            on_send,
            inputs=[session_id_state, phone_box, msg_box, pending_confirmation_state, chatbot],
            outputs=[chatbot, msg_box, pending_confirmation_state, explain_json],
        )

        def on_send_audio(session_id, phone, audio_path, pending):
            if not audio_path:
                return "", "", "", None, pending, gr.update()
            audio_bytes = Path(audio_path).read_bytes()
            result = run_audio_turn(session_id, phone, audio_bytes, pending)
            reply_path = _write_temp_wav(result.reply_audio_bytes) if result.reply_audio_bytes else None
            reply_text = result.turn.reply_text
            if result.turn.error:
                reply_text = f"{reply_text}\n\n⚠️ {result.turn.error}"
            return (
                result.transcript, result.detected_language or "", reply_text, reply_path,
                result.turn.awaiting_confirmation, get_explain(session_id),
            )

        send_audio_btn.click(
            on_send_audio,
            inputs=[session_id_state, phone_box, audio_in, pending_confirmation_state],
            outputs=[
                transcript_box, language_box, reply_box, reply_audio,
                pending_confirmation_state, explain_json,
            ],
        )

    return demo
