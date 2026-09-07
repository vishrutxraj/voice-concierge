"""
Phase 6: Gradio test harness, mounted at /ui by app/main.py.

harness.py holds every real decision (pure, dataclass-returning functions,
tested like any other provider-facing code); blocks.py is thin Gradio glue
over it. Neither module starts a server of its own -- app/main.py owns that
via gr.mount_gradio_app.
"""
