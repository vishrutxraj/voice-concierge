"""
Phase 4: audio transport layer.

Everything in this package sits ABOVE the graph, never inside it. The graph
(app/graph/) only ever deals in text -- CallState has no audio field and never
will. What lives here:

  confirmation.py        interpret a caller's spoken "yes"/"no" reply as the
                          bool run_turn's resume_value expects.
  transport.py            AudioTransport ABC -- one physical connection to a
                          caller, direction-agnostic.
  websocket_transport.py  the one AudioTransport implementation phase 4 ships:
                          a browser mic over FastAPI's WebSocket. A Twilio
                          media-stream transport would implement the same ABC
                          without touching call_loop.py at all.
  call_loop.py            orchestrates ASRClient -> run_turn -> TTSClient over
                          an AudioTransport, with streaming playback and
                          barge-in. This is the only new caller of run_turn.
"""
