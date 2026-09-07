"""
Shared error-message formatting for the two HTTP-based providers
(SarvamASRClient, SarvamTTSClient).

Why this exists: `str(httpx.HTTPStatusError)` gives you the status code and
URL but throws away the response BODY -- which for Sarvam (and most JSON
APIs) is exactly where the actual reason lives, e.g.
`{"error": {"message": "Model 'bulbul:v2' has been deprecated..."}}`. Found
the hard way: a live TTS failure was an opaque 400 in every log and trace
event until someone made the same request by hand, outside the client, just
to read the body. Never make that the only way to find out again.
"""

from __future__ import annotations

import httpx


def describe_http_error(exc: httpx.HTTPError) -> str:
    """
    A message worth putting in ASRUnavailable/TTSUnavailable and the trace
    event's `data.error` field -- includes the response body when there is
    one (a real HTTP error response), falls back to the exception's own
    string form for connection-level failures (timeout, DNS, etc.) that
    never got a response at all.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)
    body = response.text.strip()
    return f"{exc} -- response body: {body}" if body else str(exc)
