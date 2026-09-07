"""
app.providers.http_errors tests.

Regression coverage for a real bug: str(httpx.HTTPStatusError) alone drops
the response BODY, which is where a JSON API actually explains a 4xx/5xx.
A live Sarvam TTS failure was an opaque "400 Bad Request" in every log and
trace event until someone made the same request by hand, outside the
client, just to read the body (it turned out to be a deprecated model ID).
"""

from __future__ import annotations

import httpx
from app.providers.http_errors import describe_http_error


def test_includes_response_body_for_an_http_status_error():
    request = httpx.Request("POST", "https://api.sarvam.ai/text-to-speech")
    response = httpx.Response(
        400,
        json={"error": {"message": "Model 'bulbul:v2' has been deprecated."}},
        request=request,
    )
    exc = httpx.HTTPStatusError("400 Bad Request", request=request, response=response)

    message = describe_http_error(exc)

    assert "bulbul:v2" in message
    assert "deprecated" in message


def test_falls_back_to_str_when_there_is_no_response():
    """Connection-level failures (timeout, DNS, refused) never got a
    response at all -- there's no body to add, so this must not crash
    trying to read one."""
    request = httpx.Request("POST", "https://api.sarvam.ai/text-to-speech")
    exc = httpx.ConnectTimeout("connection timed out", request=request)

    message = describe_http_error(exc)

    assert "connection timed out" in message


def test_does_not_crash_on_an_empty_body():
    request = httpx.Request("GET", "https://api.sarvam.ai/health")
    response = httpx.Response(500, content=b"", request=request)
    exc = httpx.HTTPStatusError("500 Internal Server Error", request=request, response=response)

    message = describe_http_error(exc)

    assert "500" in message
