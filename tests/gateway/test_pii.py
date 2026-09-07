from __future__ import annotations

import json

from app.observability.pii import PIIVault, redact, scrub_mapping, tokenize
from app.observability.trace import EventKind, TraceEvent, TraceSink

# ---- redaction -----------------------------------------------------------


def test_redacts_indian_mobile():
    r = redact("Call me on 9876543210 please")
    assert "9876543210" not in r.text
    assert "PHONE" in r.kinds_found


def test_redacts_mobile_with_country_code():
    assert "9876543210" not in redact("reach me at +91 9876543210").text


def test_redacts_email():
    r = redact("mail anita.rao@example.com")
    assert "anita.rao@example.com" not in r.text and "EMAIL" in r.kinds_found


def test_redacts_national_id():
    r = redact("aadhaar is 1234 5678 9012")
    assert "1234 5678 9012" not in r.text


def test_redacts_address_opener():
    r = redact("I live at Flat 402, Sunrise Apartments, Kondapur")
    assert "Sunrise Apartments" not in r.text and "ADDRESS" in r.kinds_found


def test_redacts_pincode():
    assert "PINCODE" in redact("pincode 411014").kinds_found


def test_clean_text_is_untouched():
    text = "where is my parcel"
    r = redact(text)
    assert r.text == text and not r.had_pii


def test_empty_input_is_safe():
    assert redact("").text == ""


def test_multiple_pii_types_in_one_utterance():
    r = redact("I'm at Flat 12 Nehru Nagar 411014, call 9876543210")
    assert set(r.kinds_found) >= {"ADDRESS", "PHONE"}
    assert "9876543210" not in r.text


def test_redaction_is_irreversible():
    """No token, no mapping — the original must be unrecoverable from output."""
    r = redact("call 9876543210")
    assert "REDACTED" in r.text
    assert not any(ch.isdigit() for ch in r.text.replace("REDACTED", ""))


# ---- tokenization --------------------------------------------------------


def test_tokenize_is_reversible():
    vault = PIIVault()
    original = "Deliver to Flat 402, Sunrise Apts"
    tok = tokenize(original, vault)
    assert "Sunrise Apts" not in tok.text
    assert vault.rehydrate(tok.text) == original


def test_same_value_gets_stable_token():
    vault = PIIVault()
    a = tokenize("call 9876543210", vault).text
    b = tokenize("again 9876543210", vault).text
    assert a.split()[-1] == b.split()[-1], "repeat values must reuse one token"
    assert len(vault) == 1


def test_distinct_values_get_distinct_tokens():
    vault = PIIVault()
    tokenize("9876543210 and 9123456789", vault)
    assert len(vault) == 2


def test_vault_clear_forgets_everything():
    vault = PIIVault()
    tokenize("call 9876543210", vault)
    vault.clear()
    assert len(vault) == 0


def test_rehydrate_leaves_unknown_tokens_alone():
    vault = PIIVault()
    assert vault.rehydrate("hello <ADDR_9>") == "hello <ADDR_9>"


# ---- structured log scrubbing -------------------------------------------


def test_scrub_mapping_redacts_nested_strings():
    out = scrub_mapping({"turn": {"text": "my number is 9876543210"}})
    assert "9876543210" not in json.dumps(out)


def test_scrub_mapping_drops_audio_and_secrets():
    out = scrub_mapping({"audio": b"\x00\x01", "api_key": "sk-live-xyz"})
    assert out["audio"] == "[DROPPED]" and out["api_key"] == "[DROPPED]"


def test_scrub_mapping_drops_vault_reference():
    """A vault must never reach a log, even by accident."""
    assert scrub_mapping({"pii_vault": {"x": "y"}})["pii_vault"] == "[DROPPED]"


def test_scrub_mapping_handles_lists():
    out = scrub_mapping({"turns": ["call 9876543210", {"t": "at 411014"}]})
    assert "9876543210" not in json.dumps(out)


# ---- trace sink ----------------------------------------------------------


def test_trace_scrubs_pii_at_write_time(capsys):
    sink = TraceSink()
    sink.emit(
        TraceEvent(
            session_id="s1",
            kind=EventKind.DECISION,
            node="router",
            message="order_lookup",
            reasoning="caller said their number is 9876543210",
            data={"transcript": "my number is 9876543210"},
        )
    )
    capsys.readouterr()
    dumped = json.dumps(sink.session("s1"))
    assert "9876543210" not in dumped, "PII must never survive into a trace"


def test_explain_surfaces_decisions_with_confidence():
    sink = TraceSink()
    sink.emit(
        TraceEvent(
            session_id="s2",
            kind=EventKind.DECISION,
            node="router",
            message="reschedule",
            confidence=0.82,
            reasoning="caller asked to move the delivery",
            alternatives=[{"intent": "order_lookup", "score": 0.11}],
        )
    )
    out = sink.explain("s2")
    assert out["decisions"][0]["chose"] == "reschedule"
    assert out["decisions"][0]["confidence"] == 0.82
    assert out["decisions"][0]["rejected"][0]["intent"] == "order_lookup"


def test_forget_erases_session():
    sink = TraceSink()
    sink.emit(TraceEvent(session_id="s3", kind=EventKind.NODE_ENTER,
                         node="asr", message="in"))
    assert sink.forget("s3") is True
    assert sink.session("s3") == []


def test_forget_unknown_session_is_falsey():
    assert TraceSink().forget("nope") is False
