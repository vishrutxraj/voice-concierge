"""
Tests for scripts/eval_asr.py.

Focused on the pure functions (degradation, normalization, scoring, markdown
rendering) rather than the CLI/manifest-loading glue, which is straightforward
I/O. Confirms two things that matter for trusting the eventual real numbers:
the telephony round-trip actually changes the audio (it would be a silent
no-op bug if it didn't), and WER/CER scoring behaves sanely on known inputs
before it's ever pointed at real recordings.
"""

from __future__ import annotations

import pathlib
import struct
import sys
import wave

ROOT = pathlib.Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_asr  # noqa: E402


def _make_tone_wav(n_samples: int = 8000, sample_rate: int = 16000) -> bytes:
    """A simple non-silent PCM16 tone, so degradation has something to act on."""
    import math

    samples = [int(3000 * math.sin(i * 0.05)) for i in range(n_samples)]
    return struct.pack(f"<{n_samples}h", *samples)


# ---- telephony degradation --------------------------------------------


def test_degrade_telephony_changes_the_audio():
    """
    A no-op degradation pass would defeat the entire point of scoring against
    realistic conditions -- this confirms the round-trip actually alters the
    signal (mu-law is lossy) rather than silently passing audio through.
    """
    original = _make_tone_wav()
    degraded = eval_asr.degrade_telephony(original, sample_rate=16000)
    assert degraded != original


def test_degrade_telephony_preserves_roughly_the_same_duration():
    """Downsample-then-upsample should return audio close to the original
    sample count, not truncated or wildly extended."""
    original = _make_tone_wav(n_samples=16000, sample_rate=16000)
    degraded = eval_asr.degrade_telephony(original, sample_rate=16000)
    original_samples = len(original) // 2
    degraded_samples = len(degraded) // 2
    assert abs(degraded_samples - original_samples) < 100


def test_wav_round_trip_produces_valid_wav(tmp_path):
    pcm = _make_tone_wav()
    wav_bytes = eval_asr.write_wav_pcm16(pcm, 16000)

    out_path = tmp_path / "roundtrip.wav"
    out_path.write_bytes(wav_bytes)

    read_back, rate = eval_asr.read_wav_pcm16(out_path)
    assert rate == 16000
    assert read_back == pcm


# ---- normalization -------------------------------------------------------


def test_normalize_lowercases_and_strips():
    assert eval_asr.normalize_indic("  Where IS My Order  ") == "where is my order"


def test_normalize_is_idempotent():
    once = eval_asr.normalize_indic("Hello World")
    twice = eval_asr.normalize_indic(once)
    assert once == twice


# ---- scoring ---------------------------------------------------------


def test_score_identical_text_is_zero_error():
    wer, cer = eval_asr.score("where is my order", "where is my order")
    assert wer == 0.0
    assert cer == 0.0


def test_score_completely_different_text_is_high_error():
    wer, cer = eval_asr.score("where is my order", "completely unrelated text here")
    assert wer > 0.5


def test_score_is_case_and_whitespace_insensitive():
    """Scoring must compare normalized forms -- 'Where IS my order' vs
    'where is my order' should not register as an error."""
    wer, _ = eval_asr.score("Where IS my order", "where is my order")
    assert wer == 0.0


def test_score_handles_empty_hypothesis_without_crashing():
    wer, cer = eval_asr.score("where is my order", "")
    assert wer == 1.0


# ---- manifest / end-to-end pipeline ---------------------------------------


def test_load_manifest_returns_empty_list_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_asr, "MANIFEST_PATH", tmp_path / "nonexistent.json")
    assert eval_asr.load_manifest() == []


def test_render_markdown_with_no_rows_explains_whats_missing():
    report = eval_asr.render_markdown([])
    assert "No fixtures found" in report
    assert "manifest.json" in report


def test_render_markdown_groups_by_language():
    rows = [
        eval_asr.EvalRow("hi-IN", "hi_1.wav", "ref", "hyp", 0.1, 0.05, "sarvam", True),
        eval_asr.EvalRow("hi-IN", "hi_2.wav", "ref", "hyp", 0.3, 0.15, "sarvam", True),
        eval_asr.EvalRow("te-IN", "te_1.wav", "ref", "hyp", 0.5, 0.25, "sarvam", True),
    ]
    report = eval_asr.render_markdown(rows)
    assert "hi-IN" in report and "te-IN" in report
    assert "20.0%" in report  # average of 0.1 and 0.3 for hi-IN


def test_full_pipeline_runs_against_a_real_fixture_with_mock_provider(tmp_path, monkeypatch):
    """
    End-to-end smoke test: write a real fixture + manifest, run the full
    pipeline (read -> degrade -> transcribe -> score -> render) against the
    free mock provider, confirm it produces a well-formed report and never
    touches the network.
    """
    fixtures_dir = tmp_path / "audio"
    fixtures_dir.mkdir()
    wav_path = fixtures_dir / "sample.wav"
    with wave.open(str(wav_path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(_make_tone_wav())

    manifest_path = fixtures_dir / "manifest.json"
    manifest_path.write_text(
        '[{"file": "sample.wav", "language": "en-IN", "reference": "hello world"}]'
    )

    monkeypatch.setattr(eval_asr, "FIXTURES_DIR", fixtures_dir)
    monkeypatch.setattr(eval_asr, "MANIFEST_PATH", manifest_path)

    rows = eval_asr.run_eval("mock", degrade=True)
    assert len(rows) == 1
    assert rows[0].language == "en-IN"
    assert rows[0].degraded is True

    report = eval_asr.render_markdown(rows)
    assert "en-IN" in report
    assert "sample.wav" in report
