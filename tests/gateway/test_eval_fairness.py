"""
Tests for scripts/eval_fairness.py.

Mirrors test_eval_asr.py's shape: exercise the pure/testable functions
directly rather than the CLI glue. The language-invariance check is the one
piece of this eval that produces a REAL result today (no fixtures needed),
so it gets the most scrutiny; the corpus eval is tested only for its honest
placeholder behaviour and rendering, since it has no real data yet either
(see fixtures/fairness/README.md).
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_fairness  # noqa: E402

RAVI = eval_fairness.CALLER_PHONE


# ---- language-invariance check ---------------------------------------------


def test_invariance_check_finds_no_mismatches_on_canonical_utterances():
    """
    The real assertion this whole eval exists to make: router intent and
    sentiment score must be identical across every language tag for the same
    English text. This is the eval actually running for real, not a stub.
    """
    rows = eval_fairness.run_language_invariance_check()
    mismatches = [r for r in rows if not r.matches_baseline]
    assert mismatches == []
    assert len(rows) == len(eval_fairness.CANONICAL_UTTERANCES) * len(eval_fairness.LANGUAGE_TAGS)


def test_invariance_check_covers_every_requested_language():
    rows = eval_fairness.run_language_invariance_check(
        utterances=[{"label": "order_lookup", "text": "where is my order"}],
        languages=["en-IN", "hi-IN", "te-IN"],
    )
    assert {r.language for r in rows} == {"en-IN", "hi-IN", "te-IN"}
    assert all(r.intent == "order_lookup" for r in rows)
    assert all(r.matches_baseline for r in rows)


def test_invariance_check_flags_frustrated_utterance_with_negative_sentiment():
    rows = eval_fairness.run_language_invariance_check(
        utterances=[{
            "label": "frustrated", "text": "this is ridiculous, I am so frustrated",
        }],
        languages=["en-IN", "hi-IN"],
    )
    assert all(r.sentiment_score < 0 for r in rows)
    assert all(r.matches_baseline for r in rows)


def test_invariance_check_is_isolated_between_calls():
    """Two separate invocations must not interfere via shared order-store or
    checkpointer state -- each call sets up its own fresh environment."""
    first = eval_fairness.run_language_invariance_check(
        utterances=[{"label": "reschedule", "text": "reschedule to in 3 days evening"}],
        languages=["en-IN"],
    )
    second = eval_fairness.run_language_invariance_check(
        utterances=[{"label": "reschedule", "text": "reschedule to in 3 days evening"}],
        languages=["en-IN"],
    )
    assert first[0].intent == second[0].intent == "reschedule"
    assert first[0].confidence == second[0].confidence


def test_render_invariance_markdown_reports_no_mismatches():
    rows = [
        eval_fairness.InvarianceRow("order_lookup", "en-IN", "order_lookup", 0.7, 0.0, True),
        eval_fairness.InvarianceRow("order_lookup", "hi-IN", "order_lookup", 0.7, 0.0, True),
    ]
    report = eval_fairness.render_invariance_markdown(rows)
    assert "All 2 rows match" in report
    assert "MISMATCH" not in report


def test_render_invariance_markdown_flags_a_real_mismatch():
    rows = [
        eval_fairness.InvarianceRow("order_lookup", "en-IN", "order_lookup", 0.7, 0.0, True),
        eval_fairness.InvarianceRow("order_lookup", "hi-IN", "fallback", 0.2, 0.0, False),
    ]
    report = eval_fairness.render_invariance_markdown(rows)
    assert "1 MISMATCH" in report
    assert "MISMATCH" in report


# ---- corpus eval (manifest-based) -----------------------------------------


def test_load_fairness_manifest_returns_empty_list_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_fairness, "MANIFEST_PATH", tmp_path / "nonexistent.json")
    assert eval_fairness.load_fairness_manifest() == []


def test_run_corpus_eval_returns_empty_for_empty_manifest():
    assert eval_fairness.run_corpus_eval(manifest=[]) == []


def test_render_corpus_markdown_with_no_rows_explains_whats_missing():
    report = eval_fairness.render_corpus_markdown([])
    assert "No fixtures found" in report
    assert "fixtures/fairness/README.md" in report


def test_run_corpus_eval_scores_against_expected_intent():
    manifest = [
        {"language": "hi-IN", "text": "where is my order", "expected_intent": "order_lookup"},
        {"language": "te-IN", "text": "asdkjf qqzzxx nonsense", "expected_intent": "order_lookup"},
    ]
    rows = eval_fairness.run_corpus_eval(manifest)
    assert len(rows) == 2
    assert rows[0].correct is True  # actually order_lookup
    assert rows[1].correct is False  # actually fallback, expected order_lookup


def test_render_corpus_markdown_groups_by_language_and_computes_accuracy():
    rows = [
        eval_fairness.CorpusRow("hi-IN", "t1", "order_lookup", "order_lookup", 0.0),
        eval_fairness.CorpusRow("hi-IN", "t2", "reschedule", "fallback", 0.0),
        eval_fairness.CorpusRow("te-IN", "t3", "order_lookup", "order_lookup", -0.2),
    ]
    report = eval_fairness.render_corpus_markdown(rows)
    assert "hi-IN" in report and "te-IN" in report
    assert "50.0%" in report  # hi-IN: 1 of 2 correct
    assert "100.0%" in report  # te-IN: 1 of 1 correct
    assert "Accuracy spread across languages" in report
