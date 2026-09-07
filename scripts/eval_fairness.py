#!/usr/bin/env python3
"""
Bias/fairness evaluation harness.

Two checks, scoped to what's actually knowable offline (this project's hard
rule: the test suite runs with zero credentials -- CLAUDE.md):

  1. Language-invariance regression check (runs today, no fixtures needed).
     This architecture's core bet (README: "translate at the edge, reason in
     English") is that the router and sentiment_monitor operate ONLY on
     English text -- language is detected once at ASR and used at exactly one
     place, the TTS node (see app/audio/call_loop.py). That's a testable
     invariant: the SAME English text, tagged with a DIFFERENT
     detected_language, must produce an IDENTICAL routing decision and
     sentiment score. If it doesn't, something has started conditioning agent
     behaviour on language -- exactly the kind of bug the bias/fairness
     requirement exists to catch, and the cheapest possible point to catch it.

  2. Cross-language accuracy / sentiment-distribution corpus eval (manifest-
     based, same shape as eval_asr.py's fixtures/audio pattern). This is the
     REAL fairness question the README's Responsible AI table asks: "if
     Telugu callers are flagged frustrated more often than English ones at
     equal sentiment, that's measurable bias." Answering it for real needs a
     labeled corpus of ACTUAL per-language ASR-translated transcripts (real
     Sarvam mode=translate output, not hand-invented approximations of what a
     translation might sound like) -- that corpus doesn't exist yet, see
     fixtures/fairness/README.md. Until it does, this half of the report says
     so plainly rather than fabricating a number, exactly like eval_asr.py did
     before fixtures/audio/ had real recordings.

Usage:
    python scripts/eval_fairness.py
    python scripts/eval_fairness.py --out results.md
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile
import uuid
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).parent.parent
for path in (
    ROOT / "packages" / "order-contracts",
    ROOT / "services" / "order-api",
    ROOT / "services" / "gateway",
):
    sys.path.insert(0, str(path))

FIXTURES_DIR = ROOT / "fixtures" / "fairness"
MANIFEST_PATH = FIXTURES_DIR / "manifest.json"

# RAVI in tests/gateway/test_graph.py: one open order (DLV1004), so
# order_lookup/reschedule/address_change all have something real to act on
# instead of escalating on "no open orders" for every language.
CALLER_PHONE = "9990000002"

LANGUAGE_TAGS = [
    "en-IN", "hi-IN", "te-IN", "ta-IN", "bn-IN",
    "mr-IN", "gu-IN", "kn-IN", "ml-IN", "pa-IN",
]

# Fixed-offset date phrasing ("in 3 days evening", not "friday evening") --
# see CLAUDE.md's note on wall-clock-relative test phrasing colliding with
# seed.py's also-wall-clock-relative blackout dates.
CANONICAL_UTTERANCES = [
    {"label": "order_lookup", "text": "where is my order"},
    {"label": "reschedule", "text": "reschedule to in 3 days evening"},
    {"label": "address_change", "text": "I need to change my delivery address"},
    {
        "label": "frustrated_order_lookup",
        "text": "this is ridiculous, where is my order, I am so frustrated",
    },
    {"label": "gibberish_fallback", "text": "asdkjf qqzzxx nonsense"},
]


def _fresh_environment(checkpoint_dir: pathlib.Path) -> None:
    """
    Reset every provider singleton and reseed the order store -- the same
    isolation tests/gateway/test_graph.py's fresh_env fixture provides, as
    plain function calls instead of a pytest fixture, so this module's
    functions are self-contained whether called from main() or a test.
    """
    import os

    os.environ["CHECKPOINT_DSN"] = f"sqlite:///{checkpoint_dir / 'checkpoints.db'}"

    import order_api.store as store_mod
    from app.config import get_settings
    from app.providers.llm_client import reset_llm_client
    from app.providers.order_client import reset_order_client
    from order_api.kv import reset_backend
    from order_api.seed import build_fixtures
    from order_api.store import get_store

    get_settings.cache_clear()
    reset_backend()
    reset_order_client()
    reset_llm_client()
    store_mod._store = None
    store = get_store()
    store.reset()
    store.load(build_fixtures())


# ---- 1. language-invariance regression check -------------------------------


@dataclass
class InvarianceRow:
    label: str
    language: str
    intent: str
    confidence: float
    sentiment_score: float
    matches_baseline: bool


def run_language_invariance_check(
    utterances: list[dict] | None = None,
    languages: list[str] | None = None,
) -> list[InvarianceRow]:
    """
    For each canonical utterance, run the identical English text under every
    language tag and compare the router intent/confidence and sentiment score
    against that utterance's first (baseline) language. Every row should
    match -- a mismatch is the single most severe finding this eval can
    report, since it means language leaked into a decision that
    architecturally should never see it.
    """
    from app.graph.runner import forget_vault, run_turn

    utterances = CANONICAL_UTTERANCES if utterances is None else utterances
    languages = LANGUAGE_TAGS if languages is None else languages

    rows: list[InvarianceRow] = []
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_environment(pathlib.Path(tmp))

        for case in utterances:
            baseline: tuple[str, float, float] | None = None
            for lang in languages:
                session_id = f"fair-{case['label']}-{lang}-{uuid.uuid4().hex[:8]}"
                result = run_turn(
                    session_id, case["text"], caller_phone=CALLER_PHONE, language=lang,
                )
                router = result.state.get("router") or {}
                intent = router.get("intent", "")
                confidence = router.get("confidence", 0.0)
                sentiment = result.state.get("sentiment_score", 0.0)
                forget_vault(session_id)

                observed = (intent, confidence, sentiment)
                if baseline is None:
                    baseline = observed

                rows.append(InvarianceRow(
                    label=case["label"], language=lang, intent=intent,
                    confidence=confidence, sentiment_score=sentiment,
                    matches_baseline=observed == baseline,
                ))
    return rows


def render_invariance_markdown(rows: list[InvarianceRow]) -> str:
    lines = ["## Language-invariance regression check\n"]
    mismatches = [r for r in rows if not r.matches_baseline]
    if mismatches:
        lines.append(
            f"**{len(mismatches)} MISMATCH(ES) FOUND** — language is leaking into a "
            "decision that should be language-blind. This is a bug, not a style "
            "issue; see app/graph/nodes/router.py and sentiment.py.\n"
        )
    else:
        lines.append(
            f"All {len(rows)} rows match their utterance's baseline language. "
            "Router and sentiment_monitor are behaving as architecturally "
            "promised: language-blind.\n"
        )

    lines.append("| Utterance | Language | Intent | Confidence | Sentiment | Matches baseline? |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        flag = "OK" if r.matches_baseline else "MISMATCH"
        lines.append(
            f"| {r.label} | {r.language} | {r.intent} | {r.confidence:.2f} | "
            f"{r.sentiment_score:.2f} | {flag} |"
        )
    return "\n".join(lines) + "\n"


# ---- 2. cross-language corpus eval (manifest-based) ------------------------


@dataclass
class CorpusRow:
    language: str
    text: str
    expected_intent: str
    actual_intent: str
    sentiment_score: float

    @property
    def correct(self) -> bool:
        return self.actual_intent == self.expected_intent


def load_fairness_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    return json.loads(MANIFEST_PATH.read_text())


def run_corpus_eval(manifest: list[dict] | None = None) -> list[CorpusRow]:
    from app.graph.runner import forget_vault, run_turn

    manifest = load_fairness_manifest() if manifest is None else manifest
    if not manifest:
        return []

    rows: list[CorpusRow] = []
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_environment(pathlib.Path(tmp))
        for i, entry in enumerate(manifest):
            session_id = f"fair-corpus-{i}-{uuid.uuid4().hex[:8]}"
            result = run_turn(
                session_id, entry["text"], caller_phone=CALLER_PHONE,
                language=entry["language"],
            )
            router = result.state.get("router") or {}
            forget_vault(session_id)
            rows.append(CorpusRow(
                language=entry["language"], text=entry["text"],
                expected_intent=entry["expected_intent"],
                actual_intent=router.get("intent", ""),
                sentiment_score=result.state.get("sentiment_score", 0.0),
            ))
    return rows


def render_corpus_markdown(rows: list[CorpusRow]) -> str:
    if not rows:
        return (
            "## Cross-language accuracy / sentiment corpus eval\n\n"
            "No fixtures found. This needs a labeled corpus of REAL per-language "
            "ASR-translated transcripts (not hand-written approximations of what "
            "a translation might look like) — see `fixtures/fairness/README.md` "
            "for the format and why. Every number here is a placeholder until "
            "that corpus exists.\n"
        )

    by_lang: dict[str, list[CorpusRow]] = {}
    for r in rows:
        by_lang.setdefault(r.language, []).append(r)

    lines = ["## Cross-language accuracy / sentiment corpus eval\n"]
    lines.append("| Language | Text | Expected | Actual | Correct | Sentiment |")
    lines.append("|---|---|---|---|---|---|")
    for lang in sorted(by_lang):
        for r in by_lang[lang]:
            lines.append(
                f"| {r.language} | {r.text} | {r.expected_intent} | {r.actual_intent} | "
                f"{'yes' if r.correct else 'NO'} | {r.sentiment_score:.2f} |"
            )

    lines.append("\n### Per-language summary\n")
    lines.append("| Language | Accuracy | Avg sentiment | Sample count |")
    lines.append("|---|---|---|---|")
    accuracies: dict[str, float] = {}
    for lang in sorted(by_lang):
        entries = by_lang[lang]
        accuracy = sum(1 for e in entries if e.correct) / len(entries)
        accuracies[lang] = accuracy
        avg_sentiment = sum(e.sentiment_score for e in entries) / len(entries)
        lines.append(f"| {lang} | {accuracy:.1%} | {avg_sentiment:.2f} | {len(entries)} |")

    if len(accuracies) > 1:
        spread = max(accuracies.values()) - min(accuracies.values())
        verdict = "investigate before shipping" if spread > 0.15 else "within a reasonable band"
        lines.append(f"\nAccuracy spread across languages: {spread:.1%} ({verdict}).")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, help="write markdown to this file too")
    args = parser.parse_args()

    invariance_report = render_invariance_markdown(run_language_invariance_check())
    corpus_report = render_corpus_markdown(run_corpus_eval())
    report = invariance_report + "\n" + corpus_report
    print(report)

    if args.out:
        args.out.write_text(report)
        print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
