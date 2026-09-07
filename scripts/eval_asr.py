#!/usr/bin/env python3
"""
ASR evaluation harness.

Scores configured ASR provider(s) against a manifest of fixture recordings,
per language, with two deliberate choices that make the numbers trustworthy
rather than flattering:

  1. Telephony degradation FIRST. Every fixture is round-tripped through
     16kHz -> 8kHz mu-law -> 16kHz before transcription, because production
     audio is phone-call quality, not the clean read-speech that vendor
     benchmarks use. A model that scores well on the raw fixture and poorly
     after this round-trip is telling you something real about production
     accuracy that the raw number would have hidden.

  2. IndicNLP normalization, not Whisper's normalizer. Whisper's own
     normalizer strips diacritics and complex characters in a way that
     inflates Indic-language WER improvements by 20-150+ points depending on
     the language -- it's measuring how aggressively it can simplify the
     text, not how accurately it transcribed. IndicNLP's normalizer is a more
     honest baseline for comparing hypothesis to reference.

Usage:
    python scripts/eval_asr.py                    # uses fixtures/audio/manifest.json
    python scripts/eval_asr.py --provider mock     # force a specific provider
    python scripts/eval_asr.py --no-degrade        # skip telephony round-trip

Manifest format (fixtures/audio/manifest.json):
    [
      {"file": "hi_1.wav", "language": "hi-IN", "reference": "mera order kahan hai"},
      ...
    ]

Reference text should be the EXPECTED ENGLISH TRANSLATION -- ASR is run in
mode=translate (see app/providers/asr_client.py), so scoring compares English
to English regardless of what language was actually spoken.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).parent.parent
for path in (
    ROOT / "packages" / "order-contracts",
    ROOT / "services" / "order-api",
    ROOT / "services" / "gateway",
):
    sys.path.insert(0, str(path))

FIXTURES_DIR = ROOT / "fixtures" / "audio"
MANIFEST_PATH = FIXTURES_DIR / "manifest.json"


@dataclass
class EvalRow:
    language: str
    file: str
    reference: str
    hypothesis: str
    wer: float
    cer: float
    provider: str
    degraded: bool


def degrade_telephony(pcm16_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """
    16kHz PCM16 -> 8kHz mu-law -> back to 16kHz PCM16.

    This is what a real phone call does to audio quality. Scoring against
    fixtures that skip this step measures a different (better) problem than
    the one this project actually has to solve.
    """
    import audioop

    downsampled, _ = audioop.ratecv(pcm16_bytes, 2, 1, sample_rate, 8000, None)
    ulaw = audioop.lin2ulaw(downsampled, 2)
    back_to_linear = audioop.ulaw2lin(ulaw, 2)
    upsampled, _ = audioop.ratecv(back_to_linear, 2, 1, 8000, sample_rate, None)
    return upsampled


def read_wav_pcm16(path: pathlib.Path) -> tuple[bytes, int]:
    import wave

    with wave.open(str(path), "rb") as f:
        if f.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM, got {f.getsampwidth() * 8}-bit")
        return f.readframes(f.getnframes()), f.getframerate()


def write_wav_pcm16(data: bytes, sample_rate: int) -> bytes:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(data)
    return buf.getvalue()


def normalize_indic(text: str) -> str:
    """
    IndicNLP normalization for fair WER/CER scoring. Falls back to a plain
    lowercase/whitespace normalization if indic-nlp-library isn't installed
    (it's a dev-only dependency) -- degrades gracefully rather than crashing
    a CI run that only has requirements.txt, not requirements-dev.txt.
    """
    try:
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

        normalizer = IndicNormalizerFactory().get_normalizer("hi")
        return normalizer.normalize(text.strip().lower())
    except ImportError:
        return " ".join(text.strip().lower().split())


def score(reference: str, hypothesis: str) -> tuple[float, float]:
    """Returns (WER, CER), both IndicNLP-normalized first."""
    ref_norm = normalize_indic(reference)
    hyp_norm = normalize_indic(hypothesis)

    try:
        import jiwer

        wer = jiwer.wer(ref_norm, hyp_norm)
        cer = jiwer.cer(ref_norm, hyp_norm)
    except ImportError:
        print(
            "WARNING: jiwer not installed (pip install -r requirements-dev.txt) "
            "-- falling back to a crude word-overlap approximation.",
            file=sys.stderr,
        )
        ref_words, hyp_words = ref_norm.split(), hyp_norm.split()
        wer = 1.0 - (
            len(set(ref_words) & set(hyp_words)) / max(len(ref_words), 1)
        )
        cer = 1.0 - (
            len(set(ref_norm) & set(hyp_norm)) / max(len(ref_norm), 1)
        )
    return round(wer, 4), round(cer, 4)


def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    return json.loads(MANIFEST_PATH.read_text())


def run_eval(provider_name: str, degrade: bool) -> list[EvalRow]:
    from app.providers.asr_client import MockASRClient, SarvamASRClient

    manifest = load_manifest()
    if not manifest:
        return []

    if provider_name == "mock":
        client = MockASRClient()
    else:
        from app.config import get_settings

        settings = get_settings()
        if not settings.sarvam_api_key:
            print(
                "No SARVAM_API_KEY set -- falling back to the mock provider. "
                "Set the key in .env to get real accuracy numbers.",
                file=sys.stderr,
            )
            client = MockASRClient()
            provider_name = "mock"
        else:
            client = SarvamASRClient(
                settings.sarvam_api_key, settings.sarvam_asr_model,
                settings.sarvam_asr_mode, settings.sarvam_base_url,
            )

    rows: list[EvalRow] = []
    for entry in manifest:
        wav_path = FIXTURES_DIR / entry["file"]
        if not wav_path.exists():
            print(f"SKIP {entry['file']}: file not found in {FIXTURES_DIR}", file=sys.stderr)
            continue

        pcm, rate = read_wav_pcm16(wav_path)
        if degrade:
            pcm = degrade_telephony(pcm, rate)
        audio_bytes = write_wav_pcm16(pcm, rate)

        result = client.transcribe(audio_bytes, filename=entry["file"])
        wer, cer = score(entry["reference"], result.text)

        rows.append(EvalRow(
            language=entry["language"], file=entry["file"],
            reference=entry["reference"], hypothesis=result.text,
            wer=wer, cer=cer, provider=provider_name, degraded=degrade,
        ))
    return rows


def render_markdown(rows: list[EvalRow]) -> str:
    if not rows:
        return (
            "## ASR Evaluation\n\n"
            "No fixtures found. Add recordings to `fixtures/audio/` and a "
            "matching `fixtures/audio/manifest.json` (see this script's "
            "module docstring for the format) before running this again.\n"
        )

    by_lang: dict[str, list[EvalRow]] = {}
    for r in rows:
        by_lang.setdefault(r.language, []).append(r)

    lines = ["## ASR Evaluation Results\n"]
    lines.append(f"Telephony degradation: {'applied' if rows[0].degraded else 'skipped'}\n")
    lines.append("| Language | File | WER | CER | Provider |")
    lines.append("|---|---|---|---|---|")
    for lang in sorted(by_lang):
        for r in by_lang[lang]:
            lines.append(f"| {r.language} | {r.file} | {r.wer:.1%} | {r.cer:.1%} | {r.provider} |")

    lines.append("\n### Per-language average WER\n")
    lines.append("| Language | Avg WER | Sample count |")
    lines.append("|---|---|---|")
    for lang in sorted(by_lang):
        entries = by_lang[lang]
        avg = sum(e.wer for e in entries) / len(entries)
        lines.append(f"| {lang} | {avg:.1%} | {len(entries)} |")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["sarvam", "mock"], default="sarvam")
    parser.add_argument("--no-degrade", action="store_true", help="skip telephony round-trip")
    parser.add_argument("--out", type=pathlib.Path, help="write markdown to this file too")
    args = parser.parse_args()

    rows = run_eval(args.provider, degrade=not args.no_degrade)
    report = render_markdown(rows)
    print(report)

    if args.out:
        args.out.write_text(report)
        print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
