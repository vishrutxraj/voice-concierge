# ASR evaluation fixtures

## Where these came from

20 real recordings across 4 languages (`hi-IN`, `ta-IN`, `te-IN`, `bn-IN`, 5
each) — `manifest.json` alongside them. These are **not** self-recorded; they
come from [Google FLEURS](https://huggingface.co/datasets/google/fleurs)
(Conneau et al., 2022, *FLEURS: Few-shot Learning Evaluation of Universal
Representations of Speech*, [arXiv:2205.12446](https://arxiv.org/abs/2205.12446)),
licensed **CC-BY-4.0**. Real human speakers reading real sentences — not
synthetic/TTS audio, which would flatter the ASR numbers and defeat the
entire point of this eval (see the warning that used to be the only thing in
this file, still true, just no longer aspirational).

FLEURS is built on [FLoRes-101](https://github.com/facebookresearch/flores)
(Meta AI), an n-way parallel machine-translation benchmark: the same
underlying sentence is professionally translated into every FLoRes language
AND separately recorded by native speakers for FLEURS. That's what makes
this usable here specifically: each non-English clip's FLEURS `id` matches
an `en_us` FLEURS row with the *same* sentence, professionally translated —
a real English reference translation with no machine-translation step and no
one here having to write it by hand. `scripts/eval_fairness.py`'s corpus
half needs something FLEURS can't provide, though — see
`fixtures/fairness/README.md`.

If you use these fixtures elsewhere, retain the CC-BY-4.0 attribution above.

## What's here vs. what these are good for

These are general-domain sentences (geography, history, everyday facts) —
FLEURS is a broad speech benchmark, not a delivery/logistics corpus. That
makes them exactly right for measuring raw ASR/translation quality
(WER/CER, what `eval_asr.py` scores) across languages, and exactly wrong for
`eval_fairness.py`'s corpus eval, which needs *in-domain* utterances
("where is my order" spoken in Tamil, not "Argentina has one of the best
polo teams"). Don't repurpose these for that; see
`fixtures/fairness/README.md` for why that needs different material.

## Running the eval

```bash
python scripts/eval_asr.py                 # real Sarvam, needs SARVAM_API_KEY in .env
python scripts/eval_asr.py --provider mock  # free, tests the pipeline only --
                                             # MockASRClient just UTF-8-decodes
                                             # the raw WAV bytes, so its WER/CER
                                             # numbers are garbage BY DESIGN. Use
                                             # this only to confirm the file-
                                             # loading/degradation/scoring code
                                             # runs, never as an accuracy result.
python scripts/eval_asr.py --out results.md
```

**Cost, concretely**: these 20 clips total roughly 2 minutes of audio. At
this project's documented rate (₹30/hour of audio — see the README's Cost
model section), one full real run costs on the order of ₹1, once, ever —
`app/providers/cache.py` content-hashes every ASR request, so every re-run
after the first is a free filesystem read. That's comfortably inside
Sarvam's free signup credit.

Whatever comes back, put it straight into the README's evaluation section —
including if a language's WER is bad enough to cut it from v1. That's a more
useful result than a clean-looking number that isn't true.

## Adding more languages

FLEURS covers most major Indian languages (`gu_in`, `kn_in`, `ml_in`,
`mr_in`, `or_in`, `pa_in`, `sd_in`, `ur_pk`, `as_in`, plus the four already
here) via the same `id`-matched-to-`en_us` trick used to build this manifest.
Ask for more coverage and it can be extended the same way — cost scales with
total audio duration, not language count, so a handful more languages at 5
clips each stays just as cheap.
