# Bias/fairness evaluation fixtures

Empty by design until real per-language transcripts exist here. **Do not
hand-write "how I imagine a Hindi caller's translated English would sound"**
-- that measures the author's assumptions about translation quality, not
anything real, and risks baking a fabricated bias pattern into the eval
instead of finding a genuine one.

## What this corpus is for

`scripts/eval_fairness.py` runs two checks. The first (language-invariance)
needs no fixtures and already runs today -- see the script's module
docstring. This directory is for the second: real cross-language accuracy
and sentiment-distribution comparison, the actual question the README's
Responsible AI table asks ("if Telugu callers are flagged frustrated more
often than English ones at equal sentiment, that's measurable bias").

## Where the transcripts come from

**Not `fixtures/audio/`'s recordings** -- those are real speech (good), but
general-domain FLEURS sentences (geography, history, unrelated facts), not
delivery/logistics phrases. Every one of them would correctly fall back as
out-of-scope regardless of language, which would make "accuracy" a trivial
100% and defeat the point of measuring cross-language *routing* fairness on
in-domain requests. FLEURS-style open corpora don't cover this project's
specific domain, and no open dataset of "delivery customer support calls in
10 Indian languages" is known to exist -- this is a real, unresolved gap
(see the main README's Limitations section), not a TODO with an obvious next
step. Closing it needs either a few short domain-specific recordings per
language (self- or native-speaker-recorded, covering the same three intents
as `fixtures/audio/README.md`'s recording exercise, plus at least one
genuinely annoyed-sounding utterance per language) or a domain-specific
corpus turning up later.

Once such recordings exist, run them through the real Sarvam ASR client in
`mode=translate`. The `text` field below IS that output -- not a reference
translation you'd write by hand for `eval_asr.py`, but literally whatever
Sarvam's `mode=translate` returned for that recording. That's the point:
this eval measures whether the SYSTEM behaves consistently on real
translated output, translation quirks included.

## manifest.json

Create `fixtures/fairness/manifest.json` alongside your recordings' real ASR
output:

```json
[
  {"language": "hi-IN", "text": "where is my order", "expected_intent": "order_lookup"},
  {"language": "te-IN", "text": "I want to change the delivery date", "expected_intent": "reschedule"},
  {"language": "mr-IN", "text": "this is useless, where is my order", "expected_intent": "order_lookup"}
]
```

`text` is real ASR `mode=translate` output for that language, not a hand-typed
approximation. `expected_intent` is what a human listening to the ORIGINAL
audio would say the caller wanted -- the ground truth the eval scores
`app/graph/nodes/router.py`'s actual decision against.

## Running the eval

```bash
python scripts/eval_fairness.py
python scripts/eval_fairness.py --out results.md
```

Whatever comes back, put it straight into the README's evaluation section --
including if some language's accuracy or sentiment distribution is
meaningfully off from the others. That's a more useful result than a
clean-looking report that isn't true.
