# Logistics & Delivery Voice Concierge — Claude Code project memory

Read this before touching anything. It exists so you don't have to rediscover
decisions and bugs that were already found the hard way.

## What this is

A multilingual voice AI agent for delivery/logistics customer support, built
for a Jio Institute live-project assignment with Responsible-AI requirements
(observability, bias/fairness, privacy, guardrails, explainability,
human-in-the-loop). Full architecture and rationale: **read `README.md`
first, in full** — it is kept current and is the source of truth for *why*
things are built the way they are, not just what exists.

## Current status: phase 7 of 7 complete

```
0. Scaffold, config, PII layer, tracing, CI, Docker              ✅
1. Order API (Vercel), gateway (Railway), Upstash KV split        ✅
2. LangGraph core: router, 3 agents, interrupt() confirmations,   ✅
   fuzzy order-ID resolution, sentiment monitor
3. Sarvam ASR/TTS clients, content-hash cache, LLM-assisted       ✅
   date/address extraction
4. WebSocket audio, streaming, barge-in                           ✅
5. Guardrails (input/output moderation), bias/fairness eval       ✅
6. Gradio test harness, real ASR eval fixtures, docs              ✅
7. Deploy live                                                    ✅
```

All 7 phases built and verified — and phase 7 is actually deployed, not
just Docker-verified locally: order-api on Vercel (Upstash-backed) and the
gateway on Railway, pointed at each other, both confirmed with real
requests (see README.md's live URLs near the top). The plan named HF Spaces
as a third target; that got dropped for a verified, non-hypothetical
reason, not left undone — see the HF Spaces entry near the end of "Bugs
already found" below before assuming it's still just a free fallback
waiting to be pushed to.

227 tests pass, offline, with zero credentials. **If a test needs a live API
key, the test is wrong** — this has been a hard rule since phase 0 and every
provider (order client, LLM, ASR, TTS) has a mock implementation specifically
so the whole suite runs free and fast. Keep it that way.

## Non-negotiable architectural rules

Breaking any of these will look fine locally and be wrong in a way that
matters. They exist because each one was actually violated once during the
build and caught by a test or a manual check — see "Bugs already found" below
for the specifics.

1. **Never trust ASR for an alphanumeric order ID.** Fuzzy-match the
   transcript against the caller's own open orders (`app/graph/order_resolution.py`).
   This is the single most important correctness decision in the project.
2. **PII is a layer, not a node.** Redaction (irreversible, toward logs/traces)
   and tokenization (reversible, toward the LLM) both live in
   `app/observability/pii.py`. `PIIVault` is intentionally excluded from
   LangGraph's checkpoint serialization (`app/graph/checkpointer.py` raises if
   one ever reaches that boundary) — don't "fix" that exclusion.
3. **Writes need `interrupt()` confirmation first.** Reschedule and address
   changes propose, read back, and only call `OrderClient` after the caller
   confirms. Don't add a write path that skips this.
4. **Every state key needs exactly one writer per LangGraph superstep.**
   `intent_router` and `sentiment_monitor` run concurrently (both fan out from
   `START`). Domain agents write `agent_reply`; only `composer` writes the
   final `reply_text`. If you add a new concurrent branch, give it its own
   keys — see `test_router_and_sentiment_run_concurrently_without_state_collision`.
5. **Verify third-party API contracts before writing code against them.**
   The original project brief referenced a since-deprecated Sarvam endpoint
   and a dead model ID. Don't assume training-data knowledge of any external
   API (Sarvam, Groq, Twilio, HF Spaces) is current — check the live docs
   first. This project's Sarvam clients were rebuilt once already after
   verifying against `docs.sarvam.ai` mid-build.
6. **Don't guess a required field across a state change you can't fully
   resolve.** If an extraction path can't determine something required
   (e.g. `Address.state` after a city change), escalate to a human rather
   than carrying forward a stale or default value. See
   `app/graph/extraction.py`'s `city_change_blocked` handling.
7. **order_api has zero imports from the gateway.** Mechanically enforced by
   `test_module_is_liftable` (parses the AST). This is what keeps the
   Vercel/Railway split real instead of aspirational.
8. **A conditional edge cannot reliably see a concurrent sibling's write.**
   Proven empirically while building the guardrails: `route_edge`, attached
   to `intent_router`'s own outgoing edge, does NOT reliably see
   `input_guardrail`'s write even though both fan out from `START` in the
   same nominal superstep — a conditional edge fires off its source node's
   own completion, not the fully-merged superstep. A *node* in the next
   superstep DOES see the merged state (this is what lets `composer` see
   both its concurrent predecessors). So: if a concurrent branch's verdict
   needs to affect what happens next, check it inside the downstream
   node(s), never inside a conditional-edge function keyed to a different
   node. See `app/graph/nodes/input_guardrail.py::guardrail_block_patch()`
   and its call sites in `order_lookup.py`/`reschedule.py`/
   `address_change.py`/`fallback.py`.

## Bugs already found (don't reintroduce these)

- Regex substring matching silently mis-resolved "day after tomorrow" (matched
  "tomorrow") and "next monday" (matched "monday") — fixed by checking longer,
  specific phrases before the substrings they contain
  (`app/graph/regex_extractors.py`).
- A caller dictating a full new-city address got only the pincode extracted,
  silently keeping the OLD street — because a narrow single-field regex fired
  before checking for full-address signals. Ordering matters in
  `app/graph/extraction.py::extract_address`.
- Fixing that exposed a second leak: the LLM path inherited a stale
  city-specific field (old landmark) across a city change.
- `Command(resume=False)` was treated by LangGraph as "no input" because it's
  falsy — wrapped as `{"confirmed": bool}` in `app/graph/runner.py`.
- Test phrasing like `"reschedule to friday evening"` broke on some days
  because "next Friday" is wall-clock-relative and the fixtures' blackout
  dates (`T(5)`, `T(12)` in `seed.py`) are *also* wall-clock-relative. Tests
  now use `"in 3 days evening"` or similar fixed offsets. If you write a new
  test involving a date, check it against `seed.py`'s `BLACKOUT_DATES` and
  the saturated `T(3)/MORNING` slot first.
- Phase 4: a turn that pauses at `interrupt()` (reschedule/address_change)
  never reaches `composer`, so `reply_text` is `""` on exactly that turn — the
  text endpoint gets away with this because its caller reads
  `awaiting_confirmation.prompt` directly instead. `app/audio/call_loop.py`
  originally spoke `reply_text` over TTS unconditionally, so the caller heard
  silence and the call hung waiting for an `audio_end` that would never come
  on every reschedule/address-change proposal. Fixed by falling back to
  `awaiting_confirmation["prompt"]` when `reply_text` is empty, and by making
  `_deliver` always send a definite end-of-turn signal even when there's
  truly nothing to speak. Any new call_loop code path that sends `reply_text`
  to TTS needs this same fallback — check for it if you add one.
- The root `requirements.txt` that `requirements-dev.txt` requires (`-r
  requirements.txt`) didn't exist in the repo — only the two per-service
  files under `services/*/requirements.txt` did, which made `pip install -r
  requirements-dev.txt` (what CI and `dev.ps1 setup` both run) fail on a
  clean checkout. Added a root `requirements.txt` that unions the two service
  files. If you ever see this error again, check that file wasn't deleted.
- Phase 5: the first cut of `input_guardrail` tried to enforce the block by
  having `route_edge` check `input_guardrail`'s verdict — plausible-looking,
  and wrong. `route_edge` never saw the verdict (see architectural rule 8),
  so a self-harm message was silently routed to `order_lookup` and ONLY
  caught by an end-to-end probe showing the wrong node actually ran, not by
  reasoning about the code. Fixed by moving enforcement into
  `guardrail_block_patch()`, called at the top of every node `route_edge` can
  land on, instead of trying to gate the routing decision itself.
- Phase 6: `gr.Blocks()` without `analytics_enabled=False` starts a
  background thread that calls `api.gradio.app` and `huggingface.co` on
  every construction — a real, undisclosed network call from a service
  whose whole test suite is supposed to run offline. Caught by watching
  `TestClient(app)` make actual outbound HTTP requests during manual
  verification, not by reading Gradio's docs first. Fixed by passing
  `analytics_enabled=False` to `gr.Blocks(...)` in `app/ui/blocks.py`, with a
  regression test that patches `gradio.analytics.version_check` /
  `initiated_analytics` to raise if either is ever called again.
- Phase 6: adding a real `SARVAM_API_KEY` to `.env` (to run the real ASR
  eval) made a big chunk of the test suite start timing out. `Settings`'
  `env_file=".env"` means pydantic-settings falls back to reading `.env`
  whenever the process environment doesn't already have a var — every test
  going through `get_asr_client()`/`get_tts_client()` (call_loop, the WS
  endpoint, the Gradio harness, guardrail tests that run full turns) picked
  up the now-real key and started making live, slow, network calls with fake
  test "audio" bytes like `b"where is my order"`, instead of the instant
  mock. **`CLAUDE.md`'s zero-credentials rule was only ever true because no
  `.env` existed yet — nothing enforced it.** Fixed in `conftest.py`: an
  autouse fixture forces `SARVAM_API_KEY`/`GROQ_API_KEY`/both Langfuse keys
  to `""` (a real env var override beats the `.env` fallback) for every
  test, session-wide, so the suite is hermetic regardless of what's in a
  developer's real `.env`. If you ever add a new provider with its own
  credential env var, add it to that fixture's list too.
- Phase 7: `services/gateway/Dockerfile` never copied `services/order-api`
  into the image. Every test and local dev run happens from the repo root,
  where `conftest.py` puts all three package roots on `sys.path` -- so
  `InProcessOrderClient`'s `from order_api.store import get_store` (used
  whenever `ORDER_API_BASE_URL` is unset) always quietly worked, and no test
  ever ran from inside the actual built image where that isn't true.
  Building and running the real container for the first time (`docker build`
  + `docker run` + hitting `/call/turn`, not just `/health`) surfaced a
  `ModuleNotFoundError: No module named 'order_api'` on every order-touching
  request -- a gateway deployed alone, without also deploying the Vercel
  order-api service and setting `ORDER_API_BASE_URL`, would 500 on
  everything, silently contradicting the README's "runs identically on a
  laptop, in CI, and on Hugging Face Spaces" promise. `/health` reporting
  `"order_api":"in-process"` gave no hint anything was wrong -- it doesn't
  actually touch the store. Fixed by also `COPY`ing
  `services/order-api/order_api` into the image and adding it to
  `PYTHONPATH`. **Lesson: `curl /health` passing is not the same as the
  service working -- CI's own `gateway-image` job only ever checked
  `/health`; exercise at least one real request path (`/call/turn`) against
  a locally built image before trusting a Dockerfile change, or a Vercel/
  Railway split boundary change, is deploy-ready.**
- Phase 7 (Vercel deploy prep): `services/order-api/vercel.json` had
  `"runtime": "@vercel/python@4.3.1"` and `"memory": 512` in its `functions`
  config -- both obsolete/wrong per current (2026-08) docs.vercel.com,
  checked BEFORE the first real deploy rather than assumed from training
  data (rule 5). Vercel's Python support is zero-config now (auto-detected
  from `requirements.txt`); `runtime` in `vercel.json` is reserved for
  community/third-party runtimes only (e.g. `vercel-php@0.5.2`) and an
  unrecognized value there is a known cause of "Function Runtimes must have
  a valid version" deploy failures. `memory` can't be set in `vercel.json`
  at all with Fluid compute enabled (the default for new projects) -- it
  belongs in the dashboard's Functions settings. Fixed by dropping both,
  keeping `maxDuration` (still valid). Note: the `rewrites` rule mentioned
  here initially got REMOVED entirely a bit later (see the next entry) --
  and the "Include source files outside Root Directory" setting turned out
  to be the wrong lever for the `order_contracts` problem too (see the entry
  after that). Both corrections came from the actual first live deploy, not
  from re-reading docs harder.
- Phase 7 (first live Vercel deploy): even after the config fixes above,
  `api/index.py` crashed at import with `ModuleNotFoundError: No module
  named 'order_contracts'`. The "Include source files outside Root
  Directory" setting governs what's *readable during the build*, not what
  Vercel's zero-config Python bundler actually ships into the function's
  runtime filesystem -- a sibling directory outside Root Directory, reached
  only via a runtime `sys.path.insert`, is invisible to the bundler
  regardless of that setting. Fixed with `vercel.json`'s `installCommand`:
  `cp -r ../../packages/order-contracts/order_contracts . && pip install -r
  requirements.txt` -- copies the package to a sibling of `order_api/`
  *inside* Root Directory before install, so it's bundled like any other
  source file. No code change needed: `api/index.py`'s existing
  `sys.path.insert(_ROOT)` already covers this location. `packages/order-
  contracts` stays the one source of truth; the copy is gitignored
  (`services/order-api/order_contracts/`) and produced fresh on every build.
- Phase 7 (same deploy, next error): with the import fixed, every request
  404'd. Per Vercel's current FastAPI framework docs, detecting a FastAPI
  `app` at a supported entrypoint makes "the framework application handle
  all requests" directly, original path intact -- no `rewrites` needed. Our
  `rewrites` rule (`"/(.*)" -> "/api/index"`) was leftover from the older
  file-based-routing-without-a-framework-preset model, and was sending every
  request's effective path to the literal string `/api/index`, which
  matches none of our routes (`order_api/router.py`'s are already correctly
  prefixed `/api/orders...`, and `/api/health` is on `app` directly). Fixed
  by deleting `rewrites` from `vercel.json` entirely.
- Phase 7 (first live cross-service request, Railway gateway -> Vercel
  order-api): `ensure_order_id`'s `client.orders_for_phone(phone)` --
  the shared "which order is this about" call every domain agent makes
  first -- had no `OrderAPIUnavailable` handling, unlike every other
  `OrderClient` call site (`order_lookup.py`, `reschedule.py`,
  `address_change.py` all already escalate gracefully on it). A Vercel cold
  start alone was enough to exceed the 4s client timeout and crash the
  whole turn with a raw 500 instead of "let me connect you with a
  colleague." Never caught by any test because local/CI OrderClient calls
  (in-process, or mocked HTTP transport) have no real timing/network
  failure mode to trigger it -- only a genuine live cross-cloud request
  could surface this. Fixed in `app/graph/nodes/order_identify.py` with the
  same try/except-and-escalate pattern used everywhere else; regression
  test added
  (`test_order_api_unavailable_during_phone_lookup_escalates_gracefully`).
  **Known follow-up, not fixed yet:** `order_resolution.py`'s
  `resolve_order_id` and `format_disambiguation_prompt` each call
  `orders_for_phone` AGAIN independently (redundant network calls on top of
  `ensure_order_id`'s own call) and are NOT covered by this guard -- a
  caller with multiple open orders could still hit an unhandled crash if
  one of those specific calls fails after the first one succeeded.
- Phase 7 (first real `/ws/call` test against the live Railway deployment,
  after Vercel/Railway were already verified working over text): TTS failed
  on every single reply with a bare `TTSUnavailable` 400. `SarvamTTSClient`
  only calls `resp.raise_for_status()` and never logs the response BODY, so
  the real reason was invisible until a raw `httpx.post` to
  `api.sarvam.ai/text-to-speech` was made by hand to read it:
  `"Model 'bulbul:v2' has been deprecated. Please use 'bulbul:v3' instead."`
  -- config.py's own comment claimed bulbul:v2 was "verified against
  docs.sarvam.ai (Aug 2026)" and cheaper for dev use; Sarvam removed it
  entirely at some point after that, and nothing re-checked it until a real
  audio call actually needed it. `env=dev` (this project's default) always
  used the dev model/speaker pair, so this broke ALL audio replies in the
  live deployment while text-only `/call/turn` stayed completely unaffected
  (it never touches TTS) -- which is exactly why the earlier "it's deployed
  and working" checks didn't catch it. Also confirmed live: `anushka`
  (bulbul:v2's default speaker) isn't a valid bulbul:v3 speaker at all
  (Sarvam's 400 body lists the real current roster). Fixed by moving both
  dev and demo to `bulbul:v3` / speaker `shubh`. **If a Sarvam (or any
  provider) call fails with an opaque error, read the actual response body
  before guessing -- `raise_for_status()` alone throws away the message
  that would have explained it in one line.**
- Phase 7 (turning on the real Groq LLM for the first time): the SAME class
  of bug as bulbul:v2, on a different provider, caught the same way -- by
  the app's own live `validate_roster()` check, not by trusting docs.
  `groq_llm_model` (`llama-3.3-70b-versatile`) and `groq_llm_fast_model`
  (`llama-3.1-8b-instant`) were BOTH gone from the real account's
  `GET /v1/models` the moment a real key was added, despite one Groq docs
  page (fetched live, same day) still listing both in a "Production Models"
  table -- account-scoped `/models` is ground truth; a docs page can be
  stale or simply wrong. Also worth recording: the replacement models
  (`openai/gpt-oss-20b`, `qwen/qwen3.6-27b`, etc.) are reasoning-style
  models that emit `<think>...</think>` preamble on a plain unconstrained
  call -- would have silently broken `router.py`'s `json.loads(response.text)`.
  Verified before picking one: `response_format: {"type": "json_object"}`
  (already sent whenever `complete(..., json_mode=True)` is called -- see
  `llm_client.py`) keeps the output clean JSON with no leakage, confirmed
  against this project's actual router prompt, not a generic test message.
  Fixed by moving `groq_llm_model` to `openai/gpt-oss-20b`, verified against
  a live `validate_roster()` pass and a real `run_turn()` call end to end
  (`"when will my delivery arrive"` correctly classified `order_lookup` at
  0.95 confidence -- a phrase the mock's keyword list would likely have
  missed, real evidence the real LLM is actually being used, not just
  configured). **Whenever you add credentials for a provider that was
  previously running on mocks, re-verify its configured model IDs against a
  live account before trusting they still work -- this is now the second
  provider roster that rotated out from under this project's own defaults.**
- `gr.mount_gradio_app(app, build_demo(), path="/ui")` without `root_path="/ui"`
  loaded fine in a browser but broke on the first real interaction: Gradio's
  own frontend JS calls its API (`/gradio_api/queue/join`, `/gradio_api/upload`,
  ...) relative to whatever root it thinks it's served from, and without
  `root_path` it assumes the site root, so those calls 404 against the real
  mount at `/ui` while `/ui/assets/*` static files and the initial page both
  return 200. Invisible to every automated check that existed at the time —
  `TestClient.get("/ui/")` only checks the initial HTML's status code, never
  a follow-up API call — and only surfaced when the user actually opened the
  deployed URL in a real browser and hit "connection error". Root-caused via
  `railway logs`, fixed by passing `root_path="/ui"` to `mount_gradio_app`,
  and given a regression test that regex-greps the embedded `"root"` JSON
  config out of the served HTML and asserts it ends with `/ui`
  (`test_ui_declares_its_own_mount_path_as_root` in `test_ui_harness.py`) —
  the same "shallow check passes, real client interaction doesn't" lesson as
  the Docker order_api bundling bug, just on the browser side this time.

## Working conventions

- **Run tests from the repo root**: `pytest -q`. `conftest.py` wires
  `packages/order-contracts`, `services/order-api`, and `services/gateway`
  onto `sys.path` — no editable installs needed.
- **Lint**: `ruff check services packages tests scripts` before every commit.
- **PYTHONPATH for running a service directly** (not through pytest) needs
  all three paths manually — see the README's platform-specific setup
  sections. This is the one rough edge of the monorepo layout.
- Every provider (`OrderClient`, `LLMClient`, `ASRClient`, `TTSClient`) follows
  the same pattern: an ABC, a real implementation, a mock selected when
  credentials are absent, and a `get_x_client()` / `reset_x_client()` module-
  level singleton pair. Follow this pattern for any new external dependency.
- Commit messages should explain *why*, especially for bug fixes — several
  existing commits document a bug's root cause in the message body, not just
  "fix bug." Keep that habit; it's what makes `git log` useful later.

## What phase 4 built

`app/audio/` — new, and the only new caller of `run_turn()`:
`transport.py` (the `AudioTransport` ABC), `websocket_transport.py` (the
browser-mic implementation over FastAPI's `WebSocket`), `call_loop.py`
(orchestrates `ASRClient` → `run_turn` → `TTSClient`, streams TTS audio out in
chunks, and implements barge-in), and `confirmation.py` (turns a spoken
"yeah go ahead" into the `bool` `run_turn`'s `resume_value` expects — the text
endpoint never needed this because its caller already passes an explicit
`resume`). `/ws/call` in `app/main.py` wires it up. The graph itself did not
change, per the plan — `app/graph/` is untouched.

Barge-in is generation fencing, not true preemption: every inbound utterance
bumps a counter, and both the in-flight turn and the outbound audio-chunk loop
check it after every await point, discarding (not sending) a superseded
turn's output. This is deliberate — `run_turn` runs synchronously on a worker
thread via `asyncio.to_thread`, and Python cannot forcibly cancel a running
thread, so the correct move is fencing the *output*, not trying to cancel the
*work*. See `test_audio_call_loop.py` for the concurrency tests this made
possible to write deterministically (a `FakeAudioTransport` driven by asyncio
primitives, no real timing races) vs. `test_ws_endpoint.py`'s real
`/ws/call` wire-protocol tests (happy-path only, on purpose).

## What phase 5 built

`app/guardrails/` — `patterns.py` (ordered keyword/phrase tables: self_harm,
violence, illegal_activity, prompt_injection, self_harm checked first so it
can never be shadowed by a lower-severity match) and `moderation.py`
(`check_input`/`check_output`, both the same deterministic keyword check —
see its module docstring for why this project doesn't use an LLM safety
classifier). Deliberately does NOT try to catch out-of-scope chit-chat
("what's the weather") — the router's own `fallback` bucket has handled that
since phase 2; this module exists for what the router was never designed to
catch: safety-relevant input and prompt injection, screened independently of
whatever the intent classifier decides.

Wired into the graph itself (`app/graph/`), not the transport layer, so both
`/call/turn` and `/ws/call` get it for free:
- `nodes/input_guardrail.py` — a THIRD concurrent branch off `START`
  alongside `intent_router` and `sentiment_monitor` (rule 4: own state key,
  `input_guardrail_verdict`). Also exports `guardrail_block_patch()`, the
  actual enforcement point — see rule 8 and the phase-5 bug entry above for
  why enforcement lives there and not in `route_edge`.
- `nodes/output_guardrail.py` — a plain node between `composer` and `END`,
  defense-in-depth on the composed reply. Should never fire today (every
  domain agent's reply is templated), but completes the "both boundaries"
  half of the RAI requirement.
- `order_lookup.py`/`reschedule.py`/`address_change.py`/`fallback.py` each
  call `guardrail_block_patch()` as their first line.

`scripts/eval_fairness.py` — two checks, not one. (1) A language-invariance
regression check that runs TODAY with zero fixtures: the same English text,
tagged with different `language` values, must produce an identical router
decision and sentiment score, because the architecture's whole bet is that
language is only ever used at the TTS boundary. This is a real, passing
check, not a placeholder — see `test_eval_fairness.py`. (2) A manifest-based
cross-language accuracy/sentiment corpus eval, honestly reported as "no
fixtures yet" until real per-language ASR-translated transcripts exist (see
`fixtures/fairness/README.md`) — same honesty pattern as `eval_asr.py`
before `fixtures/audio/` had real recordings. Don't hand-write fake
"translated-sounding" text for that manifest; it would measure the author's
assumptions, not a real disparity.

## What phase 6 built

`app/ui/` — `harness.py` (pure logic: `run_text_turn`/`run_audio_turn` mirror
`app/audio/call_loop.py`'s per-turn shape but synchronous/one-shot, since a
test UI only ever has one utterance in flight; both reuse
`interpret_confirmation()` from phase 4 rather than a permissive
`resume: bool`) and `blocks.py` (thin Gradio glue over it). Mounted at `/ui`
in `app/main.py` via `gr.mount_gradio_app`.

`gr.Blocks(..., analytics_enabled=False)` is load-bearing. Without it,
`Blocks.__init__` starts a background thread that calls `api.gradio.app` and
`huggingface.co` — confirmed by watching it happen — which is a real network
call from what CLAUDE.md's hard rule says must be an offline suite.
`test_build_demo_disables_analytics_and_never_calls_gradios_telemetry` in
`test_ui_harness.py` patches `gradio.analytics.version_check`/
`initiated_analytics` to explode, so this can't silently come back.

`fixtures/audio/` went from empty to real: 20 clips, 4 languages (`hi-IN`,
`ta-IN`, `te-IN`, `bn-IN`), sourced from Google FLEURS (CC-BY-4.0) rather
than self-recorded — real human speech, with a genuine English reference
translation for each clip via FLEURS' shared `id` field against its `en_us`
config (same underlying FLoRes-101 sentence, professionally translated, no
MT step). See `fixtures/audio/README.md` for the sourcing method if this
needs extending to more languages — same trick, FLEURS covers most major
Indian languages. Real Sarvam scoring has now been run (a `SARVAM_API_KEY`
was added) -- see README.md's "Evaluation results" section for the actual
per-language WER and, importantly, why translation WER against a single
reference reads worse than the translations actually are (spot-checked by
hand: several "high-WER" hypotheses are fully correct translations phrased
differently from the FLoRes-101 reference). 5 samples/language is enough to
prove the pipeline and catch something badly broken, not enough to make a
confident per-language cut decision -- don't treat it as more than that.
`eval_asr.py --provider mock` against these files still proves the pipeline
runs on real audio for zero cost, but its own WER/CER numbers stay
meaningless by construction (`MockASRClient` just decodes WAV bytes as UTF-8
text) -- never confuse that smoke test's numbers with the real ones above.

**`fixtures/fairness/` is still empty, and FLEURS doesn't fix that.** Those
recordings are general-domain (geography, history) — every one would
correctly fall back as out-of-scope regardless of language, which would
make the fairness corpus's "accuracy" trivially 100% and measure nothing.
That eval needs *in-domain* (delivery/logistics) per-language utterances;
no open dataset of those is known to exist. This is a real, still-open gap,
not a solved TODO — see `fixtures/fairness/README.md`.

Nothing in this project trains or fine-tunes anything — Sarvam/Groq are
consumed as hosted inference APIs only. Both eval scripts benchmark a
third-party model's fitness for this use case, not a model this project
owns; don't add a training step trying to "complete" the eval story.

## What phase 7 built

Fixed the Dockerfile bug in the "Bugs already found" section above (bundled
`services/order-api` into the gateway image), then verified for real:
`docker build` + `docker run` + `curl -X POST .../call/turn` against
`gateway:ci` locally, checking the actual JSON response, not just that
`/health` returned 200. Strengthened `.github/workflows/ci.yml`'s
`gateway-image` job the same way — it only ever checked `/health` before,
which is exactly how the bundling bug went undetected.

**The full split is actually live**, not just Docker-verified locally --
order-api on Vercel (Upstash-backed, real persistence confirmed with a
genuine write-then-read-back cycle) and the gateway on Railway (pointed at
the live Vercel URL via `ORDER_API_BASE_URL`, `SARVAM_API_KEY` set too). See
README.md's live URLs near the top. Getting there for real surfaced three
more genuine bugs beyond the Docker one, all logged above in "Bugs already
found": two stale Vercel `vercel.json` config issues (`runtime`/`memory`
fields, then a `rewrites` rule) and one real application bug
(`ensure_order_id` not handling `OrderAPIUnavailable`, caught by an actual
Vercel cold start on the first live cross-service request). None of the
four were guessable from local dev/CI alone -- each needed the real
deployed thing to actually exist before it could surface.

**HF Spaces was attempted and deliberately dropped, not left undone.**
Building the deploy package (a Space needs `Dockerfile` + a frontmatter'd
`README.md` at ITS OWN repo root, which this repo's actual root can't
double as without wrecking the GitHub README -- so it's a purpose-built
file set pushed via the `huggingface_hub` API, not a plain `git push` of
this repo) went fine. Creating the Space itself failed: `HfApi().create_repo(...,
repo_type="space", space_sdk="docker")` returned `402 Payment Required` --
Hugging Face now requires a PRO subscription to run ANY Docker or Gradio
Space, even on free `cpu-basic` hardware; only static (HTML) Spaces are
still free. This directly contradicts what this project's own README and
cost model said before anyone actually tried to create the Space live. User
chose to skip it rather than subscribe. README.md and the cost model table
are corrected accordingly -- don't re-add "HF Spaces is a free fallback"
anywhere without re-verifying HF's current pricing first.

Two honest, still-open gaps outside the 7-phase scope, not silently
resolved: `fixtures/fairness/`'s corpus (real in-domain per-language
recordings — see its README) and expanding `fixtures/audio/` beyond the
current 4 languages/5 samples each if broader ASR coverage is wanted later.
