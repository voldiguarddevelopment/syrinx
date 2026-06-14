# CLAUDE.md — Syrinx project constitution (read fully before any task)

You are building **Syrinx**, a local, Rust-served neural TTS + zero-shot
voice-cloning engine. A Rust workspace of focused crates implements a deterministic
text frontend, a Rust inference runtime over an adopted open base (Path A), an
editable prosody control surface, speaker-latent blend/morph, paralinguistic
control, streaming, and an OpenAI-compatible server. This file is the standing law.
`DESIGN.md` is the full plan; `plan.md`/`spec.md`/`list.md` are the derived task trio.

Context is thrown away every pass and re-derived from disk. **Disk and git history
are the only memory.** Re-read the relevant files at the start of work; write your
conclusions to disk, not just into your reply. No in-context state survives a pass,
so corner-cutting in one pass cannot poison the next.

---

## Non-negotiable rules

- **No stubs, no simplified implementations, no fake passes.** If you cannot
  implement the real thing, log a blocker and stop — a green that isn't real is the
  single worst outcome in this system. Never weaken a detector, a test, or a gate to
  get past it.
- **Tests freeze at red-pass.** Never edit a frozen file (test files, `criteria_map`,
  the detectors in `.ratchet/detectors/`) in a green phase.
- **Judgment is deterministic:** the compiler, the frozen tests, the frozen checker,
  and mutation decide — not your opinion. "It sounds right" is never a verdict.
- **One task per worktree; tasks are small by design.** If your context fills up, the
  task should have been split.
- **Fix documents before code:** reconcile plan.md, spec.md, list.md before building.
- **Log all failures with raw tool output,** not paraphrases.
- **IDs are immutable.** Splits add suffixes; nothing is renumbered or deleted.

---

## THE BUILD SCOPE — what the loop builds vs. what is blocked-on-human

Syrinx is an ML system. A large fraction of its work — training, GPU inference,
numerical parity against a Python reference, corpus collection/annotation, and
**perceptual** judgments ("sounds natural", "intended emotion", SIM-o/MOS) — is **NOT
expressible as a frozen-test + mutation gate**. The Ratchet loop must NEVER attempt
those tasks, because the only way to "pass" them without the model/GPU/data/ears is
to fake a green — the one outcome this system exists to prevent.

Those tasks are marked **`status: blocked`** in `list.md` with a human/GPU blocker.
**If you are ever handed a blocked task: do not implement it, do not fabricate an
eval result, do not stub a metric. Re-confirm the blocker, log it, and stop.** They
are deliberately off the autonomous path, exactly like a manual prerequisite.

**The loop BUILDS (deterministic, frozen-test-gateable) — the engineering substrate:**

- **Phase 0 (partial):** workspace scaffold, CI wiring, the eval-harness *skeleton*
  (runs against a stub, emits a metrics JSON), the frozen-eval-set *mechanism*
  (immutable + checksummed), the license-screen matrix doc, ARCHITECTURE/CLAUDE/ethics
  docs. **NOT** the base-model A/B bench or the Python reference run (those need real
  models + a GPU → blocked).
- **Phase 1 (entire):** the deterministic text frontend — normalization, numeric/date
  expansion, lexicon/acronym overrides, the G2P/phonemizer interface, custom
  pronunciation maps, the heteronym resolver, the SSML parser, punctuation→prosody,
  context windowing, pacing/breath intervals, the test suite, the frontend→LM contract.
  This is the deterministic Rust win; everything here is golden-file / unit-test gated.
- **Phase 3 (partial):** the editable prosody-plan **data model** (T-03.01, serialize /
  round-trip), volume-automation curves as a deterministic transform. **NOT** the
  predictors, emotion steering, or anything judged by ear → blocked.
- **Phase 7 (partial):** the lip-sync timeline export (phoneme timestamps → viseme
  mapping, deterministic). **NOT** TTFB/RTF/telephony/noise targets (need the running
  model → blocked).
- **Phase 8 (partial):** the OpenAI-compatible `/v1/audio` server *scaffold* (routes,
  request/response schema, streaming endpoint shape) and the docs. **NOT** the NovaFox
  wiring or release-with-model-card (need the whole engine → blocked).

**Blocked-on-human (NOT loop tasks):** Phase 2 (Rust inference parity, quantization,
SIM-o, watermark detection — needs weights + GPU + Python reference), most of Phase 3
(predictors/emotion/perceptual), Phase 4 (blend/morph — perceptual), Phase 5 (corpus +
annotation + LoRA training, except the taxonomy/sourcing **docs** T-05.01/T-05.02),
Phase 6 (adversarial disentanglement training), most of Phase 7, and Phase 8 wiring.

When the substrate is green and a human has done the ML work (trained/ported the
model, built the corpus), the blocked tasks can be unblocked and given criteria that
*are* gateable against the then-existing artifacts.

---

## Crate contracts (the workspace, DESIGN §5)

Each crate owns one responsibility; cross-crate types flow through explicit,
versioned interfaces (never reach into another crate's internals):

| Crate | Responsibility |
|-------|----------------|
| `syrinx-frontend` | normalization, G2P, SSML, lexicon, heteronyms, context windowing |
| `syrinx-core` | tensor-ops glue, weight loading, quantization, device mgmt |
| `syrinx-lm` | AR semantic LM forward pass + paralinguistic tokens |
| `syrinx-speaker` | speaker encoder, embedding store, blend/morph, attributes |
| `syrinx-acoustic` | flow-matching decoder (DiT + ODE solver), chunk-aware streaming |
| `syrinx-vocoder` | HiFi-GAN/Vocos waveform synthesis, 48kHz/8kHz paths |
| `syrinx-prosody` | editable prosody-plan model + override API |
| `syrinx-stream` | packet streaming, ring buffer, `cpal` out, TTFB path |
| `syrinx-serve` | Axum server, OpenAI-compatible `/v1/audio`, watermarking |
| `syrinx-eval` | MOS/SIM-o/WER/latency harness, frozen-eval-set runner |
| `syrinx-cli` | local runner / dev harness |

The deterministic frontend (`syrinx-frontend`), the prosody data model
(`syrinx-prosody`), the eval-harness skeleton (`syrinx-eval`), and the server
scaffold (`syrinx-serve`) are where the loop does its work. The model crates
(`syrinx-lm`, `syrinx-acoustic`, `syrinx-vocoder`, `syrinx-speaker`, `syrinx-core`
weight loading) are human-and-GPU territory — their tasks are blocked.

---

## Where frozen tests live (read before every RED — this is load-bearing)

The harness detects a test file **only** by the repo-root `tests/` prefix
(`is_test_file` = path starts with `tests/`). A RED phase whose tests land
ANYWHERE else (a workspace member's `crates/<crate>/tests/`, or unit tests inside
`crates/<crate>/src/*.rs`) produces **"red phase produced no test files under
`tests/`"** and the task stalls. So:

- **Every frozen test file goes at the repo-root `tests/*.rs`** (e.g.
  `tests/normalize_golden.rs`), and its golden data under the repo-root
  `tests/golden/...`. NOT under `crates/*/tests/`.
- A repo-root `tests/*.rs` calls into a member crate's API
  (`use syrinx_frontend::normalize::normalize;`), so the GREEN phase adds that
  member as a dependency of the root package. The root `Cargo.toml` is BOTH a
  `[package]` (so `cargo test` runs the root `tests/`) AND a `[workspace]` — keep
  it that way; do not turn it into a virtual (package-less) workspace.
- The `COVERAGE:` line still maps each criterion to the test names you wrote in
  those repo-root files.

## Passing the mutation gate (re-read every RED and GREEN phase)

The mutation gate flips operators (`==`→`!=`, `>=`→`>`, `&&`→`||`, `+`→`-`, …) in
your implementation and requires the frozen tests to KILL every mutant. Two rules
follow; ignoring either is the usual cause of a stalled task:

- **GREEN: write the MINIMAL implementation that satisfies the frozen tests.** Every
  operator and branch you write must be killed by a frozen test. If you add validation,
  precedence, bounds, or boundary logic the frozen tests do not exercise, a mutant will
  survive and the gate rejects the pass — and you cannot edit the frozen tests to fix
  it. When in doubt, write less; do exactly what the criteria require, nothing more.
- **RED: write tests that pin every branch and both sides of every boundary.** For each
  comparison or boolean the implementation needs, assert behaviour on both sides (at
  the threshold and just past it, true and false). A happy-path-only test leaves
  operator mutants alive and dooms the green phase. Cover every criterion with
  boundary-exercising assertions.

---

## What "done" means

A task is done when its frozen tests pass honestly, the checker is clean, mutation
confirms the tests defend the real code, and every acceptance criterion maps to a
passing test — never because you believe it is. A blocked task is "done" only when a
human removes its blocker and it earns a real gate. When in doubt: re-read from disk,
do the smallest honest thing, write the result down, and let the next pass check you.
