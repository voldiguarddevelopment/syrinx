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
- **No bracket cue text and no speaker token may ever reach a backend as literal text.** No SSML tag either —
  no cue markup of any kind — ever, in any dialect, however malformed. `\[` is the only
  way to speak a literal bracket. This is enforced by property test
  (`crates/syrinx-cue/tests/invariant_property.rs`), not by review: a failure there is a
  **release blocker**, never a test to relax. The corollary is that every unescaped
  `[...]` is cue syntax (ADR-0001 §9.1 / **D5**) — `array[0]` lowers to `array` unless
  escaped, which is the accepted price of an invariant that is total instead of
  best-effort.
  **One named exception, and only one:** `syrinx_cue::legacy_emotion::parse_tagged`, the
  deprecated CosyVoice-era parser, whose weaker semantics are pinned by the frozen
  `tests/emotion_tags.rs` (ADR-0001 §11 / **D7**). It is quarantined and deprecated; no new
  code may call it, and the exception dies with CosyVoice. Do not add a second exception —
  narrow the rule only by ADR.

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
| `syrinx-frontend` | normalization, G2P, lexicon, heteronyms, context windowing |
| `syrinx-cue` | **the sole owner of expressive-cue syntax**: bracket cues, SSML subset, the `CueDoc` IR, the label vocabulary, backend `ControlCaps`, and every lowering pass |
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

The deterministic frontend (`syrinx-frontend`), the cue layer (`syrinx-cue`), the
prosody data model (`syrinx-prosody`), the eval-harness skeleton (`syrinx-eval`), and
the server scaffold (`syrinx-serve`) are where the loop does its work.

**SSML lives in `syrinx-cue`, not `syrinx-frontend`** (ADR-0001 §10 / **D6**, accepted
2026-09-03). Both authoring syntaxes — bracket cues and the SSML subset — must produce the
one `CueDoc` IR. A second producer of that IR would mean two places to enforce the hard
invariant below and two scoping implementations to keep in agreement, so `syrinx-frontend`
*consumes* `CueDoc` and never parses cue syntax itself. No backend crate may parse cue
syntax either. The model crates
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

## Model direction — Qwen only (Fish and CosyVoice are both deprecated)

**Qwen3-TTS (`syrinx-qwen`) is the TTS path**, decided 2026-09-06 on licensing.

The deciding fact is in `docs/LICENSES.md`: **every Qwen3-TTS checkpoint is Apache-2.0**,
while **every Fish checkpoint is research-only** (`s2-pro` under Fish Audio's own research
licence, `openaudio-s1-mini` under CC-BY-NC-SA-4.0). Fish cannot ship. Qwen can — use,
modify, distribute, sell and fine-tune, attribution and NOTICE preserved. No amount of
engineering changes that, and it had gone unrecorded while the effort went into Fish.

- **`syrinx-fish` is deprecated.** Its code stays in-tree and its results stand — including
  the two real codec defects found and fixed on 2026-09-04 (a bf16-rounded RoPE table and
  missing sliding-window attention), which is exactly the evidence that would be lost by
  deleting it. It gets **no new work**. It is research-only and must never be a shipping
  path.
- **CosyVoice2 / CosyVoice3 remain deprecated**, as before: code in-tree, results stand, no
  new work, `cv2`/`cv2e2e`/`cv3`/`cv3e2e` deliberately UNSET in `scripts/test-all.env` so
  they report **SKIP**, never FAIL. Do not "fix" a CosyVoice SKIP by pointing it at weights.
- **Deprecated does not mean untested.** Unlike CosyVoice, Fish has working parity fixtures
  and green gates, and the crate is still compiled into the workspace. Its *cheap* gates
  stay on the board so in-tree code cannot rot unnoticed; only the heavyweight runs are
  opt-in. Deleting a passing gate for a crate that still builds discards evidence and buys
  nothing.
- **`syrinx-stt` (Whisper, Apache-2.0) stays active** as the native WER oracle — now scoring
  Qwen renders. `syrinx-eval`'s Qwen path uses it directly, with no Python at inference.
- **Any new backend is a licence question first.** Add its row to `docs/LICENSES.md` before
  building on it. `ResembleAI/chatterbox` (**MIT**, 23 languages, voice cloning) is the
  leading candidate for a second family precisely because its licence is clean; it is a
  candidate, not an adopted path.

## Verifying the build (on the model box)

Verification is hardware-bound: the `real`-feature binaries need a GPU box with the
weights + parity fixtures present. They SIGILL on the dev box (a pre-existing CPU/candle
issue), so OFF-box you get compile-checks only — never a real pass. One command runs the
whole verification on the box:

    ./scripts/verify.sh --download        # first time — also pulls the Fish weights from HF
    ./scripts/verify.sh                   # weights already in place

It chains, in order: preflight → `cargo build --features real` (compile gate) →
create/read `scripts/test-all.env` → download the Fish weights (`--download`) →
`gen-fish-ref.py` (parity fixtures) → run every group (CV2 · CV3 · Fish s1-mini ·
Fish s2-pro · voice · emotion) → a `PASS / SKIP / MISSING / FAIL` board. Exit 0 unless
something FAILED; SKIP = that group's weights/fixtures aren't configured, MISSING = the
test file isn't built yet.

**GPU prerequisite (Blackwell / RTX 50-series boxes):** run
`./scripts/setup-cuda-blackwell.sh` ONCE before any `--features cuda` build. candle 0.8.4
pins cudarc 0.13.9, which refuses any CUDA newer than 12.8, while Blackwell (`sm_120`)
requires at least 12.8 — so the distro's CUDA 13.x cannot build this workspace, and no
candle bump fixes it (cudarc 0.17.8 still stops at 13.0). The script installs a local CUDA
12.8 + gcc 14 under `$HOME` (no root) and applies a 6-declaration host-header fix for
glibc >= 2.41. `scripts/test-all.env` then exports `CUDA_ROOT` / `NVCC_CCBIN` /
`CUDA_COMPUTE_CAP=120`.

Two steps `verify.sh` cannot do for you (it prints exactly when each is needed):
- **Fill `scripts/test-all.env`** (copy from `.env.example`) with this box's weight +
  fixture paths. Unset groups SKIP — they never FAIL on a partial box.
- **Run `scripts/gen-fish-ref.py` once** to dump the Fish parity reference. Its model-load
  is wired (it calls the reference fish-speech's own `DAC.from_indices` /
  `DualARTransformer.forward_generate` — never a reimplementation), but it needs the
  reference package, which is NOT vendored: `~/refs/fish-speech`
  (github.com/fishaudio/fish-speech, verified at `befe400`) plus its own env —
  `cd ~/refs/fish-speech && uv venv --python 3.12 .venv && uv sync --extra cpu` (torch
  2.8.0 has no cp314 wheels and `descript-audiotools` pins `protobuf<3.20`, which only
  uv's override resolves). `verify.sh` prints the exact command but never runs it: the
  s2-pro slow-AR anchor is a ~19 GB CPU/f32 job, so run it under `run-isolated.sh`,
  alone. `--codec-only` gets the cheap ~2 GB codec anchor first. No dump → the Fish
  *parity* tests SKIP; the Fish *e2e smoke* tests and all CV2/CV3 parity still run.

Narrower entry points (all read the same `test-all.env`):
- `./scripts/test-all.sh [selectors] [--exclude S] [--dry-run]` — the suite alone. A
  **selector** is a group, a **family**, or a bare test name, resolved in that order:
  `test-all.sh fish` runs one model family, `test-all.sh cue qwen` combines selectors,
  `test-all.sh real_fish_s2_e2e` runs a single test, `test-all.sh free` runs everything
  model-free. `--group/--family/--test` force the kind; `--list` shows both tables;
  `--dry-run` prints the selection without running it. `verify.sh` takes the same
  selectors. **An unknown selector is a hard error** — it used to run nothing and still
  print "all configured groups green" with exit 0, which is the one outcome this project
  cannot tolerate. Groups and families are defined once, in `scripts/test-groups.sh`,
  and shared by both runners so they cannot drift.
  **Opt-in tests** are deliberately in no group and no family, so no routine board and no
  full `verify.sh` ever fires them; `--list` prints them in their own table under the
  groups, and you reach one by naming it:
      ./scripts/test-all.sh --test real_cue_activation        # C4.2 certification run
      ./scripts/test-all.sh --test real_qwen_greedy_parity    # ~21 min, 1.7B CPU/f32
      ./scripts/test-all.sh --test real_qwen_serve            # ~12 min, 0.6B CPU/f32
      ./scripts/test-all.sh --test real_qwen_eval             # ~37 min, 1.7B CPU/f32
      SYRINX_FISH_BATCH_PARITY=1 ./scripts/test-all.sh --test real_fish_s2_batch_parity
  `real_qwen_greedy_parity` is the multi-frame greedy-decode anchor for the Qwen3-TTS
  generation loop — the only gate that covers the KV cache, position advancement, the
  per-frame talker→predictor handoff and the trailing-text schedule across many frames.
  Run it deliberately after ANY change to that loop. It left `GROUP_qwen_ckpt` because it
  alone took 21.6 min of that group's 21.6 min; the four remaining weight-backed Qwen
  gates take 23 s together. Opt-in is not optional: an unrun gate is a dead gate.
- `./scripts/run-fish.sh <s1-mini|s2-pro> "<text>" <ref.wav> [out]` — one synth; `--parity <variant>` runs that model's Fish tests.
- `./scripts/synth-samples.sh <variant> [--scale small|reply|chapter] [--lang L]` — batch-render the 610-sample corpus.

**Run heavy jobs isolated — `./scripts/run-isolated.sh <cmd>`.** The `real`-feature tests
are memory-hungry on CPU: `real_fish_s2_e2e` needs **~19 GB**, because the s2-pro weights are
BF16 on disk (9.1 GB) and the CPU parity path upcasts to F32 by design
(`s2/mod.rs`: *"CPU must stay f32 (parity); CUDA defaults to bf16 (fit)"*). Anything launched
from an editor's integrated terminal inherits that editor's systemd scope, and a scope's
default `OOMPolicy=stop` means one OOM-killed child tears down the whole scope — on
2026-09-03 this killed the VSCodium session mid-verify. The wrapper puts the job in its own
scope (`OOMPolicy=continue`), and `MEMMAX=24G scripts/run-isolated.sh …` caps it so it dies
at a bound you chose instead of starving the box. **Never run two `real`-feature suites
concurrently.** Do not "fix" this by disabling the OOM killer: with a 19 GB job the kernel
would thrash instead of killing, which is worse.

Full-coded ≠ verified: nothing is real until the box says so. Every offline-unconfirmable
numeric is marked `// PARITY:`; the Fish s2-pro EVA-GAN codec is the least-certain piece
and the first thing to scrutinize on-box.
