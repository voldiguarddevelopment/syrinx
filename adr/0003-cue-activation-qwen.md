# ADR-0003 — C4.2 re-scoped to the Qwen capability profile

Status: **PROPOSED** (acceptance is a human act — ADR-0001 §7 / C0.2′)
Date: 2026-09-06
Supersedes: the acceptance criterion of **C4.2** in
`docs/upgrades/SYRINX_UPGRADE_expressive_control.md`. The task id is unchanged; IDs are
immutable and this rewrites the AC under it.

## The problem: C4.2's criterion cannot fire on the shipping path

C4.2 gates **event activation ≥ 0.85 on `Inline::Open` backends**. After the Qwen
redirection (`docs/LICENSES.md`, 2026-09-06) that criterion is orphaned:

- the only `Inline::Open` backend is `fish-s2-pro`, which is research-licensed and can
  never ship;
- every Qwen checkpoint is `inline = none` with `event = unsupported`, so an event cue is
  filtered by `caps.can_express` in `pass_hoist` before a render happens.

`tests/real_cue_activation.rs:73` hard-wires `const BACKEND: &str = "fish-s2-pro"`.
Running the original 2.4 h sweep would certify a threshold for a backend we do not ship.

**0.85 was never measured.** It was written into the ledger before any run existed, as a
plausible-sounding number. Ledger A25 records the only data: a 2-case pilot, both *not
activated*. Retiring it is correcting the record.

## Decision

### 1. The fish-scoped criterion is retained, not deleted

`evaluate_activation`, `Thresholds`, `Measurement`, `Cell` and `Violation` keep their exact
signatures and behaviour. `tests/cue_activation_gate.rs` is frozen, calls
`evaluate_activation` at five sites, constructs `Measurement` as a struct literal and
asserts `limit == 0.85`. It stays green, untouched, and `real_cue_activation` remains the
**research-path** run. CLAUDE.md's "deprecated does not mean untested" applies.

Everything below is **additive**. No frozen file changes, so no unfreeze is needed.

### 2. No activation floor replaces 0.85

Not a smaller one either. The honest content of the current evidence is *one* cue out of
three moving the audio, on *one* sentence, with *one* preset voice. That supports no floor.

A floor nothing passes is a broken gate; a floor everything passes is decoration; a floor
chosen to sit between the two is threshold-fitting, which CLAUDE.md forbids in spirit —
"never weaken a detector, a test, or a gate to get past it" cuts both ways.

What replaces it is **a falsifiability gate plus a ratchet**: assert the things whose
failure means the *instrument* is broken, and record the level so it can only rise.

### 3. Four clauses (C4.2′)

Pre-registered here, before the run.

#### Clause A — the negative controls hold (hard assert; failure voids the run)

**A1 — pipeline control.** On `qwen3-0.6b-customvoice`, the cued and plain renders at the
same seed must be **bit-identical**. Byte equality, not a p-value: 2 renders, and a
stronger statement than any test.

*This is a control on our own code, not on the model, and the ADR says so rather than
overclaiming.* `prompt.rs:286` is `honors_instruct() = !"0b6".contains(&tts_model_size)`,
and `build_custom_voice` filters the instruct out on that basis — the 0.6B renders
identically because **we never send the string**. It proves the harness reports "no
activation" when no instruction reaches the model. It does not establish anything about
how a model responds.

**A2 — A/A calibration.** Render 2n plain takes, split into two disjoint groups of n, and
run the activation test. Observed activations must be ≤ `ceil(α·m)`. This is the
model-real analogue of `cue_activation_measure.rs`'s false-positive check, which today
runs only on synthetic LCG signals.

**A3 — sham control.** A delivery-neutral instruction of comparable token length must
**not** activate at the corrected α.

*This clause is mandatory and it is the one that makes the rest mean anything.*
`assemble_text_mode` prepends the instruct block as text tokens, so cued and plain differ
in prompt **length and content**, not only in meaning. At a fixed seed a longer prompt
gives a different AR trajectory whether or not the model attaches meaning to the words. A
model treating the instruct as pure noise would still reject the cued-vs-plain null. The
old design could not distinguish "this instruction steers delivery" from "an instruction
of this size perturbs the prompt".

**A3 has been run** (2026-09-06, `renders/2026-09-06-instruct-lang/`, n=8, 120 renders).
It passed: no sham clears the corrected acoustic threshold and the largest sham effect on
the judge is |t| = 2.07.

| | happy | sad | angry |
|---|---|---|---|
| sham-en vs plain | p=0.3164 | p=0.7453 | p=0.0325 |
| sham-zh vs plain | p=0.4059 | p=0.9091 | p=0.2862 |

So the confound is measured and rejected, and `[sad]`'s p=0.0019 stands as a measurement
of content. The `sham-en`/angry cell (p=0.0325, *larger* than the real cue's 0.7417 on
that case) is inside correction but is why the arm stays in every run rather than being
declared settled by this one.

#### Clause B — instrument sensitivity (hard assert, once earned)

One **pre-registered sentinel** case must activate at `p ≤ α/m`. Earned by this procedure,
recorded here so the run cannot become a search for one:

1. First run measures all `m` sentinels and enforces nothing (the existing
   `SYRINX_CUE_ACTIVATION_ENFORCE` pattern).
2. A case clearing `p ≤ α/m` must **replicate on a disjoint seed block** before it may be
   pinned. This turns a post-hoc pick into a pre-registered replication.
3. **If nothing clears and replicates, that is recorded and clause B is not enabled** — the
   gate ships with A + C + D. This outcome is written down *in advance* as acceptable.

#### Clause C — WER veto (hard assert, unchanged)

`wer_delta ≤ 0.5` on every (backend, kind). Backend-agnostic, already alive, already fires
on Qwen. The only part of the original C4.2 that survives intact.

#### Clause D — ratchet, not floor

`tests/golden/cue_eval/qwen_activation_baseline.json` records per sentinel: `activated`,
p-value, effect, and full provenance (checkpoint, device, dtype, voice, seed block, n, α,
commit). The gate fails when a case activated in the baseline is not activated now, **on
matching provenance**. On mismatched provenance it reports and does not assert — a number
from a different box is not a regression.

Ratchet on the **boolean only** unless GPU bit-reproducibility across runs is verified on
the box; `tests/real_qwen_seed.rs` has verified it on CPU only.

### 4. Multiple comparisons are corrected, always

"At least one cue activates" is **vacuous without correction**: at 33 measurable cases and
α=0.05, a pipeline emitting pure noise passes with probability 1 − 0.95³³ ≈ **0.82**. Every
clause uses Bonferroni over the run's full comparison count, and `min_n_for_alpha` — which
already exists — is called with the *corrected* α to set the seed-count floor.

### 5. `not_applicable`, never a fabricated zero

A cue whose kind is `unsupported` on a backend lowers to no instruct, so cued and plain are
the identical request and render bit-identically. Reporting that as `activation_rate: 0.0`
is a fabricated measurement of a channel that does not exist. It is reported as
`not_applicable` with a count. This covers all 12 event cases on every Qwen checkpoint.

### 6. `caps.toml`'s `honored` is narrowed in prose, not in value

`caps.toml` defines `honored` as "verified to affect the audio", and cites
`prompt.rs::honors_instruct` plus a test that the instruct reaches the prompt. That
establishes the instruction is **delivered**, not that it **affects the audio** — and the
n=8 result (2 of 3 cues null) is direct evidence the claim is stronger than its source.

The `source`/`notes` prose is corrected. The `Support` **value is not changed**:
`Support::Accepted` lowers exactly like `Unsupported` (`caps.rs:34-36`), so demoting it
would silently switch the cue layer off, and `tests/expressive_api.rs:155` is frozen on
`"honored"`.

## What C4.2′ cannot establish — stated so it is never claimed

- **Not that any cue produces its named emotion.** The permutation test is direction-blind
  by construction. C4.2′ is an *instrument* gate, not a fidelity gate. Fidelity stays a
  human listening call per CLAUDE.md.
- **Not that the cue system works in general.** Six cases, one sentence each, one preset
  voice, one checkpoint, one language, one device. The frozen set has three sentence
  templates per language, so sentence is confounded with placement.
- **Not that `caps.toml`'s `honored` is true.** With the sham arm it establishes that an
  instruction *with delivery content* changes the audio more than a delivery-neutral one of
  similar length — a real claim, and much stronger than what exists, but about the pipeline
  and model jointly.
- **Nothing about event cues on any shipping backend.** `not_applicable` is the correct and
  permanent answer until a backend with an event channel is adopted.
- **Nothing transferable off this box.** Provenance-keyed by design.

## Budget

7.75 s/render measured. Renders per case = 2n (plain, serving the control arm and the A/A
split) + n (cued) + n (sham) = **4n**.

| tier | cases | n | renders | GPU | asserts |
|---|---|---|---|---|---|
| gate | 6 sentinels + 3 on 0.6B | 8 | 198 | ~26 min | yes |
| sentinel replication (first run) | 1 | 8 | 32 | ~4 min | no |
| full sweep (report only) | 33 | 8 | 1056 | ~2.3 h | no |

n=8, α=0.05, Bonferroni to α/6 = 0.00833; `min_n_for_alpha(0.00833) = 5`, so n=8 has
headroom and stays numerically comparable to the recorded n=8 results.

## Consequences

- A new opt-in runner `tests/real_cue_activation_qwen.rs`; `real_cue_activation` stays as
  the fish research run.
- `evaluate_contrast` is **pure**, so it gets a model-free frozen companion
  (`tests/cue_contrast_gate.rs`) and the *decision logic* is CI-gated on every board even
  though the numbers are not.
- Ledger amendment **A27** rewrites C4.2's AC under its immutable id.
