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

---

# AMENDMENT — PROPOSED, NOT ACCEPTED

**Status of this section: PROPOSED. Nothing here is adopted.** ADR-0003's own Status line is
unchanged and this section does not change it. Acceptance is a human act (ADR-0001 §7 /
C0.2′); no run, no test and no pass of this loop may perform it. The amendment is written so
that the decision can be made against numbers instead of prose.

Raised: 2026-09-11, from `renders/2026-09-09-c42-certification/FINDINGS.md`.
Code: `crates/syrinx-eval/src/contrast.rs` (`ContrastRule`, `compare_rules`), frozen by
`tests/cue_rule_comparison.rs`. Additive — `evaluate_contrast` is byte-for-byte unchanged
and remains the only shipped verdict.

## The question

`evaluate_contrast` calls a case **content-activated** only when the cue clears *both*
contrasts at the corrected alpha:

| leg | claim |
|---|---|
| `cue vs plain` | the cue changed the audio |
| `cue vs sham` | the change is content, not prompt perturbation |

The first certification run (n=9, α=0.05/24 = 0.00208) produced this:

| case | cue/plain | sham/plain | cue/sham |
|---|---|---|---|
| happy-leading | 0.0683 | 0.4792 | 0.0598 |
| **sad-mid** | 0.0038 | 0.0233 | **0.0001** |
| angry-trailing | 0.8677 | 0.8067 | 0.8300 |
| calm-mid | 0.1594 | 0.0233 | 0.0825 |
| whisper-leading | 0.0042 | 0.4792 | 0.2068 |
| **shout-mid** | 0.0182 | 0.0233 | **0.0000** |

Two cases separate from their sham decisively while their `cue vs plain` sits at 0.0038 and
0.0182, so the conjunction reports nothing. FINDINGS.md hypothesised that sham and cue move
the audio in **different directions** from plain, and that requiring `cue vs plain`
therefore re-admits the very confound the sham arm was introduced to remove.

**The two criteria, stated exactly** (both now computable from one run):

- `ContrastRule::PlainAndSham` — shipped. `cue vs plain` **and** `cue vs sham`.
- `ContrastRule::ShamOnly` — proposed. `cue vs sham` alone.

`PlainAndSham` adds a conjunct, so it is a strict sub-rule: its verdict is always a subset
of `ShamOnly`'s, and `RuleComparison::divergent` is the whole practical difference. On the
measured run: shipped `[]`, proposed `[sad-mid, shout-mid]`.

## The evidence, argued against itself

### What the data does support

**Cue and sham do not move the audio the same way.** This is the one part of the direction
hypothesis the data carries. `crates/syrinx-eval/examples/contrast_geometry.rs` drives the real
`acoustic::activation_test` over synthetic arms of known geometry (11 dims, n=9, all 24310
labelings, isotropic Gaussian arms, 40 draws per cell, medians shown) — so this evidence is
reproducible from disk, not asserted:

    cargo run --release -p syrinx-eval --example contrast_geometry 40

A *collinear* cue and sham cannot produce the observed pattern:

| configuration | cue/plain | sham/plain | cue/sham | P(cue/sham < cue/plain) |
|---|---|---|---|---|
| collinear, \|μ\|=1.2 | 0.116 | 0.159 | **0.503** | 0.23 |
| orthogonal, \|μ\|=1.2 | 0.233 | 0.159 | **0.030** | 0.85 |
| opposed 180°, \|μ\|=1.2 | 0.229 | 0.337 | **0.041** | 0.88 |

The observed pattern is real and it rules collinearity out.

### What the data does not support — three duller readings, tested

**1. "Different directions" is unfalsified but so is "merely different directions."** The
table above shows orthogonal and opposed displacements produce statistically
indistinguishable signatures at these magnitudes (0.030 vs 0.041 — and the ordering flips
between draws). The observed p-values
cannot separate "a neutral instruction flattens delivery while a cue colours it" — the
recorded story, which predicts an *obtuse* angle — from "two arbitrary prompts perturb the
trajectory in two arbitrary directions", which in 11 dimensions is generically near-
orthogonal and needs no content interpretation at all. Any two distinct perturbations from a
common baseline satisfy ‖c−s‖ > max(‖c−p‖, ‖s−p‖) once the angle between them exceeds 60°.

**The one run that recorded effect sizes contradicts the obtuse reading.**
`renders/2026-09-06-instruct-lang/run.txt` reports all three distances per cell, so the
angle follows from the law of cosines:

| cell | ‖c−p‖ | ‖s−p‖ | ‖c−s‖ | cos θ | θ |
|---|---|---|---|---|---|
| happy en | 1.76 | 1.85 | 2.17 | +0.278 | 74° |
| happy zh | 2.39 | 1.70 | 1.24 | +0.869 | 30° |
| sad en | 2.98 | 1.41 | 2.61 | +0.483 | 61° |
| sad zh | 2.42 | 1.18 | 2.45 | +0.218 | 77° |
| angry en | 1.40 | 2.54 | 2.37 | +0.393 | 67° |
| angry zh | 1.90 | 1.86 | 2.34 | +0.226 | 77° |

**All six cosines are positive** (mean +0.41; the random-direction null in 11-d is
mean 0, sd 0.30). The cue and sham displacements are *positively correlated and oblique* —
they are not opposed, and in two of six cells ‖c−s‖ is not even the largest of the three.
The recorded hypothesis is stated in the one form the available geometry rejects.

**2. Unequal variance is NOT the explanation** — tested, and eliminated. The plausible dull
story is that the plain arm (no instruction, freest AR trajectory) is simply wider, which
would inflate every `X vs plain` p-value without any directional claim. Simulated with the
cue and sham means made *identical* and only the plain arm's spread varied:

| sd(plain) / sd(instructed) | cue/plain | sham/plain | cue/sham | P(cue/sham < cue/plain) |
|---|---|---|---|---|
| 1.0× | 0.233 | 0.231 | 0.587 | 0.17 |
| 1.6× | 0.313 | 0.356 | 0.487 | 0.40 |
| 2.0× | 0.218 | 0.334 | 0.525 | 0.25 |

Variance alone never drives `cue/sham` down. This candidate is refuted; recording it because
an eliminated alternative is worth as much as a supported one.

**3. The run is ONE observation of the effect, not three.** The sham and plain arms are
per *carrier text*, not per case — `sham_vs_plain` and `a_a` are byte-identical across
cases sharing a sentence (0.0233/0.6608 for the three `mid` cases, 0.4792/0.0507 for the two
`leading` ones). Six sentinels are **three** plain arms and **three** sham arms. Every case
showing the pattern is a cue arm against the *same* sham render set, and that set is the
most-displaced sham in the run (p=0.0233, the smallest of the three). The other two sham
arms do not agree: `whisper-leading` runs the other way entirely (cue/sham 0.2068 against
cue/plain 0.0042), and `calm-mid` shares the divergent cases' sham arm and does not clear.
So: one sham arm supports, one contradicts, one is null throughout. "The sham arm is an
outlier" is not excluded by anything in this run.

**4. Both divergent p-values are at the test's resolution floor.** With n=9 the exact test
enumerates 24310 labelings, so the smallest attainable p is 4.114e-5. `shout-mid`'s
4.1135e-5 *is* 1/24310 and `sad-mid`'s 8.227e-5 *is* 2/24310 — the first and second most
extreme labelings possible. "0.0000 versus 0.0038" reads as a 400× gap; the measurement
cannot express anything smaller, so the true gap is unbounded below and unmeasured. It is
evidence of strong separation and no evidence at all about *how much* stronger.

### The argument that actually decides it: `ShamOnly` has no null

The `cue vs plain` leg is confounded, exactly as FINDINGS.md says: it cannot separate "this
cue has content" from "an instruction was present". But dropping it does not remove the
confound — it removes the anchor and admits a **larger** one. `cue vs sham` asks whether two
*different instruction strings* produce different audio. Simulate two **delivery-neutral**
shams, orthogonal, with no content whatsoever:

| configuration | cue/plain | sham/plain | cue/sham | P(cue/sham < cue/plain) |
|---|---|---|---|---|
| sham1 vs sham2, \|μ\|=1.2 | 0.230 | 0.171 | 0.058 | 0.78 |
| sham1 vs sham2, \|μ\|=1.6 | 0.083 | 0.089 | **0.013** | 0.80 |

The signature the amendment is built on is reproduced in full by two meaningless prompts.
The run contains no measurement that excludes this, because it has no second sham. The
conjunction is crude, but `sham vs plain` at least calibrates one arm against a
no-instruction baseline; `ShamOnly` calibrates nothing.

## Recommendation

**Do not adopt `ContrastRule::ShamOnly`. Keep `PlainAndSham` as C4.2′'s criterion.** The
observation that raised this is real and the conjunction is genuinely imperfect, but the
proposed replacement is weaker, not stronger: it drops the only baseline in the design and
its distinguishing signature is one a pair of meaningless instructions reproduces.

What should change instead, in cost order — none of it a criterion change:

1. **Persist the effect vectors.** `ArmContrast::effect` is computed and then discarded by
   the runner's `report.json`; every geometric question above had to be answered from a
   *different, older* run because of it. Record `effect` per contrast, and ideally the 11-d
   arm centroid, so direction is measured rather than inferred. Zero GPU cost.
2. **Add the missing arm: `sham2`.** A second delivery-neutral instruction of comparable
   length, and the contrast `sham1 vs sham2`. That is the null `cue vs sham` currently
   lacks, and it is the single measurement that would settle this question. 3 carrier texts
   × n renders ≈ 27 renders ≈ 3.5 min at the measured 7.75 s/render. If `sham1 vs sham2`
   separates as strongly as `cue vs sham`, `ShamOnly` is void and this amendment is
   withdrawn on evidence. If it does not, `ShamOnly` becomes arguable and should then be
   proposed with that number attached.
3. **Give each case its own sham arm**, or state plainly in the report that n_independent is
   the carrier-text count, not the case count. Three cases against one shared sham arm were
   read as three observations here, and that is how a one-draw artifact becomes a criterion.
4. **Re-run at n ≥ 11** if the floor matters: at n=9 the two headline p-values are the two
   most extreme values the test can emit, so the run is measuring at its own resolution
   limit.

Until (2) exists, the honest position is the one the certification run already took: the
gate reports no content activation, and the divergence is recorded as an observation.

## What was built under this amendment

`ContrastRule` / `RuleComparison` / `activated_under` / `compare_rules` in
`crates/syrinx-eval/src/contrast.rs`, computing **both** criteria from one set of
measurements. `evaluate_contrast` is unchanged and `tests/cue_contrast_gate.rs` is untouched
and green, so `real_cue_activation_qwen`'s verdict is exactly what it was. A runner may
print both lists; only `ContrastReport::content_activated` is the verdict.

Frozen by `tests/cue_rule_comparison.rs` (9 tests, `GROUP_cue`, model-free), including the
2026-09-09 numbers verbatim: shipped rule `[]`, proposed rule
`[en-emotion-sad-mid, en-style-shout-mid]`.

`crates/syrinx-eval/examples/contrast_geometry.rs` produces every simulated table above by
driving the real `acoustic::activation_test`, so this section's evidence can be regenerated
rather than trusted. It is an example, not a gate: it has no assertions and no board row,
because "a synthetic geometry behaves as geometry predicts" is not a fact about Syrinx.

**Mutation, checked by hand** — each operator flipped in the source, the frozen test run,
the source restored. 17 mutants, 16 killed:

| # | mutant | result |
|---|---|---|
| M1–M3 | `==` → `!=` on `case_id` / `arm` / `against` in the contrast lookup | killed |
| M4–M5 | each `&&` → `\|\|` in the lookup predicate | killed |
| M6 | `beat_plain && beat_sham` → `\|\|` | killed |
| M7–M8 | drop either conjunct of `PlainAndSham` | killed |
| M9 | `ShamOnly` reads the plain leg | killed |
| M10 | drop the `!` in the `divergent` filter | killed |
| M11–M12 | `is_some_and(significant)` → `is_some()` on either leg | killed |
| M13 | swap the plain/sham lookups at the call site | killed |
| M14 | `significant_at`: `<=` → `<` | killed |
| M15 | `divergent` iterates the wrong list | killed |
| M16 | `compare_rules` swaps the two rules | killed |
| M17 | `find` → `rev().find()` in the lookup | **SURVIVED** |

**M17 is equivalent under the input contract and is reported rather than papered over.** A
run emits at most one contrast per `(case_id, arm, against)` triple, so first-match and
last-match are the same row on every input the type is defined for. Killing it would mean
pinning behaviour on duplicated triples, which are malformed input — and on which
`evaluate_contrast`'s own `BTreeMap` lookup takes the *last*, so a test pinning first-match
here would create a disagreement where none exists today. The contract is now stated in
`find_contrast`'s doc comment; the runner, not this function, is where duplicates belong
caught.
