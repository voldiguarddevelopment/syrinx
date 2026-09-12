# The two anomalies the C4.2′ run logged and did not chase

2026-09-11, follow-up to `FINDINGS.md` in this directory. Source data: `run.txt`,
`report.json`. **No GPU and no model was used** — the box's GPU was occupied by other work,
so nothing here was run under `--features real` or `--features cuda`. Everything below is
either a deterministic proof, a model-free replication, or arithmetic on the numbers the
run already wrote down. Where a question needs the model to answer, it is left open and
said so.

Both anomalies were recorded in `FINDINGS.md` as "worth watching, not acting on" with a
one-line guess attached. This file replaces the guesses with measurements.

**Summary.** Anomaly 1 is chance, and the three structural explanations that could have
made it *not* chance are each ruled out — two of them by a deterministic proof rather than
a simulation. Anomaly 2 is not the bug class it resembled: the arithmetic rules out a
text/audio mismatch. What actually caused its single word error is **undetermined**, and
will stay undetermined until the runner stops throwing its transcripts away.

New gates, both registered in `GROUP_cue`:

| file | what it pins |
|---|---|
| `tests/acoustic_permutation_exactness.rs` | the permutation null is exactly uniform; the false-positive rate at n=9; the runner's two seed blocks are exchangeable |
| `tests/cue_measure_reference_text.rs` | the C4.2′ carrier span for every sentinel, and the three WERs that tell an oracle error from a wrong-end-of-the-split bug |

---

# Anomaly 1 — the A/A calibration at p = 0.0507

## The claim being tested

`FINDINGS.md`: *"A/A on the leading text is 0.0507, a hair above nominal 0.05. It does not
violate, but the plain arm on 'I really cannot believe what you just told me.' is closer to
disagreeing with itself than any other text here."*

Three ways it could be structure rather than chance were named: the test is not exact at
n=9; some acoustic feature is heavy-tailed enough to break exchangeability; or the two seed
blocks (`0..n` and `1000..1000+n`) differ for a reason other than chance.

## First, the run made three A/A measurements, not six

The six sentinels use **three** texts, and `real_cue_activation_qwen.rs` counts A/A once
per text (`seen_texts`), which is why `report.json` shows the same three values twice each:

| text rendered (the *carrier span*, which is what the runner actually renders) | words | A/A p |
|---|---|---|
| `I really cannot believe what you just told me.` | 9 | **0.0507** |
| `before it gets any later.` | 5 | 0.6608 |
| `That was the strangest thing I have seen all week.` | 10 | 0.8011 |

That table already embarrasses the "short clips break the features" hypothesis: the
*shortest* clip in the run is the one whose plain arm agrees with itself best.

## The test is exact — proved by enumeration, not estimated

Under the null the 2n renders are exchangeable, so all `N = C(2n, n)/2` labelings are
equally likely. `activation_test` returns the fraction of labelings whose between-means
distance is at least the observed one. If it is exact, then **feeding it every labeling of
one fixed pool must return every value in `{1/N, 2/N, …, N/N}` exactly once**.

Enumerated at n=4 (N=35) and n=5 (N=126), over three pools chosen to be hostile rather
than flattering — i.i.d. Gaussian, **Cauchy** (the ratio of two normals: no finite variance
at all), and eleven-dimensional features extracted by `acoustic::features` from real 1.4-second
synthetic clips:

```
[A gauss]                    n=4 N=35   max|p_sorted - k/N| = 0.000e0   #(p<=0.05) = 1
[A cauchy]                   n=4 N=35   max|p_sorted - k/N| = 0.000e0   #(p<=0.05) = 1
[A short-clip-real-features] n=4 N=35   max|p_sorted - k/N| = 0.000e0   #(p<=0.05) = 1
[A gauss]                    n=5 N=126  max|p_sorted - k/N| = 0.000e0   #(p<=0.05) = 6
[A cauchy]                   n=5 N=126  max|p_sorted - k/N| = 0.000e0   #(p<=0.05) = 6
```

Deviation **exactly zero**, bit-for-bit, in every case. The statement is *conditional on the
pool*, and that is what makes it decisive: a distribution cannot break a property that holds
for pools drawn from it. **Hypothesis (b), heavy tails, is dead** — not "unlikely", dead.
Same for short clips.

The consequence is arithmetic rather than empirical:

> false-positive rate at alpha = `floor(N · alpha) / N`
> at n=9, alpha=0.05: `floor(24310 × 0.05) / 24310` = **1215 / 24310 = 0.049979**

An exact permutation test is never anti-conservative; the only thing discreteness can do is
round the size *down*.

## Monte-Carlo cross-check at n=9, which is the n the run used

Same-condition comparisons, 18 i.i.d. vectors split 9/9, deterministic SplitMix64 draws.
The headline row is 50 000 replications:

| pool | reps | hits | FPR | vs exact 0.049979 |
|---|---|---|---|---|
| **gaussian** | **50 000** | **2536** | **0.05072 ± 0.00098** | **+0.76 s.e.** |
| gaussian, rows shuffled before splitting | 50 000 | 2535 | 0.05070 | +0.74 s.e. |
| gaussian, one PRNG stream per row | 50 000 | 2530 | 0.05060 | +0.64 s.e. |
| **cauchy** | **20 000** | **1003** | **0.05015 ± 0.00154** | **+0.11 s.e.** |
| cauchy | 5000 | 281 | 0.0562 ± 0.0033 | +2.0 s.e. |
| cauchy (different seed base) | 1000 | 53 | 0.0530 ± 0.0071 | +0.4 s.e. |
| n=4 gaussian, against 1/35 = 0.028571 | 5000 | 162 | 0.0324 ± 0.0025 | +1.5 s.e. |

The middle two rows are controls on the *simulation* rather than on the estimator: several
smaller runs sat 1–2 s.e. high and it was worth checking that the synthetic pool was not
ordered — that 22 consecutive draws of one stream per row, split into "first nine" and
"last nine", introduced structure. It did not: shuffling the rows and giving every row its
own stream both leave the count unmoved (2535 and 2530 against 2536). The earlier
"persistent" excess was the same realization being counted more than once — the 1000- and
5000-rep gaussian runs are subsets of the 50 000.

The +2.0 s.e. row did not reproduce: 20 000 replications of the same infinite-variance pool
land on 0.05015, **+0.11 s.e.**, which is what noise looks like once it is given enough
draws. It could not have been real in any case — the enumeration above proves the
conditional size is exactly 0.049979 for Cauchy pools too, and the marginal size is an
average of conditional sizes.

## The seed blocks are not distinguishable

The runner's plain halves come from seeds `0..9` and `1000..1009`. `syrinx_qwen::sampling::
SplitMix64` takes the seed *as its state*, so the j-th draw of seed `s` is
`mix(s + j·0x9E3779B97F4A7C15)`, with `mix` a bijective avalanche finalizer. Two seeds
collide only if `s − s′ = (j′ − j)·γ (mod 2^64)`, which for small seed gaps and realistic
draw counts happens only at `j = j′`, `s = s′`. There is no structural reason for the blocks
to differ.

Measured as well as argued. A render is a deterministic function of `(prompt, seed)` and the
PRNG stream is a function of the **seed alone** — so the two blocks are *fixed* for the whole
run, and any artefact in them would contaminate every A/A in every C4.2′ run. Holding the
blocks at exactly the runner's values and varying the *prompt* instead (a per-replication
offset into each seed's stream plus a per-replication linear map onto the eleven dimensions):

```
[D seed-blocks 0..9 vs 1000..1009] reps=400  hits=20  FPR=0.0500 +/- 0.0109  median p = 0.5550
[D seed-blocks 0..9 vs 1000..1009] reps=1000 hits=48  FPR=0.0480 +/- 0.0068
```

Nominal. **Hypothesis (c) is ruled out.**

## The arithmetic on 0.0507 itself

At n=9 a p-value is a multiple of `1/24310 = 4.11e-5`, so the logged value is an exact
integer count:

```
0.05071986836692719 x 24310 = 1233
```

1233 of the 24310 labelings of that pool separate at least as far as the observed one.

* The largest value that **rejects** at alpha=0.05 is `1215/24310 = 0.049979`. **1233 > 1215:
  the A/A did not reject.** It is not a near-violation; it is a non-rejection sitting at the
  94.93rd percentile of its own null.
* Three A/A measurements were made. `P(min of 3 ≤ 0.0507) = 1 − (1−0.0507)³ = **0.1446**`.
  Seeing a value this small among three is a **1-in-7 event** — the ordinary case, not the
  exception. (`P(at least one of the three actually rejects) = 1 − (1−0.049979)³ = 0.1426`,
  which is the more relevant number and is just as unremarkable.)

  That arithmetic assumes the three are independent, and they are **not obviously** so: all
  three texts are rendered from the *same* two seed blocks, so their A/A statistics are
  three functions of one set of eighteen PRNG streams. Checked rather than assumed — 600
  simulated "worlds", each a fresh pair of seed blocks shared by three prompts:

  ```
  [corr] prompts 0x1: r = -0.0467   (s.e. under independence ~= 0.0409)
  [corr] prompts 0x2: r = -0.0309
  [corr] prompts 1x2: r = +0.0176
  [corr] P(min of 3 <= 0.0507) measured = 0.1650   (independence predicts 0.1446)
  [corr] P(at least one of 3 rejects)   = 0.1633   (independence predicts 0.1426)
  ```

  Correlations indistinguishable from zero, and the measured 1-in-6 is +1.4 s.e. from the
  1-in-7 the independence calculation gives. Sharing seed blocks across texts does not tie
  the A/A p-values together.
* Against the alpha the run actually operated at — Bonferroni `0.05/24 = 0.002083` — the value
  is **24.3x above** the bar.

## Verdict

**Chance.** One p-value of 0.0507 out of three is exactly what chance looks like: the
expected minimum of three uniforms is 0.25, and 0.0507 or lower happens 14.5 % of the time.
The instrument is exactly the size it claims, and the two mechanisms that could have made it
otherwise are excluded — heavy tails and short clips by a deterministic proof, the seed
blocks by structure and by 1400 replications.

**Confidence: high** on the calibration (this is a proof, not an estimate) and on the
arithmetic. The one thing that remains a probability statement rather than a fact is
attributing *this specific* number to chance rather than to some property of *this specific*
sentence — 14.5 % is unremarkable but it is not zero.

**The cheap way to close even that**, when a GPU is free: the A/A is a single deterministic
number per text, so it says nothing about repeatability. Render two more disjoint plain
blocks (seeds `2000..2009`, `3000..3009`) for the one text and run the three extra pairwise
A/As. Six A/A measurements on the same text, of which at most one should sit near 0.05.
Cost: 2 arms × 9 renders × 1 text, a small fraction of the 26-minute run.

---

# Anomaly 2 — `en-emotion-calm-mid` reports wer 0.200

## The claim being tested

`FINDINGS.md`: *"`calm-mid` carries wer 0.200 where every other case is 0.000 … the only case
with real transcription error and the only one whose cue is `calm` — plausibly the oracle
mishearing a quieter delivery."* That last clause is a guess. The specific worry worth
chasing is that this is another instance of the ADR-0005 defect class — **the runner holding
the wrong end of a split utterance** — which on 2026-09-09 made `segments.first()` mislabel
four of six sentinels.

## What the runner actually scores against what

`plan()` is model-free, so this is directly observable. For the six sentinels:

| case | placement | segments | carrier span (what is rendered *and* what WER scores against) | words |
|---|---|---|---|---|
| happy-leading | leading | 1 | `I really cannot believe what you just told me.` | 9 |
| whisper-leading | leading | 1 | `I really cannot believe what you just told me.` | 9 |
| angry-trailing | trailing | 1 | `That was the strangest thing I have seen all week.` | 10 |
| sad-mid | mid | 2 | `before it gets any later.` | 5 |
| calm-mid | mid | 2 | `before it gets any later.` | 5 |
| shout-mid | mid | 2 | `before it gets any later.` | 5 |

A mid cue makes `pass_hoist` split; segment 0 is `"We should probably leave "` with no
instruct and segment 1 is the carrier. The runner binds **one** variable, `clean`, from the
carrier and uses it for three things: the text handed to the engine, the reference for
`wer_cued`, and the reference for `wer_plain`. Reference and audio are the same string *by
construction* — there is no second read that could drift.

This also matches production rather than diverging from it: `QwenSynth::render_all` renders
each segment as its own request and concatenates, so the shipping path renders that fragment
standalone too.

## The arithmetic rules out the ADR-0005 class outright

`wer` is `edits / reference_word_count`. The three possible pairings give three distinct
values, and only one of them is reachable:

| what was compared | edits | reference words | WER |
|---|---|---|---|
| carrier text vs carrier audio — **correct** | 1 | 5 | **0.200** ← observed |
| whole-utterance text vs carrier audio | 4 deletions | 9 | 0.444 |
| carrier text vs whole-utterance audio | 4 insertions | 5 | 0.800 |

**0.200 is not reachable by either mismatch.** A wrong-end-of-the-split bug would have
announced itself as 0.444 or 0.800. The observed value is one word edit against the correct
five-word reference. This table is now frozen in
`tests/cue_measure_reference_text.rs::a_wrong_end_of_the_split_cannot_produce_the_observed_wer`,
so a future 0.444 is recognisable on sight instead of needing this derivation again.

## Also ruled out

* **Case and punctuation.** `wermetric::normalize_words` lowercases and replaces every
  non-alphanumeric with a space. Independently confirmed by the data: the plain arm scored
  0.000 against the same reference, which it could not have if normalization were broken.
* **A nondeterministic oracle.** `Stt::decode` takes `argmax` at every rung of the
  temperature-fallback schedule (`argmax(&v) // deterministic fallback: take the mode (no RNG
  dep)`), so the transcript is a pure function of the audio and the 0.200 is reproducible,
  not a lucky draw.
* **Cue markup reaching the model.** `plan()`'s lowered text contains no `[`, and the
  carrier span is byte-identical with and without the cue. The spoken text is the same in
  both arms; only the instruct differs.
* **`calm` being special in the lowering.** Its instruct — `"Speak in a calm tone"` — is
  built by the same `instruct_for` path as the other five, and the segmentation is identical
  to `sad-mid` and `shout-mid`, which scored 0.000 on the same reference.

## What cannot be concluded, and why

One word in five was wrong. **Which word, and whether it was a substitution, a deletion or
an insertion, is not recoverable from this run**: the runner computes `wer()` and discards
the transcript. So the recorded hypothesis — the oracle mishearing a quieter delivery —
remains exactly as unevidenced as when it was written. Confirming or refuting it needs the
model, and this pass had no GPU.

It is worth naming what the alternative hypotheses would be, since they are not equally
likely and none can be separated without the transcript: whisper-base mishearing a quiet
final word (`later` → `late`); a hallucinated insertion, which is whisper's characteristic
failure on a short clip zero-padded to 30 seconds; or a wrong language detected on 1.4
seconds of audio.

## The instrument weaknesses this exposed — where the real risk is

None of these is a correctness bug, and none was changed in this pass (see below), but the
anomaly was only *unresolvable* because of the second one.

1. **The WER veto uses one render out of nine.** `w_cued` comes from `cued[0]` and `w_plain`
   from `plain_a[0]`. The activation test uses all nine renders per arm; the veto uses one.
   On a five-word reference the veto's resolution is 0.2 per edit and its sampling error is
   at least one edit — a single draw cannot distinguish "this cue damages intelligibility"
   from "whisper slipped once". Scoring all `n` and reporting mean and max costs nothing but
   `n` transcriptions.
2. **The transcript is thrown away.** Recording `hyp_cued` / `hyp_plain` in `report.json` is
   two lines and would have turned this investigation into a five-second read. This is the
   single highest-value change on the list.
3. **The oracle auto-detects language.** The runner calls `Stt::transcribe`, which passes
   `None` and runs LID — on a 1.4-second clip — even though the sentinel set is English-only
   by construction (the runner's own doc comment says so) and `SYRINX_STT_LANG=en` is already
   in `scripts/test-all.env`. `transcribe_lang(..., Some("en"))` removes an uncontrolled
   degree of freedom from the measurement.
4. **The mid cases' WER is not comparable to the others'.** Their reference is a sentence
   fragment opening with a subordinating conjunction — the hardest thing to hand a
   base-sized ASR. That is inherent to what a mid cue scopes, not a defect, but a 0.2 on five
   words and a 0.2 on ten words are not the same evidence and the report presents them as if
   they were.

**Why none of these was changed here.** `tests/real_cue_activation_qwen.rs` is
`#![cfg(all(feature = "real", feature = "cuda"))]`. This pass could neither run it nor
compile it, and editing a gate that cannot be compiled — let alone executed — is precisely
what CLAUDE.md forbids. They are recorded as work for a pass with the GPU.

## Verdict

**No bug in the WER computation, and the ADR-0005 defect class is absent** — ruled out by
arithmetic, and now gated so it cannot return unnoticed. **Confidence: high.**

**The cause of the single word error is undetermined. Confidence in the recorded "oracle
misheard a quieter delivery" hypothesis: none** — it is consistent with everything measured
and so are two other explanations. It becomes answerable the moment the runner records what
Whisper actually said.

---

# The new gates, and the by-hand mutation check

Both files are model-free and deterministic, and both are in `GROUP_cue`. Release-profile
cost: 6.3 s and under 0.01 s.

CLAUDE.md's mutation rule is about implementation, and these tests defend code that already
exists — so instead of running the gate, each mutant below was applied by hand to the
implementation, the new tests were run, and the source was restored from a backup and
`git diff --stat` checked clean.

| # | mutant | result |
|---|---|---|
| M1 | `acoustic.rs`: `>= observed - 1e-12` → `> observed - 1e-12` | **survives — equivalent mutant.** `d_obs > d_obs − 1e-12` is true, so the observed split still counts itself. The epsilon is doing exactly the job it was written for. |
| M1b | `acoustic.rs`: `>= observed - 1e-12` → `> observed` (epsilon dropped) | killed by `the_permutation_null_is_exactly_uniform_for_every_pool` — the distribution becomes `{0/N … (N−1)/N}` |
| M2 | `acoustic.rs`: standardize over the first condition instead of the pool (label leak) | killed by `the_permutation_null_is_exactly_uniform_for_every_pool` **and** by `the_measured_false_positive_rate_at_n_nine_matches_the_nominal_size` |
| M3 | `acoustic.rs`: drop the `combo.contains(&0)` canonicalization | killed by `the_exact_size_at_n_nine_is_just_under_alpha` (`n_permutations` doubles). Note the uniformity test does **not** catch this: doubling both numerator and denominator leaves every p-value unchanged, which is why the labeling count is asserted separately. |
| M5 | `wermetric.rs`: normalize by the hypothesis length instead of the reference | killed by `a_wrong_end_of_the_split_cannot_produce_the_observed_wer` (the 4/9 and 4/5 asymmetry is the whole point) |
| M6 | `hoist.rs`: split at the cue's `span.end` instead of `span.start` | killed by three of the five: `the_carrier_span_is_what_the_runner_renders`, `only_a_mid_cue_splits_the_utterance`, `the_instruct_never_changes_the_spoken_text` |
| M7 | `hoist.rs`: a trailing manner cue applies to `out.first_mut()` instead of `out.last_mut()` (an ADR-0005 regression) | **survives these files** — the trailing sentinel has one segment, so first and last coincide. Already killed by the existing `tests/cue_trailing_manner.rs`, which is where it belongs. |

Two surviving mutants, both understood: M1 is genuinely equivalent, and M7 is out of scope
for these files and covered by an existing gate.

One thing the exactness test deliberately does **not** catch: changing the test *statistic*
(for example `mean_a − mean_b` → `mean_a + mean_b`). A permutation test is exact for any
statistic, so uniformity is silent about that choice — which is correct. The statistic's
*power* is what `tests/cue_activation_measure.rs` pins, and that division of labour is
intentional.

## Board

`source scripts/test-all.env && ./scripts/test-all.sh free` — 29 PASS / 0 SKIP / 0 FAIL
(27 before; the two new files are the difference).
