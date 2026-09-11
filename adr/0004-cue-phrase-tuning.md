# ADR-0004 — tuned instruct phrasings: where they live and what may accept one

Status: **PROPOSED** (acceptance is a human act — ADR-0001 §7 / C0.2′)
Date: 2026-09-06
Depends on: ADR-0003 (the sham arm and the corrected alpha), and on the judge swap of
2026-09-06 (`docs/LICENSES.md`).

## Why this needs an ADR at all

Not the generic "cue data is governed by ADR". The specific reason: a tuned phrase
introduces a **new kind of provenance** into `syrinx-cue`.

`vocab.toml`'s header promises its content is derived from verified upstream sources and
"**NOT invented**". `caps.toml` records facts about backends with a cited source. A tuned
phrase is neither: it was *found by automated search against a measurement, on one box, on
one checkpoint, on one date, and signed off by a person*. Mixing that into either file
destroys the property that makes it auditable, which is exactly the class of decision
ADR-0001 §2.3 exists to govern.

## Decision

### 1. Tuned rows live in `crates/syrinx-cue/instruct.toml`, in their own `[[tuned]]` table

Keyed `(backend, lang, label)` — per checkpoint because instruct semantics are per
checkpoint (ADR-0001 §2.3): CustomVoice's instruct describes *delivery*, VoiceDesign's
describes *the voice*. Every row carries `measured_on`, `incumbent`, `margin`, `judge`,
`judge_recall_on_class`, `holdout_id`, `holdout_uses`, `accepted_by`.

`judge_recall_on_class` is mandatory because a verdict quoted without it is not a verdict:
the current judge is 0.911 overall but **0.667 on `fearful`**, and a phrase tuned on that
class deserves the caveat travelling with it.

Rejected homes: `legacy_emotion.rs` (deprecated, quarantined, dies with CosyVoice) and
`vocab.toml` (backend-neutral and "not invented").

### 2. Lookup is tuned → curated → deterministic fallback, and strictly additive

A tuned row never deletes or overwrites a curated phrase. Deleting a tuned row restores the
previous behaviour **exactly**, and a frozen test asserts that with no tuned rows present
the backend lookup is identical to the curated one. Reversibility is what makes running the
loop safe to try.

### 3. **A row without `accepted_by` is inert**

Present in the file, validated on load, and returned by no lookup. The loop proposes; a
person listens and signs. CLAUDE.md: perceptual judgements are "NOT expressible as a
frozen-test + mutation gate", and no quantity of measurement changes that. An empty or
whitespace signature is not a signature.

This is the single assertion standing between "the loop proposed a phrase" and "the product
says it", so it is frozen-tested from both sides and mutation-checked.

### 4. The accept criteria are **conjunctive**, never a weighted sum

All seven must hold, on **both** splits:

| # | criterion | why |
|---|---|---|
| i | beats plain acoustically at the corrected α | it changed the delivery |
| ii | beats its **sham** at the corrected α | content, not prompt perturbation (ADR-0003) |
| iii | judge delta exceeds its own across-seed noise | the move is larger than the draw |
| iv | the **cued class gained most** | a phrase that raises `surprised` is not a win |
| v | margin over the **re-measured** incumbent; a tie keeps the incumbent | not any improvement |
| vi | WER veto | affect bought with intelligibility is not bought |
| vii | speaker-similarity floor | see below |

A scalar objective is a Goodhart machine: the moment two of these are tradeable, a search
will trade them. Conjunctive or nothing.

**(vii) is not obvious and is the one most likely to be dropped by a later simplification.**
Without it, *"Speak like a frightened old man"* is a legitimate winner — it changes delivery
(i, ii), moves `fearful` (iii, iv), keeps the words (vi), and destroys the requested voice.
`qwen::speaker_similarity` already exists and costs one embedding pair.

### 5. Round-level voids: guards that make judge failure observable

- **Sham activated** → the round is uninterpretable, per ADR-0003.
- **Counter-cue won** — every batch carries the phrase for the *wrong* label (for `[sad]`,
  the `happy` phrasing). It should lose. If it beats the incumbent on the target class, the
  judge is not tracking what it is supposed to track and **the whole round is void**. This
  converts "the judge might be weak" from a known unknown into a per-run observable, and it
  is the highest-value guard on the list.
- **Holdout expired** — a partition serves at most **K = 5** decisions. A holdout used
  repeatedly stops being held out: every accept/reject leaks a bit about it. This is the
  standard way held-out sets rot in an iterated loop, and it is bounded rather than left to
  discipline. `holdout_uses` is recorded in the row.
- **Incumbent not re-measured** — the incumbent is measured **in the same round, at the
  same seeds**, never compared against a stored number. Otherwise driver drift, a GPU
  change, or a candle bump masquerades as an improvement.

### 6. Honest limits on the guards

Stated here so they are never overstated later:

- The two measures **are not independent**. The permutation test is direction-blind and
  model-free; the judge is direction-aware and model-based — but both read the same
  waveform, and a phrase that merely raises loudness moves both. The claim is "they fail
  differently", not "agreement is confirmation".
- The counter-cue guard catches **gross** judge failure. A judge subtly biased toward, say,
  loudness passes it happily.
- The guards raise the cost of gaming. They do not eliminate it.

### 7. What is never automated

The accept decision; editing any frozen file; moving a threshold to make a candidate pass;
writing `caps.toml`; deleting or overwriting a curated phrase; any ADR status transition;
and using the judge's scores as a **training signal for weights** — `docs/LICENSES.md`
marks that as where "measurement tool" becomes "derivative work".

## Consequences

- `syrinx_eval::tune` holds the decision as a **pure** function, so `tests/cue_tune_decision.rs`
  gates it on the model-free board: a future pass that wants to relax a margin has to get
  past a test, not edit a constant.
- The driver writes a **proposal** under `.opt-reports/`, never into `crates/syrinx-cue/`.
- Ledger amendment **A29**.

---

# PROPOSED AMENDMENT — 2026-09-11: what the holdout split may require

**Status of this section: PROPOSED. Nothing below is accepted.** ADR-0004's own Status
line is unchanged and this amendment does not change it; acceptance is a human act
(ADR-0001 §7 / C0.2′). The code shipped alongside it is a *policy enum with the status quo
as its default-compatible arm*, not a change of behaviour: `decide_per_sentence` is defined
to be the existing rule and a frozen test asserts that.

## The question, and the measurement that raised it

`renders/2026-09-09-tune-sad/FINDINGS.md`, `[sad]` at n=8, `1.7B-CustomVoice`:

| | tune | holdout |
|---|:-:|:-:|
| incumbent (`"Speak in a sad, sorrowful tone"`) clears | **2 / 3** | **0 / 3** |

The holdout p-values for the shipping phrase were 0.0284, 0.0519, 0.1206 against a
corrected alpha, and on two of the three the top-gaining class was `other`/`neutral` rather
than `sad`. `decide_per_sentence` requires a candidate to clear `min_sentences` (2 of 3) on
**each** split, so a challenger must clear 2 of 3 sentences on which the shipping phrase
clears none.

That was never decided. It is the unexamined consequence of applying one absolute count to
both splits, and §4/§5 of this ADR are silent on it.

Context that makes it an ordinary occurrence rather than bad luck:
`renders/2026-09-06-sad-sentences/` and `renders/2026-09-06-angry-sentences/` establish
between-sentence variance large enough to flip a cue's verdict — `[angry]` reads
p = 0.0045 on one sentence and p = 0.7417 on another. A three-sentence partition where the
channel barely works is a normal draw from that distribution.

## The options, and what each can be gamed by

Implemented as `syrinx_eval::tune::HoldoutPolicy`; frozen in
`tests/cue_tune_holdout_policy.rs`.

### A — `Absolute`: keep absolute breadth (status quo)

Clear `min_sentences` holdout sentences, whatever the incumbent does there.

*For.* The bar is fixed, pre-registered and auditable: "it worked on 2 of 3 held-out
sentences" is a claim a reader can check without knowing anything about the incumbent's
round. It cannot be moved by a bad incumbent draw. Crucially, **its error is one-sided** —
it can only reject a phrase that deserved to win, never accept one that did not, which is
the safe direction of error in a system whose stated worst outcome is a false green.

*Against.* When the partition is hard, the loop's answer ("nothing beat the incumbent") is
determined by the sentences rather than by the candidates, and reads as a verdict about the
candidates. It is the same defect this module's own header names for the pooled design:
*"a gate the incumbent cannot pass can accept nothing"* — the failure ADR-0003 recorded for
the 0.85 activation floor, reached from the other direction.

*Gameable by.* Choosing easy holdout sentences, and by `min_sentences` itself. Both are
pre-registered human choices, which is a mitigation and not a defence.

### B — `StrictlyBroaderThanIncumbent`: relative holdout requirement

Clear strictly more holdout sentences than the incumbent does.

*For.* It states exactly what a tuning loop is for — better than what we ship — and it is
the natural reading of "confirmation": the holdout re-runs the comparison selection was
made on, rather than setting a fresh exam.

*Against.* The bar collapses precisely when the incumbent is weakest. On the 2026-09-09
round it becomes "clear 1 of 3", and one sentence is what the whole per-sentence design
exists to stop being decisive. The bar also now moves with a *measurement* of the
incumbent, so a bad draw for the reference lowers the bar for every challenger in that
round. Pinned as behaviour in
`strictly_broader_collapses_to_a_single_sentence_when_the_incumbent_clears_none`.

*Gameable by.* Anything that depresses the incumbent's measured record — harder sentences,
lower n, an unlucky seed set. It builds in an **incentive to make the reference look
worse**, which is the wrong incentive to have anywhere near a search loop.

### C — `OnlyWhereIncumbentClears`: paired / conditional breadth

Count only the holdout sentences the incumbent also clears, and require `min_sentences` of
those.

*For.* A genuine like-for-like comparison: you are asked to win only where the channel
demonstrably works, which is the cleanest answer to the "different vs better" confound.

*Against.* On the round that raised the question the eligible set is **empty**. It has to
fail closed (it does — `the_paired_arm_fails_closed_on_an_empty_eligible_set`), so it
answers nothing while looking like an ordinary rejection. It also discards the sentences
where an improvement would matter most, and a partition with one eligible sentence quietly
becomes a one-sentence test.

*Gameable by.* Making the incumbent fail on the hard sentences shrinks the eligible set to
the easy ones, so breadth evaporates without any threshold changing.

### D — `RequireFitPartition`: keep the absolute bar, and void an unfit partition

Keep A exactly. Add one round-level void: if the incumbent cannot itself clear
`min_sentences` of the **holdout** sentences, the round is
`Void::HoldoutPartitionUnfit` and no candidate is scored.

The incumbent's self-clearing is the same `check` with the margin criterion neutralised —
significance versus plain and versus the sham, a judge move outside its own noise, the cued
class gaining most, the WER veto, the speaker floor. Those are exactly the criteria the
2026-09-09 holdout sentences defeated.

*For.*
- **It moves no bar.** There is no new lever for a search to pull; on a partition the
  incumbent passes, D is byte-identical to A (pinned in
  `on_a_fit_partition_require_fit_is_indistinguishable_from_absolute`).
- **It inverts B's incentive.** A weak reference *voids* the round instead of lowering the
  bar, so there is nothing to gain by making the incumbent look bad.
- **It makes an invisible problem loud.** Today the driver prints "the incumbent stands";
  under D it prints that the partition could not answer, which is what was actually true on
  2026-09-09.
- **It is the same argument as an existing void.** `IncumbentNotRemeasured` exists because
  "without it there is nothing to have a margin over". An incumbent that is re-measured and
  fails every holdout criterion is the same hole one step further in.
- It is the honest answer to "different or better?": *this partition cannot tell you.*

*Against.* It can never accept a phrase that works only where the incumbent fails — the
case FINDINGS calls "arguably exactly what tuning should reward". The reply is that this is
precisely the case where "better" and "different" are indistinguishable from the data, and
that the case is deferred rather than forbidden: it becomes decidable as soon as a
partition exists that the incumbent can pass. It also costs a round.

*Gameable by.* Re-drawing the holdout until one is fit — p-hacking on the partition. This
is real and is **not** closed by anything in this amendment; see Open questions.

## Recommendation

**D (`RequireFitPartition`)**, as a pre-registered per-round policy. It is the only option
that answers the question without introducing a threshold that a search can move, and its
failure mode is a loud void rather than a quiet acceptance. A conjunction of A with one
more guard is in the spirit of §4: conjunctive, never a weighted sum.

The policy is a *parameter*, exactly like `min_sentences` and for the same reason — it must
be chosen before a round, never after seeing the numbers. §7's "what is never automated"
list applies to it unchanged.

## What this amendment does NOT decide (human, before the next round)

1. **Whether to adopt D at all**, and whether `Absolute` remains the default. Nothing in
   the code changes today: `decide_per_sentence` is still A, and
   `examples/tune_instruct.rs` still calls it.
2. **Whether a `HoldoutPartitionUnfit` void consumes a `holdout_uses` budget slot.** It
   serves no decision, but it does leak — you learn the incumbent fails there. Not
   implemented: the budget is the driver's bookkeeping, not the pure function's.
3. **How many holdout re-draws are allowed** before re-drawing is itself the search. D is
   gameable exactly here, and the honest bound is a human one.
4. **Whether the 2026-09-09 `[sad]` round should be re-labelled.** Under D its verdict is
   "partition unfit", not "the incumbent stands". The measured numbers do not change; the
   sentence written under them would.
5. **Whether the holdout partition should be *chosen* so the incumbent passes it.** That is
   D's requirement stated as a selection rule, and it is a different and stronger claim —
   it would make the holdout non-random by construction.

## Consequences if accepted

- `examples/tune_instruct.rs` switches to `decide_per_sentence_with_policy(..., policy)`
  with the policy printed in the round header and recorded in the proposal row.
- A `[[tuned]]` row would want the policy in its provenance, alongside `holdout_id` and
  `holdout_uses` — a row is not auditable if you cannot tell which rule admitted it.
- Ledger amendment to follow on acceptance.
