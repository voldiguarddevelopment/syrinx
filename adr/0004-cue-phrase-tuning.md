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
