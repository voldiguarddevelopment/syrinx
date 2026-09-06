# `[angry]` is sentence-dependent — the earlier verdict was about one sentence

2026-09-06. `1.7B-CustomVoice`, GPU bf16, `serena`, n=8, 6 sentences x 3 arms = 144 renders.
Judge: emotion2vec+ large (CREMA-D **1.000** recall on `angry` — its best class). Raw:
`run.txt`.

## The correction

Four independent measurements had concluded "`[angry]` does not take on this checkpoint".
**All four used the same sentence.** Measured per sentence at n=8:

| sentence | kind | acoustic p | judge Δ | t | judge p | top gain |
|---|---|---:|---:|---:|---:|---|
| accusatory-long | long accusation | 0.7417 | +0.058 | 0.03 | 0.980 | happy |
| holdout-imperative | short imperative | 0.7576 | +0.275 | 0.25 | 0.806 | surprised |
| holdout-confront | direct confrontation | 0.2824 | +5.542 | 2.38 | 0.032 | **angry** |
| holdout-betrayal | quiet reproach | 0.6033 | +1.714 | 2.63 | 0.020 | other |
| new-exasperated | exasperated question | 0.1030 | −0.857 | −0.86 | 0.402 | surprised |
| **new-command** | **terse command** | **0.0045** | **+10.179** | **4.16** | **0.0010** | **angry** |

Bonferroni: acoustic α = 0.05/12 = 0.00417, judge α = 0.05/6 = 0.00833.

**On *"Stop talking and listen to me for once."* the cue works.** Judge delta +10.179 —
by far the largest in the table — t=4.16, p=0.0010, clearing the corrected threshold, with
`angry` as the top-gaining class. The acoustic test misses its (stricter, 12-comparison)
bar by 8%: 0.0045 against 0.00417.

**On *"You told me it was handled…"* — the sentence all four prior runs used — it is flatly
dead.** t=0.03 is as close to exactly zero as this measurement produces.

So the claim "`[angry]` does not take on this checkpoint" was never supported. What was
supported, and all that was ever measured, is: *on that one sentence, it does nothing.*

## The strict flag says "0 of 6", and that is not the finding

The run's own conjunctive flag requires **both** bars — acoustic and judge — and nothing
cleared both, so it printed `0 of 6`. That is the correct gate for *accepting a tuned
phrase* (ADR-0004 is conjunctive on purpose). It is the wrong summary of *what was
learned*, and reporting only the flag would have buried the result. Both numbers are given
above for that reason.

## What the axis is not

The tempting story — "imperatives work, accusations don't" — does not survive the table.
`holdout-imperative` ("Put it back exactly where you found it, right now.") is a short
imperative and reads `surprised` at p=0.81. Two sentences move `angry` (`new-command`,
`holdout-confront`) and they are both *second-person confrontational commands*, but n=1
sentence per cell cannot separate that from length, punctuation, or lexical content. **No
mechanism is claimed here.** What is established is variance across sentences large enough
to flip the verdict, which is enough to invalidate the old conclusion without supporting a
new one.

## A concern raised mid-run and withdrawn

At three sentences in, the sham arms sat at 0.0325 / 0.0260 / 0.0168 — consistently *lower*
than the cue arms — and that was flagged as a possible systematic problem with the sham. It
was not. The full six are `[0.0325, 0.0260, 0.0168, 0.1941, 0.5801, 0.2329]`: three low,
three high, none within an order of magnitude of the corrected α. Noise, and the interim
reading was premature.

## Consequences

- **The `[angry]` tuning round is not interpretable as run.** `renders/2026-09-06-tune-angry/`
  pooled three sentences per split; this shows the effect varies by more than the pooled
  signal, so "no candidate cleared" may be a statement about the sentence mix rather than
  the candidates. Re-run per sentence, or on a sentence where the incumbent demonstrably
  works — the latter is better posed, because a tuner needs an effect to improve on.
- **The tuning holdout partition is retired.** Its three sentences were measured here, so
  ADR-0004 §5 requires a fresh `holdout_id` before the next round. The mechanism did its
  job: the expiry rule existed before it was needed.
- **Every per-cue verdict in this tree inherits the same caveat.** `[sad]`'s result
  (p=0.0019 acoustic, +4.0 judge) also came from one sentence. It is a stronger effect and
  was confirmed by three methods, but "on one sentence" applies to it too, and the same
  6-sentence sweep should be run for it before `[sad]` is described as working generally.
