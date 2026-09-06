# First tuning round: `[angry]`, four candidates, nothing proposed

2026-09-06. `Qwen3-TTS-12Hz-1.7B-CustomVoice`, GPU bf16, `serena`, n=4 seeds x 3 sentences
per arm per split (192 renders, ~40 min). Judge: emotion2vec+ large. Raw: `run.txt`.

`[angry]` was chosen because four independent measurements agreed it does nothing — the
clearest target the loop had, and the one where a win would be unambiguous.

## Outcome: no candidate cleared. The incumbent stands.

All four were rejected at the **first** criterion — none beat plain acoustically at the
Bonferroni-corrected α = 0.0125:

| candidate | p vs plain |
|---|---|
| "Speak as if you are genuinely angry" | 0.0523 |
| "Say this the way someone who is angry would say it" | 0.1633 |
| "You are angry. Speak accordingly" | 0.2388 |
| "Speak in a sharply angry tone" | 0.0232 |

This is a **result, not a failure**. Rephrasing did not rescue `[angry]` on these sentences,
and a loop that proposes nothing when nothing is better is the loop working.

## The guards held, so the round is interpretable

| guard | tune split | holdout split |
|---|---|---|
| sham vs plain | p=0.2497 (null) | p=0.5247 (null) |
| counter-cue delta on `angry` | −0.993, top-gain **happy** | −1.341, top-gain **happy** |

The counter-cue is the one worth dwelling on. The **wrong** label's phrase ("Speak in a
happy, cheerful tone") scored *below* the incumbent on `angry` and gained on `happy`
instead — on both splits. That is the judge demonstrating, inside the run, that it tracks
the label it is asked about. It is the guard that turns "the judge might be weak" from a
known unknown into a per-round observable, and here it reported healthy.

## A finding that complicates the earlier story — flagged, not buried

On the **holdout** split the incumbent behaves differently from the tune split:

| split | incumbent delta on `angry` | top gain |
|---|---|---|
| tune | +0.577 | neutral |
| holdout | **+2.386** | **angry** |

Every previous conclusion that "`[angry]` does not take on this checkpoint" — four
independent looks — used **one sentence**: *"You told me it was handled…"*. The holdout set
is different in character (short imperatives: *"Put it back exactly where you found it,
right now."*). On those, the existing phrase moves `angry` and `angry` is the top gainer.

The honest reading: **+2.386 against ±4.722 noise is inside the noise band**, so this is not
a result. At n=4 the estimates are loose. But it is a specific, cheap, testable hypothesis
that the earlier framing had no way to raise:

> `[angry]` may be sentence-dependent rather than dead — working on imperative/confrontational
> text and not on the accusatory sentence every prior run used.

That deserves a dedicated n=8 run across sentence types before anyone repeats "angry does
not take" as settled. The claim that IS still supported is narrower: on the sentence
measured four times, `[angry]` does nothing.

## Other observations, recorded rather than acted on

- Several candidates beat the **sham** on the holdout (p=0.0082–0.0319) while failing
  against **plain** (p=0.15–0.50). Beating a sham you do not beat plain against is not a
  coherent story; at n=4 it is most likely noise in the sham arm, and it is the reason the
  criteria are conjunctive rather than any-of.
- `top-gain` was `unknown` or `surprised` for several candidates — the judge's escape-hatch
  classes. A phrase whose largest movement is into `unknown` is not steering anything.

## What was NOT done

Nothing was written to `crates/syrinx-cue/`. No proposal file was produced, because there
was no proposal. Had there been one it would have been inert until a human signed
`accepted_by` (ADR-0004).

## Next, in cost order

1. **Re-measure `[angry]` across sentence types at n=8** — the cheapest way to settle the
   sentence-dependence hypothesis above, and it changes what tuning even means for this cue.
2. **Tune `[sad]` instead**, which has a demonstrated effect (p=0.0019 acoustic, +4.0 judge)
   and therefore a real incumbent to beat — a much better-posed optimisation than trying to
   create an effect from nothing.
3. Widen the candidate grammar only after 1 and 2. More candidates raise the Bonferroni bar,
   so a bigger search is strictly harder to win; shrinking the search is free.
