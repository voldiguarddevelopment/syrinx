# `[sad]` generalises across sentences; `[angry]` does not

2026-09-06. `1.7B-CustomVoice`, GPU bf16, `serena`, n=8, 6 sentences x 3 arms = 144 renders.
Judge: emotion2vec+ large (CREMA-D recall **0.867** on `sad`). Raw: `run.txt`.

The `[angry]` sweep showed a cue's verdict can be a property of the sentence it was measured
on. `[sad]`'s positive result came from one sentence too, so it owed the same test.

## Result

Acoustic α = 0.05/12 = 0.00417; judge α = 0.05/6 = 0.00833 (t-test on the cued-class delta
against pooled across-seed noise, n=8).

| sentence | acoustic p | | judge Δ | t | judge p | | top gain |
|---|---:|:-:|---:|---:|---:|:-:|---|
| anchor-vigil *(all prior runs)* | 0.0019 | ✓ | +4.007 | 3.87 | 0.0017 | ✓ | **other** |
| loss-short | 0.0042 | – | +9.538 | 5.02 | 0.0002 | ✓ | sad |
| resignation | 0.0003 | ✓ | +2.690 | 1.70 | 0.111 | – | sad |
| wistful-question | 0.1420 | – | +3.085 | 2.07 | 0.057 | – | sad |
| terse-finality | 0.0233 | – | +7.056 | 2.76 | 0.015 | – | sad |
| reflective-long | 0.0057 | – | +7.631 | 6.96 | 0.00001 | ✓ | sad |

**`sad` is the top-gaining class on 5 of 6 sentences.** Three clear the corrected judge
threshold. Against the same measurement for `[angry]`:

| | `[sad]` | `[angry]` |
|---|:-:|:-:|
| top-gain == the cued class | **5 / 6** | 2 / 6 |
| judge move significant after correction | **3 / 6** | 1 / 6 |
| both | 2 / 6 | 1 / 6 |
| acoustic clears the corrected bar | 2 / 6 | 0 / 6 |

`[sad]` is a real and reasonably general capability on this checkpoint. `[angry]` is
sentence-specific — it works on a terse command and nowhere else measured.

## The anchor is the least representative sentence in the set

The one sentence where `sad` is **not** the top gainer is *"I waited by the window until the
last light went out."* — the sentence every prior `[sad]` conclusion was drawn from. There,
`other` gains more.

That does **not** overturn A28: the `sad` delta on the anchor is +4.007 at p=0.0017, so
"the cue moves the audio toward sadness" stands on its own terms. What changes is the
framing. A28 reported a result from the anchor and could not say whether it generalised;
it does, and *better* than the anchor implied. The anchor was an unlucky draw — the one
case in six where the judge's escape-hatch class absorbs more of the movement.

This is also the concrete answer to the caveat A28 recorded against itself: *"sad is still
not the winning class"*. On the anchor that remains true. On four of the other five it is
the winning class.

## The two measures fail differently, as designed — and that is visible here

ADR-0004 claims the acoustic test and the judge "fail differently" rather than confirming
each other. The table shows it rather than asserting it:

- `resignation` clears the acoustic bar (p=0.0003) while its judge move sits inside noise —
  delivery changed, direction not established.
- `reflective-long` is the strongest judge result in the table (p=0.00001) and misses the
  acoustic bar (0.0057).
- They coincide on exactly one sentence, the anchor.

A conjunctive gate over two measures that agreed would be redundant. One over two that
disagree this often is doing real work — and it is why the run's own strict flag reports
**0 of 6**, which remains the correct gate for *accepting* a phrase and the wrong summary
of what was learned.

## What is not claimed

- **No mechanism.** Why `other` absorbs the anchor's movement, or why `wistful-question` is
  the weakest, is not established. One sentence per cell cannot separate content from
  length, punctuation or prosodic shape.
- **One voice, one checkpoint, one language.** `serena`, `1.7B-CustomVoice`, English.
- **Not that a listener would call these sad.** The judge is 0.867 on this class, not 1.0,
  and per CLAUDE.md the perceptual call is never automated into a passing test.

## Next

- `[happy]` has no sentence set yet and the sweep refuses to invent one — adding it is the
  obvious third data point, and `[happy]` is the remaining cue with a prior verdict
  ("cannot tell") drawn from a single sentence.
- Tuning is now well-posed for `[sad]`: a real, general incumbent to beat, across a sentence
  set that is known to vary. `[angry]` should be tuned only on the sentence where it works,
  or not yet.
