# `[happy]` does not work — and sentence-dependence is not a property of the channel

2026-09-06. `1.7B-CustomVoice`, GPU bf16, `serena`, n=8, 6 sentences x 3 arms = 144 renders.
Judge: emotion2vec+ large (CREMA-D recall **0.967** on `happy` — its second-best class).
Raw: `run.txt`.

The third and last cue whose verdict rested on a single sentence.

## Result

| sentence | acoustic p | judge Δ | t | judge p | top gain |
|---|---:|---:|---:|---:|---|
| anchor-goodnews *(all prior runs)* | 0.3737 | +1.946 | 1.74 | 0.103 | happy |
| confirmation | 0.9622 | −0.428 | −0.17 | 0.865 | fearful |
| announcement | 0.9475 | +1.747 | 1.24 | 0.234 | happy |
| warm-greeting | 0.7585 | −0.900 | −0.77 | 0.451 | unknown |
| **anticipation** | 0.1977 | **−4.198** | **−3.07** | **0.0083** | unknown |
| terse-triumph | 0.8085 | +0.453 | 0.26 | 0.798 | happy |

**Nothing clears the acoustic bar** — the smallest p is 0.1977, two orders of magnitude
above α. And the single judge-significant cell is `anticipation` at **−4.198**: significant,
and in the **wrong direction**, with `unknown` as top gainer.

So `[happy]`'s earlier "cannot tell" — drawn from one sentence — upgrades to *does not
work*, now on six. This is not a case where a better sentence was waiting to be found.

## The structural answer, which is the point of having run all three

| cue | acoustic clears | judge significant | top-gain == cue | both |
|---|:-:|:-:|:-:|:-:|
| **`[sad]`** | **2 / 6** | **3 / 6** | **5 / 6** | **2 / 6** |
| `[angry]` | 0 / 6 | 1 / 6 | 2 / 6 | 1 / 6 |
| `[happy]` | 0 / 6 | 1 / 6 *(wrong direction)* | 3 / 6 | 0 / 6 |

**Sentence-dependence is not a general property of this checkpoint's instruct channel.** The
three cues separate cleanly and differently:

- `[sad]` **works broadly** — the direction is right on 5 of 6 sentences and significant on 3.
- `[angry]` **works narrowly** — one terse command, nothing else measured.
- `[happy]` **does not work** — and its only significant movement is away from the target.

That matters more than any individual cue's score. Had all three degraded together, the
honest conclusion would have been "our measurement is sentence-bound and no per-cue claim
survives". They do not. The instrument resolves real differences *between cues*, and
`[sad]`'s result is a property of the cue rather than of the sentence it was measured on.

## Reproducibility, fourth time

`anchor-goodnews` returns p=0.3737 and Δ+1.946 — the exact values from
`renders/2026-09-06-instruct-lang/`, a different worktree several hours earlier. Every
anchor in all three sweeps has reproduced to the digit.

## What is not claimed

- **No mechanism**, again. Why `unknown` absorbs two of the six is not established.
- **`[happy]` may still be reachable by a different phrasing.** This measures the *shipped*
  phrase ("Speak in a happy, cheerful tone"), not the concept. That is precisely what the
  tuning loop is for — and `[happy]` is now the best-motivated tuning target after `[sad]`,
  because there is a clear deficit and a judge with 0.967 recall to detect a fix.
- One voice, one checkpoint, English.
