# `[sad]` does change the audio *toward sadness* — the first cue shown to

2026-09-06, second pass. Identical run to `FINDINGS.md` — same 120 renders, same seeds,
same checkpoint (`1.7B-CustomVoice`, GPU bf16, `serena`), n=8/arm — with the judge swapped
from the RAVDESS 8-class model (CREMA-D 0.394 overall, **0.17 on `sad`**) to
`emotion2vec+ large` (**0.911** overall, **0.867 on `sad`**). Raw output:
`run-emotion2vec.txt`.

The acoustic half is byte-identical to the first pass, as it must be — same renders. What
changed is that there is now a judge with standing on the class that mattered.

## The result

Judge deltas are **logits**, not probabilities: this model saturates, so softmax is one-hot
to float precision and would have made every delta 0 or ±1. Bonferroni over all 12 judge
comparisons, α = 0.00417.

| case | arm | delta | t | p | |
|---|---|---:|---:|---:|---|
| **sad** | **en** | **+4.007** | 3.87 | **0.00169** | **significant** |
| **sad** | **zh** | **+5.615** | 5.17 | **0.00014** | **significant** |
| sad | sham-en | −0.091 | −0.22 | 0.828 | flat |
| sad | sham-zh | +0.062 | 0.14 | 0.889 | flat |
| happy | zh | +3.443 | 2.49 | 0.026 | suggestive |
| happy | en | +1.946 | 1.74 | 0.103 | — |
| happy | sham-zh | +2.090 | 1.16 | 0.266 | — |
| angry | en | +0.058 | 0.03 | 0.980 | — |
| angry | zh | −3.426 | −1.91 | 0.077 | *away* |

**`[sad]` moves the audio toward sadness**, in both languages, clearing correction by a
factor of 2.5 (en) and 30 (zh) — while both sham arms sit flat at |t| < 0.25. That is the
cleanest result this project has produced on the cue layer:

- the **acoustic** permutation test, model-free and direction-blind, independently says
  `[sad]`/en is the only cue that changes the delivery at all (p=0.0019);
- the **judge**, model-based and direction-aware, says `[sad]` is the only cue that moves
  toward its named class;
- the **shams** say neither is prompt perturbation.

Three methods with different failure modes, one answer. The earlier "cannot tell" on `[sad]`
was a statement about the instrument, exactly as `FINDINGS.md` said it was, and replacing
the instrument resolved it.

## The caveat that matters, stated plainly

**`sad` is still not the winning class.** The plain renders read −7.68 on `sad`; cued they
read −3.68 (en) and −2.07 (zh) — a large, reliable move, and still negative. The argmax on
this sentence is `surprised` with and without the cue. So the honest claim is:

> the cue adds a substantial and reproducible amount of sadness evidence, and does not make
> the render read as sad.

Anyone quoting the p-value without that sentence is overstating it.

## The other two cues

`[happy]`: zh is suggestive (p=0.026) but does not survive correction, and `sham-zh` moves
in the same direction at comparable magnitude (+2.09) — so part of what looks like a happy
effect may be a Chinese-string effect. Not established.

`[angry]`: nothing, again. `en` is +0.058 (t=0.03 — as close to exactly zero as this
measurement gets), and `zh` is *negative* (−3.43, p=0.077 — away from anger). This is now
the fourth independent look agreeing that `[angry]` does not take on this checkpoint:
the n=1 acoustic proxy, the n=1 RAVDESS judge, the n=8 permutation test, and now a judge
with 1.000 recall on `angry`. It is the strongest candidate for the tuning loop, and the
clearest evidence that the loop has something real to work on.

## What changed about the language question

`zh` beats `en` on the cued class in both cases where there is an effect (sad +5.62 vs
+4.01; happy +3.44 vs +1.95), which is the same direction the weaker judge reported. The
acoustic `zh vs en` contrast remains null (p=0.56, 0.63, 0.25) — the two phrasings still do
not produce acoustically distinguishable audio, and the difference shows up only in what
the judge makes of it. Suggestive, consistent across two independent judges, still not
established. The default remains `En`; switching it is a maintainer call and should be made
with a Chinese-speaking listener, since both the WER oracle and the judge are English-first.
