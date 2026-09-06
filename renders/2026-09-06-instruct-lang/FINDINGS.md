# Instruct language, and whether any of it is more than prompt perturbation

2026-09-06. `Qwen3-TTS-12Hz-1.7B-CustomVoice`, GPU bf16, voice `serena`, n=8 seeds/arm,
5 arms x 3 cases = **120 renders**, ~22 min on one RTX 5070.
Driver: `crates/syrinx-eval/examples/instruct_lang_ab.rs`. Raw output: `run.txt`.

Two questions in one run, because they share their renders.

## 1. The confound, measured: is cued-vs-plain just prompt perturbation?

`assemble_text_mode` prepends the instruct block as text tokens, so cued and plain differ
in prompt **length and content**, not only in meaning. At a fixed seed a longer prompt
gives a different AR trajectory whether or not the model attaches any meaning to the
words. A model that treated the instruct as pure noise would still reject the cued-vs-plain
null — which would make every activation number this project has recorded uninterpretable.

So each language got a **sham arm**: a delivery-neutral instruction of comparable length
("Read the sentence that follows" / "请朗读下面这句话").

**The shams are null.** Acoustically none clears the corrected threshold; on the judge the
largest is |t| = 2.07 (p = 0.058), and two are under |t| = 0.75:

| arm | happy | sad | angry |
|---|---|---|---|
| sham-en vs plain (acoustic p) | 0.3164 | 0.7453 | 0.0325 |
| sham-zh vs plain (acoustic p) | 0.4059 | 0.9091 | 0.2862 |
| sham-en judge delta | +0.036 | +0.002 | −0.029 |
| sham-zh judge delta | +0.082 | −0.019 | +0.008 |

**A meaningless instruction of the same size does not move the audio.** The cued-vs-plain
contrast is therefore measuring content, not perturbation, and the earlier n=8 result
stands as a measurement of the thing it claimed to measure. This was worth testing and it
came back clean.

One blemish, recorded rather than smoothed over: `sham-en vs plain` on **angry** is
p=0.0325 — larger than the *real* English cue on the same case (p=0.7417). Uncorrected
that would read as "significant". It is inside the corrected threshold and is best read as
one draw's noise, but it is the reason the sham arm should stay in every future run rather
than being declared settled by this one.

## 2. The language: `InstructLang::default()` is `Zh`, but every render has used `En`

`SplitOptions::default()` sets `En` (`hoist.rs:32`) while `InstructLang::default()` is `Zh`
(`legacy_emotion.rs:38`), with the note that "CV3 follows Chinese instruct prompts best".
So every Qwen render this tree has ever made used the English phrasing — on a Chinese-lab
model whose upstream instruct examples are overwhelmingly Chinese.

### Acoustic — did the delivery change? (exact permutation test, Bonferroni α=0.00238/21)

| contrast | happy | sad | angry |
|---|---|---|---|
| en vs plain | 0.3737 | **0.0019** | 0.7417 |
| zh vs plain | 0.0643 | 0.0569 | 0.2800 |
| en vs sham-en | 0.1265 | 0.0165 | 0.0581 |
| zh vs sham-zh | 0.8443 | 0.0429 | 0.1049 |
| **zh vs en** | 0.6286 | 0.5566 | 0.2522 |

`[sad]`/en reproduces the earlier run **exactly** (p=0.0019) — the harness is
reproducible. Nothing else clears correction, and critically **zh vs en is null in all
three cases**: the two phrasings do not produce acoustically distinguishable audio.

### Judge — did it move toward the named class? (Bonferroni α=0.00417/12)

| arm | happy | sad | angry |
|---|---|---|---|
| en | +0.147 (p=0.030) | +0.018 (p=0.28) | +0.132 (p=0.016) |
| zh | **+0.128 (p=0.0030)** | +0.068 (p=0.0078) | +0.140 (p=0.023) |

`zh` ≥ `en` on all three cases, is the only arm to clear correction anywhere, and is
suggestive where `en` is flatly null (`sad`: p=0.0078 vs p=0.28).

## What this justifies, and what it does not

**The one corrected-significant result is on the class the judge is most biased toward,
and that undercuts it.** The n=1 study recorded both *angry* takes reading overwhelmingly
`happy` (0.458 / 0.642) — `happy` is this judge's default attractor, so "moved toward
happy" is the cheapest artifact it can produce. The `happy`/zh hit should be read as the
weakest of the three, not the strongest, despite having the smallest p.

By the same logic the `sad`/zh result is the **most** interesting one even though it is
only suggestive: `sad` is the judge's worst class (0.17 recall) and the one it almost never
predicts, so a move toward it is hard to produce by bias.

Set against that, the acoustic test says zh and en are indistinguishable. Either the
difference is real but below this test's power at n=8, or the judge is responding to
something that is not the emotion. **These two readings are not separable with the current
judge**, which is exactly the limitation `docs/backends/AFFECT_JUDGES.md` was written
about.

Honest summary:

- **Confirmed:** the sham controls are clean; cue effects are not prompt perturbation.
- **Confirmed:** `[sad]`/en changes the delivery, reproducibly (p=0.0019).
- **Suggestive, consistent, not established:** the Chinese phrasing lands closer to the
  named class than the English one on all three cases.
- **Not established:** that any cue produces its named emotion. One corrected hit, on the
  judge's biased class, on one sentence, one voice, one checkpoint.

## What was done about it

The phrase table was rebuilt in both languages (`crates/syrinx-cue/instruct.toml`), which
was needed regardless — the previous table covered 13 of 51 labels and sent the other 38 to
`format!("Speak in a {label} tone")`. Switching the default to `Zh` is **not** done here:
the evidence is suggestive rather than established, no Chinese-speaking listener has signed
off, and the WER oracle and affect judge are both English-only. That is a maintainer call,
and it should be made against the better judge.
