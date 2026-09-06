# Qwen3-TTS emotional tagging — A/B listening set (2026-09-06)

Six renders, three pairs. Within each pair **only the cue differs** — same text, same
checkpoint, same preset voice, same seed — so anything you hear is the instruction and
nothing else.

- Checkpoint: `Qwen3-TTS-12Hz-1.7B-CustomVoice`, GPU bf16, speaker `serena`, English.
- **Why the 1.7B:** it is the checkpoint that genuinely honours an instruction. On
  `0.6B-CustomVoice` the pair would come out *bit-identical* — upstream gates instruct
  support on a model-size substring test, so the 0.6B accepts the instruction and silently
  discards it. That is recorded per checkpoint in `crates/syrinx-cue/caps.toml`.
- The cue never reaches the model as text. `syrinx-cue` lowers it to an utterance-scoped
  instruction, which is the only expressive channel Qwen has (`Inline::None`).

| # | listen to | text | cue → instruction |
|---|---|---|---|
| 1 | `1-happy-plain.wav` vs `1-happy-tagged.wav` | "We finally heard back, and the news is better than we hoped." | `[happy]` → *"Speak in a happy, cheerful tone"* |
| 2 | `2-sad-plain.wav` vs `2-sad-tagged.wav` | "I waited by the window until the last light went out." | `[sad]` → *"Speak in a sad, sorrowful tone"* |
| 3 | `3-angry-plain.wav` vs `3-angry-tagged.wav` | "You told me it was handled. You looked me in the eye and said it was handled." | `[angry]` → *"Speak in an angry tone"* |

Durations (plain → tagged): happy 3.52 → 3.68 s · sad 3.68 → 3.76 s · angry 5.28 → 5.44 s.
Every pair is byte-different, so the instruction reached generation in all three.

## Intelligibility, and how much to trust it

Scored with the in-tree Whisper (`whisper-base`) against the requested text:

| pair | plain | tagged |
|---|---|---|
| happy | 0.000 | 0.000 |
| sad | 0.000 | 0.091 |
| angry | 0.235 | 0.235 |

`whisper-base` is a small model and a weak judge of synthetic speech, so read these as "the
words are the right words and no cue markup leaked", not as a quality score. The angry pair
is the clearest example: **both** members score exactly 0.235 because the oracle mis-hears
the same word ("handled" → "hand-dote" / "hand-nosed") in both. An error that appears
identically with and without the cue is the transcriber's, not the render's.

An earlier take of pair 1 used "I got the letter this morning…" and the plain render came
back as "A God in letter this morning" (WER 0.30). It was replaced rather than shipped —
the sentence above renders cleanly at 0.000 both ways, which makes for a fairer comparison.

## What the numbers cannot tell you, and why you are listening

Nothing here judges whether the emotion is *right*. WER says the words are correct; SIM-o
(see `../2026-09-06-qwen-gpu/FINDINGS.md`) says a clone landed near its reference. Neither
says "this sounds happy". Per `CLAUDE.md` that judgement is deliberately never automated
into a passing test — it needs ears, which is exactly what this set is for.

Worth listening for specifically: whether the tagged take differs in a way that matches the
*named* emotion, or merely differs. The activation harness can only ever prove the second —
a backend that responded to `[sad]` by shouting would pass every automated check in the
tree.

## Two independent measurements agree on one thing: listen to the angry pair

Added 2026-09-06 after both an acoustic proxy and a trained SER judge were run over these
six files. They disagree about almost everything — different methods, different failure
modes — and they agree about `[angry]`.

**Acoustic correlates** (`syrinx-eval/examples/affect.rs`, no model): the vocab declares
`angry` at arousal **0.90**, near the top of the scale. The tagged take came out *quieter*
(rms 0.054 -> 0.039) and *less* pitch-variable (f0 sd 47.7 -> 40.9) — the wrong direction
for high arousal. By contrast `[sad]` behaved textbook: pitch down 43 Hz, variability
nearly halved, timbre markedly darker.

**The SER judge** (apache-2.0 RAVDESS 8-class, via ONNX): both angry takes read
overwhelmingly **`happy`** (0.458 plain, 0.642 tagged), with the `angry` class flat at
0.171 -> 0.161. That null is unusually informative, because **anger is the one class this
judge is genuinely good at** — 0.80 recall cross-corpus, 0.900 on a real angry clip. It is
not failing to hear anger in general; it is not hearing it here.

| pair | cued-class delta | window floor | verdict |
|---|---|---|---|
| `[happy]` | **+0.262** | 0.222 | moved toward the cued class |
| `[sad]` | −0.033 | 0.067 | inside the noise floor — cannot tell |
| `[angry]` | −0.010 | 0.091 | inside the noise floor — cannot tell |

`[sad]` reading as "cannot tell" is expected rather than damning: sad is the judge's
**worst** class (0.17 recall; it hears real acted sadness as *happy*), and the plain take
already read 0.542 `calm`. A judge with no standing on a class cannot convict the cue on it.

So the question for your ears, concretely: **does `3-angry-tagged.wav` sound angry, or
merely different from `3-angry-plain.wav`?** Two methods say the delivery changed without
becoming anger. Candidate explanations, none yet tested: `serena`'s bright timbre
dominating, the instruction producing arousal without anger's spectral character, or the
cue genuinely not taking on this checkpoint. Your verdict decides which of those is worth
chasing.

Caveat that applies to all of the above: **n = 1 per condition.** Per-render seed control
landed the same day (`d1d0ceb`), so re-running this with 4-8 seeds per condition would
replace the window-spread floor with a real noise model. Until then these are hypotheses to
listen against, not results.

## n=8 with real seeds — and it OVERTURNS the n=1 read above

Everything above this heading was n=1 per condition, with a noise floor *simulated* by
sliding a window over the same render. Per-render seed control (`d1d0ceb`) made the real
measurement possible: 8 draws per condition, 48 renders, 6.2 min on one GPU.

    cue      delivery changed?          toward the named class?
    happy    no   p=0.3737 effect 1.76  +0.147 vs seed-noise +/-0.172   cannot tell
    sad      YES  p=0.0019 effect 2.98  +0.018 vs seed-noise +/-0.045   cannot tell
    angry    no   p=0.7417 effect 1.40  +0.132 vs seed-noise +/-0.136   cannot tell

**The `[happy]` result above is withdrawn.** At n=1 it read +0.262 against a simulated
floor of 0.222 and was reported as "moved toward the cued class" — the one apparent
success. At n=8 the same comparison is +0.147 against a *measured* seed noise of ±0.172,
i.e. inside it, and the acoustic permutation test cannot distinguish the cued renders from
the plain ones at all (p=0.37). The n=1 verdict was an artefact of having one draw and a
stand-in for noise. This is exactly the failure the acoustic module was written to prevent,
and it still took a real noise model to catch it.

**`[sad]` is the one cue that demonstrably works** — p=0.0019 on the permutation test
(n=8 resolves to 1/6435, so this is well clear of the floor), effect 2.98. That converges
with the acoustic proxy, which found textbook sad prosody in that pair: pitch down 43 Hz,
variability nearly halved, timbre markedly darker. The delivery really does change. Whether
it changes *into sadness* is still unanswerable here, because sad is the judge's worst
class (0.17 recall; it hears real acted sadness as happy).

**`[angry]` does not change the delivery at all** (p=0.74). The earlier acoustic reading —
quieter, less pitch-variable, the wrong direction for arousal 0.90 — is better explained as
one draw's noise than as a wrong-direction response. Three independent looks now agree the
angry cue is not taking on this checkpoint.

### What this means, stated carefully

No cue can be shown to move the audio *toward* its named emotion. Every affect delta sits
inside seed noise. That is a statement about the **experiment**, not a verdict on the cue
system: the judge is weak on 6 of 8 classes, n=8 is small, and one preset voice on one
checkpoint is a narrow sample. What can be said is narrower and firmer:

- the cue layer reliably reaches the model (the instruction is built, delivered, and for
  `[sad]` provably alters the audio);
- only `[sad]` produces a change larger than the model's own draw-to-draw variation;
- nothing here establishes that any change is the *named* emotion.

Which is why the six files are still worth your ears. The measurement has become honest
enough to say "cannot tell" three times, and that is the correct answer to give when it is
true.
