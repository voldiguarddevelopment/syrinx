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
