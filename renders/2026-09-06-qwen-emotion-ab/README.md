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
