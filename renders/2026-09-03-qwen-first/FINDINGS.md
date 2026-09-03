# Qwen3-TTS port — first end-to-end renders (2026-09-03)

The first synthesis this port has ever produced, plus a check of the cue layer against
all three Qwen capability tiers. GPU 0 (RTX 5070), bf16, 24 kHz out, `seed = 0`.
Transcripts from the native Rust Whisper oracle (`syrinx stt`, whisper-base), language
forced to `en`.

## 1. The port works

| render | checkpoint | instruct | frames | audio | WER |
|---|---|---|---|---|---|
| `06b-cv-plain` | 0.6B-CustomVoice | — | 34 | 2.72 s | **0.000** |
| `17b-cv-plain` | 1.7B-CustomVoice | — | 35 | 2.80 s | **0.000** |
| `17b-cv-cued` | 1.7B-CustomVoice | "Say this in a soft whisper" | 61 | 4.88 s | 0.750 |
| `17b-vd-cued` | 1.7B-VoiceDesign | "Say this in a soft whisper" | 98 | 7.84 s | 1.000 |
| `06b-cv-cued` | 0.6B-CustomVoice | "Say this in a soft whisper" | 34 | 2.72 s | (identical to plain) |

Text: `Come closer, I have something to tell you.`

Both **plain** paths transcribe perfectly (WER 0.000). Talker load is ~1.3 s and a short
utterance renders in ~4 s on the 1.7B. The pure-Rust chain — tokenizer → talker → code
predictor → split RVQ → codec decoder → WAV — is functional end to end.

## 2. The cue layer is correct at every tier

`caps.toml` claims three different behaviours across the family, and each was confirmed
against the running model rather than by reading the table:

- **Base (0.6B / 1.7B)** — `inline=none`, everything `unsupported`. Every cue is dropped
  with "backend has no channel for this control". Not rendered here: the clone path has no
  driver (nothing in-tree builds an x-vector from a WAV), so `-Base` cannot synthesize yet.
- **0.6B-CustomVoice** — `instruct = accepted`. Cue dropped with the *distinct* reason
  "backend accepts this control but does not act on it". **Proven, not assumed:** the plain
  and cued renders are `md5 24f7866572c0082b3c2ba123d0feffcf` — **bit-identical**. The
  instruction provably never reached generation. This is exactly the per-checkpoint
  distinction that motivated capability rows being per-checkpoint rather than per-crate.
- **1.7B-CustomVoice / VoiceDesign** — `instruct = honored`. The cue becomes an
  utterance-scoped instruction and the audio changes substantially.

Sub-utterance splitting (C2.3) also works. Two conflicting cues on one line:

    [happy] What a wonderful morning. [sad] But then the letter arrived.

lower to **two** requests, split on a word boundary, each with its own instruction
("Speak in a happy, cheerful tone" / "Speak in a sad, sorrowful tone").

The hard invariant holds: no bracket text appears in any transcript.

## 3. DEFECT — an instruct makes the 1.7B talker repeat itself

Reproducible, systematic, and **specific to the instruct path**. The plain path is clean;
every instruction over-generates against a 35-frame baseline:

| instruct | frames | transcript |
|---|---|---|
| *(none)* | 35 | "Come closer, I have something to tell you." |
| "Speak clearly" | 54 | "Come to...come, Towsr. I have something to tell you." |
| "Whisper" | 65 | "Come, Kelser. I have something to tell you. I have something to tell you." |
| "Say this in a soft whisper" | 61 | "Come, Couser. I have something to tell you. Have something to tell you." |
| "Speak in a happy, cheerful tone" | 122 | the sentence, **five times** |

It is repetition, not a slower delivery — the transcripts show the target text emitted
two to five times, and the frame counts scale with it. Instruction length is not the
driver ("Whisper" is one word and still doubles).

**Ruled out:**

- *Sampler defaults.* `SamplingParams::talker()` is already the published
  `generation_config.json` — temperature 0.9, top_p 1.0, top_k 50, repetition_penalty
  1.05. A missing repetition penalty would have been the obvious culprit; it is present.
- *Text duplicated in the prompt.* Adding `--instruct "Whisper"` moves the prompt from 21
  to 28 steps — exactly the `<|im_start|>user\n…<|im_end|>\n` block and nothing else.
- *ASR hallucination.* Frame counts (35 → 61/65/98/122) confirm the audio really is two
  to three times longer.

**Not yet settled:** whether `assemble_text_mode` puts the instruct block in the right
place. It is prepended as its own complete `user` turn before the prefix; the reference
may instead fold it into the same turn as the target text, or use a different role. That
cannot be decided from the checkpoint — `tokenizer_config.json` carries no chat template
and the model card only shows the Python call, not the rendered prompt.

**What would settle it:** the Qwen reference implementation, which is not vendored and has
no `gen-fish-ref.py` equivalent. This is the same gap `QWEN_PORT_STATUS.md` records as
"no Python reference dump for Qwen, so talker/code-predictor parity cannot be gated at
all yet" — and it now has a concrete symptom to chase rather than a hypothetical one.

Until it is resolved, the honest status is: **Qwen renders correctly without an
instruction, and the expressive path is not trustworthy.** The cue layer is doing its job;
what it hands to the backend is where this breaks.
