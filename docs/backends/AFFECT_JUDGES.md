# Judging affect: what could replace the current judge

Research note, 2026-09-06. Prompted by "is there an STT model that could do it itself?"
after the RAVDESS 8-class judge came back weak on 6 of 8 classes.

Every licence below was read from the model card, not recalled.

## The current judge, and why it is the floor not the ceiling

`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` (Apache-2.0), adopted because
it is permissive. Measured cross-corpus accuracy **0.394** against the card's in-domain
0.822; usable for `angry` and `neutral`, worthless for `sad` (0.17) and `fearful` (0.10).
It answers one question — which of 8 RAVDESS classes — and answers it badly for most.

## What actually exists

### A. Models that transcribe AND judge affect in one pass

| model | does | licence | commercial |
|---|---|---|---|
| **SenseVoiceSmall** (FunAudioLLM) | ASR + language ID + **SER** + **audio event detection**, 50+ languages | FunASR Model Open Source License | **yes**, attribution required |
| Qwen2-Audio-7B-Instruct | audio LLM: transcribe, then answer free-form questions about the audio | **Apache-2.0** | yes |
| Qwen2.5-Omni-7B | any-to-any multimodal | Apache-2.0 (tagged `other`) | yes |
| Qwen3-Omni-30B-A3B-Instruct | as above, 30B MoE | Apache-2.0 (tagged `other`) | yes |

### B. Dedicated affect models

| model | does | licence |
|---|---|---|
| `emotion2vec_plus_large` | self-supervised emotion representation, generally stronger than wav2vec2 SER fine-tunes | FunASR licence (permissive + attribution) |
| `audeering/w2v2-msp-dim` | **dimensional** arousal/valence/dominance in 0..1 | CC-BY-NC-SA — **rejected on licence** |

The FunASR Model Open Source License was read in full: it grants "use, copy, modify, and
share", requires attribution and retention of model names, and does **not** restrict
commercial use. It is bespoke rather than OSI-approved, so it wants the same lawyer glance
as anything else non-standard — but it is not NonCommercial.

## Recommendation: SenseVoiceSmall, and it is a strict upgrade

Not because it is newer, but because it answers **more of our questions with one model**:

1. **SER that beats the current judge.** The card claims it meets or exceeds the best
   dedicated SER models; anything above 0.394 cross-corpus is an improvement, and that is a
   low bar to clear.
2. **ASR in the same pass.** Today WER comes from `syrinx-stt` (Whisper) and affect from a
   second model, so the two describe the same audio via two independent front ends.
   SenseVoice gives both from one transcription of one waveform. Fewer moving parts, and no
   chance of the two disagreeing about what was said.
3. **Audio event detection — the capability we currently cannot test at all.** It detects
   laughter, crying, coughing, applause. `vocab.toml` carries `laughs`, `cough`, `breath`
   and `sigh` as **event** cues, and no classifier over 8 emotion classes can say anything
   about them. This is the one item on the list that adds a *new* question rather than
   answering an old one better.
4. **Speed.** Non-autoregressive: ~70 ms per 10 s of audio, ~15x faster than Whisper-large.
   At n=8 per condition that turns a measurement sweep from minutes into seconds.
5. **Runs without Python.** ONNX export is shipped upstream (`demo_onnx.py`), and there is a
   GGUF/llama.cpp runtime. C4.1 requires own-runtime models; `ort` is already in-tree and
   the existing `AffectJudge`/`OnnxJudgeSpec` seam was built for exactly this swap.

## Why an audio LLM is the wrong tool *for measurement* — and the right one for a demo

Qwen2-Audio and the Omni models are tempting: Apache-2.0, in the family we just
standardised on, and you can simply ask "how was this said?". For a **measurement** they are
worse than a small classifier, for three reasons:

- **No calibrated number.** A free-text answer cannot be differenced across conditions. Our
  entire method is "cued mean minus plain mean, against seed noise" — that needs a scalar
  per render, not a sentence.
- **Prompt sensitivity.** The answer moves with the wording of the question, which adds a
  researcher degree of freedom exactly where we are trying to remove them.
- **It can read the words.** This is the sharp one. Our probe sentences are emotionally
  loaded — "But then the letter arrived" is sad *as text*. A model that sees the transcript
  can answer from semantics rather than delivery. Since our A/B pairs hold text identical,
  a text-influenced judge returns the same answer for both and produces a null by
  construction — indistinguishable from "the cue did nothing", which is the exact
  conclusion we are trying to test.

So: an audio LLM is a good **development instrument** ("describe how this was said" while
iterating on a phrasing) and a bad **gate**. If one is adopted, it should be for qualitative
review and never wired into a pass/fail.

## Order of work, if this is picked up

1. Fetch SenseVoiceSmall, export ONNX, anchor the Rust against a Python reference — the
   same discipline that caught the missing `silu` and the two Fish codec defects. The
   existing `scripts/gen-affect-ref.py` and `tests/real_qwen_affect.rs` are the templates.
2. Calibrate it cross-corpus on **CREMA-D** (ODbL, already downloaded), not RAVDESS. State
   its per-class recall before quoting any verdict from it, exactly as was done for the
   current judge.
3. Re-run the n=8 A/B. `[sad]` currently changes the delivery provably (p=0.0019) with no
   judge able to say whether it changed *into* sadness; a better judge is the only thing
   that can answer that.
4. Only then consider event cues (`[laughs]`, `[cough]`), which are untested end to end and
   would be new ground rather than a re-measurement.

Keep the `AffectJudge` trait. Two judges have now been swapped in three days, and the next
licence or capability shift will make it three.
