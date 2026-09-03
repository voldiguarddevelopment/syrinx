# Qwen3-TTS `-Base` — the clone path, wired and anchored (2026-09-03)

`QWEN_PORT_STATUS.md` recorded that "the voice-clone path has no driver — `realize_plan`
takes the x-vector and reference frames as arguments and nothing in-tree builds them from
a WAV", so the two `-Base` checkpoints had never synthesized anything. They do now, and
the speaker encoder is anchored to the reference.

Everything below is CPU/float32 — the parity path. Reference: `QwenLM/Qwen3-TTS` @
`022e286` via `~/.venvs/qwen`. Fixture: `scripts/gen-qwen-ref-speaker.py` ->
`/home/floofy/parity-qwen/speaker.safetensors`. Gate:
`tests/real_qwen_speaker_parity.rs`.

## 1. The x-vector matches the reference to 1e-6

The generator drives the reference's own `create_voice_clone_prompt` and hooks two points
on the real call path — `extract_speaker_embedding` (the 24 kHz clip it receives, i.e.
already resampled) and `Qwen3TTSSpeakerEncoder.forward` (its input mel and output vector,
captured separately). The hooked output is cross-checked against the vector the returned
`VoiceClonePromptItem` actually carries, so a mis-placed hook cannot pass silently.

Three anchors, deliberately, so a mismatch localizes to the mel or to the encoder rather
than to "the speaker path":

| anchor | input | max abs diff | tolerance |
|---|---|---|---|
| mel `[937, 128]` | the reference's resampled clip | **0.0002508** | 5e-3 |
| x-vector, encoder alone | the reference's **own** mel | **0.0000010** | 1e-4 |
| x-vector, end to end (`embed`) | the reference's resampled clip | **0.0000010** | 1e-4 |

2048 components, L2 norm **17.028715** on both sides. The mel's looser 2.5e-4 is
dense-f32-DFT versus `torch.stft` summation order; it moves the x-vector by 1.0e-6, which
is why the end-to-end anchor lands on the same figure as the reference-mel one.

**No bug in `speaker.rs`.** It was checked against the reference source specifically for
the `silu` class of error — reflect-`same` padding, the *cascaded* Res2Net sum, the MFA
skipping `blocks.0`, population variance, the always-all-ones ASP mask — and it is right.

## 2. The gate has teeth, and each perturbation reddens only what is downstream

| perturbation | mel | encoder from ref mel | end to end | driver |
|---|---|---|---|---|
| `tdnn` loses its `ReLU` (the `text_projection` bug's exact shape) | ok 0.00025 | **FAIL 556.68** | **FAIL 556.68** | **FAIL** cos 0.2678 |
| Hann window symmetric instead of periodic (one-sample off-by-one) | **FAIL 0.0629** | ok 0.0000010 | **FAIL 0.00034** | ok |

The second row is why the x-vector bound is 1e-4 rather than something rounder: that
off-by-one is 12x tolerance at the mel but only **3.4x** at the x-vector. A looser bound
would have let a real defect through.

## 3. A real bug, found in resampling — and it was found by effect, not by reading

Every reference clip here is 16 kHz and the reference resamples with `librosa`
(soxr HQ), which is not reproducible in-tree. So the fixture stores the *post-resample*
clip and every numeric anchor starts from it: a resampler difference must never be able
to masquerade as an encoder fault. The driver still needs a resampler, so that half is
gated **by effect** — cosine >= 0.9995 against the reference x-vector.

That assertion fired immediately. The first implementation reused
`syrinx_serve::wavio::resample` (Lanczos-16, cutoff at Nyquist) and scored **cosine
0.996886**. An FFT of the difference put all of it above 8 kHz — the *input* Nyquist,
where the reference carries 1.5e-8 total energy and we were leaking **2.8e+2** of
imaging. Since `fmax` is 12 kHz, the top ~30 of the 128 mel bands saw the log floor for
the reference and an invented signal for us, which `log` then magnified.

Sweeping the kernel against the real encoder:

| lobes | passband | cosine | relative L2 |
|---|---|---|---|
| 16 | 1.00 | 0.996886 | 0.0841 |
| 32 | 0.96 | 0.999771 | 0.0229 |
| 64 | 0.95 | 0.999802 | 0.0199 |
| **64** | **0.96** | **0.999977** | **0.0070** |
| 128 | 0.96 | 0.999982 | 0.0062 |
| 128 | 0.94 | 0.997751 | 0.0673 |

64 lobes / 0.96 passband is the knee: doubling the kernel again buys 6e-6 of cosine for
twice the work, and moving the passband either way is worse — 0.98 leaves the imaging in,
0.94 starts eating real 7.5 kHz speech.

The `syrinx-serve` and `syrinx-stt` copies of that resampler were deliberately left
alone: different crates, different consumers, and this measurement is evidence about
*this* path only. Whether they have the same weakness is a separate question that
deserves its own measurement rather than a copied conclusion.

## 4. `-Base` synthesizes — both modes, WER 0.000

`crates/syrinx-qwen/examples/clone.rs`: WAV -> mono f32 -> resample -> `SpeakerEncoder::embed`
-> (`MimiEncoder::encode` for ICL) -> `build_voice_clone` -> `realize_plan` -> generate ->
split RVQ -> decoder -> WAV. Text `Come closer, I have something to tell you.`, seed 0.

| render | mode | prompt | frames | audio | WER |
|---|---|---|---|---|---|
| `xvec-only.wav` | x-vector only | 10 steps | 32 | 2.56 s | **0.000** |
| `icl.wav` | in-context, 125 reference frames | 135 steps | 37 | 2.96 s | **0.000** |

Both transcripts came back exactly right, re-verified independently against the Whisper
oracle after the agent reported them. Prompt geometry matches the reference's arithmetic
(ICL: 40 text positions padded to the 126-position codec stream, trailing `[tts_pad]`),
and the driver reproduces the reference's decode-`cat(ref_code, generated)`-then-cut-at-
`125/162` behaviour rather than decoding the halves separately.

## 5. Still open

- **CUDA / bf16 on this path has never been run.** The CUDA tolerances in the test are
  labelled slack, not evidence. Do not cite them as parity.
- **0.6B-Base is unanchored against a real clip** — the fixture is 1.7B. Re-running the
  generator with `--ckpt …0.6B-Base` closes it.
- **Clone *quality* is unjudged.** WER says the words are right; whether it sounds like
  the reference speaker is SIM-o / perceptual, and per `CLAUDE.md` that is
  blocked-on-human, never automated into a green.
- **Resampling is outside the numeric gate** by construction (see §3). It is bounded by
  effect at cosine >= 0.9995, not anchored tensor-for-tensor.
