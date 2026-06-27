<div align="center">

![Syrinx](docs/social-card.png)

# Syrinx

**A local, Rust-served neural TTS + zero-shot voice-cloning engine.**

Clone a voice from seconds of reference audio and render it near real-time on a single
consumer GPU — with editable speech-rate prosody as a typed plan, not a black box.
Inference is **pure Rust** (Candle); no Python on the hot path. (Emotional control and
sub-200 ms streaming are on the roadmap, not yet shipped — see [Status](#what-it-is).)

[![CI](https://github.com/voldiguarddevelopment/syrinx/actions/workflows/ci.yml/badge.svg)](https://github.com/voldiguarddevelopment/syrinx/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-TBD-lightgrey.svg)](#license)
[![Built with Ratchet](https://img.shields.io/badge/built%20with-Ratchet%20%28TDD%20harness%29-5865F2.svg)](#how-this-was-built)

</div>

---

## What it is

Syrinx is a text-to-speech and voice-cloning engine designed to run **entirely on your
own machine**. It pairs an autoregressive semantic language model (for control and
paralinguistics) with a non-autoregressive flow-matching acoustic decoder (for fast,
high-fidelity waveform synthesis) — each paradigm used where it is strongest.

The design goal is the rare combination of **clone quality + expressive range +
low latency + local-only**, on a single RTX&nbsp;4090-class GPU, with a **~270&nbsp;MB
4-bit footprint**.

> **Status — honest snapshot.** **Both the CosyVoice2-0.5B *and* CosyVoice3-0.5B models are
> fully reimplemented in pure-Rust [Candle](https://github.com/huggingface/candle) and
> parity-verified** (`text + ref → 24 kHz audio`; CV2 full-chain 7.7e-5, CV3 components to
> ~1e-5–1e-3). Each has a GPU runtime (RTF ≈ 1.67), a CLI + OpenAI-compatible server, measured
> eval (CV2 SIM-o ≈ 0.74, CV3 ≈ 0.88), **emotion/instruct control**, a real-SineGen quality
> source, an int4-quantized footprint, and (CV2) zh+en text-norm + speech-rate + watermark +
> sample-faithful streaming. CV3 adds a new 22-layer DiT flow + causal-f64 HiFT + v3 tokenizer.
> **Remaining:** sub-200 ms TTFB (CPU is LM-bound) and a full cross-lingual WER sweep. (Earlier
> "emotion needs an instruct checkpoint" was wrong — CV2/CV3 do instruct on the base weights.)
> The real CosyVoice2 / CosyVoice3 ports are the **default build** — a plain `cargo build`
> builds the real Candle model stack. See [Build status](#build-status) and [Roadmap](#roadmap).

---

## Highlights

- 🦀 **Pure-Rust inference.** The whole render path — frontend → LM → prosody plan →
  acoustic decoder → vocoder — is Rust. No Python runtime in production.
- 🎙️ **Zero-shot cloning.** A reference clip → a speaker embedding → a cloned voice,
  no per-speaker fine-tuning.
- ✏️ **Editable prosody.** A typed, serializable `RenderPlan` carries speech-rate and
  pitch (global + per-region). Speech-rate is faithful (≈ 1/rate); training-free **pitch
  is a weak lever** (the vocoder's mel envelope dominates — measured + documented).
  Per-*word*/phoneme targeting needs an aligner the base model doesn't expose.
- ⚡ **Streaming.** Chunk-aware incremental synthesis is implemented; **sub-200 ms TTFB is
  a design target** (needs a causal cached flow + GPU), not yet a measured result, and the
  stream is not yet sample-identical to the batch path.
- 🌍 **Cross-lingual & multi-accent** transfer — *research-tracked, not yet validated*
  (needs an ASR-based eval).
- 🔬 **Parity-gated correctness.** Every numerical stage of the real model is checked
  against the PyTorch reference within tolerance — "done" means the frozen test passes,
  never an assertion.
- 🔒 **Real, honest watermark.** A spread-spectrum watermark on every output, imperceptible
  and detectable after light processing — *not* adversarially robust (see Ethics).

---

## Architecture

```
text ─▶ [Frontend] ─▶ [AR Semantic LM] ─▶ [Prosody Plan] ─▶ [NAR Flow Decoder] ─▶ [Vocoder] ─▶ 48 kHz
        (Rust,          (control,           (editable          (acoustic, chunk-      (HiFi-GAN/
       deterministic)  paralinguistics)     dur + pitch)        aware, speaker-       Vocos)
                                                                cond.)  ▲
                                          [Speaker Encoder] ───────────┘
                                          (embed · blend · morph · attributes)
```

**Two paradigms, each where it wins:** an **autoregressive** semantic LM owns control
and paralinguistic tokens; a **non-autoregressive** flow-matching decoder owns the
acoustic frames so first-byte latency is one chunk, not one utterance.

> The diagram is the original aspirational design; the **real shipped pipeline** is the
> CosyVoice2 / CosyVoice3 ports documented under *Build status* below (24 kHz; CV2 uses a
> conformer flow + HiFT, CV3 a 22-layer DiT flow + causal HiFT).

**Realized 4-bit footprint (int4, opt-in):** **CV2 ≈ 388 MB** (the unused untied `lm_head`
dropped) · **CV3 ≈ 488 MB** (its 22-layer DiT flow dominates). The early "~270 MB" budget
under-counted the Qwen2-0.5B body; int4 is a size win, not a speed win (dequant-on-fetch).

---

## Workspace layout

A nine-crate Rust workspace; each crate owns one stage of the real pipeline.

| Crate | Responsibility |
|-------|----------------|
| [`syrinx-frontend`](crates/syrinx-frontend) | Qwen2 BPE text tokenizer, wetext-style normalization (`tn`), kaldi fbank + prompt mel, `speech_tokenizer_v2/v3.onnx` |
| [`syrinx-lm`](crates/syrinx-lm) | Real CV2/CV3 Qwen2-0.5B LM forward (Candle) + KV-cache autoregressive generation |
| [`syrinx-speaker`](crates/syrinx-speaker) | Real CAM++ x-vector speaker encoder (Candle) |
| [`syrinx-acoustic`](crates/syrinx-acoustic) | Real flow-matching mel decoder — CV2 conformer / CV3 22-layer DiT (Candle) |
| [`syrinx-vocoder`](crates/syrinx-vocoder) | Real HiFT vocoder — CV2 + CV3 causal-f64 variant (Candle) |
| [`syrinx-prosody`](crates/syrinx-prosody) | Editable prosody plan model + render-time rate/pitch transforms |
| [`syrinx-serve`](crates/syrinx-serve) | End-to-end `Synthesizer` capstone, OpenAI-compatible `/v1/audio` server, watermarking |
| [`syrinx-eval`](crates/syrinx-eval) | Measured SIM-o/WER/MOS/RTF/TTFB metrics over the real synthesizer |
| [`syrinx-cli`](crates/syrinx-cli) | `syrinx synth\|serve\|stream` (CV2 + `--cv3`) |

---

## Build status

Syrinx is built **test-first behind deterministic gates** (see
[How this was built](#how-this-was-built)). A task is `done` only when its frozen tests
pass — there are no stubbed greens.

**✅ Real CosyVoice2 model — DONE (a standalone, near-real-time Rust TTS)**

The real **CosyVoice2-0.5B** model is reimplemented in pure-Rust
**[Candle](https://github.com/huggingface/candle)** and verified numerically against the
real PyTorch model. The real ports are the **default build** (Candle is a normal
dependency):

- **LM** — Qwen2-0.5B forward + **KV-cache autoregressive generation**: logits 1.3e-4,
  per-step gen logits 2.9e-5, argmax-exact.
- **Speaker** — CAM++ x-vector (architecture recovered from the `campplus.onnx` graph): 1.3e-5, cosine 1.0.
- **Acoustic** — flow-matching mel (conformer + CFM Euler ODE + zero-shot prompt conditioning): mel 1.3e-5.
- **Vocoder** — HiFT (upsample + Snake ResBlocks + iSTFT-via-inverse-DFT): waveform 5.2e-5.
- **Frontend** — Qwen BPE tokenizer (exact) · kaldi fbank + prompt mel (1e-3) · `speech_tokenizer_v2.onnx` (exact, via `ort`).
- **`Synthesizer`** (`syrinx-serve::synth`) — `synthesize(text, ref_audio) → 24 kHz audio`,
  full-chain deterministic parity **7.7e-5**. **No Python in the inference path.**
- **GPU runtime** (`cuda` feature, Candle-CUDA) — full synth **~26× faster**, **RTF ≈ 1.67**
  (near real-time) on a single consumer GPU.

The parity fixtures (real weights + Python reference dumps) live on the model box, so these
tests are **env-gated and skip cleanly in CI** — the build stays green without the weights,
while the real path runs for real where the weights exist.

**✅ Real CosyVoice3 model — DONE (a second pure-Rust CosyVoice, feature-complete)**

The newer **CosyVoice3-0.5B** (`Fun-CosyVoice3-0.5B-2512`) is now *also* a full pure-Rust
Candle port, built the same parity-driven way and reusing ~70 % of the CV2 code (CAM++
speaker as-is, the Qwen2 LM body, the CFM Euler/CFG solver, the matcha mel + `ort` wiring):

- **LM** (`CosyVoice3LM`) — Qwen2 body + CV3 head (sos/task from `speech_embedding`, bias-free
  `llm_decoder`): teacher-forced logits **2.67e-5**.
- **Flow** — a **new 22-layer DiT** estimator (dim 1024, rotary + AdaLN) replacing CV2's U-Net,
  with a PreLookahead front-end + vocab-6561 input embedding: **2.27e-3** (the fp32 accumulation
  floor — proven: torch's own fp32-vs-fp64 on this DiT is 1.34e-3).
- **Vocoder** — `CausalHiFTGenerator` (causal convs + a **float64** f0-predictor): audio **4.9e-5**.
- **Frontend** — `speech_tokenizer_v3.onnx` (87/87 ids exact) + the matcha prompt-mel (**3.72e-5**).
- **Live synthesis** (`Cv3Synthesizer`, `text + ref → 24 kHz`) — measured **SIM-o 0.88** (voice clone,
  *better* than CV2's 0.74) and **MOS-proxy 2.21** (with the real SineGen source). The `<|endofprompt|>`
  marker is required for all CV3 inference.
- **Feature-complete:** CLI (`synth/serve --cv3`) · HTTP server (`Cv3RealSynth`) · 5-metric eval
  (`evaluate_cv3`) · emotion/instruct (`synthesize_instruct`) · real-SineGen quality source ·
  RL-LM variant (`llm.rl`) · int4 footprint (~488 MB) · **chunked-causal streaming**
  (`synthesize_streaming`, the DiT chunk-mask — finalized frames bit-stable, 0.0 vs 2.28 non-causal).
- **Measured extras:** the **RL LM is the quality winner** (SIM-o 0.845 / MOS 3.03 with the quality
  source) · **cross-lingual** zh-voice → English carries (SIM-o 0.76, MOS 4.28) · int4 is honestly a
  **size-only, lossy + slow** tradeoff (SIM-o 0.47, RTF 243) — opt-in, not the default.

> The hard win was the live decode: a repetition-aware-sampling fallback that masked the repeated
> token (which the reference doesn't) collapsed generation; a pin-reference-token diagnostic proved
> the model itself was correct (pinned → SIM-o 0.69) and isolated the one-line fix that took live
> SIM-o **0.24 → 0.88**.

---

## The parity approach (why the numbers are trustworthy)

Every numerical stage of the real port is gated against **golden fixtures dumped from the
real PyTorch CosyVoice2 / CosyVoice3 models** — the real pretrained weights and the
reference's intermediate activations. The Rust (Candle) stage must match within documented
tolerances (≈1e-4/1e-5 per stage, down to the fp32 accumulation floor on the CV3 DiT), so
"done" means a frozen parity test passes against the real model, never an assertion. See
[`PARITY.md`](PARITY.md) and [`REFERENCE.md`](REFERENCE.md).

---

## Getting started

> **Heads up:** the **default build is the real CosyVoice2 / CosyVoice3 model stack** —
> a plain `cargo build` pulls Candle and compiles the real ports. Weights are supplied at
> runtime via the `SYRINX_*` env vars (or `--model-dir`); the parity tests are env-gated and
> skip without them. `--features cuda` adds the GPU path.

```bash
# Build the whole workspace (default = the real model stack, Candle-backed)
cargo build --workspace

# Run the test suite (real parity/eval tests skip cleanly without on-disk weights)
cargo test --workspace

# CLI help
cargo run -p syrinx-cli -- --help
```

**Real synthesis** — `text + reference clip → 24 kHz wav`. Pick the model with `--cv3`
(add `--features cuda` + `--cuda` for the GPU path):

```bash
# CosyVoice2 (default): weights via SYRINX_*_WEIGHTS env (or --model-dir)
cargo run -p syrinx-cli -- synth \
  --text "Hello from Syrinx." --prompt-text "<ref transcript>" \
  --ref-wav ref.wav --out out.wav

# CosyVoice3: same CLI, add --cv3 (weights via SYRINX_CV3_*; v3 speech tokenizer)
cargo run -p syrinx-cli -- synth --cv3 \
  --text "收到好友从远方寄来的生日礼物。" --prompt-text "希望你以后能够做的比我还好呦。" \
  --ref-wav ref.wav --out out_cv3.wav

# OpenAI-compatible server (either model): `serve` / `serve --cv3`
cargo run -p syrinx-cli -- serve --cv3 --ref-wav ref.wav --port 8080
curl -s localhost:8080/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model":"syrinx-cv3","input":"hello","voice":"v","response_format":"wav"}' -o out.wav
```

**Requirements:** a stable Rust toolchain. Synthesis needs the CosyVoice2-0.5B **or**
CosyVoice3-0.5B weights + reference fixtures on disk (the parity tests are env-gated on
them); the **`cuda`** speed path needs an NVIDIA GPU + the Candle-CUDA toolchain (~26×
faster, near real-time).

---

## How this was built

Syrinx is built by **[Ratchet](https://github.com/voldiguarddevelopment/Ratchet)**, a
hardened autonomous TDD harness. Every change goes through a strict gate cascade —
integrity → checker → compile → frozen tests → mutation — and the project's three
documents (`plan.md` / `spec.md` / `list.md`) are reconciled against the code on every
pass. The core rule: **no stubs, no simplified implementations, no fake passes** — a
green that isn't real is rejected by construction. State lives in disk + git history, so
each pass re-derives correctness from scratch.

That is why the build status above is precise about what is *proven* versus *pending*:
the harness will not mark a task done on belief.

---

## Roadmap

**Done (real, verified):**
- [x] Nine-crate workspace + CI (the real ports are the default build)
- [x] **Real CosyVoice2-0.5B port** — LM (+ KV-cache gen) · CAM++ speaker · flow-matching · HiFT · frontend, all Candle, all parity-verified
- [x] **End-to-end `Synthesizer`** — `text + ref → audio`, full-chain parity 7.7e-5, no Python on the hot path
- [x] **GPU runtime** (Candle-CUDA) — ~26×, RTF ≈ 1.67 (near real-time on a consumer GPU)
- [x] **CLI + server** — `syrinx synth|serve|stream`; OpenAI-compatible `POST /v1/audio/speech` returns real audio
- [x] **Text normalization** — wetext-style zh+en (~95% match to the reference), wired into the real path (`tn` feature)
- [x] **Editable prosody** — speech-rate (faithful, ≈1/rate) + a typed `RenderPlan`; **pitch is a weak training-free lever** (the HiFT mel filter dominates perceived pitch — measured + documented)
- [x] **Measured eval — 5/5, no stub constants** — SIM-o clone fidelity (≈0.74), **WER** (Whisper CER ≈0%), **MOS-proxy** (UTMOS), RTF, TTFB. WER/MOS run via eval-side helper models (Whisper / UTMOS); the inference path stays pure-Rust
- [x] **int4 (Q4_0) LM quant** — ~2.5× (2449 → 986 MB, SIM-o 0.72 preserved); the f16 embedding tables are the remaining bulk
- [x] **Output watermark** — spread-spectrum, imperceptible + detectable after light processing (see *Ethics*)
- [x] **Real CosyVoice3-0.5B port — feature-complete** — LM (2.67e-5) · **new 22-layer DiT flow** (2.27e-3, fp32 floor) · causal f64 HiFT (4.9e-5) · v3 tokenizer (exact) · frontend (3.72e-5); live synth **SIM-o 0.88 / MOS 2.21**; CLI/server/eval/emotion/quality-source/RL-LM/int4 all wired (`--cv3`). ~70% CV2 reuse; see *Real CosyVoice3 model* above.

**Not yet (honest):**
- [x] **Sample-faithful streaming** — CV2's chunked-causal attention mask (same weights) makes the streamed mel frames **bit-stable** (`real_flow_stream_consistency`: 0.0 diff vs 0.53 for the old non-causal path), and the **streamed audio is intelligible — Whisper CER 0.0**, identical to batch. (Streamed audio is *not* sample-identical to the batch — CV2's streaming cross-fades by design; details in [`STREAMING.md`](crates/syrinx-acoustic/docs/STREAMING.md).) Sub-200 ms TTFB remains a design target (CPU TTFB is LM-bound).
- [x] **Emotion / instruct control** — `synthesize_instruct(tts, instruct, ref)` on the **same CosyVoice2-0.5B weights** (no separate checkpoint — CV2 unified instruct into the base): the instruct text takes the LM prompt-text role + the prompt speech tokens are dropped, while the flow keeps the cloned voice. Verified — emotions measurably change the output (sad / cheerful / neutral differ in MOS + SIM-o) while preserving speaker identity.
- [ ] **Cross-lingual eval set** — the SIM-o/WER/MOS harness already handles it; just needs a multilingual frozen eval set + a sweep (the Whisper helper is language-aware).
- [x] **Smaller footprint** — int4 LM linears + int4 embeddings + int4 flow/HiFT/speaker + **dropping the unused `lm_head`** (519 MB of dead weight CV2's speech path never calls) land the whole model at **388 MB** (from ~2983 fp32, **~7.7×**). The README's "270 MB" budget under-counted the Qwen2-0.5B body; 388 MB is the honest 4-bit floor. ⚠️ The int4 dequant-on-fetch is **slow to load/infer** — it's an **opt-in** path (`load_quantized`), not the default; fast int4 kernels are the follow-up.
- [x] **Perceptual-quality source + CFM noise** — `synthesize_quality` uses the real random-phase NSF SineGen (8 overtones + uv mask + Gaussian breath + learned source merge) **and** a seeded standard-normal CFM init (the model's `rand_noise`) instead of the deterministic zero-phase source + zeros. Measured UTMOS: **2.03 → 2.21 (source) → 2.36 (+`z`)**. Remaining quality headroom is the capped-gen mel + the model's true RNG byte-stream (not portable).
- [x] **Consolidation** — the original Ratchet build-methodology scaffold (the GPU-less,
  FNV-name-seeded "deterministic spec engine" proxy: `syrinx-core`, `syrinx-stream`, the toy
  LM/attention/tensor modules, the eval-harness skeleton, the toy prosody/eval tests) has been
  **removed**; the real CosyVoice2/3 ports are now the primary/default build.

See [`DESIGN.md`](DESIGN.md) for the full task-based plan.

---

## Ethics & consent

Voice cloning is powerful and abusable. Syrinx can embed a **spread-spectrum watermark**
in every synthesized output (`Synthesizer::synthesize_watermarked`): key-seeded,
imperceptible (≈ −48 dBFS), and detectable after **light** processing — high-bitrate
re-encoding, gain changes, light noise, and integer-sample crops. It is **not**
adversarially robust: aggressive low-bitrate MP3/Opus, time-stretch/resample, or
deliberate removal defeat it — that needs a *learned*, perceptually-masked scheme
(AudioSeal / WavMark), tracked as future work. See
[`crates/syrinx-serve/docs/WATERMARK.md`](crates/syrinx-serve/docs/WATERMARK.md) for the
honest robustness boundary. Cloning is meant to be gated behind a usage policy — do not
clone a voice you do not have the right to use.

---

## License

License TBD. Until a license file is added, all rights reserved by the project owners.

<div align="center">
<sub>Built with 🦀 and <a href="https://github.com/voldiguarddevelopment/Ratchet">Ratchet</a> · voldiguarddevelopment</sub>
</div>
