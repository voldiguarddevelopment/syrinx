<div align="center">

![Syrinx](docs/social-card.png)

# Syrinx

**A local, Rust-served neural TTS + zero-shot voice-cloning engine.**

Clone a voice from seconds of reference audio, render it with full prosodic and
emotional control, stream it sub-200&nbsp;ms on a single consumer GPU — and edit the
prosody as a plan, not a black box. Inference is **pure Rust**; no Python on the hot path.

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

> **Status — honest snapshot.** The deterministic engine (text frontend, the inference
> *runtime*, the prosody control surface, and audio streaming) is **built and
> test-gated** in Rust. The pieces that require real pretrained weights and a GPU
> (weight-swap, cloning *quality*, the DiT decoder / vocoder / speaker-encoder
> *quality* eval) are scaffolded with a parity-checked reference architecture and are
> the next milestone. See [Build status](#build-status).

---

## Highlights

- 🦀 **Pure-Rust inference.** The whole render path — frontend → LM → prosody plan →
  acoustic decoder → vocoder — is Rust. No Python runtime in production.
- 🎙️ **Zero-shot cloning.** A reference clip → a speaker embedding → a cloned voice,
  no per-speaker fine-tuning.
- ✏️ **Editable prosody plan.** Duration and pitch are a typed, serializable plan you
  can edit per-word and per-phoneme, then re-render — decoupled from the renderer.
- ⚡ **Streaming-first.** Chunk-aware causal flow matching targets sub-200&nbsp;ms
  time-to-first-byte; ring-buffered packet streaming with a telephony (8&nbsp;kHz) path.
- 🌍 **Cross-lingual & multi-accent** transfer (research-tracked attribute control).
- 🔬 **Parity-gated correctness.** Every numerical stage is checked against a
  byte-exact reference within tolerance — a stage is "done" only when its frozen test
  passes, never by assertion.
- 🔒 **Watermarking + consent** are first-class policy, not an afterthought.

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

**Parameter budget (pre-quant ~420M):** LM ~250M · acoustic ~120M · speaker enc ~30M ·
vocoder ~20M → **~270&nbsp;MB at 4-bit**.

---

## Workspace layout

An eleven-crate Rust workspace; each crate owns one stage of the pipeline.

| Crate | Responsibility |
|-------|----------------|
| [`syrinx-frontend`](crates/syrinx-frontend) | Text normalization, numbers/dates, G2P, SSML, lexicon, heteronyms, context windowing |
| [`syrinx-core`](crates/syrinx-core) | Tensor ops, weight loading, quantization, device management |
| [`syrinx-lm`](crates/syrinx-lm) | Autoregressive semantic LM forward pass + paralinguistic tokens |
| [`syrinx-speaker`](crates/syrinx-speaker) | Speaker encoder, embedding store, blend/morph, attributes |
| [`syrinx-acoustic`](crates/syrinx-acoustic) | Flow-matching decoder (DiT blocks + ODE solver), chunk-aware streaming |
| [`syrinx-vocoder`](crates/syrinx-vocoder) | HiFi-GAN/Vocos waveform synthesis, 48 kHz / 8 kHz paths |
| [`syrinx-prosody`](crates/syrinx-prosody) | Editable prosody plan model + override API |
| [`syrinx-stream`](crates/syrinx-stream) | Packet streaming, ring buffer, audio out, TTFB path |
| [`syrinx-serve`](crates/syrinx-serve) | Server, OpenAI-compatible `/v1/audio`, watermarking |
| [`syrinx-eval`](crates/syrinx-eval) | MOS/SIM-o/WER/latency harness, frozen eval-set runner |
| [`syrinx-cli`](crates/syrinx-cli) | Local runner / dev harness |

---

## Build status

Syrinx is built **test-first behind deterministic gates** (see
[How this was built](#how-this-was-built)). A task is `done` only when its frozen tests
pass — there are no stubbed greens.

**✅ Done — the deterministic engine + parity foundation**
- **Text frontend:** normalization, number/date/currency expansion, acronym + custom
  lexicon, G2P, heteronym resolution, SSML subset, punctuation→prosody, context
  windowing, breathing/pacing, and the typed frontend→LM contract.
- **LM inference runtime:** the full transformer forward in Rust —
  `embed → 4× (RoPE multi-head attention + SwiGLU block, pre-RMSNorm residuals) →
  final RMSNorm → untied head` — **byte-parity to a pure-Python reference within
  1e-3**, with the transformer blocks pinned numerically at activation scale.
- **Prosody control:** speech-rate scaling, intonation contours, phoneme-level plan
  edits, and plan round-tripping.
- **Audio streaming:** packet buffering and the 8&nbsp;kHz telephony resample path.
- **Substrate:** the eleven-crate workspace, core tensor ops, the deterministic
  name-seeded weight generator, and the parity harness.

**✅ Real CosyVoice2 model — DONE (a standalone, near-real-time Rust TTS)**

On top of the deterministic spec engine, the real **CosyVoice2-0.5B** model is now
reimplemented in pure-Rust **[Candle](https://github.com/huggingface/candle)** and verified
numerically against the real PyTorch model — every stage behind a `real` cargo feature
(the default build stays Candle-free):

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
tests are **env-gated and skip cleanly in CI** — the default build + CI stay green and
Candle-free, while the real path runs for real where the weights exist.

---

## The parity approach (why the numbers are trustworthy)

The inference runtime is built against a **concrete reference architecture** with
**deterministic weights derived from each tensor's name** (FNV-1a-64 hash → xorshift64
PRNG → f32), implemented **identically in Python and Rust** — so there is no weights
file to ship and the two implementations must agree *bit-for-bit on the algorithm*.

A pure-Python reference ([`reference.py`](REFERENCE.md)) emits **golden fixtures**; the
Rust code is gated against them within documented tolerances (1e-4 for single ops,
1e-3 for the full forward, 1e-4 for intermediate activations where the signal is small).
Real pretrained weights later drop into the *same verified shapes* — the structure is
already proven correct. See [`PARITY.md`](PARITY.md) and [`REFERENCE.md`](REFERENCE.md).

---

## Getting started

> **Heads up:** the default build is the deterministic spec engine (Candle-free). The
> **real CosyVoice2 model** — full `text + ref → audio` synthesis — runs behind the
> `real` / `cuda` features against on-disk weights; see *Real CosyVoice2 model* above.

```bash
# Build the whole workspace
cargo build --workspace

# Run the full test suite (frozen parity + property tests)
cargo test --workspace

# Explore a stage, e.g. the text frontend
cargo run -p syrinx-cli -- --help
```

**Requirements:** a stable Rust toolchain for the default build. The **real** model path
(`--features real`) additionally needs the CosyVoice2-0.5B weights + reference fixtures on
disk (the parity tests are env-gated on them); the **`cuda`** speed path needs an NVIDIA GPU
+ the Candle-CUDA toolchain (~26× faster, near real-time).

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

- [x] Eleven-crate workspace + CI
- [x] Deterministic spec engine — frontend, LM forward, prosody, streaming (parity-gated)
- [x] **Real CosyVoice2-0.5B port** — LM (+ KV-cache gen) · CAM++ speaker · flow-matching · HiFT · frontend, all Candle, all parity-verified
- [x] **End-to-end `Synthesizer`** — `text + ref → audio`, full-chain parity 7.7e-5, no Python on the hot path
- [x] **GPU runtime** (Candle-CUDA) — ~26×, RTF ≈ 1.67 (near real-time)
- [ ] Chunk-aware **streaming** synthesis (sub-200 ms TTFB) — *in progress*
- [ ] Syrinx's **control layer**: editable prosody plan, emotion/intensity, paralinguistic detail — *in progress*
- [ ] 4-bit quantization · cloning / cross-lingual / perceptual **quality** eval
- [ ] Server `/v1/audio` end-to-end + watermarking on every output

See [`DESIGN.md`](DESIGN.md) for the full task-based plan and resolved design decisions.

---

## Ethics & consent

Voice cloning is powerful and abusable. Syrinx treats **consent and watermarking as
requirements, not features**: every synthesized output is intended to carry a robust,
post-edit-detectable watermark, and cloning is gated behind a usage policy. Do not
clone a voice you do not have the right to use.

---

## License

License TBD. Until a license file is added, all rights reserved by the project owners.

<div align="center">
<sub>Built with 🦀 and <a href="https://github.com/voldiguarddevelopment/Ratchet">Ratchet</a> · voldiguarddevelopment</sub>
</div>
