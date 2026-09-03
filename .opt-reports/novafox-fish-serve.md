# NovaFox ↔ Fish s2-pro: measured results

All numbers measured on NovaBox (RTX 5070, 12227 MiB), `nvidia-smi` per-process,
faster-whisper large-v3 for WER, ECAPA-TDNN cosine for speaker similarity.
Nothing here is an estimate.

## Criterion status

| # | criterion | target | measured | verdict |
|---|-----------|--------|----------|---------|
| 1 | `serve --fish s2-pro` boots and stays up | — | boots in 11.2 s, holds model + ref codes | **PASS** |
| 2 | `POST /v1/audio/speech` → 44.1 kHz, reference speaker | — | 44100 Hz mono; WER **0.0000**; SIM-o **0.7115** | **PASS** |
| 3 | resident VRAM | ≤ 4096 MiB | **5294 MiB** (5488 with ref clone) | **FAIL by ~1200 MiB** |
| 4 | time to first audio, ~15 words, warm | ≤ 700 ms | **4333 ms** buffered | **FAIL** (see below) |
| 5 | CV2/CV3 paths + parity unchanged | — | 7 PASS / 1 SKIP / 0 FAIL; 11 fish unit tests pass | **PASS** |

## Gap 1 — closed

`serve` now accepts `--fish <s1-mini|s2-pro> --fish-dir <DIR>`, mutually exclusive with
`--cv3`, wired exactly like `synth` (shared `fish_device` / `fish_params` so they cannot
drift). The checkpoint is loaded **once at boot** and the reference clip is encoded to
prompt codes **once at boot**; a request is then only the dual-AR loop plus codec decode.

`syrinx-serve` gained no dependency on `syrinx-fish`: the backend lives in `syrinx-cli`
and is served through a new generic `serve_blocking_dyn(Arc<dyn Synth>)`. The CV2/CV3
entry points are untouched.

One API change was required by criterion 2: `response_format` was a **required** field, so
the specified body `{model, input, voice}` returned 422. It now defaults to `wav`, matching
OpenAI. No frozen test pinned it (every 422 test omits `voice`), so nothing was weakened.

## Gap 2 — mostly closed, with a hard floor

Int4 `Q4_0` over the dual-AR projections, through candle's **fused quantized kernels**
(`QMatMul` → `mul_mat_vec_via_q8_1`), **not** the CosyVoice dequant-on-fetch design.

| | tensors | process VRAM | overhead |
|---|---|---|---|
| bf16 | 9494 MB | 9776 MiB | 282 MB |
| **int4** | **3675 MB** | **5294 MiB** | **1619 MB** |

Weights fall **61 %** (5819 MB saved). Resident VRAM falls only **46 %**, because the
quantized path carries **~1.3 GB more runtime overhead** than the dense path. That
overhead is the reason criterion 3 misses.

Resident LM breakdown (`SYRINX_FISH_MEM_REPORT=1`):

```
int4 2277 MB + dense 1029 MB = 3305 MB   (+ 370 MB codec = 3675 MB)
  dense  797.6 MB  embeddings.weight          <- gathered, not multiplied: cannot be QMatMul
  dense  209.7 MB  codebook_embeddings.weight <- same
  dense   21.0 MB  fast_embeddings.weight
  dense    ~0 MB   norms/biases
```

Things tried that did **not** help:
* `SYRINX_FISH_DMMV=1` (fused `dequantize_mul_mat_vec`, no activation scratch):
  **5456 MiB and 5.47 s** — worse on both axes. The overhead is not the q8_1 scratch.
* Limiting f32 promotion to small tensors: no change (there were no large ones).
* The reference encode is **not** the cause: it costs only 194 MiB (5294 → 5488).

The 797.6 MB embedding table is the largest single remaining item and is not reducible
with candle's quantized types — it is read by `index_select`, and a `QTensor` cannot be
gathered from.

## The speedup nobody asked for

Quantization was requested for memory. It also made the model **6.3× faster**:

| | latency (14 words → 3.7 s audio) | ms/frame | RTF |
|---|---|---|---|
| bf16 | 27 380 ms | 342 | **7.36** |
| int4 | **4 333 ms** | **53.5** | **1.15** |

This is larger than quantization alone should give, and the likely cause is documented in
`.opt-reports/kvcache-speed.md` §4.1: the dense path's `Weights::linear` used
`x.broadcast_matmul(&w.t()?)`, which candle materialises as a **full transposed copy of
every weight on every call** (measured 42.6× on CPU). `QMatMul` bypasses that entirely —
it stores Wᵀ and never broadcasts. So the int4 path is fast partly because it routes
around a defect that still costs the bf16 path.

## Quality is not degraded

| | WER | SIM-o (ECAPA cosine) |
|---|---|---|
| bf16 | 0.0000 | 0.6814 |
| **int4** | **0.0000** | **0.7115** |
| control (unrelated clip) | — | 0.0442 |

Int4 scores marginally *higher* on speaker similarity; treat that as noise, not a win.
The point is it is not worse. The control confirms the metric discriminates.

## Which criterion gives, and by how much

**Criterion 3 gives — by ~1200 MiB.** s2-pro lands at **5.3 GB**, not 4 GB.

What that means for NovaFox's card 1 (12227 MiB):

| co-resident set | total | fits? |
|---|---|---|
| s2-pro + VTube (1.1) | 6.4 GB | yes, 5.8 GB spare |
| s2-pro + VTube + STT (2.1) | 8.5 GB | yes, 3.7 GB spare |
| s2-pro + VTube + STT + Moondream (4.6) | **13.1 GB** | **no, over by 0.9 GB** |

So Fish is usable alongside STT and VTube Studio, but **not** with Moondream on the same
card. Moving Moondream to card 0 (with the LLM) or unloading it while TTS speaks are both
viable; so is putting s2-pro on card 0 if the LLM leaves ≥ 5.5 GB.

**Criterion 4 is not met as delivered, but is no longer blocked by throughput.**
4333 ms is *whole-clip* latency: the Fish backend implements only the buffered
`Synth::synthesize`, so the first byte waits for the last sample. The relevant change is
that RTF went 7.36 → **1.15**. At 53.5 ms/frame and 46.4 ms of audio per frame,
synthesis now nearly keeps pace with playback, so streaming becomes meaningful:

* time to first audio ≈ prefill + one frame ≈ **~50 ms + prefill** — comfortably inside 700 ms
* the stream still drifts at 1.15×, i.e. ~0.15 s of deficit per second spoken, which a
  small pre-roll buffer absorbs for utterances of a few seconds

Implementing `Synth::synthesize_stream` for Fish is the remaining work for criterion 4.
It was not attempted here, so **no streaming number is claimed** — the 4333 ms above is
what the endpoint does today.
