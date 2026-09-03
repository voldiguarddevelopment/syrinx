# s2-pro codec decode: the output-length VRAM spike

Investigated 2026-08-29. Scope: `crates/syrinx-fish/src/s2/codec.rs`.
Hardware facts taken as given (measured by the operator, not re-derived): 2× RTX 5070,
12227 MiB each, sm_120; bf16 s2-pro resident baseline 10.5 GB; ~800 MiB transient spike
during synthesis; renders succeed at ~178 frames (8.3 s) and OOM at 256 frames (11.9 s).

---

## 1. Root cause — proven, with numbers

### 1.1 The real geometry

Read out of the actual checkpoint (`/home/floofy/models/s2-pro/codec.pth`, 541 keys,
unpickled without torch — a shape dump only). The decoder half matches what `codec.rs`
assumes, key for key:

| stage | op | shape | length |
|---|---|---|---|
| `decoder.model.0.conv` | conv k7 s1 | `[1536, 1024, 7]` | ×1 |
| `decoder.model.1.block.1` | convT k16 s8 | `[1536, 768, 16]` | ×8 |
| `decoder.model.1.block.{2,3,4}` | 3 × ResUnit conv k7 d∈{1,3,9} | `[768, 768, 7]` | ×1 |
| `decoder.model.2.block.1` | convT k16 s8 | `[768, 384, 16]` | ×8 |
| `decoder.model.2.block.{2,3,4}` | 3 × ResUnit k7 d∈{1,3,9} | `[384, 384, 7]` | ×1 |
| `decoder.model.3.block.1` | convT k8 s4 | `[384, 192, 8]` | ×4 |
| `decoder.model.3.block.{2,3,4}` | 3 × ResUnit k7 d∈{1,3,9} | `[192, 192, 7]` | ×1 |
| `decoder.model.4.block.1` | convT k4 s2 | `[192, 96, 4]` | ×2 |
| `decoder.model.4.block.{2,3,4}` | 3 × ResUnit k7 d∈{1,3,9} | `[96, 96, 7]` | ×1 |
| `decoder.model.6.conv` | conv k7 s1 | `[1, 96, 7]` | ×1 |

Plus `quantizer.upsample.{0,1}` = ConvTranspose(k2,s2) + ConvNeXt(dwconv k7) at 1024 ch,
×4. Total 4 × 512 = **2048× = `frame_hop`**. Compute dtype on CUDA is **bf16, 2 B/elt**
(`s2/mod.rs:113`).

### 1.2 Where the memory actually goes: candle's `im2col`

The workspace builds candle 0.8.4 **without the `cudnn` feature**, so
`CudaStorage::conv1d` takes the `USE_IM2COL_CONV1D = true` branch
(`candle-core-0.8.4/src/cuda_backend/mod.rs:1339`). It allocates, all live simultaneously:

```
col   = B · l_out · c_in · k        <-- k× the activation
res   = B · l_out · c_out           (GEMM output, [B, l_out, c_out])
res_t = B · c_out · l_out           (the strided copy back to [B, c_out, l_out])
```

For every `k7` conv in the decoder, **`col` is 7× the activation tensor**. That is the
whole story. Per-frame, in bf16:

| stage | act/frame | im2col/frame | conv peak (in+pad+col+res+res_t) |
|---|---|---|---|
| `dec.2` ResUnits, C=384, L=256·T | 0.19 MiB | 1.31 MiB | 2.06 MiB/frame |
| **`dec.3` ResUnits, C=192, L=1024·T** | **0.375 MiB** | **2.625 MiB** | **4.125 MiB/frame** |
| **`dec.4` ResUnits, C=96, L=2048·T** | **0.375 MiB** | **2.625 MiB** | **4.125 MiB/frame** |
| `dec.6` final conv, C=96 | 0.375 MiB | 2.625 MiB | 4.125 MiB/frame |

`96·2048 == 192·1024`, so the last two decoder blocks tie for the peak. Six k7 convs
(3 dilations × 2 blocks) hit it, plus the final conv.

### 1.3 The prediction vs. the observation

| T | audio | modelled generator peak | largest single alloc (`im2col`) | outcome |
|---|---|---|---|---|
| 178 | 8.27 s | **734 MiB** | 467 MiB | renders (operator measured ~**800 MiB**) |
| 256 | 11.89 s | **1056 MiB** | 672 MiB | **OOM** |

The model lands within ~9 % of the measured spike at the last working length, and the
slope is exactly linear at **4.125 MiB per frame** (measured slope implied by the
operator's 800 MiB @ 178 is 4.5 MiB/frame; the extra ~0.4 is the `Snake1d` chain, which
holds ~5 copies of the activation live, and the `cudaMallocAsync` pool's slack). Going
178 → 256 frames adds ~320 MiB to a headroom of roughly that size. That is the OOM.

### 1.4 Hypotheses ruled out

- **KV cache** — 36 layers × 8 kv heads × 128 head_dim × 2 (k,v) × 2 B = **144 KiB/token**;
  2048 tokens ≈ 0.3 GB. Also it grows monotonically and persists to the end of generation;
  the observed spike is transient and returns to baseline. Confirmed the cache is
  grow-by-`cat`, *not* preallocated at `max_seq_len` (`s2/nn.rs:244`).
- **LM head over the whole prompt** — ruled out: `SlowAr` narrows to the last position
  *before* the tied head (`s2/slow_ar.rs:209`, `:374`), so logits are `[1, 155776]`, 623 KB.
- **Prefill attention** — `[1, 32, P, P]` bf16 = 64·P² bytes. Real, but scales with the
  **prompt**, not the output; ~64 MiB at P=1000. Not the observed output-length scaling.
- **Codec bottleneck Transformer** (`post_module`, 8 layers, dim 1024) — `[1, T, 1024]`
  activations (2 KiB/frame) and a `T×T×16head` attention map (32·T² bytes → 2 MiB at
  T=256). Three orders of magnitude below the generator.

**Verdict: the codec generator is the culprit, exactly as suspected, and specifically
candle's `im2col` buffer for the k7 dilated convs of the last two decoder blocks.**

---

## 2. What changed

All in `crates/syrinx-fish/src/s2/codec.rs`.

**Split `decode` into two halves** so the chunkable part is isolated:

- `codes_to_latent(codes) -> [1, 1024, T]` — RVQ `from_codes` + the `post_module`
  Transformer. **Deliberately not chunked**: `post_module` is full-causal, position `t`
  attends over all of `0..=t`, so windowing it would change the numbers. It is also cheap
  (< 10 MiB for a minute of audio), so it always runs whole-utterance.
- `synthesize(z) -> [1, 1, L·2048]` — ConvNeXt `upsample` (×4) + the DAC generator (×512).
  Strictly causal, no right-side padding anywhere: every stride-1 causal conv has
  `extra_padding == 0`, and `causal_transpose1d` yields exactly `L · stride` after its
  `kernel − stride` right trim. This is the only length-scaled part, and the only part
  that gets chunked.

**New entry points:**

- `decode(codes)` — default path; chunks at `decode_chunk_frames_env()` frames.
- `decode_oneshot(codes)` — the one-shot **parity reference**, kept intact.
- `decode_chunked(codes, chunk_frames)` — explicit; `0` == one-shot.
- `SYRINX_FISH_CODEC_CHUNK_FRAMES` env override (`0` = one-shot), `SYRINX_FISH_*` convention.
- `DEFAULT_DECODE_CHUNK_FRAMES = 64` (≈ 3.0 s), `DECODE_LEFT_CTX_FRAMES = 16`.

**The overlap handling — not hand-waved.** Chunk `[a, b)` is synthesised from latent
frames `[a − ctx, b)` and only `[a, b)` of its output is kept. `ctx` was derived by
propagating the dependency interval backwards through the exact op list, then confirmed by
an independent forward propagation:

```
input frame -12 -> output samples [-24576, -3798)   does not reach sample 0
input frame -11 -> output samples [-22528, -1750)   does not reach sample 0
input frame -10 -> output samples [-20480,  +298)   REACHES sample 0
```

**The exact left dependency is 10 frames**, constant in chunk size, with **zero** right
dependency. `DECODE_LEFT_CTX_FRAMES = 16` is that bound plus margin (over-supplying
context is correctness-free — it only replaces zero-padding with true history).

The chunk's own tail *is* trimmed by `causal_transpose1d`, and that is correct: those raw
positions would need frames `≥ b`, which the chunk does not have — the next chunk
recomputes them with its own context.

**Equivalence claim.** The kept samples are the *same values* the one-shot path computes.
Bit-for-bit equality additionally needs the GEMM backend to accumulate identically at both
shapes; cuBLAS may pick a different tiling/split-k for a different `m`, so **on CUDA expect
agreement to rounding, not necessarily the last bit**. On CPU the reduction order is fixed
per output element, so it is bit-exact — and that is what the tests below assert.

---

## 3. Verification (CPU only — no GPU job was run)

Four unit tests were added at the bottom of `s2/codec.rs`. They build a **synthetic
small-channel codec with the real structural geometry** (`decoder_rates = [8,8,4,2]`,
`frame_hop = 2048`, the ×4 ConvNeXt upsample; only the channel widths are shrunk, and
widths do not enter the chunking arithmetic) and run the f32 CPU path.

```
cargo test -p syrinx-fish --features real s2::codec
  s2::codec::tests::oneshot_decode_is_length_exact ... ok
  s2::codec::tests::fixture_output_is_non_degenerate ... ok
  s2::codec::tests::chunked_decode_matches_oneshot ... ok
  s2::codec::tests::default_decode_matches_oneshot ... ok
test result: ok. 4 passed; 0 failed
```

`chunked_decode_matches_oneshot` asserts **max |diff| == 0.0** at T=40 for chunk sizes
2, 3, 8, 13, 16, 32, 39, 40, 64 — chunks smaller than the context, chunks that don't
divide T, and chunks at/over T.

**Negative control** (the test is sharp, not vacuous) — sweeping `DECODE_LEFT_CTX_FRAMES`:

| ctx | result |
|---|---|
| 0 | FAIL, max diff 1.458 |
| 5 | FAIL, max diff 0.101 |
| 9 | FAIL, max diff 6.1e-6 |
| **10** | **PASS** |
| 11 | PASS |

It flips to passing at exactly 10 — the value both interval propagations predicted. That
is independent confirmation of the receptive-field arithmetic against the real code, and
it also shows a too-short context fails *quietly and small* (6e-6 at ctx=9), which is
precisely the class of bug this control exists to catch.

`fixture_output_is_non_degenerate` guards against a saturated/constant fixture making the
equality assertions meaningless.

Compile gates (in a clean `git worktree` at HEAD — see §6):
- `cargo check -p syrinx-fish --features real` — clean, no warnings.
- `cargo check -p syrinx-fish --features cuda` (CUDA 12.8 / sm_120 toolchain from
  `scripts/test-all.env`) — clean, no warnings.
- `cargo clippy -p syrinx-fish --features real --tests` — the only two warnings are
  pre-existing, in `common/sampling.rs`; nothing from this change.

**Test placement deviation, stated plainly:** CLAUDE.md puts frozen Ratchet tests at the
repo-root `tests/`. These are not frozen criterion tests, and hosting them at the root
would require making `s2::codec` and `s2::nn` public — i.e. editing `s2/mod.rs`, which
another agent has in flight right now (§6). They are therefore crate unit tests inside
`s2/codec.rs`, the file this change owns. Moving them to the root later is mechanical.

---

## 4. Peak memory after the change, and what it buys

Chunked decode peak is **independent of utterance length**:

```
peak ≈ 4.5 MiB · (chunk_frames + 16)
```

| `SYRINX_FISH_CODEC_CHUNK_FRAMES` | chunk audio | decode peak | recompute overhead |
|---|---|---|---|
| 32 | 1.5 s | ~216 MiB | 1.50× |
| **64 (default)** | **3.0 s** | **~360 MiB** | **1.25×** |
| 96 | 4.5 s | ~504 MiB | 1.17× |
| 0 (one-shot) | — | 4.5 MiB · T | 1.00× |

At the default that is **less than half the current spike at the length that already
works**, and it no longer moves with T. Overhead lands on the codec stage only, which is a
small fraction of wall time next to the AR loop.

**Projected reachable utterance length.** Once decode is capped at ~360 MiB, the remaining
length-scaled terms — all live simultaneously with it, since the KV cache is fully grown
when decode runs — are:

```
KV cache            144 KiB/frame       (36 × 8 × 128 × 2 × 2 B)
output waveform      16 KiB/frame       (chunk pieces + the final cat, f32)
post_module attn     32·T bytes/frame   (quadratic: 128 MiB at T=2048)
```

Against ~1175 MiB of usable transient headroom (12227 MiB card − 10.5 GiB resident − CUDA
context):

| T | audio | projected transient |
|---|---|---|
| 512 | 24 s | ~450 MiB |
| 1024 | 48 s | ~558 MiB |
| 2048 | 95 s | ~816 MiB |
| 3072 | 143 s | ~1132 MiB |

So roughly **2–2.5 minutes per utterance**, up from 8 s — bounded now by the KV cache and
the codec bottleneck's quadratic attention, not by the generator. **This projection is
unverified** (§5); the honest claim is that the generator term is now flat, and the next
wall is somewhere else.

---

## 5. Unverified without hardware — stated plainly

1. **Every GPU number above is arithmetic, not measurement.** No GPU job was run (cuda:1
   busy, cuda:0 reserved). The allocation model is derived from candle 0.8.4's
   `cuda_backend` source, and it reproduces the operator's measured 800 MiB spike to within
   9 %, but it has not been observed.
2. **CUDA numerical equivalence of chunked vs one-shot is not verified.** Bit-exactness is
   proven on the CPU f32 path only. On CUDA, cuBLAS shape-dependent accumulation may
   introduce last-bit differences. Worth one on-box A/B:
   `SYRINX_FISH_CODEC_CHUNK_FRAMES=0` vs default on the same codes, then diff the WAVs.
3. **The projected max utterance length is a projection.** It assumes ~1175 MiB usable
   headroom, which depends on the CUDA context size and pool behaviour on this box.
4. **The measured chunking overhead is unknown.** 1.25× is a frame-count ratio, not a timing.
5. The synthetic-fixture tests validate the chunking arithmetic against the real
   *geometry*, not against the real *weights*. Nothing here changes the standing
   `// PARITY:` caveat that the EVA-GAN generator layout itself is the least-certain piece
   of the codec.

---

## 6. Blockers and notes for the operator

- **The working tree does not compile right now, and not because of this change.** Another
  agent has `crates/syrinx-fish/src/s2/load.rs`, `s2/nn.rs` and `syrinx-cli/src/main.rs` in
  flight; `load_codec` gained a 4th `CodecParts` parameter and the call site at
  `s2/mod.rs:150` was not updated:

  ```
  error[E0061]: this function takes 4 arguments but 3 arguments were supplied
     --> crates/syrinx-fish/src/s2/mod.rs:150:27
  ```

  All verification in §3 was therefore run in a detached `git worktree` at HEAD (4de558e)
  with only this change's `codec.rs` copied in, so the two edits stay independent. **Nothing
  in this repo builds until that call site is fixed.** No files outside
  `s2/codec.rs` were touched by this change.
- **Follow-up, not done (out of file scope):** `KvCache::append` (`s2/nn.rs:244`) does a
  `Tensor::cat` + `contiguous` per layer per step — 36 grow-realloc cycles every decode
  step. cudarc 0.13.9 allocates via `cuMemAllocAsync` (a stream-ordered pool), so this
  recycles better than raw `cudaMalloc` would, but a preallocated ring-buffer KV cache
  would remove both the churn and the transient 2× on the largest layer. That is the next
  length wall after this fix.
- **The encode path has the identical defect, unfixed.** `run_encoder` runs
  `encoder.block.1` at 64 ch over the *full* 44.1 kHz reference waveform with a k7 conv, so
  its `im2col` is `64 · 7 · n_samples` — roughly 395 MiB for 10 s of reference audio. It is
  chunkable by the same argument, but the encoder's mid-stack Transformer
  (`encoder.block.4.block.5`) makes it more work than the decode side. Out of scope here;
  flagged because it caps reference-audio length the same way.
