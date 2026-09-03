# Fish s2-pro resident footprint — byte-level accounting and what was done

Scope: `crates/syrinx-fish/src/s2/load.rs`, `crates/syrinx-fish/src/s2/mod.rs`
(load/dtype plumbing), `scripts/test-all.env.example`.
Checkpoint: `/home/floofy/models/s2-pro/`. Target box: RTX 5070, 12227 MiB.

Everything below was derived by reading the safetensors headers and the `codec.pth`
pickle index directly (no tensor data loaded, no GPU touched) plus reading candle
0.8.4's source. **No GPU job was run**, so every number is a static accounting, not a
measured `nvidia-smi` figure. What is and is not verified is stated at the bottom.

---

## 1. Where the ~10.5 GB is

### 1a. The LM — `model-0000{1,2}-of-00002.safetensors`

358 tensors, **all BF16 on disk**, 9,123,704,832 B on disk. `load_lm` casts each
shard tensor to the compute dtype as it reads; on CUDA that dtype is BF16, so **the
resident LM is byte-for-byte the on-disk size**. There is no dtype widening.

| group | n | shape | disk = resident (MB) |
|---|---:|---|---:|
| `layers.N.feed_forward.w1` (slow) | 36 | `[9728, 2560]` | 1793.06 |
| `layers.N.feed_forward.w2` (slow) | 36 | `[2560, 9728]` | 1793.06 |
| `layers.N.feed_forward.w3` (slow) | 36 | `[9728, 2560]` | 1793.06 |
| `layers.N.attention.wqkv` (slow, pre-fused) | 36 | `[6144, 2560]` | 1132.46 |
| `embeddings.weight` (input **and** tied head) | 1 | `[155776, 2560]` | 797.57 |
| `layers.N.attention.wo` (slow) | 36 | `[2560, 4096]` | 754.97 |
| `codebook_embeddings.weight` (MCF, 10×4096) | 1 | `[40960, 2560]` | 209.72 |
| `fast_layers.N.feed_forward.w{1,2,3}` | 12 | `[9728,2560]`/`[2560,9728]` | 597.68 |
| `fast_layers.N.attention.wqkv` | 4 | `[6144, 2560]` | 125.83 |
| `fast_layers.N.attention.wo` | 4 | `[2560, 4096]` | 83.89 |
| `fast_embeddings.weight` (shared value table) | 1 | `[4096, 2560]` | 20.97 |
| `fast_output.weight` (fast head) | 1 | `[4096, 2560]` | 20.97 |
| all RMSNorm + QK-norm vectors | 154 | `[2560]` / `[128]` | 0.44 |
| **LM total** | **358** | | **9123.70** |

### 1b. The codec — `codec.pth` (1,871,099,728 B on disk, 541 tensors)

Stored **f32** except three bf16 `freqs_cis` buffers and three **bool** `causal_mask`
buffers. Loaded via `candle_core::pickle::read_all`, folded, cast to the compute dtype.

| group | n | f32 bytes (MB) | resident bf16 (MB) | read by |
|---|---:|---:|---:|---|
| `encoder.*` | 157 | 311.60 | 155.80 | `encode` only |
| `quantizer.downsample.*` | 11+11 | 84.03 | 42.02 | `encode` only |
| `quantizer.pre_module.*` (8-layer causal TF) | 74 | 437.39 | 218.69 | `encode` only |
| `quantizer.post_module.*` (8-layer causal TF) | 74 | 437.39 | 218.69 | `decode` only |
| `decoder.*` (EVA-GAN generator) | 119 | 216.42 | 108.21 | `decode` only |
| `quantizer.upsample.*` | 11+11 | 84.03 | 42.02 | `decode` only |
| `quantizer.{,semantic_}quantizer.*` (RVQ) | 70 | 1.16 | 0.58 | both |
| **codec total** | **538** | **1572.01** | **786.01** | |

Three tensors — 301.99 MB on disk — are **never loaded at all**:

| skipped tensor | shape | dtype | disk MB |
|---|---|---|---:|
| `encoder.block.4.block.5.causal_mask` | `[16384, 16384]` | bool | 268.44 |
| `quantizer.pre_module.causal_mask` | `[4096, 4096]` | bool | 16.78 |
| `quantizer.post_module.causal_mask` | `[4096, 4096]` | bool | 16.78 |

candle 0.8.4's `DType` has no `Bool`, and `pickle::rebuild_args` bails on
`BoolStorage`; `read_pth_tensor_info` catches that per tensor and does
`eprintln!("skipping: {err:?}")`. So they are dropped at read time and never reach the
device. `codec::transformer` recomputes the triangle with `causal_mask_at`. This is
accidental but correct — it is now documented in `load_codec`, because it is the kind
of thing that silently breaks if candle ever grows a `Bool` dtype (it would then load
1.21 GB of f32 mask).

### 1c. Precomputed tables

| table | shape | dtype | MB |
|---|---|---|---:|
| slow RoPE cos+sin (`rope_cap = min(32768, 8192)`) | 2 × `[8192, 64]` | bf16 | 2.10 |
| fast RoPE cos+sin (codebook axis) | 2 × `[10, 64]` | bf16 | 0.003 |

The 8192 cap costs 2.10 MB, not GB. It is **not** worth touching.

### 1d. Reconciliation

| | MB | GiB |
|---|---:|---:|
| LM weights | 9123.70 | 8.497 |
| codec weights (bf16) | 786.01 | 0.732 |
| RoPE tables | 2.10 | 0.002 |
| **total tensors** | **9911.81** | **9.231** |
| observed resident (reported) | ~10500 | ~9.78 |
| **unaccounted** | **~590** | **~0.55** |

The ~590 MB gap is *not* tensors. It is the CUDA primary context + cuBLAS handle and
workspace + the cudarc PTX modules candle JIT-loads + the KV slabs + whatever transient
was live when the figure was taken. The slow KV cache costs
`36 layers × 2 × 8 kv-heads × 128 head_dim × 2 B` = **144 KiB per position** (288 MiB at
2048 positions); it is preallocated in slabs by `KvCache`, not grown by `cat`.

**Conclusion: the on-disk checkpoint and the resident set agree to within the CUDA
runtime's own overhead. Nothing is materialised in a wider dtype than stored, and
nothing is duplicated.**

---

## 2. Dead / reducible weight

### 2a. `lm_head` — checked, and there is none. (0 GB)

`text_config.tie_word_embeddings` is `true`, and the checkpoint genuinely ships **no**
`lm_head`/`output.weight` tensor (verified: 358 keys, none matching). `SlowAr::head`
takes the tied branch and does `linear_w(normed, embeddings.weight)`.

I also checked the failure mode that would have made the tie expensive: `linear_w` does
`x.broadcast_matmul(&w.t()?)`, and if candle forced a contiguous rhs that would be a
797 MB copy **per decode step**. It does not. `cuda_backend::gemm_config` inspects the
rhs strides and, for the broadcast-transposed `[1, 2560, 155776]` view (strides
`[0, 1, 2560]`), takes the `rhs_m1 == k && rhs_m2 == 1` branch → `CUBLAS_OP_T`. No copy.
The CosyVoice mistake is **not** repeated here.

Every one of the 358 LM tensors is fetched on the inference path. There is no dead
weight in the LM.

### 2b. The codec encoder — real, and now fixed. (**0.42 GB**)

`EvaGanDac::encode` is the only reader of `encoder.*`, `quantizer.downsample.*` and
`quantizer.pre_module.*` — 833.02 MB f32 / **416.51 MB bf16**. `EvaGanDac::decode`
touches none of them (verified by reading `decode` → `decode_codebook` → `transformer`
→ `upsample` → `run_generator`).

Both CLI entry points call `encode_reference` **exactly once**, immediately after load,
and then only ever decode:
- `crates/syrinx-cli/src/main.rs:767` (single synth)
- `crates/syrinx-cli/src/main.rs:1024` (`--batch`, explicitly "encoded ONCE")

So the encode stack sat resident for the entire generation while being read for a
fraction of a second at the start.

### 2c. The load-time f32 transient. (**~0.63 GB of peak**)

`load_codec` used to cast *every* tensor to f32 (needed only for the `‖v‖` weight-norm
fold) and cast the whole bag to bf16 afterwards. Only 306.06 MB of the codec is
weight-norm `g`/`v`; the other 1265.83 MB was being materialised f32-then-bf16 for no
reason. Peak GPU during codec load was `9123.70 (LM) + 1572.01 = 10.70 GB` — about
9.96 GiB before the CUDA context, which is uncomfortably close to 11.94 GiB.

---

## 3. What was implemented

### `s2/load.rs`

1. **`CodecParts { Decode, Encode }`** and `load_codec(path, dev, dt, parts)`. Keys are
   filtered by top-level prefix **before** `to_device`, so a skipped tensor never
   touches VRAM. `Decode` drops the three encode-only prefixes, `Encode` drops the three
   decode-only ones, and everything else (the 0.58 MB RVQ, plus any unrecognised key)
   lands in **both** bags — a key can never silently go missing. The 8 prefixes cover
   all 538 loadable tensors exactly (253 + 215 + 70 = 538).

2. **Only weight-norm components are materialised in f32.** `is_weight_norm_part`
   selects `.weight_g` / `.weight_v` / `.parametrizations.weight.original{0,1}`;
   everything else is cast straight to `dt` at read. After the fold, the `!= dt` guard
   casts just the folded outputs.

   **This is byte-identical, not approximately identical.** `f32 → f32 → bf16` and
   `f32 → bf16` are the same single rounding; the three bf16 `freqs_cis` buffers
   round-trip through f32 exactly (bf16 ⊂ f32). For `dt == F32` (the CPU parity path)
   every cast is an identity, exactly as before.

### `s2/mod.rs`

3. `S2Pro` now keeps `codec_path` and `dt`, loads the codec `CodecParts::Decode` at
   startup, and `encode_reference` materialises a temporary `CodecParts::Encode` bag,
   encodes, and drops it on return. `RvqCodec::encode` routes through it.

### `scripts/test-all.env.example`

4. `SYRINX_FISH_S2_INT4=1` **deleted**. Confirmed by `grep -rn SYRINX_FISH_S2_INT4 .`:
   before the change it appeared in exactly one place in the whole repo — line 56 of
   `test-all.env.example`, where it claimed to be a "functional int4 run on a 12 GB
   card". No source file, script, or test read it. It was a **falsely advertised knob**:
   setting it did nothing at all, silently. The block now states the real bf16 numbers.
   (The on-box `scripts/test-all.env` never had the line.)

### Net effect

| | before (MB) | after (MB) | Δ |
|---|---:|---:|---:|
| resident weights during generation | 9911.81 | **9495.30** | **−416.51** |
| peak GPU during load | ~10695.7 | ~9601.7 | −1094.0 |
| peak during `encode_reference` | ~10695.7 | ~9958.4 | −737.3 |
| working headroom on 12227 MiB (after ~590 MB runtime) | ~1.0 GB | **~1.4 GB** | **+40%** |

Cost: one extra read of `codec.pth` when cloning (a few seconds, once per run). Nothing
else changes; no weight value, no op, no order of operations.

`cargo check --workspace --features real` is clean.

---

## 4. Honest verdict on int4 for this model: **do not do it**

The claim in the env example was false, and I recommend it stay deleted rather than be
made true. The reasoning, specific to this architecture and this codebase:

**a) The dequant-on-fetch design this repo already has would make it slower and would
not even reliably shrink the working set.** The CV2/CV3 int4 paths
(`syrinx-lm/src/cv2/quant.rs`, `syrinx-vocoder/src/cv{2,3}/quant.rs`) store a `QTensor`
and reconstruct on fetch. candle 0.8.4's `QMatMul::forward` on CUDA goes through
`QCudaStorage::fwd` → `dequantize_matmul{,_vec}`, which materialises the **f32** weight
before the gemm. An f32 dequant of a `[9728, 2560]` FFN weight is 99.6 MB — *twice* the
bf16 weight it replaced. Per token you would pay 5 such dequants × 36 slow layers.
The README already records the CosyVoice outcome in plain language: *"the int4
dequant-on-fetch is slow to load/infer — it's an opt-in path, not the default"*. Doing
it again here, on a model 10× larger, would be repeating a documented mistake.

**b) The dtype does not line up.** The s2 CUDA compute dtype is bf16 throughout.
candle's quantized path is f32-only (`"only f32 can be quantized"`,
`quantized/cuda.rs:454`; the matmul dtype match at 211/266/339 accepts f32). Every
quantized linear would need `bf16 → f32 → qmatmul → f32 → bf16`, doubling activation
size and adding four casts per linear. `forward_via_f16` exists but dequantizes the
whole weight first, which is the same problem.

**c) The embeddings — the part with the best size/quality ratio — are the part that
does not fit.** `embeddings.weight` (797.57 MB) is used *both* as an `index_select`
table and as the tied output head. `QMatMul` cannot serve an `index_select`, so you
would need two representations (a `QTensor` for the head and a row-quantized store for
the gather), i.e. you'd quantize 798 MB into ~450 MB of two different encodings. Not
worth it.

**d) The quality cost is unmeasurable from here — and that is disqualifying.** Q4_0 is
per-block-32 symmetric 4-bit. In a dual-AR TTS model the slow AR samples a token per
frame and the error compounds autoregressively across hundreds of frames, then the RVQ
residuals compound again in the fast head. The only CV data point is a 0.5B model
(SIM-o 0.76 → 0.72). Extrapolating that to a 5B dual-AR is a guess. Confirming it needs
the GPU box and an ear — precisely what CLAUDE.md marks blocked. Shipping "functional
int4" without that measurement would be exactly the green that isn't real.

**e) It is not needed.** The stated problem is that long utterances OOM. After this
change the weights are 9.50 GB with ~1.4 GB of headroom, and the binding constraint is
the codec's per-conv activation, not the weights (§5). A 5.2 GB theoretical Q4_0 saving
on the LM linears buys nothing if a single decoder conv still wants 0.6 GB per 10 s.

**If int4 is ever wanted, the honest prerequisite list is:** native Q4_0 × bf16 CUDA
kernels (not dequant-on-fetch), a separate row-quantized embedding store for the tied
table, and a SIM-o/MOS A/B on the box against the bf16 baseline. That is a real project,
and it is blocked-on-GPU. It is not an env var.

---

## 5. The actual long-utterance limit is activations, not weights

Worth recording because it is what "8–10 s OOMs" really is, and it is **not in my files**.

candle 0.8.4's CUDA `conv1d` sets `USE_IM2COL_CONV1D = true` (`cuda_backend/mod.rs:1341`)
— it builds an `[b, l_out, k_size × c_in]` column buffer and gemms it. For the EVA-GAN
decoder's shallowest stages that buffer is **duration-linear and large**:

| decoder stage | channels | resolution | im2col elems / output sample | bf16 bytes at 10 s (441 000 samples) |
|---|---:|---|---:|---:|
| `decoder.model.3` residual units (k=7) | 192 | L/2 | 672 | 592 MB |
| `decoder.model.4` residual units (k=7) | 96 | L | 672 | 592 MB |
| `decoder.model.6.conv` (k=7, 96→1) | 96 | L | 672 | 592 MB |

~0.6 GB per conv call at 10 s, ~1.2 GB at 20 s, on top of the padded input and the
output. That is the OOM, and it is why the wall sits right around 8–10 s.

The owner of `s2/codec.rs` has since landed `decode_chunked` /
`SYRINX_FISH_CODEC_CHUNK_FRAMES` (default ≈3.0 s per chunk with a left-context margin),
which bounds exactly this spike. My change is complementary: theirs caps the
duration-linear activation, mine removes 0.42 GB of duration-independent weight so the
chunk has more room. Both are needed; neither substitutes for the other.

---

## 6. Deliberately not done

- **Quantizing anything.** See §4.
- **Shrinking the RoPE cap below 8192.** It is 2.10 MB. Touching it risks a wrong
  position table for a 0.02% saving.
- **Dropping `codebook_embeddings` rows.** All 10 codebooks × 4096 are read by the MCF
  sum in `SlowAr::embed`. Not dead.
- **Freeing the encoder in place after use.** `EvaGanDac`'s `w` field is private to
  `s2/codec.rs`, which another agent owns. Loading the encode bag on demand achieves the
  same resident footprint without editing their file.
- **An env knob to restore the old all-in-one codec load.** It would only ever be set to
  get back to a strictly worse state; the CPU parity path is already unchanged.
- **Touching `s2/nn.rs`, `s2/slow_ar.rs`, `s2/fast_ar.rs`, `s2/codec.rs`,
  `common/dualar.rs`.** Read only, per scope.

---

## 7. Unverified without hardware — stated plainly

Everything below is reasoning over file headers and candle's source. **None of it has
been run on a GPU** (cuda:1 rendering, cuda:0 reserved). What still needs the box:

1. **The ~590 MB runtime overhead is inferred, not measured.** It is what is left after
   subtracting a tensor-exact 9911.81 MB from a reported "~10.5 GB". Confirm with
   `nvidia-smi` before and after `S2Pro::load`.
2. **The −416.51 MB is arithmetic on the pickle index**, not an observed drop. Confirm
   with `nvidia-smi` after load, before and after this change.
3. **The byte-identity of the dtype-order change is a proof about IEEE rounding, not a
   diff of two runs.** It should be checked by re-running the s2 CPU parity test
   (`dt == F32`, where every cast is an identity, so the bar is exact equality) and the
   s2 e2e smoke test.
4. **`encode_reference` correctness after the split** — that the encode bag really
   contains every key `EvaGanDac::encode` fetches. I traced `encode` →
   `run_encoder` / `downsample` / `transformer("quantizer.pre_module")` /
   `quantize_one` (which calls `decode_codebook`, i.e. `codebook.weight` + `out_proj`,
   both in the shared RVQ group), and the prefix filter covers all of them. A missing
   key surfaces immediately and loudly as `missing weight: <name>` from `Weights::g` —
   it cannot degrade silently — but it has not been executed.
5. **The im2col figures in §5** are computed from candle's kernel shape, not profiled.
6. **The `causal_mask` skip** is read from candle's error path; the on-box run should
   show three `skipping: Msg("unsupported storage type BoolStorage")` lines on stderr
   during codec load. If those lines are *absent*, re-check §1b.

## 8. Blockers

**No blocker for the work in scope** — it is compile-verified and byte-preserving by
construction.

**One standing blocker**, unchanged: the s2-pro numeric parity and any perceptual
judgement (int4 quality included) require the GPU box and the Fish reference fixtures,
per CLAUDE.md's blocked-on-human list. Nothing in this report claims a measured result.
