# Syrinx whole-system optimization audit — read-only pass

Date: 2026-08-29 · Scope: `crates/syrinx-fish`, `crates/syrinx-stt`, `crates/syrinx-cli`,
`crates/syrinx-serve` (glue only), root `Cargo.toml`, `scripts/`.
Method: static read of the current working tree + one CPU microbenchmark (see §A3).
**No file outside this report was modified. No GPU job was run.**

> **Concurrency note.** Two other agents were editing `s2/{codec,nn,slow_ar,fast_ar}.rs`
> and `common/dualar.rs` while this ran. Three things I had independently found landed
> mid-audit and are therefore **already fixed / in flight** — I list them in §0 so nobody
> re-does them, and everything in §1–§6 is what is *still* on the table as of this diff.

---

## 0. Already in flight (do not re-do)

| Finding | Where it landed |
|---|---|
| `KvCache::append` rebuilding via `Tensor::cat` every token → preallocated slab + `slice_set` | `s2/nn.rs` `KvCache::with_capacity` / `grow` / `append` |
| Codec decoding a whole utterance in one shot → chunked synthesis, bounded peak VRAM | `s2/codec.rs` `decode_chunked` / `DECODE_LEFT_CTX_FRAMES` / `SYRINX_FISH_CODEC_CHUNK_FRAMES` |
| Encode-only codec half (`encoder.*`, `quantizer.downsample.*`, `quantizer.pre_module.*`) resident for the whole run → **~417 MB** | `s2/load.rs` `CodecParts`, `s2/mod.rs` on-demand `encode_reference` |
| `SlowAr::embed` uploading a host frame and immediately reading it back (a CUDA stream sync **per decode step**) | `s2/slow_ar.rs` `embed_ids` |
| Codec load materialising the whole bag in f32 on device before folding (1572 MB → 939 MB transient) | `s2/load.rs` `is_weight_norm_part` |

Everything below is **additive to those**.

---

## The headline number

Fish `s2-pro`, bf16, batch 1, RTX 5070 (Blackwell, ~672 GB/s, WSL2).

**Roofline (weight traffic only, one generated frame):**

| Stage | Params | bf16 bytes read / frame |
|---|---|---|
| slow AR, 36 × (wqkv 15.73M + wo 10.49M + w1/w3/w2 24.90M×3) | 3.633 B | 7.27 GB |
| tied LM head — `embeddings.weight` `[155776, 2560]` | 0.399 B | 0.80 GB |
| fast AR, 10 steps × 4 layers × 100.9M | 4.037 B | 8.07 GB |
| | | **≈ 16.2 GB** |

At 672 GB/s that is **24 ms/frame**. Frame rate = 44100 / 2048 = **21.53 Hz**, so the
memory-bandwidth floor for this model on this card is **RTF ≈ 0.52 — about 2× faster than
realtime.**

**Measured: RTF 7.4 ⇒ 344 ms/frame ⇒ 14× off the roofline.**

That ratio is the single most important fact in this report. `s2-pro` at batch 1 is **not**
bandwidth-bound and **not** compute-bound. ~320 ms of every 344 ms frame is overhead:
per-op dispatch, redundant device copies, and host round trips. **Quantization, a bigger
card, and a faster GPU all attack the 24 ms and leave the 320 ms untouched.** Fixing the
overhead is worth up to ~14×; fixing the arithmetic is worth at most 1.07×.

---

## 1. HIGHEST IMPACT — the forward issues ~6,000 CUDA kernels per frame; ~60 % are avoidable

**Confidence: high on the count and on the fix; medium on the exact time saved (needs an
on-box `nsys`/wall-clock A/B).**

**Where:** `crates/syrinx-fish/src/s2/nn.rs` — `attention` (L379–466), `attention_batched`
(L470–557), `rms_norm_w` (L100–106), `apply_rotary_emb` (L154–…), `swiglu`; mirrored in
`crates/syrinx-fish/src/s1/nn.rs` (L276, L280, L285).

Static op count for one transformer layer-pass at `t_new = 1` (decode), following each
candle call to the kernels it actually launches:

| Sub-block | Kernels | Note |
|---|---|---|
| `rms_norm_w` × 2 (attn_norm, ffn_norm) | 16 | 8 each: `to_dtype`, `sqr`, `mean_keepdim`, `+eps`, `sqrt`, `broadcast_div`, `to_dtype`, `broadcast_mul` |
| `rms_norm_w` × 2 (q_norm, k_norm) | 16 | slow AR only (`qk_norm = true`) |
| `apply_rotary_emb` × 2 | ~18 | 2 non-contiguous `narrow→reshape` copies + 6 elementwise + `stack` each |
| `q/k/v .transpose(1,2).contiguous()` | 3 | |
| `repeat_kv` × 2 (GQA 8→32) | 2 | **each materialises a full 4× copy** — `expand` then `reshape` is never a view |
| `k_full.transpose(2,3).contiguous()` | 1 | **provably unnecessary — see below** |
| `matmul`, `* scale`, `matmul`, `ctx.transpose.contiguous` | 4 | |
| `candle_nn::ops::softmax(dim)` | 5 | generic path: `max_keepdim`, `broadcast_sub`, `exp`, `sum_keepdim`, `broadcast_div` |
| `wqkv`, `wo` linears | 2 | |
| swiglu (3 matmul + silu + mul) + 2 residual adds | 7 | |
| **per slow layer** | **≈ 76** | fast-AR layer ≈ 60 (no QK-norm) |

36 slow layers + 10 fast steps × 4 layers = 76 layer-passes ⇒ **≈ 5,300 kernels**, plus the
MCF embed (~40 ops, 11 separate `index_select` + 11 H2D uploads) and the head, plus **10
blocking device→host `to_vec1` syncs** (one per fast residual, `s2/fast_ar.rs:149`; the
11th, in `embed`, was just removed). Call it **~6,000 launches + 10 stream syncs per
frame**. 320 ms / 6,000 = **53 µs per dispatch** — high, but WSL2's paravirtualised
submission path is routinely 20–40 µs (vs 3–5 µs native), and candle adds real Rust-side
per-op cost on top. The count is the lever regardless of the exact per-launch constant.

### 1a. Three fused kernels candle 0.8.4 already ships and this code doesn't use

Verified present in the vendored `candle-nn-0.8.4`:

| Replace | With | Kernels saved | Frame total |
|---|---|---|---|
| `rms_norm_w` (8 ops) | `candle_nn::ops::rms_norm(xs, alpha, eps)` — `ops.rs:641`, one fused `CustomOp2`, and `rms_norm_slow` (`ops.rs:628`) is **line-for-line the same math** including the f32-internal reduction | 7 × 4/slow-layer, 7 × 2/fast-layer | **≈ 1,600** |
| `apply_rotary_emb` (~9 ops) | `candle_nn::rotary_emb::rope_i(x, cos, sin)` — `rotary_emb.rs:203`. I diffed `rope_i_slow` (L231–249) against `apply_rotary_emb`: **identical interleaved GPT-J pairing.** Wants `[b, n_head, t, d]`, which is the layout the very next line transposes to anyway — so transpose *first*, then rope, and the two `.contiguous()` copies inside the current rope disappear too | ~8 × 2/layer | **≈ 1,200** |
| `candle_nn::ops::softmax(&scores, D::Minus1)` (5 ops) | `candle_nn::ops::softmax_last_dim(&scores)` — `ops.rs:433`, single fused kernel; also drops 3 intermediate `[b, n_head, t, T]` allocations | 4 × 1/layer | **≈ 300** |

**≈ 3,100 of ~6,000 launches removed** by three mechanical substitutions.

*Caveat (state it in the commit):* fused kernels reduce in a different order than the
op-by-op version, so bf16 output will differ in the last ulp or two. That is well inside
the documented 1e-4/1e-5 parity band, but it **must be re-run against the parity fixtures
on-box** — it is not a free swap, it is a swap that needs its gate re-run.

### 1b. `k_full.transpose(2, 3)?.contiguous()?` is a pure waste — provable from candle's source

`s2/nn.rs:440`, `s2/nn.rs:529`, `s1/nn.rs:280`.

candle's CUDA `gemm_config` (`candle-core-0.8.4/src/cuda_backend/mod.rs:1047`) selects
`CUBLAS_OP_T` when `(rhs_m1 == k) && (rhs_m2 == 1)`. For `k_full` `[b, 32, T, 128]`
contiguous, `transpose(2,3)` gives strides `[…, …, 1, 128]` ⇒ `rhs_m1 = 128 = k`,
`rhs_m2 = 1` ⇒ **second branch, `OP_T`, no copy needed.** The batch-stride arm
(`[s1, stride] if s1 == stride * dims[1]`) also matches. And `Tensor::contiguous`
(`tensor.rs`) only copies when `!is_contiguous()` — here it always is not, so it
**always copies**.

Cost: a full `[1, 32, T, 128]` bf16 copy per layer per token. At T = 450 (≈ a 10 s
utterance with a reference): 3.69 MB × 36 layers = **133 MB/token of dead memcpy**, 36
extra allocations, 36 extra launches. Fix is deleting `.contiguous()`. If candle rejects
the layout it errors loudly — there is no silent-wrong-answer mode here.

### 1c. `repeat_kv` materialises the 4× GQA expansion every layer, every token

`s2/nn.rs:436–437` / `525–526`. `unsqueeze(2).expand(...).reshape(...)`: the expanded
tensor has a stride-0 axis, so `Tensor::reshape` takes its **copy** branch (verified in
`tensor.rs`). K and V each blow up 8 kv-heads → 32 heads.

At T = 450: 2 × 3.69 MB × 36 = **266 MB/token**, plus 72 allocations, on a card with under
1 GB of headroom.

The standard avoidance is to reshape the *query* instead: `q` `[b, 32, t, d]` →
`[b, 8, 4·t, d]`, matmul against the un-expanded `k` `[b, 8, d, T]`, then reshape the
scores back. Zero copies, `n_kv`-sized attention tensors instead of `n_head`-sized. This
is a bigger change than 1a/1b and needs a parity re-run, but it removes ~266 MB/token of
traffic *and* shrinks live attention memory 4× — which bears directly on the OOM at 8–10 s.

**Combined estimate for §1 (speculative until measured on-box): 344 ms → ~180–230 ms per
frame, i.e. RTF 7.4 → ~4.0–5.0.** Stacked on the KvCache and codec-chunk fixes already in
flight.

---

## 2. HIGH — the tied LM head reads 797 MB per token to make a 4,097-way decision

**Confidence: high (arithmetic is exact).** `s2/slow_ar.rs:230–242` (`head`) and
`:392–400` (`head_batch`); consumed at `common/dualar.rs:189` / `:288`.

`linear_w(&normed, &emb)` multiplies `[1,1,2560]` by the whole `embeddings.weight`
`[155776, 2560]` — **797.6 MB of bf16 read, per token, forever** — and then the driver
constrains the result to `[semantic_begin, semantic_end] ∪ {stop}` and throws the other
151,679 logits away.

From the resolved s2 config: `stop = 151645`, `semantic_begin = 151678`,
`semantic_end = 155773`. **`stop` sits below `begin`, so a single contiguous
`narrow(0, 151645, 4129)` covers the stop id and the entire semantic range.** That is a
free view (rows are contiguous) over a `[4129, 2560]` = **21.1 MB** slice.

- Weight read for the head: **797.6 MB → 21.1 MB (37.8×)**, ≈ **1.15 ms/frame** of pure
  bandwidth back at the roofline, and a bigger fraction once §1 lands.
- Device→host transfer per token: 623 KB → 16.5 KB.
- Requires threading a `logits_offset` through `SlowStep`/`SlowStepBatch` (the driver's
  `SemanticConstraint`, `first_code`, and RAS window all work in absolute ids today), so
  it is a contained but real contract change.

**Do this in two steps.** Step one costs nothing and needs no contract change: have the
*driver* narrow before the read-back —
`step.semantic_logits.narrow(0, stop_id, 4129)?.to_vec1()?` at `dualar.rs:189` (and the
`to_vec2` at `:288`). That alone kills the 623 KB/token D2H and shrinks the sampler's work
38× (§A3). Step two — narrowing the matmul itself — is where the 776 MB/token lives.

---

## 3. HIGH (and the easiest thing in this report) — `--cuda` is inert in both Fish entry-point scripts

**Confidence: certain.** Verified: `cuda` is not a default feature of any crate
(`crates/*/Cargo.toml`), and there is **no `.cargo/config.toml`** in the repo.

- `scripts/synth-samples.sh:143` — `cargo run -q -p syrinx-cli --features real -- …`, then
  `:152` appends `--cuda` **to the binary** if the user passed `--cuda`. The *cargo*
  feature `cuda` is never passed. So `pick_device` (`syrinx-serve/src/synth/mod.rs:74`)
  compiles to its `#[cfg(not(feature = "cuda"))]` arm and returns `Device::Cpu`
  unconditionally. `S2Pro::load` then picks **f32** (`s2/mod.rs`), i.e. ~18 GB for the LM.
  The script's own `--help` at `:39` advertises "`--cuda` run on GPU (requires a
  `--features cuda` build)" — and then does not supply that build.
- `scripts/synth-samples.sh:143` also omits **`--release`**, so the 610-sample corpus
  renderer — the entry point CLAUDE.md documents — runs a **debug** build.
- `scripts/run-fish.sh:100` uses `--release` but likewise only `--features real`, and never
  offers `--cuda` at all.

Fix: `--features cuda` (which implies `real`) plus `--release` in both, gated on the
script's `--cuda`. One line each.

*(The 10.5 GB / bf16 / RTF-7.4 numbers in the brief imply the measurements were taken by
invoking the CLI by hand with `--features cuda`, per the README. The scripts are still
broken, and this is exactly the "CLI flag that silently does nothing" class.)*

---

## 4. MEDIUM-HIGH — no `[profile.release]` anywhere in the workspace

**Confidence: high that it's missing; medium on the size of the win.**

`grep -n profile Cargo.toml crates/*/Cargo.toml` → nothing. So release builds get
`lto = false`, `codegen-units = 16`, `panic = "unwind"`. Everything CPU-side in the hot
path — the sampler's sort/softmax (`common/sampling.rs`), the Lanczos resampler
(`common/audio.rs:117`), the STT decode loop, `wer` — is compiled without cross-crate
inlining.

```toml
[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"     # optional; drops unwind tables
```

Typical 5–15 % on the CPU-bound stages for a longer link. Cheap, low risk, and it applies
to every crate at once.

---

## 5. MEDIUM — dead knobs and one lurking dead-weight risk

### 5a. Documented-but-unread env vars

`grep` of every `SYRINX_*` in `scripts/` + `*.md` + `docs/` against every one in
`crates/**/*.rs`:

| Knob | Documented at | Read in code? |
|---|---|---|
| `SYRINX_FISH_S2_INT4` | `scripts/test-all.env.example:56` (`# functional int4 run on a 12 GB card`) | **No — nowhere.** (already known) |
| `SYRINX_MOS_MODEL` / `SYRINX_MOS_REPO` | `scripts/eval_mos.py:17–18` | Python-side only — fine, but they read as Rust knobs in the env file |
| `SYRINX_WER_MODEL` | `scripts/eval_wer.py:17` | same |
| `SYRINX_FISH_CKPT`, `SYRINX_FISH_REF_TEXT`, `SYRINX_REF` | `run-fish.sh`, `synth-samples.sh` | shell-only — correct, but undocumented as such |

Only `SYRINX_FISH_S2_INT4` is a genuine lie: the env file advertises an int4 path for the
12 GB card that does not exist in `syrinx-fish` at all (the int4 work is CosyVoice-side,
`SYRINX_QUANT` / `SYRINX_CV3_QUANT`). Either delete the line or make it a `todo!`-free
explicit "not implemented for Fish" note.

Conversely, 22 knobs are read by code and documented nowhere (`SYRINX_FISH_MAXFRAMES`,
`SYRINX_FISH_CODEC_ROUNDTRIP`, `SYRINX_CUDA_ORDINAL`, `SYRINX_LM_HAMMER`, …). Not a
performance issue, but the env file is not a trustworthy index of the knobs.

### 5b. `load_lm` silently retains every unrecognised checkpoint tensor

`s2/load.rs`, `remap_qwen3_key` — the final arm is *"Unknown key: keep it under its
original name … A genuinely unused tensor is harmless in the bag."* It is not harmless on a
card with under 1 GB of headroom: this is exactly the shape of the CosyVoice
519 MB-dead-`lm_head` bug the README documents. Only `audio_tower` / `visual` are dropped.

**This is measurable in five minutes on-box** and I could not do it off-box. Add a
`RefCell<HashSet<String>>` of fetched names to `Weights`, and after one `synthesize` print
every key in `map` that was never fetched, with `numel() * dtype().size_in_bytes()`. If it
prints nothing, close the finding; if it prints hundreds of MB, that is headroom for free.
Do the same for the codec bag.

*(One thing that is already accounted for: candle's pickle reader has no `Bool` dtype and
skips the codec's three `causal_mask` buffers — 302 MB, including a `[16384, 16384]`
triangle — so those never reach the device. Noted in the concurrent `s2/load.rs` work.)*

---

## 6. MEDIUM — `crates/syrinx-stt` (the WER oracle)

Three real issues, in order.

### 6a. The Whisper **encoder** is re-run for every temperature fallback

`stt.rs:315` — `model.encoder.forward(mel, true)` is the first statement of `decode`, and
`decode_with_fallback` (`:287`) calls `decode` once per entry in `m::TEMPERATURES` (6).
The encoder output depends on neither `t` nor the token history.

For `whisper-large-v3` the encoder is ~635M params — the expensive half of the model on a
30 s window. On clean TTS renders `needs_fallback` is usually false so only one pass runs,
but on any noisy or low-logprob window this is a **straight 6× multiplier on the encoder**.
`detect_language` (`:420`) runs it a further time on the same mel prefix.

Fix: hoist the encoder forward into `decode_with_fallback` and pass `audio_features` into
`decode`. Cache `detect_language`'s features for the first window. Mechanical.

### 6b. `suppress_tensor()` is rebuilt on every `decode` call, in `O(vocab × |suppress|)`

`stt.rs:328` calls `:449`, which maps over the full vocab (51,866 for large-v3) doing
`self.suppress_tokens.contains(&i)` — a **linear scan of a `Vec`** inside the map. ~90
suppress tokens ⇒ ~4.7M comparisons, a 207 KB allocation, and a host→device upload, **per
`decode` call** (so up to 6× per window). The tensor is a compile-time constant of the
model. Hoist it to an `Stt` field at load; the whole cost becomes zero.

### 6c. A redundant softmax + `to_scalar` device sync per decoded token

`stt.rs:378–383`: after choosing `next_token`, the loop runs a **second** full softmax on
the GPU and pulls one element back with `.to_scalar::<f32>()` — a blocking sync — purely to
accumulate `sum_logprob`. But on the `t == 0` branch the raw `logits` were *already*
fetched to the host at `:373` (`logits.to_vec1()`). Compute the log-prob from that host
vector. On the `t > 0` branch, fetch `logits` once and derive both `argmax` and the
unscaled prob host-side. Net per token: **1 D2H instead of 2 + ~10 GPU kernels**.

### 6d. Not a syrinx bug, but worth knowing

`candle-transformers-0.8.4`'s `whisper::model::MultiHeadAttention::forward` caches K/V
**only for cross-attention** (`xa: Some(_)`); self-attention recomputes over the full token
prefix every step. So the decoder is O(n²) in tokens. I initially flagged this as large,
then worked it through: for large-v3's ~907M-param decoder the per-step forward stays
weight-read-bound until n ≈ 250, so at typical n ≤ 100 the real penalty is ~1.3×, not 50×.
**Not worth forking the model for.** Recorded so nobody else re-derives it.

---

## 7. LOW — real but small

- **`build_prompt_with_reference` is O(t_ref² · n_cb)** — `s2/mod.rs`, the
  `for (c, s0) … for r in 0..n_cb { … ref_row(r)[c] }` nest calls `ref_row(r)`, which
  allocates a fresh `Vec<u32>` of length `t_ref`, **inside the inner loop**. At `t_ref`=215
  (10 s ref) that's ~2,150 allocations and ~460 K element copies per prompt build. ~1 ms —
  invisible next to a 5 s render, but it is genuinely quadratic and one hoist fixes it.
- **The MCF embed issues 21 kernels where 2 would do** — `s2/slow_ar.rs` `embed_ids`, the
  `for i in 0..n_cb` loop does 10 separate `index_select` + 10 `+` on `codebook_embeddings`.
  Build one `[n_cb * t]` id vector, one `index_select`, `reshape((n_cb, t, dim))`, `sum(0)`.
  ~30 launches/frame.
- **~900 `format!` + `HashMap<String, _>` SipHash lookups per frame** — every `attention`
  call builds `wqkv.bias` / `wo.bias` name strings unconditionally, even when the bias
  flags are false. Precomputing a per-layer struct of `Tensor` handles at load would remove
  all of it. ≈ 0.1 ms/frame — a tidiness item, not a perf item, and I say so because the
  temptation to "optimize" it is real and it isn't worth the churn.
- **The Lanczos resampler is 33 taps × 2 `sin()` per output sample** — `common/audio.rs:117`.
  ~30M transcendental calls for a 10 s reference at 16 k → 44.1 k. ~0.3–1 s in release, once
  per run (the batch path already encodes the reference once, `syrinx-cli/src/main.rs:1012+`).
  A precomputed polyphase table would make it ~free. Only worth doing if §4's release fix
  isn't enough — and it will be much worse than 1 s under the debug build of §3.
- **Batched generation pulls `[N, 155776]` logits to the host per frame** —
  `dualar.rs:288` `to_vec2()`. N × 623 KB per token and N × the sampler cost. The §2 narrow
  fixes this too, and it matters more here because it scales with N.

---

## Recommended order (impact × confidence × risk)

1. **§3** — one line in each of two scripts. Zero risk, and it is currently the difference
   between "runs on GPU" and "doesn't" for the documented render path.
2. **§1b** — delete two `.contiguous()` calls. Provable from candle's source; fails loudly
   if wrong. ~133 MB/token.
3. **§2 step one** — narrow the logits in the driver before `to_vec1`/`to_vec2`. No contract
   change, no numerics change.
4. **§4** — add `[profile.release]`.
5. **§6a–6c** — the STT fixes; all mechanical, all in one file.
6. **§1a** — the three fused-kernel substitutions. Biggest single win (~3,100 launches/frame)
   but **requires the on-box parity gate to be re-run**, since fused reductions differ in
   the last ulp.
7. **§5b** — instrument the weight bag for never-fetched tensors and *measure*. Cheap, and
   it either closes a risk or hands back hundreds of MB.
8. **§2 step two** and **§1c** — the two structural changes (head offset plumbing; drop
   `repeat_kv` via the reshape-q trick). Best value per byte on the OOM, most invasive.

---

## Appendix A — how the numbers were obtained

**A1. Roofline.** Param counts derived from the resolved `FishConfig::s2_pro` + the
`reconcile_fast_cfg` overrides (`s2/mod.rs`): dim 2560, 36 layers, GQA 32:8, head_dim 128,
intermediate 9728, vocab 155776, fast = 4 layers of the same block. Per slow layer:
wqkv `[6144,2560]` 15.73M, wo `[2560,4096]` 10.49M, w1/w3/w2 `[9728,2560]`×2 + `[2560,9728]`
24.90M each ⇒ 100.92M. ×36 = 3.633B; fast ×4 ×10 steps = 4.037B; embeddings 0.399B.
16.2 GB bf16 / 672 GB/s = 24 ms. Frame rate from `CodecConfig::frame_rate()` = 44100/2048.

**A2. Kernel counts** are a static walk of each candle call in `s2/nn.rs`, expanding
composite ops to the kernels they launch (e.g. `ops::softmax` → 5, verified against
`candle-nn-0.8.4/src/ops.rs:22`) and checking `is_contiguous` at each `reshape`/`narrow` to
decide view-vs-copy (`candle-core-0.8.4/src/tensor.rs`). **Not measured** — an on-box
`nsys profile` on ~20 frames would confirm or refute it in minutes and is the single most
valuable next measurement.

**A3. The one thing I measured, and it disproved my own hypothesis.** I expected the
sampler's full-vocab sort (`common/sampling.rs::logits_to_probs`, called twice per frame
over all 155,776 logits with an indirect comparator) to be a major cost. I replicated it
exactly and timed it (`rustc -O`, this box's CPU):

```
per-frame semantic sampling (2 draws):
  full vocab      2.236 ms
  sliced 4129     0.278 ms   (8.1x faster)
```

**2.2 ms of a 344 ms frame — 0.65 %.** So the sort is *not* a real win at batch 1; it is a
nitpick that rides along free with §2, and it only becomes interesting on the batch path
where it scales with N. I had estimated 40–80 ms from first principles and was wrong by
20×. Reported that way deliberately: the measurement is the verdict.

**A4. Not measured / explicitly speculative:** everything in §1 (needs `nsys` + a wall-clock
A/B on the box), the §5b dead-weight hypothesis (needs the fetched-key instrumentation),
and the §6a fallback frequency (needs a run over real audio to see how often
`needs_fallback` fires).
