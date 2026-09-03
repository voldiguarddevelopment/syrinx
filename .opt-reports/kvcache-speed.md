# s2-pro autoregressive generation: KV-cache slab rewrite + per-step waste audit

**Date:** 2026-08-29 · **Scope:** `crates/syrinx-fish/src/s2/{nn,slow_ar,fast_ar}.rs`,
`crates/syrinx-fish/src/common/dualar.rs` · **Hardware used:** none. Everything below is
either exact arithmetic, a CPU measurement, or an explicitly-labelled estimate.

> **Headline, stated up front because it contradicts the brief's premise:** the
> `Tensor::cat`-per-step KV cache *is* a real O(T²) defect and it is now fixed, but the
> arithmetic says it was worth **~0.04 % of wall time** at the measured operating point
> (T ≈ 178, RTF 7.4). It is not what makes s2-pro 7.4× slower than realtime. The audit in
> §4 found something that plausibly is, and it is one line of `nn.rs` — see §4.1.

---

## 0. Geometry used throughout

From `FishConfig::s2_pro` plus the checkpoint reconciliation in `s2/mod.rs`
(`reconcile_fast_cfg`, which makes the fast AR structurally identical to a slow layer):

| | slow AR | fast AR |
|---|---|---|
| layers | 36 | 4 |
| dim / intermediate | 2560 / 9728 | 2560 / 9728 |
| heads (q : kv) | 32 : 8 | 32 : 8 |
| head_dim | 128 | 128 |
| cache depth | prompt + T frames | 10 (one per codebook), rebuilt per frame |
| dtype on GPU | bf16 (2 B) | bf16 |

Cached bytes per position, K **and** V, all layers, batch 1:

* slow: `36 × 2 × 8 × 128 × 2 B` = **147 456 B = 144 KiB / position**
* fast: `4 × 2 × 8 × 128 × 2 B` = **16 384 B = 16 KiB / position**

Frame rate `44 100 / 2048` = 21.53 Hz → one realtime frame = **46.44 ms**.
Measured (`renders/FINDINGS.md`): 69.2 s of audio in ~511 s ⇒ **343 ms/frame, RTF 7.4**.

---

## 1. Bytes copied — before vs after

### 1.1 Slow AR

`KvCache::append` previously did `Tensor::cat(&[&ck, k], 2)?.contiguous()?`. `cat` already
returns a contiguous buffer, so `.contiguous()` was a no-op clone and `cat` is the copy: it
**rewrites every cached position, on every layer, on every step**. The very first append per
layer (the prefill) takes the `None` branch and copies nothing.

    bytes_written(cat)  = 147 456 × Σ_{j=1..T} (P + j)  =  147 456 × [ T·P + T(T+1)/2 ]
    bytes_written(slab) = 147 456 × [ (P + T) + Σ_growths cap_before_growth ]

`P` = prompt prefill length (unknown for the measured corpus, so both `P = 0` and a
representative `P = 200` are given). The slab's growth term comes from the `KV_CHUNK = 256`
reallocation schedule described in §2.

**P = 0**

| T | cat writes | slab writes | ratio | slab depth | slab VRAM |
|---:|---:|---:|---:|---:|---:|
| 178 | 2 240.3 MB | 25.03 MB | **89.5×** | 256 | 36.0 MB |
| 256 | 4 626.0 MB | 36.00 MB | **128.5×** | 256 | 36.0 MB |
| 1024 | 73 800.0 MB | 360.00 MB | **205.0×** | 1024 | 144.0 MB |

**P = 200**

| T | cat writes | slab writes | ratio | slab depth | slab VRAM |
|---:|---:|---:|---:|---:|---:|
| 178 | 7 246.5 MB | 89.16 MB | **81.3×** | 512 | 72.0 MB |
| 256 | 11 826.0 MB | 100.12 MB | **118.1×** | 512 | 72.0 MB |
| 1024 | 102 600.0 MB | 532.12 MB | **192.8×** | 1280 | 180.0 MB |

Multiply every column by `N` for the batched path.

### 1.2 Fast AR

Ten steps per frame, cache depth 1…10; position 0 is the free `None` branch, so `cat` fires
at depths 2…10 (Σ = 54).

* cat: `54 × 16 384` = **884 736 B = 864 KiB per frame**
* slab (capacity pinned to `num_codebooks` = 10, so it never grows): `10 × 16 384` =
  **163 840 B = 160 KiB per frame** → **5.40×**

| T | cat | slab | saved |
|---:|---:|---:|---:|
| 178 | 150.2 MB | 27.8 MB | 122.4 MB |
| 256 | 216.0 MB | 40.0 MB | 176.0 MB |
| 1024 | 864.0 MB | 160.0 MB | 704.0 MB |

Also removed: the transient 2× cache allocation at every cat, and 36×2 (slow) + 4×2×9
(fast) pool allocations per frame.

### 1.3 What that is worth in wall-clock — the honest part

Counting read+write = 2× the bytes, at the RTX 5070's ~672 GB/s:

| T (P = 200) | traffic saved | time saved over the utterance | per frame | share of 343 ms |
|---:|---:|---:|---:|---:|
| 178 | 6.99 GiB | 22.3 ms | 0.13 ms | **0.04 %** |
| 256 | 11.45 GiB | 36.6 ms | 0.14 ms | **0.04 %** |
| 1024 | 99.68 GiB | 318.5 ms | 0.31 ms | **0.09 %** |

The O(T²) term is genuinely gone (long-form generation no longer degrades quadratically),
but at the sequence lengths this model actually runs it was never the bottleneck. A 4.4 B
model at batch 1 reads ~15.2 GiB of *weights* per frame; a few hundred MB of KV shuffling
does not register against that.

---

## 2. What changed

### 2.1 `s2/nn.rs` — `KvCache` is now a preallocated slab (the assigned fix)

Each layer owns one contiguous `[b, n_kv, cap, head_dim]` slab. A step writes its `t_new`
positions **in place** with `slice_set` at offset `len` and returns
`slab.narrow(2, 0, len + t_new)`. New API surface: `KvCache::with_capacity(n_layers, cap)`
(exact preallocation) and `capacity()` (diagnostic). `new()`, `len()`, `append()`,
`advance()` keep their signatures, so no caller outside these files had to change.

**Sizing.** `with_capacity` honours the requested depth exactly; `new()` grows in
`KV_CHUNK = 256`-position steps. Chunked growth rather than a single up-front allocation is
deliberate: `DualArBackend::reset` is handed `cfg.slow.max_seq_len` = 32 768, and
preallocating that would be 4.8 GB on a 12 GB card already holding a 10.4 GB model.
`DriveParams::max_new_frames` defaults to 2048, so even a tight bound would be ~330 MB at
batch 1 and ~2.6 GB at N = 8. Chunking caps the over-allocation at 255 positions (≈ 37 MB
per batch element) while still removing 255/256 of the copying. Growth reallocations copy
the whole slab and happen `≈ T/256` times; that residual O(T²/256) term is included in the
`slab writes` column above.

**Bit-exactness.** `cat` and `slice_set` are both exact element copies, so the cache
*contents* are unchanged by construction. The one behavioural difference is that the cache
now hands back a **non-contiguous** `narrow` view instead of a freshly-`cat`-ed contiguous
buffer. That reaches the gemm through `repeat_kv`:

* `n_rep > 1` (the s2 slow AR and fast AR, both 32/8 = 4): `repeat_kv`'s `reshape` already
  materialised a contiguous `[b, n_head, len, head_dim]`, and still does — byte-identical
  input to the same gemm.
* `n_rep == 1` (the codec's bottleneck transformer, MHA): `repeat_kv` used to return the
  cat output unchanged, which happened to be contiguous. It now calls `.contiguous()`
  explicitly. This costs one extra copy per layer on a path that runs **once per
  utterance**, and buys a guarantee that every downstream `matmul` sees exactly the layout —
  and therefore exactly the cuBLAS/gemm configuration — it saw before.

**The batched left-padded path needed no special handling, and is on the new cache.** Left
padding lives entirely in the RoPE position ids (`gather_rope`) and the additive mask, never
in the cache layout: physical column `c` means the same thing under either storage scheme,
`slow_step_batch` writes one column for all N samples at the shared `pos`, and finished
samples are frozen by the driver, not by the cache. `attention_batched` calls the same
`append`. Verified by test (§3), not by inspection alone.

### 2.2 `s2/fast_ar.rs` — exact capacity

`expand` and `expand_batch` now build the per-frame cache with
`KvCache::with_capacity(n_layer, num_codebooks)`. The fast AR's final depth is known exactly
(1 conditioning prefix + 9 residual draws = 10), so it allocates once per frame and never
grows or re-copies.

### 2.3 `s2/slow_ar.rs` — the per-step device round-trip is gone

`embed` read its ids back off the device with
`inp.to_dtype(U32)?.flatten_all()?.to_vec1()?`. In the generation loop those ids came from
the host in the first place: `slow_step` did `Tensor::from_vec(frame.to_vec(), …)` and then
`embed` immediately copied them back. That is a host→device upload followed by a
**device→host read-back, and a `to_vec1` synchronises the CUDA stream** — once per decode
step on the single-sample path, and once *per sample* per step on the batched path.

`embed` is now split: `embed_ids(&[u32], rows, t)` holds the whole MCF body unchanged, and
`embed(&Tensor)` is a thin wrapper that does the read-back only where a real device tensor
is the input (prefill). `slow_step` passes its `frame` slice straight to `embed_ids`.

`slow_step_batch` additionally now embeds **all N frames in one MCF pass**, by treating the
batch axis as the embed's `t` axis: it builds the row-major `[1 + n_cb, N]` id matrix, calls
`embed_ids` once to get `[1, N, dim]`, and reshapes (a free view) to `[N, 1, dim]`. Every
gather and every arithmetic op in the MCF embed is per-column, so this is value-identical to
embedding each frame separately and `cat`-ing them. It replaces
`N × (1 upload + 1 sync + 11 index_selects + ~9 adds + mask/scale kernels)` plus an N-way
`cat` with one of each — for N = 16 that is roughly 16 stream syncs and ~600 kernel launches
removed *per step*.

### 2.4 `common/dualar.rs` — unchanged

No change was needed. The driver's loop structure is already correct: it computes logits for
the last position only, keeps no per-step host allocation of consequence, and its two
remaining `to_vec1`/`to_vec2` calls (semantic logits) are inherent to host-side sampling.

---

## 3. Verification

Seven CPU unit tests in `s2/nn.rs` (`cargo test -p syrinx-fish --lib s2::nn`, all passing):

* `slab_cache_matches_cat_cache_{single_sample, across_growth, batched, exact_capacity}` —
  each drives the new slab cache **and a verbatim copy of the old cat implementation**
  through the same prefill + decode sequence and asserts the returned `(k, v)` are
  element-for-element equal, at every layer of every step. `across_growth` deliberately
  straddles a `KV_CHUNK` reallocation; `batched` runs `b = 4`.
* `exact_capacity_is_honoured_and_never_grows`, `chunked_capacity_grows_only_at_chunk_boundaries` —
  pin the sizing policy on both sides of the boundary (the step that exactly fills a chunk
  must not reallocate; the next one must, by exactly one chunk).
* `repeat_kv_output_is_contiguous_from_a_narrowed_cache_view` — asserts the cache view is
  non-contiguous, that `repeat_kv` output is contiguous for both `n = 1` and `n = 4`, and
  that `n = 1` is still an exact value passthrough.

The full crate suite (11 tests) passes. `cargo check -p syrinx-fish` is clean; the `cuda`
feature compile-check was run with `scripts/test-all.env` sourced.

**Not verified:** anything on a GPU. No parity fixture was run, no synthesis was rendered,
no timing was measured on the target hardware. Per `CLAUDE.md`, full-coded ≠ verified.

---

## 4. Other per-step waste found — what I did NOT change, and why

### 4.1 ⚠ The likely real bottleneck: `Weights::linear` copies the whole weight matrix, every call

`nn.rs:61` and `nn.rs:72`:

```rust
pub fn linear(&self, x: &Tensor, wname: &str, bias: Option<&str>) -> Result<Tensor> {
    let w = self.g(wname)?;
    let y = x.broadcast_matmul(&w.t()?)?;      // <-- here
```

`x` is `[1, t, in]` (rank 3) and `w.t()` is `[in, out]` (rank 2). candle's
`broadcast_matmul` broadcasts the rhs to `[1, in, out]` and, per its own source comment
(*"TODO: Avoid concretising the broadcasted matrixes via contiguous"*), calls
`.contiguous()` on it. The broadcast layout is `strides = [0, 1, in]`, which is **not**
contiguous — so this **materialises a full transposed copy of the weight on every single
call**, and the copy is a *strided gather* (a transpose), not a memcpy.

Measured on CPU (release, `w = [9728, 2560]` — the s2 `w1` shape — `t = 1`, 100 iters):

```
rhs.t() strides=[1, 2560] contiguous=false
broadcast_as([1,k,out]) strides=[0, 1, 2560] contiguous=false
broadcast_matmul 3.937 s   matmul(broadcast_left) 0.092 s   ratio 42.6x
```

Every gemm in the model goes through this: `wqkv`, `wo`, `w1`, `w2`, `w3` in all 36 slow
layers and all 4 fast layers, `fast_output`, and the 760 MiB tied LM head (`linear_w`).

Weight bytes touched per generated frame (bf16, batch 1):

| | params | bf16 bytes |
|---|---:|---:|
| slow, 36 layers | 3.633 B | 6.77 GiB |
| tied LM head (`embeddings.weight`) | 398.8 M | 760.6 MiB |
| **slow step** | | **7.51 GiB** |
| fast step (4 layers + `fast_output`) | 414.3 M | 0.77 GiB |
| **fast, ×10 codebooks** | | **7.71 GiB** |
| **total per frame** | | **15.23 GiB** |

At 672 GB/s that is a **24.3 ms/frame roofline (RTF 0.52 — comfortably realtime)**. The
copy adds one write of the weight plus one *uncoalesced* read of it. On a bf16 tensor read
with a 2560-element stride, each 32 B sector yields one useful 2 B element, so the read can
amplify by up to 16× before L2 reuse claws any of it back:

| copy read amplification | predicted ms/frame | predicted RTF |
|---:|---:|---:|
| 1× (perfect L2) | 73.0 | 1.57 |
| 4× | 146.0 | 3.14 |
| 8× | 243.3 | 5.24 |
| 16× (no reuse) | 437.9 | 9.43 |
| — | **measured 343** | **measured 7.4** |

The measured point sits inside that band at an effective ≈ 11.6× amplification. **This is
circumstantial, not proof** — kernel-launch overhead (~3 000 launches/frame of mostly
tiny tensors), WSL's higher launch and sync cost, and the ~12 host round-trips per frame all
live in the same budget and I cannot separate them without a profiler on the box. But it is
the only candidate I found that is the right order of magnitude, and it is cheap to test.

**Why I did not fix it.** The brief's rule 4: any fix changes the cuBLAS call.

* Today: `C = A × B` with `B` a **contiguous** `[1, in, out]` copy → `CUBLAS_OP_N`, `lda = out`.
* One-line fix (`x.matmul(&w.t()?.broadcast_left(b)?)`, exactly what `candle_nn::Linear`
  does) → `CUBLAS_OP_T`, `lda = in`, stride-0 batch, **no copy**. Same mathematics, but
  cuBLAS may select a different kernel/k-split, so bf16 results are not *guaranteed*
  bit-identical. In a sampler that runs top-k/top-p over 155 776 logits, one flipped ulp can
  change the whole utterance.

Two remedies, in order of preference:

1. **Bit-exact, needs a load-time change (outside my file scope).** Store the weights
   already transposed — a contiguous `[in, out]` buffer per linear — at load
   (`s2/load.rs`), and have `linear` do `x.matmul(&wT.broadcast_left(b)?)`. That reproduces
   *today's exact* `OP_N`, `lda = out` gemm over *today's exact* bytes, with the copy paid
   once at load instead of once per call. `embeddings.weight` is used both as a gather table
   and as the tied head, so it needs care (`index_select` on dim 1, or keep both — 760 MiB,
   which the current code already allocates transiently every step anyway).
2. **One line, needs a parity re-run.** Apply the `broadcast_left` form and run the s2-pro
   parity group (`./scripts/run-fish.sh s2-pro --parity`). If it passes at the existing
   tolerance, ship it. This is the 30-minute experiment I would run first, purely to confirm
   or kill the hypothesis, before investing in (1).

### 4.2 `repeat_kv` + `transpose(2,3).contiguous()` — 86 % of the remaining KV traffic

Per layer per step at cache length `Lc` (batch 1, bf16), after this change:

| op | bytes written | status |
|---|---|---|
| ~~cat K + cat V~~ | ~~`4096 · Lc`~~ | removed by the slab cache |
| `repeat_kv` K and V (8→32 heads) | `16 384 · Lc` | **untouched** |
| `k_full.transpose(2,3).contiguous()` | `8 192 · Lc` | **untouched** |

All 36 layers: 197 MB written per step at `Lc = 200`, 394 MB at `Lc = 400`, 1 181 MB at
`Lc = 1200` → 0.61 / 1.23 / 3.69 ms of traffic per step. The cat I removed was only 14 % of
that block.

Both are avoidable — candle's `gemm_config` accepts a strided/transposed rhs and would give
`repeat_kv`-free GQA via a reshaped `[b, n_kv, n_rep·t, hd]` query, and the explicit
`.contiguous()` after `transpose(2, 3)` is unnecessary because a transposed rhs maps to
`CUBLAS_OP_T` directly. Both change the gemm configuration, so both fall under rule 4:
**reported as options, not applied.** They are worth ~0.5–1 % of frame time at T ≈ 178 and
~3 % at T ≈ 1200, so they are not urgent either way.

### 4.3 Audited and found clean / inherent

* **Logits over the last position only** — already correct: `head`/`head_batch` `narrow` to
  the last column *before* the LM-head matmul.
* **RoPE tables** — precomputed once in `SlowAr::new` / `FastAr::new`; the per-step access
  is a `narrow`/`index_select` view. No recomputation.
* **Masks** — `slow_step` passes `None` (a single new token over the full cache is
  unconditionally visible), so no mask is materialised in the single-sample hot loop.
  `slow_step_batch` does rebuild an `[N, 1, 1, pos+1]` left-pad mask each step (O(N·T) host
  work and a small upload per step, so O(N·T²) over a run) — at N = 16, T = 200 that is
  ~13 KB/step, not worth the complexity of an incremental mask. Left alone, noted here.
* **Remaining host round-trips** — 1 `to_vec1` for the semantic logits (623 KB) + 9
  `to_vec1` for the fast codebooks, per frame. These are inherent to sampling on the host
  and cannot be removed without moving the sampler onto the device, which would change the
  PRNG stream and break reproducibility. The *avoidable* one (the embed read-back) is gone.
* **`format!` + `HashMap<String, Tensor>` lookups** per weight per layer per step (~360 on
  the slow step) — real CPU overhead, but tens of microseconds against a 343 ms frame.
  Not worth touching before §4.1 is settled.

---

## 5. Honest estimate of the speedup from *this* change

Labelled an estimate: **no GPU was used, nothing here was timed on the target hardware.**

* **Single-sample path, T ≈ 178:** ~0.05 % from the KV traffic itself (§1.3), plus one
  removed stream synchronisation per decode step out of ~12 per frame. If a WSL sync costs
  1–3 ms, that is another 0.3–1 %. **Call it 0.5–2 %, and be prepared for it to be
  unmeasurable in the noise.**
* **Batched path (`--batch`), N = 16:** removing N syncs and ~600 kernel launches per step
  (§2.3) is the bigger win here, because the batched slow step's fixed overhead is amortised
  over the same single weight read. **Estimate 5–20 %, with low confidence** — it depends
  entirely on how much of the batched step is currently launch-bound, which I cannot measure.
* **Long-form (T ≥ 1024):** the O(T²) term is gone. At T = 1024 the cat was writing 100 GB
  per utterance and growing quadratically; that ceiling is removed regardless of how small
  the constant is today.
* **§4.1, if the hypothesis holds and the fix lands:** the model becomes 3–10× faster and
  crosses into realtime. That is where the 7.4× lives, if I am right.

Be skeptical of the last bullet in particular. It is a bandwidth model fitted to a single
measured number, with three other unquantified candidates (launch overhead, WSL sync cost,
L2 behaviour) in the same budget. It predicts the measurement well, which is suggestive and
nothing more.

## 6. Blocker

**No GPU access this pass** (cuda:0 reserved, cuda:1 running a production render), so:

* no parity fixture was run against the new cache — the equivalence argument rests on the
  CPU tests in §3 plus the layout analysis in §2.1, not on the s2-pro parity gate;
* no before/after timing exists, so every number in §5 is a model, not a measurement;
* the §4.1 hypothesis is untested. **The single highest-value next action is to run the
  s2-pro parity group and a one-clip timing with and without the `broadcast_left` form of
  `Weights::linear`.** That one experiment decides whether §4.1 is the answer or a red
  herring, and it takes minutes on the box.

Before trusting any of this on hardware: `./scripts/verify.sh` (Fish s2-pro group) must be
green, and a render must be A/B-compared against a pre-change render at the same seed —
identical output is the acceptance criterion for the cache rewrite, since it claims to be
bit-exact.
