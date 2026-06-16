//! T-02.02c-a — the layer-0 forward stage, frozen RED parity tests.
//!
//! `syrinx-lm` exposes the *layer-0 forward stage*: the three entry points that
//! assemble the embedding → layer-0 attention → layer-0 block path by fetching
//! the `reference.py` §3 named weights (PARITY.md §3) and composing the already
//! verified `embed`/`attention`/`block`:
//!
//!   * `embed_tokens(token_ids) -> [T, dim]` — gather of the `tok_embeddings`
//!     `[vocab, dim]` table rows by id (`embed`).
//!   * `layer_attention(x[T,dim], layer, positions) -> [T,dim]` — builds the
//!     `layers.{L}.attention.w{q,k,v,o}.weight` `[dim,dim]` projections by name
//!     and runs the verified `attention` (this fn does NOT apply the pre-norm;
//!     the caller norms its input, exactly as the block does internally).
//!   * `transformer_block(h[T,dim], layer, positions) -> [T,dim]` — builds every
//!     `layers.{L}.` weight by name (`attention_norm.weight`, the four attention
//!     projections, `ffn_norm.weight`, `feed_forward.w{1,3,2}.weight`) and runs
//!     the verified pre-RMSNorm residual `block`.
//!
//! Driven on the fixed input `[1,5,9,2,0]` (positions `[0,1,2,3,4]`) and pinned
//! at 1e-4 max-abs against the activation-scale goldens `lm_embed.json`,
//! `lm_attn0.json`, `lm_block0.json`.
//!
//! ────────────────────────────────────────────────────────────────────────────
//! WHY TWO CRITERION SUB-CLAUSES USE DETECTABLE SUBSTITUTE DISCRIMINATORS
//! ────────────────────────────────────────────────────────────────────────────
//! The criteria for C2 and C3 also name two "wrong-variant must fail" controls
//! that are NUMERICALLY UNSATISFIABLE against these frozen 1e-4 goldens, because
//! the name-seeded weights live in `[-0.02, 0.02)` (the documented tiny-weight
//! regime). Measured against the goldens (Python port of the §2 PRNG + §4/§5
//! ops, matching the goldens to ~1e-9):
//!
//!   * "swapping the adjacent RoPE pair ordering" (C2) moves `lm_attn0` by only
//!     ~3e-10. Q·K scores are ~1e-4, so softmax is ~uniform and the attention
//!     output is ~the mean of the values REGARDLESS of RoPE/scale — the score
//!     path is invisible at 1e-4.
//!   * "swapping the SwiGLU gate/up weights w1↔w3" (C3) moves `lm_block0` by only
//!     ~1.8e-9. The ENTIRE FFN sub-block output here is ~4.6e-7: dropping the FFN
//!     completely changes `lm_block0` by 4.6e-7. So `lm_block0` at 1e-4 pins
//!     embed + attention + the pre-norm residual order, and is BLIND to the FFN.
//!
//! Asserting those two swaps "diverge > 1e-4" would be FALSE, so the GREEN phase
//! could never satisfy them honestly — the exact pitfall the prior split commit
//! ("fix the unsatisfiable block-count control") and the project memory record.
//! The RoPE-realness and the SwiGLU gate/up roles ARE already gated, at a healthy
//! activation scale (×50 weights), by the frozen `attention_prop.rs` /
//! `block_prop.rs` property tests — this task is the NUMERIC PARITY task, not a
//! re-statement of those structural properties.
//!
//! So each criterion is covered by its parity gate plus DETECTABLE, intent-
//! faithful divergence controls that hold against these goldens (all measured
//! > 1e-4): a zeroed attention (C2, ~4.75e-4) and a transposed output projection
//! (C2, ~7.8e-4); an identity block (C3, ~4.75e-4) and an attention-norm↔ffn-norm
//! swap (C3, ~6.1e-4); a wrong tensor name and a transposed weight (C4,
//! ~9e-4 / ~7.8e-4). No assertion in this file is one I have not verified holds.
//!
//! RED: `syrinx-lm` exposes none of `embed_tokens`/`layer_attention`/
//! `transformer_block`, so this target fails to build. GREEN adds them as the
//! minimal §3-named-weight assembly over the verified `embed`/`attention`/`block`
//! — placed in their OWN source file (e.g. `crates/syrinx-lm/src/stage.rs`) so
//! the task-scoped mutation gate targets only the new assembly code and leaves
//! the byte-identical `lib.rs` (`attention`) and `block.rs` untouched, and so
//! every new fn is transitively exercised by the parity gates below. This file
//! is frozen at red-pass; do not edit it in GREEN.

use syrinx_core::{rmsnorm, weights, Tensor};
use syrinx_lm::{attention, block, embed_tokens, layer_attention, transformer_block};

// LM config (PARITY.md §3 / REFERENCE.md): vocab=512, dim=128, ffn_hidden=256.
const VOCAB: usize = 512;
const DIM: usize = 128;
const FFN_HIDDEN: usize = 256;
// RMSNorm epsilon (PARITY.md "Global": eps = 1e-5).
const EPS: f32 = 1e-5;

// The fixed parity input (criteria / goldens `input.token_ids`).
const TOKENS: [usize; 5] = [1, 5, 9, 2, 0];
const POSITIONS: [usize; 5] = [0, 1, 2, 3, 4];
const T: usize = 5;

// Every golden is pinned at 1e-4 max-abs (REFERENCE.md "Goldens": 1e-4 for ops /
// activation-scale stages). The goldens also carry their own `tol` field (1e-4).
const TOL: f32 = 1e-4;

// ----- golden plumbing --------------------------------------------------------

fn load(name: &str) -> serde_json::Value {
    let path = format!("{}/tests/golden/parity/{}", env!("CARGO_MANIFEST_DIR"), name);
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

/// A flat JSON number array → `Vec<f32>` (the golden `data` field).
fn flat(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .expect("data is an array")
        .iter()
        .map(|x| x.as_f64().expect("data cell is a number") as f32)
        .collect()
}

/// A JSON integer array → `Vec<usize>` (the golden `shape` field).
fn ints(v: &serde_json::Value) -> Vec<usize> {
    v.as_array()
        .expect("integer array")
        .iter()
        .map(|x| x.as_u64().expect("integer cell") as usize)
        .collect()
}

fn tol(g: &serde_json::Value) -> f32 {
    g["tol"].as_f64().expect("tol is a number") as f32
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "compared vectors must be equal length");
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

// ----- named-weight helpers (PARITY.md §3 literal tensor names) ---------------

/// The `[rows, cols]` weight matrix named `name` from the T-02.01c name-seeded
/// stream (`weights` returns `rows*cols` deterministic f32 in `[-0.02, 0.02)`).
fn mat(name: &str, rows: usize, cols: usize) -> Tensor {
    Tensor::new(weights(name, rows * cols), vec![rows, cols])
}

/// The `[d]` weight vector named `name` (an RMSNorm weight).
fn vecw(name: &str, d: usize) -> Tensor {
    Tensor::new(weights(name, d), vec![d])
}

/// Transpose a square `[n, n]` tensor (used to build the "wrong orientation"
/// `[in,out]` controls — `linear` expects `[out,in]`, so the transpose of a
/// square projection is a genuinely different, mis-oriented matrix).
fn transpose_sq(t: &Tensor) -> Tensor {
    let n = t.shape()[0];
    assert_eq!(t.shape(), &[n, n], "transpose_sq needs a square tensor");
    let src = t.data();
    let mut out = vec![0.0f32; n * n];
    for r in 0..n {
        for c in 0..n {
            out[c * n + r] = src[r * n + c];
        }
    }
    Tensor::new(out, vec![n, n])
}

/// The layer-0 attention projections `(wq, wk, wv, wo)`, each `[dim, dim]`, built
/// from the literal §3 names under prefix `layers.0.attention.`.
fn layer0_attention_weights() -> (Tensor, Tensor, Tensor, Tensor) {
    (
        mat("layers.0.attention.wq.weight", DIM, DIM),
        mat("layers.0.attention.wk.weight", DIM, DIM),
        mat("layers.0.attention.wv.weight", DIM, DIM),
        mat("layers.0.attention.wo.weight", DIM, DIM),
    )
}

/// The layer-0 FFN weights `(w1, w3, w2)` — gate `[ffn_hidden,dim]`, up
/// `[ffn_hidden,dim]`, down `[dim,ffn_hidden]` — under `layers.0.feed_forward.`.
fn layer0_ffn_weights() -> (Tensor, Tensor, Tensor) {
    (
        mat("layers.0.feed_forward.w1.weight", FFN_HIDDEN, DIM),
        mat("layers.0.feed_forward.w3.weight", FFN_HIDDEN, DIM),
        mat("layers.0.feed_forward.w2.weight", DIM, FFN_HIDDEN),
    )
}

/// The golden embedding `[5,128]` as a `Tensor` — the known token embedding for
/// `[1,5,9,2,0]` (equal to `embed(TOKENS, tok_embeddings)`). Used as the isolated
/// input to the attention/block parity so C2/C3 do not depend on the C1 impl.
fn golden_embed() -> Tensor {
    let g = load("lm_embed.json");
    Tensor::new(flat(&g["data"]), ints(&g["shape"]))
}

// =====================================================================
// C1 — token embedding parity (gather of `tok_embeddings` rows by id)
// =====================================================================

#[test]
fn test_embed_tokens_layer0_parity() {
    let g = load("lm_embed.json");
    let want = flat(&g["data"]);
    let want_shape = ints(&g["shape"]);

    // The stage gathers row `id` of the `tok_embeddings` `[vocab, dim]` table for
    // each token id; the result is `[5,128]` and matches the golden within 1e-4.
    let out = embed_tokens(&TOKENS);
    assert_eq!(out.shape(), want_shape.as_slice(), "embedding shape must be [5,128]");
    assert_eq!(want_shape, vec![T, DIM], "golden embedding shape is [5,128]");
    assert!(
        max_abs_diff(out.data(), &want) <= tol(&g),
        "embed_tokens([1,5,9,2,0]) must match lm_embed.json within 1e-4"
    );
    assert!(tol(&g) <= TOL, "golden tol must be <= 1e-4");

    // Corrupting ANY single embedding value breaks the parity assertion: the
    // check is sensitive element-wise, not just in aggregate.
    let mut corrupted = out.data().to_vec();
    corrupted[0] += 1.0;
    assert!(
        max_abs_diff(&corrupted, &want) > TOL,
        "corrupting one embedding value must fail the 1e-4 parity assertion"
    );
    // Non-vacuous: the un-corrupted output does pass, so the failure above is due
    // to the corruption, not a perpetually-failing check.
    assert!(
        max_abs_diff(out.data(), &want) <= TOL,
        "the un-corrupted embedding must pass the same assertion"
    );
}

// =====================================================================
// C2 — layer-0 attention sub-output parity
//   attention(rmsnorm(embed, attention_norm.weight, eps=1e-5), layer=0, pos)
// =====================================================================

#[test]
fn test_attention_sub_output_layer0_parity() {
    let g = load("lm_attn0.json");
    let want = flat(&g["data"]);
    let want_shape = ints(&g["shape"]);
    assert_eq!(want_shape, vec![T, DIM], "golden attn0 shape is [5,128]");

    // n1 = rmsnorm(embed, layers.0.attention_norm.weight, eps=1e-5): the exact
    // pre-norm the block feeds its attention sub-block.
    let embed = golden_embed();
    let attn_norm = vecw("layers.0.attention_norm.weight", DIM);
    let n1 = rmsnorm(&embed, &attn_norm, EPS);

    // The stage's layer-0 attention sub-output matches the golden within 1e-4.
    let out = layer_attention(&n1, 0, &POSITIONS);
    assert_eq!(out.shape(), want_shape.as_slice(), "attention sub-output shape must be [5,128]");
    assert!(
        max_abs_diff(out.data(), &want) <= tol(&g),
        "layer_attention(rmsnorm(embed,attn_norm), 0, pos) must match lm_attn0.json within 1e-4"
    );

    // Cross-check: `layer_attention(_, 0, _)` is exactly the verified `attention`
    // run on the §3-named layer-0 projections — so the named-weight assembly is
    // what reproduces the golden.
    let (wq, wk, wv, wo) = layer0_attention_weights();
    let named = attention(&n1, &wq, &wk, &wv, &wo, &POSITIONS);
    assert!(
        max_abs_diff(out.data(), named.data()) <= TOL,
        "layer_attention must equal attention() on the layers.0.attention.* weights"
    );

    // DETECTABLE control #1 — non-vacuity: a zeroed attention (all-zeros output)
    // diverges from the golden by max|lm_attn0| (~4.75e-4 > 1e-4), so the parity
    // gate is pinning a genuinely non-trivial signal.
    let zeroed = vec![0.0f32; T * DIM];
    assert!(
        max_abs_diff(&zeroed, &want) > TOL,
        "an all-zeros attention must diverge from lm_attn0 (the golden is non-trivial)"
    );

    // DETECTABLE control #2 — the output projection is real and correctly
    // oriented: running attention with `wo` transposed (a mis-oriented `[in,out]`
    // matrix) diverges from the golden by ~7.8e-4 (> 1e-4). This confirms the
    // attention path actually applies `wo` (not an identity / stub). RoPE-pair
    // ordering itself is below 1e-4 at this scale (see header) and is gated by
    // the frozen `attention_prop.rs` property test.
    let wo_t = transpose_sq(&wo);
    let wrong_wo = attention(&n1, &wq, &wk, &wv, &wo_t, &POSITIONS);
    assert!(
        max_abs_diff(wrong_wo.data(), &want) > TOL,
        "transposing the output projection wo must diverge from lm_attn0"
    );
}

// =====================================================================
// C3 — full layer-0 block parity (pre-RMSNorm residual order)
//   transformer_block(embed, layer=0, positions)
// =====================================================================

#[test]
fn test_transformer_block_layer0_parity() {
    let g = load("lm_block0.json");
    let want = flat(&g["data"]);
    let want_shape = ints(&g["shape"]);
    assert_eq!(want_shape, vec![T, DIM], "golden block0 shape is [5,128]");

    let embed = golden_embed();

    // The stage's full layer-0 block matches the golden within 1e-4.
    let out = transformer_block(&embed, 0, &POSITIONS);
    assert_eq!(out.shape(), want_shape.as_slice(), "block output shape must be [5,128]");
    assert!(
        max_abs_diff(out.data(), &want) <= tol(&g),
        "transformer_block(embed, 0, pos) must match lm_block0.json within 1e-4"
    );

    // DETECTABLE control #1 — non-vacuity: an identity block (returning the
    // embedding unchanged) diverges from the golden by ~4.75e-4 (> 1e-4), so the
    // block genuinely transforms its input (the attention residual is present).
    assert!(
        max_abs_diff(embed.data(), &want) > TOL,
        "an identity block (returning embed) must diverge from lm_block0"
    );

    // DETECTABLE control #2 — the two RMSNorm weights are distinct and correctly
    // placed: building the block with the attention-norm and ffn-norm weights
    // SWAPPED diverges from the golden by ~6.1e-4 (> 1e-4). This pins the block's
    // pre-norm structure. (The SwiGLU gate/up roles, w1↔w3, are below 1e-4 here —
    // the FFN output is ~4.6e-7, see header — and are gated by `block_prop.rs`.)
    let attn_norm = vecw("layers.0.attention_norm.weight", DIM);
    let ffn_norm = vecw("layers.0.ffn_norm.weight", DIM);
    let (wq, wk, wv, wo) = layer0_attention_weights();
    let (w1, w3, w2) = layer0_ffn_weights();
    let swapped_norms = block(
        &embed,
        &ffn_norm, // wrong: ffn_norm used where attention_norm belongs
        &attn_norm, // wrong: attention_norm used where ffn_norm belongs
        (&wq, &wk, &wv, &wo),
        (&w1, &w3, &w2),
        &POSITIONS,
    );
    assert!(
        max_abs_diff(swapped_norms.data(), &want) > TOL,
        "swapping the attention-norm and ffn-norm weights must diverge from lm_block0"
    );
}

// =====================================================================
// C4 — literal §3 layer-0 tensor names + [out,in] matrix orientation
// =====================================================================

#[test]
fn test_block_layer0_tensor_names_and_shapes() {
    let g = load("lm_block0.json");
    let want = flat(&g["data"]);

    let embed = golden_embed();
    let attn_norm = vecw("layers.0.attention_norm.weight", DIM);
    let ffn_norm = vecw("layers.0.ffn_norm.weight", DIM);
    let (wq, wk, wv, wo) = layer0_attention_weights();
    let (w1, w3, w2) = layer0_ffn_weights();

    // The literal §3 names under `layers.0.` — `attention_norm.weight`,
    // `attention.w{q,k,v,o}.weight`, `ffn_norm.weight`, `feed_forward.w{1,3,2}.
    // weight`, each in `[out,in]` orientation — reproduce the golden, exactly as
    // the stage's `transformer_block` does.
    let reference = block(
        &embed,
        &attn_norm,
        &ffn_norm,
        (&wq, &wk, &wv, &wo),
        (&w1, &w3, &w2),
        &POSITIONS,
    );
    assert!(
        max_abs_diff(reference.data(), &want) <= tol(&g),
        "the literal layers.0.* §3 names in [out,in] orientation must reproduce lm_block0"
    );
    // Tie to the stage: `transformer_block` reproduces the same golden, so it is
    // using these exact names/shapes internally.
    let out = transformer_block(&embed, 0, &POSITIONS);
    assert!(
        max_abs_diff(out.data(), &want) <= tol(&g),
        "transformer_block must reproduce lm_block0 via the layers.0.* names"
    );

    // WRONG NAME — substituting a different layer's tensor (`layers.1.attention.
    // wo.weight`) for the layer-0 output projection diverges by ~9e-4 (> 1e-4):
    // the golden is reproduced only by the layer-0 names, not just any same-shape
    // tensor.
    let wo_wrong_name = mat("layers.1.attention.wo.weight", DIM, DIM);
    let wrong_name = block(
        &embed,
        &attn_norm,
        &ffn_norm,
        (&wq, &wk, &wv, &wo_wrong_name),
        (&w1, &w3, &w2),
        &POSITIONS,
    );
    assert!(
        max_abs_diff(wrong_name.data(), &want) > TOL,
        "using a wrong tensor name (layers.1.* instead of layers.0.*) must diverge from lm_block0"
    );

    // WRONG ORIENTATION — feeding the output projection as its transpose (an
    // `[in,out]` matrix instead of `[out,in]`) diverges by ~7.8e-4 (> 1e-4):
    // matrix orientation matters, so the stage must store projections `[out,in]`.
    let wo_transposed = transpose_sq(&wo);
    let wrong_shape = block(
        &embed,
        &attn_norm,
        &ffn_norm,
        (&wq, &wk, &wv, &wo_transposed),
        (&w1, &w3, &w2),
        &POSITIONS,
    );
    assert!(
        max_abs_diff(wrong_shape.data(), &want) > TOL,
        "a transposed [in,out] output projection must diverge from lm_block0"
    );
}
