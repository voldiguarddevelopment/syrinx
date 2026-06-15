# Syrinx reference architecture (concrete, small, real — swap pretrained weights later)

A real-but-generic architecture so the Rust inference is REAL code gated on parity
against a pure-Python reference of the SAME architecture. The pretrained base later
swaps its weights into the same shapes. Init weights are deterministic (a seeded
xorshift PRNG, the SAME algorithm in Python + Rust, so both produce identical
weights without a weights file — see §Weights). All math is f32.

## Global
- dtype f32; RMSNorm eps = 1e-5; PRNG = xorshift64 seeded per-tensor (see below).
- Determinism: every forward is a pure function of (inputs, seed); no randomness at
  inference except the ODE start noise, which is itself seeded.

## syrinx-core — tensor + ops (the foundation everything uses)
- `Tensor { data: Vec<f32>, shape: Vec<usize> }` (row-major).
- ops, each a pure function with a property + golden test:
  matmul(A[m,k], B[k,n])->[m,n]; add; mul; linear(x[*,in], W[out,in], b[out])->[*,out];
  rmsnorm(x[*,d], w[d]); softmax(x, last-axis); silu; rope(x[*,h,hd], pos, theta);
  causal_mask; embed(ids, table[V,d]).

## syrinx-lm — semantic LM (decoder-only transformer, Llama-style)
- config: vocab=512, dim=128, n_layers=4, n_heads=4, head_dim=32, ffn_hidden=256
  (SwiGLU), max_seq=256, rope_theta=10000, pre-RMSNorm, causal.
- block: h += attn(rmsnorm(h)); h += swiglu_ffn(rmsnorm(h)).
- forward(token_ids[T]) -> logits[T, vocab]; final rmsnorm + tied/separate lm_head.

## syrinx-speaker — speaker encoder
- input ref_mel[T,80] -> 2-layer transformer enc (dim=128) -> mean-pool[128] ->
  linear -> speaker_emb[128] (L2-normalized).

## syrinx-acoustic — flow-matching DiT decoder
- input: x_t[T,80] (noisy mel), t[scalar in 0..1], cond = speaker_emb[128].
- dim=128, n_layers=4, n_heads=4, adaLN-zero (t + cond -> per-layer scale/shift/gate).
- velocity v(x_t,t,cond)->[T,80]. ODE: Euler, N=8 steps, x0 = seeded noise -> x1 = mel.

## syrinx-vocoder — HiFi-GAN generator (small)
- mel[T,80] -> conv1d pre(80->128) -> 2x [transposed_conv1d(up=4) + MRF residual] ->
  conv1d post(->1) -> tanh -> waveform[T*16]. (hop = 16 for the reference.)

## Weights (no file needed)
- A named tensor's weights = xorshift64(seed = hash(name)) -> u64 stream -> f32 in
  (-1,1) scaled by 0.02 (Normal-ish via uniform is fine for a REFERENCE; parity is
  what matters, not training). The Rust weight loader reproduces the SAME stream from
  the SAME name -> identical weights. The "weight loader" task therefore loads-by-name
  deterministically; a later task swaps in a real safetensors loader for pretrained.

## Goldens
- For each component, the pure-Python reference computes forward(fixed_input) and
  writes golden output tensors (JSON: {name, shape, data, tol}) under
  tests/golden/parity/. Rust frozen tests load the JSON and assert max-abs <= tol
  (tol 1e-4 for single ops, 1e-3 for full forwards). Plus property tests that need
  no golden (softmax sums to 1, rmsnorm var≈1, causal attn ignores future, rope
  preserves norm, ODE with v=0 is identity, etc.).
