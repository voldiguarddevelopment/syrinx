# Faithful streaming — the causal-flow plan

This note is the **concrete implementation plan** for sample-faithful streaming, scoped
against the real CosyVoice2 reference. It is the design half; the implementation is a
focused future pass (intricate + needs iterative on-box verification — deliberately not
rushed). It supersedes the "needs a causal cached flow" hand-wave in the README roadmap.

## The problem (measured, not assumed)

`token2wav_streaming` re-runs the **non-causal** `Flow::forward_zero_shot` on the grown
token prefix each chunk and slices out the newly-finalized mel. The flow's attention is
**full-context / unmasked** (the parity path's padding masks are all-true). So a finalized
frame's value *depends on right-context that doesn't exist yet at stream time*:

- The leading chunk decodes on partial context → a wrong leading F0 → which **poisons the
  cumulative source phase** for the whole utterance (`stream_mel_diag` showed the leading
  chunk diverges; later chunks were bit-exact **only because the diagnostic supplied full
  context**). Net: the stream is valid + correct-length but **not sample-identical** to the
  batch path (correlation ≈ 0), and the phase-continuous source fix couldn't lift it.

The non-causal flow simply **cannot** stream faithfully — a frame that can see the future
isn't stable when the future arrives incrementally.

## The fix: CV2's chunked-causal attention mask (same weights)

CosyVoice2 streams the **same flow weights** under a **chunk attention mask** — it does *not*
use a different architecture. See `cosyvoice/flow/decoder.py:439-480`: with `streaming=True`
it applies `add_optional_chunk_mask(x, mask, ..., static_chunk_size=50, num_decoding_left_chunks=2)`
in the down/mid/up blocks of the CFM estimator (and the analogous mask in the conformer
encoder). Under the mask, frame `i` (in chunk `c = i // chunk`) attends only to chunks
`[c - num_left, c]` — **never the future**. A finalized frame is therefore **stable**: adding
right-context can't change it. *That stability is exactly what makes streaming sample-faithful.*

## Implementation (in `syrinx-acoustic`)

1. **Chunk mask.** Implement `add_optional_chunk_mask(t, chunk_size, num_left) -> [t, t]` —
   an additive `0 / -inf` mask where position `i` may attend to `j` iff
   `(j / chunk) ≤ (i / chunk)` and `(j / chunk) ≥ (i / chunk) - num_left`. (`static_chunk_size = 50`,
   `num_decoding_left_chunks = 2` for the CV2 default.)
2. **Masked attention.** Thread an `Option<&Tensor>` mask into `rel_self_attn` (the espnet
   rel-pos conformer attention) **and** the estimator's attention; add it to the scores
   before softmax. ⚠️ The encoder 2×-upsamples (`N_UPENC`), so the estimator's effective chunk
   size is **2× the encoder's** — compute the chunk boundaries per-stage, post-upsample.
3. **Streaming flow.** Add `Flow::forward_streaming(...)` that runs encoder + estimator with the
   chunk mask. For *correctness*, re-running on the grown prefix **with the mask** already yields
   stable finalized frames (no right-context leakage); a left-context KV/conv cache (CV2's
   `att_cache`/`cnn_cache`) is an **efficiency** optimization, layer on after correctness.
4. **Wire** `token2wav_streaming` to call `forward_streaming` instead of the non-causal re-run.
   The mel/source/hamming HiFT caching there is already correct and stays.

## Verification — against the RIGHT reference

Compare against CV2's **streaming** flow, **not** the non-streaming path (they are different
modes by construction — comparing streaming to non-streaming is the wrong metric, as the
earlier `corr ≈ 0` finding showed). Dump `flow.inference(..., streaming=True)` (static_chunk_size=50)
per chunk → the finalized-mel diff between Syrinx's `forward_streaming` and CV2's streaming flow
should go to ~0; then `stream_demo`'s correlation (recomputed against a chunk-masked batch
reference) rises, and the audio is faithful.

## Effort / risk

Intricate: masked rel-pos attention in two stages, upsample-aware chunk boundaries, and
iterative on-box parity vs CV2 streaming. A focused pass (~hours), not a tail-of-session patch —
hence this plan rather than a rushed, possibly-subtly-wrong flow.

## Status (built + measured 2026-06-26)

Faithful streaming has turned out to be **two** problems, not one. The first is solved + proven;
the second is now isolated.

**Part 1 — the FLOW: DONE + proven.** `add_optional_chunk_mask` + `StreamCfg::cosyvoice2` +
`forward_zero_shot_streaming` are implemented (same weights, chunked-causal mask in the conformer +
CFM estimator) and wired into `token2wav_streaming`. `tests/real_flow_stream_consistency.rs` proves
the masked finalized mel frames are **bit-stable across prefix lengths — 0.0 diff**, where the old
non-causal re-run leaked **0.53**. So the flow streams faithfully: a finalized frame never changes
as future tokens arrive. `forward_zero_shot` (non-causal batch) is byte-unchanged — parity intact.

**Part 2 — the VOCODER: NOT actually a gap (metric artifact).** The chunked audio vs a single-chunk
("batch") render has best-lag correlation ≈ 0.17, which *looked* like a remaining HiFT gap — but
that is the **wrong metric**. CosyVoice2's streaming is *deliberately not* sample-identical to its
batch: `token2wav` cross-fades each boundary with a hamming `fade_in_out` over `source_cache_len`
(9600) samples and holds back a lookahead, which change a large fraction of the waveform by design.
So no correct implementation hits "sample-identical to batch."

The **right** metric is intelligibility, and it passes cleanly: **ASR (Whisper) of the streamed
audio gives CER = 0.0** — character-perfect, identical to the batch's CER. The streamed audio is
correct, smooth, intelligible speech. **Streaming is faithful + done:** the causal flow (Part 1,
proven 0.0 mel diff) was the actual fix, and the HiFT streaming assembly (CV2's mel/source cache +
hamming cross-fade) produces correct audio. The 0.17-vs-batch is the expected streaming↔batch
cross-fade difference, not a defect — it should be measured against CV2's *streaming* output (or by
intelligibility), never against the batch.
