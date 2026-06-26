# Syrinx output watermark

A **real, pure-Rust, training-free spread-spectrum watermark** stamped on Syrinx's
24 kHz mono `f32` output. It is detectable on the unmodified waveform and robust to
*light* post-editing. It is **not** an adversarial/learned watermark, and this doc
states the boundary plainly — a modest, well-characterized mark beats a fake
"robust" claim.

Source: [`crates/syrinx-serve/src/watermark.rs`](../src/watermark.rs).
Tests: [`tests/watermark.rs`](../../../tests/watermark.rs) (repo root, model-free).

## API

```rust
use syrinx_serve::watermark::{embed_watermark, detect_watermark, WmResult};

// Embed a 16-bit payload, keyed by a u64 shared secret, in place.
embed_watermark(&mut audio /* &mut [f32] @ 24 kHz */, key, payload /* u16 */);

// Detect: high z-score + recovered payload when present; ~1 when absent.
let r: WmResult = detect_watermark(&audio, key).unwrap();
// WmResult { present: bool, confidence: f64, payload: u16 }
```

Also: `embed_watermark_with_amp(audio, key, payload, amp)` to choose the amplitude,
and `Synthesizer::synthesize_watermarked(.., key, payload)` (the `real` feature)
which runs the normal synth and stamps the output. The plain `synthesize` is
unchanged (watermarking is additive/opt-in).

## How it works

- A SplitMix64 PRNG seeded by `key` derives a deterministic `±1` chip sequence of
  period `BLOCK = 1024` samples.
- Each of the 16 payload bits owns the residues `j ≡ b (mod 16)` within the block;
  its value sets the sign. Sample `p` is perturbed by
  `amp · sign(bit) · chip[p mod BLOCK]`, so **every sample carries exactly one
  chip**: `max|Δ| = rms(Δ) = amp`.
- Detection folds the signal over the `BLOCK` period (matched filter → processing
  gain ≈ `√(N/16)`), searches the `BLOCK` chip-phase offsets (so an integer-sample
  crop re-aligns), and reports a **z-score**: the correlation in units of the host
  RMS. The z-score is **scale-invariant**, hence gain-robust. Payload bits are the
  signs of the per-bit correlations.

## Imperceptibility

- Default amplitude `DEFAULT_AMP = 4e-3` (≈ **−48 dBFS**). Per-sample perturbation
  is exactly `±4e-3`: `max|Δ| = rms(Δ) = 4e-3`.
- That is ~14–15 dB below the (deliberately quiet) synthetic test host and roughly
  **28–37 dB below** a typical-level speech output (RMS ~0.1–0.3) — below the
  perceptual threshold for speech while staying well above the 16-bit PCM floor.

## Measured behaviour (from `tests/watermark.rs`, 2 s synthetic host, RMS ≈ 0.022)

Detection confidence is an RMS z-score; `present` fires at `confidence ≥ 3.0`.

| Case | present | confidence | payload |
|------|---------|-----------:|---------|
| Clean (un-watermarked) | no  | ~0.09 | — |
| Embed → detect (clean round-trip) | yes | ~9.9 | exact |
| + added Gaussian noise (~8 dB SNR) | yes | ~9.3 | exact |
| + gain ×0.5 | yes | ~9.9 | exact |
| + front crop of 137 samples | yes | ~9.9 | exact |
| **Wrong key** | no  | ~1.9 | — |

Wide separation between present (~9–10) and absent/wrong-key (~0.1–1.9), with the
threshold (3.0) ~11σ above the per-bit null mean → negligible false-positive rate.

## Robustness boundary (honest)

**Survives** (low-amplitude, redundant, correlation-recovered; detection is
block-folded and amplitude/gain-normalized):

- lossless / high-bitrate re-encoding (16-bit PCM round-trip, high-rate codecs)
- a small linear gain change (z-score is scale-invariant)
- light additive noise / dither down to a modest SNR
- cropping / trimming by an integer number of samples (chip-phase sync search)

**Does NOT survive** (not claimed):

- aggressive lossy compression (low-bitrate MP3/Opus reshapes the very low-level
  content the mark lives in)
- time-stretch, pitch-shift, or resampling to a different rate (these warp chip
  timing; only integer-sample crops are handled)
- deliberate adversarial removal (denoise-then-re-add, spectral subtraction, or an
  attacker who knows the scheme)

Those threat models require a **learned, perceptually-masked** watermark such as
[AudioSeal](https://github.com/facebookresearch/audioseal) or WavMark. This module
is the training-free baseline: real, detectable, and robust to *light* editing —
stated without overclaiming.
