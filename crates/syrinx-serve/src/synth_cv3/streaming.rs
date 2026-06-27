// ============================================================================
// Chunked-causal STREAMING synthesis (additive — low time-to-first-byte path).
//
// Mirrors CV2's `crate::synth::Synthesizer::synthesize_streaming`, swapping the CV3
// parts in: the causal DiT flow chunk-mask (`Cv3Flow::forward_zero_shot_streaming`)
// replaces the non-causal batch flow, and the CV3 causal HiFT (`Cv3Hift::decode`, which
// STFTs the source waveform internally) replaces CV2's `s_stft`-taking HiFT. The HiFT
// assembly (mel overlap cache + phase-continuous source + hamming boundary cross-fade)
// is the CV2 `token2wav_streaming` recipe. The batch `synthesize` path is untouched.
//
// ★ Correctness note: the actual fix that makes streaming faithful is the causal DiT
//   mask — a finalized mel frame never attends to the future, so it is bit-stable as the
//   token prefix grows (proven by `tests/real_cv3_flow_stream_consistency.rs`). The
//   cross-faded *audio* is, by CV2/CV3 design, NOT sample-identical to the batch render
//   (the boundary fade changes a chunk of samples); intelligibility (Whisper CER on the
//   streamed audio) is the human/on-box quality signal.
// ============================================================================

use candle_core::{Device, Tensor};

use syrinx_acoustic::cv3::Cv3StreamCfg;

use super::*;

/// Speech-token right-lookahead the CV3 flow front-end needs (the `pre_lookahead_len` of
/// `CausalMaskedDiffWithDiT`): a chunk finalizing tokens `[off .. off+hop)` needs tokens up
/// to `off+hop+PRE_LOOKAHEAD` present (mirrors CV3 `model.py`'s
/// `token_offset + this_token_hop_len + self.flow.pre_lookahead_len`).
const STREAM_PRE_LOOKAHEAD: usize = 3;
/// token -> mel ratio (CV3 `token_mel_ratio`).
const STREAM_TOKEN_MEL_RATIO: usize = 2;
/// Mel-frame overlap carried across chunks for the HiFT boundary (CV2's `mel_cache_len`).
const STREAM_MEL_CACHE: usize = 20;
/// Trailing source/speech samples held back + cross-faded at each boundary
/// (`mel_cache * f0_upsample`; CV2's `source_cache_len`).
const STREAM_SOURCE_CACHE: usize = STREAM_MEL_CACHE * F0_UPSAMPLE; // 9600

/// Timing + shape summary of a streaming run, so callers can report **time-to-first-byte**
/// (the headline latency win) and total throughput.
#[derive(Debug, Clone, Copy)]
pub struct Cv3StreamStats {
    /// Wall-clock from the call to the first emitted chunk — the TTFB.
    pub ttfb: std::time::Duration,
    /// Wall-clock for the whole streamed utterance.
    pub total: std::time::Duration,
    /// Number of audio chunks emitted.
    pub n_chunks: usize,
    /// Total 24 kHz samples emitted across all chunks.
    pub n_samples: usize,
}

impl Cv3Synthesizer {
    /// **Streaming** CV3 synthesis: same `tts_text`-in-reference-voice flow + vocoder as
    /// [`Cv3Synthesizer::synthesize`], but audio is emitted **incrementally** chunk by
    /// chunk (low time-to-first-byte) via `on_chunk`, instead of one final `Vec`.
    ///
    /// `chunk_size` is the number of *finalized* speech tokens released per chunk; a chunk
    /// for tokens `[off .. off+chunk_size)` only needs tokens up to
    /// `off+chunk_size+PRE_LOOKAHEAD` present (the flow's right lookahead), so the first
    /// chunk lands after `chunk_size + PRE_LOOKAHEAD` tokens instead of the whole
    /// utterance. Each emitted chunk is a flat `Vec<f32>` of 24 kHz samples in order;
    /// concatenating them yields the full streamed waveform.
    ///
    /// The flow runs under the chunk-causal DiT mask
    /// ([`Cv3Flow::forward_zero_shot_streaming`], [`Cv3StreamCfg::cosyvoice3`]) so finalized
    /// frames are stable as the prefix grows. The HiFT runs per chunk over a mel-overlap
    /// cache with a **phase-continuous** source (the F0 excitation phase is carried across
    /// chunks, and each chunk's overlap source is overwritten by the previous chunk's
    /// trailing source) plus a hamming cross-fade at every boundary — the CV2 streaming
    /// recipe. `inputs.z`, the LM seed/cap, and `inputs.pinned_speech_token` are honoured
    /// exactly as in `synthesize`; a fresh seeded standard-normal `z` is used when `z` is
    /// not pinned. Returns [`Cv3StreamStats`] (TTFB + totals).
    pub fn synthesize_streaming(
        &mut self,
        tts_text: &str,
        prompt_text: &str,
        ref_wav_16k: &[f32],
        ref_wav_24k: &[f32],
        inputs: &Cv3SynthInputs,
        chunk_size: usize,
        mut on_chunk: impl FnMut(Vec<f32>) -> Result<(), SynthError>,
    ) -> Result<Cv3StreamStats, SynthError> {
        assert!(chunk_size >= 1, "chunk_size must be >= 1");
        let start = std::time::Instant::now();
        let cond = self.prompt_cond(tts_text, prompt_text, ref_wav_16k, ref_wav_24k)?;

        let speech_token = match &inputs.pinned_speech_token {
            Some(ids) => ids_i64_to_tensor(ids, &self.dev)?,
            None => self.generate_speech_token(&cond, inputs.lm_seed, inputs.max_gen_steps)?,
        };

        // Full-length CFM noise z (covers the final flow length 2*(|prompt|+|token|)):
        // pinned (parity) else a seeded standard-normal (same default as `synthesize`).
        let total_flow = 2 * (cond.prompt_token.dim(1)? + speech_token.dim(1)?);
        let z_full = match inputs.z.as_ref() {
            Some(z) => z.clone(),
            None => self.seeded_normal_z(total_flow, inputs.lm_seed)?,
        };

        let n_tokens = speech_token.dim(1)?;
        let prompt_len = cond.prompt_token.dim(1)?;
        let cfg = Cv3StreamCfg::cosyvoice3();

        // HiFT carry-over caches (mirrors CosyVoice2Model.hift_cache_dict).
        let mut cache_mel: Option<Tensor> = None; // last STREAM_MEL_CACHE mel frames
        let mut cache_speech_tail: Option<Tensor> = None; // held-back fade-out leg [1, S]
        let mut cache_source_tail: Option<Tensor> = None; // trailing source samples [1, S]
        let mut phase = 0f64; // global F0 excitation phase (cycles), carried across chunks

        let mut ttfb: Option<std::time::Duration> = None;
        let mut n_chunks = 0usize;
        let mut n_samples = 0usize;

        let mut offset = 0usize;
        while offset < n_tokens {
            let want_end = (offset + chunk_size).min(n_tokens);
            let finalize = want_end == n_tokens;
            // Tokens that must be present to finalize `want_end`: finalized region plus
            // PRE_LOOKAHEAD of right context (clamped to the utterance end).
            let avail_end = (want_end + STREAM_PRE_LOOKAHEAD).min(n_tokens);
            let tok_slice = speech_token.narrow(1, 0, avail_end)?.contiguous()?;

            // z slice for this (prompt + avail) flow length.
            let flow_len = STREAM_TOKEN_MEL_RATIO * (prompt_len + avail_end);
            let z = z_full.narrow(2, 0, flow_len)?.contiguous()?;

            // Chunked-causal streaming flow over the grown prefix: the DiT mask keeps the
            // [2*offset .. 2*want_end] region bit-stable as the prefix grows.
            let mel_full = self.flow.forward_zero_shot_streaming(
                &cond.prompt_token,
                &tok_slice,
                &cond.prompt_feat,
                &cond.spk_embedding,
                &z,
                N_TIMESTEPS,
                cfg,
            )?; // [1, 80, 2*avail_end]

            // Newly-finalized mel region: [2*offset .. 2*want_end].
            let mel_start = STREAM_TOKEN_MEL_RATIO * offset;
            let mel_count = STREAM_TOKEN_MEL_RATIO * (want_end - offset);
            let mel_new = mel_full.narrow(2, mel_start, mel_count)?.contiguous()?;

            // --- HiFT chunk: mel overlap + phase-continuous source + cross-fade. ---
            let overlap_frames = match &cache_mel {
                Some(prev) => prev.dim(2)?,
                None => 0,
            };
            let mel_in = match &cache_mel {
                Some(prev) => Tensor::cat(&[prev, &mel_new], 2)?,
                None => mel_new.clone(),
            };
            let overlap_samples = overlap_frames * F0_UPSAMPLE;
            let (src_3d, source_flat, phase_out) =
                self.streaming_source(&mel_in, phase, overlap_samples, cache_source_tail.as_ref())?;
            phase = phase_out;

            let speech = self.vocode(&mel_in, &src_3d)?; // [1, L]
            let total_len = speech.dim(1)?;

            // Hold back the trailing SOURCE_CACHE samples (unless final) and trim them now.
            let (mut emit, new_tail) = if !finalize && total_len > STREAM_SOURCE_CACHE {
                let keep = total_len - STREAM_SOURCE_CACHE;
                let head = speech.narrow(1, 0, keep)?.contiguous()?;
                let tail = speech.narrow(1, keep, STREAM_SOURCE_CACHE)?.contiguous()?;
                (head, Some(tail))
            } else {
                (speech, None)
            };

            // Cross-fade the leading samples of this emit with the previous chunk's tail.
            if let Some(prev_tail) = &cache_speech_tail {
                emit = hamming_crossfade(&emit, prev_tail, &self.dev)?;
            }

            let wav: Vec<f32> = emit.flatten_all()?.to_vec1::<f32>()?;
            n_samples += wav.len();
            n_chunks += 1;
            if ttfb.is_none() {
                ttfb = Some(start.elapsed());
            }
            on_chunk(wav)?;

            // Update caches for the next chunk.
            let mlen = mel_in.dim(2)?;
            let keep_mel = STREAM_MEL_CACHE.min(mlen);
            cache_mel = Some(mel_in.narrow(2, mlen - keep_mel, keep_mel)?.contiguous()?);
            cache_speech_tail = new_tail;
            let swlen = source_flat.dim(1)?;
            let keep_src = STREAM_SOURCE_CACHE.min(swlen);
            cache_source_tail =
                Some(source_flat.narrow(1, swlen - keep_src, keep_src)?.contiguous()?);

            offset = want_end;
        }

        Ok(Cv3StreamStats {
            ttfb: ttfb.unwrap_or_default(),
            total: start.elapsed(),
            n_chunks,
            n_samples,
        })
    }

    /// Build the **streaming** CV3 HiFT source for one mel chunk with *global F0-phase
    /// continuity* (the CV3 analogue of CV2's `Synthesizer::streaming_source_phase`, but
    /// emitting the source **waveform** `[1,1,L]` that `Cv3Hift::decode` STFTs internally,
    /// not a pre-STFT'd `s_stft`).
    ///
    /// Same deterministic F0 -> single-harmonic sine core as
    /// [`Cv3Synthesizer::deterministic_source`], but: (1) the instantaneous phase continues
    /// from `phase_in` (one continuous sinusoid across chunks, not reset per chunk), and
    /// (2) the leading `overlap_samples` are overwritten by `prev_tail` (the previous
    /// chunk's trailing source), so the boundary cross-fade stays phase-coherent
    /// (CosyVoice2 `cache_source`). Returns `(source [1,1,n], source_flat [1,n], phase_out)`.
    fn streaming_source(
        &self,
        mel_in: &Tensor,
        phase_in: f64,
        overlap_samples: usize,
        prev_tail: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, f64), SynthError> {
        let f0 = self.vocoder.f0_predict(mel_in)?; // [1, T]
        let f0v: Vec<f32> = f0.flatten_all()?.to_vec1::<f32>()?;
        let n = f0v.len() * F0_UPSAMPLE;

        // New-region excitation: continue the global cumulative phase from phase_in. The
        // overlap region [0, overlap_samples) is overwritten below (so it does not advance
        // the global phase — phase_in already integrated those frames in the prev chunk).
        let mut source = vec![0f32; n];
        let mut acc = phase_in;
        let two_pi = 2.0 * std::f64::consts::PI;
        for s in overlap_samples.min(n)..n {
            let fhz = f0v[s / F0_UPSAMPLE] as f64;
            acc += fhz / MEL_SR as f64;
            source[s] = (SINE_AMP * (two_pi * acc).sin()) as f32;
        }
        let phase_out = acc;

        // Overwrite the overlap with the previous chunk's trailing source (phase-coherent).
        if let Some(t) = prev_tail {
            let prev: Vec<f32> = t.flatten_all()?.to_vec1::<f32>()?;
            let m = overlap_samples.min(prev.len()).min(n);
            let off = prev.len() - m; // align on the tail if lengths differ
            source[..m].copy_from_slice(&prev[off..off + m]);
        }

        let src_3d = Tensor::from_vec(source.clone(), (1, 1, n), &self.dev)?;
        let src_flat = Tensor::from_vec(source, (1, n), &self.dev)?;
        Ok((src_3d, src_flat, phase_out))
    }
}

/// Cross-fade `emit`'s leading `STREAM_SOURCE_CACHE` samples with `prev_tail` using a
/// hamming window of length `2*STREAM_SOURCE_CACHE` — CosyVoice2's `fade_in_out`:
/// `emit[:n] = emit[:n]*w_in + prev_tail*w_out`, `w_in` the rising (first) half and
/// `w_out` the falling (second) half. (Local copy of CV2's `hamming_crossfade`.)
fn hamming_crossfade(emit: &Tensor, prev_tail: &Tensor, dev: &Device) -> Result<Tensor, SynthError> {
    let n = STREAM_SOURCE_CACHE;
    let emit_len = emit.dim(1)?;
    let tail_len = prev_tail.dim(1)?;
    let overlap = n.min(emit_len).min(tail_len);
    if overlap == 0 {
        return Ok(emit.clone());
    }
    let win_len = 2 * n;
    let mut w_in = vec![0f32; overlap];
    let mut w_out = vec![0f32; overlap];
    let hm = |idx: usize| -> f32 {
        0.54 - 0.46 * (2.0 * std::f32::consts::PI * idx as f32 / (win_len as f32 - 1.0)).cos()
    };
    for m in 0..overlap {
        w_in[m] = hm(m);
        w_out[m] = hm(win_len - overlap + m);
    }
    let w_in = Tensor::from_vec(w_in, (1, overlap), dev)?;
    let w_out = Tensor::from_vec(w_out, (1, overlap), dev)?;

    let head = emit.narrow(1, 0, overlap)?;
    let tail_seg = prev_tail.narrow(1, tail_len - overlap, overlap)?;
    let blended = (head.broadcast_mul(&w_in)? + tail_seg.broadcast_mul(&w_out)?)?;
    if emit_len > overlap {
        let rest = emit.narrow(1, overlap, emit_len - overlap)?;
        Ok(Tensor::cat(&[&blended, &rest], 1)?)
    } else {
        Ok(blended)
    }
}
