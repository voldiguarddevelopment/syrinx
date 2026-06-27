//! Chunked-causal streaming: the [`StreamCfg`] knobs, the additive chunk mask, the
//! streaming flow forward ([`Flow::forward_zero_shot_streaming`]), and the
//! streaming/non-streaming `token2wav` glue + HiFT cross-fade. Moved verbatim from
//! `real.rs`.

use super::*;

/// Chunked-causal streaming configuration for [`Flow::forward_zero_shot_streaming`].
///
/// CosyVoice2 streams the **same** flow weights under a chunk attention mask (it does
/// *not* swap in a different architecture): a frame in chunk `c` may attend only to
/// chunks `[c - num_left, c]`, never the future, so finalized frames are stable as
/// later chunks arrive. These are the knobs that define those chunk boundaries; defaults
/// come from [`StreamCfg::cosyvoice2`].
#[derive(Clone, Copy, Debug)]
pub struct StreamCfg {
    /// Conformer-encoder **first-stage** chunk size, in speech tokens (the encoder runs
    /// pre-upsample at token rate). The up-stage runs at `ENC_UPSAMPLE`× this length and
    /// uses `enc_chunk * ENC_UPSAMPLE` as its chunk (see [`Flow::encoder_masked`]).
    pub enc_chunk: usize,
    /// CFM-estimator chunk size, in mel frames (the estimator runs at mel rate, which is
    /// `token_mel_ratio`× the token rate).
    pub est_chunk: usize,
    /// Number of left chunks a frame may additionally attend to (besides its own chunk).
    /// `usize::MAX` ⇒ all left chunks (CosyVoice2's `num_decoding_left_chunks = -1`).
    pub num_left: usize,
}

impl StreamCfg {
    /// The CosyVoice2-0.5B defaults, read straight from `CosyVoice2-0.5B/cosyvoice2.yaml`:
    ///
    /// ```text
    /// token_frame_rate: 25      token_mel_ratio: 2
    /// chunk_size: 25            # streaming chunk, IN TOKENS
    /// num_decoding_left_chunks: -1   # <0 ⇒ all left chunks
    ///   encoder (UpsampleConformerEncoder).static_chunk_size: chunk_size              = 25
    ///   estimator (CausalConditionalDecoder).static_chunk_size: chunk_size*token_mel  = 50
    /// ```
    ///
    /// Derivation of the three numbers:
    /// * `enc_chunk = 25` — the encoder's first 6 conformer layers run **pre-upsample**
    ///   at token rate, where `static_chunk_size = chunk_size = 25` tokens. The encoder
    ///   then 2× upsamples (`up_layer.stride = ENC_UPSAMPLE`) and its 4 up-stage layers
    ///   mask with `static_chunk_size * stride = 25*2 = 50` frames — derived here as
    ///   `enc_chunk * ENC_UPSAMPLE`, not stored separately.
    /// * `est_chunk = 50` — the estimator runs on the **mel** sequence, whose length is
    ///   `token_mel_ratio = 2`× the post-upsample token length; CV2 sets its chunk to
    ///   `chunk_size * token_mel_ratio = 25*2 = 50` mel frames. (So the estimator chunk
    ///   and the encoder up-stage chunk coincide at 50 — both at mel/post-upsample rate.)
    /// * `num_left = usize::MAX` — `num_decoding_left_chunks = -1` means "all left
    ///   chunks". (At runtime CV2's onnx-friendly `subsequent_chunk_mask` ignores the
    ///   left limit entirely, so all-left is also what actually executes there.)
    ///
    /// ⚠️ The 2× upsample boundary is the most likely on-box tuning target: this assumes
    /// the maintainer's `encoder()` upsamples exactly once by `ENC_UPSAMPLE` between the
    /// first-stage and up-stage blocks (it does), and that the mel length the estimator
    /// sees equals the post-upsample token length (it does: `mu_t` is `2*(Tp+Tg)`). If a
    /// future flow variant changes either ratio, re-derive these against the yaml.
    pub fn cosyvoice2() -> Self {
        Self { enc_chunk: 25, est_chunk: 50, num_left: usize::MAX }
    }
}

impl Flow {
    /// Chunked-causal **streaming** counterpart of [`Self::forward_zero_shot`].
    ///
    /// Identical conditioning, weights, noise, and ODE as `forward_zero_shot`, but the
    /// conformer encoder and the CFM estimator run under chunk-causal attention masks
    /// built from `cfg`: a finalized frame never attends to the future, so re-running on
    /// a grown token prefix leaves already-finalized frames bit-stable — the property
    /// that makes streaming sample-faithful (see `docs/STREAMING.md`). With
    /// `cfg.enc_chunk`/`cfg.est_chunk` set huge this reduces to the unmasked path; with
    /// [`StreamCfg::cosyvoice2`] it matches CosyVoice2's `inference(streaming=True)`.
    ///
    /// Returns the generated mel `[1, 80, 2*Tg]`, same shape/semantics as the non-stream.
    pub fn forward_zero_shot_streaming(
        &self,
        prompt_token: &Tensor,
        token: &Tensor,
        prompt_feat: &Tensor,
        embedding: &Tensor,
        z: &Tensor,
        n_timesteps: usize,
        cfg: StreamCfg,
    ) -> Result<Tensor> {
        let spk = self.spk_proj(embedding)?; // [1, 80]

        let tok_cat = Tensor::cat(&[prompt_token, token], 1)?; // [1, Tp+Tg]
        let enc_t = tok_cat.dim(1)?; // T = encoder first-stage (token-rate) length

        // Encoder masks: the first 6 conformer layers run at token length T with chunk
        // `enc_chunk`; after the 2× upsample the 4 up-stage layers run at length
        // `T*ENC_UPSAMPLE` with chunk `enc_chunk*ENC_UPSAMPLE`. Built at their own lengths.
        let m_enc1 = add_optional_chunk_mask(enc_t, cfg.enc_chunk, cfg.num_left, &self.dev)?;
        let m_enc2 = add_optional_chunk_mask(
            enc_t * ENC_UPSAMPLE,
            cfg.enc_chunk * ENC_UPSAMPLE,
            cfg.num_left,
            &self.dev,
        )?;

        let emb = self.input_embedding(&tok_cat)?; // [1, T, 512]
        let h = self.encoder_masked(&emb, Some(&m_enc1), Some(&m_enc2))?; // [1, 2T, 512]
        let mu = self.linear(&h, "encoder_proj.weight", Some("encoder_proj.bias"))?; // [1, 2T, 80]
        let mu_t = mu.transpose(1, 2)?.contiguous()?; // [1, 80, 2T]

        let total = mu_t.dim(2)?; // 2*(Tp+Tg) == mel length
        let mel_len1 = prompt_feat.dim(1)?; // == 2*Tp
        let mel_len2 = total - mel_len1; // == 2*Tg

        let prompt_ct = prompt_feat.transpose(1, 2)?.contiguous()?; // [1, 80, mel_len1]
        let cond_tail = Tensor::zeros((1, MEL, mel_len2), DType::F32, &self.dev)?;
        let cond = Tensor::cat(&[&prompt_ct, &cond_tail], 2)?; // [1, 80, total]

        // Estimator mask: built at the mel length `total` with chunk `est_chunk`.
        let m_est = add_optional_chunk_mask(total, cfg.est_chunk, cfg.num_left, &self.dev)?;

        let mel_full =
            self.cfm_solve_with_cond_masked(&mu_t, &spk, &cond, z, n_timesteps, Some(&m_est))?;
        // drop the prompt-mel prefix; keep only the generated mel.
        mel_full.narrow(2, mel_len1, mel_len2)
    }
}

/// Deterministic zero-shot **token2wav** glue: speech tokens -> mel -> audio.
///
/// Reproduces `CosyVoice2Model.token2wav` (non-streaming, single utterance) for the
/// zero-shot path: the prompt-conditioned flow ([`Flow::forward_zero_shot`]) yields
/// the generated mel, which the HiFT vocoder ([`syrinx_vocoder::cv2::HiftVocoder::decode`])
/// turns into a 24 kHz waveform. Both stochastic inputs are pinned and fed in:
/// the CFM noise `z` (the flow's fixed `rand_noise` slice) and the HiFT source STFT
/// `s_stft` (the SineGen source has a random initial phase in the real model, so the
/// reference captures it). Returns the waveform `[1, L]`.
#[allow(clippy::too_many_arguments)]
pub fn token2wav(
    flow: &Flow,
    vocoder: &syrinx_vocoder::cv2::HiftVocoder,
    prompt_token: &Tensor,
    token: &Tensor,
    prompt_feat: &Tensor,
    embedding: &Tensor,
    z: &Tensor,
    s_stft: &Tensor,
    n_timesteps: usize,
) -> Result<Tensor> {
    let mel = flow.forward_zero_shot(prompt_token, token, prompt_feat, embedding, z, n_timesteps)?;
    vocoder.decode(&mel, s_stft)
}

// ============================ streaming token2wav ============================

/// Flow upsamples speech tokens to mel by this factor (CosyVoice2 `token_mel_ratio`).
pub const TOKEN_MEL_RATIO: usize = 2;
/// HiFT samples emitted per mel frame: `prod(upsample_rates) * istft_hop` = 8*5*3*4.
pub const SOURCE_PER_MEL: usize = 480;
/// Mel frames overlapped from one HiFT chunk into the next (CosyVoice2 `mel_cache_len`).
pub const MEL_CACHE_LEN: usize = 20;
/// Waveform samples cross-faded at a chunk boundary (`source_cache_len`): the
/// trailing samples of one chunk blend with the leading samples of the next.
pub const SOURCE_CACHE_LEN: usize = MEL_CACHE_LEN * SOURCE_PER_MEL; // 9600

/// The **phase-continuous** streaming source builder the driver calls per chunk.
///
/// Args: `(mel_in [1,80,L], phase_in, overlap_samples, prev_source_tail)`:
///   * `mel_in` — the (overlap-extended) chunk mel to decode,
///   * `phase_in` — the global F0-excitation phase (in cycles) at the start of this
///     chunk's *new* (post-overlap) region, carried from the previous chunk so the
///     excitation is globally continuous (not reset per chunk — that reset is what
///     decorrelated streaming from the non-streaming reference),
///   * `overlap_samples` — leading source samples that belong to the carried-over
///     overlap region (== the previous chunk's `mel_cache` frames × `SOURCE_PER_MEL`),
///   * `prev_source_tail` — `Some([1, overlap_samples])` of the previous chunk's
///     trailing source waveform, used to **overwrite** the overlap region so it is
///     sample-identical to what the vocoder already emitted (CosyVoice2 `cache_source`),
///     or `None` for the first chunk.
///
/// Returns `(s_stft [1,18,T_src], source_wave [1, L·SOURCE_PER_MEL], phase_out)`:
/// the STFT for the vocoder, the full source waveform (so the driver can cache its
/// tail as the next chunk's overlap), and the global phase at this chunk's end
/// (the next chunk's `phase_in`).
pub type StreamSourceFn<'a> =
    dyn Fn(&Tensor, f64, usize, Option<&Tensor>) -> Result<(Tensor, Tensor, f64)> + 'a;

/// One emitted streaming audio chunk: the waveform `[1, L]` plus the token offset
/// (in finalized speech tokens) it ends at, for the caller's bookkeeping.
pub struct AudioChunk {
    /// The emitted (already cross-faded, cache-trimmed) waveform `[1, L]`.
    pub wav: Tensor,
    /// Number of finalized speech tokens consumed up to and including this chunk.
    pub token_offset: usize,
}

/// Per-utterance streaming state: the HiFT mel/source/speech overlap caches that
/// must persist across chunks. Mirrors `CosyVoice2Model.hift_cache_dict[uuid]`.
struct HiftCache {
    /// Last `MEL_CACHE_LEN` mel frames of the previous chunk's (extended) mel.
    mel: Option<Tensor>, // [1,80,MEL_CACHE_LEN]
    /// The trailing `SOURCE_CACHE_LEN` waveform samples held back from the previous
    /// chunk, used as the fade-out leg of the next boundary cross-fade.
    speech_tail: Option<Tensor>, // [1, SOURCE_CACHE_LEN]
    /// The trailing `SOURCE_CACHE_LEN` **source** (excitation) samples of the previous
    /// chunk. Overwrites the next chunk's overlap source so the excitation is
    /// sample-coherent across the boundary (CosyVoice2 `cache_source`).
    source_tail: Option<Tensor>, // [1, SOURCE_CACHE_LEN]
}

/// Streaming `token2wav`: drive the flow + HiFT incrementally so audio is emitted
/// chunk-by-chunk with low first-byte latency, replicating
/// `CosyVoice2Model.token2wav`'s mel/source overlap + hamming cross-fade.
///
/// `token_hop` is the number of *finalized* speech tokens released per chunk; a
/// chunk for tokens `[off .. off+hop)` only needs tokens up to `off+hop+PRE_LOOKAHEAD`
/// to be present (the flow's right lookahead), so the first chunk is produced after
/// `token_hop + PRE_LOOKAHEAD` tokens instead of the whole utterance.
///
/// ## Flow handling (chunked-causal — faithful)
/// CosyVoice2 streams the flow under a chunked-causal attention mask. This path uses
/// [`Flow::forward_zero_shot_streaming`] (same weights + a chunk mask, [`StreamCfg::cosyvoice2`]),
/// re-running over the *grown* token prefix and slicing the newly finalized mel region
/// `[2*off .. 2*(off+hop)]`. Because the mask blocks every finalized frame from attending
/// to the future, those frames are **bit-stable** as the prefix grows (verified by
/// `tests/real_flow_stream_consistency.rs`: 0.0 diff vs 0.53 for the old non-causal re-run),
/// so the stream is self-consistent — no right-context leak, no leading-chunk phase poison.
/// The HiFT caching + overlap + hamming fade below is replicated *exactly*. (Re-running the
/// masked flow per chunk is O(n²); a left-context KV cache like CV2's `flow_cache` is the
/// efficiency follow-up — correctness here does not depend on it.)
///
/// `prompt_token` / `prompt_feat` / `embedding` / `z_full` are the same zero-shot
/// conditioning as non-streaming (`z_full` must cover the full final flow length,
/// `2*(|prompt|+|token|)`). `source_fn` builds the HiFT source STFT for a mel chunk.
#[allow(clippy::too_many_arguments)]
pub fn token2wav_streaming(
    flow: &Flow,
    vocoder: &syrinx_vocoder::cv2::HiftVocoder,
    prompt_token: &Tensor,
    token: &Tensor,
    prompt_feat: &Tensor,
    embedding: &Tensor,
    z_full: &Tensor,
    source_fn: &StreamSourceFn<'_>,
    token_hop: usize,
    n_timesteps: usize,
    mut on_chunk: impl FnMut(AudioChunk) -> Result<()>,
) -> Result<()> {
    assert!(token_hop >= 1, "token_hop must be >= 1");
    let dev = z_full.device();
    let n_tokens = token.dim(1)?;
    let prompt_len = prompt_token.dim(1)?;
    let mut cache = HiftCache { mel: None, speech_tail: None, source_tail: None };

    // Global F0-excitation phase (in cycles) carried across chunks so the streaming
    // source is one continuous sinusoid, exactly like the non-streaming source —
    // rather than restarting from phase 0 each chunk (which decorrelates the audio).
    let mut phase = 0f64;
    // offset = number of finalized tokens already emitted as mel.
    let mut offset = 0usize;
    while offset < n_tokens {
        // How many tokens we want to finalize this step.
        let want_end = (offset + token_hop).min(n_tokens);
        let finalize = want_end == n_tokens;
        // Tokens that must be present for the flow to finalize `want_end`: the
        // finalized region plus PRE_LOOKAHEAD of right context (clamped to the end).
        let avail_end = (want_end + PRE_LOOKAHEAD).min(n_tokens);
        let tok_slice = token.narrow(1, 0, avail_end)?; // grown prefix [1, avail_end]

        // z slice for this (prompt + avail) flow length.
        let flow_len = TOKEN_MEL_RATIO * (prompt_len + avail_end);
        let z = z_full.narrow(2, 0, flow_len)?.contiguous()?;
        // prompt_feat is always the full prompt mel (its length matches 2*prompt_len).
        // Chunked-causal streaming flow: under the attention mask a finalized frame
        // never attends to the future, so the [2*offset .. 2*want_end] region is
        // bit-stable across the growing prefix (proven by real_flow_stream_consistency:
        // 0.0 diff vs 0.53 non-causal). That kills the leading-chunk right-context leak
        // that previously decorrelated the stream.
        let mel_full = flow.forward_zero_shot_streaming(
            prompt_token, &tok_slice, prompt_feat, embedding, &z, n_timesteps,
            StreamCfg::cosyvoice2(),
        )?; // [1, 80, 2*avail_end]

        // newly-finalized mel region: [2*offset .. 2*want_end].
        let mel_start = TOKEN_MEL_RATIO * offset;
        let mel_count = TOKEN_MEL_RATIO * (want_end - offset);
        let mel_new = mel_full.narrow(2, mel_start, mel_count)?.contiguous()?;

        // --- HiFT chunk with mel overlap, source, and waveform cross-fade. ---
        let overlap_frames = match &cache.mel {
            Some(prev) => prev.dim(2)?,
            None => 0,
        };
        let mel_in = match &cache.mel {
            Some(prev) => Tensor::cat(&[prev, &mel_new], 2)?, // prepend overlap
            None => mel_new.clone(),
        };
        // Build the phase-continuous source: the overlap region is overwritten by the
        // previous chunk's trailing source; the new region continues the global phase.
        let overlap_samples = overlap_frames * SOURCE_PER_MEL;
        let (src, source_wave, phase_out) =
            source_fn(&mel_in, phase, overlap_samples, cache.source_tail.as_ref())?;
        phase = phase_out;
        let speech = vocoder.decode(&mel_in, &src)?; // [1, L]
        let total_len = speech.dim(1)?;

        // Hold back the trailing SOURCE_CACHE_LEN samples (unless this is the last
        // chunk), and trim them from what we emit now.
        let (mut emit, new_tail) = if !finalize && total_len > SOURCE_CACHE_LEN {
            let keep = total_len - SOURCE_CACHE_LEN;
            let head = speech.narrow(1, 0, keep)?.contiguous()?;
            let tail = speech.narrow(1, keep, SOURCE_CACHE_LEN)?.contiguous()?;
            (head, Some(tail))
        } else {
            (speech, None)
        };

        // Cross-fade the leading SOURCE_CACHE_LEN samples of this emit with the held
        // tail of the previous chunk (hamming(2*SOURCE_CACHE_LEN)).
        if let Some(prev_tail) = &cache.speech_tail {
            emit = hamming_crossfade(&emit, prev_tail, dev)?;
        }

        on_chunk(AudioChunk { wav: emit, token_offset: want_end })?;

        // Update caches for the next chunk: keep the last MEL_CACHE_LEN mel frames
        // and the held waveform tail.
        let mlen = mel_in.dim(2)?;
        let keep_mel = MEL_CACHE_LEN.min(mlen);
        cache.mel = Some(mel_in.narrow(2, mlen - keep_mel, keep_mel)?.contiguous()?);
        cache.speech_tail = new_tail;
        // Cache the trailing source samples (one mel_cache worth) for the next chunk's
        // overlap overwrite — these carry the exact phase the vocoder just consumed.
        let swlen = source_wave.dim(1)?;
        let keep_src = SOURCE_CACHE_LEN.min(swlen);
        cache.source_tail = Some(source_wave.narrow(1, swlen - keep_src, keep_src)?.contiguous()?);

        offset = want_end;
    }
    Ok(())
}

/// Cross-fade `emit`'s leading `SOURCE_CACHE_LEN` samples with `prev_tail` using a
/// hamming window of length `2*SOURCE_CACHE_LEN`, replicating CosyVoice2's
/// `fade_in_out`: `emit[:n] = emit[:n]*w_in + prev_tail*w_out`, where `w_in` is the
/// rising (first) half and `w_out` the falling (second) half of the window.
fn hamming_crossfade(emit: &Tensor, prev_tail: &Tensor, dev: &Device) -> Result<Tensor> {
    let n = SOURCE_CACHE_LEN;
    let emit_len = emit.dim(1)?;
    let tail_len = prev_tail.dim(1)?;
    let overlap = n.min(emit_len).min(tail_len);
    if overlap == 0 {
        return Ok(emit.clone());
    }
    // hamming(2n): w[m] = 0.54 - 0.46 cos(2π m / (2n - 1)), m in 0..2n.
    let win_len = 2 * n;
    let mut w_in = vec![0f32; overlap]; // rising half, indices 0..overlap
    let mut w_out = vec![0f32; overlap]; // falling half, indices (win_len-overlap)..win_len
    for m in 0..overlap {
        let hm = |idx: usize| -> f32 {
            0.54 - 0.46 * (2.0 * std::f32::consts::PI * idx as f32 / (win_len as f32 - 1.0)).cos()
        };
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
        Tensor::cat(&[&blended, &rest], 1)
    } else {
        Ok(blended)
    }
}

/// Build CosyVoice2's additive chunked-causal attention mask `[1, 1, t, t]`.
///
/// Position `i` lives in chunk `c = i / chunk_size`; it may attend to position `j`
/// (chunk `cj = j / chunk_size`) iff `c - num_left <= cj <= c` — i.e. its own chunk and
/// up to `num_left` chunks of left context, **never the future**. Allowed entries are
/// `0.0`; disallowed are `f32::NEG_INFINITY`, so adding this to pre-softmax scores zeros
/// out the forbidden positions. Because every row always includes its own chunk (`cj = c`,
/// which contains `i` itself), no row is fully masked, so the softmax never sees an
/// all-`-inf` row (no NaN).
///
/// `num_left == usize::MAX` (saturating) ⇒ all left chunks (CosyVoice2's
/// `num_decoding_left_chunks = -1`). A `chunk_size` larger than `t` ⇒ a single chunk ⇒
/// an all-zeros mask (no masking) — which is why the non-streaming path passes `None`
/// rather than a degenerate mask. `chunk_size == 0` is treated as no masking too.
fn add_optional_chunk_mask(t: usize, chunk_size: usize, num_left: usize, dev: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; t * t];
    if chunk_size > 0 {
        for i in 0..t {
            let ci = i / chunk_size;
            let start_chunk = ci.saturating_sub(num_left); // num_left==MAX ⇒ 0 (all left)
            let row = i * t;
            for j in 0..t {
                let cj = j / chunk_size;
                if cj < start_chunk || cj > ci {
                    data[row + j] = f32::NEG_INFINITY;
                }
            }
        }
    }
    Tensor::from_vec(data, (1, 1, t, t), dev)
}
