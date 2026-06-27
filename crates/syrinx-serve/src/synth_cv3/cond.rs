//! CV3 frontend conditioning: tokenize text (with the `<|endofprompt|>` boundary) and
//! derive the prompt-side `embedding` / `prompt_token` / `prompt_feat` from the
//! reference voice (plus the speaker-embedding evaluation helper), with the small
//! fbank/grid tensor helpers.

use candle_core::{Device, Tensor};

use syrinx_frontend::feat::{kaldi_fbank, prompt_mel};

use super::*;

impl Cv3Synthesizer {
    /// Run the CV3 frontend half: tokenize text (with the `<|endofprompt|>` boundary on
    /// the prompt-text segment) and derive the prompt-side conditioning (`embedding`,
    /// `prompt_token`, `prompt_feat`) from the reference waveforms, applying the CosyVoice
    /// `%2` token/feat alignment.
    ///
    /// `ref_wav_16k` is the 16 kHz mono reference (fbank + speech-token input);
    /// `ref_wav_24k` is the same clip at 24 kHz (prompt-mel input). Resampling is the
    /// caller's job (only the `feat` math is under parity test), so a deterministic test
    /// can feed the exact reference-resampled waveforms.
    pub fn prompt_cond(
        &mut self,
        tts_text: &str,
        prompt_text: &str,
        ref_wav_16k: &[f32],
        ref_wav_24k: &[f32],
    ) -> Result<PromptCond, SynthError> {
        // --- text tokens: prompt_text(+<|endofprompt|>) ++ tts_text. ---
        // Text-normalization (`tn` feature, matching CV2 — identity when off). The tts
        // text is always normalized; a plain prompt transcript is normalized then gets the
        // endofprompt boundary marker; a pre-formatted instruct prompt (already carrying the
        // marker) is left byte-untouched so tn can't mangle the marker/instruction.
        let tts_text = tn_normalize(tts_text);
        let tts_text = tts_text.as_ref();
        let prompt_text = if prompt_text.contains(ENDOFPROMPT) {
            std::borrow::Cow::Borrowed(prompt_text)
        } else {
            std::borrow::Cow::Owned(format!("{}{ENDOFPROMPT}", tn_normalize(prompt_text)))
        };
        let prompt_text_ids = self.tokenizer.encode(prompt_text.as_ref())?;
        let tts_text_ids = self.tokenizer.encode(tts_text)?;
        let prompt_text_len = prompt_text_ids.len();
        let mut text_token = prompt_text_ids;
        text_token.extend_from_slice(&tts_text_ids);

        // --- speaker x-vector: kaldi fbank -> per-time mean subtraction -> CAM++. ---
        let fbank_grid = kaldi_fbank(ref_wav_16k, SR_16K, FBANK_MELS); // [T][80]
        let fbank = grid_to_tensor(&fbank_grid, &self.dev)?; // [T, 80]
        let fbank = subtract_time_mean(&fbank)?; // feat - feat.mean(dim=0)
        let fbank = fbank.unsqueeze(0)?; // [1, T, 80]
        let spk_embedding = self.speaker.forward(&fbank)?; // [1, 192]

        // --- prompt speech tokens via the v3 ONNX tokenizer (16 kHz). ---
        let prompt_token_i32 = self.speech_tokenizer.tokens_from_wav(ref_wav_16k)?;

        // --- prompt mel (24 kHz): feat returns [80, T'] mel-major; flow wants
        //     [1, T', 80] frame-major. ---
        let mel_grid = prompt_mel(
            ref_wav_24k,
            MEL_N_FFT,
            MEL_NUM_MELS,
            MEL_SR,
            MEL_HOP,
            MEL_WIN,
            MEL_FMIN,
            MEL_FMAX,
        ); // [80][T']
        let prompt_feat = mel_major_to_frame_major(&mel_grid, &self.dev)?; // [1, T', 80]

        // --- %2 alignment: token_len = min(T'/2, |prompt_token|); truncate. ---
        let n_feat_frames = prompt_feat.dim(1)?;
        let token_len = (n_feat_frames / 2).min(prompt_token_i32.len());
        let prompt_feat = prompt_feat.narrow(1, 0, 2 * token_len)?.contiguous()?;
        let prompt_token = i32_ids_to_tensor(&prompt_token_i32[..token_len], &self.dev)?; // [1, token_len]

        Ok(PromptCond {
            text_token,
            prompt_text_len,
            spk_embedding,
            prompt_token,
            prompt_feat,
        })
    }

    /// CAM++ speaker x-vector `[1, 192]` for a 16 kHz mono waveform (kaldi-fbank ->
    /// per-time mean-subtraction -> CAM++). Exposed for evaluation (e.g. SIM-o).
    pub fn speaker_embedding(&self, audio_16k: &[f32]) -> Result<Tensor, SynthError> {
        let fbank_grid = kaldi_fbank(audio_16k, SR_16K, FBANK_MELS);
        let fbank = grid_to_tensor(&fbank_grid, &self.dev)?;
        let fbank = subtract_time_mean(&fbank)?;
        let fbank = fbank.unsqueeze(0)?;
        Ok(self.speaker.forward(&fbank)?)
    }
}

// ---- fbank / grid helpers ----------------------------------------------------

/// `[T][D]` row-major grid -> `[T, D]` f32 tensor.
fn grid_to_tensor(grid: &[Vec<f32>], dev: &Device) -> candle_core::Result<Tensor> {
    let t = grid.len();
    let d = if t == 0 { 0 } else { grid[0].len() };
    let mut flat = Vec::with_capacity(t * d);
    for row in grid {
        flat.extend_from_slice(row);
    }
    Tensor::from_vec(flat, (t, d), dev)
}

/// Subtract the per-column (over time/rows) mean: `x - x.mean(dim=0, keepdim=True)`.
fn subtract_time_mean(x: &Tensor) -> candle_core::Result<Tensor> {
    let mean = x.mean_keepdim(0)?; // [1, D]
    x.broadcast_sub(&mean)
}

/// `[80][T']` mel-major grid -> `[1, T', 80]` frame-major f32 tensor.
fn mel_major_to_frame_major(grid: &[Vec<f32>], dev: &Device) -> candle_core::Result<Tensor> {
    let n_mels = grid.len();
    let t = if n_mels == 0 { 0 } else { grid[0].len() };
    let mut flat = vec![0f32; t * n_mels];
    for (m, row) in grid.iter().enumerate() {
        for (frame, &v) in row.iter().enumerate() {
            flat[frame * n_mels + m] = v;
        }
    }
    let tensor = Tensor::from_vec(flat, (t, n_mels), dev)?;
    tensor.unsqueeze(0)
}

/// i32 prompt-token ids -> `[1, n]` i64 tensor.
fn i32_ids_to_tensor(ids: &[i32], dev: &Device) -> candle_core::Result<Tensor> {
    let v: Vec<i64> = ids.iter().map(|&i| i as i64).collect();
    ids_i64_to_tensor(&v, dev)
}
