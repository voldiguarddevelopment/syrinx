//! `openaudio-s1-mini` (0.5B) backend — the Llama-style `DualARTransformer` slow AR,
//! the 4-layer fast AR (per-codebook-axis RoPE), the modded-DAC RVQ codec, and the
//! tiktoken tokenizer + prompt builder, wired into an [`S1Mini`] that implements the
//! shared [`DualArBackend`](crate::common::dualar::DualArBackend) +
//! [`RvqCodec`](crate::common::codec::RvqCodec) contracts.
//!
//! Honors the foundation's rules:
//! * the slow steps return **raw, unmasked** semantic logits over the full slow vocab
//!   (the driver owns the constraint + RAS);
//! * `fast_expand` is `&self` and allocates its tiny per-frame KV cache locally, and
//!   draws every residual via the **driver-owned** [`Sampler`](crate::common::sampling::Sampler);
//! * `hidden` is opaque (already `fast_project_in`-projected by the slow backbone).
//!
//! ⚠ Parity: the math is real but the box is offline. Every value that depends on the
//! published `config.json` / codec `.yaml` / tokenizer carries a `// PARITY:` flag here
//! or in the submodules; confirm them on-box before trusting numeric output.

// The backend body is Candle-backed and so lives behind the crate's `real` feature
// (mirroring `common::dualar`/`codec`); `config`/`sampling` remain pure-Rust.
#[cfg(feature = "real")]
mod codec;
#[cfg(feature = "real")]
mod fast_ar;
#[cfg(feature = "real")]
mod load;
#[cfg(feature = "real")]
mod nn;
#[cfg(feature = "real")]
mod slow_ar;
#[cfg(feature = "real")]
pub mod tokenizer;

#[cfg(feature = "real")]
pub use backend::S1Mini;

#[cfg(feature = "real")]
mod backend {
    use std::path::Path;

    use candle_core::{Device, Result, Tensor};

    use super::codec::ModdedDac;
    use super::fast_ar::FastAr;
    use super::load::{load_codec, load_lm};
    use super::slow_ar::SlowAr;
    use super::tokenizer::{FishTokenizer, IM_END_TOKEN, IM_START_TOKEN, VOICE_TOKEN};
    use crate::common::codec::RvqCodec;
    use crate::common::config::{CodecConfig, FishConfig};
    use crate::common::dualar::{drive, DriveParams, DualArBackend, SlowStep};
    use crate::common::sampling::Sampler;
    use crate::FishVariant;

    /// The fully-wired s1 backend: tokenizer + slow AR + fast AR + codec.
    pub struct S1Mini {
        cfg: FishConfig,
        tokenizer: FishTokenizer,
        slow: SlowAr,
        fast: FastAr,
        codec: ModdedDac,
        dev: Device,
    }

    impl S1Mini {
        /// Load every s1 component from a checkpoint directory `dir` containing
        /// `model.safetensors`, `codec.safetensors`, and `tokenizer.json` (the on-box
        /// conversion outputs; see [`super::load`] / [`super::tokenizer`]). An optional
        /// `config.json` overrides the [`FishConfig::s1_mini`] defaults.
        pub fn load(dir: impl AsRef<Path>, dev: Device) -> Result<Self> {
            let dir = dir.as_ref();
            let tok_path = dir.join("tokenizer.json");
            let tokenizer = FishTokenizer::from_file(&tok_path)
                .map_err(|e| candle_core::Error::Msg(format!("load tokenizer: {e}")))?;

            // Resolve the config: variant defaults, optionally overlaid by config.json.
            let cfg_path = dir.join("config.json");
            let mut cfg = if cfg_path.exists() {
                let json = std::fs::read_to_string(&cfg_path)
                    .map_err(|e| candle_core::Error::Msg(format!("read config.json: {e}")))?;
                FishConfig::from_fish_json(&json, FishVariant::S1Mini)
                    .map_err(candle_core::Error::Msg)?
            } else {
                FishConfig::s1_mini()
            };
            // Inject the tokenizer-resolved semantic range + stop id (reference
            // `from_pretrained` does this at load).
            cfg.semantic_begin_id = tokenizer.semantic_begin_id;
            cfg.semantic_end_id = tokenizer.semantic_end_id;
            cfg.stop_token_id = tokenizer.im_end_id;

            let lm_w = load_lm(
                dir.join("model.safetensors")
                    .to_str()
                    .ok_or_else(|| candle_core::Error::Msg("non-utf8 model path".into()))?,
                dev.clone(),
            )?;
            let slow = SlowAr::new(lm_w, cfg.clone())?;
            let fast = FastAr::new(cfg.clone(), &dev)?;

            let codec_w = load_codec(
                dir.join("codec.safetensors")
                    .to_str()
                    .ok_or_else(|| candle_core::Error::Msg("non-utf8 codec path".into()))?,
                dev.clone(),
            )?;
            let codec = ModdedDac::new(codec_w, cfg.codec.clone());

            Ok(Self {
                cfg,
                tokenizer,
                slow,
                fast,
                codec,
                dev,
            })
        }

        /// Build the encoded prompt `[1 + num_codebooks, T]` for `text` (no reference
        /// voice): row 0 = the chat-formatted token ids; rows `1..` = 0 (no audio in
        /// the prompt). Inline emotion tags like `(happy)` are plain text.
        ///
        /// PARITY: the exact chat template is reconstructed from `generate_long` +
        /// `ContentSequence` (the system instruction "convert the provided text to
        /// speech", `<|im_start|>`/`<|im_end|>` role turns, and the `<|voice|>`
        /// modality marker opening the assistant turn). `conversation.py` was not in
        /// the reference bundle, so confirm the precise spacing/newlines on-box.
        pub fn build_prompt(&self, text: &str) -> Result<Tensor> {
            let voice = VOICE_TOKEN;
            let im_s = IM_START_TOKEN;
            let im_e = IM_END_TOKEN;
            // System + user turns closed with <|im_end|>; the assistant turn opens with
            // the voice modality marker and is left open for generation.
            let prompt = format!(
                "{im_s}system\nconvert the provided text to speech{im_e}\
                 {im_s}user\n{text}{im_e}\
                 {im_s}assistant\n{voice}"
            );
            let ids = self
                .tokenizer
                .encode(&prompt)
                .map_err(|e| candle_core::Error::Msg(format!("encode prompt: {e}")))?;
            let t = ids.len();
            let n_cb = self.cfg.codec.num_codebooks;
            let mut flat = vec![0u32; (1 + n_cb) * t];
            flat[..t].copy_from_slice(&ids); // row 0
            Tensor::from_vec(flat, (1 + n_cb, t), &self.dev)
        }

        /// Synthesize `text` → mono 44.1 kHz waveform: build the prompt, [`drive`] the
        /// dual-AR loop to a `[10, T]` code matrix, then [`RvqCodec::decode`] it.
        pub fn synthesize(&mut self, text: &str, params: &DriveParams) -> Result<Vec<f32>> {
            let prompt = self.build_prompt(text)?;
            let codes = drive(self, &prompt, params)?; // [num_codebooks, T]
            let wav = self.codec.decode(&codes)?;
            wav.to_vec1::<f32>()
        }

        /// Decode a precomputed `[num_codebooks, T]` code matrix to a waveform (the
        /// codec-only path, exposed for round-tripping / cloning workflows).
        pub fn decode_codes(&self, codes: &Tensor) -> Result<Tensor> {
            self.codec.decode(codes)
        }

        /// Encode a reference waveform to `[num_codebooks, T]` cloning codes.
        pub fn encode_reference(&self, wav: &Tensor) -> Result<Tensor> {
            self.codec.encode(wav)
        }
    }

    impl DualArBackend for S1Mini {
        fn config(&self) -> &FishConfig {
            &self.cfg
        }

        fn device(&self) -> Device {
            self.dev.clone()
        }

        fn reset(&mut self, _max_seq_len: usize) -> Result<()> {
            self.slow.reset();
            Ok(())
        }

        fn prefill(&mut self, prompt: &Tensor) -> Result<SlowStep> {
            let (semantic_logits, hidden) = self.slow.prefill(prompt)?;
            Ok(SlowStep {
                semantic_logits,
                hidden,
            })
        }

        fn slow_step(&mut self, frame: &[u32], pos: usize) -> Result<SlowStep> {
            let (semantic_logits, hidden) = self.slow.slow_step(frame, pos)?;
            Ok(SlowStep {
                semantic_logits,
                hidden,
            })
        }

        fn first_code(&self, semantic_token: u32) -> u32 {
            self.fast.first_code(semantic_token)
        }

        fn fast_expand(
            &self,
            hidden: &Tensor,
            first_code: u32,
            sampler: &mut Sampler,
        ) -> Result<Vec<u32>> {
            // The fast head shares the slow backbone's checkpoint; pass its weight bag.
            self.fast
                .expand(self.slow.weights(), hidden, first_code, sampler)
        }
    }

    impl RvqCodec for S1Mini {
        fn config(&self) -> &CodecConfig {
            &self.cfg.codec
        }

        fn decode(&self, codes: &Tensor) -> Result<Tensor> {
            self.codec.decode(codes)
        }

        fn encode(&self, wav: &Tensor) -> Result<Tensor> {
            self.codec.encode(wav)
        }
    }
}
