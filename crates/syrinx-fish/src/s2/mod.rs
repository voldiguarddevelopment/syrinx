//! `s2-pro` (5B) backend — the Qwen3-4B (`fish_qwen3_omni`) slow AR, the 4-layer ~400M
//! fast audio decoder (shared embedding + codebook-axis RoPE + MCF), the 446M EVA-GAN /
//! causal-DAC RVQ codec, and the Qwen3 155k BPE tokenizer + prompt builder, wired into
//! an [`S2Pro`] that implements the shared
//! [`DualArBackend`](crate::common::dualar::DualArBackend) +
//! [`RvqCodec`](crate::common::codec::RvqCodec) contracts.
//!
//! Honours the foundation's rules (identical to s1):
//! * slow steps return **raw, unmasked** semantic logits over the full 155 776 slow
//!   vocab — the driver owns the semantic constraint + RAS;
//! * `fast_expand` is `&self` and allocates its tiny per-frame KV cache locally, drawing
//!   every residual via the **driver-owned** [`Sampler`](crate::common::sampling::Sampler);
//! * `hidden` is opaque (already `fast_project_in`-projected by the slow backbone).
//!
//! ⚠ Parity: the math is real but the box is offline. Every value that depends on the
//! published `config.json` / `codec.pth` / `tokenizer.json` carries a `// PARITY:` flag
//! here or in the submodules; confirm them on-box before trusting numeric output. The
//! **codec** (`s2::codec`) is the riskiest module — see its header.

// The backend body is Candle-backed and so lives behind the crate's `real` feature
// (mirroring `common::dualar`/`codec` and the s1 backend); `config`/`sampling` stay pure.
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
pub use backend::S2Pro;

#[cfg(feature = "real")]
mod backend {
    use std::path::Path;

    use candle_core::{Device, DType, Result, Tensor};

    use super::codec::EvaGanDac;
    use super::fast_ar::FastAr;
    use super::load::{load_codec, load_lm};
    use super::slow_ar::SlowAr;
    use super::tokenizer::{Qwen3Tokenizer, IM_END_TOKEN, IM_START_TOKEN};
    use crate::common::codec::RvqCodec;
    use crate::common::config::{CodecConfig, FishConfig};
    use crate::common::dualar::{drive, DriveParams, DualArBackend, SlowStep};
    use crate::common::sampling::Sampler;
    use crate::FishVariant;

    /// Reconcile the fast-AR geometry + residual codebook width against the REAL loaded
    /// `audio_decoder.*` tensors. In the published s2-pro checkpoint the fast AR block is
    /// structurally identical to the slow backbone (dim 2560, GQA 32:8, head_dim 128, ffn
    /// 9728, fused wqkv 6144 = q4096+k1024+v1024) — only the layer count (4), the shared
    /// 4096-wide code table, and the absence of QK-norm/qkv-o bias differ. Deriving from
    /// the tensors guarantees the right shapes even when config.json's
    /// `audio_decoder_config` omits fields (whose variant defaults are a smaller head).
    fn reconcile_fast_cfg(cfg: &mut FishConfig, w: &super::nn::Weights) -> Result<()> {
        // No fast tables loaded (e.g. a slow-only checkpoint) → leave the config as-is.
        if !w.has("fast_embeddings.weight") {
            return Ok(());
        }
        // The fast block mirrors the slow backbone's geometry.
        let s = cfg.slow.clone();
        {
            let f = &mut cfg.fast.transformer;
            f.dim = s.dim;
            f.n_head = s.n_head;
            f.n_local_heads = s.n_local_heads;
            f.head_dim = s.head_dim;
            f.intermediate_size = s.intermediate_size;
            f.rope_base = s.rope_base;
            f.norm_eps = s.norm_eps;
            // Verified against the tensors: the fast AR has no QK-norm and no qkv/o bias.
            f.attention_qk_norm = w.has("fast_layers.0.attention.q_norm.weight");
            f.attention_qkv_bias = w.has("fast_layers.0.attention.wqkv.bias");
            f.attention_o_bias = w.has("fast_layers.0.attention.wo.bias");
        }
        // Residual codebook width == the shared value/output table height (4096). This is
        // both the MCF per-codebook stride on the slow embed and the fast head's logit
        // width (and so `first_code`'s clamp ceiling).
        let residual = w.g("fast_embeddings.weight")?.dim(0)?;
        cfg.codec.residual_size = residual;
        Ok(())
    }

    /// The fully-wired s2 backend: tokenizer + slow AR + fast AR + codec.
    pub struct S2Pro {
        cfg: FishConfig,
        tokenizer: Qwen3Tokenizer,
        slow: SlowAr,
        fast: FastAr,
        codec: EvaGanDac,
        dev: Device,
    }

    impl S2Pro {
        /// Load every s2 component from a checkpoint directory `dir` containing the
        /// sharded LM (`model-*-of-*.safetensors` + `model.safetensors.index.json`),
        /// `codec.pth`, `tokenizer.json`, and `config.json`.
        ///
        /// The compute dtype is chosen by the device: **f32 on CPU** (the parity path,
        /// byte-unchanged) and **bf16 on CUDA** (so the 4.4B LM fits a 12 GB GPU — f32
        /// would need ~18 GB). Use [`S2Pro::load_with_dtype`] to override.
        pub fn load(dir: impl AsRef<Path>, dev: Device) -> Result<Self> {
            // CPU must stay f32 (parity); CUDA defaults to bf16 (fit).
            let dt = if dev.is_cuda() { DType::BF16 } else { DType::F32 };
            Self::load_with_dtype(dir, dev, dt)
        }

        /// Like [`S2Pro::load`] but with an explicit compute dtype `dt`. CPU callers
        /// should pass `DType::F32` to preserve parity; CUDA callers `DType::BF16` to fit.
        pub fn load_with_dtype(dir: impl AsRef<Path>, dev: Device, dt: DType) -> Result<Self> {
            let dir = dir.as_ref();

            let tok_path = dir.join("tokenizer.json");
            let tokenizer = Qwen3Tokenizer::from_file(&tok_path)
                .map_err(|e| candle_core::Error::Msg(format!("load tokenizer: {e}")))?;

            // Resolve the config: S2 variant defaults, overlaid by config.json (which is
            // authoritative for the verified Qwen3 dims + the semantic id range).
            let cfg_path = dir.join("config.json");
            let mut cfg = if cfg_path.exists() {
                let json = std::fs::read_to_string(&cfg_path)
                    .map_err(|e| candle_core::Error::Msg(format!("read config.json: {e}")))?;
                FishConfig::from_fish_json(&json, FishVariant::S2Pro)
                    .map_err(candle_core::Error::Msg)?
            } else {
                FishConfig::s2_pro()
            };
            // Inject the tokenizer-resolved semantic range + stop id (the reference does
            // this at load). For s2 these match config.json's 151678/155773 + 151645.
            cfg.semantic_begin_id = tokenizer.semantic_begin_id;
            cfg.semantic_end_id = tokenizer.semantic_end_id;
            cfg.stop_token_id = tokenizer.im_end_id;

            let lm_w = load_lm(dir, dev.clone(), dt)?;
            // Reconcile the fast-AR geometry + residual codebook width against the REAL
            // `audio_decoder.*` tensors before wiring the heads (see `reconcile_fast_cfg`).
            reconcile_fast_cfg(&mut cfg, &lm_w)?;
            let slow = SlowAr::new(lm_w, cfg.clone())?;
            let fast = FastAr::new(cfg.clone(), &dev, dt)?;

            let codec_w = load_codec(
                dir.join("codec.pth")
                    .to_str()
                    .ok_or_else(|| candle_core::Error::Msg("non-utf8 codec path".into()))?,
                dev.clone(),
                dt,
            )?;
            let codec = EvaGanDac::new(codec_w, cfg.codec.clone());

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
        /// voice): row 0 = the Qwen3 chat-formatted token ids; rows `1..` = 0 (no audio
        /// in the prompt). Emotion tags like `[whisper]` are plain text.
        ///
        /// PARITY: `chat_template.jinja` is the standard Qwen3 text template — the audio
        /// modality assembly lives in model code, not the jinja. The system instruction
        /// and exact turn spacing below are reconstructed from the Qwen3 chat format;
        /// confirm the production TTS system prompt + any voice-modality marker on-box.
        pub fn build_prompt(&self, text: &str) -> Result<Tensor> {
            let im_s = IM_START_TOKEN;
            let im_e = IM_END_TOKEN;
            let prompt = format!(
                "{im_s}system\nConvert the provided text to speech.{im_e}\n\
                 {im_s}user\n{text}{im_e}\n\
                 {im_s}assistant\n"
            );
            self.encode_prompt(&prompt)
        }

        /// Build a voice-cloning prompt that **prepends the reference audio** to the
        /// conversation: the reference transcript as the user turn and the reference RVQ
        /// codes `ref_codes` `[num_codebooks, T_ref]` as the assistant audio turn, then
        /// the target `text`. Row 0 carries the slow-vocab ids (text positions = BPE ids;
        /// reference-audio positions = `semantic_begin + ref_codes[0]`); rows `1..` carry
        /// the reference codes on the audio positions and 0 elsewhere.
        ///
        /// PARITY: the precise interleaving (where the `<|im_end|>`/turn markers sit
        /// relative to the audio span, and whether a `<|semantic|>`-style opener precedes
        /// the codes) is reconstructed and unconfirmed — verify against the s2 reference
        /// `encode_tokens` / `ContentSequence` on-box.
        pub fn build_prompt_with_reference(
            &self,
            ref_text: &str,
            ref_codes: &Tensor,
            text: &str,
        ) -> Result<Tensor> {
            let im_s = IM_START_TOKEN;
            let im_e = IM_END_TOKEN;
            let n_cb = self.cfg.codec.num_codebooks;

            // Text segments (their code rows are zero).
            let head = format!(
                "{im_s}system\nConvert the provided text to speech.{im_e}\n\
                 {im_s}user\n{ref_text}{im_e}\n{im_s}assistant\n"
            );
            let head_ids = self
                .tokenizer
                .encode(&head)
                .map_err(|e| candle_core::Error::Msg(format!("encode ref head: {e}")))?;

            let tail = format!("{im_e}\n{im_s}user\n{text}{im_e}\n{im_s}assistant\n");
            let tail_ids = self
                .tokenizer
                .encode(&tail)
                .map_err(|e| candle_core::Error::Msg(format!("encode ref tail: {e}")))?;

            // Reference audio span.
            let ref_codes = if ref_codes.rank() == 3 {
                ref_codes.squeeze(0)?
            } else {
                ref_codes.clone()
            };
            debug_assert_eq!(ref_codes.dim(0)?, n_cb, "ref_codes must be [num_codebooks, T]");
            let t_ref = ref_codes.dim(1)?;
            let ref_host: Vec<u32> = ref_codes
                .to_dtype(candle_core::DType::U32)?
                .flatten_all()?
                .to_vec1()?;
            let ref_row =
                |r: usize| -> Vec<u32> { (0..t_ref).map(|c| ref_host[r * t_ref + c]).collect() };

            let total = head_ids.len() + t_ref + tail_ids.len();
            let mut flat = vec![0u32; (1 + n_cb) * total];
            let put = |flat: &mut [u32], r: usize, c: usize, val: u32| {
                flat[r * total + c] = val;
            };

            // head text
            for (c, &id) in head_ids.iter().enumerate() {
                put(&mut flat, 0, c, id);
            }
            // reference audio: row0 = semantic token id from codebook-0; rows 1.. = codes.
            let begin = self.cfg.semantic_begin_id;
            let sem0 = ref_row(0);
            for (c, &s0) in sem0.iter().enumerate() {
                let col = head_ids.len() + c;
                put(&mut flat, 0, col, begin + s0);
                for r in 0..n_cb {
                    put(&mut flat, r + 1, col, ref_row(r)[c]);
                }
            }
            // tail text
            for (i, &id) in tail_ids.iter().enumerate() {
                let col = head_ids.len() + t_ref + i;
                put(&mut flat, 0, col, id);
            }

            Tensor::from_vec(flat, (1 + n_cb, total), &self.dev)
        }

        /// Encode a chat-formatted text prompt into `[1 + num_codebooks, T]` (row 0 =
        /// ids, code rows = 0).
        fn encode_prompt(&self, prompt: &str) -> Result<Tensor> {
            let ids = self
                .tokenizer
                .encode(prompt)
                .map_err(|e| candle_core::Error::Msg(format!("encode prompt: {e}")))?;
            let t = ids.len();
            let n_cb = self.cfg.codec.num_codebooks;
            let mut flat = vec![0u32; (1 + n_cb) * t];
            flat[..t].copy_from_slice(&ids); // row 0
            Tensor::from_vec(flat, (1 + n_cb, t), &self.dev)
        }

        /// Synthesize `text` → mono 44.1 kHz waveform.
        pub fn synthesize(&mut self, text: &str, params: &DriveParams) -> Result<Vec<f32>> {
            let prompt = self.build_prompt(text)?;
            let codes = drive(self, &prompt, params)?; // [num_codebooks, T]
            let wav = self.codec.decode(&codes)?;
            wav.to_vec1::<f32>()
        }

        /// Synthesize `text` in the cloned voice of `ref_text`/`ref_codes`.
        pub fn synthesize_cloned(
            &mut self,
            ref_text: &str,
            ref_codes: &Tensor,
            text: &str,
            params: &DriveParams,
        ) -> Result<Vec<f32>> {
            let prompt = self.build_prompt_with_reference(ref_text, ref_codes, text)?;
            let codes = drive(self, &prompt, params)?;
            let wav = self.codec.decode(&codes)?;
            wav.to_vec1::<f32>()
        }

        /// Decode a precomputed `[num_codebooks, T]` code matrix to a waveform.
        pub fn decode_codes(&self, codes: &Tensor) -> Result<Tensor> {
            self.codec.decode(codes)
        }

        /// Encode a reference waveform to `[num_codebooks, T]` cloning codes.
        pub fn encode_reference(&self, wav: &Tensor) -> Result<Tensor> {
            self.codec.encode(wav)
        }
    }

    impl DualArBackend for S2Pro {
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

    impl RvqCodec for S2Pro {
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
