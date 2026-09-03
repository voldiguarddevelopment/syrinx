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

    use candle_core::{DType, Device, Result, Tensor};

    use super::fast_ar::FastAr;
    use super::load::load_lm;
    use super::slow_ar::SlowAr;
    use super::tokenizer::{FishTokenizer, IM_END_TOKEN, IM_START_TOKEN, INTERLEAVE_TOKEN, SPEAKER0_TOKEN, VOICE_TOKEN};
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
        codec: crate::s2::codec::EvaGanDac,
        /// Path to `codec.pth`, so `encode_reference` can materialise the encode-side
        /// stack on demand and drop it again.
        codec_path: std::path::PathBuf,
        /// Compute dtype for the CODEC: f32 on CPU (parity), bf16 on CUDA.
        ///
        /// This is not cosmetic. candle builds every conv1d through an `im2col` buffer
        /// of `L * C_in * k` elements, and the codec's ENCODE stack runs at the full
        /// waveform length. For a 35 s reference at 44.1 kHz with C=192, k=7 that is
        /// 4.2 GB in bf16 and **8.4 GB in f32** — a single allocation large enough to
        /// fail on a 12 GB card with 8 GB free. s2 has always picked this by device;
        /// s1 hardcoded f32 and OOMed on long references.
        codec_dt: DType,
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
                {
                    // Fish ships `model.pth`; fall back to a converted safetensors.
                    let pth = dir.join("model.pth");
                    let st = dir.join("model.safetensors");
                    let p = if pth.exists() { pth } else { st };
                    p.to_str()
                        .ok_or_else(|| candle_core::Error::Msg("non-utf8 model path".into()))?
                        .to_string()
                }
                .as_str(),
                dev.clone(),
            )?;
            let slow = SlowAr::new(lm_w, cfg.clone())?;
            let fast = FastAr::new(cfg.clone(), &dev)?;

            // `codec.pth` in the s1-mini release is BYTE-IDENTICAL to s2-pro's
            // (same md5), so the codec is the same EVA-GAN/DAC stack. Drive it with the
            // s2 implementation — which is parity-checked and carries the chunked
            // decode — instead of the s1 `ModdedDac`, which was written against an
            // assumed "modded-DAC" that this checkpoint is not. The codec geometry
            // comes from s2's config for the same reason: identical weights.
            let codec_cfg = crate::common::config::FishConfig::s2_pro().codec;
            cfg.codec.semantic_size = codec_cfg.semantic_size;
            cfg.codec.codebook_dim = codec_cfg.codebook_dim;
            cfg.codec.sample_rate = codec_cfg.sample_rate;
            let codec_path = dir.join("codec.pth");
            let codec_dt = if dev.is_cuda() { DType::BF16 } else { DType::F32 };
            let codec_w = crate::s2::load::load_codec(
                codec_path
                    .to_str()
                    .ok_or_else(|| candle_core::Error::Msg("non-utf8 codec path".into()))?,
                dev.clone(),
                codec_dt,
                crate::s2::load::CodecParts::Decode,
            )?;
            let codec = crate::s2::codec::EvaGanDac::new(codec_w, codec_cfg);

            Ok(Self {
                cfg,
                tokenizer,
                slow,
                fast,
                codec,
                codec_path,
                codec_dt,
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
            let ids = self.encode_prompt_ids(text)?;
            let t = ids.len();
            let n_cb = self.cfg.codec.num_codebooks;
            let mut flat = vec![0u32; (1 + n_cb) * t];
            flat[..t].copy_from_slice(&ids); // row 0 = text ids; code rows stay 0
            Tensor::from_vec(flat, (1 + n_cb, t), &self.dev)
        }

        /// The chat-format token ids for a plain (un-cloned) request.
        ///
        /// PARITY: taken from the reference `generate_long` +
        /// `ContentSequence(modality="interleave")` in `fish_speech`, which builds
        ///
        /// ```text
        /// <|interleave|><|speaker:0|>{text}
        /// ```
        ///
        /// and leaves the sequence OPEN (`add_end=False`) so generation continues from
        /// the last text token. There is no system/user/assistant role turn and no
        /// `<|voice|>` marker on this path — an earlier reconstruction assumed a
        /// ChatML-style template, which made the model emit its stop token after three
        /// frames instead of speaking.
        fn encode_prompt_ids(&self, text: &str) -> Result<Vec<u32>> {
            let prompt = format!("{INTERLEAVE_TOKEN}{SPEAKER0_TOKEN}{text}");
            self.tokenizer
                .encode(&prompt)
                .map_err(|e| candle_core::Error::Msg(format!("encode prompt: {e}")))
        }

        /// Synthesize `text` → mono 44.1 kHz waveform: build the prompt, [`drive`] the
        /// dual-AR loop to a `[10, T]` code matrix, then [`RvqCodec::decode`] it.
        pub fn synthesize(&mut self, text: &str, params: &DriveParams) -> Result<Vec<f32>> {
            let prompt = self.build_prompt(text)?;
            if std::env::var("SYRINX_FISH_DUMP").is_ok() {
                let ids: Vec<u32> = prompt.narrow(0, 0, 1)?.flatten_all()?.to_dtype(candle_core::DType::U32)?.to_vec1()?;
                eprintln!("s1 DUMP prompt [{}x{}] row0 ids: {:?}",
                    prompt.dim(0)?, prompt.dim(1)?, &ids[..ids.len().min(24)]);
            }
            let codes = drive(self, &prompt, params)?; // [num_codebooks, T]
            if std::env::var("SYRINX_FISH_DUMP").is_ok() {
                let (nc, t) = (codes.dim(0)?, codes.dim(1)?);
                let flat: Vec<u32> = codes.to_dtype(candle_core::DType::U32)?.flatten_all()?.to_vec1()?;
                let row0: Vec<u32> = (0..t.min(16)).map(|c| flat[c]).collect();
                let mx = flat.iter().copied().max().unwrap_or(0);
                let mn = flat.iter().copied().min().unwrap_or(0);
                let distinct = flat.iter().collect::<std::collections::HashSet<_>>().len();
                eprintln!("s1 DUMP codes [{nc}x{t}] min={mn} max={mx} distinct={distinct} row0[..16]={row0:?}");
            }
            let wav = self.codec.decode(&codes)?;
            wav.to_dtype(DType::F32)?.to_vec1::<f32>()
        }


        /// Build the cloning prompt `[1 + num_codebooks, T]` from a reference
        /// transcript + its codec codes, followed by the target `text`.
        ///
        /// PARITY: reproduces the reference `generate_long` when `use_prompt` is set —
        /// `ContentSequence(modality="interleave")` with
        ///
        /// ```text
        /// <|interleave|>  <|speaker:0|>{ref_text} {ref_codes}  <|im_end|>  <|speaker:0|>{text}
        /// ```
        ///
        /// i.e. the reference turn is `[TextPart(ref_text), VQPart(ref_codes)]` closed
        /// with `add_end=True`, then the target turn is `[TextPart(text)]` left OPEN
        /// (`add_end=False`) so generation continues from it.
        ///
        /// Row 0 at a reference-audio column carries the `<|semantic:i|>` token id for
        /// codebook 0 (`semantic_id_to_token_id[code]` in the reference); rows `1..=n_cb`
        /// carry the raw RVQ codes for that frame.
        pub fn build_prompt_with_reference(
            &self,
            ref_text: &str,
            ref_codes: &Tensor,
            text: &str,
        ) -> Result<Tensor> {
            let n_cb = self.cfg.codec.num_codebooks;

            // Head: modality marker + the reference turn's TEXT part.
            let head = format!("{INTERLEAVE_TOKEN}{SPEAKER0_TOKEN}{ref_text}");
            let head_ids = self
                .tokenizer
                .encode(&head)
                .map_err(|e| candle_core::Error::Msg(format!("encode ref head: {e}")))?;

            // Tail: close the reference turn, then the OPEN target turn.
            let tail = format!("{IM_END_TOKEN}{SPEAKER0_TOKEN}{text}");
            let tail_ids = self
                .tokenizer
                .encode(&tail)
                .map_err(|e| candle_core::Error::Msg(format!("encode ref tail: {e}")))?;

            let ref_codes = if ref_codes.rank() == 3 {
                ref_codes.squeeze(0)?
            } else {
                ref_codes.clone()
            };
            if ref_codes.dim(0)? != n_cb {
                return Err(candle_core::Error::Msg(format!(
                    "ref_codes must be [{n_cb}, T], got {:?}",
                    ref_codes.dims()
                )));
            }
            let t_ref = ref_codes.dim(1)?;
            let host: Vec<u32> = ref_codes
                .to_dtype(candle_core::DType::U32)?
                .flatten_all()?
                .to_vec1()?;
            let row = |r: usize| -> Vec<u32> { (0..t_ref).map(|c| host[r * t_ref + c]).collect() };

            let total = head_ids.len() + t_ref + tail_ids.len();
            let mut flat = vec![0u32; (1 + n_cb) * total];
            let put = |flat: &mut [u32], r: usize, c: usize, val: u32| {
                flat[r * total + c] = val;
            };

            for (c, &id) in head_ids.iter().enumerate() {
                put(&mut flat, 0, c, id);
            }
            let begin = self.cfg.semantic_begin_id;
            let sem0 = row(0);
            for (c, &s0) in sem0.iter().enumerate() {
                let col = head_ids.len() + c;
                put(&mut flat, 0, col, begin + s0);
                for r in 0..n_cb {
                    put(&mut flat, r + 1, col, row(r)[c]);
                }
            }
            for (i, &id) in tail_ids.iter().enumerate() {
                let col = head_ids.len() + t_ref + i;
                put(&mut flat, 0, col, id);
            }

            Tensor::from_vec(flat, (1 + n_cb, total), &self.dev)
        }

        /// Synthesize `text` in the cloned voice described by `ref_text`/`ref_codes`.
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
            wav.to_dtype(DType::F32)?.to_vec1::<f32>()
        }

        /// Decode a precomputed `[num_codebooks, T]` code matrix to a waveform (the
        /// codec-only path, exposed for round-tripping / cloning workflows).
        pub fn decode_codes(&self, codes: &Tensor) -> Result<Tensor> {
            self.codec.decode(codes)
        }

        /// Encode a reference waveform to `[num_codebooks, T]` cloning codes.
        pub fn encode_reference(&self, wav: &Tensor) -> Result<Tensor> {
            // Only the DECODE side of the codec is resident (see `load`). The encode
            // stack is read here, used once, and dropped — cloning encodes the
            // reference exactly once per run, so this buys back its footprint for the
            // whole of generation. Mirrors `S2Pro::encode_reference`.
            let w = crate::s2::load::load_codec(
                self.codec_path
                    .to_str()
                    .ok_or_else(|| candle_core::Error::Msg("non-utf8 codec path".into()))?,
                self.dev.clone(),
                self.codec_dt,
                crate::s2::load::CodecParts::Encode,
            )?;
            crate::s2::codec::EvaGanDac::new(w, self.cfg.codec.clone()).encode(wav)
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
