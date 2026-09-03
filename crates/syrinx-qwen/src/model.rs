//! The assembled Qwen3-TTS model: weight loading, and the **dual-AR generation loop**
//! that drives the talker and the code predictor together.
//!
//! # The loop, and where it comes from
//!
//! Every step below is transcribed from the reference (`qwen_tts.core.models.
//! modeling_qwen3_tts`), not inferred. The Fish port's lesson — a guessed prompt/feedback
//! structure produced three frames of noise — applies with force here, because the
//! feedback path is the least obvious part of the model. Named sources:
//!
//! * `Qwen3TTSTalkerForConditionalGeneration.forward` — the generation branch. This is the
//!   feedback: it is **not** "re-embed code group 0". It is the SUM of all
//!   `num_code_groups` embeddings of the frame just emitted, plus one text hidden state.
//! * `Qwen3TTSTalkerCodePredictorModelForConditionalGeneration.forward` — the predictor's
//!   two-position prefill and its per-group table indexing.
//! * `Qwen3TTSForConditionalGeneration.generate` — `suppress_tokens`, `min_new_tokens`, the
//!   EOS trim, and which sampling knobs reach which head.
//!
//! ## One frame
//!
//! ```text
//!   talker hidden h[j-1]  ──codec_head──►  logits over the talker's codec vocab
//!                                          (repetition penalty → EOS guard → suppress
//!                                           → temperature → top-k → top-p → multinomial)
//!                                       └► code group 0  ─── EOS? ──► stop, frame dropped
//!
//!   code predictor, cache RESET, prefill [ h[j-1] , talker.codec_embedding(c0) ]
//!       (both at TALKER width, then `small_to_mtp_projection` narrows them)
//!     ├─ lm_head.0  on the prefill's last position → group 1
//!     ├─ codec_embedding.0(c1) → step → lm_head.1  → group 2
//!     └─ … codec_embedding.13(c14) → step → lm_head.14 → group 15
//!
//!   feedback = Σ over the 16 groups of their embeddings, at talker width:
//!       talker.model.codec_embedding(c0)
//!     + Σ_{i=0..14} code_predictor.model.codec_embedding.i(c_{i+1})
//!   plus  trailing_text_hidden[j]  if j < len(trailing)  else  tts_pad_embed
//!
//!   talker.forward(feedback) → h[j]
//! ```
//!
//! ## Three things that are easy to get wrong
//!
//! * **The predictor's cache is per FRAME, not per utterance.** Its context is exactly the
//!   16 positions of one frame — `precompute_rope` in [`crate::code_predictor`] only builds
//!   `num_code_groups` rows, so a missing reset does not merely degrade the audio, it runs
//!   off the end of the RoPE table on frame 2.
//! * **`codec_embedding.{i}` embeds group `i + 1`.** Group 0 has its own table on the
//!   talker. Confirmed twice in the reference: `forward_sub_talker_finetune` indexes
//!   `get_input_embeddings()[i-1]` for `codec_ids[:, i]`, and the generation branch pairs
//!   `get_input_embeddings()[i]` with `sequences[..., i]` (which is group `i + 1`).
//! * **The EOS frame is discarded.** HF's `_sample` appends the EOS token and stops before
//!   the next forward, so the code predictor never runs for it and no 16-code frame exists.
//!   The reference's own trim (`effective_lengths = stop_indices`) drops it a second time.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};

use crate::code_predictor::CodePredictor;
use crate::config::Qwen3TtsConfig;
use crate::load;
use crate::nn::Weights;
use crate::sampling::{self, Sampler, SamplingParams};
use crate::talker::Talker;

/// Knobs for one [`Qwen3Tts::generate`] run.
#[derive(Debug, Clone, PartialEq)]
pub struct DriveParams {
    /// PRNG seed. The talker draw and the 15 predictor draws of every frame share one
    /// stream, so `(seed, weights, prompt)` reproduces bit-for-bit.
    pub seed: u64,
    /// Hard cap on frames (the reference `max_new_tokens`, whose shipped
    /// `generation_config.json` value is 8192 — 8192 / 12.5 Hz ≈ 11 minutes).
    pub max_new_frames: usize,
    /// Frames that must exist before the codec EOS may be drawn (the reference's
    /// `min_new_tokens = 2`, hardcoded in `Qwen3TTSForConditionalGeneration.generate`).
    pub min_new_frames: usize,
    /// Warpers for code group 0 (`temperature` / `top_p` / `top_k` / `repetition_penalty`).
    pub talker: SamplingParams,
    /// Warpers for groups `1..num_code_groups` (the reference's `subtalker_*`).
    pub code_predictor: SamplingParams,
}

impl Default for DriveParams {
    fn default() -> Self {
        Self {
            // The Fish port pins seed 0 for reproducible corpus renders; same here.
            seed: 0,
            max_new_frames: 8192,
            min_new_frames: 2,
            talker: SamplingParams::talker(),
            code_predictor: SamplingParams::code_predictor(),
        }
    }
}

/// The talker's prefill, in the shape `Qwen3TTSTalkerForConditionalGeneration.generate`
/// receives it.
///
/// All three tensors are at the **talker's** hidden size and in the model's compute dtype.
/// Building them is the prompt module's job (text ids → `text_embedding` →
/// `text_projection`, plus the codec tag/speaker/BOS embeddings, plus the reference-clip
/// ICL block); this type is the contract between that module and the loop.
#[derive(Debug)]
pub struct TalkerPrompt {
    /// `[1, T, hidden]` — the whole prefill: role tokens, codec tag/speaker/BOS, and the
    /// text or ICL block, already summed the way the reference sums them.
    pub inputs_embeds: Tensor,
    /// `[1, Tt, hidden]` — the projected text hidden states consumed one per generated
    /// frame. Frame `j` adds `trailing_text_hidden[:, j]` while `j < Tt`.
    pub trailing_text_hidden: Tensor,
    /// `[1, 1, hidden]` — the projected `tts_pad_token_id` embedding, added to every frame
    /// once the trailing text is exhausted.
    pub tts_pad_embed: Tensor,
}

/// What one generation run produced.
#[derive(Debug, Clone, PartialEq)]
pub struct Generated {
    /// `frames[t][g]` — `num_code_groups` codes per frame, frame-major. This is the layout
    /// the reference hands the 12 Hz tokenizer (`{"audio_codes": codes}`, `[T, 16]`).
    pub frames: Vec<Vec<u32>>,
    /// Whether the run ended on the codec EOS rather than the frame cap.
    pub stopped_on_eos: bool,
}

impl Generated {
    /// Frames generated.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether no frame was produced (an immediate EOS).
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Transpose to `rows[g][t]` — one row per RVQ layer, which is what
    /// [`crate::codec::rvq::Rvq::decode`] consumes.
    pub fn group_rows(&self, num_code_groups: usize) -> Vec<Vec<u32>> {
        (0..num_code_groups)
            .map(|g| self.frames.iter().map(|f| f[g]).collect())
            .collect()
    }
}

/// The talker + code predictor, loaded and ready to generate.
pub struct Qwen3Tts {
    talker: Talker,
    predictor: CodePredictor,
    cfg: Qwen3TtsConfig,
    /// The reference's `suppress_tokens`, precomputed once.
    suppress: Vec<u32>,
}

impl Qwen3Tts {
    /// Load a published checkpoint directory (`config.json` + `model.safetensors`).
    ///
    /// The compute dtype is chosen **by the device**: f32 on CPU (the parity path) and
    /// bf16 on CUDA (the fit path). This is not a stylistic choice — the sibling Fish s2
    /// port hardcoded f32 on its codec and hit a single 8.4 GB allocation that OOM'd on a
    /// long reference clip. Use [`Qwen3Tts::load_with_dtype`] to override deliberately.
    pub fn load(dir: impl AsRef<Path>, dev: Device) -> Result<Self> {
        let dt = if dev.is_cuda() { DType::BF16 } else { DType::F32 };
        Self::load_with_dtype(dir, dev, dt)
    }

    /// Like [`Qwen3Tts::load`] but with an explicit compute dtype.
    pub fn load_with_dtype(dir: impl AsRef<Path>, dev: Device, dt: DType) -> Result<Self> {
        let dir = dir.as_ref();
        let json = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| candle_core::Error::Msg(format!("{}/config.json: {e}", dir.display())))?;
        let cfg = Qwen3TtsConfig::from_json(&json).map_err(candle_core::Error::Msg)?;
        let map = load::load_tensors(dir, &dev, dt)?;
        let (talker_map, predictor_map) = load::split_stacks(map);
        Self::from_maps(talker_map, predictor_map, cfg, dev, dt)
    }

    /// Build from two already-split weight bags. Kept public so a caller that materialised
    /// the checkpoint some other way (a quantised bag, a test fixture) can still assemble
    /// the model.
    pub fn from_maps(
        talker_map: HashMap<String, Tensor>,
        predictor_map: HashMap<String, Tensor>,
        cfg: Qwen3TtsConfig,
        dev: Device,
        dt: DType,
    ) -> Result<Self> {
        let talker = Talker::new(
            Weights { map: talker_map, dev: dev.clone(), dt },
            cfg.clone(),
        )?;
        let predictor = CodePredictor::new(Weights { map: predictor_map, dev, dt }, cfg.clone())?;
        let suppress = suppressed_ids(&cfg);
        Ok(Self { talker, predictor, cfg, suppress })
    }

    /// The parsed geometry.
    pub fn config(&self) -> &Qwen3TtsConfig {
        &self.cfg
    }

    /// The device everything is on.
    pub fn device(&self) -> Device {
        self.talker.device()
    }

    /// The compute dtype everything was loaded in.
    pub fn dtype(&self) -> DType {
        self.talker.dtype()
    }

    /// The talker stack (its `embed_text` is what the prompt module needs).
    pub fn talker(&self) -> &Talker {
        &self.talker
    }

    /// The talker stack, mutably.
    pub fn talker_mut(&mut self) -> &mut Talker {
        &mut self.talker
    }

    /// The code predictor stack.
    pub fn code_predictor(&self) -> &CodePredictor {
        &self.predictor
    }

    /// Mutable access to the code predictor, for callers that need to drive it directly
    /// (its `forward` advances a cache). Mirrors [`Self::talker_mut`].
    pub fn code_predictor_mut(&mut self) -> &mut CodePredictor {
        &mut self.predictor
    }

    /// The ids masked to `-inf` on every talker draw (the reference `suppress_tokens`).
    pub fn suppressed_ids(&self) -> &[u32] {
        &self.suppress
    }

    /// Embed one full frame of `num_code_groups` codes at the talker's hidden width — the
    /// sum the reference feeds back, and the same sum `generate_icl_prompt` uses to embed
    /// a reference clip's codes. Exposed because the prompt module needs it for ICL.
    ///
    /// `frame[0]` goes through the talker's own `codec_embedding`; `frame[i]` for `i >= 1`
    /// goes through the code predictor's `codec_embedding.{i - 1}`.
    pub fn embed_frame(&self, frame: &[u32]) -> Result<Tensor> {
        if frame.len() != self.cfg.num_code_groups {
            return Err(candle_core::Error::Msg(format!(
                "embed_frame: got {} codes, expected {}",
                frame.len(),
                self.cfg.num_code_groups
            )));
        }
        let mut sum = self.talker.embed_codec(&frame[0..1])?;
        for (i, &code) in frame.iter().enumerate().skip(1) {
            sum = (sum + self.predictor.embed_group(i - 1, code)?)?;
        }
        Ok(sum)
    }

    /// Drive the dual-AR loop to completion.
    ///
    /// Returns the frames generated **before** the codec EOS (or before `max_new_frames`);
    /// the EOS frame itself never exists, because the reference stops before running the
    /// code predictor for it.
    pub fn generate(&mut self, prompt: &TalkerPrompt, params: &DriveParams) -> Result<Generated> {
        self.talker.reset();
        let mut sampler = Sampler::new(params.seed);
        let eos = self.cfg.codec_eos_token_id;
        let trailing_len = prompt.trailing_text_hidden.dim(1)?;

        // Prefill: the whole prompt in one forward. `past_hidden` is the last position's
        // hidden state — the reference's `past_hidden = hidden_states[:, -1:, :]`, which is
        // both the code predictor's first prefill token and the source of this frame's
        // group-0 logits.
        let h = self.talker.forward(&prompt.inputs_embeds)?;
        let mut past_hidden = last_position(&h)?;
        let mut logits = self.talker.codec_logits(&h)?;

        let mut frames: Vec<Vec<u32>> = Vec::new();
        let mut drawn: Vec<u32> = Vec::new();
        let mut stopped_on_eos = false;

        loop {
            // The cap is checked before drawing as well as after pushing, so a cap of 0
            // yields zero frames rather than one.
            if frames.len() >= params.max_new_frames {
                break;
            }

            // --- code group 0, through HF's processor list in HF's order ----------------
            let mut lg: Vec<f32> = logits.flatten_all()?.to_vec1()?;
            sampling::apply_repetition_penalty(&mut lg, &drawn, params.talker.repetition_penalty);
            if frames.len() < params.min_new_frames {
                // MinNewTokensLengthLogitsProcessor: EOS is unreachable until enough frames
                // exist. The reference pins min_new_tokens = 2.
                sampling::block_ids(&mut lg, &[eos]);
            }
            sampling::block_ids(&mut lg, &self.suppress);
            let c0 = sampler.sample(&mut lg, &params.talker);

            if c0 == eos {
                stopped_on_eos = true;
                break;
            }
            drawn.push(c0);

            // --- the 15 residual groups, one frame of predictor context -----------------
            let frame = self.predict_frame(&past_hidden, c0, &mut sampler, &params.code_predictor)?;
            frames.push(frame);

            // --- feedback: Σ of the frame's 16 embeddings + this frame's text hidden ----
            let sum = self.embed_frame(frames.last().expect("just pushed"))?;
            // `generation_step` for the frame just emitted is its 0-based index.
            let step = frames.len() - 1;
            let text = if step < trailing_len {
                prompt.trailing_text_hidden.narrow(1, step, 1)?
            } else {
                prompt.tts_pad_embed.clone()
            };
            let next = sum.broadcast_add(&text)?;

            let h = self.talker.forward(&next)?;
            past_hidden = last_position(&h)?;
            logits = self.talker.codec_logits(&h)?;
        }

        Ok(Generated { frames, stopped_on_eos })
    }

    /// The fast AR: groups `1..num_code_groups` for one frame, given the talker's hidden
    /// state and the frame's group-0 code. Returns all `num_code_groups` codes, index 0
    /// being `c0`.
    fn predict_frame(
        &mut self,
        past_hidden: &Tensor,
        c0: u32,
        sampler: &mut Sampler,
        params: &SamplingParams,
    ) -> Result<Vec<u32>> {
        // The predictor's context is ONE frame. Without this the next frame's prefill lands
        // at cache offset `num_code_groups`, past the end of its RoPE table.
        self.predictor.reset();

        let n_res = self.cfg.code_predictor_heads();
        let mut frame = Vec::with_capacity(1 + n_res);
        frame.push(c0);

        let x = self.predictor_prefill(past_hidden, c0)?;
        let x = self.predictor.bridge(&x)?;
        let mut h = self.predictor.forward(&x)?;

        for i in 0..n_res {
            // lm_head.{i} on the last position predicts group i + 1.
            let mut lg: Vec<f32> = self.predictor.group_logits(i, &h)?.flatten_all()?.to_vec1()?;
            let code = sampler.sample(&mut lg, params);
            frame.push(code);
            if i + 1 == n_res {
                break;
            }
            // codec_embedding.{i} embeds the group i + 1 code just drawn.
            let e = self.predictor.embed_group(i, code)?;
            let e = self.predictor.bridge(&e)?;
            h = self.predictor.forward(&e)?;
        }
        Ok(frame)
    }

    /// The code predictor's two-position prefill, at the **talker's** width and before the
    /// bridge: position 0 is the talker's hidden state, position 1 is group 0's
    /// talker-side embedding.
    ///
    /// The order is the reference's `cat((past_hidden, last_id_hidden), dim=1)` and is
    /// load-bearing twice over: it decides which of the two carries RoPE position 0, and —
    /// because the stack is causal — which one can see the other. Swapping them still
    /// produces 16 plausible codes per frame, which is the failure mode worth a named
    /// helper and a test of its own.
    fn predictor_prefill(&self, past_hidden: &Tensor, c0: u32) -> Result<Tensor> {
        let e0 = self.talker.embed_codec(&[c0])?;
        Tensor::cat(&[past_hidden, &e0], 1)
    }

    /// Realise a symbolic [`crate::prompt::PromptPlan`] into the tensors
    /// [`Qwen3Tts::generate`] consumes.
    ///
    /// `prompt.rs` deliberately stops at a plan of `(text addend, codec addend)` pairs —
    /// "the model crate turns each step into one row of `inputs_embeds`" — because only
    /// this module holds both embedding stacks. Each row is the **sum** of its two
    /// addends, an absent addend contributing zero.
    ///
    /// * `speaker` is the `[1, speaker_enc_dim]` x-vector from
    ///   [`crate::speaker::SpeakerEncoder::embed`] (any shape with that many elements is
    ///   accepted and reshaped), required by any plan carrying a
    ///   [`CodecSlot::SpeakerVector`]. The reference `view(1, 1, -1)`s it straight into the
    ///   codec stream, so `speaker_enc_dim` must equal the talker's hidden size (it does on
    ///   every published checkpoint: 1024 on the 0.6B, 2048 on the 1.7B).
    /// * `ref_frames[i]` is reference frame `i`'s `num_code_groups` codes, for the
    ///   in-context clone path's [`CodecSlot::RefFrame`].
    pub fn realize_plan(
        &self,
        plan: &crate::prompt::PromptPlan,
        speaker: Option<&Tensor>,
        ref_frames: &[Vec<u32>],
    ) -> Result<TalkerPrompt> {
        use crate::prompt::{CodecSlot, TextSlot};

        let hidden = self.cfg.talker.hidden_size;
        let dev = self.device();
        let dt = self.dtype();
        let zero = || Tensor::zeros((1, 1, hidden), dt, &dev);

        let text_row = |slot: TextSlot| -> Result<Tensor> {
            match slot {
                TextSlot::Id(id) => self.talker.embed_text(&[id]),
                TextSlot::None => zero(),
            }
        };

        let mut rows: Vec<Tensor> = Vec::with_capacity(plan.steps.len());
        for (i, step) in plan.steps.iter().enumerate() {
            let t = text_row(step.text)?;
            let c = match step.codec {
                CodecSlot::Id(id) => self.talker.embed_codec(&[id])?,
                CodecSlot::None => zero()?,
                CodecSlot::SpeakerVector => {
                    let s = speaker.ok_or_else(|| {
                        candle_core::Error::Msg(format!(
                            "prompt step {i} wants the speaker vector, but none was supplied"
                        ))
                    })?;
                    s.reshape((1, 1, hidden))?.to_dtype(dt)?.to_device(&dev)?
                }
                CodecSlot::RefFrame(f) => {
                    let frame = ref_frames.get(f).ok_or_else(|| {
                        candle_core::Error::Msg(format!(
                            "prompt step {i} wants reference frame {f}, but only {} were supplied",
                            ref_frames.len()
                        ))
                    })?;
                    self.embed_frame(frame)?
                }
            };
            rows.push((t + c)?);
        }
        if rows.is_empty() {
            return Err(candle_core::Error::Msg("prompt plan has no steps".into()));
        }

        let trailing: Vec<Tensor> = plan
            .trailing_text
            .iter()
            .map(|&s| text_row(s))
            .collect::<Result<_>>()?;
        // A plan with no trailing text still needs a `[1, 0, hidden]` tensor so the loop's
        // `step < trailing_len` test is simply always false and every frame takes the pad.
        let trailing_text_hidden = if trailing.is_empty() {
            Tensor::zeros((1, 0, hidden), dt, &dev)?
        } else {
            Tensor::cat(&trailing, 1)?
        };

        Ok(TalkerPrompt {
            inputs_embeds: Tensor::cat(&rows, 1)?,
            trailing_text_hidden,
            tts_pad_embed: self.talker.embed_text(&[self.cfg.tts_pad_token_id])?,
        })
    }
}

/// Narrow `[1, t, h]` to its last position, `[1, 1, h]`.
fn last_position(h: &Tensor) -> Result<Tensor> {
    let t = h.dim(1)?;
    h.narrow(1, t - 1, 1)
}

/// The reference's `suppress_tokens`: the **top 1024 ids** of the talker's codec vocabulary,
/// minus the codec EOS.
///
/// Verbatim from `Qwen3TTSForConditionalGeneration.generate`:
///
/// ```python
/// "suppress_tokens": [
///     i for i in range(talker_config.vocab_size - 1024, talker_config.vocab_size)
///     if i not in (talker_config.codec_eos_token_id,)
/// ]
/// ```
///
/// On the published checkpoints that is ids 2048..3071 minus 2150, leaving code group 0 to
/// draw from 0..2047 — exactly the code predictor's (and the RVQ's) 2048-entry codebook.
/// Without it the talker can emit a control id that indexes nothing in the codec.
pub fn suppressed_ids(cfg: &Qwen3TtsConfig) -> Vec<u32> {
    let v = cfg.talker.vocab_size as u32;
    let begin = v.saturating_sub(1024);
    (begin..v).filter(|&i| i != cfg.codec_eos_token_id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- a synthetic checkpoint --------------------------------------------------------
    //
    // Real weights are a GPU-box concern. What IS testable off-box is the loop's
    // bookkeeping: the shapes, the per-frame reset, the stop conditions, the group/table
    // pairing and the feedback assembly. So the tests below build a tiny but *complete*
    // model — every tensor the talker and the predictor actually read, at a geometry the
    // config parser accepts — and run the real `generate` over it. Nothing is stubbed; the
    // arithmetic is simply small enough to run on a CPU in a unit test.

    /// Geometry chosen so the boundaries bite: `num_code_groups = 4` means the predictor
    /// uses exactly its 4 RoPE rows per frame (2 prefill + 2 steps), so a missing per-frame
    /// reset fails hard on frame 2 instead of silently degrading. `vocab_size = 1040` puts
    /// the 1024-id suppress block at 16..1040, leaving codes 0..15 — the predictor's vocab.
    const TINY: &str = r#"{
      "model_type": "qwen3_tts",
      "tts_model_type": "base",
      "tts_pad_token_id": 3,
      "talker_config": {
        "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 4, "intermediate_size": 8,
        "vocab_size": 1040, "rope_theta": 1000000, "rms_norm_eps": 1e-06,
        "max_position_embeddings": 64, "num_code_groups": 4,
        "text_embed_dim": 8, "text_vocab_size": 32,
        "codec_bos_id": 17, "codec_eos_token_id": 20, "codec_pad_id": 16,
        "code_predictor_config": {
          "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2,
          "num_key_value_heads": 1, "head_dim": 4, "intermediate_size": 8,
          "vocab_size": 16, "rope_theta": 1000000, "rms_norm_eps": 1e-06,
          "max_position_embeddings": 64, "num_code_groups": 4
        }
      }
    }"#;

    fn cfg() -> Qwen3TtsConfig {
        Qwen3TtsConfig::from_json(TINY).unwrap()
    }

    /// A deterministic small tensor — a fixed low-amplitude pattern, so the forward pass is
    /// numerically tame and the run reproduces exactly across machines.
    fn det(shape: &[usize], salt: u64) -> Tensor {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n)
            .map(|i| {
                let x = (i as u64).wrapping_mul(2_654_435_761).wrapping_add(salt.wrapping_mul(97));
                ((x % 1000) as f32 / 1000.0) - 0.5
            })
            .collect();
        Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
    }

    /// Every tensor one Qwen3 decoder stack reads, at `prefix`.
    fn decoder(map: &mut HashMap<String, Tensor>, prefix: &str, c: &crate::config::TransformerConfig, salt: u64) {
        let h = c.hidden_size;
        let q = c.num_attention_heads * c.head_dim;
        let kv = c.num_key_value_heads * c.head_dim;
        for l in 0..c.num_hidden_layers {
            let p = format!("{prefix}.layers.{l}");
            let s = salt + l as u64;
            map.insert(format!("{p}.input_layernorm.weight"), det(&[h], s + 1));
            map.insert(format!("{p}.post_attention_layernorm.weight"), det(&[h], s + 2));
            map.insert(format!("{p}.self_attn.q_proj.weight"), det(&[q, h], s + 3));
            map.insert(format!("{p}.self_attn.k_proj.weight"), det(&[kv, h], s + 4));
            map.insert(format!("{p}.self_attn.v_proj.weight"), det(&[kv, h], s + 5));
            map.insert(format!("{p}.self_attn.o_proj.weight"), det(&[h, q], s + 6));
            map.insert(format!("{p}.self_attn.q_norm.weight"), det(&[c.head_dim], s + 7));
            map.insert(format!("{p}.self_attn.k_norm.weight"), det(&[c.head_dim], s + 8));
            map.insert(format!("{p}.mlp.gate_proj.weight"), det(&[c.intermediate_size, h], s + 9));
            map.insert(format!("{p}.mlp.up_proj.weight"), det(&[c.intermediate_size, h], s + 10));
            map.insert(format!("{p}.mlp.down_proj.weight"), det(&[h, c.intermediate_size], s + 11));
        }
        map.insert(format!("{prefix}.norm.weight"), det(&[h], salt + 12));
    }

    fn tiny_model(c: Qwen3TtsConfig) -> Qwen3Tts {
        let t = c.talker.clone();
        let cp = c.code_predictor.clone();

        let mut talker: HashMap<String, Tensor> = HashMap::new();
        decoder(&mut talker, "talker.model", &t, 100);
        talker.insert("talker.model.codec_embedding.weight".into(), det(&[t.vocab_size, t.hidden_size], 200));
        talker.insert("talker.codec_head.weight".into(), det(&[t.vocab_size, t.hidden_size], 300));
        // The text side: a wide `text_embedding` narrowed to the model width by the
        // two-layer, biased `text_projection`. `generate` never touches these (it takes
        // `inputs_embeds` already assembled) but `realize_plan` does.
        talker.insert(
            "talker.model.text_embedding.weight".into(),
            det(&[c.text_vocab_size, c.text_embed_dim], 310),
        );
        talker.insert("talker.text_projection.linear_fc1.weight".into(), det(&[c.text_embed_dim, c.text_embed_dim], 320));
        talker.insert("talker.text_projection.linear_fc1.bias".into(), det(&[c.text_embed_dim], 330));
        talker.insert("talker.text_projection.linear_fc2.weight".into(), det(&[t.hidden_size, c.text_embed_dim], 340));
        talker.insert("talker.text_projection.linear_fc2.bias".into(), det(&[t.hidden_size], 350));

        let mut pred: HashMap<String, Tensor> = HashMap::new();
        decoder(&mut pred, "talker.code_predictor.model", &cp, 400);
        for i in 0..c.code_predictor_heads() {
            pred.insert(
                format!("talker.code_predictor.model.codec_embedding.{i}.weight"),
                // NOTE the width: the predictor's tables are at the TALKER's hidden size.
                det(&[cp.vocab_size, t.hidden_size], 500 + i as u64),
            );
            pred.insert(
                format!("talker.code_predictor.lm_head.{i}.weight"),
                det(&[cp.vocab_size, cp.hidden_size], 600 + i as u64),
            );
        }

        Qwen3Tts::from_maps(talker, pred, c, Device::Cpu, DType::F32).unwrap()
    }

    fn prompt(hidden: usize, prompt_len: usize, trailing: usize) -> TalkerPrompt {
        TalkerPrompt {
            inputs_embeds: det(&[1, prompt_len, hidden], 700),
            trailing_text_hidden: det(&[1, trailing, hidden], 800),
            tts_pad_embed: det(&[1, 1, hidden], 900),
        }
    }

    /// Greedy-ish: a near-zero temperature makes the run a pure function of the weights, so
    /// the EOS and cap tests below can pin exact behaviour without hardcoding a code.
    fn greedy() -> SamplingParams {
        SamplingParams { temperature: 0.01, top_p: 1.0, top_k: 1, repetition_penalty: 1.0 }
    }

    // ---- suppress_tokens ---------------------------------------------------------------

    /// The suppress block is the top 1024 ids minus EOS — checked on both sides of its lower
    /// edge, and with EOS explicitly spared (masking it would make the model unstoppable).
    #[test]
    fn suppressed_ids_are_the_top_1024_minus_eos() {
        let c = cfg();
        let s = suppressed_ids(&c);
        assert_eq!(s.len(), 1023, "1024 ids, minus the one EOS");
        assert_eq!(s[0], 16, "block starts at vocab_size - 1024");
        assert_eq!(*s.last().unwrap(), 1039, "and runs to vocab_size - 1");
        assert!(!s.contains(&15), "id 15 is below the block");
        assert!(s.contains(&16), "id 16 is the first in the block");
        assert!(!s.contains(&c.codec_eos_token_id), "EOS must stay reachable");
        assert!(s.contains(&19) && s.contains(&21), "…but its neighbours must not");
    }

    /// The published geometry, so a change to the rule is caught against the real numbers.
    #[test]
    fn suppressed_ids_on_the_published_geometry() {
        let mut c = cfg();
        c.talker.vocab_size = 3072;
        c.codec_eos_token_id = 2150;
        let s = suppressed_ids(&c);
        assert_eq!(s.len(), 1023);
        assert_eq!(s[0], 2048);
        assert_eq!(*s.last().unwrap(), 3071);
        assert!(!s.contains(&2047), "the 2048 usable codes stay free");
        assert!(!s.contains(&2150));
    }

    // ---- the loop ----------------------------------------------------------------------

    /// A run must produce full frames of `num_code_groups` codes, with group 0 drawn from
    /// the unsuppressed talker range and the residual groups from the predictor's vocab.
    #[test]
    fn every_frame_is_complete_and_in_range() {
        let c = cfg();
        let mut m = tiny_model(c.clone());
        let p = DriveParams { max_new_frames: 6, ..Default::default() };
        let g = m.generate(&prompt(c.talker.hidden_size, 3, 2), &p).unwrap();

        assert!(!g.is_empty(), "the loop produced nothing");
        assert!(g.len() <= 6);
        for f in &g.frames {
            assert_eq!(f.len(), c.num_code_groups, "a frame is num_code_groups codes");
            // group 0 escaped the suppress block and is not EOS
            assert!(f[0] < 16, "group 0 leaked into the suppressed block: {}", f[0]);
            assert_ne!(f[0], c.codec_eos_token_id);
            for &code in &f[1..] {
                assert!((code as usize) < c.code_predictor.vocab_size, "residual out of vocab");
            }
        }
    }

    /// Generating more than one frame at all is the proof that the predictor's cache is
    /// reset per frame: its RoPE table is exactly `num_code_groups` rows long, so a second
    /// frame prefilling at offset 4 would fail to narrow the table and error out.
    #[test]
    fn the_predictor_cache_is_reset_every_frame() {
        let c = cfg();
        let mut m = tiny_model(c.clone());
        let p = DriveParams { max_new_frames: 5, min_new_frames: 5, ..Default::default() };
        let g = m.generate(&prompt(c.talker.hidden_size, 3, 2), &p).unwrap();
        assert_eq!(g.len(), 5, "five frames means five clean predictor prefills");
    }

    /// `max_new_frames` is a hard cap, and it stops the run *without* an EOS.
    #[test]
    fn max_new_frames_caps_the_run() {
        let c = cfg();
        let hidden = c.talker.hidden_size;
        for cap in [1usize, 2, 4] {
            let mut m = tiny_model(cfg());
            // min_new_frames >= cap keeps EOS masked for the whole run, so the cap is the
            // only thing that can end it.
            let p = DriveParams { max_new_frames: cap, min_new_frames: cap, ..Default::default() };
            let g = m.generate(&prompt(hidden, 3, 2), &p).unwrap();
            assert_eq!(g.len(), cap, "cap {cap}");
            assert!(!g.stopped_on_eos, "cap {cap} must not report an EOS stop");
        }
        // Zero is a real cap, not "one frame anyway" — the boundary just below the first
        // useful value, and the only case the pre-draw check catches.
        let mut m = tiny_model(cfg());
        let p = DriveParams { max_new_frames: 0, min_new_frames: 0, ..Default::default() };
        let g = m.generate(&prompt(hidden, 3, 2), &p).unwrap();
        assert!(g.is_empty());
        assert!(!g.stopped_on_eos);
    }

    /// The repetition penalty must actually reach the group-0 draw. It only bites once an
    /// id repeats, so this runs long enough for that to happen and asserts that a strong
    /// penalty changes the codes a penalty-free run produces.
    #[test]
    fn the_repetition_penalty_reaches_the_talker_draw() {
        let hidden = cfg().talker.hidden_size;
        let run = |penalty: f32| {
            let mut m = tiny_model(cfg());
            let p = DriveParams {
                max_new_frames: 5,
                min_new_frames: 5,
                talker: SamplingParams { repetition_penalty: penalty, ..greedy() },
                code_predictor: greedy(),
                ..Default::default()
            };
            m.generate(&prompt(hidden, 3, 2), &p).unwrap().frames
        };
        let free = run(1.0);
        // The penalty-free greedy run must actually repeat a group-0 code, or the test
        // would be vacuous.
        let mut seen = std::collections::BTreeSet::new();
        assert!(
            free.iter().any(|f| !seen.insert(f[0])),
            "no group-0 code repeated, so the penalty could not bite: {free:?}"
        );
        assert_ne!(run(5.0), free, "a strong repetition penalty must change the run");
        assert_eq!(run(1.0), free, "and 1.0 must remain the no-penalty run");
    }

    /// Each head must be driven by ITS OWN knobs: `params.talker` for code group 0,
    /// `params.code_predictor` for groups 1.. . The reference keeps them separate
    /// (`top_k`/`top_p`/`temperature` versus `subtalker_*`), and a loop that fed one set to
    /// both would still generate perfectly well-formed frames.
    #[test]
    fn each_head_is_driven_by_its_own_sampling_params() {
        let hidden = cfg().talker.hidden_size;
        let run = |talker: SamplingParams, code_predictor: SamplingParams| {
            let mut m = tiny_model(cfg());
            let p = DriveParams {
                max_new_frames: 3,
                min_new_frames: 3,
                talker,
                code_predictor,
                ..Default::default()
            };
            m.generate(&prompt(hidden, 3, 2), &p).unwrap().frames
        };
        let hot = SamplingParams { temperature: 40.0, top_p: 1.0, top_k: 0, repetition_penalty: 1.0 };

        // Changing ONLY the predictor's knobs must change the run — but not group 0 of
        // frame 0, which is drawn before the predictor has run at all.
        let base = run(greedy(), greedy());
        let alt_cp = run(greedy(), hot.clone());
        assert_ne!(alt_cp, base, "the predictor's knobs must reach the residual draws");
        assert_eq!(alt_cp[0][0], base[0][0], "frame 0's group 0 predates any predictor draw");

        // And changing ONLY the talker's knobs must change group 0 of frame 0.
        let alt_talker = run(hot, greedy());
        assert_ne!(alt_talker[0][0], base[0][0], "the talker's knobs must reach group 0");
    }

    /// A golden pin over the whole loop on the synthetic checkpoint.
    ///
    /// This is a **regression** pin, not a parity claim: the weights are the deterministic
    /// fixture above, so the numbers mean nothing about real audio. What it defends is
    /// everything the structural tests can only reach indirectly — the per-group table
    /// pairing inside the residual chain, the exact prefill contents, the RoPE positions
    /// the predictor walks through, and the order the shared PRNG is consumed in. Any of
    /// those can be permuted while still producing well-formed frames, which is precisely
    /// the class of bug that cost the Fish port three frames of noise.
    ///
    /// If this changes, the change was either intentional or a bug — say which.
    #[test]
    fn the_loop_is_pinned_end_to_end_on_the_synthetic_checkpoint() {
        let hidden = cfg().talker.hidden_size;
        let mut m = tiny_model(cfg());
        let p = DriveParams {
            seed: 0,
            max_new_frames: 4,
            min_new_frames: 4,
            talker: SamplingParams { temperature: 1.0, top_p: 0.95, top_k: 8, repetition_penalty: 1.05 },
            code_predictor: SamplingParams { temperature: 1.0, top_p: 0.95, top_k: 8, repetition_penalty: 1.0 },
        };
        let g = m.generate(&prompt(hidden, 3, 2), &p).unwrap();
        assert_eq!(g.frames, GOLDEN_FRAMES.iter().map(|f| f.to_vec()).collect::<Vec<_>>());
    }

    const GOLDEN_FRAMES: [[u32; 4]; 4] =
        [[13, 9, 0, 11], [2, 6, 4, 14], [5, 15, 6, 9], [6, 9, 12, 7]];

    /// The same seed replays exactly; a different seed does not.
    #[test]
    fn a_run_is_reproducible_from_its_seed() {
        let c = cfg();
        let hidden = c.talker.hidden_size;
        let run = |seed: u64| {
            let mut m = tiny_model(cfg());
            let p = DriveParams {
                seed,
                max_new_frames: 6,
                min_new_frames: 6,
                talker: SamplingParams { temperature: 1.0, top_p: 1.0, top_k: 0, repetition_penalty: 1.0 },
                code_predictor: SamplingParams { temperature: 1.0, top_p: 1.0, top_k: 0, repetition_penalty: 1.0 },
            };
            m.generate(&prompt(hidden, 3, 2), &p).unwrap()
        };
        assert_eq!(run(0), run(0), "seed 0 must replay identically");
        assert_ne!(run(0), run(1), "seed 1 must diverge");
    }

    /// The EOS path: when the code the loop would draw first IS the EOS id, the run stops
    /// with zero frames — and raising `min_new_frames` above zero masks that same EOS and
    /// forces frames out instead. Both sides of the min-new-tokens guard, without pinning a
    /// code that the weights happen to produce.
    #[test]
    fn eos_stops_the_run_and_min_new_frames_defers_it() {
        let base = cfg();
        let hidden = base.talker.hidden_size;
        let params = |min: usize| DriveParams {
            max_new_frames: 4,
            min_new_frames: min,
            talker: greedy(),
            code_predictor: greedy(),
            ..Default::default()
        };

        // What does a greedy run draw first?
        let mut m = tiny_model(base.clone());
        let first = m.generate(&prompt(hidden, 3, 2), &params(0)).unwrap().frames[0][0];

        // Re-point the EOS at exactly that code. `first < 16` sits below the suppress block,
        // so the block itself is unchanged and only the stop id moves.
        let mut c = base.clone();
        assert!(first < 16, "greedy draw must land in the usable range, got {first}");
        c.codec_eos_token_id = first;

        // min_new_frames = 0 → EOS is reachable on the very first draw → nothing generated.
        let mut m = tiny_model(c.clone());
        let g = m.generate(&prompt(hidden, 3, 2), &params(0)).unwrap();
        assert!(g.is_empty(), "an immediate EOS must yield no frames");
        assert!(g.stopped_on_eos);

        // min_new_frames = 2 → that same EOS is masked for the first two draws.
        let mut m = tiny_model(c);
        let g = m.generate(&prompt(hidden, 3, 2), &params(2)).unwrap();
        assert!(g.len() >= 2, "the guard must hold the EOS off for two frames, got {}", g.len());
        assert!(g.frames.iter().take(2).all(|f| f[0] != first));
    }

    /// `embed_frame` must sum the talker's group-0 table with the predictor's per-group
    /// tables — never the same table 16 times, and never group `i`'s code through table
    /// `i` (the pairing is `codec_embedding.{i}` ↔ group `i + 1`).
    #[test]
    fn embed_frame_sums_the_talker_and_per_group_tables() {
        let c = cfg();
        let m = tiny_model(c.clone());
        let frame = vec![3u32, 5, 7, 9];
        let got = m.embed_frame(&frame).unwrap();
        assert_eq!(got.dims(), &[1, 1, c.talker.hidden_size]);

        let mut want = m.talker().embed_codec(&[3]).unwrap();
        want = (want + m.code_predictor().embed_group(0, 5).unwrap()).unwrap();
        want = (want + m.code_predictor().embed_group(1, 7).unwrap()).unwrap();
        want = (want + m.code_predictor().embed_group(2, 9).unwrap()).unwrap();
        let a: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(a, b);

        // A short frame is an error, not a silent partial sum.
        assert!(m.embed_frame(&[1, 2]).is_err());
    }

    /// `group_rows` must transpose frame-major codes into one row per RVQ layer, which is
    /// the layout `Rvq::decode` consumes. A transposition slip here is inaudible-but-wrong.
    #[test]
    fn group_rows_transposes_to_one_row_per_rvq_layer() {
        let g = Generated {
            frames: vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]],
            stopped_on_eos: true,
        };
        assert_eq!(
            g.group_rows(4),
            vec![vec![1, 5], vec![2, 6], vec![3, 7], vec![4, 8]]
        );
        assert_eq!(g.len(), 2);
        assert!(!g.is_empty());
    }

    /// The trailing-text hidden is consumed **one row per frame, starting at row 0**.
    ///
    /// Testing "a different trailing block changes the run" is not enough — an off-by-one
    /// that reads `trailing[j + 1]` for frame `j` passes it. So this perturbs one row at a
    /// time: with three frames the loop feeds back twice, reading exactly rows 0 and 1, so
    /// perturbing row 0 or row 1 MUST change the run and perturbing row 2 MUST NOT. That
    /// pins both the start index and the number of rows consumed.
    #[test]
    fn the_trailing_text_hidden_is_consumed_one_row_per_frame_from_row_zero() {
        let hidden = cfg().talker.hidden_size;
        // Three frames => feedback after frames 0 and 1 => rows 0 and 1 are read, row 2 is
        // not (the run is capped before frame 2's feedback).
        let run = |perturb: Option<usize>| {
            let mut m = tiny_model(cfg());
            let p = DriveParams {
                max_new_frames: 3,
                min_new_frames: 3,
                talker: greedy(),
                code_predictor: greedy(),
                ..Default::default()
            };
            let base = det(&[1, 3, hidden], 800);
            let trailing = match perturb {
                None => base,
                Some(row) => {
                    let mut parts: Vec<Tensor> = (0..3)
                        .map(|r| base.narrow(1, r, 1).unwrap())
                        .collect();
                    // A large perturbation, not a subtle one: a greedy argmax is a step
                    // function, so a small nudge can leave the drawn code unchanged and
                    // make the test lie about which row was read.
                    parts[row] = Tensor::full(6.0f32, (1, 1, hidden), &Device::Cpu).unwrap();
                    Tensor::cat(&parts, 1).unwrap()
                }
            };
            let pr = TalkerPrompt {
                inputs_embeds: det(&[1, 3, hidden], 700),
                trailing_text_hidden: trailing,
                tts_pad_embed: det(&[1, 1, hidden], 900),
            };
            m.generate(&pr, &p).unwrap().frames
        };
        let base = run(None);
        assert_eq!(base.len(), 3);
        assert_ne!(run(Some(0)), base, "frame 0's feedback must read trailing row 0");
        assert_ne!(run(Some(1)), base, "frame 1's feedback must read trailing row 1");
        assert_eq!(run(Some(2)), base, "row 2 is past the last feedback and must be unread");
    }

    /// Once the trailing block is exhausted the pad embedding takes over — the other side
    /// of `step < trailing_len`. With an empty trailing block every frame takes the pad, so
    /// changing the pad must change the run.
    #[test]
    fn the_pad_embed_takes_over_when_the_trailing_block_runs_out() {
        let hidden = cfg().talker.hidden_size;
        let run = |trailing_len: usize, pad_salt: u64| {
            let mut m = tiny_model(cfg());
            let p = DriveParams {
                max_new_frames: 3,
                min_new_frames: 3,
                talker: greedy(),
                code_predictor: greedy(),
                ..Default::default()
            };
            let pr = TalkerPrompt {
                inputs_embeds: det(&[1, 3, hidden], 700),
                trailing_text_hidden: det(&[1, 3, hidden], 800)
                    .narrow(1, 0, trailing_len)
                    .unwrap(),
                tts_pad_embed: det(&[1, 1, hidden], pad_salt),
            };
            m.generate(&pr, &p).unwrap().frames
        };
        // Empty trailing block: both feedbacks take the pad, so the pad is load-bearing.
        assert_ne!(run(0, 900), run(0, 901));
        // Trailing rows 0 and 1 both present: neither feedback reaches the pad, so changing
        // it must be invisible. The boundary, from both sides.
        assert_eq!(run(2, 900), run(2, 901));
    }

    /// The code predictor's prefill is `[talker hidden, group-0 embedding]`, in that order.
    /// Reversed it still yields 16 plausible codes per frame, so nothing downstream would
    /// notice — hence a direct assertion on the two rows.
    #[test]
    fn the_predictor_prefill_is_hidden_then_group_zero() {
        let c = cfg();
        let hidden = c.talker.hidden_size;
        let m = tiny_model(c);
        let past = det(&[1, 1, hidden], 4242);
        let got = m.predictor_prefill(&past, 6).unwrap();
        assert_eq!(got.dims(), &[1, 2, hidden]);

        let row = |t: &Tensor, i: usize| -> Vec<f32> {
            t.narrow(1, i, 1).unwrap().flatten_all().unwrap().to_vec1().unwrap()
        };
        assert_eq!(row(&got, 0), row(&past, 0), "position 0 is the talker hidden state");
        assert_eq!(
            row(&got, 1),
            row(&m.talker().embed_codec(&[6]).unwrap(), 0),
            "position 1 is group 0 through the TALKER's codec_embedding"
        );
    }

    // ---- realising a symbolic prompt plan ----------------------------------------------

    /// Each plan step becomes one row that is the SUM of its two addends, an absent addend
    /// contributing zero — and the two addends come from different tables (text goes
    /// through `text_embedding` + `text_projection`, codec through `codec_embedding`).
    #[test]
    fn realize_plan_sums_the_text_and_codec_addends_per_step() {
        use crate::prompt::{CodecSlot, PromptPlan, PromptStep, TextSlot};

        let c = cfg();
        let hidden = c.talker.hidden_size;
        let m = tiny_model(c.clone());
        let plan = PromptPlan {
            steps: vec![
                // text only (the role prefix)
                PromptStep { text: TextSlot::Id(5), codec: CodecSlot::None },
                // both streams (the body)
                PromptStep { text: TextSlot::Id(7), codec: CodecSlot::Id(9) },
                // a reference frame in the codec stream
                PromptStep { text: TextSlot::Id(7), codec: CodecSlot::RefFrame(0) },
            ],
            trailing_text: vec![TextSlot::Id(11), TextSlot::Id(13)],
            ref_frames: 1,
        };
        let refs = vec![vec![1u32, 2, 3, 4]];
        let p = m.realize_plan(&plan, None, &refs).unwrap();

        assert_eq!(p.inputs_embeds.dims(), &[1, 3, hidden]);
        assert_eq!(p.trailing_text_hidden.dims(), &[1, 2, hidden]);
        assert_eq!(p.tts_pad_embed.dims(), &[1, 1, hidden]);

        let row = |t: &Tensor, i: usize| -> Vec<f32> {
            t.narrow(1, i, 1).unwrap().flatten_all().unwrap().to_vec1().unwrap()
        };
        let flat = |t: &Tensor| -> Vec<f32> { t.flatten_all().unwrap().to_vec1().unwrap() };

        // Step 0: text only, so the row IS the text embedding — not the text embedding plus
        // some stray codec row.
        assert_eq!(row(&p.inputs_embeds, 0), flat(&m.talker().embed_text(&[5]).unwrap()));
        // Step 1: the sum of both addends, and demonstrably not either one alone.
        let want = (m.talker().embed_text(&[7]).unwrap() + m.talker().embed_codec(&[9]).unwrap()).unwrap();
        assert_eq!(row(&p.inputs_embeds, 1), flat(&want));
        assert_ne!(row(&p.inputs_embeds, 1), flat(&m.talker().embed_text(&[7]).unwrap()));
        // Step 2: the codec addend is the summed 16-group reference frame.
        let want = (m.talker().embed_text(&[7]).unwrap() + m.embed_frame(&refs[0]).unwrap()).unwrap();
        assert_eq!(row(&p.inputs_embeds, 2), flat(&want));
        // Trailing rows are text-only, in plan order.
        assert_eq!(row(&p.trailing_text_hidden, 0), flat(&m.talker().embed_text(&[11]).unwrap()));
        assert_eq!(row(&p.trailing_text_hidden, 1), flat(&m.talker().embed_text(&[13]).unwrap()));
        // The pad embed is the config's tts_pad_token_id through the text path.
        assert_eq!(
            flat(&p.tts_pad_embed),
            flat(&m.talker().embed_text(&[c.tts_pad_token_id]).unwrap())
        );
    }

    /// A plan that needs an input the caller did not supply must be a named error, not a
    /// silently-zeroed row.
    #[test]
    fn realize_plan_rejects_missing_speaker_and_reference_frames() {
        use crate::prompt::{CodecSlot, PromptPlan, PromptStep, TextSlot};

        let m = tiny_model(cfg());
        let plan = |codec: CodecSlot| PromptPlan {
            steps: vec![PromptStep { text: TextSlot::None, codec }],
            trailing_text: vec![],
            ref_frames: 0,
        };
        let e = m.realize_plan(&plan(CodecSlot::SpeakerVector), None, &[]).unwrap_err();
        assert!(e.to_string().contains("speaker vector"), "{e}");
        let e = m.realize_plan(&plan(CodecSlot::RefFrame(0)), None, &[]).unwrap_err();
        assert!(e.to_string().contains("reference frame 0"), "{e}");
        // Supplied, it succeeds — so the errors are about the missing input, not the slot.
        let spk = det(&[1, 1, cfg().talker.hidden_size], 55);
        assert!(m.realize_plan(&plan(CodecSlot::SpeakerVector), Some(&spk), &[]).is_ok());
        assert!(m
            .realize_plan(&plan(CodecSlot::RefFrame(0)), None, &[vec![1, 2, 3, 4]])
            .is_ok());
        // And an empty plan is an error rather than a zero-length prefill.
        let empty = PromptPlan { steps: vec![], trailing_text: vec![], ref_frames: 0 };
        assert!(m.realize_plan(&empty, None, &[]).is_err());
    }

    /// An empty trailing block realises to `[1, 0, hidden]`, so every frame takes the pad.
    #[test]
    fn realize_plan_handles_an_empty_trailing_block() {
        use crate::prompt::{CodecSlot, PromptPlan, PromptStep, TextSlot};

        let c = cfg();
        let m = tiny_model(c.clone());
        let plan = PromptPlan {
            steps: vec![PromptStep { text: TextSlot::Id(5), codec: CodecSlot::None }],
            trailing_text: vec![],
            ref_frames: 0,
        };
        let p = m.realize_plan(&plan, None, &[]).unwrap();
        assert_eq!(p.trailing_text_hidden.dims(), &[1, 0, c.talker.hidden_size]);
    }

    /// The defaults are the reference's: seed pinned, the shipped `max_new_tokens`, and the
    /// hardcoded `min_new_tokens = 2`.
    #[test]
    fn drive_defaults_match_the_reference() {
        let d = DriveParams::default();
        assert_eq!(d.seed, 0);
        assert_eq!(d.max_new_frames, 8192);
        assert_eq!(d.min_new_frames, 2);
        assert_eq!(d.talker, SamplingParams::talker());
        assert_eq!(d.code_predictor, SamplingParams::code_predictor());
    }
}
