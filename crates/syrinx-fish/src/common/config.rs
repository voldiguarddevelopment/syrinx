//! [`FishConfig`] — the variant-agnostic configuration for the dual-AR + RVQ stack.
//!
//! Three sub-configs:
//!   * [`TransformerConfig`] — the **slow** AR backbone (Llama for s1, Qwen3-4B for s2).
//!   * [`FastArConfig`] — the **fast** AR head that expands one semantic token into the
//!     residual RVQ codes for that frame.
//!   * [`CodecConfig`] — the RVQ codec geometry (codebook count/size, sample rate, hop).
//!
//! [`FishConfig::s1_mini`] / [`FishConfig::s2_pro`] fill these from the verified
//! reference numbers. Fields whose exact value must be confirmed against the on-box
//! `config.json` / checkpoint carry a `// PARITY:` flag and a best-effort default —
//! never a fake value. [`FishConfig::from_fish_json`] loads a real Fish `config.json`
//! (the `dual_ar` and `fish_qwen3_omni` layouts), mirroring the reference
//! `BaseModelArgs.from_pretrained`.

use serde::{Deserialize, Serialize};

use crate::FishVariant;

/// One autoregressive transformer's shape — the schema is shared by the slow backbone
/// and (via [`FastArConfig`]) the fast head. Mirrors the reference `BaseModelArgs`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransformerConfig {
    /// Model (residual stream) width.
    pub dim: usize,
    /// Number of decoder layers.
    pub n_layer: usize,
    /// Number of query attention heads.
    pub n_head: usize,
    /// Number of key/value heads (GQA). Equals `n_head` for full MHA.
    pub n_local_heads: usize,
    /// Per-head dimension (note: for Qwen3 this is **not** `dim / n_head`).
    pub head_dim: usize,
    /// SwiGLU MLP inner width.
    pub intermediate_size: usize,
    /// Token-embedding / output vocabulary size of the slow backbone.
    pub vocab_size: usize,
    /// RoPE base (θ).
    pub rope_base: f64,
    /// RMSNorm epsilon.
    pub norm_eps: f64,
    /// Maximum sequence length (KV-cache / positional bound).
    pub max_seq_len: usize,
    /// Whether `wqkv` carries a bias (Qwen3: true; Llama: false).
    pub attention_qkv_bias: bool,
    /// Whether the attention output projection carries a bias.
    pub attention_o_bias: bool,
    /// Whether per-head QK-RMSNorm is applied before RoPE (Qwen3: true).
    pub attention_qk_norm: bool,
    /// Whether the output head is tied to the input embedding.
    pub tie_word_embeddings: bool,
}

/// The **fast** AR head: a small transformer that, given the slow hidden state for a
/// frame, autoregressively emits the residual RVQ codebook indices for that frame.
///
/// s1 uses per-codebook embedding tables; s2 uses a **single shared embedding table**
/// with the codebook identity carried by the RoPE position (plus MCF fusion of the
/// slow hidden). The driver is agnostic to this; the backend owns it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FastArConfig {
    /// The fast transformer's own shape.
    pub transformer: TransformerConfig,
    /// Whether the slow hidden fed into the fast head is the post-final-RMSNorm output
    /// (`norm_fastlayer_input` in the reference) rather than the pre-norm residual.
    pub norm_fastlayer_input: bool,
}

/// RVQ codec geometry — variant-agnostic contract used by [`super::codec::RvqCodec`]
/// implementations and by the driver to size the `[num_codebooks, T]` code matrix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodecConfig {
    /// Total RVQ codebooks fed to the codec decoder (1 semantic-derived + 9 residual).
    pub num_codebooks: usize,
    /// Cardinality of the **semantic** codebook (the `<|semantic:i|>` range; 4096 in
    /// the reference tokenizer).
    pub semantic_size: usize,
    /// Cardinality of each **residual** RVQ codebook (the fast-AR output width).
    pub residual_size: usize,
    /// Per-codebook latent dimension (the VQ projection width).
    pub codebook_dim: usize,
    /// Output sample rate of the codec decoder (Hz).
    pub sample_rate: u32,
    /// Samples per semantic frame (the codec's effective hop). `sample_rate / frame_hop`
    /// is the ~21.5 Hz frame rate.
    pub frame_hop: usize,
    /// Encoder downsample strides (analysis), outermost first.
    pub encoder_rates: Vec<usize>,
    /// Decoder upsample strides (synthesis), outermost first.
    pub decoder_rates: Vec<usize>,
}

impl CodecConfig {
    /// The semantic frame rate in Hz (`sample_rate / frame_hop`).
    pub fn frame_rate(&self) -> f64 {
        self.sample_rate as f64 / self.frame_hop as f64
    }
}

/// The full variant-agnostic Fish config: slow backbone + fast head + codec, plus the
/// semantic-token range and stop id the driver needs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FishConfig {
    /// Which checkpoint this config describes.
    pub variant: FishVariant,
    /// The slow AR backbone.
    pub slow: TransformerConfig,
    /// The fast AR residual head.
    pub fast: FastArConfig,
    /// The RVQ codec geometry.
    pub codec: CodecConfig,
    /// First slow-vocab id of the contiguous semantic-token range (the `<|semantic:0|>`
    /// id). Injected from the tokenizer at load time in the reference.
    pub semantic_begin_id: u32,
    /// Last slow-vocab id of the semantic-token range (inclusive).
    pub semantic_end_id: u32,
    /// The `<|im_end|>` slow-vocab id — the end-of-utterance stop token.
    pub stop_token_id: u32,
}

impl FishConfig {
    /// Number of RVQ codebooks (the `[num_codebooks, T]` matrix height).
    pub fn num_codebooks(&self) -> usize {
        self.codec.num_codebooks
    }

    /// Config for `openaudio-s1-mini` (0.5B): a Llama-style `DualARTransformer` slow AR,
    /// a 4-layer fast AR, and the modded-DAC codec.
    ///
    /// The dual-AR + RVQ-shaped fields (10 codebooks, 4096-way semantic, 44.1 kHz) are
    /// fixed by the architecture. The slow/fast transformer dimensions and the vocab /
    /// semantic-id layout depend on the published `config.json` + tokenizer and are
    /// flagged `// PARITY:` — confirm them on-box before trusting any numeric output.
    pub fn s1_mini() -> Self {
        // PARITY: confirm the slow backbone shape from openaudio-s1-mini/config.json
        // (model_type "dual_ar"). These are best-effort 0.5B Llama-DualAR values.
        let slow = TransformerConfig {
            dim: 1024,                 // PARITY: confirm from config.json on-box
            n_layer: 24,               // PARITY: confirm from config.json on-box
            n_head: 16,                // PARITY: confirm from config.json on-box
            n_local_heads: 2,          // PARITY: confirm (GQA kv heads) on-box
            head_dim: 64,              // PARITY: confirm from config.json on-box
            intermediate_size: 4096,   // PARITY: confirm from config.json on-box
            vocab_size: 32_768,        // PARITY: confirm tiktoken vocab size on-box
            rope_base: 1_000_000.0,    // PARITY: confirm rope_base on-box
            norm_eps: 1e-6,            // PARITY: confirm norm_eps on-box
            max_seq_len: 4096,         // PARITY: confirm max_seq_len on-box
            attention_qkv_bias: false, // Llama-style: no qkv bias
            attention_o_bias: false,
            attention_qk_norm: false,  // Llama-style: no QK-norm
            tie_word_embeddings: true, // PARITY: confirm tie on-box
        };
        // Fast AR: 4 layers, per-codebook embedding tables, same width as the slow dim
        // unless the checkpoint projects (`fast_project_in`).
        let fast = FastArConfig {
            transformer: TransformerConfig {
                n_layer: 4, // PARITY: confirm n_fast_layer on-box
                // PARITY: confirm the fast head's dim/heads/intermediate on-box; the
                // reference defaults them to the slow values when unset.
                ..slow.clone()
            },
            norm_fastlayer_input: false, // PARITY: confirm norm_fastlayer_input on-box
        };
        let codec = CodecConfig {
            num_codebooks: 10,    // 1 semantic-derived + 9 residual
            semantic_size: 4096,  // `<|semantic:0..4096|>`
            residual_size: 1024,  // PARITY: confirm residual codebook size on-box
            codebook_dim: 8,      // PARITY: confirm VQ codebook_dim on-box
            sample_rate: 44_100,
            frame_hop: 2048,      // modded-DAC frame_length (hop 512 * 4) -> ~21.5 Hz
            encoder_rates: vec![2, 4, 8, 8], // PARITY: confirm modded-DAC strides on-box
            decoder_rates: vec![8, 8, 4, 2], // PARITY: confirm modded-DAC strides on-box
        };
        FishConfig {
            variant: FishVariant::S1Mini,
            slow,
            fast,
            codec,
            // PARITY: the semantic range + im_end id are injected from the tokenizer at
            // load; these placeholders keep the range internally consistent only.
            semantic_begin_id: 0,
            semantic_end_id: 4095,
            stop_token_id: 0, // PARITY: resolve <|im_end|> id from the tokenizer on-box
        }
    }

    /// Config for `s2-pro` (5B): a Qwen3-4B slow AR (`fish_qwen3_omni`), a 4-layer
    /// ~400M fast AR (single shared embedding + RoPE-position codebook identity + MCF
    /// fusion), and the 446M EVA-GAN / causal-DAC codec.
    ///
    /// The slow-backbone numbers are the verified Qwen3-4B values; `head_dim`,
    /// `rope_base`, `tie_word_embeddings`, and the full fast-head shape carry `// PARITY:`
    /// flags where the published `config.json` is the authority.
    pub fn s2_pro() -> Self {
        let slow = TransformerConfig {
            dim: 2560,
            n_layer: 36,
            n_head: 32,
            n_local_heads: 8, // GQA
            head_dim: 128,    // PARITY: Qwen3-4B uses head_dim 128 (≠ dim/n_head); confirm
            intermediate_size: 9728,
            vocab_size: 155_776,
            rope_base: 1_000_000.0, // PARITY: confirm Qwen3 rope_theta on-box
            norm_eps: 1e-6,         // PARITY: confirm rms_norm_eps on-box
            max_seq_len: 32_768,
            attention_qkv_bias: true, // fish_qwen3_omni: qkv bias on
            attention_o_bias: true,   // fish_qwen3_omni: o bias on
            attention_qk_norm: true,  // fish_qwen3_omni: QK-RMSNorm on
            tie_word_embeddings: false, // PARITY: confirm tie (4B usually untied) on-box
        };
        // Fast AR: 4 layers, ~400M. Single shared embedding table; the codebook index is
        // encoded by the RoPE position rather than per-codebook tables, and the slow
        // hidden is fused in (MCF). The backend owns that; here we record the shape.
        let fast = FastArConfig {
            transformer: TransformerConfig {
                n_layer: 4, // audio_decoder_config.n_layer
                // PARITY: confirm the fast head dim/heads/intermediate from
                // audio_decoder_config on-box; reference falls back to slow values.
                dim: 1024,             // PARITY: confirm fast_dim on-box
                n_head: 16,            // PARITY: confirm fast_n_head on-box
                n_local_heads: 16,     // PARITY: confirm fast_n_local_heads on-box
                head_dim: 64,          // PARITY: confirm fast_head_dim on-box
                intermediate_size: 3072, // PARITY: confirm fast_intermediate_size on-box
                attention_qkv_bias: true, // PARITY: confirm fast qkv bias on-box
                attention_o_bias: true,   // PARITY: confirm fast o bias on-box
                attention_qk_norm: true,  // PARITY: confirm fast QK-norm on-box
                ..slow.clone()
            },
            norm_fastlayer_input: true, // fish_qwen3_omni sets norm_fastlayer_input
        };
        let codec = CodecConfig {
            num_codebooks: 10,    // 1 semantic-derived + 9 residual
            semantic_size: 4096,
            residual_size: 1024,  // PARITY: confirm s2 codec codebook size on-box
            codebook_dim: 8,      // PARITY: confirm s2 codec codebook_dim on-box
            sample_rate: 44_100,
            frame_hop: 2048,      // PARITY: confirm s2 codec hop / frame rate on-box
            encoder_rates: vec![2, 4, 8, 8], // PARITY: confirm s2 EVA-GAN/DAC strides on-box
            decoder_rates: vec![8, 8, 4, 2], // PARITY: confirm s2 EVA-GAN/DAC strides on-box
        };
        FishConfig {
            variant: FishVariant::S2Pro,
            slow,
            fast,
            codec,
            // PARITY: tokenizer-injected at load; see s1_mini note.
            semantic_begin_id: 0,
            semantic_end_id: 4095,
            stop_token_id: 0, // PARITY: resolve <|im_end|> id from the Qwen3 tokenizer on-box
        }
    }

    /// Build the config for a [`FishVariant`].
    pub fn for_variant(variant: FishVariant) -> Self {
        match variant {
            FishVariant::S1Mini => Self::s1_mini(),
            FishVariant::S2Pro => Self::s2_pro(),
        }
    }

    /// Load a real Fish `config.json`, mirroring the reference
    /// `BaseModelArgs.from_pretrained`: the `dual_ar` layout maps directly, and the
    /// `fish_qwen3_omni` layout is flattened from its `text_config` / `audio_decoder_config`
    /// sub-objects. Returns an error string on malformed / unknown JSON.
    ///
    /// The variant-defaults from [`Self::s1_mini`] / [`Self::s2_pro`] seed any field the
    /// JSON omits, then the present keys override them. The codec geometry is not part of
    /// the LM `config.json` (it lives in the codec checkpoint), so the codec sub-config is
    /// taken from the chosen variant default and flagged for on-box confirmation.
    pub fn from_fish_json(json: &str, variant: FishVariant) -> Result<Self, String> {
        let v: serde_json::Value =
            serde_json::from_str(json).map_err(|e| format!("parse config.json: {e}"))?;
        let model_type = v
            .get("model_type")
            .and_then(|m| m.as_str())
            .unwrap_or("dual_ar");

        let mut cfg = Self::for_variant(variant);

        // The slow-backbone source object: `fish_qwen3_omni` nests it under `text_config`.
        let (text, audio) = match model_type {
            "fish_qwen3_omni" => (
                v.get("text_config").unwrap_or(&v),
                v.get("audio_decoder_config"),
            ),
            _ => (&v, None),
        };

        apply_transformer_json(&mut cfg.slow, text);
        if let Some(a) = audio {
            // The fast head's shape comes from `audio_decoder_config` in s2.
            apply_transformer_json(&mut cfg.fast.transformer, a);
            if let Some(n) = a.get("n_layer").and_then(|x| x.as_u64()) {
                cfg.fast.transformer.n_layer = n as usize;
            }
            if let Some(r) = a.get("vocab_size").and_then(|x| x.as_u64()) {
                cfg.codec.residual_size = r as usize;
            }
            if let Some(nc) = a.get("num_codebooks").and_then(|x| x.as_u64()) {
                cfg.codec.num_codebooks = nc as usize;
            }
        }

        if let Some(id) = v.get("semantic_start_token_id").and_then(|x| x.as_u64()) {
            cfg.semantic_begin_id = id as u32;
        }
        if let Some(id) = v.get("semantic_end_token_id").and_then(|x| x.as_u64()) {
            cfg.semantic_end_id = id as u32;
        }
        Ok(cfg)
    }
}

/// Overlay the transformer-shaped keys present in `obj` onto `cfg` (absent keys keep
/// the variant default). Mirrors the field names in the reference `BaseModelArgs`.
fn apply_transformer_json(cfg: &mut TransformerConfig, obj: &serde_json::Value) {
    fn u(obj: &serde_json::Value, k: &str) -> Option<usize> {
        obj.get(k).and_then(|x| x.as_u64()).map(|n| n as usize)
    }
    fn f(obj: &serde_json::Value, k: &str) -> Option<f64> {
        obj.get(k).and_then(|x| x.as_f64())
    }
    fn b(obj: &serde_json::Value, k: &str) -> Option<bool> {
        obj.get(k).and_then(|x| x.as_bool())
    }
    if let Some(x) = u(obj, "dim") {
        cfg.dim = x;
    }
    if let Some(x) = u(obj, "n_layer") {
        cfg.n_layer = x;
    }
    if let Some(x) = u(obj, "n_head") {
        cfg.n_head = x;
    }
    if let Some(x) = u(obj, "n_local_heads") {
        cfg.n_local_heads = x;
    }
    if let Some(x) = u(obj, "head_dim") {
        cfg.head_dim = x;
    }
    if let Some(x) = u(obj, "intermediate_size") {
        cfg.intermediate_size = x;
    }
    if let Some(x) = u(obj, "vocab_size") {
        cfg.vocab_size = x;
    }
    if let Some(x) = f(obj, "rope_base") {
        cfg.rope_base = x;
    }
    if let Some(x) = f(obj, "norm_eps") {
        cfg.norm_eps = x;
    }
    if let Some(x) = u(obj, "max_seq_len") {
        cfg.max_seq_len = x;
    }
    if let Some(x) = b(obj, "attention_qkv_bias") {
        cfg.attention_qkv_bias = x;
    }
    if let Some(x) = b(obj, "attention_o_bias") {
        cfg.attention_o_bias = x;
    }
    if let Some(x) = b(obj, "attention_qk_norm") {
        cfg.attention_qk_norm = x;
    }
    if let Some(x) = b(obj, "tie_word_embeddings") {
        cfg.tie_word_embeddings = x;
    }
}
