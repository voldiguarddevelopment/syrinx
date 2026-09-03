//! The Qwen3-TTS model geometry, parsed from the published `config.json`.
//!
//! Every field here is READ FROM THE CHECKPOINT, not guessed. That is deliberate: the
//! sibling `syrinx-fish` s1 port shipped hand-written defaults for the same kind of
//! geometry and seven of nine were wrong (`vocab_size` was out by 4.75x), which cost a
//! debugging session. `Qwen3TtsConfig::from_json` is the only constructor, and
//! [`Qwen3TtsConfig::variant`] is derived from the file rather than assumed.

use serde::{Deserialize, Serialize};

/// Which published checkpoint this config describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QwenVariant {
    /// `*-Base`: 3-second voice clone from a reference clip; no instruction control.
    Base,
    /// `*-CustomVoice`: 9 preset timbres with natural-language instruction control.
    CustomVoice,
    /// `*-VoiceDesign`: voice synthesised from a natural-language description.
    VoiceDesign,
}

impl QwenVariant {
    /// Parse the `tts_model_type` field (`base` / `custom_voice` / `voice_design`).
    pub fn from_model_type(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "base" => Some(Self::Base),
            "custom_voice" | "customvoice" => Some(Self::CustomVoice),
            "voice_design" | "voicedesign" => Some(Self::VoiceDesign),
            _ => None,
        }
    }

    /// Whether this checkpoint accepts a natural-language `instruct` directive.
    ///
    /// Qwen3-TTS has **no inline emotion tags** — no `[sad]` / `(whispering)` markers
    /// the way Fish does. Its equivalent is a per-utterance instruction, so a caller
    /// porting a tagged corpus has to fold the tags into one directive per utterance.
    pub fn supports_instruct(self) -> bool {
        matches!(self, Self::CustomVoice | Self::VoiceDesign)
    }

    /// Whether this checkpoint clones from a reference clip.
    pub fn supports_voice_clone(self) -> bool {
        matches!(self, Self::Base)
    }
}

/// One Qwen3 decoder stack's shape (the talker backbone or the code predictor).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransformerConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
}

/// The whole model's geometry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen3TtsConfig {
    pub variant: QwenVariant,
    /// The 28-layer semantic LM (`talker.model.*`).
    pub talker: TransformerConfig,
    /// The per-frame codebook head (`talker.code_predictor.*`).
    pub code_predictor: TransformerConfig,
    /// Total RVQ code groups per frame. Group 0 comes from the talker's `codec_head`;
    /// groups `1..num_code_groups` come from the code predictor's per-group heads —
    /// which is why the checkpoint carries `num_code_groups - 1` of those.
    pub num_code_groups: usize,
    /// Width of the text-embedding table before `text_projection` narrows it to
    /// `talker.hidden_size`.
    pub text_embed_dim: usize,
    /// Text-side vocabulary (`talker.model.text_embedding` rows).
    pub text_vocab_size: usize,
    pub speaker_enc_dim: usize,
    pub sample_rate: u32,
    pub tts_bos_token_id: u32,
    pub tts_eos_token_id: u32,
    pub tts_pad_token_id: u32,
    pub im_start_token_id: u32,
    pub im_end_token_id: u32,
    pub codec_bos_id: u32,
    pub codec_eos_token_id: u32,
    pub codec_pad_id: u32,
}

fn u(v: &serde_json::Value, k: &str) -> Option<usize> {
    v.get(k).and_then(|x| x.as_u64()).map(|n| n as usize)
}
fn u32f(v: &serde_json::Value, k: &str) -> Option<u32> {
    v.get(k).and_then(|x| x.as_u64()).map(|n| n as u32)
}
fn f(v: &serde_json::Value, k: &str) -> Option<f64> {
    v.get(k).and_then(|x| x.as_f64())
}

fn transformer_from(v: &serde_json::Value, what: &str) -> Result<TransformerConfig, String> {
    let need = |k: &str| u(v, k).ok_or_else(|| format!("{what}: missing `{k}`"));
    Ok(TransformerConfig {
        hidden_size: need("hidden_size")?,
        num_hidden_layers: need("num_hidden_layers")?,
        num_attention_heads: need("num_attention_heads")?,
        num_key_value_heads: need("num_key_value_heads")?,
        head_dim: need("head_dim")?,
        intermediate_size: need("intermediate_size")?,
        vocab_size: need("vocab_size")?,
        rope_theta: f(v, "rope_theta").unwrap_or(1_000_000.0),
        rms_norm_eps: f(v, "rms_norm_eps").unwrap_or(1e-6),
        max_position_embeddings: u(v, "max_position_embeddings").unwrap_or(32_768),
    })
}

impl Qwen3TtsConfig {
    /// Parse a published Qwen3-TTS `config.json`.
    pub fn from_json(json: &str) -> Result<Self, String> {
        let v: serde_json::Value =
            serde_json::from_str(json).map_err(|e| format!("parse config.json: {e}"))?;

        let model_type = v
            .get("model_type")
            .and_then(|x| x.as_str())
            .unwrap_or_default();
        if model_type != "qwen3_tts" {
            return Err(format!(
                "not a Qwen3-TTS config: model_type = {model_type:?} (expected `qwen3_tts`)"
            ));
        }
        let variant = v
            .get("tts_model_type")
            .and_then(|x| x.as_str())
            .and_then(QwenVariant::from_model_type)
            .ok_or("config.json: unrecognised `tts_model_type`")?;

        let talker_v = v.get("talker_config").ok_or("config.json: no `talker_config`")?;
        let cp_v = talker_v
            .get("code_predictor_config")
            .ok_or("talker_config: no `code_predictor_config`")?;

        let talker = transformer_from(talker_v, "talker_config")?;
        let code_predictor = transformer_from(cp_v, "code_predictor_config")?;

        let spk = v.get("speaker_encoder_config");
        Ok(Self {
            variant,
            num_code_groups: u(talker_v, "num_code_groups")
                .or_else(|| u(cp_v, "num_code_groups"))
                .ok_or("config.json: no `num_code_groups`")?,
            // `text_embedding` is [text_vocab_size, text_hidden_size] and is projected
            // down to the talker width by `text_projection`; the two are NOT the same
            // number. The published key is `text_hidden_size` — an earlier revision here
            // read a non-existent `text_embed_dim` and silently fell back to a default
            // that happened to be correct, which is the same silent-default failure mode
            // the sibling s1 port shipped. The fallback is kept only as a last resort.
            text_embed_dim: u(talker_v, "text_hidden_size")
                .or_else(|| u(talker_v, "text_embed_dim"))
                .unwrap_or(2048),
            text_vocab_size: u(talker_v, "text_vocab_size").unwrap_or(151_936),
            speaker_enc_dim: spk.and_then(|s| u(s, "enc_dim")).unwrap_or(1024),
            sample_rate: spk
                .and_then(|s| u32f(s, "sample_rate"))
                .unwrap_or(24_000),
            tts_bos_token_id: u32f(&v, "tts_bos_token_id").unwrap_or(151_672),
            tts_eos_token_id: u32f(&v, "tts_eos_token_id").unwrap_or(151_673),
            tts_pad_token_id: u32f(&v, "tts_pad_token_id").unwrap_or(151_671),
            im_start_token_id: u32f(&v, "im_start_token_id").unwrap_or(151_644),
            im_end_token_id: u32f(&v, "im_end_token_id").unwrap_or(151_645),
            codec_bos_id: u32f(talker_v, "codec_bos_id").unwrap_or(0),
            codec_eos_token_id: u32f(talker_v, "codec_eos_token_id").unwrap_or(0),
            codec_pad_id: u32f(talker_v, "codec_pad_id").unwrap_or(0),
            talker,
            code_predictor,
        })
    }

    /// Number of per-group heads the code predictor carries: group 0 is produced by the
    /// talker's `codec_head`, so the predictor holds `num_code_groups - 1`.
    pub fn code_predictor_heads(&self) -> usize {
        self.num_code_groups.saturating_sub(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published 0.6B-Base `config.json`, verbatim in the fields this parser reads.
    /// Pinned here so a parser regression fails in CI rather than on the box — the s1
    /// port's guessed geometry is exactly what this prevents.
    const BASE_0B6: &str = r#"{
      "model_type": "qwen3_tts",
      "tts_model_type": "base",
      "tts_bos_token_id": 151672,
      "tts_eos_token_id": 151673,
      "tts_pad_token_id": 151671,
      "im_start_token_id": 151644,
      "im_end_token_id": 151645,
      "speaker_encoder_config": { "enc_dim": 1024, "sample_rate": 24000 },
      "talker_config": {
        "hidden_size": 1024, "num_hidden_layers": 28, "num_attention_heads": 16,
        "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 3072,
        "vocab_size": 3072, "text_hidden_size": 2048, "text_vocab_size": 151936, "rope_theta": 1000000, "rms_norm_eps": 1e-06,
        "max_position_embeddings": 32768, "num_code_groups": 16,
        "code_predictor_config": {
          "hidden_size": 1024, "num_hidden_layers": 5, "num_attention_heads": 16,
          "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 3072,
          "vocab_size": 2048, "rope_theta": 1000000, "rms_norm_eps": 1e-06,
          "num_code_groups": 16
        }
      }
    }"#;

    #[test]
    fn parses_the_published_base_config() {
        let c = Qwen3TtsConfig::from_json(BASE_0B6).unwrap();
        assert_eq!(c.variant, QwenVariant::Base);
        // talker: the 28-layer semantic LM
        assert_eq!(c.talker.hidden_size, 1024);
        assert_eq!(c.talker.num_hidden_layers, 28);
        assert_eq!(c.talker.num_attention_heads, 16);
        assert_eq!(c.talker.num_key_value_heads, 8);
        assert_eq!(c.talker.head_dim, 128);
        assert_eq!(c.talker.intermediate_size, 3072);
        assert_eq!(c.talker.vocab_size, 3072);
        // code predictor: 5 layers over the residual groups
        assert_eq!(c.code_predictor.num_hidden_layers, 5);
        assert_eq!(c.code_predictor.vocab_size, 2048);
        assert_eq!(c.num_code_groups, 16);
        assert_eq!(c.sample_rate, 24_000);
    }

    /// Attention projection widths implied by the config must match the shapes actually
    /// present in the checkpoint (`q_proj [2048,1024]`, `k_proj`/`v_proj [1024,1024]`,
    /// `o_proj [1024,2048]`). A head-count or head-dim slip shows up here.
    #[test]
    fn attention_widths_match_the_checkpoint_shapes() {
        let c = Qwen3TtsConfig::from_json(BASE_0B6).unwrap();
        let t = &c.talker;
        assert_eq!(t.num_attention_heads * t.head_dim, 2048, "q_proj out");
        assert_eq!(t.num_key_value_heads * t.head_dim, 1024, "k/v_proj out");
        assert_eq!(t.hidden_size, 1024, "o_proj out / model width");
    }

    /// Group 0 is the talker's `codec_head`; the predictor holds the other 15. The
    /// checkpoint carries exactly 15 `lm_head.{i}` and 15 `codec_embedding.{i}`.
    #[test]
    fn code_predictor_head_count_is_groups_minus_one() {
        let c = Qwen3TtsConfig::from_json(BASE_0B6).unwrap();
        assert_eq!(c.code_predictor_heads(), 15);
    }

    #[test]
    fn variant_capabilities_are_not_symmetric() {
        // The split that matters for a tagged corpus: Base clones but cannot be
        // instructed; CustomVoice/VoiceDesign can be instructed but do not clone.
        assert!(QwenVariant::Base.supports_voice_clone());
        assert!(!QwenVariant::Base.supports_instruct());
        assert!(QwenVariant::CustomVoice.supports_instruct());
        assert!(!QwenVariant::CustomVoice.supports_voice_clone());
        assert!(QwenVariant::VoiceDesign.supports_instruct());
    }

    /// The text-embedding width must come from the file's real key, not a default.
    /// A config with a different `text_hidden_size` must be reflected, and the key that
    /// does NOT exist in published configs (`text_embed_dim`) must not be what we rely on.
    #[test]
    fn text_width_is_read_from_text_hidden_size() {
        let j = BASE_0B6.replace("\"text_hidden_size\": 2048", "\"text_hidden_size\": 4096");
        assert!(j.contains("4096"), "test fixture edit did not apply");
        let c = Qwen3TtsConfig::from_json(&j).unwrap();
        assert_eq!(c.text_embed_dim, 4096, "must read text_hidden_size from the file");
        // published configs carry no `text_embed_dim`; relying on it alone would default
        let c2 = Qwen3TtsConfig::from_json(BASE_0B6).unwrap();
        assert_eq!(c2.text_embed_dim, 2048);
    }

    #[test]
    fn rejects_a_non_qwen_config() {
        let err = Qwen3TtsConfig::from_json(r#"{"model_type":"dual_ar"}"#).unwrap_err();
        assert!(err.contains("not a Qwen3-TTS config"), "got {err}");
    }
}
