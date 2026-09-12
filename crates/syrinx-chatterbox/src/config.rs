//! The T3 geometry, parsed from the shipped `t3_turbo_v1.yaml`.
//!
//! # The file is a training-config superset
//!
//! `t3_turbo_v1.yaml` is a flat mapping of roughly 300 keys covering *several* models in
//! Resemble's training stack. The overwhelming majority are dead here: `dcnar_*` (their
//! non-autoregressive acoustic model), `rvc_*` (voice conversion), `taco_*` (Tacotron),
//! `hooli*` / `voc*` / `dcvoc_*` (vocoders), `eval_*`, `lora_*`, optimiser and dataloader
//! settings. Parsing all of it would be a way of pretending to know more than we do.
//!
//! [`T3Config`] reads **only the keys that are load-bearing for the T3 checkpoint**, and
//! each field below says what it is load-bearing *for*. Two keys are read and recorded
//! precisely because they are traps — see [`T3Config::declared_transformer_layers`].
//!
//! # The YAML reader
//!
//! There is no YAML dependency. The file's top level is a flat `key: scalar` mapping;
//! the only non-scalar values in it (`conv_stack_dilation`, `upsample_factors`,
//! `r_schedule`, `dcvoc_smpwd_periods`, `rvc_*` sequences) are all dead fields.
//! [`scalar_fields`] therefore takes the top-level scalar entries and ignores everything
//! indented, which is both sufficient and small enough to be checked exhaustively. It is
//! deliberately not a YAML parser and must not be used as one; if Phase 1 ever needs a
//! block value, add a real dependency then.

use std::collections::BTreeMap;

/// One GPT-2 preset's geometry.
///
/// The values are the published GPT-2 sizes, and every one of them is confirmed against
/// the shipped `t3_turbo_v1.safetensors` header by `tests/chatterbox_tensor_manifest.rs`
/// — 24 blocks `tfmr.h.0..23`, `c_attn` fusing 3x1024, `c_fc` at 4096. They are not read
/// from the YAML because **the YAML's `n_transformer_layers: 30` is wrong for this
/// checkpoint**; `gpt_transformer_type` is the field that tells the truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gpt2Backbone {
    /// The `gpt_transformer_type` spelling this came from.
    pub preset: &'static str,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_channels: usize,
    /// The MLP's inner width (`mlp.c_fc` out / `mlp.c_proj` in).
    pub ffn_dim: usize,
}

const GPT2_MEDIUM: Gpt2Backbone = Gpt2Backbone {
    preset: "gpt2-medium",
    n_layers: 24,
    n_heads: 16,
    n_channels: 1024,
    ffn_dim: 4096,
};

/// The only preset any Chatterbox checkpoint in scope declares. Deliberately not a table
/// of every GPT-2 size: an entry nothing has ever been checked against is a guess with a
/// name.
fn backbone_preset(name: &str) -> Option<Gpt2Backbone> {
    match name {
        "gpt2-medium" => Some(GPT2_MEDIUM),
        _ => None,
    }
}

/// The text side of T3's token space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextTokens {
    /// `text_tokens_dict_size` — rows of `text_emb` / `text_head` / `tfmr.wte`.
    /// GPT-2 base vocabulary **plus** the paralinguistic tags; see
    /// [`crate::contract::ModelContract`], which is where that sum is checked.
    pub dict_size: usize,
    /// `start_text_token`. **Not** a dedicated special: 255 is an ordinary GPT-2 BPE id
    /// that the model reuses as a sequence marker.
    pub start_token: u32,
    /// `stop_text_token`. Likewise ordinary — id 0.
    pub stop_token: u32,
    /// `max_text_tokens` — the text budget inside `max_total_tokens`.
    pub max_tokens: usize,
}

/// The speech side of T3's token space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechTokens {
    /// `speech_tokens_dict_size` — rows of `speech_emb` / `speech_head`. Includes the
    /// two specials, so the codebook proper is `dict_size - 2` entries wide.
    pub dict_size: usize,
    /// `start_speech_token`, which sits at `dict_size - 2`.
    pub start_token: u32,
    /// `stop_speech_token`, which sits at `dict_size - 1`.
    pub stop_token: u32,
    /// `max_speech_tokens` — the speech budget inside `max_total_tokens`.
    pub max_tokens: usize,
    /// `speech_token_type` — `tortoise`. Recorded, not derived from.
    pub token_type: String,
}

/// The waveform/mel framing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioFormat {
    /// `sample_rate` — 32 kHz, the model's output rate.
    pub sample_rate: u32,
    /// `hop_size` — the mel hop in samples. Also the vocoder's total upsample factor:
    /// the (dead-to-us, block-valued) `upsample_factors: (5, 8, 8)` multiplies to 320.
    pub hop_size: u32,
}

/// How a reference clip reaches T3.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeakerCond {
    /// `encoder_type` — `voice_encoder`, i.e. the LSTM in `ve.safetensors`. This is what
    /// makes [`crate::manifest::expected_ve_tensors`] the right manifest; `s3gen`'s own
    /// `speaker_encoder.*` is a different, unrelated network.
    pub encoder_type: String,
    /// `speaker_embed_size` — 256, the width `cond_enc.spkr_enc` consumes.
    pub embed_size: usize,
    /// `speech_cond_prompt_len` — speech-token positions the reference prompt occupies.
    pub cond_prompt_len: usize,
}

/// The T3 checkpoint's geometry: only what is load-bearing, plus the recorded traps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct T3Config {
    /// Resolved from `gpt_transformer_type`, **not** from `n_transformer_layers`.
    pub backbone: Gpt2Backbone,
    /// `max_total_tokens` — rows of the backbone's learned position table `tfmr.wpe`.
    pub max_total_tokens: usize,
    /// `input_pos_emb` — `handled_internally_by_backbone`, meaning positions come from
    /// `tfmr.wpe` and there are no separate `text_pos_emb` / `speech_pos_emb` tensors.
    /// The manifest depends on this, so [`T3Config::from_yaml`] rejects any other value.
    pub input_pos_emb: String,
    /// `n_transformer_layers` **as declared**, which for this file is 30 — and the
    /// shipped checkpoint has 24 blocks. Kept as a field, never used to derive anything,
    /// so that the contradiction is recorded rather than quietly dropped. See
    /// [`T3Config::declared_layers_match_backbone`].
    pub declared_transformer_layers: usize,
    /// `llama_config_name` as declared (`Llama_520M`). Dead: the backbone is GPT-2.
    /// Recorded for the same reason as the field above.
    pub declared_llama_config_name: String,
    pub text: TextTokens,
    pub speech: SpeechTokens,
    pub audio: AudioFormat,
    pub speaker: SpeakerCond,
}

impl T3Config {
    /// Parse the shipped `t3_turbo_v1.yaml`.
    ///
    /// Rejects, rather than papers over: an unknown `gpt_transformer_type`; a declared
    /// head count or width that disagrees with the preset; speech specials that are not
    /// the top two ids of the speech dictionary; a position table too small for the
    /// prompt + text + speech budget; a sample rate that is not a whole number of hops;
    /// a speaker encoder or position-embedding scheme other than the one the manifest is
    /// written for. Every one of those would produce a wrong port silently.
    pub fn from_yaml(yaml: &str) -> Result<Self, String> {
        let f = scalar_fields(yaml);

        let preset_name = need_str(&f, "gpt_transformer_type")?;
        let backbone = backbone_preset(preset_name).ok_or_else(|| {
            format!("t3 config: unsupported `gpt_transformer_type` {preset_name:?}")
        })?;

        let declared_heads = need_usize(&f, "n_transformer_heads")?;
        if backbone.n_heads != declared_heads {
            return Err(format!(
                "t3 config: preset {} has {} heads, file declares {declared_heads}",
                backbone.preset, backbone.n_heads
            ));
        }
        let declared_channels = need_usize(&f, "n_gpt_channels")?;
        if backbone.n_channels != declared_channels {
            return Err(format!(
                "t3 config: preset {} is {} wide, `n_gpt_channels` says {declared_channels}",
                backbone.preset, backbone.n_channels
            ));
        }
        let legacy_hidden = need_usize(&f, "legacy_gpt_hidden_size")?;
        if backbone.n_channels != legacy_hidden {
            return Err(format!(
                "t3 config: `legacy_gpt_hidden_size` {legacy_hidden} contradicts `n_gpt_channels` {}",
                backbone.n_channels
            ));
        }

        let speech = SpeechTokens {
            dict_size: need_usize(&f, "speech_tokens_dict_size")?,
            start_token: need_u32(&f, "start_speech_token")?,
            stop_token: need_u32(&f, "stop_speech_token")?,
            max_tokens: need_usize(&f, "max_speech_tokens")?,
            token_type: need_str(&f, "speech_token_type")?.to_string(),
        };
        // The two specials are appended above the codebook, so they are the last two ids.
        // A file where they are not would mean the codebook width is not `dict_size - 2`
        // and every embedding row index after it would be off.
        if speech.start_token as usize != speech.dict_size - 2 {
            return Err(format!(
                "t3 config: start_speech_token {} is not `dict_size - 2` for a speech \
                 dictionary of {}",
                speech.start_token, speech.dict_size
            ));
        }
        if speech.stop_token as usize != speech.dict_size - 1 {
            return Err(format!(
                "t3 config: stop_speech_token {} is not `dict_size - 1` for a speech \
                 dictionary of {}",
                speech.stop_token, speech.dict_size
            ));
        }

        let text = TextTokens {
            dict_size: need_usize(&f, "text_tokens_dict_size")?,
            start_token: need_u32(&f, "start_text_token")?,
            stop_token: need_u32(&f, "stop_text_token")?,
            max_tokens: need_usize(&f, "max_text_tokens")?,
        };

        let speaker = SpeakerCond {
            encoder_type: need_str(&f, "encoder_type")?.to_string(),
            embed_size: need_usize(&f, "speaker_embed_size")?,
            cond_prompt_len: need_usize(&f, "speech_cond_prompt_len")?,
        };
        if speaker.encoder_type != VOICE_ENCODER {
            return Err(format!(
                "t3 config: `encoder_type` {:?} is not {VOICE_ENCODER:?}",
                speaker.encoder_type
            ));
        }

        let max_total_tokens = need_usize(&f, "max_total_tokens")?;
        let budget = speaker.cond_prompt_len + text.max_tokens + speech.max_tokens;
        if max_total_tokens < budget {
            return Err(format!(
                "t3 config: `max_total_tokens` {max_total_tokens} cannot hold the \
                 prompt+text+speech budget {budget}"
            ));
        }

        let input_pos_emb = need_str(&f, "input_pos_emb")?.to_string();
        if input_pos_emb != BACKBONE_POS_EMB {
            return Err(format!(
                "t3 config: `input_pos_emb` {input_pos_emb:?} is not {BACKBONE_POS_EMB:?}; \
                 the manifest assumes positions come from `tfmr.wpe`"
            ));
        }

        let audio = AudioFormat {
            sample_rate: need_u32(&f, "sample_rate")?,
            hop_size: need_u32(&f, "hop_size")?,
        };
        if audio.sample_rate % audio.hop_size != 0 {
            return Err(format!(
                "t3 config: sample_rate {} is not a whole number of {}-sample hops",
                audio.sample_rate, audio.hop_size
            ));
        }

        Ok(Self {
            backbone,
            max_total_tokens,
            input_pos_emb,
            declared_transformer_layers: need_usize(&f, "n_transformer_layers")?,
            declared_llama_config_name: need_str(&f, "llama_config_name")?.to_string(),
            text,
            speech,
            audio,
            speaker,
        })
    }

    /// Whether the file's `n_transformer_layers` agrees with the backbone the file's own
    /// `gpt_transformer_type` names.
    ///
    /// **For the shipped `t3_turbo_v1.yaml` this is `false`** (30 declared, 24 real), and
    /// that is the point: the disagreement is a fact about the file, so it is exposed as
    /// a value a test can pin rather than left as a footnote nobody reads. The parser
    /// does not reject on it, because rejecting would reject the real checkpoint.
    pub fn declared_layers_match_backbone(&self) -> bool {
        self.declared_transformer_layers == self.backbone.n_layers
    }

    /// Mel frames per second — `sample_rate / hop_size`, exact by the check in
    /// [`T3Config::from_yaml`]. This is the **mel** rate, not the speech-token rate:
    /// speech tokens are produced by the S3 tokenizer inside `s3gen`, which this crate
    /// does not describe.
    pub fn mel_frame_rate_hz(&self) -> u32 {
        self.audio.sample_rate / self.audio.hop_size
    }

    /// Speech codebook entries excluding the two specials.
    pub fn speech_codebook_size(&self) -> usize {
        self.speech.dict_size - 2
    }
}

/// The `encoder_type` value the [`crate::manifest::expected_ve_tensors`] manifest is
/// written for.
pub const VOICE_ENCODER: &str = "voice_encoder";

/// The `input_pos_emb` value that means "positions live in `tfmr.wpe`".
pub const BACKBONE_POS_EMB: &str = "handled_internally_by_backbone";

// ------------------------------------------------------------------- the flat YAML read

/// Top-level `key: scalar` entries of a flat YAML mapping, values trimmed.
///
/// A line contributes only when the text before its first `:` is entirely
/// `[A-Za-z0-9_]` — which is true of every top-level key in this file and false of every
/// indented line (they begin with spaces) and every sequence item (`- 1` has no colon at
/// all). Non-scalar values therefore never appear: a key introducing a block has nothing
/// after the colon, and its children are indented.
fn scalar_fields(yaml: &str) -> BTreeMap<&str, &str> {
    let mut out = BTreeMap::new();
    for line in yaml.lines() {
        let Some((key, rest)) = line.split_once(':') else { continue };
        if key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            out.insert(key, rest.trim());
        }
    }
    out
}

fn need_str<'a>(f: &BTreeMap<&'a str, &'a str>, key: &str) -> Result<&'a str, String> {
    f.get(key).copied().ok_or_else(|| format!("t3 config: missing `{key}`"))
}

fn need_usize(f: &BTreeMap<&str, &str>, key: &str) -> Result<usize, String> {
    let v = need_str(f, key)?;
    v.parse().map_err(|_| format!("t3 config: `{key}` is not an integer: {v:?}"))
}

fn need_u32(f: &BTreeMap<&str, &str>, key: &str) -> Result<u32, String> {
    let v = need_str(f, key)?;
    v.parse().map_err(|_| format!("t3 config: `{key}` is not an integer: {v:?}"))
}
