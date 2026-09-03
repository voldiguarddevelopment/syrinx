//! Prompt assembly for the Qwen3-TTS talker — the exact input sequence for each of the
//! three generation modes.
//!
//! # Everything here is read out of the reference, not inferred
//!
//! Source: the installed `qwen_tts` package (`~/.venvs/qwen`, Apache-2.0), specifically
//!
//! * `qwen_tts/inference/qwen3_tts_model.py` — `Qwen3TTSModel._build_assistant_text`,
//!   `._build_ref_text`, `._build_instruct_text`, `.generate_custom_voice`,
//!   `.generate_voice_design`, `.generate_voice_clone`;
//! * `qwen_tts/core/models/modeling_qwen3_tts.py` — `Qwen3TTSForConditionalGeneration
//!   .generate` (the whole prefill), `.generate_icl_prompt`, and
//!   `Qwen3TTSTalkerForConditionalGeneration.forward` (how `trailing_text_hidden` is
//!   consumed during decoding).
//!
//! The three chat wrappers are literal (`qwen3_tts_model.py`):
//!
//! ```text
//! _build_assistant_text(text)  ->  "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
//! _build_ref_text(text)        ->  "<|im_start|>assistant\n{text}<|im_end|>\n"
//! _build_instruct_text(instr)  ->  "<|im_start|>user\n{instr}<|im_end|>\n"
//! ```
//!
//! With the Qwen vocabulary `<|im_start|>assistant\n` is exactly **3** ids
//! `[151644, 77091, 198]` and `<|im_end|>\n<|im_start|>assistant\n` exactly **5**
//! `[151645, 198, 151644, 77091, 198]`, which is why `generate` slices the assistant
//! text as `input_id[:, :3]` (role) and `input_id[:, 3:-5]` (body), and the reference
//! text as `ref_id[:, 3:-2]`. This module reproduces those slices rather than encoding
//! the pieces separately — separate encodes are *not* equivalent, because the
//! pre-tokenizer's `\s*[\r\n]+` rule merges a trailing newline with a leading one.
//!
//! # The talker prompt is TWO summed streams, not one token sequence
//!
//! Every prompt position adds up to two embeddings:
//!
//! * a **text** addend — `text_projection(text_embedding[id])`, and
//! * a **codec** addend — `talker.model.codec_embedding[id]`, or a speaker vector, or
//!   the summed 16-group embedding of one reference frame.
//!
//! Either may be absent (the role prefix and the instruct block are text-only). So a
//! prompt is a [`PromptPlan`] of [`PromptStep`]s, and the model crate turns each step
//! into one row of `inputs_embeds`. That mirrors `syrinx-fish`'s
//! `s2::build_prompt_with_reference`, which likewise interleaves text ids and code rows
//! instead of producing a flat id vector.
//!
//! # Verified against the reference, not just read from it
//!
//! Every layout below was diffed position-by-position against the reference's own
//! output. `Qwen3TTSForConditionalGeneration.generate` was called as an unbound
//! function on a stand-in `self` supplying only the attributes it touches — `config`,
//! `talker.{device, dtype, config, text_projection, get_text_embeddings,
//! get_input_embeddings, code_predictor.get_input_embeddings, generate}`,
//! `generate_speaker_prompt`, `generate_icl_prompt` — with every embedding table
//! replaced by a function writing an identifiable marker into the row, and
//! `talker.generate` raising so the finished `inputs_embeds` / `trailing_text_hidden`
//! could be captured. No weights, no GPU; the real reference code lays out the prompt.
//! 13 cases (custom_voice × {explicit language, auto, dialect speaker, instruct,
//! streaming}, voice_design × {instruct, bare}, voice_clone × {x-vector-only, ICL
//! streaming, ICL non-streaming, ICL with text longer than the codec stream}) matched
//! this module's [`PromptPlan`]s exactly, `trailing_text` included.
//!
//! # `position_id_per_seconds` — present in the config, unused by the model
//!
//! `talker_config.position_id_per_seconds` is `13` in all five checkpoints, but
//! `modeling_qwen3_tts.py` never reads it: `get_rope_index` computes
//! `position_ids = attention_mask.cumsum(-1) - 1`, replicated three times for the
//! mRoPE sections (`mrope_section [24, 20, 20]`, `interleaved: true`) — so all three
//! sections carry the same plain 0,1,2,… positions and mRoPE degenerates to ordinary
//! RoPE. Nothing in the prompt is scaled by 13. Grep confirms: the only hits for
//! `position_id_per_seconds` in the package are the config classes.

use std::collections::BTreeMap;

use crate::tokenizer::{QwenTokenizer, TokenizerError};

/// `generate_custom_voice(non_streaming_mode=True)` — the reference default.
pub const CUSTOM_VOICE_NON_STREAMING: bool = true;
/// `generate_voice_design(non_streaming_mode=True)` — the reference default.
pub const VOICE_DESIGN_NON_STREAMING: bool = true;
/// `generate_voice_clone(non_streaming_mode=False)` — the reference default. Note it
/// differs from the other two.
pub const VOICE_CLONE_NON_STREAMING: bool = false;

/// The nine preset timbres of the `*-CustomVoice` checkpoints, from `talker_config
/// .spk_id`. `eric` and `dylan` additionally carry a dialect (`spk_is_dialect`).
pub const PRESET_SPEAKERS: [&str; 9] = [
    "aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian",
];

/// The text addend at one prompt position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextSlot {
    /// A text-vocab id, embedded through `text_embedding` then `text_projection`.
    Id(u32),
    /// No text addend at this position.
    None,
}

/// The codec addend at one prompt position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecSlot {
    /// A codec-vocab id, embedded through `talker.model.codec_embedding`.
    Id(u32),
    /// The reference speaker vector (`extract_speaker_embedding`, `[speaker_enc_dim]`),
    /// used in place of a codec-table lookup on the clone paths.
    SpeakerVector,
    /// Reference frame `i`: the SUM over all `num_code_groups` groups, where group 0 is
    /// embedded by `talker.model.codec_embedding` and group `g>0` by
    /// `talker.code_predictor.codec_embedding[g-1]` (`generate_icl_prompt`).
    RefFrame(usize),
    /// No codec addend at this position.
    None,
}

/// One prompt position: the two addends that are summed into `inputs_embeds`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptStep {
    /// The text-stream addend.
    pub text: TextSlot,
    /// The codec-stream addend.
    pub codec: CodecSlot,
}

/// A fully assembled talker prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptPlan {
    /// The prefill positions, in order.
    pub steps: Vec<PromptStep>,
    /// The text addend for generated frame `g` (`trailing_text_hidden`): decoding step
    /// `g` adds `trailing_text[g]` if `g < trailing_text.len()`, else `<tts_pad>`
    /// (`Qwen3TTSTalkerForConditionalGeneration.forward`). In non-streaming mode this
    /// is exactly `[<tts_pad>]`, i.e. pad forever.
    pub trailing_text: Vec<TextSlot>,
    /// Reference frames spliced into the prefill (ICL clone only). The reference
    /// decodes `cat(ref_code, generated)` and then drops the leading
    /// `ref_frames / total_frames` fraction of the waveform.
    pub ref_frames: usize,
}

impl PromptPlan {
    /// Number of prefill positions.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the prefill is empty (it never is for a well-formed request).
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

/// The prompt-side ids and maps of a checkpoint's `talker_config`.
///
/// Separate from [`crate::config::Qwen3TtsConfig`], which carries the model *geometry*;
/// these are the fields `generate` reaches for while laying out the prefill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptConfig {
    /// `talker_config.codec_bos_id` (2149).
    pub codec_bos_id: u32,
    /// `talker_config.codec_eos_token_id` (2150) — the generation stop.
    pub codec_eos_id: u32,
    /// `talker_config.codec_pad_id` (2148).
    pub codec_pad_id: u32,
    /// `talker_config.codec_think_id` (2154) — opens a language-tagged prefill.
    pub codec_think_id: u32,
    /// `talker_config.codec_nothink_id` (2155) — opens an `auto`-language prefill.
    pub codec_nothink_id: u32,
    /// `talker_config.codec_think_bos_id` (2156).
    pub codec_think_bos_id: u32,
    /// `talker_config.codec_think_eos_id` (2157).
    pub codec_think_eos_id: u32,
    /// `talker_config.codec_language_id` — language name → codec id. Includes the two
    /// dialect entries on the CustomVoice checkpoints.
    pub codec_language_id: BTreeMap<String, u32>,
    /// `talker_config.spk_id` — preset speaker → codec id. Empty on Base/VoiceDesign.
    pub spk_id: BTreeMap<String, u32>,
    /// `talker_config.spk_is_dialect` — preset speaker → dialect key, if any.
    pub spk_is_dialect: BTreeMap<String, Option<String>>,
    /// `talker_config.num_code_groups` (16).
    pub num_code_groups: usize,
    /// `talker_config.position_id_per_seconds` (13). Parsed for completeness; the
    /// reference model never reads it (see the module docs).
    pub position_id_per_seconds: u32,
    /// `tts_model_size` (`"0b6"` / `"1b7"`) — gates instruction support, see
    /// [`PromptConfig::honors_instruct`].
    pub tts_model_size: String,
    /// `tts_bos_token_id` (151672).
    pub tts_bos_id: u32,
    /// `tts_eos_token_id` (151673).
    pub tts_eos_id: u32,
    /// `tts_pad_token_id` (151671).
    pub tts_pad_id: u32,
}

fn u32_at(v: &serde_json::Value, k: &str) -> Result<u32, PromptError> {
    v.get(k)
        .and_then(|x| x.as_u64())
        .map(|n| n as u32)
        .ok_or_else(|| PromptError::Config(format!("missing `{k}`")))
}

impl PromptConfig {
    /// Parse a published Qwen3-TTS `config.json`.
    pub fn from_json(json: &str) -> Result<Self, PromptError> {
        let v: serde_json::Value =
            serde_json::from_str(json).map_err(|e| PromptError::Config(format!("parse: {e}")))?;
        let t = v
            .get("talker_config")
            .ok_or_else(|| PromptError::Config("no `talker_config`".into()))?;

        let map_u32 = |k: &str| -> BTreeMap<String, u32> {
            t.get(k)
                .and_then(|m| m.as_object())
                .map(|m| {
                    m.iter()
                        .filter_map(|(k, v)| v.as_u64().map(|n| (k.clone(), n as u32)))
                        .collect()
                })
                .unwrap_or_default()
        };
        // `spk_is_dialect` values are either `false` or a dialect key string.
        let spk_is_dialect = t
            .get("spk_is_dialect")
            .and_then(|m| m.as_object())
            .map(|m| {
                m.iter()
                    .map(|(k, v)| (k.clone(), v.as_str().map(str::to_string)))
                    .collect()
            })
            .unwrap_or_default();

        Ok(Self {
            codec_bos_id: u32_at(t, "codec_bos_id")?,
            codec_eos_id: u32_at(t, "codec_eos_token_id")?,
            codec_pad_id: u32_at(t, "codec_pad_id")?,
            codec_think_id: u32_at(t, "codec_think_id")?,
            codec_nothink_id: u32_at(t, "codec_nothink_id")?,
            codec_think_bos_id: u32_at(t, "codec_think_bos_id")?,
            codec_think_eos_id: u32_at(t, "codec_think_eos_id")?,
            codec_language_id: map_u32("codec_language_id"),
            spk_id: map_u32("spk_id"),
            spk_is_dialect,
            num_code_groups: t
                .get("num_code_groups")
                .and_then(|x| x.as_u64())
                .ok_or_else(|| PromptError::Config("missing `num_code_groups`".into()))?
                as usize,
            position_id_per_seconds: u32_at(t, "position_id_per_seconds")?,
            tts_model_size: v
                .get("tts_model_size")
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string(),
            tts_bos_id: u32_at(&v, "tts_bos_token_id")?,
            tts_eos_id: u32_at(&v, "tts_eos_token_id")?,
            tts_pad_id: u32_at(&v, "tts_pad_token_id")?,
        })
    }

    /// The language names a caller may pass: `"auto"` plus every non-dialect key of
    /// `codec_language_id`. This is `Qwen3TTSForConditionalGeneration.__init__`'s
    /// `supported_languages` — dialects are reachable only indirectly, via a dialect
    /// speaker. **There is no Polish.**
    pub fn supported_languages(&self) -> Vec<String> {
        let mut out = vec!["auto".to_string()];
        out.extend(
            self.codec_language_id
                .keys()
                .filter(|k| !k.contains("dialect"))
                .cloned(),
        );
        out
    }

    /// The preset speakers this checkpoint accepts (`get_supported_speakers`).
    pub fn supported_speakers(&self) -> Vec<String> {
        self.spk_id.keys().cloned().collect()
    }

    /// Whether a `custom_voice` request's `instruct` is honoured.
    ///
    /// `generate_custom_voice` contains `if self.model.tts_model_size in "0b6":
    /// instruct = None` — a Python substring test, so the 0.6B CustomVoice checkpoint
    /// silently drops the instruction and only the 1.7B one is instructable.
    pub fn honors_instruct(&self) -> bool {
        !"0b6".contains(&self.tts_model_size)
    }
}

/// What conditions a voice-clone request (`create_voice_clone_prompt`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloneRef<'a> {
    /// `x_vector_only_mode=True`: the speaker vector alone. No reference text or codes
    /// enter the prompt, and the prefill is the ordinary text layout.
    XVectorOnly,
    /// `x_vector_only_mode=False` → `icl_mode=True`: the reference transcript and its
    /// RVQ frames are spliced in ahead of the target text (`generate_icl_prompt`).
    /// `ref_text` is mandatory in this mode.
    InContext {
        /// The reference clip's transcript.
        ref_text: &'a str,
        /// Number of frames in the reference's `[T, num_code_groups]` code matrix.
        frames: usize,
    },
}

/// Errors assembling a prompt.
#[derive(Debug)]
pub enum PromptError {
    /// `config.json` could not be parsed for the prompt-side ids.
    Config(String),
    /// The text tokenizer failed.
    Tokenize(TokenizerError),
    /// The language is not one of `supported_languages()`.
    UnknownLanguage(String),
    /// The speaker is not one of `supported_speakers()`.
    UnknownSpeaker(String),
    /// The text to synthesise produced no body tokens.
    EmptyText,
    /// An in-context clone was requested with an empty reference (no text, or no
    /// frames) — `create_voice_clone_prompt` raises here too.
    EmptyReference,
    /// The tokenized chat wrapper did not start with `<|im_start|> assistant \n`, so
    /// the reference's fixed `[:3]` / `[3:-5]` slices would silently mis-cut.
    UnexpectedChatLayout(String),
}

impl std::fmt::Display for PromptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(m) => write!(f, "qwen prompt config: {m}"),
            Self::Tokenize(e) => write!(f, "qwen prompt: {e}"),
            Self::UnknownLanguage(l) => write!(f, "qwen prompt: unsupported language {l:?}"),
            Self::UnknownSpeaker(s) => write!(f, "qwen prompt: unsupported speaker {s:?}"),
            Self::EmptyText => write!(f, "qwen prompt: the text to synthesise is empty"),
            Self::EmptyReference => write!(f, "qwen prompt: in-context clone needs ref_text and ref frames"),
            Self::UnexpectedChatLayout(m) => write!(f, "qwen prompt: {m}"),
        }
    }
}

impl std::error::Error for PromptError {}

impl From<TokenizerError> for PromptError {
    fn from(e: TokenizerError) -> Self {
        Self::Tokenize(e)
    }
}

/// The three role ids of `<|im_start|>assistant\n` / `<|im_start|>user\n`, split off
/// the front of a tokenized chat wrapper.
struct ChatText {
    /// `input_id[:, :3]` — `[<|im_start|>, assistant, \n]`.
    role: Vec<u32>,
    /// `input_id[:, 3:-5]` for the assistant wrapper, `[3:-2]` for the ref wrapper.
    body: Vec<u32>,
}

/// Tokenize `"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"` and
/// split it the way `generate` does (`[:3]` role, `[3:-5]` body).
fn assistant_text(tok: &QwenTokenizer, text: &str) -> Result<ChatText, PromptError> {
    let ids = tok.encode(&format!(
        "{s}assistant\n{text}{e}\n{s}assistant\n",
        s = crate::tokenizer::IM_START,
        e = crate::tokenizer::IM_END
    ))?;
    split_chat(tok, ids, 5, "assistant text")
}

/// Tokenize `"<|im_start|>assistant\n{text}<|im_end|>\n"` and split it the way
/// `generate` does for the reference transcript (`[3:-2]`).
fn reference_text(tok: &QwenTokenizer, text: &str) -> Result<ChatText, PromptError> {
    let ids = tok.encode(&format!(
        "{s}assistant\n{text}{e}\n",
        s = crate::tokenizer::IM_START,
        e = crate::tokenizer::IM_END
    ))?;
    split_chat(tok, ids, 2, "reference text")
}

fn split_chat(
    tok: &QwenTokenizer,
    ids: Vec<u32>,
    tail: usize,
    what: &str,
) -> Result<ChatText, PromptError> {
    if ids.len() < 3 + tail || ids[0] != tok.im_start_id {
        return Err(PromptError::UnexpectedChatLayout(format!(
            "{what}: {} ids do not open with <|im_start|> + a 3-id role prefix",
            ids.len()
        )));
    }
    Ok(ChatText {
        role: ids[..3].to_vec(),
        body: ids[3..ids.len() - tail].to_vec(),
    })
}

/// Resolve the language tag exactly as `generate` does.
///
/// `"auto"` yields `None` (the `codec_nothink_id` prefill); anything else must be in
/// `codec_language_id`. A dialect speaker then overrides it when the request language
/// is `chinese` or `auto`.
fn language_id(
    cfg: &PromptConfig,
    language: &str,
    speaker: Option<&str>,
) -> Result<Option<u32>, PromptError> {
    let lang = language.to_ascii_lowercase();
    let mut id = if lang == "auto" {
        None
    } else {
        Some(
            *cfg.codec_language_id
                .get(&lang)
                .ok_or_else(|| PromptError::UnknownLanguage(language.to_string()))?,
        )
    };
    if lang == "chinese" || lang == "auto" {
        if let Some(spk) = speaker {
            if let Some(Some(dialect)) = cfg.spk_is_dialect.get(&spk.to_ascii_lowercase()) {
                id = Some(
                    *cfg.codec_language_id
                        .get(dialect)
                        .ok_or_else(|| PromptError::UnknownLanguage(dialect.clone()))?,
                );
            }
        }
    }
    Ok(id)
}

/// The codec-stream prefill: `codec_input_emebdding` in `generate`.
///
/// `[think|nothink, think_bos, (language,) think_eos] ++ [speaker] ++ [pad, bos]`,
/// where the trailing `bos` is held back — the text stream only covers the first
/// `len - 1` entries, and the caller decides what the `bos` position becomes.
fn codec_prefill(cfg: &PromptConfig, language: Option<u32>, speaker: Option<CodecSlot>) -> Vec<CodecSlot> {
    let mut out = Vec::with_capacity(7);
    match language {
        None => {
            out.push(CodecSlot::Id(cfg.codec_nothink_id));
            out.push(CodecSlot::Id(cfg.codec_think_bos_id));
            out.push(CodecSlot::Id(cfg.codec_think_eos_id));
        }
        Some(lang) => {
            out.push(CodecSlot::Id(cfg.codec_think_id));
            out.push(CodecSlot::Id(cfg.codec_think_bos_id));
            out.push(CodecSlot::Id(lang));
            out.push(CodecSlot::Id(cfg.codec_think_eos_id));
        }
    }
    if let Some(s) = speaker {
        out.push(s);
    }
    out.push(CodecSlot::Id(cfg.codec_pad_id));
    out.push(CodecSlot::Id(cfg.codec_bos_id));
    out
}

/// Everything up to (but not including) the first text-body position.
///
/// `talker_input_embed = cat(role_ids, tts_pad*(L-2) ++ tts_bos + codec_prefill[:-1])`.
/// Returns the steps and the held-back `codec_bos` slot.
fn prefix(
    cfg: &PromptConfig,
    role: &[u32],
    codec: &[CodecSlot],
) -> (Vec<PromptStep>, CodecSlot) {
    let mut steps: Vec<PromptStep> = role
        .iter()
        .map(|&id| PromptStep { text: TextSlot::Id(id), codec: CodecSlot::None })
        .collect();
    let n = codec.len() - 1; // the trailing codec_bos is held back
    for (i, &c) in codec[..n].iter().enumerate() {
        // tts_pad everywhere except the final prefill position, which carries tts_bos.
        let text = if i + 1 == n { cfg.tts_bos_id } else { cfg.tts_pad_id };
        steps.push(PromptStep { text: TextSlot::Id(text), codec: c });
    }
    (steps, codec[n])
}

/// The plain (non-ICL) body: `generate`'s `else` branch after the prefix.
fn text_body(
    cfg: &PromptConfig,
    steps: &mut Vec<PromptStep>,
    body: &[u32],
    codec_bos: CodecSlot,
    non_streaming: bool,
) -> Vec<TextSlot> {
    if non_streaming {
        // text[0..n] then tts_eos, each over codec_pad; then tts_pad over codec_bos.
        for &id in body {
            steps.push(PromptStep {
                text: TextSlot::Id(id),
                codec: CodecSlot::Id(cfg.codec_pad_id),
            });
        }
        steps.push(PromptStep {
            text: TextSlot::Id(cfg.tts_eos_id),
            codec: CodecSlot::Id(cfg.codec_pad_id),
        });
        steps.push(PromptStep { text: TextSlot::Id(cfg.tts_pad_id), codec: codec_bos });
        vec![TextSlot::Id(cfg.tts_pad_id)]
    } else {
        // Only the FIRST text token is prefilled (over codec_bos); the rest is fed one
        // per generated frame as trailing_text_hidden, then tts_eos.
        steps.push(PromptStep { text: TextSlot::Id(body[0]), codec: codec_bos });
        let mut trailing: Vec<TextSlot> = body[1..].iter().map(|&id| TextSlot::Id(id)).collect();
        trailing.push(TextSlot::Id(cfg.tts_eos_id));
        trailing
    }
}

/// `custom_voice`: target text + one of the nine preset timbres + optional instruction.
///
/// Reference: `Qwen3TTSModel.generate_custom_voice` →
/// `Qwen3TTSForConditionalGeneration.generate(input_ids, instruct_ids, languages,
/// speakers)`. `instruct` is dropped on the 0.6B checkpoint (see
/// [`PromptConfig::honors_instruct`]); an empty instruction is treated as none.
pub fn build_custom_voice(
    tok: &QwenTokenizer,
    cfg: &PromptConfig,
    text: &str,
    speaker: &str,
    instruct: Option<&str>,
    language: &str,
    non_streaming: bool,
) -> Result<PromptPlan, PromptError> {
    let spk = speaker.to_ascii_lowercase();
    let spk_id = *cfg
        .spk_id
        .get(&spk)
        .ok_or_else(|| PromptError::UnknownSpeaker(speaker.to_string()))?;
    check_language(cfg, language)?;

    let instruct = instruct.filter(|s| !s.is_empty() && cfg.honors_instruct());
    let lang = language_id(cfg, language, Some(&spk))?;
    let codec = codec_prefill(cfg, lang, Some(CodecSlot::Id(spk_id)));
    assemble_text_mode(tok, cfg, text, instruct, &codec, non_streaming)
}

/// `voice_design`: target text + a natural-language description of the voice.
///
/// Reference: `Qwen3TTSModel.generate_voice_design`. There is no speaker slot at all —
/// `speakers` is `None`, so `speaker_embed is None` and the codec prefill goes straight
/// from `think_eos` to `[pad, bos]`.
pub fn build_voice_design(
    tok: &QwenTokenizer,
    cfg: &PromptConfig,
    text: &str,
    instruct: &str,
    language: &str,
    non_streaming: bool,
) -> Result<PromptPlan, PromptError> {
    check_language(cfg, language)?;
    let lang = language_id(cfg, language, None)?;
    let codec = codec_prefill(cfg, lang, None);
    let instruct = (!instruct.is_empty()).then_some(instruct);
    assemble_text_mode(tok, cfg, text, instruct, &codec, non_streaming)
}

/// `voice_clone` (the `*-Base` checkpoints): target text + a reference clip.
///
/// Reference: `Qwen3TTSModel.generate_voice_clone` +
/// `Qwen3TTSForConditionalGeneration.generate_icl_prompt`. The speaker slot always
/// holds the extracted x-vector ([`CodecSlot::SpeakerVector`]); with
/// [`CloneRef::InContext`] the reference transcript and its RVQ frames are additionally
/// spliced in. This path takes no instruction — `Base` is not instructable.
pub fn build_voice_clone(
    tok: &QwenTokenizer,
    cfg: &PromptConfig,
    text: &str,
    reference: CloneRef<'_>,
    language: &str,
    non_streaming: bool,
) -> Result<PromptPlan, PromptError> {
    check_language(cfg, language)?;
    let lang = language_id(cfg, language, None)?;
    let codec = codec_prefill(cfg, lang, Some(CodecSlot::SpeakerVector));

    let target = assistant_text(tok, text)?;
    if target.body.is_empty() {
        return Err(PromptError::EmptyText);
    }
    let (mut steps, codec_bos) = prefix(cfg, &target.role, &codec);

    let (ref_text, frames) = match reference {
        CloneRef::XVectorOnly => {
            // Identical to the text modes: icl_mode is false, so `generate` falls into
            // the same `else` branch, with the x-vector as the speaker slot.
            let trailing = text_body(cfg, &mut steps, &target.body, codec_bos, non_streaming);
            return Ok(PromptPlan { steps, trailing_text: trailing, ref_frames: 0 });
        }
        CloneRef::InContext { ref_text, frames } => (ref_text, frames),
    };
    if ref_text.is_empty() || frames == 0 {
        return Err(PromptError::EmptyReference);
    }
    let reference = reference_text(tok, ref_text)?;
    if reference.body.is_empty() {
        return Err(PromptError::EmptyReference);
    }

    // generate_icl_prompt: text stream = ref body ++ target body ++ tts_eos …
    let mut text_stream: Vec<TextSlot> = reference
        .body
        .iter()
        .chain(target.body.iter())
        .map(|&id| TextSlot::Id(id))
        .collect();
    text_stream.push(TextSlot::Id(cfg.tts_eos_id));
    // … codec stream = codec_bos ++ one summed 16-group embedding per reference frame.
    // NOTE the prefix's held-back codec_bos is discarded on this path; generate_icl_prompt
    // prepends its own.
    let _ = codec_bos;
    let mut codec_stream: Vec<CodecSlot> = Vec::with_capacity(frames + 1);
    codec_stream.push(CodecSlot::Id(cfg.codec_bos_id));
    codec_stream.extend((0..frames).map(CodecSlot::RefFrame));

    let trailing = if non_streaming {
        // The two streams are laid END TO END, not aligned: T1 text-over-codec_pad
        // positions, then T2 tts_pad-over-codec positions.
        for t in &text_stream {
            steps.push(PromptStep { text: *t, codec: CodecSlot::Id(cfg.codec_pad_id) });
        }
        for c in &codec_stream {
            steps.push(PromptStep { text: TextSlot::Id(cfg.tts_pad_id), codec: *c });
        }
        vec![TextSlot::Id(cfg.tts_pad_id)]
    } else {
        // Aligned position by position; whichever stream is shorter is padded, and any
        // text left over becomes trailing_text_hidden.
        for (i, c) in codec_stream.iter().enumerate() {
            let text = text_stream.get(i).copied().unwrap_or(TextSlot::Id(cfg.tts_pad_id));
            steps.push(PromptStep { text, codec: *c });
        }
        if text_stream.len() > codec_stream.len() {
            text_stream[codec_stream.len()..].to_vec()
        } else {
            vec![TextSlot::Id(cfg.tts_pad_id)]
        }
    };
    Ok(PromptPlan { steps, trailing_text: trailing, ref_frames: frames })
}

/// Reject a language the wrapper's `_validate_languages` would reject. Dialect keys are
/// deliberately NOT accepted here: they are reachable only through a dialect speaker.
fn check_language(cfg: &PromptConfig, language: &str) -> Result<(), PromptError> {
    let lang = language.to_ascii_lowercase();
    if cfg.supported_languages().iter().any(|l| *l == lang) {
        Ok(())
    } else {
        Err(PromptError::UnknownLanguage(language.to_string()))
    }
}

/// The shared body of the two non-clone modes.
fn assemble_text_mode(
    tok: &QwenTokenizer,
    cfg: &PromptConfig,
    text: &str,
    instruct: Option<&str>,
    codec: &[CodecSlot],
    non_streaming: bool,
) -> Result<PromptPlan, PromptError> {
    let target = assistant_text(tok, text)?;
    if target.body.is_empty() {
        return Err(PromptError::EmptyText);
    }

    // The instruct block is prepended whole and is TEXT ONLY — `generate` appends
    // `text_projection(text_embedding(instruct_id))` with no codec addend.
    let mut steps = Vec::new();
    if let Some(instruct) = instruct {
        let ids = tok.encode(&format!(
            "{s}user\n{instruct}{e}\n",
            s = crate::tokenizer::IM_START,
            e = crate::tokenizer::IM_END
        ))?;
        steps.extend(ids.into_iter().map(|id| PromptStep {
            text: TextSlot::Id(id),
            codec: CodecSlot::None,
        }));
    }

    let (body_steps, codec_bos) = prefix(cfg, &target.role, codec);
    steps.extend(body_steps);
    let trailing = text_body(cfg, &mut steps, &target.body, codec_bos, non_streaming);
    Ok(PromptPlan { steps, trailing_text: trailing, ref_frames: 0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::tests::checkpoint_dir;

    fn load(name: &str) -> Option<(QwenTokenizer, PromptConfig)> {
        let any = checkpoint_dir()?;
        let dir = any.parent()?.join(name);
        if !dir.join("config.json").exists() {
            return None;
        }
        let tok = QwenTokenizer::from_dir(&dir).expect("tokenizer");
        let cfg = PromptConfig::from_json(&std::fs::read_to_string(dir.join("config.json")).unwrap())
            .expect("prompt config");
        Some((tok, cfg))
    }

    macro_rules! load_or_skip {
        ($name:expr) => {
            match load($name) {
                Some(x) => x,
                None => {
                    eprintln!("SKIP: no {} checkpoint (set SYRINX_QWEN_DIR)", $name);
                    return;
                }
            }
        };
    }

    const BASE: &str = "Qwen3-TTS-12Hz-0.6B-Base";
    const CUSTOM: &str = "Qwen3-TTS-12Hz-1.7B-CustomVoice";
    const DESIGN: &str = "Qwen3-TTS-12Hz-1.7B-VoiceDesign";

    fn texts(p: &PromptPlan) -> Vec<TextSlot> {
        p.steps.iter().map(|s| s.text).collect()
    }
    fn codecs(p: &PromptPlan) -> Vec<CodecSlot> {
        p.steps.iter().map(|s| s.codec).collect()
    }

    #[test]
    fn parses_the_prompt_ids_from_the_checkpoint() {
        let (_, cfg) = load_or_skip!(CUSTOM);
        assert_eq!(cfg.codec_pad_id, 2148);
        assert_eq!(cfg.codec_bos_id, 2149);
        assert_eq!(cfg.codec_eos_id, 2150);
        assert_eq!(cfg.codec_think_id, 2154);
        assert_eq!(cfg.codec_nothink_id, 2155);
        assert_eq!(cfg.codec_think_bos_id, 2156);
        assert_eq!(cfg.codec_think_eos_id, 2157);
        assert_eq!(cfg.num_code_groups, 16);
        assert_eq!(cfg.position_id_per_seconds, 13);
        assert_eq!(cfg.codec_language_id["english"], 2050);
        assert_eq!(cfg.codec_language_id["german"], 2053);
        assert_eq!(cfg.codec_language_id["chinese"], 2055);

        // Ten languages plus `auto`, and no Polish.
        let langs = cfg.supported_languages();
        assert_eq!(langs.len(), 11, "{langs:?}");
        assert!(langs.contains(&"auto".to_string()));
        assert!(!langs.iter().any(|l| l.contains("dialect")), "{langs:?}");
        assert!(!langs.contains(&"polish".to_string()));

        let mut spk = cfg.supported_speakers();
        spk.sort();
        assert_eq!(spk, PRESET_SPEAKERS);
        assert!(cfg.honors_instruct(), "1.7B CustomVoice is instructable");
    }

    #[test]
    fn the_06b_custom_voice_checkpoint_drops_the_instruction() {
        let (tok, cfg) = load_or_skip!("Qwen3-TTS-12Hz-0.6B-CustomVoice");
        assert!(!cfg.honors_instruct());
        let with = build_custom_voice(&tok, &cfg, "Hallo Welt.", "ryan", Some("be cheerful"), "german", true).unwrap();
        let without = build_custom_voice(&tok, &cfg, "Hallo Welt.", "ryan", None, "german", true).unwrap();
        assert_eq!(with, without, "0b6 must ignore `instruct`");
    }

    /// custom_voice, non-streaming, explicit language: the full layout, position by
    /// position, against `generate`'s prefill.
    #[test]
    fn custom_voice_prompt_shape() {
        let (tok, cfg) = load_or_skip!(CUSTOM);
        let text = "Hello, world!";
        let plan = build_custom_voice(&tok, &cfg, text, "ryan", None, "english", true).unwrap();

        let body = tok.encode(text).unwrap();
        let (p, e, b) = (cfg.tts_pad_id, cfg.tts_eos_id, cfg.tts_bos_id);
        let mut want_text = vec![
            TextSlot::Id(tok.im_start_id),
            TextSlot::Id(77091), // "assistant"
            TextSlot::Id(198),   // "\n"
            TextSlot::Id(p),     // codec_think
            TextSlot::Id(p),     // codec_think_bos
            TextSlot::Id(p),     // language id
            TextSlot::Id(p),     // codec_think_eos
            TextSlot::Id(p),     // speaker
            TextSlot::Id(b),     // codec_pad  <- tts_bos sits here
        ];
        want_text.extend(body.iter().map(|&i| TextSlot::Id(i)));
        want_text.push(TextSlot::Id(e));
        want_text.push(TextSlot::Id(p));
        assert_eq!(texts(&plan), want_text);

        let cp = CodecSlot::Id(cfg.codec_pad_id);
        let mut want_codec = vec![
            CodecSlot::None,
            CodecSlot::None,
            CodecSlot::None,
            CodecSlot::Id(cfg.codec_think_id),
            CodecSlot::Id(cfg.codec_think_bos_id),
            CodecSlot::Id(cfg.codec_language_id["english"]),
            CodecSlot::Id(cfg.codec_think_eos_id),
            CodecSlot::Id(cfg.spk_id["ryan"]),
            cp,
        ];
        want_codec.extend(std::iter::repeat(cp).take(body.len() + 1));
        want_codec.push(CodecSlot::Id(cfg.codec_bos_id));
        assert_eq!(codecs(&plan), want_codec);

        assert_eq!(plan.trailing_text, vec![TextSlot::Id(p)]);
        assert_eq!(plan.ref_frames, 0);
        assert_eq!(plan.len(), 3 + 6 + body.len() + 2);
    }

    /// `auto` drops the language id from the prefill and swaps think → nothink, which
    /// makes the whole prompt exactly one position shorter.
    #[test]
    fn auto_language_uses_the_nothink_prefill() {
        let (tok, cfg) = load_or_skip!(CUSTOM);
        let auto = build_custom_voice(&tok, &cfg, "Hello.", "ryan", None, "auto", true).unwrap();
        let en = build_custom_voice(&tok, &cfg, "Hello.", "ryan", None, "english", true).unwrap();
        assert_eq!(auto.len() + 1, en.len());
        assert_eq!(codecs(&auto)[3], CodecSlot::Id(cfg.codec_nothink_id));
        assert_eq!(codecs(&auto)[4], CodecSlot::Id(cfg.codec_think_bos_id));
        assert_eq!(codecs(&auto)[5], CodecSlot::Id(cfg.codec_think_eos_id));
        assert_eq!(codecs(&en)[3], CodecSlot::Id(cfg.codec_think_id));
        assert_eq!(codecs(&en)[5], CodecSlot::Id(cfg.codec_language_id["english"]));
    }

    /// `eric` is a Sichuan-dialect preset: with `chinese` OR `auto` the language slot
    /// becomes the dialect id, so even `auto` gets the 4-entry think prefill.
    #[test]
    fn a_dialect_speaker_overrides_the_language_slot() {
        let (tok, cfg) = load_or_skip!(CUSTOM);
        let sichuan = cfg.codec_language_id["sichuan_dialect"];
        for lang in ["chinese", "auto"] {
            let p = build_custom_voice(&tok, &cfg, "你好。", "eric", None, lang, true).unwrap();
            assert_eq!(codecs(&p)[3], CodecSlot::Id(cfg.codec_think_id), "{lang}");
            assert_eq!(codecs(&p)[5], CodecSlot::Id(sichuan), "{lang}");
        }
        // A non-dialect speaker, and a non-Chinese language, both leave it alone.
        let p = build_custom_voice(&tok, &cfg, "你好。", "ryan", None, "chinese", true).unwrap();
        assert_eq!(codecs(&p)[5], CodecSlot::Id(cfg.codec_language_id["chinese"]));
        let p = build_custom_voice(&tok, &cfg, "Hallo.", "eric", None, "german", true).unwrap();
        assert_eq!(codecs(&p)[5], CodecSlot::Id(cfg.codec_language_id["german"]));
    }

    /// The instruct block is prepended whole, text-only, and shifts nothing else.
    #[test]
    fn voice_design_prompt_shape_with_and_without_instruct() {
        let (tok, cfg) = load_or_skip!(DESIGN);
        let instruct = "A warm, low, unhurried narrator.";
        let plan = build_voice_design(&tok, &cfg, "Hello, world!", instruct, "english", true).unwrap();
        let bare = build_voice_design(&tok, &cfg, "Hello, world!", "", "english", true).unwrap();

        let ins = tok
            .encode(&format!("<|im_start|>user\n{instruct}<|im_end|>\n"))
            .unwrap();
        assert_eq!(plan.len(), bare.len() + ins.len());
        for (i, &id) in ins.iter().enumerate() {
            assert_eq!(plan.steps[i], PromptStep { text: TextSlot::Id(id), codec: CodecSlot::None });
        }
        assert_eq!(&plan.steps[ins.len()..], &bare.steps[..]);

        // VoiceDesign has no speaker slot at all: think_eos is followed straight by the
        // codec_pad that carries tts_bos.
        // role(3) + think, think_bos, language, think_eos, THEN codec_pad at index 7 —
        // one earlier than CustomVoice, which has a speaker slot in between.
        assert_eq!(codecs(&bare)[6], CodecSlot::Id(cfg.codec_think_eos_id));
        assert_eq!(codecs(&bare)[7], CodecSlot::Id(cfg.codec_pad_id));
        assert_eq!(texts(&bare)[7], TextSlot::Id(cfg.tts_bos_id));
        assert!(cfg.spk_id.is_empty());
    }

    /// Streaming mode prefills only the first text token and feeds the rest through
    /// `trailing_text_hidden`, one per generated frame.
    #[test]
    fn streaming_mode_moves_the_text_tail_into_trailing_text() {
        let (tok, cfg) = load_or_skip!(CUSTOM);
        let text = "Hello, world!";
        let body = tok.encode(text).unwrap();
        let s = build_custom_voice(&tok, &cfg, text, "ryan", None, "english", false).unwrap();
        let ns = build_custom_voice(&tok, &cfg, text, "ryan", None, "english", true).unwrap();

        assert_eq!(s.len(), 3 + 6 + 1);
        assert_eq!(s.steps[s.len() - 1], PromptStep {
            text: TextSlot::Id(body[0]),
            codec: CodecSlot::Id(cfg.codec_bos_id),
        });
        let mut want: Vec<TextSlot> = body[1..].iter().map(|&i| TextSlot::Id(i)).collect();
        want.push(TextSlot::Id(cfg.tts_eos_id));
        assert_eq!(s.trailing_text, want);
        // Non-streaming instead pads forever and carries the whole body in the prefill.
        assert_eq!(ns.trailing_text, vec![TextSlot::Id(cfg.tts_pad_id)]);
        assert!(ns.len() > s.len());
    }

    /// x-vector-only clone: no reference frames, the ordinary text layout, and the
    /// speaker slot holds the extracted vector rather than a codec id.
    #[test]
    fn voice_clone_x_vector_only_prompt_shape() {
        let (tok, cfg) = load_or_skip!(BASE);
        let plan =
            build_voice_clone(&tok, &cfg, "Hello, world!", CloneRef::XVectorOnly, "english", false).unwrap();
        assert_eq!(plan.ref_frames, 0);
        assert_eq!(codecs(&plan)[7], CodecSlot::SpeakerVector);
        assert!(!codecs(&plan).iter().any(|c| matches!(c, CodecSlot::RefFrame(_))));
        // `generate_voice_clone` defaults to non_streaming_mode=False.
        assert!(!VOICE_CLONE_NON_STREAMING);
        assert_eq!(plan.len(), 3 + 6 + 1);
    }

    /// In-context clone, streaming (the reference default): the reference transcript
    /// and target text ride the text stream while codec_bos + the reference frames ride
    /// the codec stream, aligned position by position.
    #[test]
    fn voice_clone_in_context_prompt_shape_streaming() {
        let (tok, cfg) = load_or_skip!(BASE);
        let (text, ref_text, frames) = ("Hello, world!", "Dies ist der Referenztext.", 40);
        let plan = build_voice_clone(
            &tok,
            &cfg,
            text,
            CloneRef::InContext { ref_text, frames },
            "english",
            false,
        )
        .unwrap();

        let rb = tok.encode(&format!("<|im_start|>assistant\n{ref_text}<|im_end|>\n")).unwrap();
        let rb = &rb[3..rb.len() - 2];
        let tb = tok.encode(text).unwrap();
        let t1 = rb.len() + tb.len() + 1; // ref ++ target ++ tts_eos
        let t2 = frames + 1; // codec_bos ++ frames
        assert!(t2 > t1, "this fixture exercises the codec-longer branch");

        assert_eq!(plan.ref_frames, frames);
        assert_eq!(plan.len(), 3 + 6 + t2);
        // The prefix's speaker slot is the x-vector.
        assert_eq!(codecs(&plan)[7], CodecSlot::SpeakerVector);
        // Then codec_bos, then one RefFrame per reference frame.
        assert_eq!(codecs(&plan)[9], CodecSlot::Id(cfg.codec_bos_id));
        for i in 0..frames {
            assert_eq!(codecs(&plan)[10 + i], CodecSlot::RefFrame(i));
        }
        // The text stream is ref body, target body, tts_eos, then tts_pad filler.
        let ts = texts(&plan);
        assert_eq!(ts[9], TextSlot::Id(rb[0]));
        assert_eq!(ts[9 + rb.len()], TextSlot::Id(tb[0]));
        assert_eq!(ts[9 + t1 - 1], TextSlot::Id(cfg.tts_eos_id));
        assert_eq!(ts[9 + t1], TextSlot::Id(cfg.tts_pad_id));
        assert_eq!(plan.trailing_text, vec![TextSlot::Id(cfg.tts_pad_id)]);
    }

    /// When the text stream outruns the codec stream the overflow becomes
    /// trailing_text_hidden instead of being padded away.
    #[test]
    fn voice_clone_in_context_streaming_spills_long_text_into_trailing() {
        let (tok, cfg) = load_or_skip!(BASE);
        let text = "This is a deliberately long target sentence, long enough that its \
                    tokens outnumber the handful of reference frames we hand the model.";
        let ref_text = "Kurzer Referenztext.";
        let frames = 3;
        let plan = build_voice_clone(
            &tok,
            &cfg,
            text,
            CloneRef::InContext { ref_text, frames },
            "english",
            false,
        )
        .unwrap();
        let rb = tok.encode(&format!("<|im_start|>assistant\n{ref_text}<|im_end|>\n")).unwrap();
        let rb = &rb[3..rb.len() - 2];
        let tb = tok.encode(text).unwrap();
        let t1 = rb.len() + tb.len() + 1;
        let t2 = frames + 1;
        assert!(t1 > t2, "this fixture exercises the text-longer branch");
        assert_eq!(plan.len(), 3 + 6 + t2);
        assert_eq!(plan.trailing_text.len(), t1 - t2);
        assert_eq!(plan.trailing_text[plan.trailing_text.len() - 1], TextSlot::Id(cfg.tts_eos_id));
    }

    /// Non-streaming ICL lays the two streams END TO END, so the prompt is T1 + T2 long
    /// rather than max(T1, T2).
    #[test]
    fn voice_clone_in_context_non_streaming_concatenates_the_streams() {
        let (tok, cfg) = load_or_skip!(BASE);
        let (text, ref_text, frames) = ("Hello, world!", "Dies ist der Referenztext.", 40);
        let plan = build_voice_clone(
            &tok,
            &cfg,
            text,
            CloneRef::InContext { ref_text, frames },
            "english",
            true,
        )
        .unwrap();
        let rb = tok.encode(&format!("<|im_start|>assistant\n{ref_text}<|im_end|>\n")).unwrap();
        let rb = &rb[3..rb.len() - 2];
        let tb = tok.encode(text).unwrap();
        let (t1, t2) = (rb.len() + tb.len() + 1, frames + 1);
        assert_eq!(plan.len(), 3 + 6 + t1 + t2);
        // The first T1 positions are text over codec_pad …
        for i in 0..t1 {
            assert_eq!(codecs(&plan)[9 + i], CodecSlot::Id(cfg.codec_pad_id), "pos {i}");
        }
        // … the next T2 are tts_pad over the codec stream.
        for i in 0..t2 {
            assert_eq!(texts(&plan)[9 + t1 + i], TextSlot::Id(cfg.tts_pad_id), "pos {i}");
        }
        assert_eq!(codecs(&plan)[9 + t1], CodecSlot::Id(cfg.codec_bos_id));
        assert_eq!(codecs(&plan)[9 + t1 + 1], CodecSlot::RefFrame(0));
        assert_eq!(plan.trailing_text, vec![TextSlot::Id(cfg.tts_pad_id)]);
    }

    #[test]
    fn rejects_unsupported_language_speaker_and_empty_inputs() {
        let (tok, cfg) = load_or_skip!(CUSTOM);
        assert!(matches!(
            build_custom_voice(&tok, &cfg, "Cześć.", "ryan", None, "polish", true),
            Err(PromptError::UnknownLanguage(_))
        ));
        // A dialect key is in codec_language_id but NOT in supported_languages.
        assert!(matches!(
            build_custom_voice(&tok, &cfg, "你好。", "ryan", None, "sichuan_dialect", true),
            Err(PromptError::UnknownLanguage(_))
        ));
        assert!(matches!(
            build_custom_voice(&tok, &cfg, "Hello.", "nobody", None, "english", true),
            Err(PromptError::UnknownSpeaker(_))
        ));
        assert!(matches!(
            build_custom_voice(&tok, &cfg, "", "ryan", None, "english", true),
            Err(PromptError::EmptyText)
        ));
        // Speaker and language lookups are case-insensitive, as in the reference.
        assert!(build_custom_voice(&tok, &cfg, "Hello.", "Ryan", None, "English", true).is_ok());
    }

    #[test]
    fn in_context_clone_requires_a_reference() {
        let (tok, cfg) = load_or_skip!(BASE);
        for r in [
            CloneRef::InContext { ref_text: "", frames: 10 },
            CloneRef::InContext { ref_text: "Referenz.", frames: 0 },
        ] {
            assert!(matches!(
                build_voice_clone(&tok, &cfg, "Hello.", r, "english", false),
                Err(PromptError::EmptyReference)
            ));
        }
    }
}

