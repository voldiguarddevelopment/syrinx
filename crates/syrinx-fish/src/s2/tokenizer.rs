//! The s2 Qwen3 tokenizer over the `fish_qwen3_omni` 155 776-entry BPE vocabulary.
//!
//! Loads the published `tokenizer.json` via the Hugging Face `tokenizers` crate (Qwen3
//! is a byte-level BPE) and pins the structural special tokens the dual-AR driver needs.
//! Per `s2-pro/config.json` the authoritative ids are:
//!   * `semantic_start_token_id = 151678`, `semantic_end_token_id = 155773` (the 4096
//!     `<|semantic:i|>` ids; `155773 − 151678 + 1 == 4096`),
//!   * `eos_token_id = 151645` (`<|im_end|>`) — the end-of-utterance stop.
//!
//! Emotion/style tags (`[whisper]`, `[angry]`, `(happy)`) are **plain text** — they flow
//! through BPE like any other characters; there is no special-token path for them.
//!
//! PARITY: confirm the converted `tokenizer.json` reproduces the Qwen3 ids 1:1 on-box,
//! and that `<|semantic:i|>` are registered as atomic added tokens over `[151678,
//! 155773]`. The backend additionally injects the config-provided semantic range + stop
//! id at load (so a `tokenizer.json` lacking the explicit `<|semantic:i|>` table still
//! drives correctly).

use std::path::Path;

use tokenizers::Tokenizer;

/// `<|im_start|>` — role-turn opener.
pub const IM_START_TOKEN: &str = "<|im_start|>";
/// `<|im_end|>` — the end-of-utterance stop token (Qwen3 `eos`, id 151645).
pub const IM_END_TOKEN: &str = "<|im_end|>";
/// `<|voice|>` — the audio/voice modality marker placed after `assistant\n`; generation
/// of semantic tokens begins immediately after it (reference `fish_qwen3_omni` prompt).
/// PARITY: confirm `<|voice|>` is a registered added-token (single id) in the on-box
/// `tokenizer.json`; if it is NOT, it will be split into BPE pieces here.
pub const VOICE_TOKEN: &str = "<|voice|>";
/// `<|speaker:0|>` — the speaker tag prepended to the reference transcript in the cloning
/// prompt. PARITY: confirm it resolves to a single id on-box.
pub const SPEAKER0_TOKEN: &str = "<|speaker:0|>";
/// The number of contiguous `<|semantic:i|>` ids in the s2 vocabulary.
pub const N_SEMANTIC: usize = 4096;
/// config.json `semantic_start_token_id` — the fallback `<|semantic:0|>` id.
pub const SEMANTIC_START_ID: u32 = 151678;
/// config.json `semantic_end_token_id` — the fallback `<|semantic:4095|>` id (inclusive).
pub const SEMANTIC_END_ID: u32 = 155773;
/// config.json `eos_token_id` (`<|im_end|>`).
pub const EOS_TOKEN_ID: u32 = 151645;

/// A loaded Qwen3 tokenizer with its resolved special-id layout.
pub struct Qwen3Tokenizer {
    inner: Tokenizer,
    /// First `<|semantic:0|>` slow-vocab id.
    pub semantic_begin_id: u32,
    /// Last `<|semantic:4095|>` slow-vocab id (inclusive).
    pub semantic_end_id: u32,
    /// `<|im_end|>` slow-vocab id (the driver's stop token).
    pub im_end_id: u32,
}

impl Qwen3Tokenizer {
    /// Load from the published Qwen3 `tokenizer.json`. Resolves the semantic range +
    /// `<|im_end|>` id from the vocabulary, falling back to the config.json constants
    /// when the `<|semantic:i|>` added-token table is absent.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let inner =
            Tokenizer::from_file(path.as_ref()).map_err(|e| TokenizerError::Load(e.to_string()))?;

        // Resolve the contiguous `<|semantic:i|>` ids if present; else use config ids.
        let mut valid: Vec<u32> = Vec::with_capacity(N_SEMANTIC);
        for i in 0..N_SEMANTIC {
            if let Some(id) = inner.token_to_id(&format!("<|semantic:{i}|>")) {
                valid.push(id);
            }
        }
        let (semantic_begin_id, semantic_end_id) = if valid.is_empty() {
            (SEMANTIC_START_ID, SEMANTIC_END_ID)
        } else {
            (*valid.iter().min().unwrap(), *valid.iter().max().unwrap())
        };

        let im_end_id = inner.token_to_id(IM_END_TOKEN).unwrap_or(EOS_TOKEN_ID);

        Ok(Self {
            inner,
            semantic_begin_id,
            semantic_end_id,
            im_end_id,
        })
    }

    /// Look up a single token's id.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Encode `text` to slow-vocab token ids with inline special tokens honored and no
    /// automatic BOS/EOS (Qwen3 chat strings carry their own `<|im_start|>`/`<|im_end|>`).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| TokenizerError::Encode(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode token ids back to text (debugging / round-trips).
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, false)
            .map_err(|e| TokenizerError::Encode(e.to_string()))
    }
}

/// Errors loading or running the Qwen3 tokenizer.
#[derive(Debug)]
pub enum TokenizerError {
    /// The `tokenizer.json` could not be read or parsed.
    Load(String),
    /// Encoding / decoding failed inside the Hugging Face tokenizer.
    Encode(String),
    /// A required special token was absent from the vocabulary.
    MissingSpecial(String),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::Load(m) => write!(f, "failed to load tokenizer.json: {m}"),
            TokenizerError::Encode(m) => write!(f, "tokenizer encode/decode failed: {m}"),
            TokenizerError::MissingSpecial(m) => write!(f, "missing special token: {m}"),
        }
    }
}

impl std::error::Error for TokenizerError {}
