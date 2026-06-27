//! The s1 `FishTokenizer` over the Fish tiktoken vocabulary.
//!
//! Ports `fish_speech/tokenizer.py`: a wrapper over the model's tokenizer that pins
//! the **`allowed_special="all"`** behavior (special markers — `<|im_start|>`,
//! `<|im_end|>`, `<|voice|>`, the 4096 `<|semantic:i|>` ids — are parsed inline as
//! atomic tokens, never BPE-split) and exposes the semantic-id range the dual-AR
//! driver needs.
//!
//! ## File format
//! Fish ships `tokenizer.tiktoken` + `special_tokens.json`. The Hugging Face
//! `tokenizers` crate consumes a serialized `tokenizer.json`; the on-box conversion
//! step (`AutoTokenizer.from_pretrained(...).backend_tokenizer.save("tokenizer.json")`)
//! produces it with every special token (incl. the `<|semantic:i|>` range) registered
//! as an added token. We load that file. PARITY: confirm the converted `tokenizer.json`
//! reproduces the tiktoken ids 1:1 on-box (the `<|semantic:0|>`/`im_end` ids in
//! particular, which seed the driver's semantic constraint + stop).

use std::path::Path;

use tokenizers::Tokenizer;

/// `<|im_start|>` — role-turn opener.
pub const IM_START_TOKEN: &str = "<|im_start|>";
/// `<|im_end|>` — the end-of-utterance stop token.
pub const IM_END_TOKEN: &str = "<|im_end|>";
/// `<|voice|>` — the voice-modality marker prepended to the assistant turn.
pub const VOICE_TOKEN: &str = "<|voice|>";
/// The number of contiguous `<|semantic:i|>` ids in the Fish vocabulary.
pub const N_SEMANTIC: usize = 4096;

/// A loaded Fish tokenizer with its resolved special-id layout.
pub struct FishTokenizer {
    inner: Tokenizer,
    /// First `<|semantic:0|>` slow-vocab id.
    pub semantic_begin_id: u32,
    /// Last `<|semantic:4095|>` slow-vocab id (inclusive).
    pub semantic_end_id: u32,
    /// `<|im_end|>` slow-vocab id (the driver's stop token).
    pub im_end_id: u32,
}

impl FishTokenizer {
    /// Load from a serialized Hugging Face `tokenizer.json` (the converted Fish
    /// tiktoken vocabulary). Resolves the semantic range + `<|im_end|>` id from the
    /// added-token table, mirroring `FishTokenizer.__init__`.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let inner = Tokenizer::from_file(path.as_ref())
            .map_err(|e| TokenizerError::Load(e.to_string()))?;

        // Resolve the contiguous `<|semantic:i|>` ids (reference scans i in 0..4096).
        let mut valid: Vec<u32> = Vec::with_capacity(N_SEMANTIC);
        for i in 0..N_SEMANTIC {
            if let Some(id) = inner.token_to_id(&format!("<|semantic:{i}|>")) {
                valid.push(id);
            }
        }
        if valid.is_empty() {
            return Err(TokenizerError::MissingSpecial(
                "no <|semantic:i|> tokens found in vocab".to_string(),
            ));
        }
        let semantic_begin_id = *valid.iter().min().unwrap();
        let semantic_end_id = *valid.iter().max().unwrap();

        let im_end_id = inner
            .token_to_id(IM_END_TOKEN)
            .ok_or_else(|| TokenizerError::MissingSpecial(IM_END_TOKEN.to_string()))?;

        Ok(Self {
            inner,
            semantic_begin_id,
            semantic_end_id,
            im_end_id,
        })
    }

    /// Look up a single token's id (the reference `get_token_id`).
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Encode `text` to slow-vocab token ids, **with inline special tokens honored**
    /// and no automatic BOS/EOS (the reference `encode(..., add_special_tokens=False,
    /// allowed_special="all")`). Inline emotion/style tags like `(happy)` are ordinary
    /// text and flow through BPE unchanged — there is no special-token path for them.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| TokenizerError::Encode(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode token ids back to text (used for debugging / round-trips).
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, false)
            .map_err(|e| TokenizerError::Encode(e.to_string()))
    }
}

/// Errors loading or running the Fish tokenizer.
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
