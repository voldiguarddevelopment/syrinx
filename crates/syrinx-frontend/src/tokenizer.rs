//! Real CosyVoice2 text tokenizer — the Qwen2 BPE / tiktoken front-half of the
//! pipeline (the `text -> token ids` step that `cli/frontend.py`'s
//! `_extract_text_token` performs via `self.tokenizer.encode`).
//!
//! Gated behind the crate's `real` feature: it pulls in the Hugging Face
//! `tokenizers` crate and loads the model's serialized `tokenizer.json` (the
//! Qwen2 BPE merges + vocab + the CosyVoice2 *added* special tokens —
//! `<|endofprompt|>`, `[breath]`, `<strong>`, ... `[mn]`). The default,
//! Candle-free build of the crate does not compile this module.
//!
//! Parity contract (mirrors `CosyVoice2Tokenizer.encode`):
//!   * encoding is `tokenizer([text])["input_ids"][0]` — i.e. the BPE token ids
//!     for the (already-normalized) input, with the added special tokens split
//!     out to their own ids, and **no** BOS/EOS/template tokens injected;
//!   * the special markers are recognized as atomic tokens, not BPE-split.

use std::path::Path;

use tokenizers::Tokenizer;

/// Loaded CosyVoice2 text tokenizer.
///
/// Thin wrapper over the Hugging Face fast tokenizer loaded from the model's
/// `tokenizer.json`; the only behavior it pins is the no-special-token-injection
/// encode that matches the Python reference.
pub struct TextTokenizer {
    inner: Tokenizer,
}

impl TextTokenizer {
    /// Load the tokenizer from a serialized Hugging Face `tokenizer.json`.
    ///
    /// This is the file the model ships (or that is produced by
    /// `save_pretrained`/`backend_tokenizer.save` after the CosyVoice2 special
    /// tokens are added); it is fully self-contained — vocab, merges, the
    /// pre-tokenizer, and every added special token with its id.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let inner = Tokenizer::from_file(path.as_ref())
            .map_err(|e| TokenizerError::Load(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Encode `text` to token ids, reproducing `CosyVoice2Tokenizer.encode`.
    ///
    /// The added special tokens are honored (they remain atomic ids); no
    /// BOS/EOS or chat-template tokens are added, matching the plain
    /// `tokenizer([text])` call on the Python side.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        // `add_special_tokens = false`: the Qwen2 tokenizer adds no automatic
        // BOS/EOS, and `CosyVoice2Tokenizer.encode` performs a bare call with no
        // template — so we must NOT have the encoder inject any. The *recognition*
        // of the added special markers (e.g. `[breath]`) is a property of the
        // tokenizer's added-token table and is unaffected by this flag.
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| TokenizerError::Encode(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }
}

/// Errors raised loading or running the text tokenizer.
#[derive(Debug)]
pub enum TokenizerError {
    /// The `tokenizer.json` could not be read or parsed.
    Load(String),
    /// Encoding a string failed inside the Hugging Face tokenizer.
    Encode(String),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::Load(m) => write!(f, "failed to load tokenizer.json: {m}"),
            TokenizerError::Encode(m) => write!(f, "failed to encode text: {m}"),
        }
    }
}

impl std::error::Error for TokenizerError {}
