//! The cross-file checks — the ones neither the YAML nor the tokenizer JSONs can make
//! alone.
//!
//! [`T3Config`] knows how wide the text embedding table is. [`TokenizerContract`] knows
//! how many base BPE ids there are and how many tags sit above them. Only together can
//! they say whether those two agree, and a disagreement means the downloaded files come
//! from different revisions — the failure that would otherwise surface as a silent
//! off-by-nineteen in every tag id.

use crate::config::T3Config;
use crate::tokenizer::TokenizerContract;

/// A [`T3Config`] and a [`TokenizerContract`] that have been checked against each other.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelContract {
    pub config: T3Config,
    pub tokenizer: TokenizerContract,
}

impl ModelContract {
    /// Cross-check the model geometry against the tokenizer files.
    ///
    /// Rejects a `text_tokens_dict_size` that is not exactly base vocabulary + tags, and
    /// a text sequence marker that has strayed up into the tag block (where it would
    /// collide with a paralinguistic tag rather than reusing a base BPE id).
    pub fn new(config: T3Config, tokenizer: TokenizerContract) -> Result<Self, String> {
        let implied = tokenizer.implied_text_vocab_size();
        if config.text.dict_size != implied {
            return Err(format!(
                "`text_tokens_dict_size` {} != base vocabulary {} + {} tags = {implied}",
                config.text.dict_size,
                tokenizer.base_vocab_size,
                tokenizer.tags.len()
            ));
        }

        for (what, id) in [
            ("start_text_token", config.text.start_token),
            ("stop_text_token", config.text.stop_token),
        ] {
            if id as usize >= tokenizer.base_vocab_size {
                return Err(format!(
                    "`{what}` {id} is not below the base vocabulary size {}; it would \
                     collide with a paralinguistic tag",
                    tokenizer.base_vocab_size
                ));
            }
        }

        Ok(Self { config, tokenizer })
    }

    /// How many paralinguistic tags this checkpoint carries.
    pub fn tag_count(&self) -> usize {
        self.tokenizer.tags.len()
    }
}
