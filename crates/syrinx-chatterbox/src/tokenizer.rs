//! The text tokenizer contract, from `added_tokens.json`, `tokenizer_config.json` and
//! `special_tokens_map.json`.
//!
//! Chatterbox Turbo's text side is a stock GPT-2 BPE with nineteen **paralinguistic
//! tags** appended above the base vocabulary. That is the whole of the model's expressive
//! channel, and it is the reason this backend is interesting at all
//! (`docs/backends/CHATTERBOX_PORT_SCOPE.md`).
//!
//! Two files describe those tags and they must agree: `added_tokens.json` maps tag text
//! to id, and `tokenizer_config.json`'s `added_tokens_decoder` maps id back to tag text
//! *and* marks which entries are `special`. [`TokenizerContract::from_files`] checks them
//! against each other, and checks that the tags form one unbroken run starting exactly at
//! the base vocabulary size — the property that lets a port say "id - base_vocab_size" is
//! a tag index without a lookup table, and the property a swapped or truncated file would
//! break.
//!
//! # What this module deliberately does not decide
//!
//! It does not classify the tags. Grouping them into emotions, styles and events is an
//! editorial judgement — **the model card states no taxonomy** — and the label vocabulary
//! is `syrinx-cue`'s responsibility, not a backend crate's. This module reports the
//! nineteen tags exactly as they ship, in id order.
//!
//! It also does not tokenize. `vocab.json` (999 KB) and `merges.txt` (456 KB) carry the
//! base BPE and are not committed here; the tag ids above the base are all Phase 0 needs.

use std::collections::BTreeMap;

use serde_json::Value;

/// One paralinguistic tag exactly as the checkpoint spells it, with its token id.
///
/// The spelling is the backend's own — `[whispering]`, not `[whisper]`; `[clear throat]`
/// with a space. Mapping a canonical `syrinx-cue` label onto it is `syrinx-cue`'s job;
/// nothing here may be passed through from user text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tag {
    pub text: String,
    pub id: u32,
}

/// What the three tokenizer files jointly assert.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerContract {
    /// `tokenizer_class` — `GPT2Tokenizer`.
    pub tokenizer_class: String,
    /// The one special token, `<|endoftext|>`, which serves as bos, eos, pad and unk.
    pub eot_text: String,
    /// Its id, 50256 — the last id of the GPT-2 base vocabulary.
    pub eot_id: u32,
    /// `eot_id + 1`: the number of base BPE ids, and the id the first tag sits at.
    pub base_vocab_size: usize,
    /// The nineteen tags, in ascending id order.
    pub tags: Vec<Tag>,
    /// `model_max_length` — 1024. This is the *tokenizer's* stock GPT-2 default, not the
    /// model's bound; T3's text budget is `max_text_tokens: 402`. Recorded so the two are
    /// not confused.
    pub model_max_length: usize,
    /// `add_bos_token` — false. T3 supplies `start_text_token` itself, so a tokenizer
    /// that prepended one would double it.
    pub add_bos_token: bool,
    /// `add_prefix_space` — false. A port that inserts one tokenizes differently.
    pub add_prefix_space: bool,
}

impl TokenizerContract {
    /// Parse and cross-check the three shipped tokenizer files.
    pub fn from_files(
        added_tokens_json: &str,
        tokenizer_config_json: &str,
        special_tokens_map_json: &str,
    ) -> Result<Self, String> {
        let added: Value = serde_json::from_str(added_tokens_json)
            .map_err(|e| format!("added_tokens.json: {e}"))?;
        let tcfg: Value = serde_json::from_str(tokenizer_config_json)
            .map_err(|e| format!("tokenizer_config.json: {e}"))?;
        let smap: Value = serde_json::from_str(special_tokens_map_json)
            .map_err(|e| format!("special_tokens_map.json: {e}"))?;

        let tokenizer_class = string_field(&tcfg, "tokenizer_class", "tokenizer_config.json")?;
        let model_max_length = usize_field(&tcfg, "model_max_length", "tokenizer_config.json")?;
        let add_bos_token = bool_field(&tcfg, "add_bos_token", "tokenizer_config.json")?;
        let add_prefix_space = bool_field(&tcfg, "add_prefix_space", "tokenizer_config.json")?;

        // ---- the decoder: split the one special from the tags -------------------------
        let decoder = tcfg
            .get("added_tokens_decoder")
            .and_then(Value::as_object)
            .ok_or("tokenizer_config.json: no `added_tokens_decoder` object")?;

        let mut specials: Vec<(u32, String)> = Vec::new();
        let mut decoder_tags: BTreeMap<String, u32> = BTreeMap::new();
        for (id_str, entry) in decoder {
            let id: u32 = id_str
                .parse()
                .map_err(|_| format!("added_tokens_decoder: {id_str:?} is not an id"))?;
            let content = entry
                .get("content")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("added_tokens_decoder[{id}]: no `content`"))?;
            let special = entry
                .get("special")
                .and_then(Value::as_bool)
                .ok_or_else(|| format!("added_tokens_decoder[{id}]: no `special`"))?;
            if special {
                specials.push((id, content.to_string()));
            } else {
                decoder_tags.insert(content.to_string(), id);
            }
        }

        if specials.len() != 1 {
            return Err(format!(
                "tokenizer_config.json: expected exactly one special token, found {}",
                specials.len()
            ));
        }
        let (eot_id, eot_text) = specials.remove(0);
        if eot_text != EOT {
            return Err(format!(
                "tokenizer_config.json: the special token is {eot_text:?}, expected {EOT:?}"
            ));
        }

        // ---- the two tag listings must be the same map --------------------------------
        let added_map: BTreeMap<String, u32> = added
            .as_object()
            .ok_or("added_tokens.json: not an object")?
            .iter()
            .map(|(k, v)| {
                let id = v
                    .as_u64()
                    .ok_or_else(|| format!("added_tokens.json[{k}]: not an id"))?;
                Ok((k.clone(), id as u32))
            })
            .collect::<Result<_, String>>()?;

        if added_map != decoder_tags {
            return Err(format!(
                "added_tokens.json ({} tags) and added_tokens_decoder ({} non-special \
                 entries) do not describe the same tag set",
                added_map.len(),
                decoder_tags.len()
            ));
        }

        // ---- contiguity, anchored at the base vocabulary size -------------------------
        let base_vocab_size = eot_id as usize + 1;
        let mut tags: Vec<Tag> = added_map
            .into_iter()
            .map(|(text, id)| Tag { text, id })
            .collect();
        tags.sort_by_key(|t| t.id);
        for (i, t) in tags.iter().enumerate() {
            if t.id as usize != base_vocab_size + i {
                return Err(format!(
                    "tag {:?} has id {} but the tag run must start at \
                     {base_vocab_size}; the ids are not one unbroken block above the \
                     base vocabulary",
                    t.text, t.id
                ));
            }
        }

        // ---- special_tokens_map: bos = eos = pad = unk = the one special --------------
        for key in SPECIAL_ROLES {
            let text = smap
                .get(key)
                .and_then(token_text)
                .ok_or_else(|| format!("special_tokens_map.json: no `{key}`"))?;
            if text != eot_text {
                return Err(format!(
                    "special_tokens_map.json: `{key}` is {text:?}, expected {eot_text:?}"
                ));
            }
        }

        Ok(Self {
            tokenizer_class,
            eot_text,
            eot_id,
            base_vocab_size,
            tags,
            model_max_length,
            add_bos_token,
            add_prefix_space,
        })
    }

    /// The id of a tag spelled exactly as the checkpoint spells it.
    pub fn tag_id(&self, text: &str) -> Option<u32> {
        self.tags.iter().find(|t| t.text == text).map(|t| t.id)
    }

    /// Total text vocabulary the tokenizer files imply: base BPE plus the tags. This is
    /// the number `text_tokens_dict_size` must equal — checked by
    /// [`crate::contract::ModelContract`].
    pub fn implied_text_vocab_size(&self) -> usize {
        self.base_vocab_size + self.tags.len()
    }
}

/// The single special token, which fills every role.
pub const EOT: &str = "<|endoftext|>";

/// The roles `special_tokens_map.json` assigns, all to [`EOT`].
pub const SPECIAL_ROLES: [&str; 4] = ["bos_token", "eos_token", "pad_token", "unk_token"];

/// A `special_tokens_map.json` entry is either a bare string (`pad_token`) or an
/// `AddedToken` object with a `content` field (the other three). Both forms ship in the
/// same file, so both arms are live.
fn token_text(v: &Value) -> Option<&str> {
    match v.as_str() {
        Some(s) => Some(s),
        None => v.get("content").and_then(Value::as_str),
    }
}

fn string_field(v: &Value, key: &str, what: &str) -> Result<String, String> {
    v.get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("{what}: no string `{key}`"))
}

fn usize_field(v: &Value, key: &str, what: &str) -> Result<usize, String> {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .ok_or_else(|| format!("{what}: no integer `{key}`"))
}

fn bool_field(v: &Value, key: &str, what: &str) -> Result<bool, String> {
    v.get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{what}: no boolean `{key}`"))
}
