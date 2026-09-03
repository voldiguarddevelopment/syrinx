//! The Qwen3-TTS **text** tokenizer — a Qwen2 byte-level BPE, built from the
//! checkpoint's `vocab.json` + `merges.txt` + `tokenizer_config.json`.
//!
//! ## Why this file builds the tokenizer instead of loading one
//!
//! The published Qwen3-TTS checkpoints ship **no `tokenizer.json`** — only
//! `vocab.json` (151 665 entries), `merges.txt` (151 387 rules) and
//! `tokenizer_config.json` (`tokenizer_class = "Qwen2Tokenizer"`, 33 added tokens).
//! The sibling Fish port needed an on-box tiktoken→HF conversion script for the same
//! reason (`syrinx_fish::s1::tokenizer`); here no conversion is needed, because the
//! three files are exactly the inputs Hugging Face's own `Qwen2Converter` consumes.
//! So we assemble the equivalent serialized tokenizer in memory and hand it to
//! `Tokenizer::from_str` — the same deserialization path a `tokenizer.json` takes.
//!
//! ## What the assembled spec reproduces (verbatim from the reference)
//!
//! `transformers/convert_slow_tokenizer.py::Qwen2Converter.converted()` (v4.57.3, the
//! version in `~/.venvs/qwen`) builds:
//!
//! ```text
//! BPE(vocab, merges, dropout=None, unk_token=None, continuing_subword_prefix="",
//!     end_of_word_suffix="", fuse_unk=False, byte_fallback=False)
//! normalizer      = NFC()
//! pre_tokenizer   = Sequence([ Split(PRETOKENIZE_REGEX, behavior="isolated", invert=False),
//!                              ByteLevel(add_prefix_space=<tokenizer_config>, use_regex=False) ])
//! decoder         = ByteLevel()
//! post_processor  = ByteLevel(trim_offsets=False)
//! ```
//!
//! and `transformers/models/qwen2/tokenization_qwen2.py::PRETOKENIZE_REGEX` is
//! [`QWEN2_PRETOKENIZE_REGEX`] below. Cross-checked against the live object: the
//! `backend_tokenizer.to_str()` of `AutoTokenizer.from_pretrained("~/models/
//! Qwen3-TTS-12Hz-0.6B-Base")` serializes to exactly this shape (`version "1.0"`,
//! `normalizer {"type":"NFC"}`, the Sequence pre-tokenizer above, ByteLevel
//! post-processor with `trim_offsets:false`, ByteLevel decoder, 33 added tokens).
//!
//! ## The one genuine ambiguity: `fix_mistral_regex`
//!
//! `Qwen3TTSModel.from_pretrained` (`qwen_tts/inference/qwen3_tts_model.py`) loads the
//! processor as `AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)`.
//! With transformers ≥ 4.57.3 that flag reaches
//! `PreTrainedTokenizerBase._patch_mistral_regex`, which — for ANY *local* model
//! directory whose `config.json` records `transformers_version > 4.57.2`, which every
//! Qwen3-TTS checkpoint does — **replaces the pre-tokenizer's regex** with the
//! Mistral-corrected one ([`MISTRAL_FIXED_PRETOKENIZE_REGEX`]). That is almost
//! certainly accidental on Alibaba's part (the flag exists to silence a Mistral
//! warning), but it is what the reference wrapper actually executes.
//!
//! Measured divergence between the two regexes over this repo's 837-line sample corpus
//! (`samples/fish-samples.jsonl`): **21 texts differ, all Arabic** — the fixed regex
//! folds `\p{M}` (combining marks) into its letter classes, the canonical one does not.
//! **Zero** divergence across the ten languages Qwen3-TTS actually supports. So the
//! default here is [`PreTokenizeRegex::Qwen2`] — what the checkpoint's own
//! `tokenizer_class` declares and what the model was trained with — and
//! [`PreTokenizeRegex::ReferenceWrapper`] is available for bit-exact agreement with a
//! parity fixture generated through `Qwen3TTSModel.from_pretrained`.

use std::collections::BTreeMap;
use std::path::Path;
use std::str::FromStr;

use tokenizers::Tokenizer;

use crate::config::Qwen3TtsConfig;

/// `<|im_start|>` — chat-turn opener (config `im_start_token_id`, 151644).
pub const IM_START: &str = "<|im_start|>";
/// `<|im_end|>` — chat-turn closer (config `im_end_token_id`, 151645).
pub const IM_END: &str = "<|im_end|>";
/// `<tts_pad>` — the text-stream filler (config `tts_pad_token_id`, 151671).
pub const TTS_PAD: &str = "<tts_pad>";
/// `<tts_text_bos>` — the text-stream BOS (config `tts_bos_token_id`, 151672).
pub const TTS_TEXT_BOS: &str = "<tts_text_bos>";
/// `<tts_text_eod>` — the text-stream EOS (config `tts_eos_token_id`, 151673).
pub const TTS_TEXT_EOD: &str = "<tts_text_eod>";

/// `transformers/models/qwen2/tokenization_qwen2.py::PRETOKENIZE_REGEX`, verbatim.
pub const QWEN2_PRETOKENIZE_REGEX: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The regex `PreTrainedTokenizerBase._patch_mistral_regex` substitutes when
/// `fix_mistral_regex=True` — verbatim from `transformers/tokenization_utils_base.py`.
pub const MISTRAL_FIXED_PRETOKENIZE_REGEX: &str =
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Which pre-tokenizer regex to install. See the module docs for why there are two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreTokenizeRegex {
    /// The checkpoint's declared `Qwen2Tokenizer` regex. Default.
    Qwen2,
    /// The regex the `qwen_tts` wrapper ends up with via `fix_mistral_regex=True`.
    ReferenceWrapper,
}

impl PreTokenizeRegex {
    /// The regex source this variant installs.
    pub fn pattern(self) -> &'static str {
        match self {
            Self::Qwen2 => QWEN2_PRETOKENIZE_REGEX,
            Self::ReferenceWrapper => MISTRAL_FIXED_PRETOKENIZE_REGEX,
        }
    }
}

/// A loaded Qwen3-TTS text tokenizer with its resolved special ids.
pub struct QwenTokenizer {
    inner: Tokenizer,
    /// `<|im_start|>`.
    pub im_start_id: u32,
    /// `<|im_end|>`.
    pub im_end_id: u32,
    /// `<tts_pad>` — the text addend at every position with no real text.
    pub tts_pad_id: u32,
    /// `<tts_text_bos>`.
    pub tts_bos_id: u32,
    /// `<tts_text_eod>`.
    pub tts_eos_id: u32,
}

impl QwenTokenizer {
    /// Load from a checkpoint directory, using the canonical Qwen2 regex.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        Self::from_dir_with(dir, PreTokenizeRegex::Qwen2)
    }

    /// Load from a checkpoint directory holding `vocab.json`, `merges.txt` and
    /// `tokenizer_config.json`.
    pub fn from_dir_with(
        dir: impl AsRef<Path>,
        regex: PreTokenizeRegex,
    ) -> Result<Self, TokenizerError> {
        let dir = dir.as_ref();
        let read = |name: &str| -> Result<String, TokenizerError> {
            std::fs::read_to_string(dir.join(name))
                .map_err(|e| TokenizerError::Load(format!("{name}: {e}")))
        };
        let spec = build_spec(&read("vocab.json")?, &read("merges.txt")?, &read("tokenizer_config.json")?, regex)?;
        let inner =
            Tokenizer::from_str(&spec).map_err(|e| TokenizerError::Load(format!("build tokenizer: {e}")))?;

        let id = |t: &str| {
            inner
                .token_to_id(t)
                .ok_or_else(|| TokenizerError::MissingSpecial(t.to_string()))
        };
        Ok(Self {
            im_start_id: id(IM_START)?,
            im_end_id: id(IM_END)?,
            tts_pad_id: id(TTS_PAD)?,
            tts_bos_id: id(TTS_TEXT_BOS)?,
            tts_eos_id: id(TTS_TEXT_EOD)?,
            inner,
        })
    }

    /// Encode to text-vocab ids, with the 33 added tokens honoured inline.
    ///
    /// Mirrors `Qwen3TTSModel._tokenize_texts` → `Qwen3TTSProcessor.__call__` →
    /// `Qwen2TokenizerFast(text)`, which uses `add_special_tokens=True`; the ByteLevel
    /// post-processor adds no ids, so this only affects offsets.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        self.inner
            .encode(text, true)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| TokenizerError::Encode(e.to_string()))
    }

    /// Decode ids back to text, keeping special tokens (transformers' default).
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, false)
            .map_err(|e| TokenizerError::Encode(e.to_string()))
    }

    /// Look up a single token's id.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Cross-check the resolved special ids against the checkpoint's `config.json`.
    ///
    /// This is the guard the Fish port learned the hard way: an id off by one there
    /// made the model emit its stop token after three frames. Cheap, so do it at load.
    pub fn check_special_ids(&self, cfg: &Qwen3TtsConfig) -> Result<(), TokenizerError> {
        let pairs = [
            (IM_START, self.im_start_id, cfg.im_start_token_id),
            (IM_END, self.im_end_id, cfg.im_end_token_id),
            (TTS_PAD, self.tts_pad_id, cfg.tts_pad_token_id),
            (TTS_TEXT_BOS, self.tts_bos_id, cfg.tts_bos_token_id),
            (TTS_TEXT_EOD, self.tts_eos_id, cfg.tts_eos_token_id),
        ];
        for (name, got, want) in pairs {
            if got != want {
                return Err(TokenizerError::IdMismatch {
                    token: name.to_string(),
                    vocab: got,
                    config: want,
                });
            }
        }
        Ok(())
    }
}

/// Assemble the serialized tokenizer spec from the checkpoint's three files.
fn build_spec(
    vocab_json: &str,
    merges_txt: &str,
    tokenizer_config_json: &str,
    regex: PreTokenizeRegex,
) -> Result<String, TokenizerError> {
    let vocab: BTreeMap<String, u32> = serde_json::from_str(vocab_json)
        .map_err(|e| TokenizerError::Load(format!("vocab.json: {e}")))?;

    // `merges.txt` is one `"<left> <right>"` rule per line, most-frequent first. The
    // byte-level alphabet never contains a literal space (it is `Ġ`), so the first
    // space is the only separator. Comment lines (`#version: …`) are skipped — the
    // Qwen3-TTS files carry none, but GPT-2-lineage merge files may.
    let mut merges: Vec<(&str, &str)> = Vec::new();
    for line in merges_txt.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let pair = line
            .split_once(' ')
            .ok_or_else(|| TokenizerError::Load(format!("merges.txt: unsplittable rule {line:?}")))?;
        merges.push(pair);
    }

    let tc: serde_json::Value = serde_json::from_str(tokenizer_config_json)
        .map_err(|e| TokenizerError::Load(format!("tokenizer_config.json: {e}")))?;
    let add_prefix_space = tc
        .get("add_prefix_space")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // `added_tokens_decoder` maps id → {content, single_word, lstrip, rstrip,
    // normalized, special}; that is exactly the `added_tokens` entry shape, plus `id`.
    let decoder = tc
        .get("added_tokens_decoder")
        .and_then(|v| v.as_object())
        .ok_or_else(|| TokenizerError::Load("tokenizer_config.json: no `added_tokens_decoder`".into()))?;
    let mut added: Vec<serde_json::Value> = Vec::with_capacity(decoder.len());
    for (id, spec) in decoder {
        let id: u32 = id
            .parse()
            .map_err(|_| TokenizerError::Load(format!("added_tokens_decoder: bad id {id:?}")))?;
        let mut obj = spec
            .as_object()
            .cloned()
            .ok_or_else(|| TokenizerError::Load(format!("added_tokens_decoder[{id}] is not an object")))?;
        obj.insert("id".into(), serde_json::json!(id));
        added.push(serde_json::Value::Object(obj));
    }
    added.sort_by_key(|v| v["id"].as_u64().unwrap_or_default());

    let spec = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": { "type": "NFC" },
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                { "type": "Split",
                  "pattern": { "Regex": regex.pattern() },
                  "behavior": "Isolated",
                  "invert": false },
                { "type": "ByteLevel",
                  "add_prefix_space": add_prefix_space,
                  "trim_offsets": true,
                  "use_regex": false }
            ]
        },
        "post_processor": {
            "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true
        },
        "decoder": {
            "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": vocab,
            "merges": merges,
        }
    });
    Ok(spec.to_string())
}

/// Errors building or running the Qwen3-TTS text tokenizer.
#[derive(Debug)]
pub enum TokenizerError {
    /// A checkpoint file was missing, unreadable, or malformed.
    Load(String),
    /// Encoding or decoding failed inside the Hugging Face tokenizer.
    Encode(String),
    /// A required special token was absent from the vocabulary.
    MissingSpecial(String),
    /// A special token's vocabulary id disagrees with `config.json`.
    IdMismatch {
        /// The token whose ids disagree.
        token: String,
        /// The id the vocabulary assigns.
        vocab: u32,
        /// The id `config.json` declares.
        config: u32,
    },
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load(m) => write!(f, "qwen tokenizer load: {m}"),
            Self::Encode(m) => write!(f, "qwen tokenizer encode: {m}"),
            Self::MissingSpecial(t) => write!(f, "qwen tokenizer: special token {t} not in vocabulary"),
            Self::IdMismatch { token, vocab, config } => write!(
                f,
                "qwen tokenizer: {token} is id {vocab} in the vocabulary but {config} in config.json"
            ),
        }
    }
}

impl std::error::Error for TokenizerError {}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Locate a Qwen3-TTS checkpoint directory: `$SYRINX_QWEN_DIR`, else the standard
    /// on-box download location. All five published checkpoints ship byte-identical
    /// `vocab.json` (md5 `613b8e4a…`) and `merges.txt` (md5 `e78882c2…`), so any of
    /// them serves for the text tokenizer.
    pub(crate) fn checkpoint_dir() -> Option<std::path::PathBuf> {
        if let Ok(d) = std::env::var("SYRINX_QWEN_DIR") {
            let p = std::path::PathBuf::from(d);
            return p.join("vocab.json").exists().then_some(p);
        }
        let home = std::env::var("HOME").ok()?;
        for name in [
            "Qwen3-TTS-12Hz-0.6B-Base",
            "Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "Qwen3-TTS-12Hz-1.7B-Base",
            "Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        ] {
            let p = std::path::Path::new(&home).join("models").join(name);
            if p.join("vocab.json").exists() {
                return Some(p);
            }
        }
        None
    }

    macro_rules! tok_or_skip {
        () => {
            match checkpoint_dir() {
                Some(d) => QwenTokenizer::from_dir(&d).expect("build tokenizer"),
                None => {
                    eprintln!("SKIP: no Qwen3-TTS checkpoint (set SYRINX_QWEN_DIR)");
                    return;
                }
            }
        };
    }

    /// Golden ids captured from the reference:
    /// `AutoTokenizer.from_pretrained("~/models/Qwen3-TTS-12Hz-0.6B-Base")(text)`
    /// under `~/.venvs/qwen/bin/python` (transformers 4.57.3, `Qwen2TokenizerFast`).
    const GOLDEN: &[(&str, &[u32])] = &[
        ("Hello, world!", &[9707, 11, 1879, 0]),
        (
            "Größenwahn: die Bäckerei überrascht mit Straßenfesten.",
            &[
                6464, 2956, 26824, 86, 29560, 25, 2746, 425, 2305, 377, 485, 72, 10489, 65, 84384,
                13920, 5451, 143089, 52764, 268, 13,
            ],
        ),
        (
            "我叫通义千问，是阿里云的开源大模型。",
            &[
                35946, 99882, 31935, 64559, 99320, 56007, 3837, 20412, 102661, 99718, 9370, 115462,
                26288, 104949, 1773,
            ],
        ),
        (
            "<|im_start|>assistant\nHello, world!<|im_end|>\n<|im_start|>assistant\n",
            &[151644, 77091, 198, 9707, 11, 1879, 0, 151645, 198, 151644, 77091, 198],
        ),
        (
            "<|im_start|>user\nSpeak in a calm, low voice.<|im_end|>\n",
            &[151644, 872, 198, 95845, 304, 264, 19300, 11, 3347, 7743, 13, 151645, 198],
        ),
        (
            "<|im_start|>assistant\nDies ist der Referenztext.<|im_end|>\n",
            &[151644, 77091, 198, 47789, 5999, 2694, 28634, 16597, 1318, 13, 151645, 198],
        ),
    ];

    #[test]
    fn matches_the_reference_tokenizer_ids() {
        let tok = tok_or_skip!();
        for (text, want) in GOLDEN {
            assert_eq!(&tok.encode(text).unwrap(), want, "ids for {text:?}");
        }
    }

    #[test]
    fn round_trips_english_german_and_chinese() {
        let tok = tok_or_skip!();
        for text in [
            "Hello, world! It's 3:15 — don't wait.",
            "Größenwahn: die Bäckerei überrascht mit Straßenfesten.",
            "我叫通义千问，是阿里云的开源大模型。",
            "混合 text with 中文 und Umlauten: schön, グレート, 한국어.",
        ] {
            let ids = tok.encode(text).unwrap();
            assert!(!ids.is_empty(), "empty encoding for {text:?}");
            assert_eq!(tok.decode(&ids).unwrap(), text, "round trip {text:?}");
        }
    }

    #[test]
    fn special_tokens_survive_a_round_trip_unsplit() {
        let tok = tok_or_skip!();
        let text = "<|im_start|>assistant\nHallo Welt<|im_end|>\n";
        let ids = tok.encode(text).unwrap();
        assert_eq!(ids[0], tok.im_start_id);
        assert_eq!(ids[ids.len() - 2], tok.im_end_id);
        assert_eq!(tok.decode(&ids).unwrap(), text);
        // Each special is ONE id, never BPE-split into its characters.
        for t in [IM_START, IM_END, TTS_PAD, TTS_TEXT_BOS, TTS_TEXT_EOD] {
            assert_eq!(tok.encode(t).unwrap().len(), 1, "{t} split into pieces");
        }
    }

    #[test]
    fn special_ids_match_the_checkpoint_config() {
        let Some(dir) = checkpoint_dir() else {
            eprintln!("SKIP: no Qwen3-TTS checkpoint (set SYRINX_QWEN_DIR)");
            return;
        };
        let tok = QwenTokenizer::from_dir(&dir).expect("build tokenizer");
        let cfg = Qwen3TtsConfig::from_json(&std::fs::read_to_string(dir.join("config.json")).unwrap())
            .expect("parse config.json");

        // The literal values `config.json` carries, pinned so a silent upstream
        // renumber fails here rather than after three frames of noise.
        assert_eq!(cfg.im_start_token_id, 151_644);
        assert_eq!(cfg.im_end_token_id, 151_645);
        assert_eq!(cfg.tts_pad_token_id, 151_671);
        assert_eq!(cfg.tts_bos_token_id, 151_672);
        assert_eq!(cfg.tts_eos_token_id, 151_673);

        assert_eq!(tok.im_start_id, cfg.im_start_token_id);
        assert_eq!(tok.im_end_id, cfg.im_end_token_id);
        assert_eq!(tok.tts_pad_id, cfg.tts_pad_token_id);
        assert_eq!(tok.tts_bos_id, cfg.tts_bos_token_id);
        assert_eq!(tok.tts_eos_id, cfg.tts_eos_token_id);
        tok.check_special_ids(&cfg).expect("special ids agree");
    }

    #[test]
    fn check_special_ids_rejects_a_shifted_config() {
        let tok = tok_or_skip!();
        let mut cfg = Qwen3TtsConfig::from_json(
            &std::fs::read_to_string(checkpoint_dir().unwrap().join("config.json")).unwrap(),
        )
        .unwrap();
        cfg.tts_bos_token_id += 1; // the exact Fish failure mode: off by one
        let err = tok.check_special_ids(&cfg).unwrap_err();
        assert!(
            matches!(&err, TokenizerError::IdMismatch { token, .. } if token == TTS_TEXT_BOS),
            "got {err}"
        );
    }

    #[test]
    fn the_two_pretokenizer_regexes_agree_on_the_supported_languages() {
        let Some(dir) = checkpoint_dir() else {
            eprintln!("SKIP: no Qwen3-TTS checkpoint (set SYRINX_QWEN_DIR)");
            return;
        };
        let a = QwenTokenizer::from_dir_with(&dir, PreTokenizeRegex::Qwen2).unwrap();
        let b = QwenTokenizer::from_dir_with(&dir, PreTokenizeRegex::ReferenceWrapper).unwrap();
        // One sample per supported language; the corpus-wide divergence is Arabic-only
        // (combining marks), and Arabic is not one of the ten supported languages.
        for text in [
            "The quick brown fox jumps over the lazy dog.",
            "我叫通义千问，是阿里云的开源大模型。",
            "Größenwahn: die Bäckerei überrascht mit Straßenfesten.",
            "Bonjour, l'été sera chaud cette année.",
            "Ciao, come stai oggi? Tutto bene!",
            "こんにちは、はじめまして。",
            "안녕하세요, 반갑습니다.",
            "Olá, tudo bem por aí?",
            "Здравствуйте, как ваши дела?",
            "¡Hola! ¿Cómo estás hoy?",
        ] {
            assert_eq!(a.encode(text).unwrap(), b.encode(text).unwrap(), "{text:?}");
        }
        // …and they are genuinely different objects: a combining-mark script splits.
        assert_ne!(
            a.encode("مهلاً، هل هذا أنت حقاً؟").unwrap(),
            b.encode("مهلاً، هل هذا أنت حقاً؟").unwrap()
        );
    }
}
