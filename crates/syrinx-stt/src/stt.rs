//! The real Whisper STT engine (Candle). Feature `real`.
//!
//! Mirrors the official `candle-examples/whisper` decode loop, packaged as a
//! reusable [`Stt`] with a structured [`Transcript`] result. Model weights live
//! on the box; this file is compile-verified off-box (see `// PARITY:` notes for
//! the model-layout assumptions that can only be checked against real weights).

use std::cell::RefCell;
use std::fmt;
use std::path::{Path, PathBuf};

use candle_core::{Device, IndexOp, Tensor};
use candle_nn::ops::softmax;
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, audio, model::Whisper, Config};
use tokenizers::Tokenizer;

mod resample;

/// A decoded transcript: the joined text, the detected/forced language code, and
/// the per-chunk segments (Whisper processes audio in 30 s windows).
#[derive(Debug, Clone)]
pub struct Transcript {
    /// The full transcript (all segments joined with spaces, trimmed).
    pub text: String,
    /// The language code that drove decoding (e.g. `"en"`), if known. `None` for
    /// English-only models (which carry no language tokens).
    pub language: Option<String>,
    /// One entry per decoded 30-second window, in order.
    pub segments: Vec<Segment>,
}

/// One decoded audio window.
#[derive(Debug, Clone)]
pub struct Segment {
    /// Start offset of this window in seconds.
    pub start: f64,
    /// Window length in seconds.
    pub duration: f64,
    /// Text decoded from this window.
    pub text: String,
}

/// Errors from loading or running the Whisper STT engine.
#[derive(Debug)]
pub enum SttError {
    /// A required model file was missing from the model directory.
    MissingFile(PathBuf),
    /// Failed to read/parse `config.json`.
    Config(String),
    /// Failed to load the tokenizer (`tokenizer.json`).
    Tokenizer(String),
    /// A required special token was absent from the tokenizer vocab.
    MissingToken(String),
    /// A Candle tensor / weight-loading error.
    Candle(String),
    /// The input waveform was empty.
    EmptyInput,
}

impl fmt::Display for SttError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SttError::MissingFile(p) => write!(f, "missing model file: {}", p.display()),
            SttError::Config(e) => write!(f, "config.json: {e}"),
            SttError::Tokenizer(e) => write!(f, "tokenizer.json: {e}"),
            SttError::MissingToken(t) => write!(f, "tokenizer is missing token {t}"),
            SttError::Candle(e) => write!(f, "candle: {e}"),
            SttError::EmptyInput => write!(f, "empty input waveform"),
        }
    }
}

impl std::error::Error for SttError {}

impl From<candle_core::Error> for SttError {
    fn from(e: candle_core::Error) -> Self {
        SttError::Candle(e.to_string())
    }
}

/// Whisper's fixed input sample rate.
const SAMPLE_RATE: u32 = m::SAMPLE_RATE as u32;

/// A loaded Whisper STT model: encoder/decoder + tokenizer + the precomputed
/// mel filter bank and the special-token ids.
///
/// `transcribe` takes `&self`; the underlying Whisper forward needs `&mut`
/// (per-step KV cache), so the model lives behind a [`RefCell`].
pub struct Stt {
    model: RefCell<Whisper>,
    tokenizer: Tokenizer,
    device: Device,
    mel_filters: Vec<f32>,
    num_mel_bins: usize,
    max_source_positions: usize,
    max_target_positions: usize,
    vocab_size: usize,
    suppress_tokens: Vec<u32>,
    is_multilingual: bool,
    sot_token: u32,
    transcribe_token: u32,
    eot_token: u32,
    no_timestamps_token: u32,
    no_speech_token: u32,
}

impl Stt {
    /// Load a Whisper model from a directory holding the HF `openai/whisper-*`
    /// layout: `config.json`, `tokenizer.json`, and `model.safetensors`.
    ///
    /// Works for any Whisper size (tiny / base / small / medium / large-v3) —
    /// all dimensions are read from `config.json`, and the mel filter bank is
    /// selected by `num_mel_bins` (80, or 128 for large-v3).
    ///
    /// `device` is the Candle device (CPU is the reference; `cuda:i` needs a
    /// `--features cuda` build).
    //
    // PARITY: assumes the HF Transformers Whisper safetensors layout — weight
    // keys prefixed `model.encoder.*` / `model.decoder.*` (what `Whisper::load`
    // expects, and what the `openai/whisper-*` repos ship). The candle-converted
    // `lmz/candle-whisper` repo uses the same key prefixes.
    pub fn load(model_dir: impl AsRef<Path>, device: Device) -> Result<Self, SttError> {
        let dir = model_dir.as_ref();
        let config_path = require(dir.join("config.json"))?;
        let tokenizer_path = require(dir.join("tokenizer.json"))?;
        let weights_path = require(dir.join("model.safetensors"))?;

        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(&config_path).map_err(|e| SttError::Config(e.to_string()))?,
        )
        .map_err(|e| SttError::Config(e.to_string()))?;

        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| SttError::Tokenizer(e.to_string()))?;

        // The mel filter bank is fixed per bin-count; embedded so the crate needs
        // no side-car file (same bytes the candle whisper example ships).
        let mel_bytes: &[u8] = match config.num_mel_bins {
            80 => include_bytes!("melfilters.bytes").as_slice(),
            128 => include_bytes!("melfilters128.bytes").as_slice(),
            n => {
                return Err(SttError::Config(format!(
                    "unsupported num_mel_bins {n} (expected 80 or 128)"
                )))
            }
        };
        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        for (i, f) in mel_filters.iter_mut().enumerate() {
            let b = &mel_bytes[i * 4..i * 4 + 4];
            *f = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        }

        // Special tokens (vocab-dependent). Multilingual models carry per-language
        // tokens; detect that by probing for a non-English language token.
        let is_multilingual = tokenizer.token_to_id("<|zh|>").is_some();
        let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
        let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|t| tokenizer.token_to_id(t))
            .ok_or_else(|| SttError::MissingToken(m::NO_SPEECH_TOKENS.join(" / ")))?;

        let num_mel_bins = config.num_mel_bins;
        let max_source_positions = config.max_source_positions;
        let max_target_positions = config.max_target_positions;
        let vocab_size = config.vocab_size;
        let suppress_tokens = config.suppress_tokens.clone();

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], m::DTYPE, &device)
                .map_err(SttError::from)?
        };
        let model = Whisper::load(&vb, config).map_err(SttError::from)?;

        Ok(Self {
            model: RefCell::new(model),
            tokenizer,
            device,
            mel_filters,
            num_mel_bins,
            max_source_positions,
            max_target_positions,
            vocab_size,
            suppress_tokens,
            is_multilingual,
            sot_token,
            transcribe_token,
            eot_token,
            no_timestamps_token,
            no_speech_token,
        })
    }

    /// Transcribe a mono waveform, auto-detecting the language (multilingual
    /// models) or running English-only.
    pub fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<Transcript, SttError> {
        self.transcribe_lang(samples, sample_rate, None)
    }

    /// Transcribe a mono waveform with an optional forced language code (e.g.
    /// `Some("en")`); `None` auto-detects on multilingual models.
    pub fn transcribe_lang(
        &self,
        samples: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<Transcript, SttError> {
        if samples.is_empty() {
            return Err(SttError::EmptyInput);
        }

        // Whisper consumes 16 kHz mono. Resample if the caller's rate differs.
        let pcm: Vec<f32> = if sample_rate == SAMPLE_RATE {
            samples.to_vec()
        } else {
            resample::resample(samples, sample_rate, SAMPLE_RATE)
        };

        // Log-mel front end (reused from candle-transformers). `pcm_to_mel` reads
        // only `num_mel_bins` from the `Config`, so hand it a minimal interned one.
        let mel = audio::pcm_to_mel(mel_config_for(self.num_mel_bins), &pcm, &self.mel_filters);
        let n_frames = mel.len() / self.num_mel_bins;
        let mel = Tensor::from_vec(mel, (1, self.num_mel_bins, n_frames), &self.device)
            .map_err(SttError::from)?;

        // Language token: forced, auto-detected, or none (English-only model).
        let (language_token, language_code) = self.resolve_language(&mel, language)?;

        // Decode each 30-second window.
        let (_, _, content_frames) = mel.dims3().map_err(SttError::from)?;
        let mut seek = 0usize;
        let mut segments = Vec::new();
        let mut texts = Vec::new();
        while seek < content_frames {
            let time_offset = (seek * m::HOP_LENGTH) as f64 / SAMPLE_RATE as f64;
            let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
            let segment_duration = (segment_size * m::HOP_LENGTH) as f64 / SAMPLE_RATE as f64;
            let mel_segment = mel.narrow(2, seek, segment_size).map_err(SttError::from)?;
            seek += segment_size;

            let dr = self.decode_with_fallback(&mel_segment, language_token)?;
            if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
                // Silence / non-speech window — skip it.
                continue;
            }
            let text = dr.text.trim().to_string();
            if !text.is_empty() {
                texts.push(text.clone());
            }
            segments.push(Segment {
                start: time_offset,
                duration: segment_duration,
                text,
            });
        }

        Ok(Transcript {
            text: texts.join(" ").trim().to_string(),
            language: language_code,
            segments,
        })
    }

    /// Pick the language token + code: forced override, auto-detect, or none.
    fn resolve_language(
        &self,
        mel: &Tensor,
        language: Option<&str>,
    ) -> Result<(Option<u32>, Option<String>), SttError> {
        match (self.is_multilingual, language) {
            (false, _) => Ok((None, None)),
            (true, Some(code)) => {
                let tok = token_id(&self.tokenizer, &format!("<|{code}|>"))?;
                Ok((Some(tok), Some(code.to_string())))
            }
            (true, None) => {
                let (tok, code) = self.detect_language(mel)?;
                Ok((Some(tok), Some(code)))
            }
        }
    }

    /// Greedy/temperature-fallback decode of one mel window for a language token.
    fn decode_with_fallback(
        &self,
        segment: &Tensor,
        language_token: Option<u32>,
    ) -> Result<DecodingResult, SttError> {
        let mut last: Option<DecodingResult> = None;
        for (i, &t) in m::TEMPERATURES.iter().enumerate() {
            let dr = self.decode(segment, t, language_token)?;
            let is_last = i == m::TEMPERATURES.len() - 1;
            let needs_fallback = dr.compression_ratio > m::COMPRESSION_RATIO_THRESHOLD
                || dr.avg_logprob < m::LOGPROB_THRESHOLD;
            if is_last || (!needs_fallback || dr.no_speech_prob > m::NO_SPEECH_THRESHOLD) {
                return Ok(dr);
            }
            last = Some(dr);
        }
        // TEMPERATURES is non-empty, so `last` (or an earlier return) is set.
        last.ok_or_else(|| SttError::Candle("empty temperature schedule".into()))
    }

    /// One greedy (or temperature-sampled) decode pass over a mel window.
    fn decode(
        &self,
        mel: &Tensor,
        t: f64,
        language_token: Option<u32>,
    ) -> Result<DecodingResult, SttError> {
        let mut model = self.model.borrow_mut();
        let audio_features = model.encoder.forward(mel, true).map_err(SttError::from)?;

        let sample_len = self.max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens: Vec<u32> = vec![self.sot_token];
        if let Some(lt) = language_token {
            tokens.push(lt);
        }
        tokens.push(self.transcribe_token);
        // No-timestamps mode: one segment per window, plain text out.
        tokens.push(self.no_timestamps_token);

        let suppress = self.suppress_tensor()?;
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())
                .map_err(SttError::from)?
                .unsqueeze(0)
                .map_err(SttError::from)?;
            let ys = model
                .decoder
                .forward(&tokens_t, &audio_features, i == 0)
                .map_err(SttError::from)?;

            if i == 0 {
                let logits = model
                    .decoder
                    .final_linear(&ys.i(..1).map_err(SttError::from)?)
                    .map_err(SttError::from)?
                    .i(0)
                    .map_err(SttError::from)?
                    .i(0)
                    .map_err(SttError::from)?;
                no_speech_prob = softmax(&logits, 0)
                    .map_err(SttError::from)?
                    .i(self.no_speech_token as usize)
                    .map_err(SttError::from)?
                    .to_scalar::<f32>()
                    .map_err(SttError::from)? as f64;
            }

            let (_, seq_len, _) = ys.dims3().map_err(SttError::from)?;
            let logits = model
                .decoder
                .final_linear(&ys.i((..1, seq_len - 1..)).map_err(SttError::from)?)
                .map_err(SttError::from)?
                .i(0)
                .map_err(SttError::from)?
                .i(0)
                .map_err(SttError::from)?;
            let logits = logits.broadcast_add(&suppress).map_err(SttError::from)?;

            let next_token = if t > 0f64 {
                let prs = softmax(&(&logits / t).map_err(SttError::from)?, 0)
                    .map_err(SttError::from)?;
                let v: Vec<f32> = prs.to_vec1().map_err(SttError::from)?;
                argmax(&v) // deterministic fallback: take the mode (no RNG dep)
            } else {
                let v: Vec<f32> = logits.to_vec1().map_err(SttError::from)?;
                argmax(&v)
            };
            tokens.push(next_token);

            let prob = softmax(&logits, candle_core::D::Minus1)
                .map_err(SttError::from)?
                .i(next_token as usize)
                .map_err(SttError::from)?
                .to_scalar::<f32>()
                .map_err(SttError::from)? as f64;
            if next_token == self.eot_token || tokens.len() > self.max_target_positions {
                break;
            }
            sum_logprob += prob.ln();
        }

        let text = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(|e| SttError::Candle(e.to_string()))?;
        let avg_logprob = sum_logprob / tokens.len() as f64;
        let compression_ratio = compression_ratio(&text);

        Ok(DecodingResult {
            text,
            avg_logprob,
            no_speech_prob,
            compression_ratio,
        })
    }

    /// Detect the spoken language from the first encoder window. Returns the
    /// `<|lang|>` token id and its two-letter code.
    fn detect_language(&self, mel: &Tensor) -> Result<(u32, String), SttError> {
        let (_b, _bins, seq_len) = mel.dims3().map_err(SttError::from)?;
        let mel = mel
            .narrow(2, 0, usize::min(seq_len, self.max_source_positions))
            .map_err(SttError::from)?;

        let langs = LANGUAGES;
        let lang_token_ids = langs
            .iter()
            .map(|(code, _)| token_id(&self.tokenizer, &format!("<|{code}|>")))
            .collect::<Result<Vec<_>, _>>()?;

        let mut model = self.model.borrow_mut();
        let audio_features = model.encoder.forward(&mel, true).map_err(SttError::from)?;
        let tokens = Tensor::new(&[[self.sot_token]], mel.device()).map_err(SttError::from)?;
        let lang_ids_t = Tensor::new(lang_token_ids.as_slice(), mel.device())
            .map_err(SttError::from)?;
        let ys = model
            .decoder
            .forward(&tokens, &audio_features, true)
            .map_err(SttError::from)?;
        let logits = model
            .decoder
            .final_linear(&ys.i(..1).map_err(SttError::from)?)
            .map_err(SttError::from)?
            .i(0)
            .map_err(SttError::from)?
            .i(0)
            .map_err(SttError::from)?;
        let logits = logits.index_select(&lang_ids_t, 0).map_err(SttError::from)?;
        let probs: Vec<f32> = softmax(&logits, candle_core::D::Minus1)
            .map_err(SttError::from)?
            .to_vec1()
            .map_err(SttError::from)?;

        let best = argmax(&probs) as usize;
        let code = langs[best].0.to_string();
        let token = token_id(&self.tokenizer, &format!("<|{code}|>"))?;
        Ok((token, code))
    }

    /// The non-speech-token suppression mask as a `[vocab]` logit-offset tensor.
    fn suppress_tensor(&self) -> Result<Tensor, SttError> {
        let mask: Vec<f32> = (0..self.vocab_size as u32)
            .map(|i| {
                if self.suppress_tokens.contains(&i) {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        Tensor::new(mask.as_slice(), &self.device).map_err(SttError::from)
    }
}

/// Per-window decode bookkeeping (mirrors the candle example's `DecodingResult`).
struct DecodingResult {
    text: String,
    avg_logprob: f64,
    no_speech_prob: f64,
    compression_ratio: f64,
}

/// Index of the maximum element (greedy `argmax`); ties go to the first.
fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// gzip-free text "compression ratio" proxy: bytes / unique-bytes-ish. Whisper
/// uses gzip; we approximate degenerate repetition with the ratio of total
/// length to distinct whitespace-split tokens, which is enough to trip the
/// fallback on pathological loops without a compression dependency.
fn compression_ratio(text: &str) -> f64 {
    let total = text.chars().count();
    if total == 0 {
        return 0.0;
    }
    let distinct: std::collections::HashSet<&str> = text.split_whitespace().collect();
    let words = text.split_whitespace().count().max(1);
    // High when many repeated words (distinct << words).
    (words as f64 / distinct.len().max(1) as f64).max(total as f64 / (distinct.len().max(1) as f64 * 8.0))
}

/// Look up a single special token's id or error.
fn token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32, SttError> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| SttError::MissingToken(token.to_string()))
}

/// Error unless `path` exists.
fn require(path: PathBuf) -> Result<PathBuf, SttError> {
    if path.exists() {
        Ok(path)
    } else {
        Err(SttError::MissingFile(path))
    }
}

/// Interned minimal `Config` (only `num_mel_bins` matters to `pcm_to_mel`) so we
/// can hand `audio::pcm_to_mel` a `&Config` without borrowing the model.
fn mel_config_for(num_mel_bins: usize) -> &'static Config {
    use std::sync::OnceLock;
    static C80: OnceLock<Config> = OnceLock::new();
    static C128: OnceLock<Config> = OnceLock::new();
    let make = |bins: usize| Config {
        num_mel_bins: bins,
        max_source_positions: 1500,
        d_model: 0,
        encoder_attention_heads: 0,
        encoder_layers: 0,
        vocab_size: 0,
        max_target_positions: 0,
        decoder_attention_heads: 0,
        decoder_layers: 0,
        suppress_tokens: Vec::new(),
    };
    if num_mel_bins == 128 {
        C128.get_or_init(|| make(128))
    } else {
        C80.get_or_init(|| make(80))
    }
}

/// Whisper's language table (code, english name) — the 99 languages it supports.
/// Mirrors `candle-examples/whisper/multilingual.rs`.
const LANGUAGES: [(&str, &str); 99] = [
    ("en", "english"), ("zh", "chinese"), ("de", "german"), ("es", "spanish"),
    ("ru", "russian"), ("ko", "korean"), ("fr", "french"), ("ja", "japanese"),
    ("pt", "portuguese"), ("tr", "turkish"), ("pl", "polish"), ("ca", "catalan"),
    ("nl", "dutch"), ("ar", "arabic"), ("sv", "swedish"), ("it", "italian"),
    ("id", "indonesian"), ("hi", "hindi"), ("fi", "finnish"), ("vi", "vietnamese"),
    ("he", "hebrew"), ("uk", "ukrainian"), ("el", "greek"), ("ms", "malay"),
    ("cs", "czech"), ("ro", "romanian"), ("da", "danish"), ("hu", "hungarian"),
    ("ta", "tamil"), ("no", "norwegian"), ("th", "thai"), ("ur", "urdu"),
    ("hr", "croatian"), ("bg", "bulgarian"), ("lt", "lithuanian"), ("la", "latin"),
    ("mi", "maori"), ("ml", "malayalam"), ("cy", "welsh"), ("sk", "slovak"),
    ("te", "telugu"), ("fa", "persian"), ("lv", "latvian"), ("bn", "bengali"),
    ("sr", "serbian"), ("az", "azerbaijani"), ("sl", "slovenian"), ("kn", "kannada"),
    ("et", "estonian"), ("mk", "macedonian"), ("br", "breton"), ("eu", "basque"),
    ("is", "icelandic"), ("hy", "armenian"), ("ne", "nepali"), ("mn", "mongolian"),
    ("bs", "bosnian"), ("kk", "kazakh"), ("sq", "albanian"), ("sw", "swahili"),
    ("gl", "galician"), ("mr", "marathi"), ("pa", "punjabi"), ("si", "sinhala"),
    ("km", "khmer"), ("sn", "shona"), ("yo", "yoruba"), ("so", "somali"),
    ("af", "afrikaans"), ("oc", "occitan"), ("ka", "georgian"), ("be", "belarusian"),
    ("tg", "tajik"), ("sd", "sindhi"), ("gu", "gujarati"), ("am", "amharic"),
    ("yi", "yiddish"), ("lo", "lao"), ("uz", "uzbek"), ("fo", "faroese"),
    ("ht", "haitian creole"), ("ps", "pashto"), ("tk", "turkmen"), ("nn", "nynorsk"),
    ("mt", "maltese"), ("sa", "sanskrit"), ("lb", "luxembourgish"), ("my", "myanmar"),
    ("bo", "tibetan"), ("tl", "tagalog"), ("mg", "malagasy"), ("as", "assamese"),
    ("tt", "tatar"), ("haw", "hawaiian"), ("ln", "lingala"), ("ha", "hausa"),
    ("ba", "bashkir"), ("jw", "javanese"), ("su", "sundanese"),
];
