//! Inline **emotion tagging** for the CV3 instruct path — pure-Rust, **model-free**.
//!
//! The user writes emotion tags inline in the text — `"[happy] hello there"` sounds
//! happy; `"[happy] hi [sad] bye"` changes emotion mid-utterance — and this module turns
//! that into (a) a sequence of [`Segment`]s (text spans + the emotion in effect) and (b)
//! the CV3 **instruct string** for each emotion, which the synthesizer feeds to
//! [`Cv3Synthesizer::synthesize_instruct`](crate::synth_cv3::Cv3Synthesizer::synthesize_instruct).
//!
//! Nothing here touches Candle / `Tensor` (the tensor work lives in
//! [`crate::synth_cv3`]), so the registry, the tag parser, and the equal-power cross-fade
//! are unit-testable at the repo root **without the model** (like [`crate::watermark`]).
//! This module is therefore NOT behind the `real` feature.
//!
//! ## The three pieces
//!   * [`EmotionRegistry`] — a `tag -> instruct` map seeded with a rich,
//!     Fish-Speech-S1-inspired vocabulary ([`EmotionRegistry::default`]), each tag mapped
//!     to BOTH a Chinese and an English natural-language instruct phrase. CV3 was trained
//!     on Chinese instruct prompts (e.g. `用开心的语气说`), so the **zh** form is the
//!     default (the on-box A/B confirmed it follows); the **en** form is selectable via
//!     [`InstructLang`]. Extend it with [`EmotionRegistry::register`].
//!   * [`parse_tagged`] — split tagged text into [`Segment`]s. Accepts `[tag]` (the user's
//!     form) and `(tag)` (Fish-Speech S1 style), selectable via [`TagSyntax`]. A leading
//!     tag sets the first segment's emotion; a mid-text tag starts a new segment; text
//!     before any tag and unknown tags become neutral (`emotion: None`), with a logged
//!     warning for unknown tags. Malformed input (an unclosed bracket) is treated as
//!     literal text — it never panics.
//!   * [`equal_power_crossfade`] / [`concat_crossfade`] — join the per-segment waveforms
//!     with a short equal-power cross-fade at each boundary, so an emotion change does not
//!     click. (Used by `Cv3Synthesizer::synthesize_tagged`.)
//!
//! ## The `<|endofprompt|>` marker
//! CV3 instruct requires a trailing `<|endofprompt|>` on the instruct text. The registry's
//! instruct strings do **not** carry it — `synthesize_instruct` appends it idempotently
//! (see its docs) — so the stored strings stay clean, human-readable instruct phrases.

use std::collections::BTreeMap;

/// Which instruct-language variant a registry hands back. CV3 follows Chinese instruct
/// prompts best (the on-box A/B), so [`InstructLang::Zh`] is the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstructLang {
    /// Chinese instruct phrase (e.g. `用开心的语气说`) — the default, on-box-confirmed.
    Zh,
    /// English instruct phrase (e.g. `Speak in a happy tone`).
    En,
}

impl Default for InstructLang {
    fn default() -> Self {
        InstructLang::Zh
    }
}

/// Which bracket syntax [`parse_tagged`] recognizes as a tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagSyntax {
    /// `[tag]` only (the user's form).
    Brackets,
    /// `(tag)` only (Fish-Speech S1 style).
    Parens,
    /// Both `[tag]` and `(tag)` (the default).
    Both,
}

impl Default for TagSyntax {
    fn default() -> Self {
        TagSyntax::Both
    }
}

impl TagSyntax {
    /// If `c` opens a tag under this syntax, the delimiter that closes it.
    fn close_for(self, c: char) -> Option<char> {
        match (c, self) {
            ('[', TagSyntax::Brackets | TagSyntax::Both) => Some(']'),
            ('(', TagSyntax::Parens | TagSyntax::Both) => Some(')'),
            _ => None,
        }
    }
}

/// The two instruct-language variants stored for one emotion tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmotionInstruct {
    /// Chinese instruct phrase (the default the synthesizer uses).
    pub zh: String,
    /// English instruct phrase.
    pub en: String,
}

impl EmotionInstruct {
    /// The instruct phrase for `lang`.
    pub fn for_lang(&self, lang: InstructLang) -> &str {
        match lang {
            InstructLang::Zh => &self.zh,
            InstructLang::En => &self.en,
        }
    }
}

/// One parsed span of tagged text: the `text` to speak and the `emotion` tag in effect for
/// it (`None` = neutral — spoken plainly, with no instruct).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    /// The canonical (lower-cased) emotion tag in effect, or `None` for a neutral span.
    pub emotion: Option<String>,
    /// The text to synthesize for this span (trimmed of surrounding whitespace).
    pub text: String,
}

/// A `tag -> instruct` map plus the active [`InstructLang`] / [`TagSyntax`]. Seed a rich
/// default set with [`EmotionRegistry::default`], extend or override with
/// [`EmotionRegistry::register`].
#[derive(Debug, Clone)]
pub struct EmotionRegistry {
    entries: BTreeMap<String, EmotionInstruct>,
    lang: InstructLang,
    syntax: TagSyntax,
}

/// The seed vocabulary: `(tag, zh-instruct, en-instruct)`. Fish-Speech-S1-inspired and
/// covering the required core set (happy / sad / angry / excited / calm / gentle / serious
/// / fearful / surprised / disgusted / whisper / shout) plus the tone markers
/// (in-a-hurry / soft / slow), with a handful of natural aliases. The zh form is `用X的语气说`
/// ("speak in an X tone"), the dominant CV3 instruct shape; the volume markers
/// (whisper/shout) and tone markers use the matching natural phrasing. None of these carry
/// the `<|endofprompt|>` marker — the synthesizer appends it.
const DEFAULT_EMOTIONS: &[(&str, &str, &str)] = &[
    // --- core emotions ---
    ("happy", "用开心愉悦的语气说", "Speak in a happy, cheerful tone"),
    ("sad", "用悲伤难过的语气说", "Speak in a sad, sorrowful tone"),
    ("angry", "用愤怒生气的语气说", "Speak in an angry tone"),
    ("excited", "用兴奋激动的语气说", "Speak in an excited tone"),
    ("calm", "用平静冷静的语气说", "Speak in a calm tone"),
    ("gentle", "用温柔体贴的语气说", "Speak in a gentle, tender tone"),
    ("serious", "用严肃认真的语气说", "Speak in a serious tone"),
    ("fearful", "用害怕恐惧的语气说", "Speak in a fearful, frightened tone"),
    ("surprised", "用惊讶吃惊的语气说", "Speak in a surprised tone"),
    ("disgusted", "用厌恶嫌弃的语气说", "Speak in a disgusted, disdainful tone"),
    // --- volume / delivery markers ---
    ("whisper", "用气声小声地耳语", "Say this in a soft whisper"),
    ("shout", "提高音量大声喊着说", "Shout this loudly"),
    // --- tone markers ---
    ("in-a-hurry", "用急促匆忙的语气快速地说", "Speak quickly, as if in a hurry"),
    ("soft", "用柔和轻柔的语气说", "Speak in a soft, mellow tone"),
    ("slow", "用缓慢从容的语气慢慢地说", "Speak slowly and unhurriedly"),
    // --- aliases (point at the same delivery as a canonical tag's phrasing) ---
    ("afraid", "用害怕恐惧的语气说", "Speak in a fearful, frightened tone"),
    ("disdainful", "用厌恶嫌弃的语气说", "Speak in a disgusted, disdainful tone"),
    ("hurried", "用急促匆忙的语气快速地说", "Speak quickly, as if in a hurry"),
    ("cheerful", "用开心愉悦的语气说", "Speak in a happy, cheerful tone"),
];

impl Default for EmotionRegistry {
    /// The seed registry: the full [`DEFAULT_EMOTIONS`] vocabulary, [`InstructLang::Zh`],
    /// and [`TagSyntax::Both`].
    fn default() -> Self {
        let mut reg = EmotionRegistry {
            entries: BTreeMap::new(),
            lang: InstructLang::default(),
            syntax: TagSyntax::default(),
        };
        for &(tag, zh, en) in DEFAULT_EMOTIONS {
            reg.register(tag, zh, en);
        }
        reg
    }
}

impl EmotionRegistry {
    /// An empty registry (no tags), default lang/syntax. Build a bespoke vocabulary on top
    /// with [`EmotionRegistry::register`].
    pub fn empty() -> Self {
        EmotionRegistry {
            entries: BTreeMap::new(),
            lang: InstructLang::default(),
            syntax: TagSyntax::default(),
        }
    }

    /// Register (or **override**) a `tag` with its Chinese + English instruct phrases. The
    /// tag is canonicalized (trimmed + lower-cased) so lookups are case-insensitive. The
    /// instruct strings must NOT carry `<|endofprompt|>` (the synthesizer appends it).
    pub fn register(&mut self, tag: &str, zh: &str, en: &str) {
        self.entries.insert(
            canonical_tag(tag),
            EmotionInstruct {
                zh: zh.to_string(),
                en: en.to_string(),
            },
        );
    }

    /// Builder-style: set the active instruct language.
    pub fn with_lang(mut self, lang: InstructLang) -> Self {
        self.lang = lang;
        self
    }

    /// Builder-style: set the recognized tag syntax.
    pub fn with_syntax(mut self, syntax: TagSyntax) -> Self {
        self.syntax = syntax;
        self
    }

    /// The active instruct language.
    pub fn lang(&self) -> InstructLang {
        self.lang
    }

    /// The active tag syntax.
    pub fn syntax(&self) -> TagSyntax {
        self.syntax
    }

    /// Is `tag` known (case-insensitively)?
    pub fn contains(&self, tag: &str) -> bool {
        self.entries.contains_key(&canonical_tag(tag))
    }

    /// The active-language instruct phrase for `tag`, or `None` if the tag is unknown.
    pub fn instruct(&self, tag: &str) -> Option<&str> {
        self.entries
            .get(&canonical_tag(tag))
            .map(|e| e.for_lang(self.lang))
    }

    /// Both instruct variants for `tag`, or `None` if unknown.
    pub fn instruct_pair(&self, tag: &str) -> Option<&EmotionInstruct> {
        self.entries.get(&canonical_tag(tag))
    }

    /// All known tags, sorted (the `--list-emotions` source).
    pub fn tags(&self) -> Vec<&str> {
        self.entries.keys().map(String::as_str).collect()
    }

    /// Number of registered tags.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry has no tags.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Parse `text` into [`Segment`]s using this registry's syntax + vocabulary. Unknown
    /// tags resolve to neutral (`emotion: None`) with a logged warning — see
    /// [`parse_tagged`].
    pub fn parse(&self, text: &str) -> Vec<Segment> {
        parse_tagged(text, self)
    }

    /// Does `text` carry at least one **known** emotion tag? (The CLI auto-detect for the
    /// tagged-synthesis path.) Unknown bracketed tokens do not count.
    pub fn has_emotion_tags(&self, text: &str) -> bool {
        self.parse(text).iter().any(|s| s.emotion.is_some())
    }
}

/// Canonicalize a tag: trim, lower-case. Used for both registration and lookup so tags are
/// matched case-insensitively and whitespace-insensitively.
fn canonical_tag(tag: &str) -> String {
    tag.trim().to_lowercase()
}

/// A bracketed token is treated as a tag only if its (trimmed) inner is a short, tag-shaped
/// name: non-empty, <= 32 chars, and only ascii letters / digits / space / `-` / `_`.
/// Anything else (e.g. Chinese text after a stray `[`) is left as literal text, so the
/// parser never mis-segments real content.
fn is_tag_shaped(name: &str) -> bool {
    !name.is_empty()
        && name.chars().count() <= 32
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == ' ' || c == '-' || c == '_')
}

/// Scan forward from `start` for the matching `close` delimiter, returning its index. Bails
/// (returns `None`) if it hits another opening bracket of either kind, or scans past a small
/// window — so an unclosed/garbage bracket is rejected rather than swallowing the rest of
/// the text.
fn find_close(chars: &[char], start: usize, close: char) -> Option<usize> {
    const MAX_TAG_SPAN: usize = 40;
    let mut j = start;
    while j < chars.len() && j - start <= MAX_TAG_SPAN {
        let c = chars[j];
        if c == close {
            return Some(j);
        }
        if c == '[' || c == '(' {
            return None; // a new opener before the close — not a well-formed tag
        }
        j += 1;
    }
    None
}

/// Split `text` into [`Segment`]s, resolving tags against `registry`.
///
/// Grammar (informal):
///   * a **tag** is `[name]` or `(name)` (per `registry.syntax()`) whose trimmed inner
///     `name` is tag-shaped (see [`is_tag_shaped`]);
///   * the text up to the first tag is a **neutral** segment (`emotion: None`);
///   * each tag starts a new segment whose text runs until the next tag (or end);
///   * a **known** tag sets that segment's `emotion` to the canonical tag; an **unknown**
///     tag sets it to `None` (neutral) and logs a warning to stderr;
///   * a bracket that is not a well-formed, tag-shaped token (e.g. an unclosed `[`, or
///     `[` before Chinese text) is kept as **literal** text — never a panic;
///   * each segment's text is trimmed; empty segments are dropped (so a leading tag does
///     not emit an empty neutral segment). Empty / whitespace-only input yields `[]`.
pub fn parse_tagged(text: &str, registry: &EmotionRegistry) -> Vec<Segment> {
    let syntax = registry.syntax();
    let chars: Vec<char> = text.chars().collect();
    let mut segments: Vec<Segment> = Vec::new();
    let mut cur_text = String::new();
    let mut cur_emotion: Option<String> = None;
    let mut i = 0;

    let flush = |segments: &mut Vec<Segment>, emotion: &Option<String>, text: &str| {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            segments.push(Segment {
                emotion: emotion.clone(),
                text: trimmed.to_string(),
            });
        }
    };

    while i < chars.len() {
        let c = chars[i];
        if let Some(close) = syntax.close_for(c) {
            if let Some(j) = find_close(&chars, i + 1, close) {
                let inner: String = chars[i + 1..j].iter().collect();
                let name = inner.trim().to_lowercase();
                if is_tag_shaped(&name) {
                    // A well-formed tag: close the running segment and open a new one.
                    flush(&mut segments, &cur_emotion, &cur_text);
                    cur_text.clear();
                    if registry.contains(&name) {
                        cur_emotion = Some(name);
                    } else {
                        eprintln!(
                            "syrinx emotion: unknown tag `{name}` — speaking this span \
                             neutrally (known tags: run `syrinx synth --list-emotions`)"
                        );
                        cur_emotion = None;
                    }
                    i = j + 1;
                    continue;
                }
            }
            // Not a well-formed tag — keep the bracket as literal text.
            cur_text.push(c);
            i += 1;
        } else {
            cur_text.push(c);
            i += 1;
        }
    }
    flush(&mut segments, &cur_emotion, &cur_text);
    segments
}

/// Default cross-fade length (samples) between adjacent emotion segments — ~10 ms at
/// 24 kHz. Short enough not to blur the content, long enough to kill the boundary click.
pub const DEFAULT_XFADE_SAMPLES: usize = 240;

/// Equal-power cross-fade joining waveform `a` into waveform `b`.
///
/// Over the overlap region of length `L = min(fade, a.len(), b.len())`, the tail of `a` is
/// faded out by `cos(θ)` and the head of `b` faded in by `sin(θ)` (θ sweeping `0..π/2`), so
/// the summed power `cos²+sin² = 1` stays constant (no dip/bump at the seam). The result is
/// `a[..a.len()-L] ++ blend(L) ++ b[L..]`, i.e. length `a.len() + b.len() - L`. With
/// `fade == 0` (or either side empty) this is a plain concatenation.
pub fn equal_power_crossfade(a: &[f32], b: &[f32], fade: usize) -> Vec<f32> {
    let l = fade.min(a.len()).min(b.len());
    if l == 0 {
        let mut out = Vec::with_capacity(a.len() + b.len());
        out.extend_from_slice(a);
        out.extend_from_slice(b);
        return out;
    }
    let mut out = Vec::with_capacity(a.len() + b.len() - l);
    out.extend_from_slice(&a[..a.len() - l]);
    let a_tail = &a[a.len() - l..];
    for i in 0..l {
        // Centered phase so the fade is symmetric across the seam.
        let t = (i as f32 + 0.5) / l as f32;
        let theta = t * std::f32::consts::FRAC_PI_2;
        let g_out = theta.cos();
        let g_in = theta.sin();
        out.push(a_tail[i] * g_out + b[i] * g_in);
    }
    out.extend_from_slice(&b[l..]);
    out
}

/// Concatenate per-segment waveforms left-to-right with an [`equal_power_crossfade`] of
/// `fade` samples at every boundary. Empty segments are skipped; the total length is
/// `sum(len) - fade*(boundaries crossed)` (each boundary's overlap clamped to the shorter
/// side). Returns an empty `Vec` for no (non-empty) segments.
pub fn concat_crossfade(segments: &[Vec<f32>], fade: usize) -> Vec<f32> {
    let mut iter = segments.iter().filter(|s| !s.is_empty());
    let mut acc = match iter.next() {
        Some(first) => first.clone(),
        None => return Vec::new(),
    };
    for seg in iter {
        acc = equal_power_crossfade(&acc, seg, fade);
    }
    acc
}
