//! **Legacy** CosyVoice-era emotion tagging — migrated verbatim from
//! `syrinx-serve::emotion` per ADR-0001 **D1**. Deprecated; kept only for the CV3 tagged
//! path and the CLI's `--list-emotions`.
//!
//! # Do not build on this
//!
//! New work uses [`crate::parse`] (bracket cues) or [`crate::ssml`] (SSML), both of which
//! produce the one `CueDoc` IR and satisfy the hard invariant. This module is a SECOND
//! bracket parser with **different, weaker semantics**, retained because
//! `tests/emotion_tags.rs` is frozen and pins them:
//!
//!   * an unclosed `[` stays in the text as literal characters (`"[happy hello"`),
//!   * with [`TagSyntax::Parens`] selected, `[happy] hi` reaches the backend verbatim,
//!   * a bracket whose content is not tag-shaped stays literal.
//!
//! Each of those puts a literal bracket in front of a backend, which the strict rule
//! (ADR-0001 §9.1 / D5) forbids. The conflict is recorded in **ADR-0001 §11**; the
//! resolution is that this path is frozen and deprecated, not that the invariant bends.
//! The invariant property test covers `parse`/`parse_ssml`; it deliberately does not cover
//! this function, because this function does not satisfy it.
//!
//! Migrating a caller means switching to [`crate::parse`] and accepting D5's semantics.

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

