//! SSML subset parser (C1.3) → the **same** [`CueDoc`] IR as the bracket parser.
//!
//! Two authoring syntaxes, one IR: everything downstream (lowering, caps, the hard
//! invariant) sees a `CueDoc` and never learns which syntax produced it. That is the whole
//! point of putting this next to `parse.rs` rather than in a backend or the frontend.
//!
//! ## The supported subset (spec §2.2)
//!
//! * `<speak>` — optional document root, carries no cue.
//! * `<prosody rate= pitch= volume=>…</prosody>` → [`CueKind::Prosody`] over its content.
//! * `<emphasis level=>…</emphasis>` → [`CueKind::Emphasis`] over its content.
//! * `<break time=|strength=/>` → [`CueKind::Pause`], a zero-width point event.
//!
//! Anything else is a hard error, never silently ignored: an unsupported tag that is
//! quietly dropped would leave the author believing a control was honoured.
//!
//! ## Mixed syntax is a hard error
//!
//! SSML and bracket cues may not be combined in one input. The two have different scoping
//! rules (XML nesting vs. next-cue-or-sentence-end), so a mixed document has no single
//! defensible reading — and guessing would silently change prosody. See [`parse_any`].

use crate::ir::{Cue, CueDoc, CueKind, Level};
use crate::parse::{parse, ParseOptions};
use crate::vocab::Vocab;

/// Why an SSML document was rejected. Every variant carries a byte offset into the source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SsmlError {
    /// Bracket cue markup and SSML tags in the same input.
    MixedSyntax { at: usize },
    /// `<foo>` where `foo` is outside the supported subset.
    UnsupportedTag { name: String, at: usize },
    /// A `<prosody>`/`<emphasis>` element was never closed.
    UnclosedTag { name: String, at: usize },
    /// `</b>` closing an element that is not open.
    MismatchedClose { expected: Option<String>, found: String, at: usize },
    /// An attribute value that is not in the supported form (e.g. `rate="quickly"`).
    BadAttribute { tag: String, attr: String, value: String, at: usize },
    /// A `<` that never becomes a well-formed tag.
    Malformed { at: usize },
}

impl std::fmt::Display for SsmlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MixedSyntax { at } => write!(
                f,
                "byte {at}: SSML tags and bracket cues cannot be mixed in one input; \
                 use one syntax, or escape brackets as \\[ \\]"
            ),
            Self::UnsupportedTag { name, at } => write!(
                f,
                "byte {at}: <{name}> is outside the supported SSML subset \
                 (speak, prosody, emphasis, break)"
            ),
            Self::UnclosedTag { name, at } => write!(f, "byte {at}: <{name}> is never closed"),
            Self::MismatchedClose { expected, found, at } => match expected {
                Some(e) => write!(f, "byte {at}: </{found}> closes <{e}>"),
                None => write!(f, "byte {at}: </{found}> closes nothing"),
            },
            Self::BadAttribute { tag, attr, value, at } => {
                write!(f, "byte {at}: <{tag} {attr}=\"{value}\"> is not a supported value")
            }
            Self::Malformed { at } => write!(f, "byte {at}: malformed markup"),
        }
    }
}

impl std::error::Error for SsmlError {}

/// Which authoring syntax an input uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    Brackets,
    Ssml,
}

/// Does this input contain an SSML tag? `<` followed by a name char or `/`.
///
/// Deliberately does not fire on `<|speaker:1|>`: `|` is neither, so a speaker token stays
/// bracket-dialect and never trips the mixed-syntax error on its own.
fn has_ssml_tag(input: &str) -> Option<usize> {
    let b = input.as_bytes();
    for i in 0..b.len() {
        if b[i] == b'<' {
            match b.get(i + 1) {
                Some(c) if c.is_ascii_alphabetic() || *c == b'/' => return Some(i),
                _ => {}
            }
        }
    }
    None
}

/// Does this input contain unescaped bracket cue markup or a speaker token?
fn has_bracket_cue(input: &str) -> Option<usize> {
    let b = input.as_bytes();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'\\' => i += 2, // escaped: `\[` is literal, never a cue
            b'[' | b']' => return Some(i),
            b'<' if input[i..].starts_with("<|speaker") => return Some(i),
            _ => i += 1,
        }
    }
    None
}

/// Detect the dialect, rejecting a mixed document.
pub fn detect(input: &str) -> Result<Dialect, SsmlError> {
    match (has_ssml_tag(input), has_bracket_cue(input)) {
        (Some(s), Some(br)) => Err(SsmlError::MixedSyntax { at: s.min(br) }),
        (Some(_), None) => Ok(Dialect::Ssml),
        (None, _) => Ok(Dialect::Brackets),
    }
}

/// Parse either syntax into the one IR, erroring on a mixed document.
///
/// This is the entry point every caller should use; `parse` and `parse_ssml` are the
/// dialect-specific paths behind it.
pub fn parse_any(input: &str, vocab: &Vocab, opts: &ParseOptions) -> Result<CueDoc, SsmlError> {
    match detect(input)? {
        Dialect::Brackets => Ok(parse(input, vocab, opts)),
        Dialect::Ssml => parse_ssml(input),
    }
}

// ---------------------------------------------------------------- value parsers

/// `rate="slow" | "150%" | "1.5"` → a multiplier where 1.0 is unchanged.
fn parse_rate(v: &str) -> Option<f32> {
    match v.trim().to_lowercase().as_str() {
        "x-slow" => return Some(0.5),
        "slow" => return Some(0.75),
        "medium" | "default" => return Some(1.0),
        "fast" => return Some(1.25),
        "x-fast" => return Some(1.5),
        _ => {}
    }
    let t = v.trim();
    if let Some(p) = t.strip_suffix('%') {
        return p.parse::<f32>().ok().filter(|x| *x > 0.0).map(|x| x / 100.0);
    }
    t.parse::<f32>().ok().filter(|x| *x > 0.0)
}

/// `pitch="+2st" | "high" | "+10%"` → semitones relative to the speaker's baseline.
fn parse_pitch_st(v: &str) -> Option<f32> {
    match v.trim().to_lowercase().as_str() {
        "x-low" => return Some(-6.0),
        "low" => return Some(-3.0),
        "medium" | "default" => return Some(0.0),
        "high" => return Some(3.0),
        "x-high" => return Some(6.0),
        _ => {}
    }
    let t = v.trim();
    let lower = t.to_lowercase();
    if let Some(p) = lower.strip_suffix("st") {
        return p.trim().parse::<f32>().ok();
    }
    if let Some(p) = t.strip_suffix('%') {
        // Percent of frequency → semitones: 12 * log2(1 + pct/100).
        let pct = p.parse::<f32>().ok()?;
        let ratio = 1.0 + pct / 100.0;
        if ratio <= 0.0 {
            return None;
        }
        return Some(12.0 * ratio.log2());
    }
    None
}

/// `volume="+6dB" | "loud" | "silent"` → decibels relative to the speaker's baseline.
fn parse_volume_db(v: &str) -> Option<f32> {
    match v.trim().to_lowercase().as_str() {
        "silent" => return Some(-60.0),
        "x-soft" => return Some(-12.0),
        "soft" => return Some(-6.0),
        "medium" | "default" => return Some(0.0),
        "loud" => return Some(6.0),
        "x-loud" => return Some(12.0),
        _ => {}
    }
    let lower = v.trim().to_lowercase();
    let p = lower.strip_suffix("db")?;
    p.trim().parse::<f32>().ok()
}

/// `time="400ms" | "0.4s"` → milliseconds.
fn parse_break_time(v: &str) -> Option<u32> {
    let t = v.trim().to_lowercase();
    if let Some(p) = t.strip_suffix("ms") {
        return p.trim().parse::<f32>().ok().filter(|x| *x >= 0.0).map(|x| x as u32);
    }
    if let Some(p) = t.strip_suffix('s') {
        return p.trim().parse::<f32>().ok().filter(|x| *x >= 0.0).map(|x| (x * 1000.0) as u32);
    }
    None
}

/// `strength="weak" | "strong" | "none"` → milliseconds, per the SSML strength ladder.
fn parse_break_strength(v: &str) -> Option<u32> {
    match v.trim().to_lowercase().as_str() {
        "none" => Some(0),
        "x-weak" => Some(100),
        "weak" => Some(200),
        "medium" => Some(400),
        "strong" => Some(700),
        "x-strong" => Some(1000),
        _ => None,
    }
}

fn parse_emphasis_level(v: &str) -> Option<Level> {
    match v.trim().to_lowercase().as_str() {
        "reduced" | "none" => Some(Level::Reduced),
        "moderate" | "default" => Some(Level::Moderate),
        "strong" => Some(Level::Strong),
        _ => None,
    }
}

// ---------------------------------------------------------------- scanner

/// The five predefined XML entities. Anything else is left verbatim — an unknown entity is
/// far more likely to be a literal ampersand in prose than a control the author meant.
fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        let mut matched = false;
        for (ent, ch) in [("&amp;", '&'), ("&lt;", '<'), ("&gt;", '>'), ("&quot;", '"'), ("&apos;", '\'')] {
            if tail.starts_with(ent) {
                out.push(ch);
                rest = &tail[ent.len()..];
                matched = true;
                break;
            }
        }
        if !matched {
            out.push('&');
            rest = &tail[1..];
        }
    }
    out.push_str(rest);
    out
}

struct Tag {
    name: String,
    attrs: Vec<(String, String)>,
    closing: bool,
    self_closing: bool,
    /// Byte offset just past `>`.
    end: usize,
}

/// Scan one tag starting at `input[at] == '<'`.
fn scan_tag(input: &str, at: usize) -> Result<Tag, SsmlError> {
    let b = input.as_bytes();
    let close = input[at..].find('>').map(|d| at + d).ok_or(SsmlError::Malformed { at })?;
    let inner = &input[at + 1..close];
    let (closing, inner) = match inner.strip_prefix('/') {
        Some(r) => (true, r),
        None => (false, inner),
    };
    let (self_closing, inner) = match inner.strip_suffix('/') {
        Some(r) => (true, r),
        None => (false, inner),
    };
    let inner = inner.trim();
    let name_end = inner.find(|c: char| c.is_whitespace()).unwrap_or(inner.len());
    let name = inner[..name_end].to_lowercase();
    if name.is_empty() || !name.starts_with(|c: char| c.is_ascii_alphabetic()) {
        return Err(SsmlError::Malformed { at });
    }
    let mut attrs = Vec::new();
    let mut rest = inner[name_end..].trim_start();
    while !rest.is_empty() {
        let eq = rest.find('=').ok_or(SsmlError::Malformed { at })?;
        let key = rest[..eq].trim().to_lowercase();
        let after = rest[eq + 1..].trim_start();
        let quote = after.chars().next().ok_or(SsmlError::Malformed { at })?;
        if quote != '"' && quote != '\'' {
            return Err(SsmlError::Malformed { at });
        }
        let vend = after[1..].find(quote).ok_or(SsmlError::Malformed { at })? + 1;
        attrs.push((key, decode_entities(&after[1..vend])));
        rest = after[vend + 1..].trim_start();
    }
    debug_assert_eq!(b[close], b'>');
    Ok(Tag { name, attrs, closing, self_closing, end: close + 1 })
}

/// An element open on the stack, awaiting its close tag.
struct OpenEl {
    name: String,
    kind: Option<CueKind>,
    raw: String,
    source_start: usize,
    text_start: usize,
}

/// Parse an SSML-subset document into a [`CueDoc`].
///
/// Errors rather than degrading: an unsupported tag or an unparseable attribute is a
/// mistake the author must see, not a control to silently drop.
pub fn parse_ssml(input: &str) -> Result<CueDoc, SsmlError> {
    if let (Some(s), Some(br)) = (has_ssml_tag(input), has_bracket_cue(input)) {
        return Err(SsmlError::MixedSyntax { at: s.min(br) });
    }

    let mut text = String::with_capacity(input.len());
    let mut cues: Vec<Cue> = Vec::new();
    let mut stack: Vec<OpenEl> = Vec::new();
    let mut i = 0usize;
    let mut lit_start = 0usize;

    while i < input.len() {
        if input.as_bytes()[i] != b'<' {
            i += 1;
            continue;
        }
        // Only `<name` / `</name` opens a tag; a bare `<` is literal text.
        match input.as_bytes().get(i + 1) {
            Some(c) if c.is_ascii_alphabetic() || *c == b'/' => {}
            _ => {
                i += 1;
                continue;
            }
        }
        text.push_str(&decode_entities(&input[lit_start..i]));
        let tag = scan_tag(input, i)?;

        if tag.closing {
            match stack.pop() {
                Some(open) if open.name == tag.name => {
                    if let Some(kind) = open.kind {
                        cues.push(Cue {
                            span: open.text_start..text.len(),
                            kind,
                            raw: open.raw,
                            source: open.source_start..tag.end,
                        });
                    }
                }
                Some(open) => {
                    return Err(SsmlError::MismatchedClose {
                        expected: Some(open.name),
                        found: tag.name,
                        at: i,
                    })
                }
                None => {
                    return Err(SsmlError::MismatchedClose {
                        expected: None,
                        found: tag.name,
                        at: i,
                    })
                }
            }
        } else {
            let attr = |k: &str| tag.attrs.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str());
            let bad = |a: &str, v: &str| SsmlError::BadAttribute {
                tag: tag.name.clone(),
                attr: a.to_string(),
                value: v.to_string(),
                at: i,
            };
            let raw = input[i..tag.end].to_string();
            match tag.name.as_str() {
                "speak" => {
                    stack.push(OpenEl {
                        name: tag.name.clone(),
                        kind: None,
                        raw,
                        source_start: i,
                        text_start: text.len(),
                    });
                }
                "prosody" => {
                    let rate = match attr("rate") {
                        Some(v) => Some(parse_rate(v).ok_or_else(|| bad("rate", v))?),
                        None => None,
                    };
                    let pitch_st = match attr("pitch") {
                        Some(v) => Some(parse_pitch_st(v).ok_or_else(|| bad("pitch", v))?),
                        None => None,
                    };
                    let volume_db = match attr("volume") {
                        Some(v) => Some(parse_volume_db(v).ok_or_else(|| bad("volume", v))?),
                        None => None,
                    };
                    stack.push(OpenEl {
                        name: tag.name.clone(),
                        kind: Some(CueKind::Prosody { rate, pitch_st, volume_db }),
                        raw,
                        source_start: i,
                        text_start: text.len(),
                    });
                }
                "emphasis" => {
                    let level = match attr("level") {
                        Some(v) => parse_emphasis_level(v).ok_or_else(|| bad("level", v))?,
                        None => Level::Moderate,
                    };
                    stack.push(OpenEl {
                        name: tag.name.clone(),
                        kind: Some(CueKind::Emphasis { level }),
                        raw,
                        source_start: i,
                        text_start: text.len(),
                    });
                }
                "break" => {
                    let ms = match (attr("time"), attr("strength")) {
                        (Some(v), _) => parse_break_time(v).ok_or_else(|| bad("time", v))?,
                        (None, Some(v)) => {
                            parse_break_strength(v).ok_or_else(|| bad("strength", v))?
                        }
                        (None, None) => 400,
                    };
                    cues.push(Cue {
                        span: text.len()..text.len(),
                        kind: CueKind::Pause { ms },
                        raw,
                        source: i..tag.end,
                    });
                }
                _ => {
                    return Err(SsmlError::UnsupportedTag { name: tag.name, at: i });
                }
            }
            // `<break/>` and `<prosody .../>` never take content, so a self-closing tag
            // that OPENED an element must close it here.
            //
            // The guard is `source_start == i` — "the top of the stack is the element this
            // very tag pushed" — and not a bare `pop()`. `<break/>` pushes nothing (it
            // emits its Pause cue directly above), so a bare pop closed whatever element
            // happened to enclose it: `<speak>wait<break/>then</speak>` popped the
            // `<speak>` and then rejected its own `</speak>` as a MismatchedClose. Every
            // fixture that exercised `<break/>` did so at the document root, so the
            // combination that fails is the ordinary one — a real document.
            if tag.self_closing && stack.last().is_some_and(|o| o.source_start == i) {
                if let Some(open) = stack.pop() {
                    if let Some(kind) = open.kind {
                        cues.push(Cue {
                            span: open.text_start..text.len(),
                            kind,
                            raw: open.raw,
                            source: open.source_start..tag.end,
                        });
                    }
                }
            }
        }
        i = tag.end;
        lit_start = tag.end;
    }

    if let Some(open) = stack.pop() {
        return Err(SsmlError::UnclosedTag { name: open.name, at: open.source_start });
    }
    text.push_str(&decode_entities(&input[lit_start..]));
    cues.sort_by_key(|c| c.source.start);
    Ok(CueDoc { text, cues, speakers: Vec::new() })
}
