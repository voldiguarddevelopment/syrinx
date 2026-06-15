//! Recursive-descent parser for the documented SSML subset (T-01.07).
//!
//! [`parse`] is a total function from a `&str` of SSML-or-plain-text to a
//! `Result<Vec<ControlEvent>, SsmlError>`: every input yields either typed
//! [`ControlEvent`]s in source order or a typed [`SsmlError`], and never panics.
//!
//! The fixed subset is exactly the tags `break`, `emphasis`, `prosody`, `say-as`,
//! `phoneme`, and `sub`. `break` is a void element that must self-close
//! (`<break .../>`) and emits a single [`ControlEvent::Break`]. The other five are
//! container tags that open (`<tag ...>`), wrap inner text, and close (`</tag>`),
//! each emitting its open variant, the inner [`ControlEvent::Text`], and its
//! matching `*End` variant. Plain text with no markup becomes one `Text` event.
//!
//! Malformed input (an unclosed void tag) and any out-of-subset tag are errors,
//! not silently dropped. Out of scope: no DTD/namespace validation or external
//! entity resolution.

/// Emphasis strength carried by an [`ControlEvent::Emphasis`] open event. The
/// fixed subset pins the `strong` level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmphasisLevel {
    /// `<emphasis level="strong">`.
    Strong,
}

/// A typed control event produced by [`parse`]. Container tags emit an open
/// variant, then their inner [`Text`](ControlEvent::Text), then a matching `*End`;
/// the void `break` emits a single [`Break`](ControlEvent::Break).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlEvent {
    /// Literal text run between (or outside) tags.
    Text(String),
    /// `<break time="..ms"/>` — a pause of `ms` milliseconds.
    Break { ms: u32 },
    /// `<emphasis level="..">` open.
    Emphasis { level: EmphasisLevel },
    /// `</emphasis>` close.
    EmphasisEnd,
    /// `<prosody rate="..">` open.
    Prosody { rate: String },
    /// `</prosody>` close.
    ProsodyEnd,
    /// `<say-as interpret-as="..">` open.
    SayAs { interpret_as: String },
    /// `</say-as>` close.
    SayAsEnd,
    /// `<phoneme ph="..">` open.
    Phoneme { ph: String },
    /// `</phoneme>` close.
    PhonemeEnd,
    /// `<sub alias="..">` open.
    Sub { alias: String },
    /// `</sub>` close.
    SubEnd,
}

/// The typed error returned for malformed or out-of-subset SSML.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SsmlError;

/// Parse `input` into typed control events in source order.
///
/// Text outside tags becomes [`ControlEvent::Text`]; each subset tag becomes its
/// typed variant. An unterminated or out-of-subset tag returns [`SsmlError`].
pub fn parse(input: &str) -> Result<Vec<ControlEvent>, SsmlError> {
    let mut events = Vec::new();
    let mut rest = input;
    loop {
        match rest.find('<') {
            None => {
                if !rest.is_empty() {
                    events.push(ControlEvent::Text(rest.to_string()));
                }
                return Ok(events);
            }
            Some(lt) => {
                let (before, from_lt) = rest.split_at(lt);
                if !before.is_empty() {
                    events.push(ControlEvent::Text(before.to_string()));
                }
                let close = from_lt.find('>').ok_or(SsmlError)?;
                let raw = &from_lt[1..close];
                parse_tag(raw, &mut events)?;
                rest = &from_lt[close + 1..];
            }
        }
    }
}

/// Dispatch the body of a single `<...>` tag (the text between the angle
/// brackets) into the appropriate event(s), appended to `events`.
fn parse_tag(raw: &str, events: &mut Vec<ControlEvent>) -> Result<(), SsmlError> {
    if let Some(name) = raw.strip_prefix('/') {
        return push_close(name, events);
    }
    if let Some(inner) = raw.strip_suffix('/') {
        return push_void(inner, events);
    }
    push_open(raw, events)
}

/// Handle a void (self-closing) tag body, e.g. `break time="200ms"`. Only `break`
/// is a valid void element; anything else is out of subset.
fn push_void(inner: &str, events: &mut Vec<ControlEvent>) -> Result<(), SsmlError> {
    let (name, attrs) = split_name(inner);
    match name {
        "break" => {
            let ms = parse_break_ms(attrs)?;
            events.push(ControlEvent::Break { ms });
            Ok(())
        }
        _ => Err(SsmlError),
    }
}

/// Handle a container open tag body, e.g. `prosody rate="slow"`, mapping the fixed
/// subset to its open variant. `break` is void-only here, so it (and any unknown
/// tag) is an error.
fn push_open(raw: &str, events: &mut Vec<ControlEvent>) -> Result<(), SsmlError> {
    let (name, attrs) = split_name(raw);
    match name {
        "emphasis" => {
            let value = attr_value(attrs).ok_or(SsmlError)?;
            match value {
                "strong" => events.push(ControlEvent::Emphasis {
                    level: EmphasisLevel::Strong,
                }),
                _ => return Err(SsmlError),
            }
        }
        "prosody" => {
            let rate = attr_value(attrs).ok_or(SsmlError)?.to_string();
            events.push(ControlEvent::Prosody { rate });
        }
        "say-as" => {
            let interpret_as = attr_value(attrs).ok_or(SsmlError)?.to_string();
            events.push(ControlEvent::SayAs { interpret_as });
        }
        "phoneme" => {
            let ph = attr_value(attrs).ok_or(SsmlError)?.to_string();
            events.push(ControlEvent::Phoneme { ph });
        }
        "sub" => {
            let alias = attr_value(attrs).ok_or(SsmlError)?.to_string();
            events.push(ControlEvent::Sub { alias });
        }
        _ => return Err(SsmlError),
    }
    Ok(())
}

/// Handle a closing tag body, e.g. `emphasis` from `</emphasis>`, mapping the
/// fixed subset to its `*End` variant.
fn push_close(name: &str, events: &mut Vec<ControlEvent>) -> Result<(), SsmlError> {
    match name {
        "emphasis" => events.push(ControlEvent::EmphasisEnd),
        "prosody" => events.push(ControlEvent::ProsodyEnd),
        "say-as" => events.push(ControlEvent::SayAsEnd),
        "phoneme" => events.push(ControlEvent::PhonemeEnd),
        "sub" => events.push(ControlEvent::SubEnd),
        _ => return Err(SsmlError),
    }
    Ok(())
}

/// Split a tag body into its element name and the remaining attribute text,
/// dividing at the first whitespace (no whitespace ⇒ no attributes).
fn split_name(inner: &str) -> (&str, &str) {
    match inner.find(char::is_whitespace) {
        Some(i) => (&inner[..i], &inner[i..]),
        None => (inner, ""),
    }
}

/// Extract the quoted value of the single attribute in `attrs`
/// (e.g. `rate="slow"` ⇒ `slow`), or `None` if it is missing or unquoted.
fn attr_value(attrs: &str) -> Option<&str> {
    let eq = attrs.find('=')?;
    let rest = &attrs[eq + 1..];
    rest.trim().strip_prefix('"')?.strip_suffix('"')
}

/// Parse the `time="..ms"` attribute of a `break` into whole milliseconds.
fn parse_break_ms(attrs: &str) -> Result<u32, SsmlError> {
    let value = attr_value(attrs).ok_or(SsmlError)?;
    let digits = value.strip_suffix("ms").ok_or(SsmlError)?;
    digits.parse::<u32>().map_err(|_| SsmlError)
}
