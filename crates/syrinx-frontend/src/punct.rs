//! Punctuation → prosody mapping (T-01.08).
//!
//! Maps the four recognized punctuation marks in a normalized `&str` to typed
//! prosody markers, one per mark, in source order:
//!
//!   * `.` -> `Boundary { tone: Falling, strength: Full }`
//!   * `,` -> `Break    { kind: Short }`
//!   * `?` -> `Boundary { tone: Rising,  strength: Full }`
//!   * `!` -> `Boundary { tone: Falling, strength: Exclamatory }`
//!
//! Unpunctuated text yields no markers. Markers are typed metadata only — no
//! acoustic realization here. Out of scope: semicolon/colon/dash.

/// Terminal pitch movement of a boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tone {
    /// Falling terminal tone (period, exclamation).
    Falling,
    /// Rising terminal tone (question).
    Rising,
}

/// Boundary strength.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strength {
    /// A full terminal boundary (period, question).
    Full,
    /// An exclamatory boundary (exclamation mark).
    Exclamatory,
}

/// The kind of an internal break.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakKind {
    /// A short internal break (comma).
    Short,
}

/// A typed prosody marker keyed to a recognized punctuation mark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProsodyHint {
    /// A terminal boundary with a pitch movement and strength.
    Boundary { tone: Tone, strength: Strength },
    /// An internal break.
    Break { kind: BreakKind },
}

/// Map the recognized punctuation marks in `input` to prosody hints, in source
/// order — one marker per recognized mark, none for unpunctuated text.
pub fn hints(input: &str) -> Vec<ProsodyHint> {
    let mut out = Vec::new();
    for ch in input.chars() {
        let hint = match ch {
            '.' => ProsodyHint::Boundary {
                tone: Tone::Falling,
                strength: Strength::Full,
            },
            ',' => ProsodyHint::Break {
                kind: BreakKind::Short,
            },
            '?' => ProsodyHint::Boundary {
                tone: Tone::Rising,
                strength: Strength::Full,
            },
            '!' => ProsodyHint::Boundary {
                tone: Tone::Falling,
                strength: Strength::Exclamatory,
            },
            _ => continue,
        };
        out.push(hint);
    }
    out
}
