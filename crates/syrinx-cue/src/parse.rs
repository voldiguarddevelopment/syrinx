//! The cue parser (C1.2). Single owner of bracket syntax in the workspace — ADR-0001 §2.1.
//!
//! ## Semantics (spec §2.1, Fish S2 boundary rules)
//!
//! * `[free text]` scopes the text that FOLLOWS it, until the next span cue or the end of
//!   the sentence — whichever comes first. There is no blending.
//! * Point events (`[laughs]`, `[pause 400ms]`, …) are zero-width and do NOT scope.
//!   Whether a label is a point event is decided by the vocabulary's `Kind::Event`, not by
//!   spelling, so the two cannot drift apart.
//! * `[emphasis]word[/emphasis]` is the explicit span form; any span cue may use it.
//! * `<|speaker:N|>` and `[speaker N]` open a turn.
//! * `\[` and `\]` are literal brackets and never open a cue.
//! * Legacy `(tag)` mode is OFF by default (S1 scripts opt in).
//!
//! ## What is deliberately NOT a cue
//!
//! Under the strict rule (ADR-0001 §9.1 / D5) every unescaped `[...]` IS cue syntax, so
//! this list is short and structural, not a prose heuristic: empty content, content
//! spanning a newline, an unmatched closer, and an opener with no close within
//! [`MAX_SCAN_CHARS`]. All of those are **dropped**, never emitted as literal text — that
//! totality is what makes the hard invariant machine-checkable instead of best-effort.
//! A stage direction is therefore a `Free` cue, not prose; `\[He turns\]` is the way to
//! say it out loud.

use crate::ir::{Cue, CueDoc, CueKind, Level, SpeakerRef};
use crate::vocab::{Kind, Vocab};

/// Which delimiters open a cue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Syntax {
    /// `[tag]` only. The default.
    Brackets,
    /// `(tag)` only — Fish S1 legacy scripts.
    Parens,
    /// Both. Opt-in; ambiguous for prose containing parentheses.
    Both,
}

impl Default for Syntax {
    fn default() -> Self {
        Self::Brackets
    }
}

impl Syntax {
    fn close_for(self, c: char) -> Option<char> {
        match (c, self) {
            ('[', Self::Brackets | Self::Both) => Some(']'),
            ('(', Self::Parens | Self::Both) => Some(')'),
            _ => None,
        }
    }
}

/// Parser options.
#[derive(Debug, Clone)]
pub struct ParseOptions {
    pub syntax: Syntax,
    /// Default intensity for an emotion with no modifier word.
    pub default_intensity: f32,
}

impl Default for ParseOptions {
    fn default() -> Self {
        Self { syntax: Syntax::default(), default_intensity: 0.6 }
    }
}

/// Hard ceiling on how far the parser will scan for a closing delimiter.
///
/// Not a prose heuristic — under the strict rule (ADR-0001 §9 option a) every unescaped
/// bracket is cue syntax. This only stops a *missing* `]` from swallowing a whole
/// paragraph: beyond this distance the `[` is treated as a stray and dropped.
const MAX_SCAN_CHARS: usize = 200;

/// Does `inner` form a cue body?
///
/// Under the strict rule this is deliberately NOT a vocabulary or prose test — Fish S2
/// takes free text, so `[He turns to the window, slowly]` is a legitimate (if unusual)
/// style instruction and becomes a `Free` cue. The only rejections are shapes that cannot
/// be a cue at all: empty content, or content spanning a newline (which means the `]`
/// belongs to a different line and the `[` is a stray).
///
/// Anything rejected here is **stripped**, never emitted as literal text — that is what
/// makes the hard invariant total rather than heuristic.
pub fn is_tag_shaped(inner: &str) -> bool {
    !inner.trim().is_empty() && !inner.contains('\n')
}

/// Intensity modifiers. Applied multiplicatively to the default.
fn intensity_of(words: &[&str], default: f32) -> (f32, usize) {
    const UP: [(&str, f32); 5] = [
        ("very", 1.4), ("super", 1.5), ("extremely", 1.6), ("really", 1.3), ("so", 1.2),
    ];
    const DOWN: [(&str, f32); 4] =
        [("slightly", 0.5), ("a bit", 0.55), ("mildly", 0.6), ("somewhat", 0.7)];
    if let Some(first) = words.first() {
        let lower = first.to_lowercase();
        for (w, m) in UP.iter().chain(DOWN.iter()) {
            if lower == *w {
                return ((default * m).clamp(0.0, 1.0), 1);
            }
        }
    }
    (default, 0)
}

/// Parse `[pause 400ms]` / `[pause 0.4s]` / `[break 250]`.
fn parse_pause(inner: &str) -> Option<u32> {
    let t = inner.trim().to_lowercase();
    let rest = t.strip_prefix("pause").or_else(|| t.strip_prefix("break"))?.trim();
    if rest.is_empty() {
        return Some(300); // bare `[pause]`
    }
    if let Some(v) = rest.strip_suffix("ms") {
        return v.trim().parse::<f32>().ok().map(|x| x.max(0.0) as u32);
    }
    if let Some(v) = rest.strip_suffix('s') {
        return v.trim().parse::<f32>().ok().map(|x| (x.max(0.0) * 1000.0) as u32);
    }
    rest.parse::<f32>().ok().map(|x| x.max(0.0) as u32)
}

/// Parse `[speaker 2]`.
fn parse_speaker_alias(inner: &str) -> Option<u32> {
    let t = inner.trim().to_lowercase();
    let rest = t.strip_prefix("speaker")?.trim();
    rest.parse::<u32>().ok()
}

fn emphasis_level(inner: &str) -> Option<Level> {
    match inner.trim().to_lowercase().as_str() {
        "emphasis" | "emphasise" | "emphasize" | "stress" => Some(Level::Moderate),
        "strong emphasis" | "strong" => Some(Level::Strong),
        "reduced emphasis" | "reduced" | "de-emphasis" => Some(Level::Reduced),
        _ => None,
    }
}

/// A cue opened but not yet closed by the next boundary.
struct Open {
    kind: CueKind,
    raw: String,
    source_start: usize,
    text_start: usize,
    /// Set when opened by an explicit `[x]…[/x]` form; closes only on the matching tag.
    explicit: Option<String>,
}

/// Parse authoring text into a [`CueDoc`].
pub fn parse(input: &str, vocab: &Vocab, opts: &ParseOptions) -> CueDoc {
    let b = input.as_bytes();
    let mut text = String::with_capacity(input.len());
    let mut cues: Vec<Cue> = Vec::new();
    let mut speakers: Vec<SpeakerRef> = Vec::new();
    let mut open: Vec<Open> = Vec::new();
    let mut i = 0usize;

    // Close every span cue that is still open, ending at the current clean-text offset.
    fn close_all(open: &mut Vec<Open>, cues: &mut Vec<Cue>, at: usize, src_end: usize) {
        while let Some(o) = open.pop() {
            cues.push(Cue {
                span: o.text_start..at,
                kind: o.kind,
                raw: o.raw,
                source: o.source_start..src_end,
            });
        }
    }

    while i < b.len() {
        let c = input[i..].chars().next().unwrap();
        let clen = c.len_utf8();

        // --- escapes: \[ and \] are literal, and never open a cue -----------------
        if c == '\\' && i + 1 < b.len() {
            let n = input[i + clen..].chars().next().unwrap();
            if n == '[' || n == ']' || n == '(' || n == ')' {
                text.push(n);
                i += clen + n.len_utf8();
                continue;
            }
        }

        // --- speaker token <|speaker:N|> -----------------------------------------
        if input[i..].starts_with("<|speaker:") {
            // A speaker-token-shaped sequence must NEVER survive into the clean text,
            // even when malformed — it is exactly the leak the hard invariant names.
            // Well-formed => a turn; well-formed delimiters with a bad id => drop the
            // whole token; no closing delimiter => drop the marker prefix.
            match input[i..].find("|>") {
                Some(end) => {
                    let inner = &input[i + "<|speaker:".len()..i + end];
                    if inner.trim().parse::<u32>().is_err() {
                        i += end + 2;
                        continue;
                    }
                }
                None => {
                    i += "<|speaker:".len();
                    continue;
                }
            }
            if let Some(end) = input[i..].find("|>") {
                let inner = &input[i + "<|speaker:".len()..i + end];
                if let Ok(id) = inner.trim().parse::<u32>() {
                    // A speaker turn is a hard boundary: nothing scopes across it.
                    close_all(&mut open, &mut cues, text.len(), i + end + 2);
                    cues.push(Cue {
                        span: text.len()..text.len(),
                        kind: CueKind::SpeakerTurn { id },
                        // Canonical form, identical for `<|speaker:N|>` and `[speaker N]`,
                        // so serialise->parse is lossless whichever the author wrote.
                        raw: format!("speaker:{id}"),
                        source: i..i + end + 2,
                    });
                    speakers.push(SpeakerRef { id, at: text.len() });
                    i += end + 2;
                    continue;
                }
            }
        }

        // --- bracketed / parenthesised cue ---------------------------------------
        if let Some(close) = opts.syntax.close_for(c) {
            let window_end = input.len().min(
                i + clen
                    + input[i + clen..]
                        .char_indices()
                        .nth(MAX_SCAN_CHARS)
                        .map(|(o, _)| o)
                        .unwrap_or(input.len() - i - clen),
            );
            if let Some(rel) = input[i + clen..window_end].find(close) {
                let j = i + clen + rel;
                let inner = &input[i + clen..j];
                if is_tag_shaped(inner) {
                    let src = i..j + close.len_utf8();
                    let trimmed = inner.trim();

                    // explicit close: [/emphasis]
                    if let Some(name) = trimmed.strip_prefix('/') {
                        let want = name.trim().to_lowercase();
                        if let Some(pos) =
                            open.iter().rposition(|o| o.explicit.as_deref() == Some(want.as_str()))
                        {
                            let o = open.remove(pos);
                            cues.push(Cue {
                                span: o.text_start..text.len(),
                                kind: o.kind,
                                raw: o.raw,
                                source: o.source_start..src.end,
                            });
                        }
                        i = src.end;
                        continue;
                    }

                    let raw = trimmed.to_string();
                    let lower = raw.to_lowercase();

                    // point events that are not vocabulary lookups
                    if let Some(ms) = parse_pause(&lower) {
                        cues.push(Cue {
                            span: text.len()..text.len(),
                            kind: CueKind::Pause { ms },
                            raw,
                            source: src.clone(),
                        });
                        i = src.end;
                        continue;
                    }
                    if let Some(id) = parse_speaker_alias(&lower) {
                        close_all(&mut open, &mut cues, text.len(), src.end);
                        cues.push(Cue {
                            span: text.len()..text.len(),
                            kind: CueKind::SpeakerTurn { id },
                            raw: format!("speaker:{id}"),
                            source: src.clone(),
                        });
                        speakers.push(SpeakerRef { id, at: text.len() });
                        i = src.end;
                        continue;
                    }

                    // emphasis (span cue, may be explicit-closed)
                    if let Some(level) = emphasis_level(&lower) {
                        close_all(&mut open, &mut cues, text.len(), src.end);
                        open.push(Open {
                            kind: CueKind::Emphasis { level },
                            raw,
                            source_start: src.start,
                            text_start: text.len(),
                            explicit: Some(lower.clone()),
                        });
                        i = src.end;
                        continue;
                    }

                    // vocabulary lookup, with an optional leading intensity modifier
                    let words: Vec<&str> = lower.split_whitespace().collect();
                    let (intensity, skip) = intensity_of(&words, opts.default_intensity);
                    let label_text = words[skip..].join(" ");
                    let resolved = vocab
                        .resolve(&label_text)
                        .or_else(|| vocab.resolve(&lower));

                    match resolved {
                        // Events are zero-width: they do not scope following text.
                        Some((Kind::Event, e)) => {
                            cues.push(Cue {
                                span: text.len()..text.len(),
                                kind: CueKind::Event { label: e.id.clone() },
                                raw,
                                source: src.clone(),
                            });
                        }
                        Some((Kind::Emotion, e)) => {
                            close_all(&mut open, &mut cues, text.len(), src.end);
                            open.push(Open {
                                kind: CueKind::Emotion { label: e.id.clone(), intensity },
                                raw,
                                source_start: src.start,
                                text_start: text.len(),
                                explicit: Some(lower.clone()),
                            });
                        }
                        Some((Kind::Style, e)) => {
                            close_all(&mut open, &mut cues, text.len(), src.end);
                            open.push(Open {
                                kind: CueKind::Style { label: e.id.clone() },
                                raw,
                                source_start: src.start,
                                text_start: text.len(),
                                explicit: Some(lower.clone()),
                            });
                        }
                        // Unrecognised: a Free SPAN cue carrying `raw` verbatim. Never
                        // dropped — ADR-0001 §2.2, and the reason Fish S2 loses nothing.
                        None => {
                            close_all(&mut open, &mut cues, text.len(), src.end);
                            open.push(Open {
                                kind: CueKind::Free,
                                raw,
                                source_start: src.start,
                                text_start: text.len(),
                                explicit: Some(lower.clone()),
                            });
                        }
                    }
                    i = src.end;
                    continue;
                }
            }
            // Strict rule (ADR-0001 §9a): an unescaped opener that forms no cue is a
            // stray. Drop it rather than emit it — a bracket must never be speakable.
            i += clen;
            continue;
        }

        // An unmatched closing delimiter is a stray under the strict rule: drop it.
        if c == ']' || (opts.syntax.close_for('(').is_some() && c == ')') {
            i += clen;
            continue;
        }

        // --- sentence boundary closes span cues (Fish semantics) ------------------
        text.push(c);
        i += clen;
        if matches!(c, '.' | '!' | '?' | '\n') {
            close_all(&mut open, &mut cues, text.len(), i);
        }
    }

    close_all(&mut open, &mut cues, text.len(), input.len());
    cues.sort_by_key(|c| (c.source.start, c.span.start));
    CueDoc { text, cues, speakers }
}
