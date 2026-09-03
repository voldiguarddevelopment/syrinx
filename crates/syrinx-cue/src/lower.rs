//! Lowering: `CueDoc` + [`ControlCaps`] → what a backend actually receives (C2.2).
//!
//! Lowering is a pipeline of small passes, each of which may only *narrow* what a cue
//! asks for, and each of which must record what it did in the [`LoweringReport`]. Nothing
//! is ever silently discarded: a user whose `[whisper]` did nothing must be able to find
//! out why, which is what `--explain` / `?explain=1` surface.
//!
//! ## Passes (C2.2 implements 1–3)
//!
//! 1. **normalize** — resolve every label against the vocabulary to a canonical id, so
//!    later passes never see synonyms or casing variants. Unrecognised labels stay
//!    [`CueKind::Free`].
//! 2. **pass-through** — on an [`Inline::Open`] backend (Fish S2) emit the author's cue
//!    **byte-identical**. This backend has an open vocabulary, so anything we "helpfully"
//!    canonicalise is expressive range thrown away.
//! 3. **vocab map** — on an [`Inline::Closed`] backend, translate the canonical id into
//!    that backend's native spelling. A label with no native spelling is left for the
//!    later fallback passes (C2.4), recorded as [`Action::Unmapped`].
//!
//! Hoisting/splitting (C2.3) and scalar projection, fallbacks and the final strip pass
//! (C2.4) run after these and are implemented in their own tasks.

use crate::caps::{ControlCaps, Inline, Support};
use serde::Serialize;
use crate::ir::{Cue, CueDoc, CueKind, Span};
use crate::vocab::{Kind, Vocab};

/// What lowering did with one cue.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Action {
    /// Emitted verbatim into the text stream (open-vocabulary backend).
    PassedThrough { rendered: String },
    /// A synonym or casing variant was resolved to its canonical id.
    Normalized { from: String, to: String },
    /// Translated to the backend's native spelling.
    Mapped { to: String },
    /// Recognised, and the backend supports the axis, but this label has no native
    /// spelling. Left for the fallback passes.
    Unmapped,
    /// Converted into something the backend can take (a free-text instruction).
    Projected { to: String },
    /// The backend cannot express this at all. Carries why.
    Dropped { reason: DropReason },
}

/// Why a cue could not be delivered. Distinguishing these is the point of the report —
/// "unsupported" and "accepted but ignored" look identical to a user and are not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DropReason {
    /// The backend has no channel for this axis.
    Unsupported,
    /// The backend *accepts* this input and does not act on it. See
    /// [`Support::Accepted`] — e.g. Qwen3-TTS 0.6B-CustomVoice and `instruct`.
    AcceptedButIgnored,
    /// The axis is supported but this specific label has no representation.
    NoNativeSpelling,
}

impl DropReason {
    pub fn explain(self) -> &'static str {
        match self {
            Self::Unsupported => "backend has no channel for this control",
            Self::AcceptedButIgnored => {
                "backend accepts this control but does not act on it (see caps.toml notes)"
            }
            Self::NoNativeSpelling => "no native spelling for this label on this backend",
        }
    }
}

/// One line of the explain report.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReportEntry {
    /// Byte range of the original markup in the author's source.
    pub source: Span,
    /// What the author wrote.
    pub raw: String,
    pub action: Action,
}

impl ReportEntry {
    /// Did this cue actually reach the backend?
    pub fn delivered(&self) -> bool {
        !matches!(self.action, Action::Dropped { .. })
    }
}

/// Everything lowering did, in source order.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct LoweringReport {
    pub entries: Vec<ReportEntry>,
}

impl LoweringReport {
    pub fn push(&mut self, source: Span, raw: String, action: Action) {
        self.entries.push(ReportEntry { source, raw, action });
    }

    pub fn dropped(&self) -> impl Iterator<Item = &ReportEntry> {
        self.entries.iter().filter(|e| !e.delivered())
    }

    pub fn delivered(&self) -> impl Iterator<Item = &ReportEntry> {
        self.entries.iter().filter(|e| e.delivered())
    }

    /// A human-readable explanation, one line per cue. This is what `--explain` prints.
    pub fn explain(&self) -> String {
        let mut out = String::new();
        for e in &self.entries {
            let line = match &e.action {
                Action::PassedThrough { rendered } => {
                    format!("{:?} -> passed through verbatim as {rendered:?}", e.raw)
                }
                Action::Normalized { from, to } => format!("{from:?} -> normalized to {to:?}"),
                Action::Mapped { to } => format!("{:?} -> native spelling {to:?}", e.raw),
                Action::Unmapped => {
                    format!("{:?} -> recognised but has no native spelling here", e.raw)
                }
                Action::Projected { to } => format!("{:?} -> projected to {to:?}", e.raw),
                Action::Dropped { reason } => {
                    format!("{:?} -> DROPPED: {}", e.raw, reason.explain())
                }
            };
            out.push_str(&line);
            out.push('\n');
        }
        out
    }
}

/// The result of lowering: the cues as narrowed for one backend, plus the report.
#[derive(Debug, Clone, PartialEq)]
pub struct Lowered {
    /// Clean text, unchanged by passes 1–3 (text rewriting happens in C2.4's strip pass
    /// and C2.3's splitting).
    pub text: String,
    /// The cues that survived, each rewritten for the target.
    pub cues: Vec<Cue>,
    pub report: LoweringReport,
}

/// The label a cue carries, if it carries one.
fn label_of(kind: &CueKind) -> Option<&str> {
    match kind {
        CueKind::Emotion { label, .. } | CueKind::Style { label } | CueKind::Event { label } => {
            Some(label)
        }
        _ => None,
    }
}

fn with_label(kind: &CueKind, new: String) -> CueKind {
    match kind {
        CueKind::Emotion { intensity, .. } => {
            CueKind::Emotion { label: new, intensity: *intensity }
        }
        CueKind::Style { .. } => CueKind::Style { label: new },
        CueKind::Event { .. } => CueKind::Event { label: new },
        other => other.clone(),
    }
}

/// **Pass 1 — normalize.** Resolve every label to its canonical vocabulary id.
///
/// After this pass no later pass has to know that "cheerful" and "Happy" are "happy",
/// which is what keeps the mapping table small and the behaviour testable.
pub fn pass_normalize(doc: &mut CueDoc, vocab: &Vocab, report: &mut LoweringReport) {
    for cue in &mut doc.cues {
        let Some(label) = label_of(&cue.kind) else { continue };
        let Some((_, entry)) = vocab.resolve(label) else { continue };
        if entry.id != label {
            let from = label.to_string();
            let to = entry.id.clone();
            cue.kind = with_label(&cue.kind, to.clone());
            report.push(cue.source.clone(), cue.raw.clone(), Action::Normalized { from, to });
        }
    }
}

/// The native spelling of `id` on this backend, if any.
fn native_spelling(caps: &ControlCaps, vocab: &Vocab, id: &str) -> Option<String> {
    let (_, e) = vocab.by_id(id)?;
    match caps.krate.as_str() {
        "syrinx-fish" if caps.id == "fish-s1-mini" => e.fish_s1.clone(),
        "syrinx-fish" => e.fish_s2.clone(),
        "syrinx-serve" => e.cosyvoice.clone(),
        _ => None,
    }
}

/// **Pass 2 — pass-through**, and **Pass 3 — vocab map**.
///
/// Run together because they are the two halves of one decision: what a backend's inline
/// channel can carry. Splitting them would mean walking the cue list twice to ask the same
/// question.
pub fn pass_passthrough_and_map(
    doc: &mut CueDoc,
    caps: &ControlCaps,
    vocab: &Vocab,
    report: &mut LoweringReport,
) {
    let mut kept = Vec::with_capacity(doc.cues.len());
    for cue in std::mem::take(&mut doc.cues) {
        let support = caps.support_for(&cue.kind);
        // `Accepted` is not `Honored`: it lowers exactly like unsupported, but the report
        // must say which it was, because the two are indistinguishable to a listener.
        if !support.is_effective() {
            let reason = match support {
                Support::Accepted => DropReason::AcceptedButIgnored,
                _ => DropReason::Unsupported,
            };
            report.push(cue.source.clone(), cue.raw.clone(), Action::Dropped { reason });
            continue;
        }

        match caps.inline {
            // Pass 2. Byte-identical: `raw` is exactly what the author typed.
            Inline::Open => {
                report.push(
                    cue.source.clone(),
                    cue.raw.clone(),
                    Action::PassedThrough { rendered: cue.raw.clone() },
                );
                kept.push(cue);
            }
            // Pass 3.
            Inline::Closed => match label_of(&cue.kind) {
                Some(label) => match native_spelling(caps, vocab, label) {
                    Some(native) => {
                        let mut c = cue.clone();
                        c.kind = with_label(&cue.kind, native.clone());
                        report.push(
                            cue.source.clone(),
                            cue.raw.clone(),
                            Action::Mapped { to: native },
                        );
                        kept.push(c);
                    }
                    None => {
                        report.push(cue.source.clone(), cue.raw.clone(), Action::Unmapped);
                        kept.push(cue);
                    }
                },
                // Labelless cues (pause, emphasis, speaker) need no spelling.
                None => {
                    report.push(cue.source.clone(), cue.raw.clone(), Action::Unmapped);
                    kept.push(cue);
                }
            },
            // No inline channel: whatever survives must reach the backend by another
            // route (instruct), which is C2.3's hoist pass.
            Inline::None => {
                report.push(cue.source.clone(), cue.raw.clone(), Action::Unmapped);
                kept.push(cue);
            }
        }
    }
    doc.cues = kept;
}

/// Run passes 1–3.
pub fn lower(doc: &CueDoc, caps: &ControlCaps, vocab: &Vocab) -> Lowered {
    let mut doc = doc.clone();
    let mut report = LoweringReport::default();
    pass_normalize(&mut doc, vocab, &mut report);
    pass_passthrough_and_map(&mut doc, caps, vocab, &mut report);
    Lowered { text: doc.text, cues: doc.cues, report }
}

/// Run the **whole** pipeline: passes 1–3, then C2.4's projection/fallbacks.
///
/// `lower` deliberately stops after pass 3 (it is what the per-pass tests pin); callers
/// that want the finished result — the CLI, the server's `?explain=1` — want this. The
/// projection pass runs BEFORE the inline decision, so a prosody cue that becomes a
/// free-text instruction still gets a chance to reach an instructable backend.
pub fn lower_full(doc: &CueDoc, caps: &ControlCaps, vocab: &Vocab) -> Lowered {
    let mut doc = doc.clone();
    let mut report = LoweringReport::default();
    pass_normalize(&mut doc, vocab, &mut report);
    pass_project_fallbacks(&mut doc, caps, &mut report);
    pass_passthrough_and_map(&mut doc, caps, vocab, &mut report);
    let text = pass_strip(&doc.text);
    Lowered { text, cues: doc.cues, report }
}

/// Convenience: the vocabulary kind a cue belongs to, for reporting.
pub fn kind_of(cue: &Cue, vocab: &Vocab) -> Option<Kind> {
    let label = label_of(&cue.kind)?;
    vocab.by_id(label).map(|(k, _)| k)
}

// ==================================================================== C2.4

/// Render a prosody request as a natural-language phrase.
///
/// Used when a backend has no prosody axis but *does* take an instruction: a `<prosody
/// rate="slow">` is then still honoured in spirit rather than thrown away. Thresholds are
/// deliberate and pinned by tests on both sides, because "roughly slower" is not a spec.
pub fn project_prosody(rate: Option<f32>, pitch_st: Option<f32>, volume_db: Option<f32>)
    -> Option<String>
{
    let mut parts: Vec<&str> = Vec::new();
    if let Some(r) = rate {
        // 1.0 is unchanged; a 10% deviation is the smallest one worth a word.
        if r <= 0.9 {
            parts.push("slowly");
        } else if r >= 1.1 {
            parts.push("quickly");
        }
    }
    if let Some(p) = pitch_st {
        // One semitone is about the smallest reliably audible step.
        if p <= -1.0 {
            parts.push("in a lower pitch");
        } else if p >= 1.0 {
            parts.push("in a higher pitch");
        }
    }
    if let Some(v) = volume_db {
        if v <= -3.0 {
            parts.push("quietly");
        } else if v >= 3.0 {
            parts.push("loudly");
        }
    }
    if parts.is_empty() {
        return None;
    }
    Some(format!("Speak {}", parts.join(", ")))
}

/// Render an emphasis level as a phrase, for backends with no emphasis channel.
pub fn project_emphasis(level: crate::ir::Level) -> &'static str {
    match level {
        crate::ir::Level::Reduced => "Speak this part with less emphasis",
        crate::ir::Level::Moderate => "Emphasise this part",
        crate::ir::Level::Strong => "Emphasise this part strongly",
    }
}

/// **Pass 5 — scalar projection and fallbacks.**
///
/// A cue the backend cannot express natively is converted into something it *can* take —
/// a free-text instruction — rather than being dropped, whenever the backend has any
/// instruction channel at all. Cues on a backend with no channel are dropped with a
/// reason, as always.
pub fn pass_project_fallbacks(
    doc: &mut CueDoc,
    caps: &ControlCaps,
    report: &mut LoweringReport,
) {
    // Only worth projecting if the backend actually obeys an instruction.
    let instructable = caps.instruct.is_effective();
    let mut kept = Vec::with_capacity(doc.cues.len());
    for cue in std::mem::take(&mut doc.cues) {
        let projected = match (&cue.kind, caps.support_for(&cue.kind).is_effective()) {
            // Natively supported: leave it alone.
            (_, true) => Some(cue.clone()),
            (CueKind::Prosody { rate, pitch_st, volume_db }, false) if instructable => {
                project_prosody(*rate, *pitch_st, *volume_db).map(|phrase| {
                    let mut c = cue.clone();
                    c.kind = CueKind::Free;
                    c.raw = phrase;
                    c
                })
            }
            (CueKind::Emphasis { level }, false) if instructable => {
                let mut c = cue.clone();
                c.kind = CueKind::Free;
                c.raw = project_emphasis(*level).to_string();
                Some(c)
            }
            _ => None,
        };
        match projected {
            Some(c) => {
                if c.kind == CueKind::Free && c.raw != cue.raw {
                    report.push(
                        cue.source.clone(),
                        cue.raw.clone(),
                        Action::Projected { to: c.raw.clone() },
                    );
                }
                kept.push(c);
            }
            None => {
                let reason = if caps.support_for(&cue.kind) == Support::Accepted {
                    DropReason::AcceptedButIgnored
                } else {
                    DropReason::Unsupported
                };
                report.push(cue.source.clone(), cue.raw.clone(), Action::Dropped { reason });
            }
        }
    }
    doc.cues = kept;
}

/// **Pass 6 — strip.** The last line of defence for the hard invariant.
///
/// By construction the parser already guarantees the clean text carries no cue markup, so
/// on a correct pipeline this is a no-op. It exists because "by construction" is exactly
/// the kind of claim that quietly stops being true, and the cost of being wrong here is a
/// backend speaking `[happy]` aloud. Escaped brackets are restored to their literal form,
/// which is the one place `\[` becomes `[`.
pub fn pass_strip(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        // Restore an escape to its literal character.
        if b[i] == b'\\' && matches!(b.get(i + 1), Some(b'[') | Some(b']')) {
            out.push(b[i + 1] as char);
            i += 2;
            continue;
        }
        if b[i] == b'[' || b[i] == b']' {
            i += 1; // stray markup: never spoken
            continue;
        }
        if text[i..].starts_with("<|speaker") {
            match text[i..].find("|>") {
                Some(end) => i += end + 2,
                None => i += "<|speaker".len(),
            }
            continue;
        }
        let ch = text[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}
