//! **syrinx-cue** — the canonical expressive-cue IR and all lowering.
//!
//! Per ADR-0001, this crate owns cue parsing and every lowering pass; no backend crate
//! may parse bracket syntax itself. See `docs/backends/CONTROL_SURVEY.md` for the
//! verified per-backend control surfaces this must lower onto.
//!
//! ## Hard invariant
//!
//! No bracket cue text and no speaker token may ever reach a backend as literal text.
//! Enforced by property test; a failure is a release blocker.

pub mod caps;
pub mod hoist;
pub mod ir;
pub mod legacy_emotion;
pub mod lower;
pub mod offsets;
pub mod parse;
pub mod ssml;
pub mod vocab;

pub use caps::{BackendId, CapsTable, ControlCaps, ExpressiveBackend, Granularity, Inline, Support};
pub use hoist::{instruct_for, pass_hoist, SplitOptions, UtteranceSegment};
pub use ir::{Cue, CueDoc, CueKind, Level, SpeakerRef, Span};
pub use legacy_emotion::{parse_tagged, EmotionInstruct, EmotionRegistry, InstructLang, Segment, TagSyntax};
pub use lower::{lower, lower_full, Action, DropReason, Lowered, LoweringReport, ReportEntry};
pub use offsets::{interleave, whitespace_spans, CueAnchor, Emission, OffsetMap, TokenSpan};
pub use parse::{parse, ParseOptions, Syntax};
pub use ssml::{detect, parse_any, parse_ssml, Dialect, SsmlError};
pub use vocab::{Entry, Kind, Vocab, VocabError};
