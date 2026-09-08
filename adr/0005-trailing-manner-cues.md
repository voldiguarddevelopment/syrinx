# ADR-0005 — a cue written after its text is a manner, not an event

Status: **PROPOSED** (acceptance is a human act — ADR-0001 §7 / C0.2′)
Date: 2026-09-09
Narrows: ADR-0001's point/span distinction. Supersedes nothing.

## The defect

`"That was the strangest thing I have seen all week. [angry]"` produced **no instruction and
no drop record**. The sentence was spoken plainly and nothing said otherwise.

The parser gives a trailing cue an **empty span** (`51..51`), correctly: a spanning cue
scopes the text that *follows* it, and nothing follows. `Cue::is_point()` is
`span.start == span.end`, so the cue was filed as a point event. `pass_hoist` routes points
to `UtteranceSegment::cues` for backends that can place them inline, and never offers them
to `instruct_for`. On Qwen — no inline channel, no event channel — the cue was inert.

`placement: trailing` is one of the three placements in the frozen C4.2 cue set, so that
set has been measuring a no-op on those cases since it was written.

## Decision

**A zero-span cue whose kind is `Emotion` or `Style` is a *manner* cue, not a point**, and
applies to the segment preceding it.

The distinction is not a convention, it is what the kinds mean. An **event** is a sound
occurring at an instant — `[laughs]`, `[cough]` — and a zero-span event is exactly right.
An **emotion or style** is a manner of speaking; it cannot occur at an instant, so a
zero-span one is never an event. It is a cue written after the text it describes, which is
an ordinary way to write one.

Consequences, all pinned by `tests/cue_trailing_manner.rs`:

- a trailing emotion or style sets the last segment's instruction;
- a zero-span **event** stays a point and sets no instruction — collapsing this would make
  `[laughs]` emit "Speak in a laugh tone", the nonsense `instruct.toml` deliberately has no
  row for;
- if the last segment already carries an instruction, the trailing cue **loses and is
  reported** as `Dropped`. Two deliveries for one span is a conflict, and silent discard is
  the original sin being fixed;
- leading and mid placements are untouched, and segments still concatenate to the exact
  spoken text.

## Why this reading rather than "it scopes nothing"

The alternative — treat a zero-span manner cue as scoping nothing and drop it with a
report — is more conservative and was rejected on the evidence. The frozen cue set includes
`trailing` as a placement to be measured for *activation*, which only makes sense if the
project expects it to do something. Dropping it would make those cases permanently
non-activating by design, and would encode "trailing cues are meaningless" as a decision
nobody actually took.

Either reading is an improvement on the current behaviour, because the current behaviour is
**silent**: no instruction and no report, which the project's own principle forbids.

## How it was found, which matters more than the fix

Not by review, and not by any existing test. The C4.2′ runner asserts the difference
between "the backend cannot express this kind" (legitimately `not_applicable`) and "caps say
it can and yet nothing carried an instruct" (a lowering bug). That assertion was added on
2026-09-09 after the runner's *own* first version mislabelled four sentinels as
`not_applicable` by reading `segments.first()` instead of the segment carrying the instruct.

Fixing that bug made the runner able to see this one. The general lesson is in the shape:
a gate that distinguishes *kinds* of null result finds defects that a gate reporting a bare
null cannot.

## What is not addressed

- **Whether a trailing cue should out-rank a leading one.** Currently the cue already in
  effect wins and the trailing one is reported. That is a defensible default, not a
  researched one.
- **SSML.** The equivalent question for a trailing SSML tag is untested.
- **Multiple trailing cues.** They are applied in order, so the first takes the segment and
  the rest are reported as conflicts. Not separately motivated.
