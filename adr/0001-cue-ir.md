# ADR-0001 — Cue IR: positional-first, free-text-retaining

- **Status:** **ACCEPTED** 2026-09-03. All of §6 resolved; see §8 Decisions.
- **Date:** 2026-09-03
- **Supersedes:** the §1 table of `docs/upgrades/SYRINX_UPGRADE_expressive_control.md`,
  now superseded by `docs/backends/CONTROL_SURVEY.md` (C0.1).

## 1. Context

Syrinx drives backends whose expressive control surfaces are not merely different in
spelling but different in *kind*: open-vocabulary word-positional (Fish S2), closed-set
word-positional (Fish S1), utterance-level natural language (Qwen3-TTS
CustomVoice/VoiceDesign), utterance-level with closed inline point/span events
(CosyVoice 2/3), and none at all (Qwen3-TTS Base). See C0.1 for the verified matrix.

We want one authoring surface that is also the training transcript format for a future
Syrinx-native model.

## 2. Decision

Adopt a **positional-first** IR (`CueDoc`) in a new crate `syrinx-cue` that owns parsing
and all lowering, per spec §2.1–§2.3.

### 2.1 Why positional-first

Positional → utterance is a *total* function; utterance → positional is not. A word-scoped
cue can always be hoisted by choosing a summary rule (spec §2.3 pass 4), and the loss is
recorded in the `LoweringReport`. The reverse requires inventing timing information that
the author never supplied. Since the survey shows the two most capable backends we drive
(Fish S1/S2) are word-scoped and the future Syrinx-native model is specified as word-scoped,
choosing the weaker representation would discard information at the authoring boundary
and could never recover it.

### 2.2 Why free text is always retained

`Cue.raw` keeps the author's exact string even when the cue canonicalises. Three reasons,
each load-bearing:

1. **Fish S2 is open-vocabulary** — the card documents "15,000+ unique tags". Any closed
   canonical vocabulary is a strict subset, so canonicalising destructively would *reduce*
   S2's capability. `Free` cues must pass through byte-identically (spec §2.3 pass 2).
2. **Training format.** §2.6 makes this IR the transcript format. Discarding the author's
   wording would bake our canonical vocabulary into the training data permanently.
3. **Honest degradation.** A report saying `Dropped("no native surface")` is only auditable
   if the original text survives to be named in it.

### 2.3 Capability manifests are per **checkpoint variant**, not per family

Forced by two survey findings: Chatterbox base has no tag set while Turbo/Nano do, and
Qwen3-TTS `Base` has no instruct while `CustomVoice` does. A family-level manifest would
claim capabilities a loaded checkpoint does not have, making the `LoweringReport` wrong.

`caps.toml` additionally needs an **`honored: bool`** distinct from *accepted*: Qwen
`0.6B-CustomVoice` accepts an `instruct` argument and silently discards it (measured
2026-09-01). Modelling "accepts" alone would produce a report claiming `Hoisted` for a cue
that had no effect. This is a case where the backend lies and the manifest must not.

## 3. Consequences

- One parser, in `syrinx-cue`. Backends receive `NativeConditioning`, never raw text.
- Every backend needs a `caps.toml` (C2.1), including the deprecated CosyVoice ports and
  the no-control Qwen `Base` (whose manifest is "all-none" and whose report is all-`Dropped`).
- The hard invariant (§5) becomes a property test and a CLAUDE.md rule.

## 4. Conflicts with the existing architecture — NOT silently reconciled

Per session rules these are recorded, not resolved.

### 4.1 `ARCHITECTURE.md` does not exist and never has
The spec's opening line instructs the reader to read it; C5.1 instructs us to *update* it.
It is task **T-00.08 in `plan.md`, `status: blocked`**, explicitly because it "needs human
architecture decisions (final paradigm and contract choices) that are judgment calls the
loop must not invent." Creating it is therefore out of scope for an autonomous pass, and
C5.1 cannot be completed as written.

### 4.2 The spec's crate list is stale in both directions
Spec §0 names `syrinx-core` and `syrinx-stream`; both were **deleted** in commit `d09b11e`
("retire the deterministic spec-engine proxy"). It omits `syrinx-fish`, `syrinx-qwen` and
`syrinx-stt`, which exist. The workspace is 12 crates, not 11.

Consequence for §2.4: "**`syrinx-lm`**: for Syrinx-native and Fish-style backends, cues are
emitted into the text token stream" names the wrong crate — `syrinx-lm` is the CosyVoice
LM. Fish's LM is in `syrinx-fish`, Qwen's in `syrinx-qwen`. Similarly the §2.3 pass-6
prosody fallback points at `syrinx-acoustic`/`syrinx-vocoder`, which are CosyVoice-only and
deprecated.

### 4.3 A second bracket parser already exists and ships
`crates/syrinx-serve/src/emotion.rs` (414 lines, wired into `syrinx-cli`) already parses
`[tag]` **and** `(tag)`, segments text into spans, maps tags to zh/en instruct strings,
splits into sub-utterances and equal-power-crossfades them. That is spec passes 1 and 4 and
task C2.3, implemented, in a **backend crate** — which §2.1's "No backend crate may parse
bracket syntax itself" forbids.

This is the single largest architectural decision in this upgrade and it is not addressed
by the spec. Options: (a) migrate it into `syrinx-cue` and reduce `emotion.rs` to a thin
adapter; (b) delete it and re-implement; (c) let it stand and scope the invariant to new
code, accepting two parsers. **(a) is recommended** — it preserves the measured crossfade
behaviour while satisfying the invariant — but it changes `syrinx-serve`'s public API and
the CLI's `--emotion-tags` surface, so it needs a decision.

### 4.4 CosyVoice is deprecated but in-tree
`CLAUDE.md` states the CosyVoice ports "get no new work". C2.1 requires a `caps.toml` for
every backend. Writing manifests for CV2/CV3 is new work on a deprecated path; omitting
them makes the C2.1 gate vacuous. Needs a ruling.

## 5. The hard invariant

> **No bracket cue text and no speaker token may ever reach a backend as literal text.**

Enforced by a property test over generated inputs for every backend whose caps declare
`Inline::None` or `Inline::Closed`, asserting the lowered text contains no unescaped `[`,
`]`, or `<|speaker` sequence. To be added to `CLAUDE.md` as a hard rule. Any failure is a
release blocker.

## 6. Open questions (resolved — see §8)

1. **§4.1** — Is C5.1 descoped to `CLAUDE.md` only, or is ARCHITECTURE.md unblocked and
   authored by you?
2. **§4.3** — Migrate `emotion.rs` into `syrinx-cue` (a), delete and rewrite (b), or accept
   two parsers (c)?
3. **§4.4** — Do the deprecated CosyVoice backends get `caps.toml`?
4. ~~Ledger AC rewrites (§7)~~ — **ACCEPTED 2026-09-03**; recorded as ledger amendments
   A1 (C0.2′), A2 (C5.1a/b), A3 (ARCHITECTURE.md blocked), A4 (C3.2/C2.3 pin values).

## 7. Proposed AC rewrites

The session rule is that every AC is enforced by a test or CI gate, never by judgement.
Two ACs fail that bar as written.

| Task | AC as written | Problem | Proposed rewrite |
|---|---|---|---|
| **C0.2** | "ADR accepted per repo process." | **No ADR process exists** — no `adr/` directory, and nothing in `CLAUDE.md`, `rule.md` or `plan.md` defines acceptance. Unenforceable and self-referential. | "`adr/NNNN-*.md` exists, its `Status:` line reads `ACCEPTED`, and a test asserts every ledger task marked done references an ADR that is not `PROPOSED`." Acceptance stays a human act; the *gate* checks the recorded outcome. |
| **C5.1** | "docs reviewed; CLAUDE.md lists the invariant as a hard rule." | "docs reviewed" is unfalsifiable by machine. The second clause is greppable. | Split: **C5.1a** "a test asserts `CLAUDE.md` contains the literal invariant sentence" (enforceable now); **C5.1b** "`docs/TRAINING_DATA_FORMAT.md` exists and documents all four §2.6 stages" (structure-checkable). Drop "reviewed", or make it a `CODEOWNERS` approval, which is a CI gate rather than a claim. |

Additionally flagged, enforceable only once numbers are supplied: **C3.2** ("within
tolerance" — needs explicit ms/semitone bounds) and **C2.3** ("correct prefixes" — needs
golden fixtures to define *correct*). Both are fixable by pinning values at task start;
neither blocks acceptance of this ADR.


## 8. Decisions (2026-09-03)

Recorded here rather than in conversation, per the session doctrine.

### D1 — `emotion.rs` migrates into `syrinx-cue` (§4.3 option a)
The parser, the tag→instruct registry and the span segmentation move into `syrinx-cue`.
`crates/syrinx-serve/src/emotion.rs` is reduced to a thin adapter that keeps its current
public function signatures and delegates. `equal_power_crossfade` / `concat_crossfade`
**stay in `syrinx-serve`**: joining waveforms is synthesis, not cue parsing, and §2.1's
prohibition is specifically on parsing bracket syntax.

Consequences: the CV3 zh/en instruct vocabulary and the on-box-confirmed crossfade
behaviour are preserved rather than re-derived. `syrinx-serve`'s public API changes and the
CLI's `--emotion-tags` path is re-pointed; both need a deprecation note in the migrating
PR. The migration is a prerequisite of C2.2, not of C1.1 — the crate can be scaffolded
before the move.

### D2 — CosyVoice 2/3 ship `caps.toml` (§4.4)
A capability manifest is data describing a backend that already exists, not new work on a
deprecated model path, so it does not conflict with the `CLAUDE.md` freeze. It is also the
**only** backend we drive with native *span* forms (`<strong>…</strong>`,
`<laughter>…</laughter>`) and the richest closed inline set (7 point events), which makes it
the load-bearing fixture for the span-lowering path. Excluding it would leave that path
untested. The C2.1 gate therefore stays "every backend ships a manifest", with no
deprecation carve-out.

### D3 — C5.1's ARCHITECTURE.md clause is descoped (§4.1)
C5.1 is delivered as **C5.1a** (CLAUDE.md invariant, machine-checked) and **C5.1b**
(`TRAINING_DATA_FORMAT.md` structure). `ARCHITECTURE.md` remains `T-00.08`,
`status: blocked`, and is **not** authored by this upgrade. Ledger amendment A3 stands as
the record that the clause was consciously deferred, not dropped.

### D4 — §7 AC rewrites accepted
Recorded as ledger amendments A1 (C0.2′), A2 (C5.1a/b), A4 (C3.2/C2.3 pin values before
start).

## 9. Open conflict raised during C1.2 — bracketed prose (2026-09-03)

**Status: ACCEPTED (2026-09-03) — option (a), with the D1 qualification in §9.1.** Surfaced by the hard-invariant property test, which
fails on exactly two shapes:

```
  [He turns to the window, slowly] he said.   ->  reaches the backend verbatim
  a ] stray                                    ->  stray close bracket reaches the backend
```

### The conflict
The session invariant is *"no bracket cue text or speaker token may ever reach a backend as
literal text."* Spec §2.3 pass 8 states it more strongly: *"no bracket syntax ever reaches a
backend that would read it as literal text."*

But the parser currently preserves non-cue-shaped brackets as prose — deliberately, to match
the pre-existing `syrinx-serve::emotion` guard (`is_tag_shaped`), which ADR-0001 D1 says to
preserve so migrating scripts do not change meaning. A stage direction stays in the text.

These cannot both hold. Either unescaped brackets are always cue syntax, or brackets can
reach a backend and be spoken aloud.

Note the failure mode is not theoretical: Fish S2 would render `[He turns to the window,
slowly]` as an instruction or speak it; a knob-less backend would speak the brackets.

### Options
- **(a) Strict — every unescaped `[...]` is cue syntax.** Unrecognised content becomes a
  `Free` cue: stripped from the text, carried as `raw`, passed through on `Inline::Open`
  (Fish S2, where a stage direction reads naturally as a style instruction) and `Dropped`
  with a report elsewhere. `\[` is the documented way to get a literal bracket. Invariant
  holds absolutely. **Cost:** `array[0]` loses `0` unless escaped; diverges from
  `emotion.rs`, so D1's "no meaning change" needs qualifying.
- **(b) Prose-preserving — status quo.** Matches `emotion.rs` exactly. **Cost:** the
  invariant weakens to "no *recognised* cue markup leaks", and brackets can be spoken.
- **(c) Strip-but-never-reinterpret.** Non-cue-shaped brackets are removed and recorded as
  `Dropped("bracketed prose")`. Invariant holds; nothing is misread as a cue. **Cost:** the
  author's text is deleted even on backends that could have used it.

**Recommendation: (a).** It is what spec §2.1's escape rule implies, it keeps the invariant
absolute and machine-checkable, and it loses nothing on the one backend whose vocabulary is
open. D1 is then qualified: the *lowering* behaviour of `emotion.rs` is preserved; its
prose-bracket tolerance is not.

### 9.1 Decision (2026-09-03) — D5: strict bracket semantics

Option **(a)** is adopted. Every unescaped `[...]` is cue syntax; `\[` is the documented
escape for a literal bracket. Implemented in `crates/syrinx-cue/src/parse.rs` as:

- `is_tag_shaped(inner)` accepts any non-empty, single-line bracket content. A bracket whose
  content spans a newline is not a cue — its `[` is treated as a stray and dropped, and the
  content is preserved as prose, because an unclosed bracket earlier in a paragraph should
  not swallow the paragraph.
- Content that is not a known label becomes a `Free` cue: stripped from the clean text and
  carried in `raw` for lowering to decide.
- A non-cue bracket is **dropped**, never emitted as literal text. Delimiters are dropped and
  inner content preserved, so `[   ] spaces.` yields `"    spaces."`.
- Unmatched closers (`a ] stray`) are strays and are dropped.
- `MAX_SCAN_CHARS = 200` bounds the search for a closing delimiter, so a lone `[` cannot
  consume an arbitrarily long document.
- A `<|speaker:...|>` sequence never survives into the clean text **even when malformed**:
  well-formed with a `u32` id becomes a turn; well-formed delimiters with a bad id drops the
  whole token; a missing `|>` drops the marker prefix. This closes the one leak the property
  test found after the bracket work.

**D1 is qualified accordingly:** the *lowering* behaviour of `syrinx-serve::emotion` is
preserved on migration; its tolerance for bracketed prose is not. Scripts that relied on a
stage direction being spoken aloud must escape it as `\[...\]`. The C2.2 migration PR
carries this in its deprecation note.

**Accepted cost, recorded:** `array[0] index.` lowers to `array index.` unless escaped. This
is the documented price of an absolute, machine-checkable invariant and is pinned as a
parser fixture so it can never change silently.

## 10. Conflict raised during C1.3 — where SSML lives (2026-09-03)

**Status: ACCEPTED (2026-09-03). D6 confirmed by the maintainer; `CLAUDE.md` amended.**

`CLAUDE.md`'s crate-contract table assigns SSML to `syrinx-frontend` ("normalization, G2P,
**SSML**, lexicon, heteronyms, context windowing"), and Phase 1 lists an SSML parser as a
frontend task. C1.3 puts the SSML parser in `syrinx-cue` instead. That is a real conflict
with the stated architecture, so it is recorded here rather than reconciled in silence.

**Finding:** the frontend SSML parser was never built — `crates/syrinx-frontend/src/`
contains `feat.rs`, `lib.rs`, `speech_token.rs`, `textnorm.rs`, `tokenizer.rs` and no SSML
module. So nothing is being duplicated or displaced; the question is only where the *first*
implementation belongs.

**D6 — SSML parses in `syrinx-cue`.** ADR-0001 §2 makes `CueDoc` the single IR and this
crate its single producer. An SSML parser in `syrinx-frontend` would be a second producer
of the same IR, which means two places to enforce the hard invariant, two scoping
implementations to keep in agreement, and a dependency edge from the frontend to the cue
vocabulary. `syrinx-frontend` consumes `CueDoc` (C3.1's offset map) rather than producing
it.

**Amendment made (2026-09-03), on maintainer approval:** `CLAUDE.md` now carries a
`syrinx-cue` row naming it *the sole owner of expressive-cue syntax* (bracket cues, SSML
subset, the `CueDoc` IR, the vocabulary, `ControlCaps`, and every lowering pass); `SSML` is
removed from the `syrinx-frontend` row; and a paragraph records that the frontend
*consumes* `CueDoc` and that no backend crate may parse cue syntax. The **hard invariant**
was added to the non-negotiable rules in the same edit (the standing session instruction,
and C5.1a's AC). Both are enforced by `tests/claude_md_invariant_gate.rs`, so the doc
cannot silently regress.

**Superseded ask:** this implied a one-line amendment to the `CLAUDE.md` crate table — moving "SSML"
from the `syrinx-frontend` row to a `syrinx-cue` row — which C5.1a will make together with
the hard-invariant rule. Flagged for the maintainer to confirm or overturn; overturning it
costs the C1.3 module move and nothing else, since the parser is self-contained and the IR
would not change.

## 11. Conflict raised during C2.2 — D5 vs. the frozen `tests/emotion_tags.rs` (2026-09-03)

**Status: RESOLVED as D7. Surfaced by the D1 migration; recorded because it narrows a
claim made in §9.1 and in `CLAUDE.md`.**

### The conflict
D1 says migrate `syrinx-serve::emotion` into `syrinx-cue` "so migrating scripts do not
change meaning". The obvious implementation — reimplement `parse_tagged` on top of the
strict parser so there is genuinely ONE parser — **breaks three frozen assertions** in
`tests/emotion_tags.rs`, which is frozen under the CLAUDE.md rule *"never edit a frozen
file"*:

```
unclosed_bracket_is_literal_text_not_a_panic     "[happy hello there" -> text "[happy hello there"
unclosed_bracket_after_a_valid_tag_stays_literal "[happy] hi [sad bye" -> text "hi [sad bye"
parens_only_syntax_treats_brackets_as_literal    "[happy] hi" (Parens) -> text "[happy] hi"
```

Each of those puts a **literal bracket in front of a backend**, which is exactly what the
hard invariant and D5 forbid. So three rules collide: the frozen test may not be edited,
the invariant may not be weakened, and D1 wants one parser.

### D7 — the legacy path is frozen and quarantined, not reconciled
`syrinx_cue::legacy_emotion` is the old parser moved **verbatim**, semantics intact, so the
25 frozen tests pass unedited. It is marked deprecated in its module docs, which name the
three divergences explicitly and point here. `syrinx-serve::emotion` becomes a re-export
plus the crossfade (which is audio, not cue logic, and correctly stays).

`syrinx-cue` therefore contains **two** bracket parsers for now. That does not contradict
the D6 crate contract — the crate is still the single *owner* of cue syntax — but it does
mean "one parser" is a goal not yet reached, and pretending otherwise would be the fake
green this project exists to prevent.

### The claim this narrows — stated plainly
The hard invariant is **total for `parse` and `parse_ssml`**, which is what the property
test covers and what all new code uses. It is **not** satisfied by
`legacy_emotion::parse_tagged`, whose frozen semantics predate the rule. `CLAUDE.md` records
this as the single named exception rather than letting the rule read as universal when it
is not.

**Retiring the exception** requires unfreezing `tests/emotion_tags.rs`, which is a
maintainer decision, not a loop decision. It is cheap when CosyVoice is finally removed:
the legacy path exists only for CV2/CV3, which are already deprecated.

