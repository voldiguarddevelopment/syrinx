# Syrinx upgrade: backbone-agnostic expressive control ("cues")

You are working in the Syrinx repo (Rust workspace, ~11 crates: `syrinx-frontend`, `syrinx-core`, `syrinx-lm`, `syrinx-speaker`, `syrinx-acoustic`, `syrinx-vocoder`, `syrinx-prosody`, `syrinx-stream`, `syrinx-serve`, `syrinx-eval`, `syrinx-cli`). Read `PLAN.md`, `ARCHITECTURE.md` and `CLAUDE.md` first. This document is a spec plus task ledger; follow the doctrine there (spec-first, machine-enforced gates, append-only ledgers, worktree per task, minimal justified dependencies, AGPL/MIT-compatible only).

## 0. Goal

Add one authoring surface for emotion / style / paralinguistic / prosody control that works across *every* backend Syrinx can drive — Fish S2-style inline-tag models, Qwen3-TTS-style instruct-prefix models, Chatterbox-style scalar-knob models, CosyVoice-style hybrid models, and knob-less models (MeloTTS, Kokoro) — with deterministic, reported degradation when a backend cannot honor a cue. The same representation must later be the *training format* for Syrinx's own model, so it is a data contract, not just an API convenience.

## 1. Background you must internalize before designing

How Fish Audio S2 does it (arXiv 2603.08823): there is no control mechanism. Tags like `[whisper]`, `[laughs]`, `[in a hurry]` are ordinary text tokens placed at the word position in an interleaved text/audio token stream. The capability comes from (a) a rich-transcription ASR (fine-tuned Qwen3-Omni-30B) that writes those tags inline into training transcripts at the timestamp where the event occurs, (b) fine-grained text/audio interleaving (~10 text tokens / 20 audio tokens, 70% of sequences) that makes a tag causally local to the audio that follows, and (c) GRPO post-training where the same ASR re-transcribes the output and penalizes missed tags and wrong speaker IDs. Each tag governs the text that follows it until the next tag or sentence end; there is no blending.

How the other families differ (verify each against current docs/repos and record findings in `docs/backends/CONTROL_SURVEY.md` before writing code):

| Family | Native control surface | Granularity |
|---|---|---|
| Fish S2 / S1 | inline `[free text]` (S2) or fixed `(tag)` set (S1) | word/sub-word |
| Qwen3-TTS | natural-language instruct prefix; no positional inline tags | utterance |
| CosyVoice 2/3 | instruct prefix + small inline event/emphasis tokens | utterance + point events |
| Chatterbox | `exaggeration` / `cfg` scalars, small paralinguistic tag set | utterance (+ point events) |
| Step-Audio-EditX | rich inline tags | word |
| MeloTTS / Kokoro | speed (and speaker id) only | utterance |
| Syrinx native (future) | inline text tags, our vocabulary | word |

Design consequence: a *positional* cue language that can be *hoisted* to utterance scope, *projected* to scalars, or *dropped with a report* — never silently ignored, and never pasted as literal text into a backend that would read it aloud.

## 2. Architecture

### 2.1 New crate: `syrinx-cue`
Owns the canonical intermediate representation and all lowering. No backend crate may parse bracket syntax itself.

**Authoring syntax** (superset of Fish S2, so existing scripts work unmodified):
- `[free text]` inline anywhere. Applies to following text until next cue or sentence end (Fish semantics).
- Point events: `[laughs]`, `[sigh]`, `[breath]`, `[pause 400ms]`, `[cough]` … — zero-width, do not scope following text.
- `[emphasis]` before a word or `[emphasis]word[/emphasis]` explicit span form. Any span cue may take the explicit closing form.
- Speaker turns: `<|speaker:N|>` (Fish-compatible) and `[speaker N]` alias.
- Escapes: `\[` `\]` for literal brackets. A legacy `(tag)` mode for S1 scripts, off by default.
- Optional SSML subset (`<prosody rate pitch volume>`, `<break time>`, `<emphasis>`) parsed into the same IR; SSML and brackets may not be mixed in one input (hard error).

**IR** (`CueDoc`):
```
CueDoc { text: NormalizedText, cues: Vec<Cue>, speakers: Vec<SpeakerRef> }
Cue { span: Span,            // char range on normalized text; empty range = point event
      kind: CueKind,
      raw: String }          // ALWAYS retained: the exact free text the author wrote
CueKind =
  | Emotion { label: CanonEmotion, intensity: f32 /*0..1*/ }
  | Style   { label: CanonStyle }                 // whisper, shout, broadcast, narrator, …
  | Event   { label: CanonEvent }                 // laugh, sigh, breath, cough, gasp, …
  | Prosody { rate: Option<f32>, pitch_st: Option<f32>, volume_db: Option<f32> }
  | Emphasis { level: Level }
  | Pause   { ms: u32 }
  | SpeakerTurn { id: u32 }
  | Free    { }                                   // unrecognized; carried via `raw`
```
Canonical vocabularies live in one file (`cue/vocab.toml`): ~40–80 emotions/styles/events chosen by frequency in target use (NovaFox streaming, narration, dialogue). Each entry has: canonical id, arousal/valence scalars, synonyms, and per-backend native spellings. Parsing free text → canonical id is a synonym table plus a small rule set (intensity words: "very", "slightly", "super"). Anything unmatched becomes `Free` and still carries `raw`. No ML in the parser.

### 2.2 Backend capability manifest
Every backend implements:
```
trait ExpressiveBackend {
    fn caps(&self) -> &ControlCaps;
    fn lower(&self, doc: &CueDoc) -> (NativeConditioning, LoweringReport);
}
ControlCaps {
    inline: Inline::{None, Closed(Vec<NativeTag>), Open},   // positional tags
    prefix: Prefix::{None, NaturalLanguage, Enum(Vec<..>)},  // utterance-level instruct
    scalars: Vec<ScalarKnob { name, range, maps_from: CueDimension }>,
    events: Vec<CanonEvent>,             // point events it can render
    prosody: ProsodyCaps { rate, pitch, volume },  // native or post-process
    speakers: SpeakerCaps::{Single, MultiToken, MultiPrompt},
    granularity: Word | Sentence | Utterance,
}
```
`NativeConditioning` is an enum per backend (inline-tagged text, prefix + clean text, clean text + scalar map, …). Manifests are data (`backends/<name>/caps.toml`), loaded at build time, with a test that asserts every backend ships one.

### 2.3 Lowering passes (in `syrinx-cue::lower`), applied in order, each idempotent and unit-tested
1. **Normalize**: canonicalize labels, merge adjacent identical cues, clamp intensities, resolve explicit spans.
2. **Pass-through**: backend `Inline::Open` → re-serialize each cue as native inline text at its position, using the backend spelling table; `Free` cues pass `raw` verbatim (this is the Fish path).
3. **Vocabulary map**: `Inline::Closed` → map canonical → nearest native tag via `vocab.toml`; unmapped → next pass.
4. **Hoist**: backend has `prefix` but no inline → summarize positional cues into one utterance-level instruct string. Rule: dominant emotion by span coverage weighted by intensity; styles listed; point events dropped (or rendered by pass 6). When cues conflict inside one utterance and the backend is utterance-scoped, **split into sub-utterances at cue boundaries** and synthesize sequentially (respect sentence boundaries; never split inside a word). Splitting is opt-out via config.
5. **Scalar projection**: map arousal/valence/intensity → backend scalars (e.g. Chatterbox `exaggeration`), via per-backend affine tables in `caps.toml`. Clamp; log.
6. **Prosody fallback**: rate/pitch/volume/pause cues on backends without native support → `syrinx-prosody` plan overrides or signal-domain post-processing (time-stretch / pitch-shift in `syrinx-acoustic`/`vocoder`), clearly flagged as degraded.
7. **Event fallback**: events the backend cannot render → drop, or optionally splice a bank sample (`[breath]`, `[pause]`) — off by default.
8. **Strip**: any cue that survived to here is removed from the text. Assert: **no bracket syntax ever reaches a backend that would read it as literal text.** Property test this.

`LoweringReport` = per-cue outcome `{cue, outcome: Native | Mapped(to) | Hoisted | Projected(knob, value) | Degraded(how) | Dropped(reason)}`. Returned through the API and CLI (`--explain`). Stable, machine-readable JSON.

### 2.4 Integration points
- `syrinx-frontend`: text normalization must run **after** cue extraction so char offsets are stable; expose `NormalizedText` with an offset map to original input.
- `syrinx-lm`: for Syrinx-native and Fish-style backends, cues are emitted into the text token stream at position; implement the interleaving-aware tokenizer path so a cue lands in the text chunk immediately preceding its audio.
- `syrinx-prosody`: accept `Prosody`/`Pause`/`Emphasis` cues as plan overrides.
- `syrinx-serve`: `/v1/audio/speech` accepts cues in `input`; add optional `cues: CueDoc` JSON field for clients that pre-parse; return `X-Syrinx-Lowering` header (summary) and full report on `/v1/audio/speech?explain=1`. Add `GET /v1/backends/{name}/caps`.
- `syrinx-cli`: `syrinx cue parse`, `syrinx cue lower --backend X --explain`.

### 2.5 Verification: tag-activation harness in `syrinx-eval`
Mirror Fish's reward design; this doubles as the future RL reward.
- ASR (Whisper-class, run through our own inference path or ONNX, no PyTorch) → WER on the clean text.
- Paralinguistic event detector (laugh/breath/sigh/cough) + timestamps.
- Segment-level emotion classifier (emotion2vec-class), scored on the cue's span.
- Speaker similarity (ECAPA/WavLM-class).
- Metric: **tag activation rate** = fraction of cues whose expected signal is detected within the cue span (± tolerance), reported per cue kind and per backend. Plus WER delta vs. uncued synthesis (cues must not cost intelligibility).
- Frozen cue eval set: ~200 scripted lines covering every canonical label, several intensities, multi-speaker, and adversarial cases (bracket in literal text, conflicting cues, cue at sentence end). Checksummed; gate on it.

### 2.6 Training-side contract (design now, execute in a later phase)
The authoring syntax **is** the transcript format for Syrinx-native training data. Specify in `docs/TRAINING_DATA_FORMAT.md`:
- Annotator pipeline stages: separation+VAD → quality filter → rich transcription with inline cues at word position (bootstrapped from an open audio-understanding model + forced aligner + event detector, human-checked sample for precision).
- Cues stored as text, at position; speaker turns as `<|speaker:N|>`; reference audio prefix loss-masked.
- Text/audio interleaving parameters (chunk sizes, probability).
- Reward spec = §2.5 harness, with a heavy penalty term for missed cues and wrong speaker IDs; GRPO-style group advantages without std normalization.
Do not implement training in this upgrade; make sure nothing in the IR prevents it.

## 3. Non-goals for this upgrade
- No new model weights. No Python at inference. No LLM-in-the-loop cue parsing.
- No attempt at cross-tag blending; Fish boundary semantics are the spec.
- No open-vocabulary emotion on backends that lack it — degrade honestly instead.

## 4. Task ledger (append-only; one worktree per task; sizes S/M/L)

- [ ] **C0.1** `docs/backends/CONTROL_SURVEY.md`: verify §1 table against current upstream repos/docs for every backend Syrinx targets; record native syntax, scope, and quirks with links. **AC:** file committed; each row has a source link and a date. **S**
- [ ] **C0.2** ADR `adr/NNNN-cue-ir.md`: IR design, why positional-first, why free text is retained. **AC:** ADR accepted per repo process. **S**
- [ ] **C1.1** Crate `syrinx-cue` scaffold + `vocab.toml` (≥40 canonical labels with synonyms, arousal/valence, per-backend spellings). **AC:** `cargo test` green; vocab schema validated by a test. **M**
- [ ] **C1.2** Parser: bracket syntax, point vs span, explicit close, speaker tokens, escapes, S1 legacy mode; offset-stable with normalization. **AC:** ≥50 parser fixtures incl. adversarial; property test: `serialize(parse(x))` round-trips for the Fish path. **M**
- [ ] **C1.3** SSML subset parser into the same IR; mixed-syntax hard error. **AC:** fixtures; error path tested. **S**
- [ ] **C2.1** `ControlCaps` + `ExpressiveBackend` trait; `caps.toml` for every existing backend; build-time test that no backend lacks one. **AC:** test fails if a backend is added without caps. **S**
- [ ] **C2.2** Lowering passes 1–3 (normalize, pass-through, vocab map) + `LoweringReport`. **AC:** unit tests per pass; Fish-style backend receives byte-identical tags for `Free` cues. **M**
- [ ] **C2.3** Hoist pass with sub-utterance splitting. **AC:** conflicting cues on an utterance-scoped backend produce N sequential segments with correct prefixes; no split inside a word; opt-out honored. **M**
- [ ] **C2.4** Scalar projection + prosody fallback + event fallback + strip pass. **AC:** property test: output text for any backend with `Inline::None` contains no `[`/`]`/`<|speaker` sequences unless escaped in source. **M**
- [ ] **C3.1** Wire into `syrinx-frontend` (offset map) and `syrinx-lm` (position-correct emission in interleaved stream). **AC:** golden token dumps for 5 fixtures; cue token index == first token of its span. **M**
- [ ] **C3.2** `syrinx-prosody` override path for Prosody/Pause/Emphasis cues. **AC:** measured duration/pitch deltas on frozen fixtures within tolerance. **M**
- [ ] **C3.3** `syrinx-serve` + `syrinx-cli` surfaces (§2.4). **AC:** OpenAI-compatible request with inline cues works unchanged; `?explain=1` returns full report; caps endpoint documented in OpenAPI. **M**
- [ ] **C4.1** `syrinx-eval` activation harness (§2.5) with ONNX/own-runtime models only. **AC:** emits JSON with per-kind/per-backend activation rate and WER delta; runs in CI on the frozen cue set. **L**
- [ ] **C4.2** Frozen cue eval set + gate thresholds (start: activation ≥ 0.85 on `Inline::Open` backends for events; WER delta ≤ +0.5 abs everywhere). **AC:** set checksummed; CI blocks on regression. **M**
- [ ] **C5.1** `docs/TRAINING_DATA_FORMAT.md` (§2.6) + `ARCHITECTURE.md`/`CLAUDE.md` updates. **AC:** docs reviewed; CLAUDE.md lists the "no literal bracket leakage" invariant as a hard rule. **S**

## 5. Definition of done
Same script with `[whispering]`, `[laughs]`, `[excited]`, a `[pause 500ms]`, and two `<|speaker:N|>` turns synthesizes on every configured backend with (a) no cue text audible, (b) a lowering report that explains every cue, (c) activation metrics in CI, and (d) an IR that can be written straight into a training manifest for the Syrinx-native model.

## 6. Ledger amendments (append-only)

Entries above are never edited. Amendments supersede by reference and record why.

### A1 — 2026-09-03 — supersedes the AC of **C0.2**
Accepted per ADR-0001 §7. Original AC ("ADR accepted per repo process") was not
machine-enforceable: no `adr/` directory existed and no acceptance process is defined in
`CLAUDE.md`, `rule.md` or `plan.md`.

- [ ] **C0.2′** ADR `adr/0001-cue-ir.md`. **AC:** the file exists; its `Status:` line reads
  `ACCEPTED`; and a test asserts that no ledger task may be marked done while the ADR it
  references is still `PROPOSED`. Acceptance remains a human act — the gate checks only the
  recorded outcome. **S**

### A2 — 2026-09-03 — supersedes the AC of **C5.1**
Accepted per ADR-0001 §7. "Docs reviewed" is unfalsifiable by machine; the ARCHITECTURE.md
clause is additionally blocked (see A3).

- [ ] **C5.1a** **AC:** a test asserts `CLAUDE.md` contains the literal hard-invariant
  sentence from ADR-0001 §5, byte-for-byte. **S**
- [ ] **C5.1b** **AC:** `docs/TRAINING_DATA_FORMAT.md` exists and contains a section for each
  of the four §2.6 stages (annotator pipeline, cue storage, interleaving parameters, reward
  spec); a test asserts all four headings are present. **S**

### A3 — 2026-09-03 — **C5.1's `ARCHITECTURE.md` clause is BLOCKED, not descoped**
`ARCHITECTURE.md` does not exist and never has. It is `T-00.08` in `plan.md`,
`status: blocked`, because it "needs human architecture decisions (final paradigm and
contract choices) that are judgment calls the loop must not invent." Recorded here rather
than silently dropped. Awaiting a ruling (ADR-0001 §6 Q1).

### A4 — 2026-09-03 — **C3.2 and C2.3 need values pinned before they are startable**
Both ACs are enforceable in principle but under-specified: C3.2 says "within tolerance"
without bounds, C2.3 says "correct prefixes" without a definition of correct. Neither
blocks ADR-0001. Each task must pin its numbers/golden fixtures in its own PR description
before implementation begins, and the pinned values become part of that task's gate.

### A5 — 2026-09-03 — ADR-0001 accepted; scope changes it implies
Per ADR-0001 §8:
- **D1** adds a prerequisite to **C2.2**: migrate `syrinx-serve::emotion` into `syrinx-cue`
  (parser + registry + segmentation), leaving a delegating adapter and keeping the
  crossfade in `syrinx-serve`. Deprecation note required in that PR.
- **D2** confirms **C2.1** covers CosyVoice 2/3 with no deprecation carve-out.
- **D3** replaces **C5.1** with C5.1a/C5.1b only; the ARCHITECTURE.md clause is deferred
  under A3 and is not part of this upgrade.

### A6 — 2026-09-03 — process relaxation
Worktree-per-task and PR-per-task are suspended at the maintainer's direction; work lands
directly on the working tree. The gates that protect correctness are NOT relaxed:
machine-enforced ACs, append-only ledger, ADR-recorded decisions, Rust-only at inference,
and license justification for every new dependency all still apply.

### A7 — 2026-09-03 — **C1.1 COMPLETE**
`crates/syrinx-cue` scaffolded and registered in the workspace; `vocab.toml` carries **54**
canonical labels (26 emotion / 14 style / 14 event), derived from the C0.1 verified native
sets rather than invented.
**AC met:** `cargo test -p syrinx-cue` green (8 tests). Schema validated by test, including
id/synonym uniqueness, scalar ranges, mandatory valence on emotions, and cross-checks that
every `fish_s1` spelling is inside the documented 65-tag closed set and every `cosyvoice`
spelling inside its 7-token closed set.
**New dependency:** `toml` 0.8 — MIT OR Apache-2.0 — required because ADR-0001 specifies the
vocabulary as data; pulls only `serde`, already in the graph.

### A8 — 2026-09-03 — **C1.2 COMPLETE** (bracket/tag parser → Cue IR)
`crates/syrinx-cue` parses bracket cues, parenthetical cues and speaker turns into the
`CueDoc` IR of ADR-0001 §3. Spans are byte offsets into the **clean** text.
**Decision applied:** ADR-0001 §9 is resolved as **D5 / option (a) — strict bracket
semantics**; see adr/0001-cue-ir.md §9.1 for the rule set and the accepted `array[0]` cost.
**AC met:** `cargo test -p syrinx-cue` green — 8 vocab + 7 fixture + 3 property tests.
- 50 table-driven parser fixtures pin clean text and cue count for every documented shape,
  including the escape `\[`, unmatched delimiters, whitespace-only brackets, newline-spanning
  brackets and malformed speaker tokens.
- **Hard invariant is now machine-enforced**, not asserted: `invariant_property.rs` generates
  2,000 inputs from 30 adversarial fragments with a deterministic xorshift PRNG and asserts
  that no unescaped `[`, `]` or `<|speaker` survives into the clean text, over both the
  generated corpus and the full fixture corpus. A failure of this test is a release blocker.
- `fish_path_round_trips` checks that escaped brackets survive the round trip intact.
**No new dependency.**

### A9 — 2026-09-03 — **C1.3 COMPLETE** (SSML subset → the same IR)
`crates/syrinx-cue/src/ssml.rs` parses the spec §2.2 subset — `<speak>`, `<prosody rate
pitch volume>`, `<emphasis level>`, `<break time|strength>` — into the **same** `CueDoc`,
so nothing downstream can tell which syntax an author used.
**AC met:** `cargo test -p syrinx-cue` green — **31 tests** (8 vocab + 7 bracket fixtures +
11 SSML fixtures + 5 property).
- **Fixtures:** 16 table-driven source→(clean text, cue count) cases plus span/offset,
  point-event and scalar-axis tests. Every attribute axis is pinned at both ends
  (`rate` keyword/percent/multiplier, `pitch` semitone/keyword/percent, `volume`,
  `strength` from `none` to `x-strong`, `emphasis` either side of moderate), and the
  percent→semitone conversion is checked against the octave relation.
- **Error path tested (the AC's explicit demand):** mixed syntax in both orders,
  unsupported tag, unclosed tag, mismatched close, close-with-nothing-open, malformed tag,
  and a bad value on all six attributes — plus the boundary that `rate="0"` and
  `rate="-1"` are not rates.
- **Mixed-syntax hard error** is property-tested, not just fixture-tested: 800 crossed
  documents from both generators, every one carrying both dialects must be rejected. An
  escaped `\[` is correctly NOT a mix.
- **Hard invariant extended to the SSML dialect:** 4,000 generated documents assert no tag
  survives into the clean text, with the generator forced to exercise both the valid
  (>100) and rejected (>100) arms so the test cannot pass vacuously.
- **Conflict recorded, not reconciled:** `CLAUDE.md` assigns SSML to `syrinx-frontend`.
  See ADR-0001 §10 / **D6** — the frontend parser was never built, and a second `CueDoc`
  producer would mean two places to enforce the hard invariant. Flagged for maintainer
  confirmation; implies a one-line `CLAUDE.md` table amendment, folded into C5.1a.
**No new dependency** — the scanner is hand-rolled (no `quick-xml`/`roxmltree`), which also
keeps the inference path Rust-only with a zero license surface.

### A10 — 2026-09-03 — **C2.1 COMPLETE** (`ControlCaps` + `ExpressiveBackend` + caps table)
`crates/syrinx-cue/src/caps.rs` + `caps.toml`: **9 rows, one per checkpoint variant** —
fish-s1-mini, fish-s2-pro, qwen3-{0.6b,1.7b}-base, qwen3-{0.6b,1.7b}-customvoice,
qwen3-1.7b-voicedesign, cosyvoice2, cosyvoice3 (CosyVoice included per ADR-0001 **D2**).
**AC met — "test fails if a backend is added without caps" — and PROVEN by negative
control, not merely asserted.** Three escape routes are each closed and each was verified
to bite by temporarily breaking the tree:
1. `BackendId` variant with no `caps.toml` row → *"backend `new-shiny-tts` has no caps.toml
   entry"*.
2. variant added to the enum but forgotten in `ALL` (which would make check 1 vacuous) →
   *"BackendId has 10 variants but ALL lists 9"*. Counted from the source text.
3. a whole backend **crate** added without touching `BackendId` — the failure that actually
   happens, and which neither 1 nor 2 catches → *"crate `syrinx-newtts` is neither listed
   as a non-backend nor covered by a caps.toml row"*.
All three restored to green afterwards. `cargo test -p syrinx-cue` = **39 tests**.
**`accepted` vs `honored` is load-bearing, and the on-box evidence is stronger than the
spec's:** the split is by **checkpoint size, not variant**. `prompt.rs::honors_instruct`
gates on a substring test for the 1.7B size, so **0.6B-CustomVoice accepts an `instruct`
string and silently discards it** while 1.7B-CustomVoice obeys it — identical API, opposite
truth. `Support::Accepted.is_effective() == false`, so it lowers exactly like `Unsupported`
and is always reported; a test pins both checkpoints against each other so the distinction
cannot quietly collapse.
**No new dependency** (`toml` + `serde` already present from C1.1).

### A11 — 2026-09-03 — **D6 accepted; `CLAUDE.md` amended** (part of C5.1a, landed early)
Maintainer approved ADR-0001 §10 / **D6**. `CLAUDE.md` amended:
- crate table gains **`syrinx-cue` — the sole owner of expressive-cue syntax** (bracket
  cues, SSML subset, `CueDoc` IR, vocabulary, `ControlCaps`, every lowering pass);
- `SSML` removed from the `syrinx-frontend` row;
- a paragraph records why: one IR, one producer — a second producer would mean two places
  to enforce the hard invariant and two scoping implementations to keep in agreement. The
  frontend *consumes* `CueDoc`; no backend crate may parse cue syntax.
- the **hard invariant** is now a numbered non-negotiable rule, naming speaker tokens and
  SSML tags as well as brackets, pointing at `invariant_property.rs` as its enforcement,
  and recording the D5 corollary (`array[0]` needs escaping).
**Both amendments are machine-gated**, per the session rule that no AC rests on judgement:
`tests/claude_md_invariant_gate.rs` (2 tests) fails if the invariant leaves the rules
section, if it stops naming its scope/enforcement, if `syrinx-cue` leaves the crate table,
or if the frontend row re-claims SSML.
This discharges the `CLAUDE.md` half of **C5.1a** ahead of its ledger position; the
`docs/TRAINING_DATA_FORMAT.md` half remains open.

### A12 — 2026-09-03 — **C2.2 COMPLETE** (lowering passes 1–3 + `LoweringReport`, incl. the D1 migration)
`crates/syrinx-cue/src/lower.rs`. **AC met:** unit tests per pass, and the headline
criterion — *a Fish-style backend receives byte-identical tags for `Free` cues* — is pinned
over five awkward shapes (odd spacing, mixed case, long free text): what Fish S2 receives is
`cue.raw`, the author's exact bytes. `cargo test -p syrinx-cue` = **54 tests**.
- **Pass 1 normalize** resolves synonyms/casing to canonical ids. Testing it exposed that
  **the parser already canonicalises**, so pass 1 is the defensive net for IR that did not
  come from the parser (API callers, the SSML path). Both facts are now pinned so the two
  stages cannot silently swap responsibility — the original test asserted the wrong stage.
- **Pass 2 pass-through** (open vocabulary) and **pass 3 vocab map** (closed vocabulary,
  via the `fish_s1`/`fish_s2`/`cosyvoice` columns) run as one walk over the cue list.
- **`LoweringReport`**: nothing is ever dropped silently — a test asserts every cue that
  enters lowering leaves a report line, on **all nine backends**. `Dropped` carries a
  reason, and `AcceptedButIgnored` is distinct from `Unsupported`: the 0.6B-CustomVoice and
  0.6B-Base cases produce different explanations for what a listener hears identically.
- Passes 1–3 provably **do not touch the clean text** (text rewriting is C2.3/C2.4), so
  spans stay valid; and the hard invariant is re-asserted after lowering on every backend.

### A13 — 2026-09-03 — **D1 migration done; conflict recorded as D7**
`syrinx-serve::emotion` → `syrinx_cue::legacy_emotion` (parser + registry + segmentation),
leaving a re-export adapter; `equal_power_crossfade`/`concat_crossfade` stay in
`syrinx-serve` as audio, not cue logic. `syrinx-serve` gains a `syrinx-cue` path dependency.
**The 25 frozen tests in `tests/emotion_tags.rs` pass unedited.**
**Conflict found and NOT silently reconciled (ADR-0001 §11 / D7):** the clean
implementation — reimplement `parse_tagged` on the strict parser, giving genuinely one
parser — **breaks three frozen assertions**, because they require a literal `[` to reach
the backend (`"[happy hello there"`, `"hi [sad bye"`, and `[happy] hi` under `Parens`).
Three rules collide: the frozen test may not be edited, the invariant may not be weakened,
D1 wants one parser. Resolution: the legacy parser is migrated **verbatim** and
quarantined as deprecated, so `syrinx-cue` holds two bracket parsers for now — "one parser"
is a goal not yet reached, and claiming otherwise would be a fake green.
**This narrows a claim made in A8 and in `CLAUDE.md`, so it is stated rather than glossed:**
the hard invariant is total for `parse`/`parse_ssml` (all new code) and is **not** satisfied
by `legacy_emotion::parse_tagged`. `CLAUDE.md` now records that as the single named
exception, and `tests/claude_md_invariant_gate.rs` fails if the exception loses its name,
its ADR citation, or stops being singular. Retiring it means unfreezing
`tests/emotion_tags.rs` — a maintainer decision — and is cheap once CosyVoice is removed.

### A14 — 2026-09-03 — **C2.3 AC pinned before implementation** (discharges A4 for C2.3)
A4 recorded that C2.3's *"N sequential segments with correct prefixes"* is unenforceable
until "correct prefix" is defined. Pinned now, before any code, so the gate is a fact and
not an opinion:

**A segment's prefix is its `instruct` string, derived from the single cue in effect for
that segment by this total function, in order:**
1. `CueKind::Free` → the author's `raw`, **verbatim**. Free text is already a
   natural-language instruction, which is exactly what an utterance-scoped backend wants.
2. A vocabulary label with a legacy instruct phrase → that phrase, in the requested
   `InstructLang` (`Zh` is CosyVoice's default; `En` for Qwen).
3. A vocabulary label with no phrase → `"Speak in a {label} tone"` (en) /
   `"用{label}的语气说"` (zh).
4. No cue in effect → `None`; the segment is spoken plainly with no instruct.

**Segmentation rules, all machine-checked:**
- Split points are the start offsets of span-scoping cues that *conflict* — two cues
  conflict when both are effective on that backend and their labels differ.
- **No split inside a word:** a split offset is snapped to the nearest preceding
  whitespace; a split that would land mid-token moves left to the token boundary.
- Point events (zero-width) never split; they attach to the segment containing them.
- **Opt-out** (`SplitOptions::allow_split = false`): exactly one segment is emitted, using
  the first effective cue; every other cue is `Dropped` with a reason in the report.
- Concatenating every segment's text reproduces the clean text exactly (modulo the
  boundary whitespace consumed by the snap) — property-tested, so splitting can never
  invent, drop, or reorder words.

### A15 — 2026-09-03 — **C2.3 COMPLETE** (hoist pass + sub-utterance splitting)
`crates/syrinx-cue/src/hoist.rs`, to the definition pinned in **A14**. 12 tests.
**AC met:** `[happy] good morning [sad] but not for long` on Qwen 1.7B-CustomVoice yields
**2** sequential segments with prefixes *"Speak in a happy, cheerful tone"* / *"Speak in a
sad, sorrowful tone"*; a three-way conflict yields 3 in order; **no split lands inside a
word** (offsets snap left to a token boundary, verified over mid-token cues like
`some[sad]thing`); **opt-out honored** — one segment, first cue wins, and the losing cues
are `Dropped` in the report rather than ignored.
- Identical consecutive cues do **not** split — that would cost a synthesis pass and a join
  artefact for no expressive gain.
- Word-granular backends (Fish) are never split, and get no prefix: they steer inline.
- The 0.6B-CustomVoice is not split and gets no instruction, because it would ignore one.
- **Bug caught by the round-trip property, not by review:** a cue snapping left past a
  space carved off a whitespace-only sliver which was then dropped, silently deleting a
  space from the utterance. Slivers are now carried onto the next segment. The property
  *"concatenating every segment reproduces the clean text exactly"* holds over 6 shapes.
- A test expectation of mine was wrong and was corrected rather than the code: `[laughs]`
  on Qwen is genuinely **dropped** (Qwen has no event channel), so the honest assertion is
  that it does not split *and* appears in the report; a companion test shows the point cue
  surviving on CosyVoice, which does have an inline event token.

### A16 — 2026-09-03 — **C2.4 COMPLETE** (scalar projection + fallbacks + strip)
`pass_project_fallbacks` / `pass_strip` / `project_prosody` / `project_emphasis` in
`lower.rs`. 10 tests.
**AC met — the property test:** for every `Inline::None` backend, over **800 seeds x 4
lengths x 3 backends (>5,000 checks)** of adversarially generated input (stray brackets,
malformed speaker tokens, escapes, CJK, emoji), the output text contains no `[`, `]` or
`<|speaker` **unless escaped in the source**.
- **Projection thresholds are pinned on both sides of every boundary** (rate 0.9/1.1, pitch
  ±1 st, volume ±3 dB, inclusive), so "roughly slower" is never the spec and an operator
  mutation cannot survive.
- A prosody cue on an instructable backend becomes a phrase (*"Speak slowly"*) instead of
  being thrown away; on a backend with no instruction channel it is dropped **with a
  reason**; on the 0.6B it is dropped rather than projected, because projecting into a
  channel that is `accepted` but not `honored` would be theatre.
- A projection that says nothing (rate 1.0) is a drop, never an empty instruction.
- `pass_strip` is **deliberately not idempotent** — it restores `\[` to `[`, so a second
  run would strip what the first produced. That ordering constraint is pinned by test so
  the next pass author sees it.

### A17 — 2026-09-03 — **C3.1 COMPLETE**, and **C3.2 AC rewrite proposed** (discharges A4 for C3.2)

**C3.1 COMPLETE.** `crates/syrinx-cue/src/offsets.rs` + `tests/cue_token_alignment.rs`
(repo-root, per the CLAUDE.md convention) with goldens under `tests/golden/cue_tokens/`.
**AC met:** golden token dumps for exactly **5** fixtures (leading cue, mid-sentence cue,
point event, explicit span, multi-speaker turn), and *cue token index == first token of its
span* asserted directly as well as implied by the dumps. 6 tests.
- The module is **tokenizer-agnostic**: it consumes the byte spans a tokenizer produced, so
  the alignment arithmetic is gateable without model weights while the same code serves the
  real `syrinx-frontend` tokenizer at run time. Goldens use a deterministic whitespace
  tokenizer.
- The goldens were **read, not just generated**: `[angry]` fires immediately before `"the"`
  (the first token of its span) and the two speaker turns fire before `"hello"` and
  `"and"`. An unreviewed golden is a rubber stamp, not a gate.
- `interleave` emits a cue **before** the token it governs, and a property pins that the
  token stream is never dropped, duplicated or reordered.

**C3.2 AC REWRITE — proposed, and implemented against, per the session rule that an AC
which cannot be machine-enforced as written must be rewritten first.**
As written — *"measured duration/pitch deltas on frozen fixtures within tolerance"* — the
AC implies measuring rendered **audio**, which needs the acoustic model + GPU and is
therefore blocked-on-human, not loop-gateable.
**Rewritten to:** the deltas are measured on `syrinx_prosody::RenderPlan` over a frozen
synthetic mel, which is fully deterministic and needs no model:
- **duration:** `RenderPlan::apply` time-warps the frame axis, so `T_out / T_in` must equal
  `1/rate` within **±1 frame** (the warp is integral in frames);
- **pitch:** the returned `f0_mult` must equal `2^(semitones/12)` within **1e-6**;
- both on frozen fixtures, for global and per-region cues.
**Scope stated honestly:** this measures the *plan's* effect, which is the deterministic
half. Whether the acoustic model then follows the plan is a perceptual/GPU question and
stays blocked — this AC does not claim otherwise.

### A18 — 2026-09-03 — **C3.2 COMPLETE** (prosody override path)
`crates/syrinx-prosody/src/cues.rs` — `plan_from_cues` turns `Prosody`/`Emphasis` cues into
`RenderPlan` global knobs and `Region`s. `syrinx-prosody` gains a `syrinx-cue` dependency;
the direction is one-way (D6) and nothing here parses cue syntax.
**AC met, to the A17 rewrite** — `tests/prosody_cue_overrides.rs`, 9 tests, all *measured*
via `RenderPlan::apply` on a frozen synthetic mel:
- **duration:** output/input frame ratio equals `1/rate` within ±1 frame, across
  rate ∈ {0.5, 0.75, 0.9, 1.0, 1.25, 2.0};
- **pitch:** the returned `f0_mult` equals `2^(st/12)` within 1e-6, across ±12 st;
- a narrow cue becomes a region and the global knob stays untouched, with both shifted and
  unshifted frames asserted present so the test cannot pass on a plan that did nothing;
- emphasis deltas are pinned *and ordered* (strong slows/lifts more than moderate), so a
  sign flip cannot survive;
- emotion/style/event/pause/speaker cues leave the plan at identity — they steer the model,
  and producing regions here would double-apply against the backend's own control.

### A19 — 2026-09-03 — **C3.3 COMPLETE** (`syrinx-serve` + `syrinx-cli` surfaces)
**AC met**, `tests/expressive_api.rs` (9 tests) + the CLI:
- **"OpenAI-compatible request with inline cues works unchanged"** — six inputs (plain,
  bracket cues, multi-cue, point event, speaker turns, SSML) all still return `200` +
  `audio/*`, and the existing 400/422 error contract is re-asserted untouched.
- **`?explain=1` returns the full report** as JSON instead of audio: backend, the exact
  text the backend receives (asserted free of cue markup — the hard invariant over the
  wire), and one entry per cue with `raw`, source range and action. `?explain=0` and a
  stray query param must NOT swallow the audio, which is also pinned.
- **Caps endpoint** `GET /v1/audio/caps` returns all 9 rows, and the accepted-vs-honored
  split is asserted to survive to the wire (0.6B `accepted`, 1.7B `honored`).
- **`docs/api/openapi.yaml`** documents both, including the `Support` enum with the
  `accepted` trap spelled out ("taken without error but NOT acted on"). A gate test scans
  the router's registered routes out of the source and fails if any endpoint is
  undocumented — so a new route cannot ship without OpenAPI.
- **CLI:** `syrinx cue --text <T> [--backend <ID>] [--explain]` prints the same report
  without synthesizing. Mixed syntax exits non-zero with the parser's message; an unknown
  backend lists the valid ids.

### A20 — 2026-09-03 — bug found by using the CLI, not by a test
Running `syrinx cue` on real input showed an SSML `<prosody rate="slow">` reported as
**DROPPED** on `qwen3-1.7b-customvoice` — a backend that plainly *can* take it as a
natural-language instruction. Two defects, both fixed:
1. `ControlCaps::support_for` treated `CueKind::Free` as expressible only on an **open
   inline vocabulary**, ignoring the instruction channel entirely. Free text now resolves
   to `Honored` when `instruct` is honored, and to `Accepted` when instruct is merely
   accepted — so the 0.6B still correctly drops it.
2. `lower` stops after pass 3 by design (that is what the per-pass tests pin), so no caller
   was running C2.4's projection. Added **`lower_full`** = passes 1–3 + projection +
   strip, with projection ordered *before* the inline decision so a projected instruction
   still gets its chance to reach an instructable backend. The CLI and the server's
   `?explain=1` now use it.
Worth recording as process: the unit tests were green throughout. Exercising the surface by
hand is what surfaced it.

### A21 — 2026-09-03 — **C4.1 + C4.2 COMPLETE** (activation harness, frozen set, gate)
`crates/syrinx-eval/src/activation.rs` + `tests/golden/cue_eval/` +
`tests/cue_activation_gate.rs` (9 tests).
**C4.2 frozen set:** 54 cases — emotion/style/event x {en, de, pl} x
{leading, mid, trailing} plus a **neutral control group** (no cues) that gives WER delta a
baseline. SHA-256 checksummed in `cue_set.sha256`; a mismatch fails loudly and says that
historical numbers are not comparable across sets. A test also parses every case through
`syrinx-cue` and asserts each cued case yields exactly one cue and leaks no markup — a
case whose cue silently failed to parse would score as a permanent non-activation.
**C4.1 harness AC met:** emits JSON with per-kind / per-backend `activation_rate` and
`wer_delta`, plus violations and a `passed` flag.
**C4.2 gate AC met:** blocks event activation below **0.85** on `Inline::Open` backends and
any WER delta above **+0.5** absolute, with the boundary pinned (a delta *exactly* at the
limit passes). Event activation is gated **only** where events can be expressed — gating a
backend with no event channel would be a permanent meaningless red.
**Honest scope, stated in the module docs and pinned by test:** the harness computes and
gates; it does **not** synthesize. Activation requires running the models on a GPU, so the
numbers are only as real as the run behind them. `an_empty_run_is_reported_as_empty_not_as_a_pass_with_no_data`
exists so a run that never happened cannot masquerade as a clean one in CI.

### A22 — 2026-09-03 — **C5.1a + C5.1b COMPLETE**
**C5.1a AC met literally:** the gate no longer checks a paraphrase — it extracts the
invariant sentence *from ADR-0001 §5 itself* (`include_str!` + parse of the bold
blockquote) and asserts `CLAUDE.md` contains it **byte-for-byte**, so the two cannot drift.
`CLAUDE.md`'s rule was reworded to carry the ADR sentence verbatim.
**C5.1b AC met:** `docs/TRAINING_DATA_FORMAT.md` written, covering all four §2.6 stages —
annotator pipeline (separation+VAD → quality filter → forced-aligned rich transcription →
human-checked precision sample), cue storage (text at position, `<|speaker:N|>`,
loss-masked reference prefix, escaping, the invariant applied to corpora), interleaving
parameters (table of knobs; cue emission position fixed to *before the first token of its
span*, matching `interleave` and the C3.1 goldens), and the reward spec (activation minus
WER delta, heavy penalties for missed cues and wrong speaker ids, GRPO-style group
advantages **without std normalization**). The gate checks the four headings **and** eight
specific required contents, so a heading with nothing under it cannot pass.
§5 of the doc cross-checks the shipped IR against every training requirement; nothing in
the IR blocks the design, and the one deliberate omission (per-cue timestamps) is recorded
as additive rather than a redesign.

### A23 — 2026-09-03 — **ledger complete: C1.1 … C5.1b all built**
Every task in §4 is implemented and gated. Remaining open items are **not** loop work:
- **ADR-0001 §6 Q1 / A3** — `ARCHITECTURE.md` does not exist (`T-00.08`, blocked on human
  architecture decisions). C5.1's ARCHITECTURE clause stays blocked, not descoped.
- **D7 / ADR-0001 §11** — retiring the legacy-parser exception needs `tests/emotion_tags.rs`
  unfrozen, which is a maintainer decision; cheap once CosyVoice is removed.
- **C4.2 thresholds are un-certified** — the gate logic is green, but no GPU run has
  produced real activation measurements for the frozen set. That run is the next thing
  worth doing on the box, and until it happens no claim is made about real activation rates.

### A24 — 2026-09-03 — on-box verification, and a **false SKIP** found in the runner
`./scripts/verify.sh` → **VERIFIED, exit 0** (PASS 9 · SKIP 34 · MISSING 0 · **FAIL 0**).
The 34 SKIPs are the intended state: CosyVoice groups are deliberately unconfigured (model
direction: Fish only), and the Fish parity fixtures need the one `# TODO(on-box)` in
`gen-fish-ref.py`.

**Defect found in the harness, pre-existing and not caused by this upgrade:** `emotion_tags`
reported **SKIP** on every board while actually passing 25/25. `run_one` classified a test
as SKIP whenever its log matched `grep -qiE 'skip|skipping'` — case-insensitively, on any
substring — and the frozen `tests/emotion_tags.rs` contains a *test name*,
`concat_crossfade_skips_empty_segments_and_handles_none`, which matched. A green test was
therefore hidden behind a SKIP.
Fixed in **both** `scripts/test-all.sh` and `scripts/verify.sh` by matching the two actual
self-skip conventions case-sensitively — `SKIP <name>: …` and `skipping <name>: …` (the
trailing space is load-bearing). Audited every skip message in `tests/` first to confirm
those are the only two forms. This **tightens** the classifier rather than weakening it: a
passing test can no longer hide as SKIP, and FAIL detection is by exit code and untouched.
`emotion_tags` now reports PASS; the model-free group is **8/8 PASS, 0 SKIP**.

**Coverage gap closed:** the six repo-root gates added by this upgrade were in no group, so
the sanctioned board did not exercise them at all. Added **`GROUP_cue`** (control_survey_gate,
claude_md_invariant_gate, cue_token_alignment, prosody_cue_overrides, expressive_api,
cue_activation_gate) to both scripts, registered in `ALL_GROUPS` and included in `--quick`,
since all six are model-free and must never SKIP. Group result: **6/6 PASS**.


### A25 — 2026-09-03 — **C4.2 measurement built; the certification run is STARTED-NOT-FINISHED**
A23 recorded that C4.2's thresholds were uncertified because nothing could produce a real
`Measurement` — the only caller was the frozen gate test, with synthetic values. That gap is
now closed in code, but **the certification run itself has not completed and no activation
number is claimed.**

**Landed (committed, green):**
- `syrinx_eval::acoustic` — an 11-dimension acoustic summary plus an EXACT two-sample
  permutation test. Defines `activated` as "moved beyond this model's own run-to-run
  variation", measured from `n` renders per condition at distinct `DriveParams::seed`s.
  Both naive readings are useless and were rejected: one-vs-one comparison and fixed-seed
  bit-comparison each answer `true` unconditionally, so either would report ~100 %
  activation for a backend that ignores cues entirely. `n >= 4` is enforced —
  `C(8,4)/2 = 35` labelings put the smallest attainable p at 1/35, and a smaller `n`
  cannot reject at all, which would look exactly like a real 0 %.
- `tests/cue_activation_measure.rs` — 17 model-free tests, including a **calibration**
  assertion (20 same-condition comparisons must not exceed the alpha-implied
  false-positive rate). Without it the metric is a rubber stamp that no GPU time exposes.
- `tests/real_cue_activation.rs` — the run. Opt-in on `SYRINX_CUE_ACTIVATION_OUT`, in no
  group. Asserts structural soundness and writes the JSON; does **not** assert the C4.2
  thresholds unless `SYRINX_CUE_ACTIVATION_ENFORCE=1`, because thresholds that have never
  met reality can only rubber-stamp the model or red the board for an unagreed reason.

**To resume (≈2.5 h, owns the GPU; run nothing else heavy alongside — see the run note below):**

    source scripts/test-all.env
    SYRINX_CUE_ACTIVATION_OUT=.opt-reports/cue-activation.json \
      MEMMAX=28G scripts/run-isolated.sh cargo test --features "real cuda" --release \
      --test real_cue_activation -- --nocapture

**Run note, learned the hard way:** any other `cargo` invocation blocks on the same
target-dir lock and stalls the run before it starts. Either leave cargo alone for the
duration or give the run its own `CARGO_TARGET_DIR`.

**The only data so far — a 2-case pilot, which certifies NOTHING:**
`en-emotion-happy-leading` p=0.371 and `en-emotion-sad-mid` p=0.086, both **not activated**,
WER 0.000 in both conditions. Two cases cannot distinguish "s2-pro barely moves on bracket
cues" from noise. It is recorded only so the full run has something to be compared against,
and it is the reason the thresholds are not yet an assertion. **No claim is made about real
activation rates until the full 54-case run completes.**

### A26 — 2026-09-06 — **documentation reconciled; two superseding amendments recorded**

A full sweep of every doc, ADR and `FINDINGS.md` found **17 contradictions**. Most were
harmless staleness; six were `CLAUDE.md` — the file every pass reads in full — being *wrong
about the current tree*, which misdirects work rather than merely aging. Corrected:

- **The crate table listed `syrinx-core` and `syrinx-stream`**, both deleted in `d09b11e`,
  and omitted `syrinx-qwen`, `syrinx-fish`, `syrinx-stt`. The workspace has **13** crates;
  the table now says so, and records where the two deleted crates' responsibilities went.
- **BUILD SCOPE Phase 1 still claimed the SSML parser** for `syrinx-frontend`, contradicting
  D6 *in the same file*. A second `CueDoc` producer is exactly what D6 forbids.
- **SIM-o was listed as blocked-on-human.** It is a cosine between speaker embeddings —
  objective, and computed today by `syrinx_eval::qwen::speaker_similarity` against a
  measured ceiling (0.997) and floor (0.925). Removed from the blocked list, with the
  general test stated instead: does the number need *ears*?
- **The opt-in test list named five of seven.** `real_qwen_seed` and `real_qwen_affect` were
  missing. By this file's own rule — an unrun gate is a dead gate — that is a defect.
- **"Verifying the build" described a Fish-only board** and omitted `qwen`, `qwen_ckpt`,
  `cue`, `unit`. It now defers to `scripts/test-groups.sh` as the authority.
- **Phase 2's blocker was imprecise.** *Parity* is what is blocked; the weight-free
  substrate was always buildable and was built. `list.md`'s seven done Phase-2 tasks are
  not a contradiction, and a reader concluding "one of these is lying" was reasonable.

Also: `README.md` still said "Fish Audio is the only TTS path under active development" in
two places — the most misleading text left in the tree — and `DESIGN.md`'s own current-state
banner was Fish-centric. Both now say Qwen and cite `docs/LICENSES.md`. `spec.md` was a
byte-identical copy of `plan.md` (md5 `12c5b94…`) and is now a pointer; `plan.md` and
`list.md` carry a HISTORICAL banner. `QWEN_PORT_STATUS.md` §2.3 had four rows the same file
later contradicted (e2e synthesis, the Python reference dump, CUDA, bf16) — struck through
with their closing dates. `adr/0002`'s "Impact while unfixed" and "Why the obvious fix is
blocked" now say *historical*; they read as a live defect on a skim.

**Two superseding amendments, recorded because this ledger is append-only and its earlier
entries still read as current:**

- **A6 is superseded.** Worktree-per-task was suspended there; `CLAUDE.md` reinstated it on
  2026-09-06 after a `git add -A` in the shared checkout swept two subagents' in-progress
  files into an unrelated commit. `scripts/worktree.sh` exists to make it cheap.
- **A24 is superseded.** "Model direction: Fish only" was reversed on 2026-09-06 on
  **licence**: every Fish checkpoint is research-only, every Qwen3-TTS checkpoint is
  Apache-2.0. See `docs/LICENSES.md`.

**Not changed, because it is a behavioural decision and not a doc fix:**
`crates/syrinx-serve/src/lib.rs:765` still defaults the `backend=` query parameter to Fish
S2-pro, described there as "the primary TTS path". Under a Qwen-only direction the server's
default points at a research-licensed backend. Flagged for the maintainer.

### A27 — 2026-09-06 — **C4.2's acceptance criterion is rewritten (ADR-0003); the 0.85 event floor is retired**

C4.2's AC gated **event activation ≥ 0.85 on `Inline::Open` backends**. After the Qwen
redirection that criterion is orphaned: the only `Inline::Open` backend is `fish-s2-pro`,
which is research-licensed and cannot ship, and every Qwen checkpoint is `inline = none`
with `event = unsupported`, so an event cue is filtered before a render happens.
`tests/real_cue_activation.rs:73` hard-wires `const BACKEND = "fish-s2-pro"`. Completing
the 2.4 h run recorded as outstanding in **A25** would have certified a threshold for a
backend we do not ship — which is why it was *not* completed, and this replaces it.

**0.85 was never measured.** It was written here before any run existed. Retiring it is
correcting the record, and it is deliberately **not** replaced by a smaller
plausible-sounding number: a floor nothing passes is a broken gate, a floor everything
passes is decoration, and one chosen to sit between the two is threshold-fitting.

The new criterion (C4.2′, ADR-0003) is a falsifiability gate plus a ratchet — four clauses,
pre-registered:

- **A — negative controls.** A1 the 0.6B renders bit-identically (a control on *our own*
  `honors_instruct` filter, and the ADR says so rather than calling it a model control);
  A2 A/A calibration at the nominal alpha; **A3 a sham arm**.
- **B — one pre-registered sentinel must activate**, earned by replication on a disjoint
  seed block, with "clause B not enabled" recorded in advance as an acceptable outcome.
- **C — the WER veto**, `wer_delta ≤ 0.5`, carried over unchanged. The only part of the
  original C4.2 that survives intact.
- **D — a provenance-keyed ratchet** rather than a floor.

**A3 is the substantive addition and it has already been run.** `assemble_text_mode`
prepends the instruct as text tokens, so cued and plain differ in prompt *length*, not only
meaning — a model treating the instruct as noise would still reject the cued-vs-plain null.
120 renders on 2026-09-06 (`renders/2026-09-06-instruct-lang/`) put a delivery-neutral
instruction of comparable length in its own arm per language. **All four shams are null**,
so the confound is measured and rejected and `[sad]`'s p=0.0019 stands as a measurement of
content. Recorded blemish: `sham-en`/angry came in at p=0.0325, larger than the real cue on
that case — inside correction, and the reason the arm stays in every future run.

Also, per ADR-0003 §6: `caps.toml`'s `source`/`notes` for the 1.7B-CustomVoice
`emotion`/`style` rows are narrowed to say what was actually verified — the instruction is
*delivered* — with the n=8 nulls cited. The `Support` **values are unchanged**:
`Accepted` lowers exactly like `Unsupported`, so demoting would silently switch the cue
layer off, and `tests/expressive_api.rs:155` is frozen on `"honored"`.

Nothing frozen was edited. `evaluate_activation` and its types keep their exact signatures,
`tests/cue_activation_gate.rs` stays green untouched, and `real_cue_activation` remains the
fish research run. The new logic is additive (`syrinx_eval::contrast`) and, being pure, is
gated on the model-free board by `tests/cue_contrast_gate.rs` — 22 assertions, every clause
pinned on both sides of its boundary.

**Still outstanding for C4.2′:** the run itself (~26 min, opt-in) and clause B's sentinel,
which cannot be pinned until a first run earns one.
### A28 — 2026-09-06 — **the judge is replaced, and `[sad]` is shown to move the audio toward sadness**

The affect judge was the blocker on the only question that mattered. The RAVDESS 8-class
model measured **0.394** cross-corpus and **0.17 on `sad`** — and `[sad]` was the one cue
with a demonstrated acoustic effect (p=0.0019), so the judge had no standing to say whether
the change was *toward sadness*. Every verdict came back "cannot tell", correctly.

`emotion2vec/emotion2vec_plus_large` (FunASR licence, 42,526 h, dedicated SER) replaces it.
Same 180-clip CREMA-D probe, same protocol, held out from both models: **0.911** overall,
**0.867 on `sad`**, 1.000 on `angry`, 0.667 on `fearful` (weakest, confused with sad).
`surprised`/`other`/`unknown` are not probed and no recall is claimed for them.

**Not SenseVoiceSmall**, which `AFFECT_JUDGES.md` had recommended: it is CTC, so emotion
arrives as tokens in a `[1,T,vocab]` stream rather than a `[1,labels]` vector; its unique
offer was audio-event detection, which no Qwen checkpoint supports; and its ASR argument
double-counted Whisper. The research note is retained with the reversal recorded in it.

**The re-run answers the question.** Identical 120 renders, same seeds, new judge:

- `[sad]`/en **+4.007 logits, p=0.0017**; `[sad]`/zh **+5.615, p=0.00014** — both clear
  Bonferroni over all 12 comparisons (α=0.00417), while both shams sit flat at |t| < 0.25.
- Three methods with different failure modes agree: the direction-blind acoustic test says
  `[sad]` is the only cue that changes delivery, the direction-aware judge says it is the
  only cue that moves toward its class, and the shams say neither is perturbation.
- `[angry]` is null for the fourth independent time — and now on a judge with **1.000**
  recall on `angry`, so this is a real null and not an instrument failure. It is the
  clearest target the tuning loop has.

**The caveat, recorded so it is not dropped:** `sad` is still not the winning class. Plain
reads −7.68, cued −3.68/−2.07 — a large reliable move that stays negative; the argmax is
`surprised` either way. The claim is "the cue adds substantial sadness evidence", not "the
render reads as sad". Quoting the p-value without that sentence overstates it.

Two defects were caught by anchoring the judge against funasr rather than trusting the
export, and both would have produced plausible wrong numbers: the dynamo ONNX exporter
emitted a length-dependent graph (right at the traced length, ~3 logits out elsewhere), and
the model saturates so funasr's own softmax is one-hot to float precision — scoring on
probabilities would have flattened every delta to 0 or ±1. The judge is anchored and scored
on **logits**, and the export is capped at 160,079 samples (10.005 s), found by binary
search and enforced by a test that asserts it errors rather than truncating.

### A29 — 2026-09-06 — **the tuning loop is built and has run once; it proposed nothing, and it raised a better question**

ADR-0004 defines where tuned phrasings live, what may accept one, and the guards. The
decision is a pure function (`syrinx_eval::tune`) gated on the model-free board by
`tests/cue_tune_decision.rs` — 22 assertions, 18 mutants all killed, every criterion pinned
on both sides of its boundary. The human gate (`accepted_by`, without which a row is inert)
is mutation-checked three ways.

**First round: `[angry]`, 4 candidates, 192 renders, ~40 min. Nothing proposed.** All four
were rejected at the first criterion — none beat plain acoustically at the corrected
α = 0.0125. That is the loop working, not failing: a round that proposes nothing when
nothing is better is the outcome the guards exist to produce.

**The guards reported healthy.** The sham was null on both splits (p=0.25, 0.52), and the
counter-cue — the *wrong* label's phrase — scored below the incumbent on `angry` and gained
on `happy` instead, on both splits. That is the judge demonstrating inside the run that it
tracks the label it is asked about.

**A finding that complicates A28 and is recorded rather than buried.** On the holdout split
the incumbent scores +2.386 on `angry` with `angry` as top gainer; on the tune split it
scores +0.577 with `neutral`. Every prior conclusion that "`[angry]` does not take" — four
independent looks — used the **same single sentence**. The holdout sentences are short
imperatives, and on those the existing phrase moves the right class.

+2.386 against ±4.722 noise is **inside the noise band**, so this is not a result, and n=4
is loose. But it is a cheap testable hypothesis the earlier framing could not raise:
`[angry]` may be **sentence-dependent** rather than dead. Until an n=8 run across sentence
types settles it, the supported claim is narrower than previously written — on *that*
sentence, `[angry]` does nothing.

Next in cost order: (1) re-measure `[angry]` across sentence types; (2) tune `[sad]`, which
has a demonstrated effect and therefore a real incumbent to beat — a far better-posed
optimisation than trying to create an effect from nothing; (3) widen the candidate grammar
only after those, since more candidates raise the Bonferroni bar and shrinking the search
is free.

Also landed: `tests/real_cue_activation_qwen.rs`, the C4.2′ runner ADR-0003 specified and
left outstanding. Opt-in, ~26 min, first run enforces nothing until a sentinel is earned.

### A30 — 2026-09-06 — **`[angry]` is sentence-dependent; the four-times-repeated verdict was about one sentence**

A29 raised this as a hypothesis from an n=4 side-observation. Measured properly — n=8, six
sentences, per sentence, three arms each, 144 renders:

| sentence | acoustic p | judge Δ | judge p | top gain |
|---|---:|---:|---:|---|
| accusatory-long (the one all four prior runs used) | 0.7417 | +0.058 | 0.980 | happy |
| **new-command** ("Stop talking and listen to me for once.") | **0.0045** | **+10.179** | **0.0010** | **angry** |

On a terse second-person command the cue **works**: the largest judge delta in the table,
t=4.16, clearing the Bonferroni-corrected threshold (α=0.05/6), with `angry` as the
top-gaining class — on the judge's *best* class (CREMA-D recall 1.000). The acoustic test
misses its stricter 12-comparison bar by 8% (0.0045 vs 0.00417).

On the sentence every previous run used, it is dead: t=0.03.

**So "`[angry]` does not take on this checkpoint" was never supported.** What was measured,
four times, is that it does nothing *on that one sentence*. The record is corrected here
rather than quietly restated.

Two things this does NOT establish, stated so they are not read in: no mechanism (the
obvious "imperatives work" story fails — `holdout-imperative` reads `surprised` at p=0.81,
and n=1 sentence per cell cannot separate confrontation from length or lexis), and no
general claim about `[angry]`, only that the variance across sentences is large enough to
flip the verdict.

The run's own conjunctive flag printed `0 of 6`, because ADR-0004 requires both bars to
accept a phrase. That is the right gate for *accepting* and the wrong summary of what was
*learned*; both numbers are reported.

**Consequences.** The `[angry]` tuning round (A29) pooled sentences per split and is not
interpretable as run — "no candidate cleared" may be about the sentence mix. The tuning
holdout partition is retired, since its sentences were measured here; ADR-0004 §5's
`holdout_id` expiry existed before it was needed and now applies. And the caveat
generalises: **`[sad]`'s result also came from one sentence.** It is stronger and
triangulated by three methods, but the same six-sentence sweep is owed before `[sad]` is
described as working generally.

### A31 — 2026-09-06 — **`[sad]` generalises across sentences; `[angry]` does not. The anchor was the least representative sentence.**

A30 showed a cue's verdict can be a property of the sentence it was measured on, so `[sad]`
owed the same six-sentence sweep. n=8, 144 renders, per sentence.

| | `[sad]` | `[angry]` |
|---|:-:|:-:|
| top-gain == the cued class | **5 / 6** | 2 / 6 |
| judge move significant after Bonferroni | **3 / 6** | 1 / 6 |
| both | 2 / 6 | 1 / 6 |
| acoustic clears the corrected bar | 2 / 6 | 0 / 6 |

**`[sad]` is a real and reasonably general capability on this checkpoint.** `[angry]` is
sentence-specific: it works on a terse command and nowhere else measured.

**The anchor turned out to be the least representative sentence in its set.** *"I waited by
the window…"* — the source of every prior `[sad]` conclusion — is the only one of six where
`sad` is not the top-gaining class (`other` is). A28 is not overturned: the `sad` delta
there is +4.007 at p=0.0017. But A28 could not say whether that generalised, and it does —
*better* than the anchor implied. This is also the concrete answer to the caveat A28
recorded against itself ("sad is still not the winning class"): true on the anchor, false
on four of the other five.

**ADR-0004's "the two measures fail differently" is now demonstrated rather than asserted.**
`resignation` clears the acoustic bar (p=0.0003) with its judge move inside noise;
`reflective-long` is the strongest judge result (p=0.00001) and misses the acoustic bar.
They coincide on exactly one sentence. A conjunctive gate over two measures that agreed
would be redundant; over two that disagree this often it is doing real work — and it is why
the strict flag reads 0/6, which stays the right gate for *accepting* a phrase and the wrong
summary of what was learned.

Not claimed: any mechanism (one sentence per cell cannot separate content from length or
prosodic shape), anything beyond `serena`/1.7B/English, and nothing perceptual — the judge
is 0.867 on this class, not 1.0.

Next: `[happy]` is the remaining cue whose verdict ("cannot tell") rests on a single
sentence, and the sweep deliberately refuses to run without a purpose-built sentence set for
it. Tuning is now well-posed for `[sad]` — a real, general incumbent to beat.

### A32 — 2026-09-06 — **`[happy]` does not work; sentence-dependence is NOT a property of the channel**

The third and last cue whose verdict rested on one sentence. n=8, six sentences, 144 renders.

Nothing clears the acoustic bar (smallest p 0.1977), and the one judge-significant cell is
`anticipation` at **−4.198** — significant in the **wrong direction**, top gain `unknown`.
`[happy]`'s "cannot tell" upgrades to *does not work*, on six sentences instead of one.

With all three swept, the structural question A30 raised is answered:

| cue | acoustic | judge sig | top-gain == cue | both |
|---|:-:|:-:|:-:|:-:|
| `[sad]` | 2/6 | 3/6 | **5/6** | 2/6 |
| `[angry]` | 0/6 | 1/6 | 2/6 | 1/6 |
| `[happy]` | 0/6 | 1/6 *(wrong way)* | 3/6 | 0/6 |

**Sentence-dependence is not a general property of the instruct channel.** `[sad]` works
broadly, `[angry]` narrowly, `[happy]` not at all. Had all three degraded together the
honest conclusion would have been that no per-cue claim survives sentence variation; they
did not, so the instrument resolves real differences *between cues*, and `[sad]`'s result is
a property of the cue rather than of its sentence.

Fourth exact reproduction: `anchor-goodnews` returns p=0.3737 and Δ+1.946, matching the
instruct-lang run digit for digit from a different worktree.

Not claimed: any mechanism, and **not** that `[happy]` is unreachable — this measures the
shipped phrase, not the concept. That makes `[happy]` the best-motivated tuning target after
`[sad]`: a clear deficit, and a judge with 0.967 recall to detect a fix.

### A33 — 2026-09-07 — **the tuner pooled sentences and could accept nothing; and the server's default backend moves to Qwen**

**The pooling defect.** The first two tuning rounds pooled 3 sentences x n seeds into one
permutation test per arm. The `[sad]` round exposed it: **the incumbent itself could not
clear the acoustic bar** — p=0.0166 on the tune split, 0.3580 on the holdout — while the
same phrase scores 0.0003–0.0057 measured per sentence at n=8. Twelve pooled samples doing
worse than eight per-sentence samples is the signature of inflated variance, and A30–A32
say exactly where it comes from: sentences differ enough to swamp the cue.

A gate the incumbent cannot pass can accept nothing. That is the failure ADR-0003 named for
the 0.85 activation floor, reached from the other side, and it explains both "no candidate
cleared" results without needing the candidates to have been bad.

Fixed additively: `decide_per_sentence` tests each sentence on its own and aggregates the
**decisions** — a candidate must clear every criterion on a majority of sentences, on each
split. `decide`, `TuneMeasurement` and `TuneThresholds` keep their signatures, so
`tests/cue_tune_decision.rs` stays frozen and green. New frozen companion
`tests/cue_tune_per_sentence.rs`, 12 assertions, 8 mutants all killed. This also makes
"works on some sentences" expressible, which A30–A32 showed is the actual shape of the
phenomenon.

**The server default.** `crates/syrinx-serve/src/lib.rs` defaulted `backend=` to
`fish-s2-pro`, described in the code as "the primary TTS path". That stopped being true on
2026-09-06. A26 flagged it and deliberately did not change it, because it is a behavioural
change to a shipped surface rather than a documentation fix. **Changed now, on maintainer
instruction:** the default is `qwen3-1.7b-customvoice`. An explicit `backend=fish-s2-pro`
still works — the research path is deprecated, not removed.

That required a **maintainer-authorised unfreeze**: `tests/expressive_api.rs` asserted the
old default explicitly, which is what a frozen test is for. The assertion was updated rather
than deleted — what the default *is* still matters, and a silent change should still fail
there — with the authorisation and reason recorded in the test itself, following the
ADR-0002 precedent.

### A34 — 2026-09-08 — **the instruct channel cannot produce a laugh; `[laughs]` is unreachable, not unwired**

`event = unsupported` on all five Qwen checkpoints means `pass_hoist` drops `[laughs]`
before a render. Every backend that honours events is deprecated. But that is a narrower
claim than "the model cannot laugh": Qwen has one expressive channel and nobody had asked
it to laugh through that.

144 renders, 3 sentences x 6 arms at n=8, four phrasings meaning four different things by
"laugh". **No duration increase anywhere** — largest magnitude 0.23 s and *negative*, mean
|Δ| under 0.07 s, where a laugh costs 0.5–1.5 s. Nothing near α=0.00208. WER flat, which
matters because a laugh is non-lexical audio the oracle must transcribe or skip.

The sharpest reading is comparative: **the sham moved the audio more than every laugh
instruction on two of three sentences.** These are not weakly effective; they are less
consequential than a meaningless string of the same length. The model is not *partially*
laughing, which is why "the right words would unlock it" is unlikely on four negatives.

So the current behaviour — drop, record `Dropped { reason: Unsupported }`, speak cleanly —
is correct. It also puts the 2026-09-06 judge decision on measurement rather than
assumption: audio-event detection was SenseVoiceSmall's one capability over emotion2vec+,
passed over because no shipping backend had an event channel. There is none.
Full record and 18 WAVs: `renders/2026-09-08-event-induction/`.

### A35 — 2026-09-09 — **a feasibility check was used as a power check; and a trailing cue was silently inert (ADR-0005)**

**The power defect.** `min_n_for_alpha` answers whether the exact permutation test can
*ever* reject at α. The tuning driver used it as a floor as though it answered whether the
test can reject a *real effect*. At α=0.01 that floor is n=5 with **1.3x** headroom — below
the n=6 the round actually ran, so passing the check said nothing. The `[sad]` incumbent,
independently shown to work on 5 of 6 sentences, scored 0.0022 on its best sentence: that
**is** 1/462, the single most extreme labeling available at n=6. A gate only the most
extreme draw can clear rejects working phrases and reads as "nothing is better".
`min_n_for_headroom(α, 20)` is now required by both the tuner and the C4.2′ runner.
At n=8 the same phrase scores 0.0073 / 0.0002 / 0.0564 — 2 of 3, the gate reachable.

**ADR-0005 (PROPOSED).** `"…all week. [angry]"` produced no instruction *and no drop
record*. A trailing cue gets an empty span — a spanning cue scopes what follows, and nothing
follows — so `is_point()` was true and it was filed as a point **event**, never reaching
`instruct_for`. Right for `[laughs]`, a category error for an emotion: a manner of speaking
cannot occur at an instant. `placement: trailing` is one of three placements in the frozen
C4.2 set, so that set had been measuring a no-op on those cases since it was written.

Found by the C4.2′ runner, and only because that runner distinguishes "the backend cannot
express this kind" from "caps say it can, yet nothing carried an instruct" — an assertion
that exists only because the runner's *own* first version had the same class of bug.

### A36 — 2026-09-09 — **C4.2′ certified: clean, all controls hold, verdict "no"**

n=9 (20x headroom), 6 pre-registered sentinels, 4 arms each. A1 holds (the 0.6B renders
bit-identically), A2 no calibration violation, A3 no sham activated, C no WER regression,
and **0 cases `not_applicable`**. No sentinel shows content activation. Clause B earns no
sentinel and ships disabled, which ADR-0003 pre-registered as acceptable.

Getting there cost **three runner defects**, none visible from a green suite:
`segments.first()` mislabelled 4 of 6 as `not_applicable`; the A1 control OOM'd because the
1.7B was never freed; and A/A was counted per case rather than per text, manufacturing a
`CalibrationFailed` from one measurement counted twice.

**Recorded, not acted on:** two cases separate from their *sham* decisively (`sad-mid`
p=0.0001, `shout-mid` p=0.0000) while their cue-vs-plain sits at 0.0038 and 0.0182 — the cue
is further from a delivery-neutral instruction than from no instruction at all. If sham and
cue move the audio in different directions from plain, the conjunctive requirement may
measure the wrong thing: `cue vs plain` re-admits the very confound the sham arm removes.
Redefining `content_activated` is an ADR-0003 amendment and a maintainer decision, and six
sentinels is thin evidence. Data in `renders/2026-09-09-c42-certification/report.json`.

Also open from the same day: the `[sad]` tuning incumbent clears **0 of 3 holdout
sentences**, so a challenger must clear sentences the incumbent cannot. Defensible (a phrase
working where the incumbent fails is what tuning should reward) but undecided — a threshold
question on `min_sentences`, not a bug.

### A37b — 2026-09-11 — **the holdout question gets four named options (ADR-0004 PROPOSED); and the tuned-row path turned out to be wired to nothing**

*(Numbered **A37b**, not A37. Two agents working in parallel worktrees allocated A37 on the same day; the earlier-committed one kept the plain number. CLAUDE.md's rule is that IDs are immutable and splits add suffixes, so a suffix is the convention this file already has for exactly this shape of clash — and it keeps the 2026-09-11 work ahead of A38, which is dated 2026-09-12. Neither entry had reached `main`, so nothing published was renumbered.)*

**The holdout threshold (A36's open item).** `decide_per_sentence` demands `min_sentences`
on *each* split as an absolute count, so on the 2026-09-09 `[sad]` round a challenger had to
clear 2 of 3 holdout sentences the incumbent cleared **none** of. Four options are now
written down and implemented as `syrinx_eval::tune::HoldoutPolicy`, each with what it can be
gamed by: `Absolute` (status quo — one-sided error, but a bar the incumbent cannot pass
accepts nothing, the same defect A33 fixed on the pooled side); `StrictlyBroaderThanIncumbent`
(collapses to "clear one sentence" exactly when the incumbent is weakest, and rewards
whoever makes the reference look worst); `OnlyWhereIncumbentClears` (like-for-like, but the
eligible set on the round in question is empty, so it fails closed and answers nothing); and
`RequireFitPartition` — keep the absolute bar, and **void** the round when the incumbent
cannot itself clear `min_sentences` holdout sentences.

`RequireFitPartition` is the recommendation, argued in ADR-0004 §PROPOSED (2026-09-11). It
moves no bar, so it adds no lever a search can pull; it inverts the incentive the relative
rules create, because a weak reference voids rather than lowers; and it is the same argument
as the existing `IncumbentNotRemeasured` void one step further in. Nothing is accepted:
`decide_per_sentence` is still `Absolute` by definition, and `examples/tune_instruct.rs`
still calls it. Frozen in `tests/cue_tune_holdout_policy.rs`, 14 hand-injected mutants, no
survivors.

**The tuned-row path had never run, and it was wired to nothing.** ADR-0004's storage
mechanism had unit coverage of inertness and zero end-to-end coverage. Driving it with
fixture rows (`tests/instruct_tuned_path.rs`, 23 assertions) found four defects:

1. **`phrase_for_backend` was called from tests and from no production code.** Every prefix
   in the pipeline came from `hoist::instruct_for`, which is backend-blind, so an accepted
   row a human had signed would have been inert with no diagnostic. ADR-0004 §2's
   tuned → curated → fallback order did not exist anywhere — tiers 1–2 in `instruct.rs`,
   tier 3 in `hoist.rs`, nothing joining them. Fixed: `instruct_with` / `pass_hoist_with`
   resolve against `caps.id`, the per-checkpoint key §1 writes a row under.
2. **A tuned phrase was exempt from the hard invariant.** The frozen well-formedness test
   walks curated rows only, so a signed row containing `[sad]` would have reached a backend
   as literal text — from the one source no human reads before it ships.
   `syrinx_cue::instruct::phrase_is_safe` now gates at load, and a frozen test asserts it
   agrees case for case with `syrinx_eval::tune::phrase_is_safe`, which gates at proposal.
3. **Provenance was accepted blank or absurd**: `judge = ""`, `measured_on = ""`,
   `judge_recall_on_class = 3.0`, `margin = -1.0`, and a `lang` no lookup can ever produce
   (silently inert forever). All are load errors now, for signed and unsigned rows alike.
   `backend` deliberately keeps no such check: it is an open set, so an unknown id is not
   *provably* unreachable the way an unknown `lang` is.
4. **Two accepted rows for one `(backend, lang, label)` resolved by row order.** Now a load
   error: a file that does not say what will be spoken must not load.

Also fixed: `examples/tune_instruct.rs` hard-coded `measured_on = "2026-09-06"` and
`holdout_id = "{label}-2026-09-06-a"` into every proposal it wrote, so the 2026-09-09 rounds
would have signed a false date. It now requires `SYRINX_TUNE_MEASURED_ON` and refuses to
default it — the load-time validation can tell that a date is blank and can never tell that
it is a lie.

`crates/syrinx-cue/instruct.toml` still has **zero** `[[tuned]]` rows and a frozen test
asserts the shipped file contains no `accepted_by` at all. Signing a real tuned phrase
remains a human act.
