# Training data format — the Syrinx-native expressive transcript

**Status: design only.** Nothing in this document is implemented in this upgrade
(spec §2.6: *"Do not implement training in this upgrade; make sure nothing in the IR
prevents it"*). It exists so the IR that C1–C4 built can be checked against the training
requirements now, while changing it is still cheap.

The load-bearing idea: **the authoring syntax is the transcript format.** The same string a
user types into `/v1/audio/speech` is the same string an annotator produces, and the same
string a training example stores. One syntax, one parser (`syrinx-cue`), one IR
(`CueDoc`) — so a cue can never mean one thing at training time and another at inference.

---

## 1. Annotator pipeline

Stages, in order. Each stage's output is the next stage's input; every stage is
reproducible from its inputs plus a pinned model version.

1. **Separation + VAD.** Source separation to isolate speech from music/effects, then voice
   activity detection to cut continuous audio into utterance candidates. Output: audio
   segments with sample-accurate boundaries.
2. **Quality filter.** Reject segments on measurable criteria — estimated SNR, clipping
   ratio, bandwidth (a segment upsampled from 8 kHz is not 44.1 kHz training data),
   duration bounds, and single-speaker confidence. Thresholds are recorded per corpus, not
   per run, so a corpus can be re-derived.
3. **Rich transcription with inline cues at word position.** Bootstrapped, not
   hand-written:
   - an open audio-understanding model proposes emotion/style labels and paralinguistic
     events;
   - a **forced aligner** places each proposal at a word position — this is what makes a
     cue's position meaningful rather than decorative;
   - an **event detector** supplies point events (laughter, breath, cough) with timestamps.
   The three are merged into one cue-annotated transcript.
4. **Human-checked sample for precision.** A fixed random sample of each corpus is reviewed
   by a human, and the measured precision per cue kind is recorded with the corpus. A
   corpus ships with its precision numbers or it does not ship. Recall is *not* claimed:
   the pipeline under-annotates by design, because a missing cue costs less than a wrong one.

**Vocabulary.** Labels are the canonical ids in `crates/syrinx-cue/vocab.toml`. An
annotator proposing a label outside it stores the free text as-is — the IR carries it as
`CueKind::Free`, which is exactly what an open-vocabulary model can learn from. Extending
the vocabulary is a vocab.toml change, never a format change.

---

## 2. Cue storage

- **Cues are stored as text, at position**, inline in the transcript, in the authoring
  syntax: `he stopped [sigh] and turned away`. Not a side-car table of offsets — offsets
  rot the moment the transcript is edited, and a side-car cannot be read by a human.
- **Point events vs. spans.** A point event occupies no text and scopes nothing; a span cue
  scopes the text that follows it to the next cue or the end of the sentence. Whether a
  label is a point event is decided by its `Kind` in the vocabulary, not by its spelling,
  so the two can never drift apart.
- **Speaker turns are `<|speaker:N|>`**, numbered per example from 0. The token is part of
  the transcript text, exactly as at inference.
- **Reference audio prefix is loss-masked.** A training example is
  `[reference audio][reference transcript][target transcript]`; the loss is computed on the
  target only. Without the mask the model learns to reconstruct the prompt, which is the
  classic zero-shot-cloning failure.
- **Escaping.** A literal bracket in a transcript is `\[` / `\]`, the same escape the
  authoring syntax uses. This matters more in training data than at inference: a corpus
  full of unescaped stage directions would teach the model to speak them.
- **The hard invariant applies to training data too.** No unescaped cue markup may survive
  into the text the model is asked to *speak*; the cue is a control token, not a word.
  `syrinx-cue`'s parser is the single arbiter of that split, at training and inference
  alike.

---

## 3. Text/audio interleaving parameters

Syrinx's AR path consumes an interleaved text/audio stream, so the format must pin how an
example is chunked. These are the knobs, with their defaults to be fixed empirically before
any training run:

| Parameter | Meaning | Notes |
|-----------|---------|-------|
| `text_chunk_tokens` | text tokens emitted per interleave step | too large and the model loses fine alignment; too small and it loses linguistic context |
| `audio_chunk_frames` | codec frames emitted per interleave step | must divide evenly into the codec's frame rate |
| `interleave_probability` | fraction of examples trained interleaved vs. fully sequential | a mix teaches both streaming and one-shot behaviour |
| `cue_emission_position` | where a cue token is emitted relative to its span | **before** the first token of its span — see below |
| `max_context_chunks` | interleave steps per training window | bounds attention cost |

**Cue emission position is not a free parameter in the same sense as the others.** A cue is
emitted *immediately before the first token of the span it governs*, which is what
`syrinx_cue::offsets::interleave` already implements and what
`tests/cue_token_alignment.rs` pins with golden dumps. Training and inference must use the
same rule or every cue is learned one position off.

---

## 4. Reward spec

Reward is the §2.5 activation harness (`syrinx_eval::activation`), reused rather than
reinvented, so the thing being optimised is the thing being measured.

- **Base reward:** per-sample activation (did the cue measurably change the audio?) minus
  the WER delta against the un-cued baseline, so expressiveness bought with intelligibility
  is not rewarded.
- **Heavy penalty for missed cues.** A cue present in the transcript that does not activate
  is penalised well above the base scale. Under-expression is the failure mode that makes
  the whole control surface feel broken to a user, so it must dominate.
- **Heavy penalty for wrong speaker IDs.** A turn attributed to the wrong speaker is worse
  than a flat delivery: it is a factually wrong output. Penalised at the same weight as a
  missed cue or higher.
- **GRPO-style group advantages, without std normalization.** Advantages are computed
  within a group of samples from the same prompt, mean-centred but **not** divided by the
  group standard deviation. Dividing amplifies noise in low-variance groups — exactly the
  groups where every sample is already good — and destabilises training.
- **The eval set is frozen and checksummed** (`tests/golden/cue_eval/cue_set.jsonl`), so a
  reward number is comparable across runs. Changing the set invalidates every historical
  number, which the checksum test says out loud.

---

## 5. What the IR must not prevent (checked against C1–C4)

| Requirement | Satisfied by |
|-------------|--------------|
| cue stored as text at a word position | `Cue::raw` + `Cue::span` (byte offsets into clean text) |
| position survives to the token stream | `OffsetMap::anchor` / `interleave` (C3.1) |
| free-vocabulary labels not lost | `CueKind::Free` carries `raw` verbatim (C2.2) |
| speaker turns first-class | `CueKind::SpeakerTurn` + `SpeakerRef` |
| point events distinguishable from spans | `Cue::is_point()`, driven by the vocabulary |
| the author's exact text always recoverable | `Cue::raw` is never rewritten by canonicalisation |
| reward reuses the eval harness | `syrinx_eval::activation::evaluate_activation` |

Nothing in the current IR blocks the training design. The one thing it deliberately does
**not** carry is a timestamp per cue: cues are anchored to text positions, and audio
alignment is the aligner's job at corpus-build time. If training later needs frame-accurate
cue timing, that is an additive field on `Cue`, not a redesign.
