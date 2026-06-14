# Syrinx — Task-Based Build Plan

> Codename **Syrinx** (the avian vocal organ — placeholder, swap freely).
> A local, Rust-served neural TTS + zero-shot voice-cloning engine.

**Status:** Planning · **Owner:** NovaDevelopment · **Last updated:** 2026-06-14
**Target hardware:** single RTX 4090-class GPU · **Inference:** Rust · **Training/research:** Python

---

## Conventions

- **Task IDs:** `T<phase>.<n>`. Checkbox = status. `- [ ]` open · `- [x]` done · `- [~]` in progress · `- [!]` blocked.
- **AC (Acceptance Criteria):** a task is *done* only when its AC passes the eval harness or CI gate — not by assertion. Where possible AC is machine-checkable (an unfakeable gate). "Looks fine" is never an AC.
- **Size:** S (≤1 day) · M (≤1 week) · L (multi-week). L tasks should be split before execution.
- **Deps:** only non-obvious / cross-phase dependencies are listed; same-phase sequential deps are implied.
- **Track:** `core` (critical path) · `data` (Phase 5 corpus, runs in parallel) · `research` (Phase 6, parallel, partial-result tolerant).
- Plan is structured to drop into a stateless harness loop (one task per pass, AC as the gate).

---

## 1. Mission

Clone a voice zero-shot from seconds of reference, render it with full prosodic + emotional control and human paralinguistic detail, stream sub-200ms on one consumer GPU, and expose an editable prosody plan. Inference is pure Rust; the model is an open hybrid base extended with our own control, speaker-latent, and paralinguistic layers. Closes the TTS gap in NovaFox.

## 2. Scope

**In:** zero-shot cloning, cross-lingual/multi-accent, emotion + prosody control, editable prosody plan, paralinguistic artifacts, speaker blend/morph, deterministic Rust text frontend, streaming + telephony, 48kHz, Rust runtime.
**Out (this iteration):** scratch codec-LM pretraining (Path B), perfect attribute disentanglement (partial only), singing voice, full conversation simulation.

## 3. Resolved design decisions

| # | Tension | Decision | Rationale |
|---|---------|----------|-----------|
| D1 | `<100M` vs cloning + range | ~350–400M model, **4-bit served** (~270MB); lean acoustic decoder (~120M) on hot path | `<100M` was a proxy for cheap-local + low-latency; quant meets the intent, literal sub-100M can't also clone with range |
| D2 | "Non-AR" globally vs control + streaming | **Paradigm per stage:** AR control layer + non-AR flow-matching acoustic layer | Each paradigm where it's strong |
| D3 | Sub-200ms TTFB vs whole-utterance NAR | **Chunk-aware causal flow matching** | First-byte = one chunk |
| D4 | Phoneme-level editing vs NAR renderer | **Separate prosody plan from renderer** | Edit the plan; render in parallel |
| D5 | Morph/blend vs disentanglement | **Ships now** via speaker-embedding interpolation | Interpolation ≠ attribute control |
| D6 | Age / gender / dialect axes | **Research-track**, attribute-conditioned + adversarial disentanglement, partial | Not solved by vanilla encoder; bounded method exists |
| D7 | Build vs adopt | **Adopt open hybrid base**, extend; Path B explicit alternative | Value is control/latent/Rust runtime, not re-deriving a codec LM |

**Cut:** literal `<100M` + full range (→ D1). **Open fork:** if "tiny above all" beats cloning, flip to enrolled-voice (per-speaker LoRA). Plan assumes cloning.

## 4. Architecture (planning-level)

```
text ─▶ [Frontend] ─▶ [AR Semantic LM] ─▶ [Prosody Plan] ─▶ [NAR Flow Decoder] ─▶ [Vocoder] ─▶ 48kHz
        (Rust,         (control,            (editable          (acoustic, chunk-      (HiFi-GAN/
       deterministic) paralinguistics)      dur + pitch)        aware, speaker-       Vocos)
                                                                cond.)  ▲
                                          [Speaker Encoder] ───────────┘
                                          (embed · blend · morph · attributes)
```

**Param budget (pre-quant ~420M):** LM ~250M · acoustic ~120M · speaker enc ~30M · vocoder ~20M → ~270MB @ 4-bit.

## 5. Crate layout (Rust workspace)

| Crate | Responsibility |
|-------|----------------|
| `syrinx-frontend` | normalization, G2P, SSML, lexicon, heteronyms, context windowing |
| `syrinx-core` | tensor ops glue, weight loading, quantization, device mgmt |
| `syrinx-lm` | AR semantic LM forward pass + paralinguistic tokens |
| `syrinx-speaker` | speaker encoder, embedding store, blend/morph, attributes |
| `syrinx-acoustic` | flow-matching decoder (DiT + ODE solver), chunk-aware streaming |
| `syrinx-vocoder` | HiFi-GAN/Vocos waveform synthesis, 48kHz/8kHz paths |
| `syrinx-prosody` | editable prosody plan model + override API |
| `syrinx-stream` | packet streaming, ring buffer, `cpal` out, TTFB path |
| `syrinx-serve` | Axum server, OpenAI-compatible `/v1/audio`, watermarking |
| `syrinx-eval` | MOS/SIM-o/WER/latency harness, frozen eval set runner |
| `syrinx-cli` | local runner / dev harness |

## 6. Build paths

- **Path A (default):** adopt open base (Chatterbox MIT / CosyVoice2 / F5) → reimplement inference in Rust → add control, speaker latent, paralinguistics. Weeks-to-months.
- **Path B (alt):** scratch codec-LM pretrain. Multi-GPU-month + large corpus. Separate greenlit track only if from-scratch *is* the goal.

Tasks below assume **Path A**.

---

## Phase 0 — Foundations  `core`

- **T0.1** Repo + Rust workspace scaffold per §5 crate map. **AC:** `cargo build` green on empty crates; CI runs. **Size:** S
- **T0.2** CI pipeline: build, clippy, fmt, test, frozen-eval gate. **AC:** PR blocked on any failing gate. **Size:** S
- **T0.3** Eval harness skeleton in `syrinx-eval`: hooks for SIM-o, WER (ASR), MOS-proxy, TTFB, RTF. **AC:** harness runs against a stub and emits a metrics JSON. **Size:** M
- **T0.4** Frozen eval set: held-out reference clips + transcripts + target thresholds, age-encrypted. **AC:** set is immutable + checksummed; cannot be edited without breaking the gate. **Size:** S
- **T0.5** Base-model license + arch screen (Chatterbox / CosyVoice2 / F5): matrix of license, params, streaming, cloning, multilingual. **AC:** matrix doc committed; disqualifiers flagged. **Size:** S
- **T0.6** Base-model A/B bench: SIM-o, WER, latency on the frozen set → **select base**. **AC:** ranked results + decision recorded in `ARCHITECTURE.md`. **Deps:** T0.3, T0.5. **Size:** M
- **T0.7** Reproducible reference (Python) inference of the chosen base, end-to-end. **AC:** deterministic audio from text+reference; seed-pinned. **Deps:** T0.6. **Size:** M
- **T0.8** `ARCHITECTURE.md` v0 + `CLAUDE.md` scaffold (doctrines, gates, crate contracts). **AC:** both committed. **Size:** S
- **T0.9** Consent/ethics + watermarking policy doc. **AC:** usage policy + watermark requirement defined before any cloning ships. **Size:** S

## Phase 1 — Text frontend (deterministic Rust win)  `core`

- **T1.1** Text normalization core (unicode, whitespace, casing). **AC:** golden-file normalization suite passes. **Size:** M
- **T1.2** Number/date/currency/ordinal expansion (numeric context). **AC:** test set incl. `$1,200`, `1/2/26`, `3.14`, `1st` resolves correctly. **Size:** M
- **T1.3** Acronym expansion + custom dictionary override system. **AC:** user lexicon overrides default; precedence tested. **Size:** S
- **T1.4** G2P / phonemizer (pure-Rust or espeak-ng-backed) with IPA output. **AC:** phoneme accuracy ≥ target on a labeled set. **Size:** L
- **T1.5** Custom pronunciation mapping (per-word IPA overrides). **AC:** override replaces G2P output for mapped words. **Deps:** T1.4. **Size:** S
- **T1.6** Heteronym resolver (POS/context classifier). **AC:** `read`/`lead`/`bow` disambiguated ≥ target on test set. **Deps:** T1.4. **Size:** M
- **T1.7** SSML parser (subset: `prosody`, `break`, `emphasis`, `say-as`, `phoneme`, `sub`). **AC:** parses spec subset → typed control events; malformed input errors cleanly. **Size:** M
- **T1.8** Punctuation → prosody hints (pauses, boundary tones). **AC:** punctuation maps to break/intonation markers. **Size:** S
- **T1.9** Cross-sentence context windowing (surrounding text → LM conditioning). **AC:** window assembled + passed across the typed boundary. **Size:** S
- **T1.10** Paragraph pacing + automated breathing-interval calculation. **AC:** breath markers inserted at computed intervals; multi-paragraph pacing stable. **Size:** M
- **T1.11** Frontend test suite + normalization golden tests. **AC:** suite in CI gate. **Size:** S
- **T1.12** Frontend→LM interface contract (typed token/phoneme + control stream). **AC:** schema documented + versioned; consumed by `syrinx-lm`. **Size:** S

## Phase 2 — Rust inference runtime + cloning baseline  `core`

- **T2.1** Weight loader: port base weights → Rust tensor format (Candle/Burn). **AC:** weights load; shapes/dtypes match reference. **Deps:** T0.7. **Size:** M
- **T2.2** Semantic LM forward pass in Rust. **AC:** logits parity vs Python within tolerance on fixed input. **Deps:** T2.1. **Size:** L
- **T2.3** Speaker encoder forward (reference → embedding). **AC:** embedding parity vs reference. **Deps:** T2.1. **Size:** M
- **T2.4** Flow-matching acoustic decoder (DiT blocks + ODE solver). **AC:** mel parity within tolerance at fixed seed/steps. **Deps:** T2.1. **Size:** L
- **T2.5** Vocoder (HiFi-GAN/Vocos) in Rust, 48kHz. **AC:** waveform parity; no audible artifacts vs reference. **Deps:** T2.1. **Size:** M
- **T2.6** End-to-end pipeline wiring (frontend→LM→plan→decoder→vocoder). **AC:** text+reference → audio, no Python in path. **Deps:** T1.12, T2.2, T2.3, T2.4, T2.5. **Size:** M
- **T2.7** Numerical parity harness (Rust vs Python, per-stage tolerances). **AC:** all stages within tolerance in CI. **Deps:** T2.6. **Size:** M
- **T2.8** 4-bit quantization path (ISQ-style) + per-bit-width quality eval; fp16 fallback. **AC:** SIM-o/WER degradation within budget at 4-bit; fallback selectable. **Deps:** T2.6. **Size:** M
- **T2.9** Zero-shot cloning validation. **AC:** SIM-o ≥ baseline on frozen set. **Deps:** T2.6. **Size:** S
- **T2.10** Cross-lingual + multi-accent validation. **AC:** intelligible cross-lingual transfer; accent retained; WER ≤ target. **Deps:** T2.9. **Size:** M
- **T2.11** Watermark embedding (PerTh-style) on every output. **AC:** watermark detectable post-MP3/edit; near-100% detect. **Deps:** T0.9, T2.6. **Size:** M
- **T2.12** Footprint check @ 4-bit. **AC:** resident ≤ ~300MB; runs on one 4090. **Deps:** T2.8. **Size:** S

## Phase 3 — Control surface  `core`

- **T3.1** Prosody plan data model (typed editable duration + pitch arrays). **AC:** plan serializes/round-trips; versioned schema. **Size:** M
- **T3.2** Duration predictor exposure + override. **AC:** overriding durations changes timing predictably. **Deps:** T3.1. **Size:** M
- **T3.3** Pitch/F0 contour predictor exposure + override (word + phoneme level). **AC:** per-word and per-phoneme pitch edits audibly apply. **Deps:** T3.1. **Size:** M
- **T3.4** Speech-rate stretching control. **AC:** rate scales without pitch shift. **Size:** S
- **T3.5** Volume automation curves. **AC:** per-segment volume envelope applied. **Size:** S
- **T3.6** Emotion steering (text-prompted + intensity scale). **AC:** A/B perceptual check confirms intended emotion + monotonic intensity. **Size:** M
- **T3.7** Intonation contour manipulation API. **AC:** contour presets + manual curves apply. **Deps:** T3.3. **Size:** S
- **T3.8** Sarcasm/irony inflection control. **AC:** inflection toggles produce the expected contour shift in eval. **Deps:** T3.6, T3.7. **Size:** M
- **T3.9** Phoneme-level manual edit API (the plan editor). **AC:** edit any phoneme's dur/pitch; renderer honors it. **Deps:** T3.2, T3.3. **Size:** M
- **T3.10** Edited-plan round-trip test. **AC:** edited plan → rendered audio matches the edit (deterministic). **Deps:** T3.9. **Size:** S
- **T3.11** Automated prosody prediction quality eval. **AC:** default prosody MOS-proxy ≥ target. **Size:** S

## Phase 4 — Speaker latent: blending & morphing  `core`

- **T4.1** Speaker-embedding space audit (structure, distances, clustering). **AC:** report on interpolability + caveats. **Deps:** T2.3. **Size:** S
- **T4.2** Multi-speaker profile blending (weighted interpolation). **AC:** blend of 2+ enrolled voices is coherent in eval. **Deps:** T4.1. **Size:** M
- **T4.3** Real-time voice morphing (live interpolation across chunks). **AC:** morph transitions artifact-free at chunk boundaries. **Deps:** T4.2, T7.1. **Size:** M
- **T4.4** Bilingual seamless switching (mid-utterance language change). **AC:** language flips without timbre break. **Deps:** T2.10. **Size:** M
- **T4.5** Enrollment pipeline (store/manage speaker profiles). **AC:** enroll from clip → persisted embedding; recall stable. **Deps:** T2.3. **Size:** S
- **T4.6** Blend/morph perceptual eval. **AC:** blind listening passes for coherence. **Size:** S

## Phase 5 — Paralinguistics + corpus  `data` (parallel; start early)

**5a — Spec**
- **T5.1** Paralinguistic taxonomy + label schema (breath, laugh, sigh, throat-clear, hesitation; phonation modes). **AC:** schema doc + annotation guide. **Size:** S
- **T5.2** Corpus sourcing plan + licensing/consent manifest. **AC:** per-source provenance + consent tracked. **Size:** M

**5b — Collect / annotate**
- **T5.3** Recording/collection pipeline. **AC:** ingest → normalized clips with metadata. **Deps:** T5.2. **Size:** M
- **T5.4** Forced-alignment tooling. **AC:** clips aligned to phoneme timestamps. **Size:** M
- **T5.5** Annotate breaths/laughter/sighs/throat-clears/hesitations. **AC:** labeled set ≥ target hours, inter-annotator agreement ≥ target. **Deps:** T5.3, T5.4. **Size:** L

**5c — Phonation modes**
- **T5.6** Whisper-mode data + control label. **AC:** whispered set collected + labeled. **Size:** M
- **T5.7** Shout/projection data + label. **AC:** collected + labeled. **Size:** M
- **T5.8** Vocal fry data + label. **AC:** collected + labeled. **Size:** M
- **T5.9** Vocal fatigue data + label. **AC:** collected + labeled. **Size:** M

**5d — Train / integrate**
- **T5.10** Paralinguistic token vocabulary + LM extension. **AC:** tokens emit + decode through the pipeline. **Deps:** T2.2, T5.5. **Size:** M
- **T5.11** LoRA/fine-tune for insertion + control. **AC:** controllable triggering of each artifact. **Deps:** T5.10. **Size:** L
- **T5.12** Whisper↔spoken + whispered-to-spoken transitions. **AC:** mode switches mid-utterance are clean. **Deps:** T5.6, T5.11. **Size:** M
- **T5.13** Contextual/dynamic triggering (laughter, hesitation injection, vocal-fry level, organic throat-clear). **AC:** context-driven + level-adjustable in eval. **Deps:** T5.11. **Size:** M
- **T5.14** Blind-listening organic-ness eval. **AC:** artifacts rated natural ≥ target vs human. **Size:** S

## Phase 6 — Disentanglement (research)  `research` (parallel; partial-tolerant)

- **T6.1** Attribute label set (age/gender/accent) + data tagging. **AC:** tagged subset ready. **Size:** M
- **T6.2** Attribute conditioning inputs into model. **AC:** attributes wired as separate conditioning. **Deps:** T6.1, T2.4. **Size:** M
- **T6.3** Adversarial disentanglement loss (strip attributes from timbre embedding). **AC:** classifier-on-timbre accuracy drops toward chance. **Deps:** T6.2. **Size:** L
- **T6.4** Age-progression axis eval. **AC:** age knob shifts perceived age with measurable independence. **Size:** S
- **T6.5** Gender-neutral synthesis eval. **AC:** neutral target reachable; timbre stable. **Size:** S
- **T6.6** Dialect-shifting axis eval. **AC:** dialect knob shifts accent with partial independence. **Size:** S
- **T6.7** Disentanglement metrics report. **AC:** independence scores documented; honest partial-result writeup. **Size:** S

## Phase 7 — Performance & integration hardening  `core`

- **T7.1** Streaming packet path (chunked emit, ring buffer, `cpal`). **AC:** continuous audio with no underruns. **Deps:** T2.6. **Size:** M
- **T7.2** TTFB tuning (chunk size vs quality). **AC:** TTFB < 200ms p50 streaming. **Deps:** T7.1. **Size:** M
- **T7.3** RTF optimization (kernel fusion, batching). **AC:** RTF < target on one 4090. **Size:** M
- **T7.4** Telephony path (8kHz resample, band-limit, codec). **AC:** validated 8kHz output; intelligible over narrowband. **Size:** M
- **T7.5** Noise robustness (reference denoise + augmentation). **AC:** cloning stable on noisy reference within budget. **Size:** M
- **T7.6** Lip-sync timeline export (phoneme timestamps → viseme/timeline). **AC:** timeline aligns to audio within tolerance. **Deps:** T1.4, T2.6. **Size:** S
- **T7.7** Footprint + load/stress test (concurrency on one 4090). **AC:** ≤300MB @ 4-bit; concurrency target met without OOM. **Deps:** T2.12. **Size:** S
- **T7.8** Background-noise-robust reference enrollment. **AC:** enroll from noisy clip without quality cliff. **Size:** S

## Phase 8 — Integration + release  `core`

- **T8.1** OpenAI-compatible `/v1/audio` server (Axum) in `syrinx-serve`. **AC:** drop-in client works; streaming endpoint live. **Deps:** T7.1. **Size:** M
- **T8.2** (Optional) Anthropic-style surface for stack consistency. **AC:** parity endpoint. **Size:** S
- **T8.3** NovaFox v2 wiring (replace OpenVoice/Piper). **AC:** NovaFox runs end-to-end on Syrinx. **Deps:** T8.1. **Size:** M
- **T8.4** Docs (API, control surface, deployment, ethics). **AC:** complete + examples. **Size:** S
- **T8.5** Release packaging + model card + ethics statement. **AC:** versioned release; model card published. **Deps:** T0.9. **Size:** S

---

## 7. Parallel tracks & critical path

- **Critical path:** P0 → P1 → P2 → P3 → P4 → P7 → P8.
- **`data` track (P5)** runs in parallel from day one — it is the long pole. T5.10–T5.14 rejoin the core after P2 (need T2.2).
- **`research` track (P6)** runs in parallel, partial-result tolerant, never blocks a release.
- **Hard cross-track joins:** T4.3 needs T7.1 (streaming) · T5.10 needs T2.2 (LM) · T6.2 needs T2.4 (decoder).

## 9. Data plan (the critical path)

Architecture is settled; **data makes or breaks Syrinx.** Base cloning data comes from the adopted base. The paralinguistic + phonation corpus (P5) has no good off-the-shelf source — build/license it, with provenance + consent tracked (T5.2). Attribute labels (T6.1) feed disentanglement. Start P5 immediately, in parallel.

## 10. Risks

| Risk | Sev | Mitigation |
|------|-----|------------|
| Paralinguistic data unavailable/expensive | High | P5 parallel from day 1; own milestones |
| Disentanglement underdelivers | Med | Research-track, partial, off critical path |
| 4-bit degrades cloning | Med | Multi-bit eval (T2.8) + fp16 fallback |
| TTFB target missed | Med | Chunk-aware decoder from P2; tune T7.2 |
| Base license incompatibility | Med | License-screen T0.5 before commit |
| Scope creep into Path B | Med | Scratch pretrain separately greenlit |
| Cloning misuse | High | Consent policy + watermark (T0.9, T2.11) from first cloning output |

## 11. Success metrics

- **Quality:** MOS/CMOS vs reference; SIM-o for cloning.
- **Intelligibility:** WER via ASR on output.
- **Latency:** TTFB p50 < 200ms; RTF < target on one 4090.
- **Footprint:** ≤ ~300MB @ 4-bit.
- **Control:** A/B that each control produces the intended change.

## 12. Harness scope note (Ratchet)

This plan is driven by the Ratchet harness. Only tasks with a deterministic,
frozen-test-gateable AC are autonomous build tasks (the text frontend, the eval-harness
skeleton, the prosody data model, the server scaffold, the docs). ML/GPU/perceptual
tasks (Phases 2, 4, 5 collection/training, 6, most of 7) are **blocked-on-human** in
`list.md` — see `CLAUDE.md` → "THE BUILD SCOPE". They are unblocked once a human has
trained/ported the model and built the corpus, at which point they earn real gates.
