# Syrinx control surfaces — feasibility survey (Phase 3+ differentiator layer)

This note records what control the **CosyVoice2 base** (as reimplemented in Rust
across `syrinx-{frontend,lm,acoustic,vocoder,speaker,serve}`) actually exposes
*today*, versus what needs new training. It is the design half of the first slice
of Syrinx's differentiator layer (DESIGN §6 Phase 3 "editable prosody + control",
Phase 5 "paralinguistics"). It is descriptive of the code as built, not
aspirational.

## The three real control surfaces of the base

| Surface | Where it lives in the pipeline | Status |
|---|---|---|
| Paralinguistic markers | text → tokenizer added-token table → LM | **Usable now**, no training |
| Speech-rate (duration) | generated mel → vocoder (frame count) | **Usable now**, no training |
| Instruct / emotion text prompt | instruction text → LM text stream | **Needs the instruct checkpoint / training** |

### 1. Paralinguistic markers — usable now

The text tokenizer (`syrinx-frontend::tokenizer::TextTokenizer`) is loaded from
the model's `tokenizer.json`, which carries CosyVoice2's *added special tokens*:
`<|endofprompt|>`, `[breath]`, `<strong>`/`</strong>`, `[laughter]`, `[noise]`,
`[cough]`, `[lipsmack]`, `[mn]`, … Those are atomic ids (not BPE-split). A marker
placed in the text therefore tokenizes to its own id and flows straight through
`Qwen2Lm::build_lm_input` → `text_embed` → `generate`, where the LM can emit the
matching speech tokens. **No model or LM change is needed** — only a clean way to
position markers in a string. That is `syrinx_prosody::markup` (`Markup`,
`Marker`).

What is *not* settled by the code alone: whether each marker audibly changes the
rendered speech is a property of the trained base model. The base was trained with
these tokens, so they should take effect; the on-box smoke A/B-tests this
(`[laughter]` vs none → different speech tokens / audio).

### 2. Speech-rate control — usable now (mel time-scale)

The acoustic stack fixes a **token→mel ratio of 2** (`forward_zero_shot` produces
a mel of length `2 * (|prompt_token| + |speech_token|)` via the encoder's
Upsample1D ×2). The HiFT vocoder then maps a fixed hop per mel frame, so the mel's
**frame count sets the duration**. Pitch is carried by the per-frame spectral
envelope (which mel bands have energy), independent of frame count.

So a pitch-preserving rate knob = **time-scale the generated mel along its frame
axis by `1/rate`** before the vocoder. `syrinx_prosody::render::time_scale_mel`
does this with deterministic linear interpolation on the frame axis only; the
synth hook `Synthesizer::synthesize_with_rate` wires it between flow and vocoder.
This is the classic length-regulator move and needs no training.

Two layers exist, intentionally distinct:
* `ProsodyPlan::scale_rate` (`rate.rs`) scales the **editable plan's**
  `durations_ms` — for the plan editor / display; it does *not* by itself change
  a waveform.
* `render::time_scale_mel` is the **audio-affecting** transform actually applied
  at render time.

Limitation: this is a *uniform* utterance-level stretch. Per-phoneme rate (from an
edited plan driving the length regulator token-by-token) needs a duration-aware
regulator hook into the flow encoder, which the base does not expose as a clean
seam — that is future work (see below).

### 3. Instruct / emotion — needs the instruct checkpoint

CosyVoice2's `inference_instruct2` steers style/emotion by putting an
**instruction string** into the LM's text stream (e.g. "用开心的语气说" /
"speak in a happy tone") ahead of the content, the same slot our
`build_lm_input` fills with `prompt_text ++ text`. Mechanically the base LM
*could* consume an instruction the same way. But the base checkpoint on the box
(`CosyVoice2-0.5B`, the zero-shot LM) is **not** the instruct-tuned variant;
feeding it instruction text would not reliably produce the intended emotion. Real
emotion steering needs either the CosyVoice2-Instruct weights or our own
attribute-conditioning training (DESIGN Phase 6). **Flagged as needs-training** —
not implemented here, to avoid a knob that does not actually change the audio.

## Implemented in this slice

* `markup` — typed paralinguistic markup → text (point markers + `<strong>` span).
* `render::time_scale_mel` — pitch-preserving render-time rate transform.
* `serve::synth::Synthesizer::synthesize_with_rate` — additive synth hook applying
  the rate to the generated mel.

## Deferred (the fuller differentiator set)

* **Emotion steering** — instruct checkpoint or attribute-conditioning training;
  then a text-prompt + intensity-scale API (DESIGN T3.6).
* **Pitch/F0 contour at render time** — `pitch.rs`/`contour.rs` edit the plan's
  `pitch_hz`, but nothing yet *drives the flow/vocoder F0 from the plan*. The
  HiFT F0 predictor is internal; exposing an F0 override seam is the next concrete
  audio-affecting pitch knob (DESIGN T3.3).
* **Per-phoneme duration via the length regulator** — drive the token→mel
  expansion per token from an edited plan (DESIGN T3.2/T3.9), not just a uniform
  utterance stretch.
* **Phoneme-level edit round-trip** — `roundtrip.rs` round-trips the plan JSON;
  the edited-plan-→-rendered-audio loop (T3.10) needs the pitch/duration render
  seams above.

## What later testing should cover

* **Paralinguistic effect**: A/B that each marker changes the generated speech
  tokens (and audibly the audio) vs the unmarked text; that markers are robust to
  position; that an unmarked utterance is unaffected.
* **Rate effect**: duration scales ≈ `1/rate` (monotone across e.g. 0.7/1.0/1.3),
  `rate == 1.0` ≈ default-speed audio, and pitch is preserved (F0 estimate stable
  across rates) — the pitch-preservation claim is the load-bearing one and needs a
  real F0 measurement, deferred to perceptual/eval, not unit-gated.
* **Emotion** (once trained): perceptual A/B for intended emotion + monotone
  intensity (DESIGN T3.6 AC).
