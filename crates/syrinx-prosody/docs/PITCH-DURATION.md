# Editable pitch + duration on CosyVoice2 — feasibility & the render plan

This note is the design half of Syrinx's **editable-prosody differentiator**: what
pitch and duration control is *faithfully achievable* on the CosyVoice2 acoustic
stack (as reimplemented across `syrinx-{acoustic,vocoder,serve}`) **without any
retraining**, and the `RenderPlan` API that wires the achievable subset into the
synthesizer. It is descriptive of the code as built, not aspirational. It extends
`CONTROL-SURFACES.md` (which covered markers + the uniform rate knob) with the
**pitch** axis and **per-region** duration.

## The two places pitch + duration live in the pipeline

The synthesizer (`syrinx_serve::synth::Synthesizer`) ends in:

```
flow.forward_zero_shot(...) -> mel [1, 80, T]      # spectral envelope per frame
f0 = hift.f0_predict(mel)    -> [1, T]             # one F0 (Hz) per mel frame
source = sine(f0) -> STFT    -> s_stft [1, 18, T]  # harmonic excitation
audio = hift.decode(mel, s_stft)                   # NSF: source (pitch) + filter (mel)
```

The HiFT vocoder is a **Neural Source Filter** (HiFTNet / iSTFTNet): the *source*
branch carries the harmonic excitation (pitch), and the *mel* carries the spectral
envelope (timbre/formants + duration). That split is what makes faithful,
training-free control possible — but only along the seams the split actually
exposes.

### Duration — fully faithful, no training

The acoustic stack fixes a token→mel ratio of 2, and the vocoder maps a **fixed hop
per mel frame**, so the mel's *frame count* sets the duration and the per-frame
spectral shape (and hence pitch) is independent of it. Time-scaling the mel along
its frame axis by `1/rate` is therefore an exact, pitch-preserving duration control
(the classic length-regulator move). This was already proven for the *global* knob
(`render::time_scale_mel` + `Synthesizer::synthesize_with_rate`).

* **Global rate** — faithful. (existing)
* **Per-region rate** — faithful, and *added here*: time-scale only the chosen
  frame ranges, concatenating the rest unchanged. Variable-rate time-warp; pitch is
  preserved frame-for-frame exactly as in the global case.

The honest limit (unchanged from `CONTROL-SURFACES.md`): "region" here means a
**generated-mel frame range**, not a word or phoneme. CosyVoice2 exposes no
token→audio alignment, so we cannot say "slow down *this word*" — only "slow down
frames `[a, b)`". Mapping words→frames needs an aligner the base does not provide
(see *What needs alignment / training* below).

### Pitch — faithful via the F0 source; lower-fidelity via mel-bin shift

There are exactly two training-free levers on pitch, and they are **not** equal:

1. **Scale the F0 fed to the vocoder (the faithful lever — the default).** Multiply
   the predicted per-frame F0 by `2^(semitones/12)` before building the sine source.
   Because the NSF source carries the harmonic comb and the mel carries the
   envelope, this raises/lowers the *pitch* while leaving the mel-derived
   **formants/timbre intact** — i.e. it is a *formant-preserving* pitch shift, the
   good kind. This is the one wired by default in `synthesize_with_plan`, both
   globally and per region.

   Honest caveats, with specifics:
   * The mel is **not** a pure envelope. At 24 kHz / 80 bins / hop 480 it still
     encodes some harmonic fine structure, and `f0_predict` *reads the mel*. So
     when the source is retuned but the mel is not, the two disagree slightly; the
     vocoder leans on the source for voicing, so moderate shifts (roughly **±5–6
     semitones**) sound like clean pitch changes, while large shifts (≳ an octave)
     get progressively rougher as residual mel harmonics fight the retuned source.
     This is a genuine quality ceiling, not a tunable bug — closing it needs joint
     mel+source resynthesis or retraining.
   * It rides on the **deterministic smoke source** (zero-phase, noise-free), the
     same non-parity source `synthesize_with_rate` uses. A pinned/parity `s_stft`
     is length- and pitch-locked to the unscaled mel and cannot describe a retuned
     one, so `synthesize_with_plan` always rebuilds the source (pinned `s_stft` is
     ignored, exactly like the rate path).

   **EMPIRICAL UPDATE (measured 2026-06-26 — corrects the optimism above).** On
   full-generation audio, `yin` (gold-standard F0) shows the rendered pitch shift is
   *far weaker* than the "±5–6 semitones sound clean" estimate: a **−4-semitone**
   request moved the median F0 ≈ **0 Hz** (230→230) and a **+4** request only
   ≈ **1.4 semitones** (230→249), not 4. The HiFT/NSF **mel filter dominates the
   perceived pitch** much more than expected, so the F0-source lever is **not an
   effective standalone pitch knob** training-free — it would need joint mel+source
   resynthesis or retraining. The lever still *applies* and measurably alters the
   signal (the `RenderPlan` + per-region machinery are correct and unit-tested); it is
   the *perceptual transfer* that is weak. **Duration/rate (below) is unaffected and
   remains faithful** — that is the solid prosody differentiator today.

2. **Shift mel energy along the mel-bin (frequency) axis (lower fidelity — opt-in).**
   Warping the mel up/down the bin axis by the pitch ratio moves the *whole spectral
   envelope*, which **shifts the formants too** — the "chipmunk / Darth Vader"
   artifact — because it cannot separate source from filter. It also only
   approximates a constant frequency ratio (the mel axis is quasi-log, so a constant
   bin offset is not a constant ratio; we warp by ratio in bin space as a documented
   approximation). It is provided (`shift_mel_bins`, and the `mel_envelope_shift`
   flag) for completeness and A/B, **off by default**, and is honestly worse than
   the F0 lever for pitch alone. Its legitimate use is deliberate *timbre* change,
   not faithful pitch.

## What needs alignment / training (not built — flagged honestly)

* **Word- / phoneme-level pitch or duration** — needs a token→frame aligner
  CosyVoice2 does not expose. The per-region API operates on **frame ranges**; a
  caller who wants "this word" must supply the frame range themselves (or build an
  aligner). No fake word knob is offered.
* **Octave-plus pitch shifts at full quality** — needs joint mel+F0 resynthesis or
  retraining (see caveat 1 above).
* **Emotion / style** — still the instruct-checkpoint / attribute-conditioning item
  from `CONTROL-SURFACES.md §3`; unrelated to this slice.

## What this slice implements

* `render_plan::RenderPlan` — a typed, JSON-round-tripping render-level plan:
  `global_rate`, `global_pitch_semitones`, an opt-in `mel_envelope_shift`, and a
  list of `Region { start_frame, end_frame, rate?, pitch_semitones? }` frame-range
  overrides (last region wins on overlap). Validated against the mel frame count.
* `render_plan::RenderPlan::apply(mel)` — the pure (Candle-free) transform: a
  variable-rate time-warp of the mel frame axis (global + per-region rate) that
  simultaneously resamples a per-frame **F0-multiplier** profile (global + per-region
  semitone shift), plus the opt-in mel-bin envelope shift. Returns the new mel grid
  and the per-output-frame F0 multiplier.
* `Synthesizer::synthesize_with_plan` — additive synth hook: flow → mel →
  `plan.apply` → `f0_predict` × the F0 multiplier → rebuild source → `decode`. The
  existing `synthesize` / `synthesize_with_rate` are untouched.

## What later testing should cover

* **Pitch effect** (the load-bearing claim): estimated F0 of the rendered audio
  rises at `+semitones`, falls at `-semitones`, monotone across e.g. `-4 / 0 / +4`,
  and timbre stays recognizably the same voice (formant preservation) — a real F0 +
  formant measurement, only smoke-asserted here.
* **Per-region duration**: total duration tracks the per-region `1/rate` integral;
  non-region frames are bit-unchanged.
* **Quality ceiling**: characterize where F0-source pitch shift audibly degrades
  (the ±octave roughness above) to set the documented usable range.
