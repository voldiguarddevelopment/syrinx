# C4.2′ first certification run — clean, and it says "no"

2026-09-09. `1.7B-CustomVoice`, GPU bf16, `serena`, **n=9** (20x headroom at α=0.00208),
6 pre-registered sentinels, 4 arms each. Raw: `run.txt`, `report.json`.

The gate ADR-0003 specified, run for the first time. Not enforcing: clause B has no
sentinel until one is earned, which ADR-0003 pre-registered as an acceptable outcome.

## All four controls hold

| clause | result |
|---|---|
| **A1** pipeline control | 0.6B renders **bit-identically** cued vs plain ✓ |
| **A2** A/A calibration | no violation (1 activation, budget 1) ✓ |
| **A3** sham | nothing activated — 0.0233 to 0.8067, against α=0.00208 ✓ |
| **C** WER veto | no regression ✓ |
| — | **0 cases `not_applicable`** — every sentinel measured |

## The verdict: no content activation

| case | cue/plain | sham/plain | **cue/sham** | A/A |
|---|---|---|---|---|
| happy-leading | 0.0683 | 0.4792 | 0.0598 | 0.0507 |
| **sad-mid** | 0.0038 | 0.0233 | **0.0001** | 0.6608 |
| angry-trailing | 0.8677 | 0.8067 | 0.8300 | 0.8011 |
| calm-mid | 0.1594 | 0.0233 | 0.0825 | 0.6608 |
| whisper-leading | 0.0042 | 0.4792 | 0.2068 | 0.0507 |
| **shout-mid** | 0.0182 | 0.0233 | **0.0000** | 0.6608 |

`content_activated` requires clearing **both** cue-vs-plain and cue-vs-sham. Nothing does,
so the gate reports none. That is the correct conservative verdict.

## What the gate's verdict omits, and it is the most interesting thing here

**Two cases separate from their sham decisively — `sad-mid` at p=0.0001 and `shout-mid` at
p=0.0000 — while their cue-vs-plain sits at 0.0038 and 0.0182.** The cue is *further from a
delivery-neutral instruction of the same length* than it is from no instruction at all.

That is not noise and it is not nothing. The natural reading is that the sham and the cue
move the audio in **different directions** from plain: a neutral "read this" flattens
delivery, a cue colours it, and the pair are furthest apart from each other. Both differ
moderately from plain; they differ from each other strongly.

If that reading is right, the conjunctive requirement may be measuring the wrong thing.
`cue vs plain` conflates "this cue has content" with "an instruction was present at all" —
which is precisely the confound the sham arm was introduced to remove. Requiring the cue to
*also* beat plain re-admits it.

**This is recorded as an observation, not acted on.** Changing what `content_activated`
means is an ADR-0003 amendment and a maintainer decision, and one run on six sentinels is
thin evidence for redefining a criterion. The data is in `report.json` either way.

## Power, and why n=9

The first clean run used n=8 on `min_n_for_alpha` — 13.4x headroom, below the 20x adopted
for the tuner the same day, with three cases within 1.2x of the bar. That is the regime
where power decides the verdict, so the runner was moved onto `min_n_for_headroom(α, 20)`
and re-run at n=9.

**The extra power did not rescue the borderline cases** — `sad-mid` went 0.0025 → 0.0038 and
`whisper-leading` 0.0023 → 0.0042. They are genuinely not at that level, which is a firmer
answer than the underpowered run could give. Worth having spent 30 minutes on.

## Two smaller things, recorded

- **A/A on the leading text is 0.0507**, a hair above nominal 0.05. It does not violate, but
  the plain arm on *"I really cannot believe what you just told me."* is closer to
  disagreeing with itself than any other text here. Worth watching, not acting on.
- **`calm-mid` carries wer 0.200** where every other case is 0.000. Well inside the 0.5
  veto, but it is the only case with real transcription error and the only one whose cue is
  `calm` — plausibly the oracle mishearing a quieter delivery.

## Getting here cost three runner defects

Each produced a plausible wrong answer, and none was visible from a green suite:
`segments.first()` mislabelled 4 of 6 as `not_applicable`; the A1 control OOM'd because the
1.7B was never freed; and A/A was counted per case rather than per text, manufacturing a
`CalibrationFailed` from one measurement counted twice. The first of those, once fixed,
immediately exposed a real defect in `syrinx-cue` — trailing cues silently inert (ADR-0005).
