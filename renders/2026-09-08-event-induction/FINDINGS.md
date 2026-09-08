# Qwen's instruct channel cannot produce a laugh

2026-09-08. `1.7B-CustomVoice`, GPU bf16, `serena`, n=8, 3 sentences x 6 arms = 144 renders.
Raw: `run.txt`. WAVs for every arm are beside this file — the decisive test is ears.

## The question

`[laughs]` and the other twelve event cues are `event = unsupported` on all five Qwen
checkpoints, so `pass_hoist` filters them before a render happens. Every backend that *does*
honour events (fish-s1-mini, fish-s2-pro, cosyvoice2/3) is deprecated.

But "the event channel is unsupported" is a narrower claim than "the model cannot laugh".
Qwen has one expressive channel — a natural-language instruction — and nobody had asked it
to laugh through that. Four phrasings, meaning four different things by "laugh".

## Result: no

| arm | Δ duration (s) | acoustic p vs plain |
|---|---|---|
| sham (control) | −0.19, −0.09, +0.13 | 0.0322, 0.2120, 0.2887 |
| "Laugh briefly, then speak" | −0.08, +0.03, +0.01 | 0.7506, 0.2222, 0.2149 |
| "Begin with a short laugh, then say the line" | +0.02, +0.03, +0.02 | 0.6760, 0.6567, 0.1192 |
| "Speak with laughter in your voice" | −0.03, −0.23, −0.05 | 0.8418, 0.0435, 0.1700 |
| "Speak as if you are trying not to laugh" | +0.04, −0.10, +0.10 | 0.6087, 0.1127, 0.3138 |

Bonferroni α = 0.05/24 = 0.00208. **Nothing comes near it.**

Three signals agree:

- **No duration increase.** A laugh costs half a second to a second and a half. The largest
  magnitude here is 0.23 s, and it is *negative*. Across twelve instructed cells the mean
  |Δ| is under 0.07 s — seed noise.
- **No acoustic change.** Nothing clears α, and most sit between 0.11 and 0.84.
- **WER is flat** (0.000–0.083, no systematic movement). A laugh is non-lexical audio the
  oracle must transcribe as something or skip; either perturbs WER, and neither happened.

**The sharpest reading is comparative.** The delivery-neutral sham moved the audio *more*
than every laugh instruction on two of three sentences (0.0322 vs 0.61–0.84 on s0). The
instructions are not weakly effective; they are less consequential than a meaningless
string of the same length.

## What this settles

**`[laughs]` is unreachable on the shipping path, not merely unwired.** `caps.toml`'s
`event = unsupported` is the whole truth for Qwen, not a wiring gap someone could close by
lowering the cue to prose instead of dropping it. The current behaviour — drop it, record
`Dropped { reason: Unsupported }`, speak the sentence cleanly — is correct, and the
alternative was worth ten minutes of GPU to rule out rather than assume.

It also **confirms the judge decision** taken on 2026-09-06. Audio-event detection was
SenseVoiceSmall's one unique capability over emotion2vec+, and it was passed over partly
on the argument that no shipping backend has an event channel to detect. That argument was
contingent on this experiment; it now holds on measurement rather than assumption.

## What it does not settle

- **Other phrasings exist.** Four were tried. A negative over four is not a proof over all,
  though the total absence of duration change makes "the right words would unlock it"
  unlikely: the model is not partially laughing.
- **Other checkpoints.** VoiceDesign's instruct designs the *voice* rather than the
  delivery, and was not tested here.
- **A backend with a real event channel** would answer differently by construction. That is
  a scope decision (`ResembleAI/chatterbox`, MIT, is the logged unadopted candidate), not a
  measurement.
