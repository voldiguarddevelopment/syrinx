# Findings from earlier runs

The audio and per-run directories for these were deleted (regenerable, and large). What
they established is kept here because it *configures* later runs.

## 2026-08-29 — s2-pro, 50 sentences, en/de/pl, emotion-tagged

Verified with faster-whisper large-v3. WER scored with the language forced, against the
prompt text with emotion tags stripped.

| lang | n  | median WER | mean WER | lang-ID | tag-leak |
|------|----|------------|----------|---------|----------|
| en   | 17 | 0.000      | 0.032    | 17/17   | 0        |
| de   | 17 | 0.000      | 0.037    | 17/17   | 0        |
| pl   | 16 | 0.100      | 0.155    | 11/16   | 0        |
| all  | 50 | 0.000      | 0.073    | 45/50   | **0**    |

- **Zero tag leakage in 50/50**, across all nine placements (leading, mid, trailing, wrap,
  multi, per_sentence, combined, special, neutral). No `[sad]` / `[excited]` / `[whispers]`
  was ever spoken aloud. Tag *consumption* is objectively clean; tag *following* is
  perceptual and not scored here.
- en/de are effectively solved at this scale, including 18-26 word conversational turns
  transcribing word-for-word.

## The Polish result — cross-lingual cloning, not Polish

Same 16 Polish sentences, rendered twice:

| Polish                     | median WER | mean WER | lang-ID |
|----------------------------|------------|----------|---------|
| cloned from the English ref| 0.100      | 0.155    | 11/16   |
| no reference (uncloned)    | **0.000**  | **0.075**| **14/16**|

Dropping the English reference halves the error and fixes 3 of 5 language-ID misses
(`small_pl_mid_sad_01`: 0.667 -> 0.000). **The model's Polish is fine; cloning it from an
English voice is what degrades it** — the reference clip's phoneme coverage bounds the
output, exactly as `README.md` warns.

**Consequence for later runs:** do not clone Polish from an English reference. Either
render Polish uncloned, or source a Polish reference clip.

## Throughput on NovaBox (measured, not estimated)

- **7.4x slower than realtime**, single GPU, bf16, `--batch-size 1` (16 clips, 69.2 s of
  audio, ~511 s wall).
- **Only cuda:1 is usable while the desktop is running.** cuda:0 holds ~1 GB of desktop
  VRAM; the 10.4 GB bf16 model plus CUDA context and KV cache does not fit the remainder
  and OOMs at load. Free that GPU to double throughput.
