# Qwen on the GPU, and three "blockers" that were not (2026-09-06)

Yesterday I reported four remaining Qwen items as "genuinely out of reach today rather
than merely undone". Challenged on it, three turned out to be cost or process constraints
I had imposed myself, and the fourth was stale. Written down because the failure mode —
relaying a subagent's cost decision as a capability limit — is worth not repeating.

| claimed | actually |
|---|---|
| VoiceDesign "wants the 1.7B-VoiceDesign checkpoint" | the checkpoint was already on disk, 4.3 GB. The agent's reason was "the 1.7B is off-limits for **cost** here" — my CPU-only rule |
| CUDA/bf16 "untested" | both GPUs idle, cudarc built, `--features "real cuda"` used all session. Untested because I told every agent "CPU only, I am holding the GPU" |
| multi-segment split "would double a 12-minute job" | CPU cost only |
| SIM-o "blocked-on-human" | not perceptual at all, and unblocked by our own speaker-encoder work (see §3) |

Genuinely blocked: **MOS** — "does it sound good" needs ears or a MOS-prediction model.

## 1. A real defect the challenge surfaced

`syrinx qwen --cuda` could never reach the GPU. `crates/syrinx-cli/Cargo.toml`'s `cuda`
feature read

```toml
cuda = ["real", "syrinx-serve/cuda", "syrinx-fish/cuda", "syrinx-stt/cuda"]
```

— no `syrinx-qwen/cuda`. When wiring the `qwen` subcommand I added the crate to the CLI's
`real` feature and forgot `cuda`, so the flag was accepted, the device probe failed, and it
silently fell back to CPU with a one-line warning that is easy to miss in a long log. Fixed.
The first VoiceDesign render in this directory was produced on CPU for exactly this reason,
before the fix.

## 2. What the GPU changes — the number behind "off-limits for cost"

Same input, same binary, `[happy] What a wonderful morning. [sad] But then the letter
arrived.` (two cue-driven segments), 1.7B-CustomVoice:

| device | wall clock |
|---|---|
| CUDA bf16 | **8.2 s** |
| CPU f32 | **486 s** |

**59x.** That is the entire content of every "too expensive" judgement in this session's
subagent reports. Nothing was impossible; everything was an hour instead of a sip.

Renders, all transcribing correctly against the Whisper oracle:

| render | path | frames | audio |
|---|---|---|---|
| `voicedesign.wav` | 1.7B-VoiceDesign, `--voice-design`, cue -> instruct | 35 | 2.80 s |
| `split-cuda.wav` | 1.7B-CustomVoice, 2 segments, GPU bf16 | 22 + 31 | 4.23 s |
| `split-cpu.wav` | same, CPU f32 | 22 + 29 | 4.07 s |

The two split renders differ (31 vs 29 frames on segment 1). That is bf16-vs-f32 sampling
divergence — the same phenomenon documented for Fish `drive_batch`, where a last-bit
difference flips a draw and the utterance diverges from there. Both are valid; they are not
the same audio.

## 3. SIM-o, which was never perceptual

`CLAUDE.md` groups SIM-o with MOS under blocked-on-human. That was right when written and
is not now. SIM-o is a **cosine between speaker embeddings** — entirely objective. It was
blocked because nothing in-tree could produce the embeddings; `SpeakerEncoder` now can, and
it is anchored to the reference at 1e-6 (2048-wide) and 6e-7 (1024-wide).

`syrinx_eval::qwen::speaker_similarity` + `crates/syrinx-eval/examples/simo.rs`. Measured
against `voice_en_10s.wav`, 1024-wide encoder from 0.6B-Base:

| clip | SIM-o | what it is |
|---|---|---|
| `voice_en.wav` | **0.997** | the SAME speaker, a different clip — the ceiling |
| `xvec-only.wav` | **0.985** | clone, x-vector only |
| `icl.wav` | **0.984** | clone, in-context |
| `voice_pl.wav` | **0.925** | a genuinely DIFFERENT real speaker — the floor |
| `voicedesign.wav` | 0.902 | not a clone: the instruction designed the voice |
| `whisper.wav` | 0.905 | not a clone: CustomVoice preset `serena` |

The clones land at 0.985 against a same-speaker ceiling of 0.997 and a different-speaker
floor of 0.925. That gap is the result: the clone path really does move the render toward
its reference.

**Read the gap, never the absolute value.** 0.90 looks like a high score and is in fact
*below* a different speaker — this encoder's cosine is compressed into a narrow band, so a
number without a floor measured on the same encoder means nothing. Note also the two
synthetic non-clone voices score *below* the different-speaker floor, which is probably a
synthetic-vs-real domain effect as much as an identity one.

**What this is not.** The encoder scoring the clone belongs to the same family that
produced it, so this measures "did the render land where the reference lands in Qwen's own
speaker space". It will catch a clone that ignored its reference; it is not an independent
verifier the way CosyVoice's CAM++ path is, and these numbers are not comparable with
CAM++ SIM-o from the literature. An independent speaker-verification model would be the
stronger arrangement and is still worth having.
