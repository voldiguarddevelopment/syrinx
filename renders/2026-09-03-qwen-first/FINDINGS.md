# Qwen3-TTS port — first end-to-end renders (2026-09-03)

The first synthesis this port has ever produced, plus a check of the cue layer against
all three Qwen capability tiers. GPU 0 (RTX 5070), bf16, 24 kHz out, `seed = 0`.
Transcripts from the native Rust Whisper oracle (`syrinx stt`, whisper-base), language
forced to `en`.

## 1. The port works

| render | checkpoint | instruct | frames | audio | WER |
|---|---|---|---|---|---|
| `06b-cv-plain` | 0.6B-CustomVoice | — | 34 | 2.72 s | **0.000** |
| `17b-cv-plain` | 1.7B-CustomVoice | — | 35 | 2.80 s | **0.000** |
| `17b-cv-cued` | 1.7B-CustomVoice | "Say this in a soft whisper" | 61 | 4.88 s | 0.750 |
| `17b-vd-cued` | 1.7B-VoiceDesign | "Say this in a soft whisper" | 98 | 7.84 s | 1.000 |
| `06b-cv-cued` | 0.6B-CustomVoice | "Say this in a soft whisper" | 34 | 2.72 s | (identical to plain) |

Text: `Come closer, I have something to tell you.`

Both **plain** paths transcribe perfectly (WER 0.000). Talker load is ~1.3 s and a short
utterance renders in ~4 s on the 1.7B. The pure-Rust chain — tokenizer → talker → code
predictor → split RVQ → codec decoder → WAV — is functional end to end.

## 2. The cue layer is correct at every tier

`caps.toml` claims three different behaviours across the family, and each was confirmed
against the running model rather than by reading the table:

- **Base (0.6B / 1.7B)** — `inline=none`, everything `unsupported`. Every cue is dropped
  with "backend has no channel for this control". Not rendered here: the clone path has no
  driver (nothing in-tree builds an x-vector from a WAV), so `-Base` cannot synthesize yet.
- **0.6B-CustomVoice** — `instruct = accepted`. Cue dropped with the *distinct* reason
  "backend accepts this control but does not act on it". **Proven, not assumed:** the plain
  and cued renders are `md5 24f7866572c0082b3c2ba123d0feffcf` — **bit-identical**. The
  instruction provably never reached generation. This is exactly the per-checkpoint
  distinction that motivated capability rows being per-checkpoint rather than per-crate.
- **1.7B-CustomVoice / VoiceDesign** — `instruct = honored`. The cue becomes an
  utterance-scoped instruction and the audio changes substantially.

Sub-utterance splitting (C2.3) also works. Two conflicting cues on one line:

    [happy] What a wonderful morning. [sad] But then the letter arrived.

lower to **two** requests, split on a word boundary, each with its own instruction
("Speak in a happy, cheerful tone" / "Speak in a sad, sorrowful tone").

The hard invariant holds: no bracket text appears in any transcript.

## 3. DEFECT — an instruct makes the 1.7B talker repeat itself

Reproducible, systematic, and **specific to the instruct path**. The plain path is clean;
every instruction over-generates against a 35-frame baseline:

| instruct | frames | transcript |
|---|---|---|
| *(none)* | 35 | "Come closer, I have something to tell you." |
| "Speak clearly" | 54 | "Come to...come, Towsr. I have something to tell you." |
| "Whisper" | 65 | "Come, Kelser. I have something to tell you. I have something to tell you." |
| "Say this in a soft whisper" | 61 | "Come, Couser. I have something to tell you. Have something to tell you." |
| "Speak in a happy, cheerful tone" | 122 | the sentence, **five times** |

It is repetition, not a slower delivery — the transcripts show the target text emitted
two to five times, and the frame counts scale with it. Instruction length is not the
driver ("Whisper" is one word and still doubles).

**Ruled out:**

- *Sampler defaults.* `SamplingParams::talker()` is already the published
  `generation_config.json` — temperature 0.9, top_p 1.0, top_k 50, repetition_penalty
  1.05. A missing repetition penalty would have been the obvious culprit; it is present.
- *Text duplicated in the prompt.* Adding `--instruct "Whisper"` moves the prompt from 21
  to 28 steps — exactly the `<|im_start|>user\n…<|im_end|>\n` block and nothing else.
- *ASR hallucination.* Frame counts (35 → 61/65/98/122) confirm the audio really is two
  to three times longer.

**Not yet settled:** whether `assemble_text_mode` puts the instruct block in the right
place. It is prepended as its own complete `user` turn before the prefix; the reference
may instead fold it into the same turn as the target text, or use a different role. That
cannot be decided from the checkpoint — `tokenizer_config.json` carries no chat template
and the model card only shows the Python call, not the rendered prompt.

**What would settle it:** the Qwen reference implementation, which is not vendored and has
no `gen-fish-ref.py` equivalent. This is the same gap `QWEN_PORT_STATUS.md` records as
"no Python reference dump for Qwen, so talker/code-predictor parity cannot be gated at
all yet" — and it now has a concrete symptom to chase rather than a hypothetical one.

## 4. RESOLVED — the cause was a dropped activation in `text_projection`

Chased with the reference (`github.com/QwenLM/Qwen3-TTS` at `022e286`, read only — the
existing `~/.venvs/qwen` already had torch 2.13 and the pinned transformers 4.57.3).

**The reference does not repeat.** Same checkpoint, text, speaker and instructions:
3.04 s plain, 3.12 s "Whisper", 3.20 s "Say this in a soft whisper", all transcribing
cleanly. Against our 2.80 s -> 5.2 s / 4.88 s. So this was a port bug, not model
behaviour.

Everything structural then checked out and was eliminated in turn: the instruct template
is byte-identical (`_build_instruct_text` is exactly `<|im_start|>user\n{instruct}<|im_end|>\n`),
the instruct embeds are prepended in the same position with no codec addend, the token
ids match exactly (`[151644, 872, 198, 1639, 27370, 151645, 198]`), the prompt geometry
matches (21 -> 28 positions on both sides, trailing text `(1,1,2048)`), `suppress_tokens`
is implemented, and the sampler defaults are already the published `generation_config`.

Diffing the realized prompt embeddings step by step is what found it:

    before: max abs diff 0.781   norms 1.3x-1.9x the reference's, EVERY step
    after:  max abs diff 0.0     (f32, bit-exact)

`Talker::embed_text` implemented `text_projection` as `linear_fc2(linear_fc1(x))`. The
reference's `Qwen3TTSTalkerResizeMLP.forward` is `linear_fc2(act_fn(linear_fc1(x)))` with
`hidden_act = silu`. **The activation was missing** — which collapses the projection to a
composition of two linear maps, i.e. a plain linear map. The old doc comment asserted, in
prose, that "the reference applies no activation between them". It does.

Restoring `silu` fixes it. Frame counts for the same four cases go
35 / 65 / 61 / 122 -> **34 / 38 / 41 / 34**, and all four now transcribe at **WER 0.000**,
instructions included.

Why nothing caught it: the error was large enough to corrupt behaviour but small enough
to sound fine. Plain synthesis stayed perfectly intelligible (WER 0.000 throughout), so
every self-consistent Rust test passed. Only the reference could settle it — which is
exactly the gap that now has a gate: `scripts/gen-qwen-ref.py` dumps the reference's
`inputs_embeds` in f32 and `tests/real_qwen_prompt_parity.rs` compares against it,
per-step, with a dtype-aware tolerance (f32 measured 0.00000, bf16 0.0201, both far under
the bug's 0.78). Verified to fail without the fix.

## 5. Full-stack parity — the whole chain is now anchored to the reference

The prompt gate only covered the first of four stages, and the `silu` bug had just shown
that an error can be numerically small, audibly invisible and still wreck behaviour. So
every remaining stage got its own deterministic anchor
(`scripts/gen-qwen-ref.py` -> `tests/real_qwen_stack_parity.rs`):

| anchor | what it gates | max abs diff |
|---|---|---|
| `prompt.{plain,instruct}` | embedding tables, projections, assembly | **0.00000** |
| `talker.prefill_logits` | attention, RoPE, norms, codec head | **0.00003** |
| `predictor.logits` | the fast-AR stack, fed the reference's own input | **0.00003** |
| `codec.wav` | RVQ + decoder, no LM in the path | **0.000022** |
| `codec_edge.wav` | same, every group at its last valid row | **0.000005** |

The predictor is deliberately fed the reference's captured `inputs_embeds` rather than one
derived here: its real input depends on the sampled group-0 code, so deriving it would
make the test depend on sampling and stop being a parity check.

### The codec "failure" that wasn't

The codec anchor first came out at **0.031** max abs — 15x its tolerance, and alarming for
a path with no language model in it. It was not a port bug. The fixture had been generated
on CUDA while the test runs the CPU parity path, and the reference decoding the same codes
on the two devices **disagrees with itself by 0.030827**. Against a CPU-generated
reference the port lands at 0.000022 with correlation 1.0000000.

The decoder is a deep conv stack, so accumulation order matters and cross-device drift
dominates any real signal. `gen-qwen-ref.py` therefore defaults to CPU/float32 — the same
rule the Fish port already follows ("CPU must stay f32 (parity)"). Generating on CUDA
would spend the entire error budget on the device and leave nothing to catch a fault with.

### Two smaller things worth knowing

- `decoder_config` advertises `codebook_size: 2048` beside `semantic_codebook_size: 4096`,
  which is the exact ambiguity behind the Fish codec's out-of-range bug. It does **not**
  apply here, and that was settled against the checkpoint rather than the config: every
  decoder table — `rvq_first` and all 15 `rvq_rest` layers — is `[2048, 256]`. The 4096
  belongs to the encoder path.
- The reference's `decode` clamps `min=0` and **nothing above**, so an out-of-range code
  is a CUDA device-side assert, not a clamp. Feeding 4095 in group 0 kills it. The
  contract is "codes must already be in range".

### Final renders, after the fix

All five at **WER 0.000**, and the capability claims still hold:

| render | frames | audio | WER |
|---|---|---|---|
| `final-06b-plain` | 47 | 3.76 s | 0.000 |
| `final-06b-cued` | 47 | 3.76 s | 0.000 (bit-identical to plain — instruct discarded, as caps says) |
| `final-17b-plain` | 34 | 2.72 s | 0.000 |
| `final-17b-cued` | 41 | 3.28 s | 0.000 (instruct honored, no repetition) |
| `final-17b-vd` | 35 | 2.80 s | 0.000 |
