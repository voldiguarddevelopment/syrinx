# Chatterbox Turbo — port scope

Written 2026-09-12. **Nothing is adopted by this document.** It records verified upstream
facts, what the port would cost, and what it would buy, so the scope decision can be made
on evidence. Qwen3-TTS remains the TTS path and the fallback throughout.

## Why now

On 2026-09-08 we measured that **Qwen's instruct channel cannot produce a laugh**
(`renders/2026-09-08-event-induction/`): four phrasings, three sentences, n=8, no duration
change, and the laugh instructions moved the audio *less* than a delivery-neutral sham. The
conclusion was that `[laughs]` is unreachable on the shipping path — with one recorded
caveat: *"a backend with a real event channel would answer differently by construction."*

Chatterbox Turbo is documented as having exactly that channel. That turns a logged candidate
into a specific question.

## Verified upstream facts

Read from the model card and repository files, not recalled.

| | |
|---|---|
| licence | **MIT** — `license: mit` in the card's YAML front matter |
| size | 350M params, **English only** |
| sample rate | 32 kHz (`sample_rate: 32000`) |
| voice cloning | zero-shot from a reference clip |
| watermark | **PerTh, always on** — every generated file is watermarked by the model |

### The paralinguistic tag set — all 19, enumerated

`CONTROL_SURVEY.md` recorded this as unverifiable upstream ("`[cough]`, `[laugh]`,
`[chuckle]`, and more"). It is enumerated in `added_tokens.json`, contiguous ids
50257–50275 immediately after the 50257-entry GPT-2 base vocabulary:

| kind | tags |
|---|---|
| emotion | `[angry]` `[fear]` `[surprised]` `[happy]` `[sarcastic]` `[crying]` |
| style | `[whispering]` `[dramatic]` `[narration]` `[advertisement]` |
| event | `[laugh]` `[chuckle]` `[sigh]` `[gasp]` `[cough]` `[clear throat]` `[groan]` `[sniff]` `[shush]` |

**The native syntax is literally `[tag]`** — the same bracket form `syrinx-cue` parses. That
is a coincidence of convention, not a licence to pass cue text through: lowering must still
map a canonical vocab id to the backend's own spelling, exactly as the `fish_s2` column
does. `[whispering]` is not `[whisper]`, `[fear]` is not `[afraid]`, and `[gasp]` is our
`quick_breath`. A pass-through would ship the wrong token or an unknown one.

### Architecture, from `t3_turbo_v1.yaml`

- **T3** — `llama_config_name: Llama_520M`, 30 layers, 16 heads, 1024 channels. Text tokens
  (GPT-2 BPE, `text_tokens_dict_size: 50276` = 50257 base + 19 tags) → speech tokens
  (`speech_tokens_dict_size: 6563`, tortoise-style, start 6561 / stop 6562).
- **s3gen** — speech tokens → mel → waveform. **Distilled to a single step**
  (`s3gen_meanflow.safetensors`), down from 10.
- **ve** — voice encoder, 5.7 MB, `speaker_embed_size: 256`.

Weights: `t3_turbo_v1.safetensors` 1.9 GB, `s3gen*.safetensors` ~1.06 GB each, `ve` 5.7 MB.

## What is reusable in-tree — the port is not green-field

The card acknowledges **CosyVoice, HiFT-GAN, S3Tokenizer and Llama 3**, and this workspace
already has ports of that lineage:

| Chatterbox component | in-tree precedent |
|---|---|
| T3 (Llama-family transformer, RoPE, KV cache) | `syrinx-qwen`'s talker — same family, same problems solved |
| s3gen (flow-matching token→mel) | `syrinx-acoustic` — built for CosyVoice2/3 |
| HiFT-style vocoder | `syrinx-vocoder` — same |
| voice encoder | `syrinx-speaker` |
| GPT-2 BPE tokenizer | standard; `vocab.json` + `merges.txt` ship with the model |

The deprecated CosyVoice work is the reason this is tractable. That is the second time
keeping deprecated code in-tree has paid — the first was the Fish codec defects.

## What it buys, and what it costs

**Buys:** a real event channel (`[laugh]` and eight more), MIT with no attribution burden,
350M params (a third of Qwen 1.7B), one-step decoding, 32 kHz.

**Costs:** English only — the multilingual Chatterbox is a *different* 500M checkpoint
**without** the tags, so tags and 23 languages cannot be had from one model. A second
backend to keep parity-anchored. And an always-on third-party watermark, which interacts
with `syrinx-serve`'s own watermarking obligation and needs a decision rather than a
default.

## Phasing

**Phase 0 — model-free, gateable now, no GPU.** Licence and capability records; the
`chatterbox_turbo` spelling column in `vocab.toml` and a `caps.toml` row; a `BackendId`;
crate skeleton with config parsing as pure functions of the shipped YAML/JSON; the
tokenizer contract and the cue → native tag → token-id round trip. All frozen-test gated
against files that can be downloaded without the 4 GB of weights.

**Phase 1 — needs weights, blocked on GPU availability.** Reference dump against the
upstream Python (`pip install chatterbox-tts`), then component-by-component parity: T3
forward, s3gen, ve. The discipline that caught the missing `silu` and both Fish codec
defects — anchor against the reference's own forward pass, never a reimplementation.

**Phase 2 — the question that motivated this.** Does `[laugh]` actually produce a laugh?
The same measurement the Qwen negative used, with the sham arm, on a backend that claims
the channel. This is the payoff and it is one experiment.

## Decision this document does not make

Whether to adopt it at all. Qwen stays the TTS path; this is a candidate second family, and
Phase 0 is cheap enough to be worth doing before the question is settled — it is data and
documents, and none of it commits the project to a port.
