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
| watermark | PerTh — but **applied by upstream's Python over the finished waveform, not by the weights**. A Rust port therefore inherits *no* watermark. "Always on" is true of upstream and would silently become false of us; see `docs/LICENSES.md` |

### The paralinguistic tag set — all 19, enumerated

`CONTROL_SURVEY.md` recorded this as unverifiable upstream ("`[cough]`, `[laugh]`,
`[chuckle]`, and more"). It is enumerated in `added_tokens.json`, contiguous ids
50257–50275 immediately after the 50257-entry GPT-2 base vocabulary:

Grouping into emotion/style/event below is **ours, editorial** — `added_tokens.json` carries
no kind field and the card states no taxonomy. It is weakly corroborated by the events
occupying a contiguous id block (50267–50275) and by nothing else.

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

### Architecture — CORRECTED 2026-09-12, read from the checkpoint, not the YAML

**The first version of this section was wrong, and wrong in the way it warned against.**
It said *"T3 — `llama_config_name: Llama_520M`, 30 layers"*, quoting two fields that are
**dead**: `t3_turbo_v1.yaml` is a training-config superset for several models in Resemble's
stack, and this document said so two paragraphs earlier before repeating dead keys anyway.

The checkpoint settles it. A `safetensors` file begins with a u64 length and a JSON tensor
index, so a **29 KB HTTP range request** against the 1.9 GB file reads the complete
name/shape listing without downloading any weights:

```
tfmr.h.0 .. tfmr.h.23           -> 24 blocks, not 30
tfmr.h.0.attn.c_attn.weight     [1024, 3072]   fused QKV, GPT-2 Conv1D layout
tfmr.wpe.weight                 [8196, 1024]   LEARNED absolute positions
rotary / RoPE tensors           none
total                           478.9M params  (= 1.9 GB / 4, i.e. F32)
```

- **T3 is a 24-block GPT-2**, not a Llama. The live field is `gpt_transformer_type:
  gpt2-medium`; `llama_config_name` and `n_transformer_layers` are dead. Heads (16) and
  channels (1024) happen to be right. Text tokens (GPT-2 BPE, `text_tokens_dict_size:
  50276` = 50257 base + 19 tags) → speech tokens (`speech_tokens_dict_size: 6563`,
  tortoise-style, start 6561 / stop 6562).
- **"350M params" is not what ships.** T3 alone is 478.9M; s3gen 264.0M, meanflow 266.1M,
  ve 1.4M. The card's figure is unreconciled and this document no longer repeats it as fact.
- **Two unrelated speaker encoders**, not one: `ve.safetensors` (a 1.4M-param 3-layer LSTM,
  40 mel bins in, 256 out, feeding T3's `cond_enc.spkr_enc`) and s3gen's own
  `speaker_encoder.*` (937 tensors, CAMPPlus-shaped). A port assuming a single path is
  wrong before it starts. `ve_hidden_size: 768` is another dead field.
- `s3gen_meanflow` differs from `s3gen` by exactly **two** tensors.
- **s3gen** — speech tokens → mel → waveform. **Distilled to a single step**
  (`s3gen_meanflow.safetensors`), down from 10.
- **ve** — voice encoder, 5.7 MB, `speaker_embed_size: 256`.

Weights: `t3_turbo_v1.safetensors` 1.9 GB, `s3gen*.safetensors` ~1.06 GB each, `ve` 5.7 MB.

## What is reusable in-tree — the port is not green-field

The card acknowledges **CosyVoice, HiFT-GAN, S3Tokenizer and Llama 3**, and this workspace
already has ports of that lineage:

| Chatterbox component | in-tree precedent |
|---|---|
| T3 (**GPT-2**: learned positions, LayerNorm, GELU, Conv1D weights) | **weaker than first claimed.** `syrinx-qwen`'s talker is RoPE + RMSNorm + SwiGLU; almost none of that transfers. GPT-2 is the *simpler* shape, so this is still tractable — but as new work, not reuse. The Conv1D `[in, out]` layout is the transpose of what candle's `Linear` wants |
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
