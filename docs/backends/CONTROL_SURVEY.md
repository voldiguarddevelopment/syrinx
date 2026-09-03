# Backend control-surface survey (C0.1)

Verification date for every row: **2026-09-03**. Each row was checked against the
backend's current upstream source, model card, or the checkpoint shipped on this box —
**not** against the table in `docs/upgrades/SYRINX_UPGRADE_expressive_control.md` §1,
which this survey exists to audit. Discrepancies found against that table are listed in
§3 and are inputs to ADR-0001.

Method note: where a claim could be settled from code rather than prose, it was. A model
card describes intent; a tokenizer's `additional_special_tokens` list is the contract.

## 1. Backends Syrinx drives today

These are the backends that exist in this workspace and therefore need a `caps.toml`
under C2.1.

| Backend | Crate | Native control surface | Granularity | Verified from |
|---|---|---|---|---|
| **Fish S2-pro** | `syrinx-fish` | Inline `[free text]`. **Open vocabulary** — the card states "15,000+ unique tags supported" and gives free-form examples (`[whisper in small voice]`, `[professional broadcast tone]`). Not a fixed set. | **Word** | Shipped card `~/models/s2-pro/README.md`; [HF](https://huggingface.co/fishaudio/s2-pro); [tech report](https://huggingface.co/papers/2603.08823) |
| **Fish S1-mini** | `syrinx-fish` | Inline `(tag)`. **Closed set of 65**: 50 emotional + 5 tone + 10 special markers, enumerated verbatim in the card. | **Word** | Shipped card `~/models/openaudio-s1-mini/README.md`; [HF](https://huggingface.co/fishaudio/s1-mini) |
| **Qwen3-TTS `*-CustomVoice`** | `syrinx-qwen` | Natural-language `instruct` string + one of 9 preset `speaker` timbres. **No inline tag syntax exists.** | **Utterance** | `generate_custom_voice(text, speaker, language, instruct)` in installed `qwen_tts`; [HF](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) |
| **Qwen3-TTS `*-VoiceDesign`** | `syrinx-qwen` | Natural-language `instruct` describing the voice. No inline tags. | **Utterance** | `generate_voice_design(text, instruct, language)` in installed `qwen_tts`; [HF](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) |
| **Qwen3-TTS `*-Base`** | `syrinx-qwen` | **No expressive control at all.** Clones from `ref_audio`; the API takes no `instruct` parameter. | — | `generate_voice_clone(text, language, ref_audio, ref_text)` in installed `qwen_tts`; [HF](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) |
| **CosyVoice 2** | `syrinx-lm` + `syrinx-acoustic` + `syrinx-vocoder` (deprecated) | Instruct prefix + **closed inline set**: `[breath]` `[noise]` `[laughter]` `[cough]` `[clucking]` `[accent]` `[quick_breath]`, plus **span** forms `<strong>…</strong>` and `<laughter>…</laughter>`, and `<\|endofprompt\|>`. | **Utterance + point events + spans** | [`cosyvoice/tokenizer/tokenizer.py` `additional_special_tokens`](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/tokenizer/tokenizer.py) |
| **CosyVoice 3** | as above (deprecated) | Same inline set; instruct is the primary surface. `<\|endofprompt\|>` is required for all CV3 inference. | **Utterance + point events + spans** | [tokenizer.py](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/tokenizer/tokenizer.py); in-tree `crates/syrinx-serve/src/synth_cv3` |

## 2. Backends named in the spec that Syrinx does NOT drive

Surveyed because §1 of the spec lists them and the IR must not preclude them. **None of
these has a crate, and none needs a `caps.toml` until it does.**

| Backend | Native control surface | Granularity | License | Verified from |
|---|---|---|---|---|
| **Chatterbox** (base multilingual) | `exaggeration` (0.5 default) and `cfg`/`cfg_weight` (0.5 default). **No inline tags documented.** | **Utterance** | MIT | [HF card](https://huggingface.co/ResembleAI/chatterbox) |
| **Chatterbox Turbo / Nano** | Same scalars **plus native inline paralinguistic tags** — `[cough]`, `[laugh]`, `[chuckle]`, "and more"; the full set is not enumerated upstream. 350M params. | **Utterance + point events** | MIT | [HF card](https://huggingface.co/ResembleAI/chatterbox-turbo); [Resemble docs](https://www.resemble.ai/learn/models/chatterbox-turbo) |
| **Step-Audio-EditX** | Rich inline bracket tags: ~14 emotions (`[angry]`, `[happy]`, …), 30+ speaking styles (`[whisper]`, `[serious]`, `[news]`, …), paralinguistics (`[sigh]`, `[laugh]`, `[cough]`, `[breath]`, `[uhm]`, `[Surprise-oh]`, `[Question-ei]`). Demonstrated mid-sentence. | **Word / phrase** | Apache-2.0 | [GitHub](https://github.com/stepfun-ai/Step-Audio-EditX) |
| **MeloTTS** | `speaker_id` plus **four scalars**: `speed` (1.0), `sdp_ratio` (0.2), `noise_scale` (0.6), `noise_scale_w` (0.8). No tag syntax. The latter three are VITS variance knobs, not emotion controls. | **Utterance** | MIT | [`melo/api.py::tts_to_file`](https://github.com/myshell-ai/MeloTTS/blob/main/melo/api.py) |
| **Kokoro** | `voice` (speaker) and `speed` only. No emotion/style/paralinguistic markup. | **Utterance** | Apache-2.0 | [GitHub](https://github.com/hexgrad/kokoro) |

## 3. Corrections to the spec's §1 table

Recorded here because the spec instructs that this survey supersedes it.

1. **Chatterbox is two different capability classes, not one row.** The base multilingual
   model documents *no* tag set; tags are native to **Turbo/Nano** only. The spec's single
   row ("scalars, small paralinguistic tag set") is true of Turbo and false of base. A
   `caps.toml` must therefore be per **checkpoint variant**, not per model family.
2. **Same problem inside Qwen3-TTS.** `Base` has *no* expressive control, while
   `CustomVoice`/`VoiceDesign` take an instruct string. The spec's single "Qwen3-TTS" row
   hides a variant that cannot honour any cue. Additionally, measured on this box on
   2026-09-01: **`0.6B-CustomVoice` silently discards `instruct`** (`if
   self.model.tts_model_size in "0b6": instruct = None`), so it advertises a capability it
   does not apply. `caps.toml` must record this or the lowering report will lie.
3. **MeloTTS has four scalars, not "speed only."** `sdp_ratio` / `noise_scale` /
   `noise_scale_w` are additional prosody-variance knobs.
4. **CosyVoice's inline set is closed and includes span forms.** `<strong>…</strong>` and
   `<laughter>…</laughter>` are span-scoped, not point events — directly relevant to the
   spec's `[emphasis]word[/emphasis]` design, which has a native analogue here.
5. **Fish S1's set is exactly 65 tags** and is enumerated in the card; the spec says only
   "fixed `(tag)` set". The concrete list is what a `Inline::Closed` manifest needs.
6. Unverifiable upstream: the **complete** Chatterbox Turbo tag set ("and more"), and
   whether Alibaba intends Qwen's mistral-patched pre-tokenizer regex. Both are recorded
   as open questions rather than assumed.
