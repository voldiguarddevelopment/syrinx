# Backend control-surface survey (C0.1)

Verification date for every row: **2026-09-03**, except the two **Chatterbox** rows in §2
and items 6–9 of §3, re-verified **2026-09-12** (see below and
`CHATTERBOX_PORT_SCOPE.md`). Each row was checked against the
backend's current upstream source, model card, or the checkpoint shipped on this box —
**not** against the table in `docs/upgrades/SYRINX_UPGRADE_expressive_control.md` §1,
which this survey exists to audit. Discrepancies found against that table are listed in
§3 and are inputs to ADR-0001.

Method note: where a claim could be settled from code rather than prose, it was. A model
card describes intent; the tokenizer's own token table is the contract. Read the whole
table, not one key: Chatterbox Turbo's 19 tags are **not** in `additional_special_tokens`
(the key is absent) — they are added tokens in `added_tokens.json` / `added_tokens_decoder`
carrying `"special": false`. A survey that grepped only for the 2026-09-03 wording of this
note would have concluded Turbo had no tags at all.

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
| **Chatterbox** (the `ResembleAI/chatterbox` repo: Multilingual V3 500M / 23 languages, an English-only 500M, and six single-language finetunes) | `exaggeration` (0.5 default) and `cfg`/`cfg_weight` (0.5 default). **No inline tags** — none documented on the card and none in the repo's tokenizer files. | **Utterance** | MIT | [HF card](https://huggingface.co/ResembleAI/chatterbox), re-read 2026-09-12 |
| **Chatterbox Turbo** | Same scalars **plus a closed set of 19 native inline paralinguistic tags**, natively spelled `[tag]`: `[advertisement]` `[angry]` `[chuckle]` `[clear throat]` `[cough]` `[crying]` `[dramatic]` `[fear]` `[gasp]` `[groan]` `[happy]` `[laugh]` `[narration]` `[sarcastic]` `[shush]` `[sigh]` `[sniff]` `[surprised]` `[whispering]`. Token ids **50257–50275**, contiguous, immediately after the 50257-entry GPT-2 base vocabulary (`text_tokens_dict_size: 50276` = 50257 + 19). English only. | **Utterance + point events** | MIT | The checkpoint's own `added_tokens.json`, `tokenizer_config.json` and `t3_turbo_v1.yaml` (`/data/models/chatterbox-turbo`, read 2026-09-12); [HF card](https://huggingface.co/ResembleAI/chatterbox-turbo) |
| **Step-Audio-EditX** | Rich inline bracket tags: ~14 emotions (`[angry]`, `[happy]`, …), 30+ speaking styles (`[whisper]`, `[serious]`, `[news]`, …), paralinguistics (`[sigh]`, `[laugh]`, `[cough]`, `[breath]`, `[uhm]`, `[Surprise-oh]`, `[Question-ei]`). Demonstrated mid-sentence. | **Word / phrase** | Apache-2.0 | [GitHub](https://github.com/stepfun-ai/Step-Audio-EditX) |
| **MeloTTS** | `speaker_id` plus **four scalars**: `speed` (1.0), `sdp_ratio` (0.2), `noise_scale` (0.6), `noise_scale_w` (0.8). No tag syntax. The latter three are VITS variance knobs, not emotion controls. | **Utterance** | MIT | [`melo/api.py::tts_to_file`](https://github.com/myshell-ai/MeloTTS/blob/main/melo/api.py) |
| **Kokoro** | `voice` (speaker) and `speed` only. No emotion/style/paralinguistic markup. | **Utterance** | Apache-2.0 | [GitHub](https://github.com/hexgrad/kokoro) |

## 3. Corrections to the spec's §1 table

Recorded here because the spec instructs that this survey supersedes it. Items 6–9 were
added on 2026-09-12 and correct **this survey's own 2026-09-03 rows** as well — a survey
that only ever corrected someone else's table would be the least trustworthy document in
the directory.

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
6. **RESOLVED 2026-09-12 — the Chatterbox Turbo tag set is enumerated, and it is 19.** The
   2026-09-03 row said `[cough]`, `[laugh]`, `[chuckle]`, "and more", and listed the
   complete set as unverifiable upstream. It was unverifiable *from the card* — the card
   still says "and more" — but not from the checkpoint: `added_tokens.json` lists all 19
   with their ids, and `t3_turbo_v1.yaml`'s `text_tokens_dict_size: 50276` independently
   corroborates the count (50257 GPT-2 base + 19). The §2 row now carries the enumeration
   and cites those files. What changed is the evidence, not the model: **we checked, where
   before we assumed**, and the hedge was correct at the time it was written.
   The grouping into emotion / style / event used by `CHATTERBOX_PORT_SCOPE.md` is **ours,
   not upstream's** — the JSON has no kind field. It is weakly corroborated by the id
   layout (the nine event tags occupy a contiguous block, 50267–50275) and nothing more.
   **Still unverifiable:** whether Alibaba intends Qwen's mistral-patched pre-tokenizer
   regex. That one remains an open question rather than an assumption.
7. **"Nano" was never verified and is not on the card.** The 2026-09-03 row read
   "Chatterbox Turbo / Nano". The Turbo card's model zoo names three models — Turbo (350M,
   English), Multilingual (500M, 23+), and Chatterbox (500M, English) — and no Nano. The
   row has been narrowed to Turbo, which is what the evidence covers. Nothing is claimed
   about a Nano checkpoint either way.
8. **`ResembleAI/chatterbox` is a repo, not a checkpoint.** It holds Multilingual V3
   (500M, 23 languages), an English-only 500M, and six single-language finetunes. Calling
   it "base multilingual" hid that, and it compounds §3.1: `caps.toml` must be keyed per
   **checkpoint variant**, and here the variants do not even live in separate repos.
9. **Turbo's "350M" is the card's figure and is not reconciled with the shipped files.**
   `t3_turbo_v1.yaml` names `llama_config_name: Llama_520M` and `t3_turbo_v1.safetensors`
   is 1.92 GB. Recorded as the card says it, not adopted as measured; a port will settle it
   by loading the weights, and nothing depends on it in the meantime.
