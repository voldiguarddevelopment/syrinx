# Model licence matrix

`CLAUDE.md` lists a "license-screen matrix doc" as a Phase 0 deliverable. It did not
exist; this is it. Written 2026-09-06, prompted by a direct question about whether a
tuned Qwen could be used commercially.

**Not legal advice.** This records what the licence files and model cards say, and the
structure of the questions they raise. A real commercial decision needs a lawyer.

## What is on this box

| model | used for | licence | commercial use |
|---|---|---|---|
| `Qwen3-TTS-12Hz-*` (all five checkpoints) | the Qwen TTS path | **Apache-2.0** | **yes** |
| `Qwen3-TTS-Tokenizer-12Hz` | Qwen codec (encode + decode) | **Apache-2.0** | **yes** |
| `whisper-base` | the native WER oracle (`syrinx-stt`) | **Apache-2.0** | **yes** |
| `s2-pro` | Fish TTS path | `fish-audio-research-license` ("other") | **no — research only** |
| `openaudio-s1-mini` | Fish TTS path | CC-BY-NC-SA-4.0 | **no** |
| `w2v2-msp-dim` (audEERING) | dimensional affect — **REJECTED 2026-09-06 on licence** | CC-BY-NC-SA-4.0 | **no** |
| `emotion2vec/emotion2vec_plus_large` | affect judge (**adopted 2026-09-06**): 9-class SER | FunASR Model Open Source License | **yes**, attribution required |
| `ehcalabres/wav2vec2-lg-xlsr-en-...` | affect judge (**superseded 2026-09-06**): RAVDESS 8-class | **Apache-2.0** | yes |
| `ResembleAI/chatterbox-turbo` | **candidate only — NOT adopted.** Metadata only on this box (4 config/tokenizer files, **no weights**) | **MIT** (card front matter) | **yes** |

### Datasets

| dataset | used for | licence | commercial |
|---|---|---|---|
| CREMA-D (180-clip probe subset) | determining the affect judge's head activation, and calibrating it cross-corpus | **ODbL-1.0** | **yes** |
| RAVDESS | **deliberately NOT downloaded** | CC-BY-NC-SA-4.0 | no |

RAVDESS is avoided twice over: it is non-commercial, *and* it is the affect judge's own
training set, so measuring the judge on it would report memorisation as accuracy. CREMA-D
is both permissively licensed and genuinely held out, which is why the judge's real
cross-corpus accuracy (0.394) could be established at all against the card's in-domain
claim of 0.822.

## The headline — acted on 2026-09-06

This matrix changed the project's direction the day it was written. `CLAUDE.md` had said
"**Fish Audio (`syrinx-fish`) is the TTS path**"; it now says Qwen, and Fish joins
CosyVoice as deprecated. The reasoning:

- **Every Fish checkpoint is research-only.** `s2-pro` ships under Fish Audio's own
  research licence, `openaudio-s1-mini` under CC-BY-NC-SA-4.0. Neither can be used in a
  commercial product. `crates/syrinx-fish/src/lib.rs` already says so at the crate level.
- **Every Qwen3-TTS checkpoint is Apache-2.0** — use, modify, distribute, sell, and
  fine-tune, with attribution and NOTICE preservation. Fine-tunes are yours.

So the two paths are not interchangeable: **Fish is the research path, Qwen is the
shippable one.** Nothing in the tree recorded that, which is exactly how a project ends up
having invested its effort in the unlicensable half. **Decision (2026-09-06): Qwen is the
TTS path; `syrinx-fish` is deprecated** — code and results stay, no new work, never a
shipping path.

### Candidate affect judges (researched 2026-09-06, see backends/AFFECT_JUDGES.md)

| model | licence | commercial | note |
|---|---|---|---|
| `FunAudioLLM/SenseVoiceSmall` | FunASR Model Open Source License | **yes**, attribution required | ASR + SER + audio-event detection in one pass. Licence read in full: grants use/copy/modify/share, requires attribution and model-name retention, does NOT restrict commercial use. Bespoke rather than OSI — wants a lawyer's glance, but it is not NonCommercial. |
| `Qwen/Qwen2-Audio-7B-Instruct` | **Apache-2.0** | yes | audio LLM; good development instrument, poor gate (see the research note) |
| `Qwen/Qwen2.5-Omni-7B`, `Qwen3-Omni-30B-A3B` | Apache-2.0 (tagged `other`, `license_name: apache-2.0`) | yes | heavier; same caveat |
| `emotion2vec/emotion2vec_plus_large` | FunASR licence | yes, attribution | dedicated affect representation |

### Candidate second family — Chatterbox (Resemble AI)

**Not adopted, and nothing here adopts it.** Qwen3-TTS remains the TTS path and the
fallback. This section exists to discharge `CLAUDE.md`'s standing rule — *"any new backend
is a licence question first; add its row to `docs/LICENSES.md` before building on it"* —
**before** anyone builds. Whether to take on a second family at all is a scope decision
under ADR-0001 §7, it is a human act, and it has not been made. Capability detail lives in
`backends/CONTROL_SURVEY.md`; cost and phasing in `backends/CHATTERBOX_PORT_SCOPE.md`.

**One row was not enough, because this is two checkpoints with different capabilities.**
The tag channel and the languages live in different weights and cannot be had from one
model:

| checkpoint | what it is | paralinguistic tags | licence | commercial |
|---|---|---|---|---|
| `ResembleAI/chatterbox-turbo` | **English only** (`language: [en]`), 32 kHz, decoder distilled to one step. Card says **350M**. | **19 native inline tags**, enumerated — see `backends/CONTROL_SURVEY.md` §2 | **MIT** | **yes** |
| `ResembleAI/chatterbox` | **500M**, **23 languages** (Multilingual V3), plus an English-only 500M and six single-language finetunes in the same repo. `exaggeration` / `cfg_weight` scalars. | **none** — no tag syntax documented or shipped | **MIT** | **yes** |

Read on 2026-09-12 from the two model cards and, for Turbo, from the checkpoint's own
tokenizer metadata now on this box at `/data/models/chatterbox-turbo` — `added_tokens.json`,
`tokenizer_config.json`, `special_tokens_map.json`, `t3_turbo_v1.yaml`. **The weights are
not downloaded** (~4 GB); every claim below is from a card or from those four files.

#### What the licence says

Both cards declare **`license: mit`** in their YAML front matter. MIT grants use, copy,
modify, merge, publish, distribute, sublicense and sell, conditioned only on carrying the
copyright notice and the permission notice. **No NonCommercial clause, no ShareAlike, no
NOTICE file to maintain — and no patent grant**, which is the one thing Apache-2.0 gives
that MIT does not. The inference code, `github.com/resemble-ai/chatterbox`, is MIT with a
LICENSE file checked in.

One thing to hand a lawyer rather than settle here: **neither Hugging Face weight repo
ships a LICENSE file.** The grant on the weights is the card's front-matter tag and nothing
else. That is how Hugging Face expects a licence to be declared and it is almost certainly
what Resemble AI intends, but it is thinner than checked-in text, and the code repo's
LICENSE covers the code rather than self-evidently the checkpoints. Contrast `s2-pro`,
where the restriction is a licence *file* inside the checkpoint and there is nothing left
to argue about. The asymmetry is worth noticing: we were willing to treat Fish's file as
binding, so we should not treat a tag as binding on a different standard of evidence.

#### What it means for us

- **It is the cleanest licence of any TTS family screened here** — cleaner than Apache-2.0
  on obligations, weaker on patents. That is a reason it *stays* a candidate, not a reason
  to move: Qwen is Apache-2.0, already ported, and anchored end to end. **No licence
  problem is driving this**, unlike the 2026-09-06 decision that displaced Fish.
- **The two checkpoints are a capability fork, not a version bump.** Anything built on
  Turbo's tag channel is English-only by construction, and anything built on the 23
  languages has no tag channel. A `caps.toml` keyed on "Chatterbox" would be wrong for one
  of them — precisely the defect `CONTROL_SURVEY.md` §3.1 already records, and a licence
  screen cannot fix it.
- **Every generated file is watermarked by default, and the watermark is not ours.** Both
  cards state that every audio file Chatterbox generates carries Resemble AI's PerTh
  (Perceptual Threshold) watermark. Under this file's own rule — anything that ends up *in*
  a shipped artifact inherits its lineage — audio we sold would carry a third party's mark.
  Three things that look like one and are not:
  1. **Licence: clean.** The watermarker is `github.com/resemble-ai/perth`, **MIT**, a
     separate package from the model. Nothing about it is restrictive.
  2. **What actually applies it: the Python package, not the weights.** The card documents
     watermarking as a property of upstream's generate path and demonstrates detection with
     an ordinary `import perth` over the finished waveform. It is post-processing, so **a
     Rust port would not inherit it** — it would emit unwatermarked audio unless the
     watermarker were deliberately implemented or linked. "Always on" is a fact about
     upstream's code; do not carry it forward as a fact about a port.
  3. **Our own obligation is separate and unmet by theirs.** `syrinx-serve` owes a
     watermark of its own. Two neural watermarks in one waveform raises questions nobody
     has answered (do they survive each other? which detector answers a provenance
     claim?), and *dropping* the upstream one is equally a decision rather than a default.
     Neither is settled here, and neither should be settled by whichever happens to be
     easier to code.

MIT is the right shape for a second family. **Adopting one is a scope decision, not a
licence one**, and this section does not make it.

## The judge was replaced again, 2026-09-06 — measured, not assumed

`emotion2vec/emotion2vec_plus_large` replaces the RAVDESS fine-tune. Both are permissive,
so this was a **capability** decision, and it was made on a measurement rather than a card:

| judge | CREMA-D overall | `sad` | `angry` | `happy` | `fearful` |
|---|---|---|---|---|---|
| `ehcalabres` (RAVDESS 8-class) | 0.394 | 0.17 | 0.80 | — | 0.10 |
| **`emotion2vec+ large`** | **0.911** | **0.867** | **1.000** | 0.967 | 0.667 |

Same corpus, same protocol (`scripts/calibrate-emotion2vec.py`, 180 clips, ODbL, held out
from both models). The decisive cell is `sad`: 0.17 -> 0.867. `[sad]` is the one cue with a
demonstrated acoustic effect (p=0.0019) and the old judge had no standing to say whether the
change was *toward sadness*. This one does.

`fearful` (0.667, confused with `sad`) is the weakest class and the one to quote a caveat
against. `surprised`, `other` and `unknown` are **not probed** — CREMA-D has no such clips,
so no recall is claimed for them.

Licence note: FunASR's is bespoke rather than OSI-approved. It grants use, copy, modify and
share, requires attribution and model-name retention, and does **not** restrict commercial
use. It wants the same lawyer's glance as anything non-standard, but it is not
NonCommercial — unlike audEERING's, which was rejected outright.

## Measurement tools are a separate question from shipped artifacts

The affect model (`w2v2-msp-dim`) is a **measuring instrument**, not a component: nothing
it produces is linked into a binary or shipped. That distinction matters but does not
dissolve the licence. Two separate questions, which are worth keeping apart:

1. **May we run it?** CC's NonCommercial clause restricts use "primarily intended for or
   directed toward commercial advantage". Running it inside commercial R&D is plausibly
   exactly that, regardless of whether anything ships. The conservative reading is that
   commercial use of the *instrument* needs a licence — and audEERING's own model card
   says a commercial licence can be bought from them, which implies they read it that way
   too.
2. **Is the output derivative?** Much weaker, and it depends entirely on how it is used:
   - **As a screen** — rendering candidates, measuring arousal/valence, and a human
     picking a phrasing — the artifact is a short English sentence like "Speak in a soft
     whisper". Short phrases and factual measurements are poor candidates for copyright,
     and nothing of the model's expressive content ends up in them.
   - **As a training signal** — using its scores as a reward to fine-tune weights — is a
     far stronger derivation argument, because the model's judgements are then baked into
     the weights being shipped. Avoid that entirely for anything commercial.

The gap between those two is the whole risk surface, and it is one design decision wide.

## Clean paths, in order of preference

1. **Buy the commercial licence** from audEERING. Explicitly offered on the model card;
   removes the question rather than reasoning around it.
2. **Swap in a permissive judge for anything commercial.** Several SER models are
   Apache-2.0 (`speechbrain/emotion-recognition-wav2vec2-IEMOCAP`,
   `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`,
   `firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3`). They are
   **categorical** rather than dimensional, so they lose the direct arousal/valence
   mapping to `vocab.toml` — but a categorical label can still be projected onto the
   vocab's coordinates, which is enough for a direction test. **Keep the judge pluggable
   so this swap is a config change, not a rewrite.**
3. **Keep the NC model strictly research-side**: dev-time screening only, never in a
   shipped artifact, never as a training signal, and never in a commercially-directed
   programme.

## Rules of thumb for this repo

- A model's licence follows the **weights**, not the inference code. Every Syrinx crate is
  ours; the restriction rides on the checkpoint it loads.
- Anything that ends up **in** a shipped artifact (weights, embeddings baked into a
  product, generated audio you sell) inherits the strictest licence in its lineage.
- A **measurement** about an artifact is not usually part of it — but running the
  instrument is still a use of the instrument.
- When adding any new checkpoint, add a row here first. That is cheaper than discovering
  the constraint after building on it.

## The affect judge: why the permissive one won despite being worse

audEERING's model was the better technical fit — it regresses **arousal and valence
directly in 0..1**, the same scale `vocab.toml` already declares, so a direction test needed
no mapping at all. It is CC-BY-NC-SA-4.0 and was **rejected on licence** once a commercial
direction was confirmed.

The replacement, `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` (Apache-2.0),
is **categorical**: 8 RAVDESS classes rather than two continuous axes. That is a real
downgrade and worth naming rather than glossing:

- The 8 classes (neutral, calm, happy, sad, angry, fearful, disgust, surprised) do map
  almost 1:1 onto the vocab's labels, so a direction test is still possible — "did the
  tagged render shift probability toward the cued class" instead of "did arousal rise".
- Use the **full probability vector**, never the argmax: a shift from 0.2 to 0.4 on the
  right class is the signal, and argmax discards it.
- RAVDESS is **acted** speech applied to **synthetic** speech — two domain gaps stacked.
  Treat a null result as "this judge cannot tell", not "the cue did nothing".

Because that trade-off may be revisited (a commercial audEERING licence, or a better
permissive model appearing), **the judge is required to sit behind a pluggable interface**
so swapping it is configuration rather than a rewrite.
