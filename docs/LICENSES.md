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
| `ehcalabres/wav2vec2-lg-xlsr-en-...` | affect judge (**adopted**): RAVDESS 8-class | **Apache-2.0** | **yes** |

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

### Candidate second family

| model | licence | notes |
|---|---|---|
| `ResembleAI/chatterbox` | **MIT** | 23 languages, voice cloning. Fully permissive — no NC clause, no ShareAlike. A candidate, not adopted; it would need the same reference-anchored port treatment Qwen got before anyone trusts it. |

MIT is if anything cleaner than Apache-2.0 here (no patent grant, but no NOTICE obligation
either). It is the right shape for a second family; adopting it is a scope decision, not a
licence one.

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
