# Fish Audio sample corpus (`s1-mini`, `s2-pro`)

A large, structured corpus for exercising Syrinx's Fish Audio models. It is organized
around **three scales** and deliberately places **emotion tags in varied positions** so
each model's tag-following behaviour can be probed systematically across **13 languages**.

- **610 entries** total — `small` 364, `reply` 182, `chapter` 64.
- Authoring is **box-independent** (pure text). Synthesis runs on the GPU box via
  `scripts/synth-samples.sh`.

## The three scales (primary organizing axis)

| scale | size | role | count |
|---|---|---|---|
| `small`   | short sentences, ≈3–12 words | the quick-test bread-and-butter; many per language | 364 |
| `reply`   | conversational / assistant turns, ≈1–4 sentences | realistic dialogue turns and responses | 182 |
| `chapter` | long-form paragraph, ≈120–300+ words | a real narrative arc; emotion **progresses** across the passage | 64 |

## Emotion-tag placement (the explicit second axis)

Every scale exercises tags in different positions:

| placement | meaning |
|---|---|
| `leading`      | tag at the very start — sets the whole utterance |
| `mid`          | tag mid-sentence at a natural clause break |
| `trailing`     | tag at the end |
| `multi`        | 2–4 emotion **switches** within one item |
| `wrap`         | a tag immediately before one specific word/phrase to colour just that word |
| `special`      | a special-audio insert mid-text (laugh / sigh / gasp / chuckle) |
| `combined`     | two tags together = tone + emotion |
| `per_sentence` | a tag opening most sentences (natural for `reply` / `chapter`) |
| `neutral`      | untagged baseline |

For `chapter`, tags ride a real emotional **arc** across the paragraph
(e.g. calm → tense → sad → hopeful → joyful) at sentence/scene boundaries.

## Fish tag syntax (per model)

- **s1** (`s1-mini`) — parenthesized inline markers:
  `(happy) (sad) (angry) (excited) (surprised) (sarcastic) (fearful) (disdainful)
  (whispering) (shouting) (screaming) (in a hurry tone) (laughing) (chuckling)
  (sobbing) (crying loudly) (sighing) (panting) (groaning)` …
- **s2** (`s2-pro`) — bracketed + free-form descriptive:
  `[laugh] [whispers] [super happy] [angry] [pitch up] [prolonged laugh]
  [whisper in a small voice] [breathing heavily] [sad] [excited] [surprised]
  [sigh] [gasp] [chuckle] [trembling voice] [soft] [fast]` …

Each entry's `model` is `s1`, `s2`, or `both`. `both` is reserved for `neutral`
(untagged) baselines, whose text contains no tags and is valid for either model.
Inline tags inside `text` are always written in the chosen model's syntax; the language
of the spoken content is in `text`, never in the tags.

## Schema — `fish-samples.jsonl`

One JSON object per line (UTF-8, compact). Validate with `jq -c . fish-samples.jsonl`.

```json
{
  "id": "small_en_leading_happy_01",
  "scale": "small",            // small | reply | chapter
  "lang": "en",                // en zh ja ko fr de es ar ru nl it pl pt
  "model": "s1",               // s1 | s2 | both  (both => neutral, no tags)
  "placement": "leading",      // leading mid trailing multi wrap special combined per_sentence neutral
  "tags": ["happy"],           // tag words used, without ()/[]; [] for neutral
  "text": "(happy) We finally did it!",   // carries the inline tags in the model's syntax
  "desc": "leading happy, celebration"    // short English note
}
```

IDs are stable and categorized:
- `small` / `reply`: `<scale>_<lang>_<placement>_<tagslug>_<NN>` (e.g. `reply_zh_per_sentence_happy_01`).
- `chapter`: `chapter_<lang>_arc_<NN>` (e.g. `chapter_zh_arc_03`).

## Breakdown — scale × language × placement

### `small` (364)

| lang | leading | mid | trailing | multi | wrap | special | combined | neutral | **tot** |
|---|---|---|---|---|---|---|---|---|---|
| en | 5 | 4 | 4 | 3 | 3 | 3 | 3 | 3 | **28** |
| zh | 5 | 4 | 4 | 3 | 3 | 3 | 3 | 3 | **28** |
| ja | 5 | 4 | 4 | 3 | 3 | 3 | 3 | 3 | **28** |
| ko | 5 | 4 | 4 | 3 | 3 | 3 | 3 | 3 | **28** |
| fr | 5 | 4 | 4 | 3 | 3 | 3 | 3 | 3 | **28** |
| de | 5 | 4 | 4 | 3 | 3 | 3 | 3 | 3 | **28** |
| es | 5 | 4 | 4 | 3 | 3 | 3 | 3 | 3 | **28** |
| ar | 5 | 4 | 4 | 3 | 3 | 3 | 3 | 3 | **28** |
| ru | 5 | 4 | 4 | 3 | 3 | 3 | 3 | 3 | **28** |
| nl | 5 | 4 | 4 | 3 | 3 | 3 | 3 | 3 | **28** |
| it | 5 | 4 | 4 | 3 | 3 | 3 | 3 | 3 | **28** |
| pl | 5 | 4 | 4 | 3 | 3 | 3 | 3 | 3 | **28** |
| pt | 5 | 4 | 4 | 3 | 3 | 3 | 3 | 3 | **28** |
| **tot** | **65** | **52** | **52** | **39** | **39** | **39** | **39** | **39** | **364** |

### `reply` (182)

| lang | leading | per_sentence | mid | trailing | multi | special | combined | neutral | **tot** |
|---|---|---|---|---|---|---|---|---|---|
| en | 2 | 3 | 2 | 2 | 1 | 1 | 1 | 2 | **14** |
| zh | 2 | 3 | 2 | 2 | 1 | 1 | 1 | 2 | **14** |
| ja | 2 | 3 | 2 | 2 | 1 | 1 | 1 | 2 | **14** |
| ko | 2 | 3 | 2 | 2 | 1 | 1 | 1 | 2 | **14** |
| fr | 2 | 3 | 2 | 2 | 1 | 1 | 1 | 2 | **14** |
| de | 2 | 3 | 2 | 2 | 1 | 1 | 1 | 2 | **14** |
| es | 2 | 3 | 2 | 2 | 1 | 1 | 1 | 2 | **14** |
| ar | 2 | 3 | 2 | 2 | 1 | 1 | 1 | 2 | **14** |
| ru | 2 | 3 | 2 | 2 | 1 | 1 | 1 | 2 | **14** |
| nl | 2 | 3 | 2 | 2 | 1 | 1 | 1 | 2 | **14** |
| it | 2 | 3 | 2 | 2 | 1 | 1 | 1 | 2 | **14** |
| pl | 2 | 3 | 2 | 2 | 1 | 1 | 1 | 2 | **14** |
| pt | 2 | 3 | 2 | 2 | 1 | 1 | 1 | 2 | **14** |
| **tot** | **26** | **39** | **26** | **26** | **13** | **13** | **13** | **26** | **182** |

### `chapter` (64) — full paragraphs, emotion arc at sentence/scene boundaries

| lang | per_sentence | multi | neutral | **tot** |
|---|---|---|---|---|
| en | 3 | 2 | 1 | **6** |
| zh | 3 | 2 | 1 | **6** |
| ja | 3 | 2 | 1 | **6** |
| fr | 3 | 2 | 1 | **6** |
| de | 3 | 2 | 1 | **6** |
| es | 3 | 2 | 1 | **6** |
| ko | 2 | 1 | 1 | **4** |
| ar | 2 | 1 | 1 | **4** |
| ru | 2 | 1 | 1 | **4** |
| nl | 2 | 1 | 1 | **4** |
| it | 2 | 1 | 1 | **4** |
| pl | 2 | 1 | 1 | **4** |
| pt | 2 | 1 | 1 | **4** |
| **tot** | **32** | **19** | **13** | **64** |

### Model balance

| scale | s1 | s2 | both (neutral) |
|---|---|---|---|
| small   | 177 | 148 | 39 |
| reply   | 85  | 71  | 26 |
| chapter | 32  | 19  | 13 |
| **all** | **294** | **238** | **78** |

## Running it

```sh
scripts/synth-samples.sh <s1-mini|s2-pro> [--scale small|reply|chapter] \
    [--lang L] [--placement P] [--limit N] [--ref REF.wav] [--out DIR]
```

The runner filters the corpus by the variant's model (`s1-mini` → `s1`+`both`,
`s2-pro` → `s2`+`both`) plus any `--scale` / `--lang` / `--placement` / `--limit`, and
for each match invokes the Fish synth front door:

```sh
cargo run -p syrinx-cli --features real -- \
    synth --fish <variant> --text "<text>" --ref <REF.wav> --out <DIR>/<id>.wav
```

It writes `manifest.tsv` (id, scale, lang, placement, text → wav) and `counts.txt`
(per-scale / per-language / per-placement summary) under `--out`
(default `samples/out/<variant>`). Examples:

```sh
# Every s2-pro entry (dry-run if --fish isn't wired in yet)
scripts/synth-samples.sh s2-pro --ref voice.wav

# Only the Japanese short sentences on s1-mini, first 10
scripts/synth-samples.sh s1-mini --scale small --lang ja --limit 10 --ref voice.wav

# Only the long-form chapters, to probe long-form coherence/streaming
scripts/synth-samples.sh s2-pro --scale chapter --ref voice.wav
```

**`--fish` pending integration.** The CLI does not yet expose `synth --fish`. Until it
does, the runner probes `synth --help`, detects the missing flag, and runs in **dry-run**
mode: it prints each command it *would* run (tagged `[PENDING INTEGRATION]`) and still
emits the manifest + counts, so the corpus is fully inspectable off-box. Once the flag
lands, the same invocation synthesizes for real (a `--ref` voice clip is then required).

## Honest quality note

This corpus tests **whether the model follows its learned tags**, not whether the text is
hard to pronounce. A few caveats to keep expectations calibrated:

- **Expressiveness depends on the model.** The emotion/special tags only take effect to
  the extent `s1-mini` / `s2-pro` actually learned them. A tag in the right place is a
  *request*; the rendered audio is the model's *response*, and it may under- or
  over-express, ignore a tag, or bleed an emotion past where it was switched. The varied
  `placement` axis exists precisely to surface where that following is strong vs weak.
- **Tag inventories differ by model**, so s1 and s2 entries are not 1:1 translations of
  one another — they use each model's native syntax and vocabulary.
- **`chapter` items test more than emotion.** At 120–300+ words they also stress
  long-form coherence, prosodic stamina, and streaming/chunking behaviour; a clean short
  sentence says little about how the same voice holds up across a full paragraph arc.
- **Text is human-authored, idiomatic, native phrasing** (not machine-translated stubs),
  but it has not been reviewed by a native-speaker panel; minor regional-register
  variation is expected across the 13 languages.
