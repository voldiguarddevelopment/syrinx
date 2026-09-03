# `renders/` — TTS output runs and their verification

Rendered audio from the Fish Audio models, one directory per run. **`samples/` holds the
inputs** (the corpus text); **`renders/` holds the outputs** (audio + what a speech
recognizer heard back).

Audio is **not** committed — `*.wav` is gitignored repo-wide. What *is* committed is the
part that makes a run auditable after the audio is gone: the exact corpus that was
rendered, the manifest, the box provenance, and the verification report.

## Layout

```
renders/<date>-<variant>-<slug>/
  corpus.jsonl    the exact entries rendered (a frozen copy — samples/ may move on)
  manifest.tsv    written by `syrinx synth --fish --batch`: id -> wav, sample count
  run-meta.txt    box provenance: host, CPU/GPU, driver, toolchain, commit
  report.tsv      per-clip ASR verification (see below)
  *.wav           the audio itself — gitignored, regenerate with the command below
```

## Watching a run (live UI)

```bash
scripts/render-ui.py --run renders/<run-dir>      # then open http://localhost:8080
```

A stdlib-only local server (no Flask, no build step) that reads the run directory as it
fills. It shows per-GPU progress and what each worker is currently synthesizing, overall
throughput and ETA, and — once `report.tsv` exists — the ASR verdict for every clip with
inline audio playback. Filter by language, scale, tag placement, or by the things you
actually want to find: `WER > 0.20`, language mismatch, tag leak. Click a row to see what
the recognizer heard next to what was asked for.

It binds to `127.0.0.1` and only serves `.wav` files from inside the run directory.

## Verifying a run

Renders are graded objectively, never by ear alone. `scripts/verify-renders.py` runs
faster-whisper over every clip and writes `report.tsv`:

- **`det_lang` / `lang_ok`** — the language the ASR *detected*. A Polish prompt that comes
  back `en` is a cross-lingual failure even if the words are fine.
- **`wer`** — word error rate with the language **forced** to the entry's own, so a
  language-ID miss is not double-counted as unintelligibility.
- **`tag_leak`** — whether an emotion tag (`[sad]`, `(excited)`) was *spoken*. Tags are
  control markers and must never appear in the audio; a leak is a real defect.

WER is computed against the prompt text with tags stripped, casefolded, punctuation
removed, Unicode-normalized (so Polish diacritics and German umlauts compare correctly).

## Reproducing a run

`corpus.jsonl` is a frozen copy, but it is not hand-made — `scripts/pick-render-subset.py`
regenerates it deterministically from `samples/fish-samples.jsonl` (the exact invocation is
recorded in each run's `run-meta.txt`):

```bash
scripts/pick-render-subset.py --variant s2 --lang en=17 --lang de=17 --lang pl=16 \
    --out renders/<run>/corpus.jsonl
```

Then render and verify:

```bash
source scripts/test-all.env          # weights, CUDA 12.8 toolchain, device ordinal
RUN=renders/<the-run-dir>

# render (SYRINX_FISH_DEVICE picks the GPU; the 5B bf16 LM needs ~10.4 GB free)
SYRINX_FISH_DEVICE=1 ./target/release/syrinx synth --fish s2-pro \
  --fish-dir "$SYRINX_FISH_S2_DIR" --cuda --ref-wav "$SYRINX_FISH_REF_WAV" \
  --batch "$RUN/corpus.jsonl" --out-dir "$RUN" --batch-size 1

# verify
scripts/verify-renders.py --jsonl "$RUN/corpus.jsonl" --out-dir "$RUN" \
  --model large-v3 --device-index 1 --report "$RUN/report.tsv"
```

## Reading the results honestly

The caveats in [`samples/README.md`](../samples/README.md) apply — a corpus entry is a
*request*, and a weak render is data about the model, not automatically a bug. Two more
that bite here:

- **Cross-lingual cloning is bounded by the reference clip.** Cloning a German or Polish
  line from an English reference is limited by the phonemes that reference contains. If
  `de`/`pl` WER is worse than `en` with an English ref, isolate the cause by re-rendering
  without `--ref-wav` before blaming the model.
- **The ASR is a floor, not truth.** `large-v3` has its own error rate, and it is higher on
  short, emotional, or accented speech. A WER of 0.1 on a six-word line is one word.
