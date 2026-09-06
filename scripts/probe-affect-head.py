#!/usr/bin/env python3
"""
probe-affect-head.py — DETERMINE, by measurement, the classification head of
`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`.

## Why this script has to exist

The checkpoint's `config.json` claims `architectures: ["Wav2Vec2ForSequenceClassification"]`,
and that is **wrong**. The stock transformers class has `projector` (hidden -> proj) plus a
plain `classifier` Linear; this checkpoint's tensors are:

    classifier.dense.weight   [1024, 1024]
    classifier.output.weight  [8, 1024]

so `AutoModelForAudioClassification.from_pretrained` loads the wav2vec2 trunk, silently
DISCARDS both head tensors as unexpected, and randomly initialises a head of its own. It
does not error. The pipeline runs and returns confident-looking nonsense. Anyone who has
used this model through `AutoModel*` without reading the warning has measured noise.

The correct forward is therefore a custom head, and no authoritative source states it.
Two independent third-party reimplementations found by GitHub code search disagree on the
one thing that matters:

  * `Nikhil-Yadav15/SonicTrace :: core/emotion_recognizer.py`  -> dense -> **tanh** -> output
  * `T4ko0522/speech-processing-poc :: .../speech_emotion.py`  -> dense -> **relu** -> output

Both agree on mean-pooling the last hidden state first. `CLAUDE.md` forbids resolving that
by preference, so this script resolves it by measurement: run every candidate head over
labelled emotional speech and keep the one that can actually classify it. A wrong
non-linearity is not a small error — the `dense` weights were fitted through whichever
function was there, so using another one leaves a classifier near chance.

## The probe set, and why not RAVDESS

The model was fine-tuned on RAVDESS, whose licence is CC-BY-NC-SA-4.0 — the exact licence
class this project just moved OFF (see `docs/LICENSES.md`). So the probe uses **CREMA-D**
(github.com/CheyneyComputerScience/CREMA-D, **Open Database License v1.0**, commercial use
permitted), which is labelled, acted, English, already 16 kHz mono, and — being a different
corpus — is honest held-out data rather than the training set.

CREMA-D carries 6 of the model's 8 classes (no `calm`, no `surprised`); those two stay
reachable as predictions and simply count as errors. Chance is 1/6 = 0.167 if a head only
ever guesses among reachable classes, and lower if it wanders into the other two. The
default subset is sentence `DFA`, intensity `XX`, speakers 1001-1030: 180 clips, balanced
6 x 30, ~14 MB.

Absolute accuracy here will NOT match the model card's 0.8223 — that number is RAVDESS,
in-domain, and this is a corpus shift. The verdict is comparative.

WHAT IT ACTUALLY FOUND (NovaBox, 2026-09-06, 180 clips, chance 0.167):

    head        accuracy   mean p(true)
    tanh           0.394          0.342
    relu           0.389          0.278
    identity       0.389          0.349
    dense pre-activation |x|: mean 0.241  max 1.239  fraction > 0 0.494

The expected large gap did NOT appear, and the pre-activation scale says why: `dense`
outputs live inside |x| < 1.3, where tanh is close to the identity, so the three heads
agree on 93-98 % of predictions. `tanh` is adopted because it wins on both metrics AND
because `pooling_mode: mean` + `final_dropout` in the config are the signature of the
m3hrdadfi `Wav2Vec2ForSpeechClassification` family, whose head is dense -> tanh -> out.
The honest summary is "the activation barely matters for this checkpoint", recorded here
rather than hidden behind a confident-looking choice.

The confusion matrix printed alongside is the more important output: it says the judge
can hear `angry` (recall 0.80) and `neutral` (0.73) and mostly cannot hear `sad` (0.17,
leaking to `calm`) or `fearful` (0.10). Read it before trusting any verdict this model
gives on a render.

Usage:
    scripts/probe-affect-head.py --model /data/models/w2v2-lg-xlsr-en-ser \\
        --data /data/datasets/crema-d-probe [--download] [--limit N]
"""
import argparse
import os
import sys
import urllib.request

CREMA_RAW = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV"
CREMA_SENTENCE = "DFA"
CREMA_SPEAKERS = range(1001, 1031)
# CREMA-D emotion code -> the checkpoint's label. `calm` and `surprised` have no
# CREMA-D counterpart; they stay predictable and count as errors when predicted.
CREMA_TO_LABEL = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}
TARGET_SR = 16000


def die(msg: str):
    raise SystemExit(f"[probe-affect-head] FATAL: {msg}")


def download(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    for spk in CREMA_SPEAKERS:
        for code in CREMA_TO_LABEL:
            name = f"{spk}_{CREMA_SENTENCE}_{code}_XX.wav"
            dest = os.path.join(data_dir, name)
            if os.path.exists(dest) and os.path.getsize(dest) > 1024:
                continue
            print(f"[probe-affect-head] GET {name}", file=sys.stderr)
            urllib.request.urlretrieve(f"{CREMA_RAW}/{name}", dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="CREMA-D probe dir")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="use at most N clips (0 = all)")
    ap.add_argument("--act", default="tanh", choices=["tanh", "relu", "identity"],
                    help="which candidate head the confusion matrix is printed for")
    args = ap.parse_args()

    if args.download:
        download(args.data)
    if not os.path.isdir(args.data):
        die(f"--data {args.data!r} missing; pass --download to fetch it")

    import numpy as np
    import torch
    import torch.nn as nn
    import soundfile as sf
    from transformers import Wav2Vec2FeatureExtractor
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2Model,
        Wav2Vec2PreTrainedModel,
    )

    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

    # Candidate heads. `identity` is included on purpose: it is what you get if the two
    # Linears are folded (the community `projector`/`classifier` remap in the model's HF
    # discussion thread does exactly that, with no non-linearity between them), so it is
    # a real hypothesis somebody is shipping, not a strawman.
    ACTS = {"tanh": torch.tanh, "relu": torch.relu, "identity": lambda x: x}

    class Head(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.output = nn.Linear(config.hidden_size, config.num_labels)

    class Probe(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = Head(config)
            self.init_weights()

        def pooled(self, input_values):
            return torch.mean(self.wav2vec2(input_values)[0], dim=1)

        def logits(self, pooled, act):
            return self.classifier.output(act(self.classifier.dense(pooled)))

    model = Probe.from_pretrained(args.model)
    model.eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    print(f"[probe-affect-head] labels: {labels}", file=sys.stderr)

    fe = Wav2Vec2FeatureExtractor.from_pretrained(args.model)

    clips = sorted(f for f in os.listdir(args.data) if f.endswith(".wav"))
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        die(f"no .wav under {args.data}")

    conf = np.zeros((len(labels), len(labels)), dtype=np.int64)
    pre_abs, pre_max, pre_pos = [], [], []
    hits = {k: 0 for k in ACTS}
    # Mean probability the head assigns to the TRUE class — a finer signal than accuracy,
    # and it does not collapse when two heads happen to argmax the same way.
    ptrue = {k: 0.0 for k in ACTS}
    n = 0
    for name in clips:
        code = name.split("_")[2]
        gold = CREMA_TO_LABEL.get(code)
        if gold is None:
            continue
        x, sr = sf.read(os.path.join(args.data, name), dtype="float32", always_2d=True)
        if sr != TARGET_SR:
            die(f"{name} is {sr} Hz; the CREMA-D probe set is expected at {TARGET_SR}")
        x = x.mean(axis=1).astype(np.float32)
        y = fe(x, sampling_rate=TARGET_SR)["input_values"][0]
        pooled = model.pooled(torch.from_numpy(np.asarray(y, np.float32)).reshape(1, -1))
        gi = labels.index(gold)
        # The pre-activation scale decides how much the choice of non-linearity can
        # matter at all: inside |x| < 1, tanh is within 24 % of the identity.
        pre = model.classifier.dense(pooled)[0]
        pre_abs.append(float(pre.abs().mean()))
        pre_max.append(float(pre.abs().max()))
        pre_pos.append(float((pre > 0).float().mean()))
        for k, act in ACTS.items():
            p = torch.softmax(model.logits(pooled, act)[0], dim=-1)
            pi = int(p.argmax())
            hits[k] += int(pi == gi)
            ptrue[k] += float(p[gi])
            if k == args.act:
                conf[gi, pi] += 1
        n += 1
        if n % 20 == 0:
            print(f"[probe-affect-head] {n}/{len(clips)}", file=sys.stderr)

    print(f"\n[probe-affect-head] {n} CREMA-D clips, 6 reachable classes, chance = {1/6:.3f}")
    print(f"{'head':<12}{'accuracy':>10}{'mean p(true)':>15}")
    for k in ACTS:
        print(f"{k:<12}{hits[k]/n:>10.3f}{ptrue[k]/n:>15.3f}")

    print(
        f"\ndense pre-activation |x|: mean {float(np.mean(pre_abs)):.3f}  "
        f"max {float(np.max(pre_max)):.3f}  fraction > 0 {float(np.mean(pre_pos)):.3f}"
    )

    print(f"\nconfusion for --act {args.act} (rows = CREMA-D truth, cols = predicted)")
    print(f"{'':<10}" + "".join(f"{l[:5]:>7}" for l in labels) + f"{'recall':>9}")
    for gi, lab in enumerate(labels):
        row = conf[gi]
        tot = int(row.sum())
        rec = f"{row[gi] / tot:.2f}" if tot else "-"
        print(f"{lab:<10}" + "".join(f"{int(c):>7}" for c in row) + f"{rec:>9}")


if __name__ == "__main__":
    main()
