#!/usr/bin/env python3
"""
gen-affect-ref.py — dump the **affect judge** reference for `tests/real_qwen_affect.rs`.

The fourth sibling of `gen-fish-ref.py` / `gen-qwen-ref*.py`, and it exists for the same
reason they do: the Rust side must be anchored against the real upstream implementation
running on the real weights, never against numbers anybody typed in.

## What is being anchored

`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` — wav2vec2-large-xlsr-53-english
fine-tuned on RAVDESS to classify 8 emotions (angry, calm, disgust, fearful, happy, neutral,
sad, surprised). **Apache-2.0**, which is why it and not audEERING's technically-better
dimensional model: see `docs/LICENSES.md`.

## The trap this script is written around

The checkpoint's `config.json` says `architectures: ["Wav2Vec2ForSequenceClassification"]`
and that is FALSE. The stock class has `projector` + a plain `classifier` Linear; this
checkpoint has `classifier.dense` [1024,1024] and `classifier.output` [8,1024].
`AutoModelForAudioClassification.from_pretrained` therefore loads the trunk, throws BOTH
head tensors away as unexpected, initialises a random head, and returns a model that runs
happily and predicts noise. So this script:

  * defines the real head (mean-pool -> dense -> **tanh** -> output) explicitly, and
  * asserts, via `output_loading_info`, that every head tensor actually loaded.

The `tanh` is not a preference. No authoritative source states the activation and two
third-party reimplementations disagree (tanh vs relu), so `scripts/probe-affect-head.py`
decided it by measurement on 180 labelled CREMA-D clips. Read that script before changing
it; its finding was that the choice barely matters here (the dense pre-activations sit
inside |x| < 1.3, where tanh is nearly the identity) but that tanh scores best on both
accuracy and mean probability of the true class.

## What it captures, and why each tensor is here

  wav16/<stem>    [N]      the 16 kHz mono clip AS FED to the model, i.e. after
                           `librosa.resample`. Captured for the same reason
                           `gen-qwen-ref-speaker.py` captures `wav24`: Rust has no soxr, so
                           a resampler difference must never be mistaken for a model-port
                           fault. The Rust parity anchor reads THIS; the driver-path
                           resampler gets its own, separately measured, bound.
  wav_in/<stem>   [M]      the clip as `soundfile` read it (24 kHz here), for that
                           driver-path diagnostic.
  sr_in/<stem>    [1] i64  its sample rate (safetensors carries tensors, not ints).
  logits/<stem>   [8]      raw head output, in `config.id2label` order.
  probs/<stem>    [8]      softmax of the above. Report the whole vector, never the argmax:
                           a shift from 0.2 to 0.4 on the cued class IS the signal.
  hidden/<stem>   [1024]   the mean-pooled last hidden state, i.e. the head's INPUT. Two
                           anchors instead of one: if `logits` disagrees, `hidden` says
                           immediately whether the fault is the trunk or the head.

CPU/float32 only, and that is not a preference: the reference is the parity path and a GPU
fixture would spend the error budget on accumulation order (the lesson already written into
`gen-qwen-ref.py`).

Environment: any python with torch + transformers + librosa + soundfile + safetensors;
on this box `/home/floofy/.venvs/qwen/bin/python`.

Usage:
    MEMMAX=10G scripts/run-isolated.sh /home/floofy/.venvs/qwen/bin/python \\
        scripts/gen-affect-ref.py \\
        --model /data/models/w2v2-lg-xlsr-en-ser \\
        --out   /home/floofy/parity-affect/affect.safetensors \\
        renders/2026-09-06-qwen-emotion-ab/*.wav
"""
import argparse
import os
import sys

TARGET_SR = 16000


def die(msg: str):
    raise SystemExit(f"[gen-affect-ref] FATAL: {msg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local snapshot of the checkpoint")
    ap.add_argument("--out", required=True, help="output .safetensors")
    ap.add_argument("wavs", nargs="+", help="clips to score (any rate; resampled here)")
    args = ap.parse_args()

    if not os.path.isdir(args.model):
        die(f"--model {args.model!r} is not a directory")

    try:
        import numpy as np
        import torch
        import torch.nn as nn
        import librosa
        import soundfile as sf
        from safetensors.torch import save_file
        from transformers import Wav2Vec2FeatureExtractor
        from transformers.models.wav2vec2.modeling_wav2vec2 import (
            Wav2Vec2Model,
            Wav2Vec2PreTrainedModel,
        )
    except Exception as e:  # noqa: BLE001 — refuse to run rather than approximate
        die(f"cannot import the reference stack ({e}). This script never reimplements it.")

    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

    class Head(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.output = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, pooled):
            return self.output(torch.tanh(self.dense(pooled)))

    class AffectJudge(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = Head(config)
            self.init_weights()

        def forward(self, input_values):
            pooled = torch.mean(self.wav2vec2(input_values)[0], dim=1)
            return self.classifier(pooled), pooled

    model, info = AffectJudge.from_pretrained(args.model, output_loading_info=True)
    model.eval()

    # THE guard. A silently-random head is this checkpoint's signature failure mode, and it
    # produces plausible-looking numbers, so it must be impossible to reach a fixture.
    missing = [k for k in info.get("missing_keys", []) if k.startswith("classifier.")]
    unexpected = [k for k in info.get("unexpected_keys", []) if k.startswith("classifier.")]
    if missing or unexpected:
        die(
            f"head did not load: missing={missing} unexpected={unexpected}. "
            "The checkpoint's head is `classifier.dense` + `classifier.output`; a mismatch "
            "here means the model is running a RANDOM classifier."
        )
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    print(f"[gen-affect-ref] loaded {args.model}", file=sys.stderr)
    print(f"[gen-affect-ref] head loaded clean; labels {labels}", file=sys.stderr)

    fe = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
    if fe.sampling_rate != TARGET_SR:
        die(f"feature extractor wants {fe.sampling_rate} Hz, this script assumes {TARGET_SR}")
    if not fe.do_normalize:
        die("feature extractor has do_normalize=False; the checkpoint was trained with it on")

    tensors = {}
    print(
        "[gen-affect-ref] " + " ".join(f"{l[:5]:>7}" for l in labels) + "   clip",
        file=sys.stderr,
    )
    for path in args.wavs:
        stem = os.path.splitext(os.path.basename(path))[0]
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(axis=1).astype(np.float32)  # mono
        # `.copy()` matters even in the no-resample case: safetensors refuses to write two
        # keys that alias one buffer, and `wav_in` / `wav16` coincide for a 16 kHz input.
        x16 = x.copy() if sr == TARGET_SR else librosa.resample(x, orig_sr=sr, target_sr=TARGET_SR)
        x16 = np.ascontiguousarray(x16, dtype=np.float32)

        y = fe(x16, sampling_rate=TARGET_SR)["input_values"][0]
        logits, hidden = model(torch.from_numpy(np.asarray(y, np.float32)).reshape(1, -1))
        logits = logits[0].float()
        hidden = hidden[0].float()
        probs = torch.softmax(logits, dim=-1)
        print(
            "[gen-affect-ref] " + " ".join(f"{float(p):>7.3f}" for p in probs) + f"   {stem}",
            file=sys.stderr,
        )

        tensors[f"wav_in/{stem}"] = torch.from_numpy(np.ascontiguousarray(x))
        tensors[f"sr_in/{stem}"] = torch.tensor([sr], dtype=torch.int64)
        tensors[f"wav16/{stem}"] = torch.from_numpy(x16)
        tensors[f"logits/{stem}"] = logits.contiguous()
        tensors[f"probs/{stem}"] = probs.contiguous()
        tensors[f"hidden/{stem}"] = hidden.contiguous()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # The label order is part of the contract; carry it in the file so the Rust side can
    # assert it instead of assuming it.
    save_file(tensors, args.out, metadata={"labels": ",".join(labels)})
    print(f"[gen-affect-ref] wrote {len(tensors)} tensors -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
