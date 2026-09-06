#!/usr/bin/env python3
"""
export-affect-onnx.py — export the affect judge to ONNX so Rust can run it with `ort`.

`C4.1` forbids Python at inference. The audEERING model that was first considered came with
a ready-made ONNX export; `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`
(Apache-2.0, the judge actually adopted — see `docs/LICENSES.md`) does not, so we export it.

## Two things this export deliberately bakes into the graph

1. **The head.** The checkpoint's `config.json` lies: it says
   `Wav2Vec2ForSequenceClassification`, but the tensors are `classifier.dense` [1024,1024]
   and `classifier.output` [8,1024], which that class does not have. Loading it with
   `AutoModelForAudioClassification` silently drops both and initialises a RANDOM head —
   no error, just noise. The head used here is mean-pool -> dense -> **tanh** -> output,
   chosen by measurement, not by preference: see `scripts/probe-affect-head.py`, which
   scores all three candidate non-linearities on 180 labelled CREMA-D clips.

2. **The feature extractor.** `Wav2Vec2FeatureExtractor` with `do_normalize=true` is exactly
   `(x - mean) / sqrt(var + 1e-7)` over the clip. Putting it inside the graph means the Rust
   side hands over a raw 16 kHz waveform and owns no feature-extraction code that could
   drift from HF's. It also matches the shape of audEERING's published export (`signal` in,
   `logits` out), so a future judge swap does not change the Rust contract. The exported
   graph is therefore gain-invariant by construction, and `verify` below asserts it.

Outputs, matching that convention:

    signal        [1, time]  f32   raw 16 kHz mono, ANY amplitude
    logits        [1, 8]     f32   RAW logits (not softmaxed — the caller decides)
    hidden_states [1, 1024]  f32   the mean-pooled last hidden state, i.e. the head's input

`hidden_states` is exported for the same reason `gen-qwen-ref-speaker.py` dumps the mel as
well as the x-vector: if `logits` ever disagrees between Rust and the reference, this says
in one glance whether the fault is the transformer trunk or the four numbers of the head.

`torch.onnx.export` needs the `onnx` package (torch writes the protobuf through it). It is
NOT in the qwen venv and this script does not add it there — installing into somebody
else's environment is how a working env stops working. Put it beside instead:

    /home/floofy/.venvs/qwen/bin/pip install --target /home/floofy/.venvs/affect-libs onnx

Usage:
    PYTHONPATH=/home/floofy/.venvs/affect-libs MEMMAX=10G scripts/run-isolated.sh \\
        /home/floofy/.venvs/qwen/bin/python \\
        scripts/export-affect-onnx.py --model /data/models/w2v2-lg-xlsr-en-ser
"""
import argparse
import os
import sys

TARGET_SR = 16000
NORM_EPS = 1e-7  # transformers' Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm


def die(msg: str):
    raise SystemExit(f"[export-affect-onnx] FATAL: {msg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local snapshot of the checkpoint")
    ap.add_argument("--out", default=None, help="output .onnx (default <model>/model.onnx)")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    out = args.out or os.path.join(args.model, "model.onnx")

    import numpy as np
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2FeatureExtractor
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2Model,
        Wav2Vec2PreTrainedModel,
    )

    torch.set_grad_enabled(False)

    class Head(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.output = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, pooled):
            return self.output(torch.tanh(self.dense(pooled)))

    class AffectJudge(Wav2Vec2PreTrainedModel):
        """The whole judge: normalise -> wav2vec2 -> mean-pool -> head."""

        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = Head(config)
            self.init_weights()

        def forward(self, signal):
            mean = signal.mean(dim=-1, keepdim=True)
            var = signal.var(dim=-1, keepdim=True, unbiased=False)
            x = (signal - mean) / torch.sqrt(var + NORM_EPS)
            pooled = torch.mean(self.wav2vec2(x)[0], dim=1)
            return self.classifier(pooled), pooled

    model = AffectJudge.from_pretrained(args.model)
    model.eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    print(f"[export-affect-onnx] labels {labels}", file=sys.stderr)

    # A real-length dummy: 3 s. Too short a trace can bake in a degenerate conv shape.
    dummy = torch.randn(1, 3 * TARGET_SR, dtype=torch.float32) * 0.05

    torch.onnx.export(
        model,
        (dummy,),
        out,
        input_names=["signal"],
        output_names=["logits", "hidden_states"],
        dynamic_axes={"signal": {1: "time"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[export-affect-onnx] wrote {out} ({os.path.getsize(out)/1e6:.0f} MB)", file=sys.stderr)

    # ---- verify the export against the torch model it came from --------------------
    import onnxruntime as rt

    sess = rt.InferenceSession(out, providers=["CPUExecutionProvider"])
    ins = [(i.name, i.shape) for i in sess.get_inputs()]
    outs = [(o.name, o.shape) for o in sess.get_outputs()]
    print(f"[export-affect-onnx] graph in {ins} out {outs}", file=sys.stderr)

    fe = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
    rng = np.random.default_rng(7)
    worst_logits = 0.0
    worst_hidden = 0.0
    for secs in (1.0, 2.5, 4.75):  # three lengths: the time axis must really be dynamic
        x = (rng.standard_normal(int(secs * TARGET_SR)) * 0.05).astype(np.float32)
        t_log, t_hid = model(torch.from_numpy(x).reshape(1, -1))
        o_log, o_hid = sess.run(None, {"signal": x.reshape(1, -1)})
        worst_logits = max(worst_logits, float(np.abs(t_log.numpy() - o_log).max()))
        worst_hidden = max(worst_hidden, float(np.abs(t_hid.numpy() - o_hid).max()))

        # And the normalisation really is inside the graph: same clip, different gain.
        ref = sess.run(None, {"signal": x.reshape(1, -1)})[0]
        for g in (0.1, 10.0):
            g_out = sess.run(None, {"signal": (x * g).reshape(1, -1).astype(np.float32)})[0]
            d = float(np.abs(g_out - ref).max())
            if d > 1e-2:
                die(f"graph is not gain-invariant at {g} (max|d| {d:.3e})")

        # ...and it agrees with the HF feature extractor it replaces.
        hf = np.asarray(fe(x, sampling_rate=TARGET_SR)["input_values"][0], dtype=np.float32)
        ours = (x - x.mean()) / np.sqrt(x.var() + NORM_EPS)
        d = float(np.abs(hf - ours).max())
        if d > 1e-4:
            die(f"in-graph normalisation differs from Wav2Vec2FeatureExtractor by {d:.3e}")

    print(
        f"[export-affect-onnx] onnx vs torch: logits max|d| {worst_logits:.3e}, "
        f"hidden max|d| {worst_hidden:.3e}",
        file=sys.stderr,
    )
    # Loose enough for f32 accumulation-order differences between two runtimes over 24
    # transformer layers, tight enough that a wrong graph cannot slip through.
    if worst_logits > 1e-3 or worst_hidden > 1e-3:
        die("export does not reproduce the torch model")
    print("[export-affect-onnx] OK", file=sys.stderr)


if __name__ == "__main__":
    main()
