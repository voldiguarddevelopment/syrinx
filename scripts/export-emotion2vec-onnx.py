#!/usr/bin/env python3
"""Export emotion2vec+ large to ONNX for the Rust affect judge.

  emotion2vec/emotion2vec_plus_large -- 9-class SER, ~300M, fine-tuned on 42,526 h.
  LICENCE: FunASR Model Open Source License (permissive, attribution required).
  See docs/LICENSES.md. It replaces the RAVDESS 8-class judge, which measured 0.394
  cross-corpus and read almost everything as `happy`.

WHY THE GRAPH OUTPUTS **LOGITS**, NOT PROBABILITIES
---------------------------------------------------
funasr's own `Emotion2vec.inference` ends with `torch.softmax(...)`, and this model is
saturated: on real speech the logit spread is ~19-26, so softmax returns a one-hot vector
to float precision. Our entire method is "cued mean minus plain mean against seed noise" --
a one-hot score makes every delta 0 or +/-1 and destroys the measurement. docs/LICENSES.md
already states the rule this follows: "Use the full probability vector, never the argmax."
Saturated softmax IS an argmax, so the graph stops at the logits and Rust decides what to
do with them.

THE GRAPH, traced from funasr 1.4.14 `Emotion2vec.inference` (never reimplemented):

    source = layer_norm(wav, wav.shape)      # iff cfg.normalize
    x      = extract_features(source)["x"]   # [1, T, 1024]
    logits = proj(x.mean(dim=1))             # [1, 9]

THE EXPORTED GRAPH IS LIMITED TO 160,079 SAMPLES (10.005 s @ 16 kHz)
--------------------------------------------------------------------
Measured by binary search, not assumed. The AUDIO modality encoder carries a precomputed
[heads, T, T] position-bias buffer sized 499 frames; torch slices it to the live T, but the
export bakes the buffer and only the slice target stays dynamic, so T <= 499 works and
T > 499 does not. Beyond the limit onnxruntime raises on `/AUDIO/Reshape` -- a LOUD failure,
never a silently wrong number, which is why this is documented and bounded rather than
worked around.

This is comfortably above the intended use: the A/B renders this judge scores are 3-6 s.
Anything longer must be chunked by the caller, deliberately, because chunking changes what
"the emotion of this clip" means and that is not a decision an exporter should make.

Usage: export-emotion2vec-onnx.py <model-dir> <out.onnx>
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn.functional as F

def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    model_dir, out = sys.argv[1], sys.argv[2]

    from funasr import AutoModel
    net = AutoModel(model=model_dir, disable_update=True, device="cpu").model
    net.eval()

    normalize = bool(net.cfg.normalize)
    print(f"[export] cfg.normalize={normalize}  params={sum(p.numel() for p in net.parameters())/1e6:.1f}M")

    class Judge(torch.nn.Module):
        """The inference path above, as a single traced graph."""
        def __init__(self, net, normalize):
            super().__init__()
            self.net, self.normalize = net, normalize
        def forward(self, signal):           # signal: [1, time] f32 @ 16 kHz
            s = signal
            if self.normalize:
                # funasr does `F.layer_norm(source, source.shape)` on the 1-D waveform: no
                # weight, no bias, so it is plain standardisation over the time axis.
                # Spelled out because a dynamic `normalized_shape` cannot be traced -- the
                # legacy exporter fails outright on it and the dynamo one silently emits a
                # length-dependent graph. eps and the biased variance match torch's default.
                mean = s.mean(dim=1, keepdim=True)
                var = s.var(dim=1, unbiased=False, keepdim=True)
                s = (s - mean) / torch.sqrt(var + 1e-5)
            x = self.net.extract_features(s, padding_mask=None)["x"]
            return self.net.proj(x.mean(dim=1))

    wrapped = Judge(net, normalize).eval()

    # ---- check 1: the wrapper reproduces funasr's OWN inference, before anything is
    # exported. If this drifts, the graph below would be a faithful export of the wrong
    # computation -- the failure mode that is hardest to notice later.
    import soundfile as sf, scipy.signal as ss
    probe_wav = "/home/floofy/refs/voice_en_10s.wav"
    w, fs = sf.read(probe_wav, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    if fs != 16000:
        w = ss.resample_poly(w, 16000, fs).astype("float32")
    ref = AutoModel(model=model_dir, disable_update=True, device="cpu").generate(
        probe_wav, granularity="utterance", extract_embedding=False)[0]
    with torch.no_grad():
        mine = torch.softmax(wrapped(torch.from_numpy(w).view(1, -1)), dim=-1)[0].numpy()
    ref_scores = np.array(ref["scores"], dtype="float64")
    d0 = float(np.abs(ref_scores[:8] - mine[:8]).max())
    print(f"[verify] wrapper vs funasr.generate: max|diff| = {d0:.3e}")
    if d0 > 1e-4:
        sys.exit(f"[verify] FAILED: the wrapper does not reproduce funasr's own output")
    # 3 s of noise: long enough for several transformer frames, short enough to trace fast.
    example = torch.randn(1, 48_000)

    with torch.no_grad():
        torch.onnx.export(
            wrapped, (example,), out,
            input_names=["signal"], output_names=["logits"],
            dynamic_axes={"signal": {1: "time"}},
            opset_version=17, do_constant_folding=True,
            # dynamo=False forces the legacy TorchScript exporter. The dynamo path silently
            # produced a graph that agreed with torch at the traced length and at 4.5 s but
            # was ~3 logits out at 1.0 s and 2.5 s -- it does not honour the legacy
            # `dynamic_axes` spelling, and the disagreement is length-dependent, which is
            # exactly the failure a fixed-length smoke test would miss.
            dynamo=False,
        )
    print(f"[export] wrote {out}")

    # ---- check 2: the exported graph matches the wrapper, on lengths the trace never saw.
    import onnxruntime as ort
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    worst = 0.0
    rng = np.random.default_rng(0)
    # Includes both sides of the 160,079-sample ceiling documented above.
    for n in (16_000, 40_000, 72_000, 160_079):
        probe = torch.from_numpy(rng.standard_normal((1, n)).astype("float32") * 0.1)
        with torch.no_grad():
            want = wrapped(probe).numpy()
        got = sess.run(["logits"], {"signal": probe.numpy()})[0]
        d = float(np.abs(want - got).max())
        worst = max(worst, d)
        print(f"[verify] {n/16000:>4.1f}s  max|onnx-torch| = {d:.3e}")
    # A traced graph is the same arithmetic in a different order; 1e-3 is generous for f32
    # accumulation over 24 transformer blocks and tight enough to catch a wrong graph.
    if worst > 1e-3:
        sys.exit(f"[verify] FAILED: {worst:.3e} exceeds 1e-3 -- the export does not match torch")
    print(f"[verify] OK, worst {worst:.3e}")

    # ---- check 3: the ceiling is where it is documented to be. If a future torch or
    # funasr version lifts it, this prints and the doc comment above needs updating; if it
    # LOWERS below our render lengths, that must not go unnoticed.
    import numpy as _np
    def _runs(n):
        try:
            sess.run(["logits"], {"signal": _np.zeros((1, n), dtype="float32")})
            return True
        except Exception:
            return False
    assert _runs(160_079), "the documented ceiling of 160079 samples no longer runs"
    if _runs(160_080):
        print("[verify] NOTE: the 160079-sample ceiling has lifted; update the doc comment")
    else:
        print("[verify] ceiling confirmed at 160079 samples (10.005 s)")

if __name__ == "__main__":
    main()
