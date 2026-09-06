#!/usr/bin/env python3
"""Dump the emotion2vec+ torch reference the Rust judge is anchored against.

Same discipline as `gen-fish-ref.py` and `gen-affect-ref.py`: the reference is the upstream
package's OWN forward pass (funasr's `Emotion2vec`), never a reimplementation. What Rust
must reproduce is what funasr computes, so funasr is what gets dumped.

Keys, per input stem:

  wav_in/<stem>   [n]    f32  the file exactly as read
  sr_in/<stem>    [1]    i64  its sample rate (safetensors carries tensors, not ints)
  wav16/<stem>    [m]    f32  after resampling to 16 kHz -- the graph's actual input
  logits/<stem>   [1,9]  f32  proj(features.mean(1)), pre-softmax. THE anchor.
  probs/<stem>    [1,9]  f32  softmax(logits), dumped for completeness and NOT the anchor:
                              the model saturates, so these are one-hot to float precision
                              and would compare equal for two clearly different clips.

Usage:
  gen-emotion2vec-ref.py --model /data/models/emotion2vec-plus-large \\
                         --out $HOME/parity-affect/emotion2vec.safetensors \\
                         wav [wav ...]
"""
import argparse, os, sys, warnings
warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("wavs", nargs="+")
    a = ap.parse_args()

    import numpy as np, torch, soundfile as sf, scipy.signal as ss
    from safetensors.torch import save_file
    from funasr import AutoModel

    net = AutoModel(model=a.model, disable_update=True, device="cpu").model
    net.eval()
    labels = [l.split("/")[-1] for l in open(os.path.join(a.model, "tokens.txt")).read().split()]
    labels = [l.strip("<>") if l.startswith("<") else l for l in labels]
    print(f"[ref] labels: {labels}")

    tensors, seen = {}, set()
    for path in a.wavs:
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem in seen:
            sys.exit(f"[ref] duplicate stem {stem!r} -- safetensors keys must be unique")
        seen.add(stem)

        x, sr = sf.read(path, dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        x16 = x if sr == 16000 else ss.resample_poly(x, 16000, sr).astype("float32")
        # `.copy()`/ascontiguousarray matters even without resampling: safetensors refuses
        # to write two keys that alias the same buffer.
        # `.copy()` matters even when no resampling happened: `x16` would then BE `x`, and
        # safetensors refuses to write two keys backed by the same buffer. gen-affect-ref.py
        # carries the same note for the same reason.
        x16 = np.ascontiguousarray(x16, dtype=np.float32).copy()

        s = torch.from_numpy(x16).view(1, -1)
        # funasr's own path: layer_norm over the waveform, features, mean-pool, project.
        if net.cfg.normalize:
            m, v = s.mean(dim=1, keepdim=True), s.var(dim=1, unbiased=False, keepdim=True)
            s = (s - m) / torch.sqrt(v + 1e-5)
        with torch.no_grad():
            feats = net.extract_features(s, padding_mask=None)["x"]
            logits = net.proj(feats.mean(dim=1))
            probs = torch.softmax(logits, dim=-1)

        tensors[f"wav_in/{stem}"] = torch.from_numpy(np.ascontiguousarray(x))
        tensors[f"sr_in/{stem}"] = torch.tensor([sr], dtype=torch.int64)
        tensors[f"wav16/{stem}"] = torch.from_numpy(x16)
        tensors[f"logits/{stem}"] = logits.contiguous()
        tensors[f"probs/{stem}"] = probs.contiguous()
        top = labels[int(logits.argmax())]
        print(f"[ref] {stem:<24} {len(x16)/16000:>5.2f}s  top={top:<10} "
              f"spread={float(logits.max()-logits.min()):.2f}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_file(tensors, a.out, metadata={"labels": ",".join(labels),
                                        "model": "emotion2vec/emotion2vec_plus_large"})
    print(f"[ref] wrote {a.out}  ({len(seen)} clips)")


if __name__ == "__main__":
    main()
