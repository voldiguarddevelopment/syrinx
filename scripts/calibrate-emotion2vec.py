#!/usr/bin/env python3
"""Cross-corpus calibration for the emotion2vec+ judge, on CREMA-D.

A judge's number means nothing until you know how often it is right, and on WHICH classes.
The previous judge's card claimed 0.822 and measured **0.394** here, worthless on `sad`
(0.17) and `fearful` (0.10) — which is why `[sad]`, the one cue with a demonstrated
acoustic effect, could not be adjudicated at all.

CREMA-D (ODbL-1.0, commercial use permitted) is used rather than the training corpus of any
candidate: measuring a model on its own training set reports memorisation as accuracy.
The 180-clip probe subset is 6 classes x 30, balanced.

Prints per-class recall. Quote it beside any verdict this judge is used to support.

Usage: calibrate-emotion2vec.py <model-dir> <crema-dir>
"""
import sys, os, glob, warnings, collections
warnings.filterwarnings("ignore")

# CREMA-D filename code -> emotion2vec+ class. `NEU` maps to `neutral`; CREMA-D has no
# `surprised`, so two of the model's nine classes are simply not probed and this script
# says so rather than reporting a recall it did not measure.
CREMA = {"ANG": "angry", "DIS": "disgusted", "FEA": "fearful",
         "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    model_dir, crema = sys.argv[1], sys.argv[2]

    import numpy as np, torch, soundfile as sf, scipy.signal as ss
    from funasr import AutoModel

    net = AutoModel(model=model_dir, disable_update=True, device="cpu").model
    net.eval()

    def logits(path):
        x, sr = sf.read(path, dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        if sr != 16000:
            x = ss.resample_poly(x, 16000, sr).astype("float32")
        s = torch.from_numpy(np.ascontiguousarray(x)).view(1, -1)
        if net.cfg.normalize:
            m, v = s.mean(dim=1, keepdim=True), s.var(dim=1, unbiased=False, keepdim=True)
            s = (s - m) / torch.sqrt(v + 1e-5)
        with torch.no_grad():
            f = net.extract_features(s, padding_mask=None)["x"]
            return net.proj(f.mean(dim=1))[0].numpy()

    files = sorted(glob.glob(os.path.join(crema, "*.wav")))
    if not files:
        sys.exit(f"no wavs under {crema}")

    hit = collections.Counter()
    tot = collections.Counter()
    confusion = collections.Counter()
    for p in files:
        code = os.path.basename(p).split("_")[2]
        truth = CREMA.get(code)
        if truth is None:
            continue
        pred = LABELS[int(logits(p).argmax())]
        tot[truth] += 1
        hit[truth] += int(pred == truth)
        confusion[(truth, pred)] += 1

    n = sum(tot.values())
    print(f"\n=== emotion2vec+ large on CREMA-D, n={n} ===")
    print(f"{'class':<12}{'recall':>8}{'n':>5}   most common error")
    for cls in sorted(tot):
        errs = [(v, p) for (t, p), v in confusion.items() if t == cls and p != cls]
        worst = max(errs)[1] if errs else "-"
        print(f"{cls:<12}{hit[cls]/tot[cls]:>8.3f}{tot[cls]:>5}   {worst}")
    print(f"{'OVERALL':<12}{sum(hit.values())/n:>8.3f}{n:>5}")
    print("\nNot probed by CREMA-D: surprised, other, unknown "
          "(the corpus has no such clips; no recall is claimed for them).")


if __name__ == "__main__":
    main()
