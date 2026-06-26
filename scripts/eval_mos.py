#!/usr/bin/env python3
"""MOS-proxy eval helper — predict a reference-free MOS for a synthesized WAV.

This is the **eval-only** quality bridge for `syrinx-eval`'s `mos_proxy` metric: the
pure-Rust inference path never touches Python, but a perceptual MOS estimate needs a
learned model. Uses **UTMOS** (UTMOS22, the VoiceMOS-Challenge-winning reference-free MOS
predictor) via `torch.hub` (`tarepan/SpeechMOS`). The Rust eval shells out to this script
when `SYRINX_MOS_HELPER` is set.

Usage:
    eval_mos.py <wav> [ignored...]

Prints a single float (predicted MOS, ~1..5) on the last stdout line. Loads the WAV as an
array to avoid the ffmpeg dependency, and resamples to 16 kHz (UTMOS's input rate).

Env:
    SYRINX_MOS_MODEL  torch.hub entry (default "utmos22_strong")
    SYRINX_MOS_REPO   torch.hub repo  (default "tarepan/SpeechMOS:v1.2.0")
"""
import os
import sys
import wave

import numpy as np
import torch
import librosa


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: eval_mos.py <wav>\n")
        return 2
    wav_path = sys.argv[1]
    repo = os.environ.get("SYRINX_MOS_REPO", "tarepan/SpeechMOS:v1.2.0")
    entry = os.environ.get("SYRINX_MOS_MODEL", "utmos22_strong")

    w = wave.open(wav_path, "rb")
    n, sr = w.getnframes(), w.getframerate()
    x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if sr != 16000:
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)

    predictor = torch.hub.load(repo, entry, trust_repo=True)
    predictor.eval()
    with torch.no_grad():
        wave_t = torch.from_numpy(x).unsqueeze(0)  # [1, T]
        score = predictor(wave_t, 16000)
    mos = float(score.reshape(-1)[0].item())
    sys.stderr.write(f"utmos: {mos:.3f}\n")
    print(f"{mos:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
