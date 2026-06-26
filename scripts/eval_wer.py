#!/usr/bin/env python3
"""WER/CER eval helper — ASR a synthesized WAV with Whisper, print the error rate.

This is the **eval-only** ASR bridge for `syrinx-eval`'s `wer` metric: the pure-Rust
inference path never touches Python, but measuring word/character error rate needs an
ASR model, and the box already has `openai-whisper`. The Rust eval (`syrinx_eval::real`)
shells out to this script when `SYRINX_WER_HELPER` is set.

Usage:
    eval_wer.py <wav> <reference_text>

Prints a single float (the error rate, CER for CJK / WER for space-delimited scripts)
on the last stdout line, so the caller can parse it. Loads the WAV as an array to
avoid the ffmpeg dependency `whisper.transcribe(path)` would otherwise need.

Env:
    SYRINX_WER_MODEL  whisper model name (default "small")
    SYRINX_WER_LANG   language hint (default "zh")
"""
import os
import sys
import wave
import unicodedata

import numpy as np
import whisper
import librosa


def _normalize(s: str) -> str:
    """Strip punctuation + whitespace so the rate measures content, not formatting."""
    return "".join(c for c in s if not unicodedata.category(c).startswith("P")).replace(" ", "")


def _edit_rate(ref: str, hyp: str, tokens) -> float:
    r = tokens(ref)
    h = tokens(hyp)
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + (r[i - 1] != h[j - 1]),
            )
    return d[len(r)][len(h)] / max(1, len(r))


def error_rate(ref: str, hyp: str, lang: str) -> float:
    # CJK has no word boundaries -> character error rate (on normalized text);
    # otherwise word error rate (token = whitespace-split word).
    if lang in ("zh", "ja", "yue"):
        return _edit_rate(_normalize(ref), _normalize(hyp), tokens=list)
    return _edit_rate(ref.lower(), hyp.lower(), tokens=str.split)


def main() -> int:
    if len(sys.argv) < 3:
        sys.stderr.write("usage: eval_wer.py <wav> <reference_text>\n")
        return 2
    wav_path, ref = sys.argv[1], sys.argv[2]
    model_name = os.environ.get("SYRINX_WER_MODEL", "small")
    lang = os.environ.get("SYRINX_WER_LANG", "zh")

    w = wave.open(wav_path, "rb")
    n, sr = w.getnframes(), w.getframerate()
    x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if sr != 16000:
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)

    hyp = whisper.load_model(model_name).transcribe(x, language=lang)["text"]
    rate = error_rate(ref, hyp, lang)
    sys.stderr.write(f"ref: {ref}\nhyp: {hyp.strip()}\n")
    print(f"{rate:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
