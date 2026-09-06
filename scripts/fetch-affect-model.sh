#!/usr/bin/env bash
# =============================================================================
# fetch-affect-model.sh — acquire the affect judge and export it to ONNX.
#
#   ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
#   wav2vec2-large-xlsr-53-english fine-tuned on RAVDESS, 8 emotion classes:
#   angry, calm, disgust, fearful, happy, neutral, sad, surprised.
#
#   LICENCE: Apache-2.0 (verified on the model card's YAML front matter and the
#   Hub API's cardData.license). Commercial use permitted, which is the whole
#   reason it is here — see docs/LICENSES.md. audEERING's dimensional model was
#   the better technical fit and was REJECTED on licence (CC-BY-NC-SA-4.0).
#
# Three steps, all idempotent:
#
#   1. HF snapshot   config + weights, pinned to a commit.
#   2. ONNX export   the checkpoint ships no ONNX, so we export it ourselves.
#                    `scripts/export-affect-onnx.py` bakes in the real (non-stock)
#                    classification head and the feature extractor, and verifies the
#                    result against the torch model before returning.
#   3. Fixture       `scripts/gen-affect-ref.py` dumps the torch reference the Rust
#                    anchor in `tests/real_qwen_affect.rs` compares against.
#
# Usage:  scripts/fetch-affect-model.sh [dest-dir]
#         SKIP_EXPORT=1 scripts/fetch-affect-model.sh   (weights only)
# =============================================================================
set -euo pipefail

REPO="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
# Pinned so a re-fetch is the same bytes. HF main as of 2026-09-06.
REV="b520c9c46a719e36e1b9a91cad2cb5d0668757d8"
DEST="${1:-/data/models/w2v2-lg-xlsr-en-ser}"
PY="${PYTHON:-/home/floofy/.venvs/qwen/bin/python}"
# `torch.onnx.export` needs the `onnx` package. It is deliberately NOT installed into the
# qwen venv (that env belongs to the Qwen work); it lives beside it and is added to
# PYTHONPATH for the export step only.
ONNX_LIBS="${ONNX_LIBS:-/home/floofy/.venvs/affect-libs}"
REF_OUT="${REF_OUT:-$HOME/parity-affect/affect.safetensors}"

# Measured on the artefact this script fetched on 2026-09-06. A mismatch means upstream
# changed or the download truncated — investigate, never edit past it. The ONNX is NOT
# pinned: it is produced locally and its bytes depend on the torch version, so
# `export-affect-onnx.py` verifies it numerically against torch instead.
SHA_SAFETENSORS="33bd858e5a4dc3241a60e99dd46818bead624f889244ce5d6b6618426915fea0"

say() { printf '[fetch-affect-model] %s\n' "$*" >&2; }
die() { printf '[fetch-affect-model] FATAL: %s\n' "$*" >&2; exit 1; }

mkdir -p "$DEST"
say "destination: $DEST"

# ---- 1. HF snapshot ----------------------------------------------------------
BASE="https://huggingface.co/$REPO/resolve/$REV"
for f in config.json preprocessor_config.json README.md model.safetensors; do
  if [ -s "$DEST/$f" ]; then
    say "have $f"
  else
    say "GET  $f"
    curl -sSLf -o "$DEST/$f" "$BASE/$f" || die "download of $f failed"
  fi
done

got=$(sha256sum "$DEST/model.safetensors" | cut -d' ' -f1)
[ "$got" = "$SHA_SAFETENSORS" ] || die "sha256(model.safetensors) = $got, expected $SHA_SAFETENSORS"
say "sha256 ok: model.safetensors"

# The licence claim this whole choice rests on, re-checked from the Hub rather than
# trusted from a comment.
lic=$(curl -sSLf "https://huggingface.co/api/models/$REPO" | "$PY" -c \
  'import json,sys; print(json.load(sys.stdin).get("cardData",{}).get("license"))' 2>/dev/null || echo "?")
say "hub-reported licence: $lic"
[ "$lic" = "apache-2.0" ] || die "expected apache-2.0 on the Hub, got '$lic' — STOP and re-screen"

if [ "${SKIP_EXPORT:-0}" = "1" ]; then
  say "SKIP_EXPORT=1: stopping after the weights"
  exit 0
fi

# ---- 2. ONNX export ----------------------------------------------------------
if [ -s "$DEST/model.onnx" ]; then
  say "have model.onnx"
else
  [ -d "$ONNX_LIBS" ] || {
    say "installing the onnx package beside the qwen venv -> $ONNX_LIBS"
    "$PY" -m pip install --quiet --target "$ONNX_LIBS" onnx || die "pip install onnx failed"
  }
  say "exporting ONNX (a few minutes; ~1.3 GB out)"
  PYTHONPATH="$ONNX_LIBS" MEMMAX="${MEMMAX:-10G}" "$(dirname "$0")/run-isolated.sh" \
    "$PY" "$(dirname "$0")/export-affect-onnx.py" --model "$DEST" \
    || die "ONNX export failed"
fi

# ---- 3. the python reference fixture ----------------------------------------
if [ -s "$REF_OUT" ]; then
  say "have $REF_OUT"
else
  say "capturing the torch reference -> $REF_OUT"
  MEMMAX="${MEMMAX:-10G}" "$(dirname "$0")/run-isolated.sh" \
    "$PY" "$(dirname "$0")/gen-affect-ref.py" \
    --model "$DEST" --out "$REF_OUT" \
    renders/2026-09-06-qwen-emotion-ab/*.wav \
    /home/floofy/refs/voice_en_10s.wav \
    /data/datasets/crema-d-probe/1001_DFA_ANG_XX.wav \
    /data/datasets/crema-d-probe/1001_DFA_HAP_XX.wav \
    /data/datasets/crema-d-probe/1001_DFA_SAD_XX.wav \
    /data/datasets/crema-d-probe/1001_DFA_NEU_XX.wav \
    || die "reference capture failed (CREMA-D calibration clips: run scripts/probe-affect-head.py --download)"
fi

cat >&2 <<EOF
[fetch-affect-model] done.

  export SYRINX_AFFECT_ONNX=$DEST/model.onnx
  export SYRINX_AFFECT_REF=$REF_OUT

  cargo test --features affect --test real_qwen_affect -- --nocapture
  cargo run  --features affect -p syrinx-eval --example affect --release -- \\
      "\$SYRINX_AFFECT_ONNX" renders/2026-09-06-qwen-emotion-ab/*.wav

Licence: Apache-2.0. The CREMA-D clips used for calibration are ODbL-1.0.
EOF
