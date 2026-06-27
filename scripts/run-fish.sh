#!/usr/bin/env bash
#
# run-fish.sh — one-shot runner for the Fish Audio (syrinx-fish) TTS port.
#
# SKELETON (foundation wave). The arg parsing + the `cargo run` invocation SHAPE are
# scaffolded here; the integration wave wires the actual `syrinx synth --fish` front
# door in `crates/syrinx-cli` and fills the TODOs below.
#
# Usage:
#   scripts/run-fish.sh <s1-mini|s2-pro> "<text>" <ref.wav> [out.wav]
#   scripts/run-fish.sh --parity <s1-mini|s2-pro> <ref.wav>      # on-box parity check
#
# Notes:
#   * NON-COMMERCIAL weights (see crate docs). Point SYRINX_FISH_CKPT at the local
#     checkpoint dir (default: checkpoints/<variant-dir>).
#   * The codec emits 44.1 kHz mono; <out.wav> defaults to fish-out.wav.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PARITY=0
if [[ "${1:-}" == "--parity" ]]; then
    PARITY=1
    shift
fi

VARIANT="${1:-}"
case "$VARIANT" in
    s1-mini|s2-pro) ;;
    *)
        echo "usage: run-fish.sh [--parity] <s1-mini|s2-pro> \"<text>\" <ref.wav> [out.wav]" >&2
        exit 2
        ;;
esac

# Checkpoint dir: the variant's canonical name under checkpoints/, overridable.
case "$VARIANT" in
    s1-mini) CKPT_DIR_DEFAULT="checkpoints/openaudio-s1-mini" ;;
    s2-pro)  CKPT_DIR_DEFAULT="checkpoints/s2-pro" ;;
esac
CKPT="${SYRINX_FISH_CKPT:-$CKPT_DIR_DEFAULT}"

if [[ "$PARITY" == "1" ]]; then
    REF_WAV="${2:-}"
    [[ -n "$REF_WAV" ]] || { echo "--parity needs <ref.wav>" >&2; exit 2; }
    # TODO(integration wave): a `--parity` mode that encodes <ref.wav> to codes, decodes
    # back, and compares the round-trip + a fixed-seed generation against the Python
    # reference dump (codes_*.npy / fake.wav) saved on-box. For now just a placeholder
    # so the harness shape is fixed.
    echo "[run-fish] PARITY mode is a placeholder until the integration wave (variant=$VARIANT, ckpt=$CKPT, ref=$REF_WAV)" >&2
    exit 0
fi

TEXT="${2:-}"
REF_WAV="${3:-}"
OUT="${4:-fish-out.wav}"
if [[ -z "$TEXT" || -z "$REF_WAV" ]]; then
    echo "usage: run-fish.sh <s1-mini|s2-pro> \"<text>\" <ref.wav> [out.wav]" >&2
    exit 2
fi

# Invocation SHAPE the integration wave wires up: a `--fish <variant>` front door on the
# existing `syrinx synth` command (mirrors the `--cv3` switch). The reference voice is
# resampled to 44.1 kHz and encoded to prompt codes by the s1/s2 codec; emotion/style
# tags in <text> (e.g. "[happy]") are plain text.
#
# TODO(integration wave): implement `synth --fish` in crates/syrinx-cli and drop the echo.
echo "[run-fish] would run: cargo run --release -p syrinx-cli --features real -- \\" >&2
echo "             synth --fish $VARIANT --ckpt $CKPT --ref-wav $REF_WAV --text \"$TEXT\" --out $OUT" >&2

# cargo run --release -p syrinx-cli --features real -- \
#     synth --fish "$VARIANT" --ckpt "$CKPT" \
#     --ref-wav "$REF_WAV" --text "$TEXT" --out "$OUT"
