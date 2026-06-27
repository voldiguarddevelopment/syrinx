#!/usr/bin/env bash
#
# run-fish.sh — one-shot runner for the Fish Audio (syrinx-fish) TTS port.
#
# Wires the real `syrinx synth --fish` front door (mirrors the `--cv3` switch) and an
# on-box `--parity` mode that runs the `real_fish_*` tests with the env from
# scripts/test-all.env.
#
# Usage:
#   scripts/run-fish.sh <s1-mini|s2-pro> "<text>" <ref.wav> [out.wav]
#   scripts/run-fish.sh --parity <s1-mini|s2-pro> [ref.wav]      # on-box parity + e2e
#
# Notes:
#   * NON-COMMERCIAL weights (see crate docs). Point SYRINX_FISH_CKPT at the local
#     checkpoint dir (default: checkpoints/<variant-dir>).
#   * The codec emits 44.1 kHz mono; <out.wav> defaults to fish-out.wav.
#   * s1-mini has no reference-cloning path: <ref.wav> is passed but the CLI ignores it
#     (text-only synthesis). s2-pro clones from <ref.wav>; set SYRINX_FISH_REF_TEXT to
#     supply the optional reference transcript (--prompt-text).
#
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

# ---------------------------------------------------------------------------
# --parity: run the variant's real_fish_* parity + e2e tests with the on-box
# env from scripts/test-all.env (weights + parity-fixture paths). The tests
# self-skip cleanly when their env vars are unset, so this is safe to run on a
# partial box.
# ---------------------------------------------------------------------------
if [[ "$PARITY" == "1" ]]; then
    ENV_FILE="$REPO_ROOT/scripts/test-all.env"
    if [[ -f "$ENV_FILE" ]]; then
        # shellcheck disable=SC1090
        source "$ENV_FILE"
        echo "[run-fish] sourced $ENV_FILE for SYRINX_FISH_* paths" >&2
    else
        echo "[run-fish] no scripts/test-all.env — tests will SKIP without their env vars" >&2
    fi
    # An optional <ref.wav> overrides the e2e reference voice.
    REF_WAV="${2:-}"
    if [[ -n "$REF_WAV" ]]; then
        export SYRINX_FISH_REF_WAV="$REF_WAV"
        echo "[run-fish] SYRINX_FISH_REF_WAV=$REF_WAV (e2e clone path)" >&2
    fi
    case "$VARIANT" in
        s1-mini) TESTS=(real_fish_s1_parity real_fish_s1_e2e) ;;
        s2-pro)  TESTS=(real_fish_s2_parity real_fish_s2_e2e) ;;
    esac
    rc=0
    for t in "${TESTS[@]}"; do
        echo "── $t ──────────────────────────────────────────────" >&2
        cargo test --features real --release --test "$t" -- --nocapture || rc=$?
    done
    exit "$rc"
fi

# ---------------------------------------------------------------------------
# synth mode: build + invoke the real `synth --fish` CLI.
# ---------------------------------------------------------------------------
TEXT="${2:-}"
REF_WAV="${3:-}"
OUT="${4:-fish-out.wav}"
if [[ -z "$TEXT" || -z "$REF_WAV" ]]; then
    echo "usage: run-fish.sh <s1-mini|s2-pro> \"<text>\" <ref.wav> [out.wav]" >&2
    exit 2
fi

# The reference transcript (optional; s2-pro cloning). Empty => the codec audio codes
# still condition the cloned voice.
PROMPT_ARGS=()
if [[ -n "${SYRINX_FISH_REF_TEXT:-}" ]]; then
    PROMPT_ARGS=(--prompt-text "$SYRINX_FISH_REF_TEXT")
fi

echo "[run-fish] synth --fish $VARIANT  ckpt=$CKPT  ref=$REF_WAV  out=$OUT" >&2
exec cargo run --release -p syrinx-cli --features real -- \
    synth --fish "$VARIANT" --fish-dir "$CKPT" \
    --ref-wav "$REF_WAV" --text "$TEXT" --out "$OUT" "${PROMPT_ARGS[@]}"
