#!/usr/bin/env bash
# =============================================================================
# Syrinx — ONE-RUN test/parity suite for the WHOLE project.
#
#   CosyVoice2 · CosyVoice3 · Fish s1-mini · Fish s2-pro · voice · emotion · eval
#
# Run on the model box (weights + parity fixtures present). Each component group
# either PASSES, FAILS, or SKIPS (when its weights/fixtures aren't configured) —
# so a partial box still gives a clean report. Fill paths in scripts/test-all.env
# (copy from scripts/test-all.env.example) first.
#
#   ./scripts/test-all.sh                 # run every group, print a summary
#   ./scripts/test-all.sh --group cv3     # one group (see --list)
#   ./scripts/test-all.sh --list          # list groups + their tests
#   ./scripts/test-all.sh --compile-only  # build + compile tests, run nothing (off-box)
#   ./scripts/test-all.sh --download-fish # hf download the Fish weights (s1-mini gated: hf auth login)
#
# Exit code: 0 if nothing FAILED (passes + skips are fine), 1 if any group FAILED.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
ROOT="$(pwd)"

# ---- config ----------------------------------------------------------------
ENV_FILE="$ROOT/scripts/test-all.env"
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  echo "config: sourced $ENV_FILE"
else
  echo "config: no scripts/test-all.env (copy from .env.example) — groups will SKIP"
fi

CARGO_FLAGS="--features real --release"

# ---- group -> test files ----------------------------------------------------
# (model-free runs everywhere; the rest self-skip without their env vars)
GROUP_modelfree="voice_lib emotion_tags watermark audio_server health_endpoint server_hardening real_cv3_quality_source real_cv3_multinomial"
GROUP_cv2="real_lm_parity real_lm_gen_parity real_lm_kvcache real_lm_quant real_feat_parity real_tokenizer_parity real_textnorm_parity real_speech_token_parity real_flow_parity real_flow_stream_consistency real_vocoder_parity real_speaker_parity real_token2wav_parity real_quant_footprint"
GROUP_cv2e2e="real_synth_e2e real_eval_metrics real_eval_multilingual real_lm_hammer"
GROUP_cv3="real_cv3_lm_parity real_cv3_flow_parity real_cv3_flow_stream_consistency real_cv3_hift_parity real_cv3_stok_parity real_cv3_ras real_cv3_quant_parity real_cv3_quant_footprint"
GROUP_cv3e2e="real_cv3_e2e_parity real_cv3_eval_metrics real_cv3_voice real_cv3_emotion"
# Fish groups — test files are added by the syrinx-fish integration wave; the runner
# tolerates absent files (reports SKIP) so this script is stable as the port lands.
GROUP_fish_s1="real_fish_s1_parity real_fish_s1_e2e"
GROUP_fish_s2="real_fish_s2_parity real_fish_s2_e2e"
# STT (pure-Rust Whisper) — the audio->text reverse path + the native TTS oracle.
# Env-gated on a Whisper model dir + a test clip; self-skips off-box. Download:
#   hf download openai/whisper-base --local-dir "$SYRINX_STT_MODEL_DIR"
GROUP_stt="real_stt"

ALL_GROUPS="modelfree cv2 cv2e2e cv3 cv3e2e fish_s1 fish_s2 stt"

group_tests() { local v="GROUP_$1"; echo "${!v:-}"; }

# ---- modes -----------------------------------------------------------------
if [[ "${1:-}" == "--list" ]]; then
  for g in $ALL_GROUPS; do printf '  %-10s %s\n' "$g" "$(group_tests "$g")"; done
  exit 0
fi
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then sed -n '2,22p' "$0"; exit 0; fi
if [[ "${1:-}" == "--download-fish" ]]; then
  HF=""; command -v hf >/dev/null && HF=hf || { command -v huggingface-cli >/dev/null && HF=huggingface-cli; }
  [ -n "$HF" ] || { echo "need the hf CLI: pip install -U huggingface_hub"; exit 2; }
  echo ">> downloading fishaudio/openaudio-s1-mini (gated — needs 'hf auth login')"
  "$HF" download fishaudio/openaudio-s1-mini --local-dir "${SYRINX_FISH_S1_DIR:-/root/models/openaudio-s1-mini}"
  echo ">> downloading fishaudio/s2-pro (public, ~11 GB)"
  "$HF" download fishaudio/s2-pro --local-dir "${SYRINX_FISH_S2_DIR:-/root/models/s2-pro}"
  exit $?
fi
if [[ "${1:-}" == "--compile-only" ]]; then
  echo ">> compile-only: build + compile all tests, run nothing"
  cargo build $CARGO_FLAGS && cargo test $CARGO_FLAGS --no-run
  exit $?
fi

SELECT=""
if [[ "${1:-}" == "--group" ]]; then SELECT="${2:?--group needs a name}"; fi

# ---- run -------------------------------------------------------------------
PASS=0; FAIL=0; SKIP=0; MISS=0
declare -a FAILED_TESTS=()
LOG="$(mktemp)"

run_one() {
  local t="$1"
  if [ ! -f "$ROOT/tests/$t.rs" ]; then
    printf '  %-40s \033[2mMISSING (not built yet)\033[0m\n' "$t"; ((MISS++)); return
  fi
  cargo test $CARGO_FLAGS --test "$t" -- --nocapture >"$LOG" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    printf '  %-40s \033[31mFAIL\033[0m\n' "$t"; ((FAIL++)); FAILED_TESTS+=("$t")
  elif grep -qiE 'skip|skipping' "$LOG"; then
    printf '  %-40s \033[33mSKIP\033[0m\n' "$t"; ((SKIP++))
  else
    printf '  %-40s \033[32mPASS\033[0m\n' "$t"; ((PASS++))
  fi
}

echo "=============================================================="
echo " Syrinx full suite  ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "=============================================================="
for g in $ALL_GROUPS; do
  [ -n "$SELECT" ] && [ "$g" != "$SELECT" ] && continue
  echo; echo "── group: $g ─────────────────────────────────────────────────"
  for t in $(group_tests "$g"); do run_one "$t"; done
done
rm -f "$LOG"

echo
echo "=============================================================="
printf ' PASS %d   SKIP %d   MISSING %d   \033[1mFAIL %d\033[0m\n' "$PASS" "$SKIP" "$MISS" "$FAIL"
if [ "$FAIL" -gt 0 ]; then
  printf ' failed: %s\n' "${FAILED_TESTS[*]}"
  echo "=============================================================="
  exit 1
fi
echo " all configured groups green (skips = unconfigured weights/fixtures)"
echo "=============================================================="
exit 0
