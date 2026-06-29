#!/usr/bin/env bash
# =============================================================================
# Syrinx — ONE-FILE end-to-end verification. Run this on the model box and it
# does everything: preflight → build → config → (download) → parity fixtures →
# test the WHOLE project (CV2 · CV3 · Fish s1-mini · Fish s2-pro · voice · emotion).
#
#   ./scripts/verify.sh                 full verify (uses scripts/test-all.env)
#   ./scripts/verify.sh --download      also download the Fish weights from HF first
#   ./scripts/verify.sh --group cv3     verify one group only (see the GROUPS list)
#   ./scripts/verify.sh --quick         build + model-free tests + compile-check, no heavy runs
#   ./scripts/verify.sh --help
#
# First run with no scripts/test-all.env: it creates one from the template and
# tells you to fill in your weight paths — the weight-backed groups SKIP until you do
# (build + model-free tests still run, so you always get a useful result).
#
# Exit 0 if nothing FAILED (SKIP/MISSING are fine); 1 if any test FAILED; 2 on setup error.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.." || { echo "cannot cd to repo root"; exit 2; }
ROOT="$(pwd)"

DOWNLOAD=0 QUICK=0 ONLY=""
while [ $# -gt 0 ]; do case "$1" in
  --download) DOWNLOAD=1 ;;
  --quick)    QUICK=1 ;;
  --group)    ONLY="${2:?--group needs a name}"; shift ;;
  -h|--help)  sed -n '2,17p' "$0"; exit 0 ;;
  *) echo "unknown arg: $1 (try --help)"; exit 2 ;;
esac; shift; done

c() { printf '\033[%sm%s\033[0m' "$1" "$2"; }
step() { printf '\n%s\n' "$(c '1;36' "══ $1 ══")"; }

# ── group → test files (kept in sync with scripts/test-all.sh) ───────────────
GROUP_modelfree="voice_lib emotion_tags watermark audio_server health_endpoint server_hardening real_cv3_quality_source real_cv3_multinomial"
GROUP_cv2="real_lm_parity real_lm_gen_parity real_lm_kvcache real_lm_quant real_feat_parity real_tokenizer_parity real_textnorm_parity real_speech_token_parity real_flow_parity real_flow_stream_consistency real_vocoder_parity real_speaker_parity real_token2wav_parity real_quant_footprint"
GROUP_cv2e2e="real_synth_e2e real_eval_metrics real_eval_multilingual real_lm_hammer"
GROUP_cv3="real_cv3_lm_parity real_cv3_flow_parity real_cv3_flow_stream_consistency real_cv3_hift_parity real_cv3_stok_parity real_cv3_ras real_cv3_quant_parity real_cv3_quant_footprint"
GROUP_cv3e2e="real_cv3_e2e_parity real_cv3_eval_metrics real_cv3_voice real_cv3_emotion"
GROUP_fish_s1="real_fish_s1_parity real_fish_s1_e2e"
GROUP_fish_s2="real_fish_s2_parity real_fish_s2_e2e"
# STT (pure-Rust Whisper): audio->text + the native TTS oracle. Self-skips off-box.
#   hf download openai/whisper-base --local-dir "$SYRINX_STT_MODEL_DIR"
GROUP_stt="real_stt"
ALL_GROUPS="modelfree cv2 cv2e2e cv3 cv3e2e fish_s1 fish_s2 stt"
[ "$QUICK" = 1 ] && ALL_GROUPS="modelfree"
group_tests() { local v="GROUP_$1"; echo "${!v:-}"; }

# ── 0. preflight ─────────────────────────────────────────────────────────────
step "0/6  preflight"
command -v cargo >/dev/null || { echo "$(c '1;31' FATAL): cargo/rust not found — install rustup"; exit 2; }
echo "  rust    : $(rustc --version 2>/dev/null || echo '?')"
command -v jq >/dev/null && echo "  jq      : yes" || echo "  jq      : no (sample runner has a fallback)"
command -v python >/dev/null && echo "  python  : $(python --version 2>&1)" || echo "  python  : no (Fish parity fixtures need it)"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/  gpu     : /'
else echo "  gpu     : none detected (CPU parity only; s2-pro will be slow/needs int4)"; fi

# ── 1. build (compile gate) ──────────────────────────────────────────────────
step "1/6  build (compile gate — nothing runs until this passes)"
if cargo build --features real 2>&1 | tail -2 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "$(c '1;31' FATAL): build failed — fix the compile errors above first"; exit 1
fi
echo "  $(c '1;32' 'build OK')"

# ── 2. config ────────────────────────────────────────────────────────────────
step "2/6  config (scripts/test-all.env)"
ENVF="$ROOT/scripts/test-all.env"
if [ ! -f "$ENVF" ]; then
  cp "$ROOT/scripts/test-all.env.example" "$ENVF"
  echo "  $(c '1;33' 'created scripts/test-all.env from the template.')"
  echo "  $(c '1;33' '>> EDIT it with your weight/fixture paths, then re-run for the weight-backed groups.')"
  echo "     (continuing now — unconfigured groups will SKIP, build + model-free still run)"
fi
# shellcheck disable=SC1090
source "$ENVF" 2>/dev/null && echo "  sourced scripts/test-all.env" || echo "  (could not source $ENVF)"

# ── 3. download Fish weights (opt-in) ────────────────────────────────────────
if [ "$DOWNLOAD" = 1 ]; then
  step "3/6  download Fish weights (~11 GB)"
  HF=""; command -v hf >/dev/null && HF=hf || { command -v huggingface-cli >/dev/null && HF=huggingface-cli; }
  if [ -n "$HF" ]; then
    "$HF" download fishaudio/openaudio-s1-mini --local-dir "${SYRINX_FISH_S1_DIR:-checkpoints/openaudio-s1-mini}" \
      || echo "  $(c '1;33' 's1-mini download failed — it is GATED: run `hf auth login` + accept the license')"
    "$HF" download fishaudio/s2-pro --local-dir "${SYRINX_FISH_S2_DIR:-checkpoints/s2-pro}" \
      || echo "  $(c '1;33' 's2-pro download failed')"
  else echo "  $(c '1;33' 'no hf CLI — run: pip install -U huggingface_hub  (gives the `hf` command)')"; fi
else
  step "3/6  download (skipped — pass --download to fetch Fish weights)"
fi

# ── 4. Fish parity fixtures ──────────────────────────────────────────────────
step "4/6  Fish parity fixtures (scripts/gen-fish-ref.py)"
if [ "$QUICK" = 1 ]; then
  echo "  (skipped in --quick)"
elif command -v python >/dev/null && [ -f "$ROOT/scripts/gen-fish-ref.py" ]; then
  if python "$ROOT/scripts/gen-fish-ref.py" 2>&1 | tail -4 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "  $(c '1;33' 'gen-fish-ref.py did not complete (likely the `# TODO(on-box)` model-load).')"
    echo "  $(c '1;33' '   → Fish PARITY tests will SKIP; the Fish e2e smoke tests still run.')"
  fi
else
  echo "  $(c '1;33' 'python or gen-fish-ref.py missing — Fish parity tests will SKIP')"
fi

# ── 5. test everything ───────────────────────────────────────────────────────
step "5/6  test the whole project"
PASS=0 FAIL=0 SKIP=0 MISS=0; FAILED=""
LOG="$(mktemp)"; trap 'rm -f "$LOG"' EXIT
run_one() {
  local t="$1"
  [ -f "$ROOT/tests/$t.rs" ] || { printf '  %-40s %s\n' "$t" "$(c '2' 'MISSING')"; MISS=$((MISS+1)); return; }
  if cargo test --features real --release --test "$t" -- --nocapture >"$LOG" 2>&1; then
    if grep -qiE 'skip|skipping' "$LOG"; then printf '  %-40s %s\n' "$t" "$(c '33' SKIP)"; SKIP=$((SKIP+1))
    else printf '  %-40s %s\n' "$t" "$(c '32' PASS)"; PASS=$((PASS+1)); fi
  else printf '  %-40s %s\n' "$t" "$(c '31' FAIL)"; FAIL=$((FAIL+1)); FAILED="$FAILED $t"; fi
}
for g in $ALL_GROUPS; do
  [ -n "$ONLY" ] && [ "$g" != "$ONLY" ] && continue
  printf '\n  %s\n' "$(c '1;34' "── $g ──")"
  for t in $(group_tests "$g"); do run_one "$t"; done
done

# ── 6. summary ───────────────────────────────────────────────────────────────
step "6/6  summary"
printf '  %s   %s   %s   %s\n' \
  "$(c '32' "PASS $PASS")" "$(c '33' "SKIP $SKIP")" "$(c '2' "MISSING $MISS")" "$(c '1;31' "FAIL $FAIL")"
if [ "$FAIL" -gt 0 ]; then
  printf '  %s%s\n' "$(c '1;31' 'failed:')" "$FAILED"
  echo "  $(c '1;31' 'VERIFICATION FAILED')"; exit 1
fi
echo "  $(c '1;32' 'VERIFIED') — all configured groups green (SKIP = unconfigured weights/fixtures, MISSING = test not built)"
