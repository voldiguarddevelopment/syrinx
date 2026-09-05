#!/usr/bin/env bash
# =============================================================================
# Syrinx — ONE-FILE end-to-end verification. Run this on the model box and it
# does everything: preflight → build → config → (download) → parity fixtures →
# test the WHOLE project (CV2 · CV3 · Fish s1-mini · Fish s2-pro · Qwen3-TTS · voice · emotion).
#
#   ./scripts/verify.sh                 full verify (uses scripts/test-all.env)
#   ./scripts/verify.sh --download      also download the Fish weights from HF first
#   ./scripts/verify.sh fish            verify one model family (or group, or test name)
#   ./scripts/verify.sh --group cv3     force the selector to be read as a group
#   ./scripts/verify.sh --exclude cv2   subtract a selector
#   ./scripts/verify.sh --list          groups + families
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

DOWNLOAD=0 QUICK=0
SYRINX_ROOT="$ROOT"
# shellcheck source=scripts/test-groups.sh
source "$ROOT/scripts/test-groups.sh"

# Selection shares its vocabulary (and its resolver) with scripts/test-all.sh.
SELECTED="" EXCLUDED="" BAD=0
while [ $# -gt 0 ]; do case "$1" in
  --download) DOWNLOAD=1 ;;
  --quick)    QUICK=1 ;;
  --list)     echo "groups:"; list_groups; echo "families:"; list_families; exit 0 ;;
  --group|--family|--test)
    kind="$1"; name="${2:?$1 needs a name}"; shift
    case "$kind" in
      --group)  is_group  "$name" || { echo "not a group: '$name'" >&2; BAD=1; }; ;;
      --family) is_family "$name" || { echo "not a family: '$name'" >&2; BAD=1; }; ;;
      --test)   is_test   "$name" || { echo "not a test: no tests/$name.rs, and not one of: $PSEUDO_TESTS" >&2; BAD=1; }; ;;
    esac
    got="$(resolve_selector "$name")" && SELECTED="$SELECTED $got" || BAD=1 ;;
  --exclude)
    name="${2:?--exclude needs a name}"; shift
    got="$(resolve_selector "$name")" && EXCLUDED="$EXCLUDED $got" || BAD=1 ;;
  -h|--help)  sed -n '2,20p' "$0"; exit 0 ;;
  -*) echo "unknown arg: $1 (try --help)"; exit 2 ;;
  *)  got="$(resolve_selector "$1")" && SELECTED="$SELECTED $got" || BAD=1 ;;
esac; shift; done
[ "$BAD" -eq 0 ] || { echo "refusing to verify: fix the selectors above" >&2; exit 2; }

c() { printf '\033[%sm%s\033[0m' "$1" "$2"; }
step() { printf '\n%s\n' "$(c '1;36' "══ $1 ══")"; }

# Groups and families come from scripts/test-groups.sh (sourced above) — one
# definition shared with scripts/test-all.sh, so the two runners cannot drift.
# --quick keeps only what needs no weights: those must never SKIP.
[ "$QUICK" = 1 ] && { SELECTED=""; for g in $(family_groups free); do SELECTED="$SELECTED $(group_tests "$g")"; done; }

# No selector means the whole board.
if [ -z "${SELECTED// /}" ]; then
  for g in $ALL_GROUPS; do SELECTED="$SELECTED $(group_tests "$g")"; done
fi
SELECTED="$(dedupe "$SELECTED")"
if [ -n "${EXCLUDED// /}" ]; then
  keep=""
  for t in $SELECTED; do
    skip=0; for x in $EXCLUDED; do [ "$t" = "$x" ] && { skip=1; break; }; done
    [ "$skip" -eq 0 ] && keep="$keep $t"
  done
  SELECTED="$keep"
fi
# Never verify nothing and call it a pass.
[ -n "${SELECTED// /}" ] || { echo "selection is empty — nothing would be verified" >&2; exit 2; }

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
# NOT run automatically. Dumping the fixtures loads the REFERENCE fish-speech model:
# the codec anchor costs ~2 GB, the s2-pro slow-AR anchor ~19 GB in CPU/f32 (the same
# footprint as real_fish_s2_e2e). Firing that off in the middle of a verify would race
# the suite below for memory, so this step only reports readiness and prints the exact
# commands. Run them once, under scripts/run-isolated.sh, then set SYRINX_FISH_*_REF.
if [ -n "${SYRINX_FISH_S1_REF:-}${SYRINX_FISH_S2_REF:-}" ]; then
  for v in S1 S2; do
    eval "p=\${SYRINX_FISH_${v}_REF:-}"
    [ -n "$p" ] || continue
    if [ -f "$p" ]; then echo "  $(c '32' "SYRINX_FISH_${v}_REF present") $p"
    else echo "  $(c '31' "SYRINX_FISH_${v}_REF set but missing") $p"; fi
  done
else
  echo "  $(c '1;33' 'no SYRINX_FISH_S1_REF / SYRINX_FISH_S2_REF — Fish PARITY tests will SKIP')"
  echo "  $(c '1;33' '   (the Fish e2e smoke tests still run)')"
fi
FISHPY="${SYRINX_FISH_REF_PY:-$HOME/refs/fish-speech/.venv/bin/python}"
FISHROOT="${FISH_SPEECH_ROOT:-$HOME/refs/fish-speech}"
if [ -x "$FISHPY" ]; then
  echo "  reference interpreter: $FISHPY"
else
  echo "  $(c '1;33' "no reference interpreter at $FISHPY")"
  echo "     cd $FISHROOT && uv venv --python 3.12 .venv && uv sync --extra cpu"
fi
echo "  to (re)generate — one at a time, never alongside another real-feature suite:"
echo "     MEMMAX=24G scripts/run-isolated.sh $FISHPY scripts/gen-fish-ref.py \\"
echo "        --variant s2-pro --ckpt \"\${SYRINX_FISH_S2_DIR:-/data/models/s2-pro}\" \\"
echo "        --out \$HOME/parity-fish/s2/ref.safetensors --fish-speech-root $FISHROOT"
echo "     (add --codec-only for the cheap ~2 GB codec anchor first)"

# ── 5. test everything ───────────────────────────────────────────────────────
step "5/6  test the whole project"
PASS=0 FAIL=0 SKIP=0 MISS=0; FAILED=""
LOG="$(mktemp)"; trap 'rm -f "$LOG"' EXIT
# One board row. Everything that varies BETWEEN rows — the cargo argv, whether a
# SKIP marker is the row's own verdict, what detail it reports — comes from
# scripts/test-groups.sh, so the `crate_unit_tests` workspace-lib row is not
# special-cased here and in test-all.sh separately.
run_one() {
  local t="$1"
  test_present "$t" || { printf '  %-40s %s\n' "$t" "$(c '2' 'MISSING')"; MISS=$((MISS+1)); return; }
  local -a targs=(); local a
  while IFS= read -r a; do targs+=("$a"); done < <(test_cargo_args "$t")
  cargo test --features real --release "${targs[@]}" -- --nocapture >"$LOG" 2>&1
  local rc=$?
  local detail; detail="$(test_detail "$t" "$LOG")"
  [ -n "$detail" ] && detail="  $(c '2' "($detail)")"
  if [ "$rc" -eq 0 ]; then
    # Case-SENSITIVE, trailing space required: the two real self-skip conventions are
    # `SKIP <name>: ...` and `skipping <name>: ...`. A case-insensitive bare `skip` also
    # matched a passing test's *name* (emotion_tags' concat_crossfade_skips_...), so a
    # green model-free test was reported as SKIP on every board.
    if test_can_skip "$t" && grep -qE 'SKIP |skipping ' "$LOG"; then
      printf '  %-40s %s%s\n' "$t" "$(c '33' SKIP)" "$detail"; SKIP=$((SKIP+1))
    else printf '  %-40s %s%s\n' "$t" "$(c '32' PASS)" "$detail"; PASS=$((PASS+1)); fi
  else printf '  %-40s %s%s\n' "$t" "$(c '31' FAIL)" "$detail"; FAIL=$((FAIL+1)); FAILED="$FAILED $t"; fi
}
shown=" "
for g in $ALL_GROUPS; do
  members=""
  for t in $(group_tests "$g"); do
    case " $SELECTED " in *" $t "*) members="$members $t" ;; esac
  done
  [ -z "${members// /}" ] && continue
  printf '\n  %s\n' "$(c '1;34' "── $g ──")"
  for t in $members; do run_one "$t"; shown="$shown$t "; done
done
loose=""
for t in $SELECTED; do case "$shown" in *" $t "*) ;; *) loose="$loose $t" ;; esac; done
if [ -n "${loose// /}" ]; then
  printf '\n  %s\n' "$(c '1;34' "── (ungrouped) ──")"
  for t in $loose; do run_one "$t"; done
fi

# ── 6. summary ───────────────────────────────────────────────────────────────
step "6/6  summary"
printf '  %s   %s   %s   %s\n' \
  "$(c '32' "PASS $PASS")" "$(c '33' "SKIP $SKIP")" "$(c '2' "MISSING $MISS")" "$(c '1;31' "FAIL $FAIL")"
if [ "$FAIL" -gt 0 ]; then
  printf '  %s%s\n' "$(c '1;31' 'failed:')" "$FAILED"
  echo "  $(c '1;31' 'VERIFICATION FAILED')"; exit 1
fi
echo "  $(c '1;32' 'VERIFIED') — all configured groups green (SKIP = unconfigured weights/fixtures, MISSING = test not built)"
