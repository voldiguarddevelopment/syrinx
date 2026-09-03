#!/usr/bin/env bash
# =============================================================================
# Syrinx — ONE-RUN test/parity suite for the WHOLE project.
#
#   CosyVoice2 · CosyVoice3 · Fish s1-mini · Fish s2-pro · Qwen3-TTS · voice · emotion · eval
#
# Run on the model box (weights + parity fixtures present). Each component group
# either PASSES, FAILS, or SKIPS (when its weights/fixtures aren't configured) —
# so a partial box still gives a clean report. Fill paths in scripts/test-all.env
# (copy from scripts/test-all.env.example) first.
#
#   ./scripts/test-all.sh                    # everything
#   ./scripts/test-all.sh fish               # one model family
#   ./scripts/test-all.sh cue qwen           # several selectors at once
#   ./scripts/test-all.sh real_fish_s2_e2e   # a single test by name
#   ./scripts/test-all.sh free --exclude cv2 # meta-family, minus something
#   ./scripts/test-all.sh --dry-run fish     # show what would run, run nothing
#   ./scripts/test-all.sh --list             # groups; --list-families for families
#   ./scripts/test-all.sh --compile-only     # build + compile tests, run nothing (off-box)
#   ./scripts/test-all.sh --download-fish    # hf download the Fish weights (s1-mini gated)
#
# A selector is a GROUP, a FAMILY, or a bare test name — resolved in that order,
# and the three namespaces never overlap. --group/--family/--test force one kind.
# An unknown selector is a hard error: it must never quietly run nothing.
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

# ---- groups / families -------------------------------------------------------
SYRINX_ROOT="$ROOT"
# shellcheck source=scripts/test-groups.sh
source "$ROOT/scripts/test-groups.sh"

# ---- modes -----------------------------------------------------------------
if [[ "${1:-}" == "--list" ]]; then
  echo "groups:";   list_groups
  echo "families:"; list_families
  exit 0
fi
if [[ "${1:-}" == "--list-families" ]]; then list_families; exit 0; fi
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then sed -n '2,32p' "$0"; exit 0; fi
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

# ---- selection ---------------------------------------------------------------
# Selectors accumulate; --exclude subtracts. Anything unresolvable stops the run.
SELECTED=""; EXCLUDED=""; DRY=0; BAD=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --group|--family|--test)
      kind="$1"; name="${2:?$1 needs a name}"; shift 2
      case "$kind" in
        --group)  is_group  "$name" || { echo "not a group: '$name' (groups: $ALL_GROUPS)" >&2; BAD=1; continue; } ;;
        --family) is_family "$name" || { echo "not a family: '$name' (families: $ALL_FAMILIES)" >&2; BAD=1; continue; } ;;
        --test)   is_test   "$name" || { echo "not a test: no tests/$name.rs" >&2; BAD=1; continue; } ;;
      esac
      got="$(resolve_selector "$name")" || { BAD=1; continue; }
      SELECTED="$SELECTED $got" ;;
    --exclude)
      name="${2:?--exclude needs a name}"; shift 2
      got="$(resolve_selector "$name")" || { BAD=1; continue; }
      EXCLUDED="$EXCLUDED $got" ;;
    -*) echo "unknown flag: $1" >&2; BAD=1; shift ;;
    *)
      got="$(resolve_selector "$1")" || { BAD=1; shift; continue; }
      SELECTED="$SELECTED $got"; shift ;;
  esac
done
[ "$BAD" -eq 0 ] || { echo "refusing to run: fix the selectors above" >&2; exit 2; }

# No selector means the whole board.
if [ -z "${SELECTED// /}" ]; then
  for g in $ALL_GROUPS; do SELECTED="$SELECTED $(group_tests "$g")"; done
fi
SELECTED="$(dedupe "$SELECTED")"
if [ -n "${EXCLUDED// /}" ]; then
  keep=""
  for t in $SELECTED; do
    skip=0
    for x in $EXCLUDED; do [ "$t" = "$x" ] && { skip=1; break; }; done
    [ "$skip" -eq 0 ] && keep="$keep $t"
  done
  SELECTED="$keep"
fi

# An empty selection is an error, never a green board: the old runner answered a
# typo'd --group with "PASS 0 ... all configured groups green" and exit 0.
if [ -z "${SELECTED// /}" ]; then
  echo "selection is empty (everything was excluded) — nothing would be tested" >&2
  exit 2
fi

if [ "$DRY" -eq 1 ]; then
  echo "would run $(echo $SELECTED | wc -w) test(s):"
  for t in $SELECTED; do printf '  %s\n' "$t"; done
  exit 0
fi

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
  elif grep -qE 'SKIP |skipping ' "$LOG"; then
    # Case-SENSITIVE, and the trailing space matters: the two real self-skip
    # conventions are `SKIP <name>: ...` and `skipping <name>: ...`. The old
    # case-insensitive bare `skip` also matched a passing test's *name* —
    # `concat_crossfade_skips_empty_segments_and_handles_none` in emotion_tags —
    # so a green model-free test was reported as SKIP on every board.
    printf '  %-40s \033[33mSKIP\033[0m\n' "$t"; ((SKIP++))
  else
    printf '  %-40s \033[32mPASS\033[0m\n' "$t"; ((PASS++))
  fi
}

echo "=============================================================="
echo " Syrinx full suite  ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "=============================================================="
# Print grouped, but drive off the resolved selection so a bare-test selector and
# an --exclude are both honoured. Anything selected that belongs to no group is
# reported under "(ungrouped)" rather than silently dropped.
shown=" "
for g in $ALL_GROUPS; do
  members=""
  for t in $(group_tests "$g"); do
    case " $SELECTED " in *" $t "*) members="$members $t" ;; esac
  done
  [ -z "${members// /}" ] && continue
  echo; echo "── group: $g ─────────────────────────────────────────────────"
  for t in $members; do run_one "$t"; shown="$shown$t "; done
done
loose=""
for t in $SELECTED; do
  case "$shown" in *" $t "*) ;; *) loose="$loose $t" ;; esac
done
if [ -n "${loose// /}" ]; then
  echo; echo "── (ungrouped) ───────────────────────────────────────────────"
  for t in $loose; do run_one "$t"; done
fi
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
