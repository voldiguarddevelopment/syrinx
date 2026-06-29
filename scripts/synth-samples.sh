#!/usr/bin/env bash
#
# synth-samples.sh — drive the Fish Audio sample corpus through the Syrinx batch synth CLI.
#
# Reads samples/fish-samples.jsonl and renders every entry matching the chosen variant /
# scale / language / placement in a SINGLE CLI invocation that loads the 5 GB model (and,
# for s2-pro + a reference, encodes the reference) exactly ONCE:
#
#   cargo run -p syrinx-cli --features real -- \
#       synth --fish <variant> --fish-dir <CKPT> --batch <corpus.jsonl> \
#             --out-dir <DIR> --ref-wav <REF.wav> \
#             [--filter-lang L] [--filter-scale S] [--limit N] [--cuda]
#
# The script's filters map onto the CLI's native batch filters:
#   --lang  -> --filter-lang     --scale -> --filter-scale     --limit -> --limit
# Family (model) filtering (s2-pro => s2/both, s1-mini => s1/both) is done by the CLI.
# --placement has no native CLI filter, so it is pre-applied here by writing a temp,
# placement-filtered corpus (requires jq) that is then passed as --batch.
#
# The CLI writes the canonical <out-dir>/manifest.tsv
# (id <tab> scale <tab> lang <tab> text <tab> wav <tab> n_samples). This script keeps a
# model-free counts.txt + run-meta.txt summary so the corpus can be inspected off-box.
#
# The corpus is box-independent to *author*; actual synthesis runs on the GPU box.
# Pass --dry-run to print the single command that WOULD run, tagged [DRY RUN] — nothing
# is synthesized (no model load), but the matched counts are still produced.
#
# Usage:
#   scripts/synth-samples.sh <s1-mini|s2-pro> [options]
#
# Options:
#   --scale       small|reply|chapter   only synthesize this scale (-> --filter-scale)
#   --lang        L                     only this language code (-> --filter-lang)
#   --placement   P                     only this placement (pre-filtered here; needs jq)
#   --limit       N                     stop after N matching entries (-> --limit)
#   --ref         REF.wav               reference voice clip (default: $SYRINX_REF or none)
#   --prompt-text TEXT                  reference transcript (s2-pro; -> --prompt-text)
#   --max-steps   N                     LM frame cap (-> --max-steps)
#   --cuda                              run on GPU (requires a --features cuda build)
#   --out         DIR                   output dir (default: samples/out/<variant>)
#   --fish-dir    DIR                   checkpoint dir (default: $SYRINX_FISH_S{1,2}_DIR
#                                       or checkpoints/<variant-dir>)
#   --dry-run                           print the command that WOULD run; synthesize
#                                       nothing (off-box authoring; no model load)
#   -h|--help                           this help
#
# Outputs (under the chosen --out DIR):
#   manifest.tsv   id <tab> scale <tab> lang <tab> text <tab> wav <tab> n_samples  (CLI-written)
#   counts.txt     per-scale (and per-lang/per-placement) summary  (script-written)
#   run-meta.txt   host + gpu + toolchain specs                    (script-written)
#
set -euo pipefail

# ---------------------------------------------------------------------------
# locate repo root + corpus
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
JSONL="$REPO_ROOT/samples/fish-samples.jsonl"

die() { echo "synth-samples: $*" >&2; exit 1; }

usage() {
  sed -n '2,52p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

[[ $# -ge 1 ]] || usage 1
case "$1" in -h|--help) usage 0 ;; esac

VARIANT="$1"; shift
case "$VARIANT" in
  s1-mini) MODEL_FILTER="s1"; FISH_DIR_DEFAULT="${SYRINX_FISH_S1_DIR:-$REPO_ROOT/checkpoints/openaudio-s1-mini}" ;;
  s2-pro)  MODEL_FILTER="s2"; FISH_DIR_DEFAULT="${SYRINX_FISH_S2_DIR:-$REPO_ROOT/checkpoints/s2-pro}" ;;
  *) die "unknown variant '$VARIANT' (expected s1-mini or s2-pro)" ;;
esac

# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------
F_SCALE=""; F_LANG=""; F_PLACEMENT=""; LIMIT=0
REF="${SYRINX_REF:-}"
PROMPT_TEXT=""
MAX_STEPS=""
CUDA=0
OUT=""
FISH_DIR=""
DRYRUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scale)       F_SCALE="${2:?--scale needs a value}"; shift 2 ;;
    --lang)        F_LANG="${2:?--lang needs a value}"; shift 2 ;;
    --placement)   F_PLACEMENT="${2:?--placement needs a value}"; shift 2 ;;
    --limit)       LIMIT="${2:?--limit needs a value}"; shift 2 ;;
    --ref)         REF="${2:?--ref needs a value}"; shift 2 ;;
    --prompt-text) PROMPT_TEXT="${2:?--prompt-text needs a value}"; shift 2 ;;
    --max-steps)   MAX_STEPS="${2:?--max-steps needs a value}"; shift 2 ;;
    --cuda)        CUDA=1; shift ;;
    --out)         OUT="${2:?--out needs a value}"; shift 2 ;;
    --fish-dir)    FISH_DIR="${2:?--fish-dir needs a value}"; shift 2 ;;
    --dry-run)     DRYRUN=1; shift ;;
    -h|--help)     usage 0 ;;
    *) die "unknown option '$1' (see --help)" ;;
  esac
done
FISH_DIR="${FISH_DIR:-$FISH_DIR_DEFAULT}"

[[ -f "$JSONL" ]] || die "corpus not found: $JSONL"
OUT="${OUT:-$REPO_ROOT/samples/out/$VARIANT}"
mkdir -p "$OUT"
COUNTS="$OUT/counts.txt"

# ---------------------------------------------------------------------------
# Live vs dry run.
# ---------------------------------------------------------------------------
if [[ $DRYRUN -eq 1 ]]; then
  echo "synth-samples: --dry-run -> printing the single command that WOULD run; no audio is produced." >&2
fi
if [[ $DRYRUN -eq 0 && -z "$REF" ]]; then
  die "a reference voice is required to synthesize; pass --ref REF.wav (or set \$SYRINX_REF), or use --dry-run"
fi
REF_ARG="${REF:-<REF.wav>}"

# ---------------------------------------------------------------------------
# Placement pre-filter: the CLI has no --filter-placement, so when --placement is
# requested we materialize a temp, placement-filtered corpus and feed THAT to --batch.
# (jq required only for this case.) Otherwise the whole corpus is the batch input.
# ---------------------------------------------------------------------------
BATCH_CORPUS="$JSONL"
TMP_CORPUS=""
cleanup() { [[ -n "$TMP_CORPUS" && -f "$TMP_CORPUS" ]] && rm -f "$TMP_CORPUS"; }
trap cleanup EXIT
if [[ -n "$F_PLACEMENT" ]]; then
  command -v jq >/dev/null 2>&1 || die "--placement filtering needs jq (no native CLI filter); install jq or drop --placement"
  TMP_CORPUS="$(mktemp "${TMPDIR:-/tmp}/fish-samples.placement.XXXXXX.jsonl")"
  jq -c --arg p "$F_PLACEMENT" 'select(.placement == $p)' "$JSONL" > "$TMP_CORPUS"
  BATCH_CORPUS="$TMP_CORPUS"
fi

# ---------------------------------------------------------------------------
# Build the ONE batch command.
# ---------------------------------------------------------------------------
cmd=(cargo run -q -p syrinx-cli --features real -- \
     synth --fish "$VARIANT" --fish-dir "$FISH_DIR" \
     --batch "$BATCH_CORPUS" --out-dir "$OUT")
[[ -n "$REF" ]] && cmd+=(--ref-wav "$REF")
[[ -n "$PROMPT_TEXT" ]] && cmd+=(--prompt-text "$PROMPT_TEXT")
[[ -n "$F_LANG" ]] && cmd+=(--filter-lang "$F_LANG")
[[ -n "$F_SCALE" ]] && cmd+=(--filter-scale "$F_SCALE")
[[ "$LIMIT" -gt 0 ]] && cmd+=(--limit "$LIMIT")
[[ -n "$MAX_STEPS" ]] && cmd+=(--max-steps "$MAX_STEPS")
[[ $CUDA -eq 1 ]] && cmd+=(--cuda)

if [[ $DRYRUN -eq 1 ]]; then
  printf '[DRY RUN] '
  printf '%q ' "${cmd[@]}"; printf '\n'
else
  echo ">> rendering Fish '$VARIANT' batch (model loads ONCE) -> $OUT" >&2
  "${cmd[@]}"
fi

# ---------------------------------------------------------------------------
# Model-free counts (works off-box; mirrors the CLI's family+lang+scale+limit set so a
# --dry-run can preview how many entries the live batch will render). Placement is already
# applied to BATCH_CORPUS, so the scan reads BATCH_CORPUS.
# ---------------------------------------------------------------------------
emit_rows() {
  if command -v jq >/dev/null 2>&1; then
    jq -r --arg m "$MODEL_FILTER" '
      select(.model == $m or .model == "both")
      | [.id, .scale, .lang, .placement] | @tsv
    ' "$BATCH_CORPUS"
  else
    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      get() { printf '%s' "$line" | sed -n 's/.*"'"$1"'"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p'; }
      local id scale lang placement model
      id=$(get id); scale=$(get scale); lang=$(get lang)
      placement=$(get placement); model=$(get model)
      [[ "$model" == "$MODEL_FILTER" || "$model" == "both" ]] || continue
      printf '%s\t%s\t%s\t%s\n' "$id" "$scale" "$lang" "$placement"
    done < "$BATCH_CORPUS"
  fi
}

declare -A SC_COUNT LANG_COUNT PL_COUNT
total=0
while IFS=$'\t' read -r id scale lang placement; do
  [[ -n "$F_SCALE" && "$scale" != "$F_SCALE" ]] && continue
  [[ -n "$F_LANG"  && "$lang"  != "$F_LANG"  ]] && continue
  SC_COUNT["$scale"]=$(( ${SC_COUNT["$scale"]:-0} + 1 ))
  LANG_COUNT["$lang"]=$(( ${LANG_COUNT["$lang"]:-0} + 1 ))
  PL_COUNT["$placement"]=$(( ${PL_COUNT["$placement"]:-0} + 1 ))
  total=$(( total + 1 ))
  if [[ "$LIMIT" -gt 0 && "$total" -ge "$LIMIT" ]]; then break; fi
done < <(emit_rows)

# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
{
  echo "variant:   $VARIANT  (model filter: $MODEL_FILTER + both)"
  echo "filters:   scale=${F_SCALE:-*} lang=${F_LANG:-*} placement=${F_PLACEMENT:-*} limit=${LIMIT:-0}"
  echo "ref:       $REF_ARG"
  echo "fish-dir:  $FISH_DIR"
  echo "out dir:   $OUT"
  echo "mode:      $([[ $DRYRUN -eq 1 ]] && echo 'DRY RUN (--dry-run)' || echo 'LIVE synthesis')"
  echo "matched:   $total entries"
  echo
  echo "per-scale:"
  for k in small reply chapter; do printf '  %-8s %d\n' "$k" "${SC_COUNT[$k]:-0}"; done
  echo
  echo "per-language:"
  for k in $(printf '%s\n' "${!LANG_COUNT[@]}" | sort); do printf '  %-4s %d\n' "$k" "${LANG_COUNT[$k]}"; done
  echo
  echo "per-placement:"
  for k in $(printf '%s\n' "${!PL_COUNT[@]}" | sort); do printf '  %-12s %d\n' "$k" "${PL_COUNT[$k]}"; done
} | tee "$COUNTS"

# ---------------------------------------------------------------------------
# run metadata — capture the BOX SPECS + toolchain so every batch of audio is
# reproducible and attributable to the host it was synthesized on. Written even
# on a dry run (where it documents the authoring host instead of the GPU box).
# ---------------------------------------------------------------------------
META="$OUT/run-meta.txt"
have() { command -v "$1" >/dev/null 2>&1; }
{
  echo "# synth-samples run metadata"
  echo "timestamp:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "variant:     $VARIANT (model $MODEL_FILTER + both)"
  echo "mode:        $([[ $DRYRUN -eq 1 ]] && echo 'DRY RUN' || echo 'LIVE')"
  echo "filters:     scale=${F_SCALE:-*} lang=${F_LANG:-*} placement=${F_PLACEMENT:-*} limit=${LIMIT:-0}"
  echo "ref:         $REF_ARG"
  echo "fish-dir:    $FISH_DIR"
  echo "cuda:        $([[ $CUDA -eq 1 ]] && echo 'yes (--cuda)' || echo 'no (CPU)')"
  echo "matched:     $total entries"
  echo
  echo "## host"
  echo "hostname:    $(hostname 2>/dev/null || echo '?')"
  echo "user:        ${USER:-?}"
  echo "uname:       $(uname -a 2>/dev/null || echo '?')"
  if [[ -r /etc/os-release ]]; then echo "os:          $(. /etc/os-release; echo "$PRETTY_NAME")"; fi
  echo
  echo "## cpu / memory"
  if have lscpu; then lscpu | grep -E '^(Model name|Socket|Core|Thread|CPU\(s\))' | sed 's/^/  /'; fi
  if have nproc; then echo "  nproc:     $(nproc)"; fi
  if have free;  then echo "  mem:       $(free -h | awk '/^Mem:/{print $2" total, "$7" avail"}')"; fi
  echo
  echo "## gpu"
  if have nvidia-smi; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
      --format=csv,noheader 2>/dev/null | sed 's/^/  /' || nvidia-smi 2>&1 | sed 's/^/  /'
  else
    echo "  (no nvidia-smi on this host)"
  fi
  echo
  echo "## toolchain"
  have rustc && echo "  rustc:     $(rustc --version)"
  have cargo && echo "  cargo:     $(cargo --version)"
  if have nvcc; then echo "  nvcc:      $(nvcc --version | tail -1)"; fi
  echo
  echo "## repo"
  echo "  commit:    $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"
  echo "  branch:    $(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  echo "  corpus:    $JSONL ($(wc -l < "$JSONL" 2>/dev/null || echo '?') entries)"
} > "$META"

echo
echo "manifest  -> $OUT/manifest.tsv   (written by the synth CLI on a live run)"
echo "counts    -> $COUNTS"
echo "run-meta  -> $META   (host + gpu + toolchain specs)"
