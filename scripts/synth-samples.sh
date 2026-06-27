#!/usr/bin/env bash
#
# synth-samples.sh — drive the Fish Audio sample corpus through the Syrinx synth CLI.
#
# Reads samples/fish-samples.jsonl, filters by model variant / scale / language /
# placement, and for every matching entry invokes the Fish synth front door:
#
#   cargo run -p syrinx-cli --features real -- \
#       synth --fish <variant> --fish-dir <CKPT> --text "<text>" \
#             --ref-wav <REF.wav> --out <DIR>/<id>.wav
#
# The corpus is box-independent to *author*; actual synthesis runs on the GPU box.
# Pass --dry-run to print each line as the command it WOULD run, tagged [DRY RUN] —
# nothing is synthesized (no model load), but the manifest + counts are still produced
# so the corpus can be inspected anywhere off-box.
#
# Usage:
#   scripts/synth-samples.sh <s1-mini|s2-pro> [options]
#
# Options:
#   --scale     small|reply|chapter   only synthesize this scale
#   --lang      L                     only this language code (en, zh, ja, ...)
#   --placement P                     only this placement (leading, mid, ...)
#   --limit     N                     stop after N matching entries
#   --ref       REF.wav               reference voice clip (default: $SYRINX_REF or none)
#   --out       DIR                   output dir (default: samples/out/<variant>)
#   --fish-dir  DIR                   checkpoint dir (default: $SYRINX_FISH_S{1,2}_DIR
#                                     or checkpoints/<variant-dir>)
#   --dry-run                         print the commands that WOULD run; synthesize
#                                     nothing (off-box authoring; no model load)
#   -h|--help                         this help
#
# Outputs (under the chosen --out DIR):
#   manifest.tsv   id <tab> scale <tab> lang <tab> placement <tab> text <tab> wav
#   counts.txt     per-scale (and per-lang/per-placement) summary
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
  sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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
OUT=""
FISH_DIR=""
DRYRUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scale)     F_SCALE="${2:?--scale needs a value}"; shift 2 ;;
    --lang)      F_LANG="${2:?--lang needs a value}"; shift 2 ;;
    --placement) F_PLACEMENT="${2:?--placement needs a value}"; shift 2 ;;
    --limit)     LIMIT="${2:?--limit needs a value}"; shift 2 ;;
    --ref)       REF="${2:?--ref needs a value}"; shift 2 ;;
    --out)       OUT="${2:?--out needs a value}"; shift 2 ;;
    --fish-dir)  FISH_DIR="${2:?--fish-dir needs a value}"; shift 2 ;;
    --dry-run)   DRYRUN=1; shift ;;
    -h|--help)   usage 0 ;;
    *) die "unknown option '$1' (see --help)" ;;
  esac
done
FISH_DIR="${FISH_DIR:-$FISH_DIR_DEFAULT}"

[[ -f "$JSONL" ]] || die "corpus not found: $JSONL"
OUT="${OUT:-$REPO_ROOT/samples/out/$VARIANT}"
mkdir -p "$OUT"
MANIFEST="$OUT/manifest.tsv"
COUNTS="$OUT/counts.txt"

# ---------------------------------------------------------------------------
# Live vs dry run. `--fish` is wired into the CLI (synth --fish <variant>); a LIVE
# run loads the model and synthesizes, a --dry-run prints the commands only.
# ---------------------------------------------------------------------------
if [[ $DRYRUN -eq 1 ]]; then
  echo "synth-samples: --dry-run -> printing the commands that WOULD run; no audio is produced." >&2
fi
if [[ $DRYRUN -eq 0 && -z "$REF" ]]; then
  die "a reference voice is required to synthesize; pass --ref REF.wav (or set \$SYRINX_REF), or use --dry-run"
fi
REF_ARG="${REF:-<REF.wav>}"

# ---------------------------------------------------------------------------
# JSONL reader: jq when available, else a grep/sed fallback.
# Emits TAB-separated: model<TAB>scale<TAB>lang<TAB>placement<TAB>text, one per line.
# An entry matches the chosen variant if its model is the variant's model OR "both".
# ---------------------------------------------------------------------------
emit_rows() {
  if command -v jq >/dev/null 2>&1; then
    jq -r --arg m "$MODEL_FILTER" '
      select(.model == $m or .model == "both")
      | [.id, .scale, .lang, .placement, (.text|gsub("\t";" "))] | @tsv
    ' "$JSONL"
  else
    # Fallback parser (no jq). Pulls the six string fields per line with sed.
    # Assumes one compact JSON object per line (as produced by the generator).
    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      get() { printf '%s' "$line" | sed -n 's/.*"'"$1"'"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p'; }
      local id scale lang placement text model
      id=$(get id); scale=$(get scale); lang=$(get lang)
      placement=$(get placement); model=$(get model)
      text=$(printf '%s' "$line" | sed -n 's/.*"text"[[:space:]]*:[[:space:]]*"\(.*\)"[[:space:]]*,[[:space:]]*"desc".*/\1/p')
      [[ "$model" == "$MODEL_FILTER" || "$model" == "both" ]] || continue
      printf '%s\t%s\t%s\t%s\t%s\n' "$id" "$scale" "$lang" "$placement" "${text//	/ }"
    done < "$JSONL"
  fi
}

# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------
: > "$MANIFEST"
printf 'id\tscale\tlang\tplacement\ttext\twav\n' >> "$MANIFEST"

declare -A SC_COUNT LANG_COUNT PL_COUNT
total=0

while IFS=$'\t' read -r id scale lang placement text; do
  [[ -n "$F_SCALE"     && "$scale"     != "$F_SCALE"     ]] && continue
  [[ -n "$F_LANG"      && "$lang"      != "$F_LANG"      ]] && continue
  [[ -n "$F_PLACEMENT" && "$placement" != "$F_PLACEMENT" ]] && continue

  wav="$OUT/$id.wav"
  cmd=(cargo run -q -p syrinx-cli --features real -- \
       synth --fish "$VARIANT" --fish-dir "$FISH_DIR" --text "$text" --ref-wav "$REF_ARG" --out "$wav")

  if [[ $DRYRUN -eq 1 ]]; then
    printf '[DRY RUN] '
    printf '%q ' "${cmd[@]}"; printf '\n'
  else
    echo ">> $id"
    "${cmd[@]}"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$scale" "$lang" "$placement" "$text" "$wav" >> "$MANIFEST"
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
echo "manifest  -> $MANIFEST"
echo "counts    -> $COUNTS"
echo "run-meta  -> $META   (host + gpu + toolchain specs)"
