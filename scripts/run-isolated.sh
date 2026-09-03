#!/usr/bin/env bash
# =============================================================================
# run-isolated.sh — run a heavy build/test/render in its OWN systemd scope.
#
# Why: anything launched from the VSCodium integrated terminal inherits codium's
# scope (app-codium-*.scope). systemd's default OOMPolicy for a scope is `stop`,
# so when the kernel OOM-kills ONE child, systemd tears down the whole scope —
# the editor with it. That is exactly what happened on 2026-09-03: the kernel
# killed real_fish_s2_e2e (19.3 GB anon RSS) and took the session down.
#
# This wrapper puts the job in its own scope with OOMPolicy=continue, so an OOM
# kills only the job. It does NOT disable the OOM killer — with a 19 GB job on a
# 60 GB box that would trade a clean kill for a thrash/freeze.
#
#   scripts/run-isolated.sh cargo test --features real --release --test real_fish_s2_e2e
#   scripts/run-isolated.sh ./scripts/verify.sh
#
# Add MEMMAX=24G to also cap the job's memory, so it dies at a bound you chose
# instead of starving the machine first:
#   MEMMAX=24G scripts/run-isolated.sh cargo test ...
# =============================================================================
set -uo pipefail

if [ $# -eq 0 ]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

ARGS=(--user --scope --collect -p OOMPolicy=continue)
[ -n "${MEMMAX:-}" ] && ARGS+=(-p MemoryMax="$MEMMAX" -p MemorySwapMax=0)
# Keep the job's stdio attached so output still streams to the caller.
exec systemd-run "${ARGS[@]}" --quiet -- "$@"
