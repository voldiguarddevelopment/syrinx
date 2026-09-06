#!/usr/bin/env bash
# =============================================================================
# worktree.sh — make a git worktree that can actually build and test Syrinx.
#
#   scripts/worktree.sh new <name> [base-ref]   create + wire it up
#   scripts/worktree.sh list                    show worktrees and their disk use
#   scripts/worktree.sh rm <name>               remove one (refuses if dirty)
#
# Why a helper rather than plain `git worktree add`: two things in this repo do
# NOT come along with a checkout, and without them a fresh worktree looks broken
# in ways that are easy to misread.
#
#   1. `scripts/test-all.env` is gitignored (`.gitignore` has `*.env`) because it
#      holds this box's absolute weight and fixture paths. A worktree without it
#      does not fail — every weight-backed test SKIPs, the board prints green, and
#      you conclude the suite passed when it tested almost nothing. That is the
#      one failure mode this project cares about most, so the env is symlinked
#      back to the main tree: one source of truth for the box's paths.
#   2. `target/` is 31 GB here. Worktrees must NOT share one: cargo fingerprints
#      path dependencies by path, so two worktrees at different paths rebuild
#      rather than reuse, and a shared dir would just thrash. Each worktree gets
#      its own under $WT_ROOT/.targets/<name>, written into a per-worktree
#      `.cargo/config.toml`. Cold builds are the price of isolation; disk is
#      cheap here (742 GB free) and correctness is not.
#
# Both artifacts are added to `.git/info/exclude` (local, not the shared
# .gitignore) so they never show up as untracked noise in `git status`.
# =============================================================================
set -uo pipefail

MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WT_ROOT="${SYRINX_WT_ROOT:-$(dirname "$MAIN")/syrinx-wt}"

die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
ok()  { printf '\033[32m  ok\033[0m  %s\n' "$*"; }

cmd_new() {
  local name="${1:?usage: worktree.sh new <name> [base-ref]}"
  local base="${2:-HEAD}"
  local path="$WT_ROOT/$name"
  [ -e "$path" ] && die "$path already exists"

  mkdir -p "$WT_ROOT"
  # A branch per worktree: two worktrees cannot share one checked-out branch, and
  # naming it after the task makes `git worktree list` self-documenting.
  git -C "$MAIN" worktree add -b "wt/$name" "$path" "$base" || die "git worktree add failed"
  ok "worktree at $path on branch wt/$name (from $base)"

  # 1. the env file — symlink, so a path fixed here is fixed everywhere
  if [ -f "$MAIN/scripts/test-all.env" ]; then
    ln -s "$MAIN/scripts/test-all.env" "$path/scripts/test-all.env"
    ok "scripts/test-all.env -> main tree (weight-backed tests will RUN, not SKIP)"
  else
    printf '\033[33m  !!\033[0m  no scripts/test-all.env in the main tree — weight-backed tests will SKIP\n'
  fi

  # 2. its own target dir
  local tdir="$WT_ROOT/.targets/$name"
  mkdir -p "$tdir" "$path/.cargo"
  cat > "$path/.cargo/config.toml" <<EOF
# Written by scripts/worktree.sh. Worktrees must not share a target dir: cargo
# fingerprints path dependencies by path, so sharing rebuilds rather than reuses.
[build]
target-dir = "$tdir"
EOF
  ok "CARGO_TARGET_DIR -> $tdir (cold build on first use)"

  # 3. keep both out of git status, locally
  local ex="$MAIN/.git/info/exclude"
  grep -qxF '.cargo/config.toml' "$ex" 2>/dev/null || echo '.cargo/config.toml' >> "$ex"
  grep -qxF 'scripts/test-all.env' "$ex" 2>/dev/null || echo 'scripts/test-all.env' >> "$ex"
  ok "added to .git/info/exclude (local, not the shared .gitignore)"

  cat <<EOF

  cd $path
  source scripts/test-all.env && ./scripts/test-all.sh free

Merge back with:  git -C "$MAIN" merge wt/$name
Remove with:      scripts/worktree.sh rm $name
EOF
}

cmd_list() {
  git -C "$MAIN" worktree list | while read -r p rest; do
    local_name="$(basename "$p")"
    t="$WT_ROOT/.targets/$local_name"
    sz="$( [ -d "$t" ] && du -sh "$t" 2>/dev/null | cut -f1 || echo '-' )"
    printf '  %-52s %-28s target %s\n' "$p" "$rest" "$sz"
  done
}

cmd_rm() {
  local name="${1:?usage: worktree.sh rm <name>}"
  local path="$WT_ROOT/$name"
  [ -d "$path" ] || die "no worktree at $path"
  # Refuse on uncommitted work: a worktree is where in-progress changes live, and
  # this session already lost track of some by being casual with them.
  if [ -n "$(git -C "$path" status --porcelain)" ]; then
    git -C "$path" status --short | sed 's/^/    /'
    die "$name has uncommitted changes — commit, stash, or remove by hand"
  fi
  git -C "$MAIN" worktree remove "$path" || die "git worktree remove failed"
  rm -rf "${WT_ROOT:?}/.targets/$name"
  ok "removed $name and its target dir"
}

case "${1:-}" in
  new)  shift; cmd_new "$@" ;;
  list) cmd_list ;;
  rm)   shift; cmd_rm "$@" ;;
  *)    sed -n '3,8p' "$0"; exit 2 ;;
esac
