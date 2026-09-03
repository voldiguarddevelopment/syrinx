#!/usr/bin/env bash
# =============================================================================
# setup-dev-drive.sh — bring an UNUSED NVMe up as the development drive.
#
# This box has two identical 1 TB Kingston SNV3S1000G. nvme0n1 holds /boot, /
# and /home; nvme1n1 has no partition table, no filesystem and no mount. This
# script turns the second one into /data and moves the heavy, regenerable things
# there: cargo target dirs, model weights, renders, and (optionally) swap.
#
#   scripts/setup-dev-drive.sh --check              inspect only (default, SAFE)
#   sudo scripts/setup-dev-drive.sh --format /dev/nvme1n1
#
# --check makes NO changes. --format is DESTRUCTIVE and refuses to run if the
# target has any partition table, filesystem signature, mount, or holder — pass
# --force only if you have read the --check output and accept the erase.
#
# Why ext4: matches / and /home, no surprises, and the workload (cargo target
# dirs = many small files, rewritten constantly) is exactly what it is good at.
# Model weights are already-compressed tensors, so btrfs compression would buy
# nothing for the extra moving parts.
# =============================================================================
set -euo pipefail

DEV="${2:-/dev/nvme1n1}"
MODE="${1:---check}"
MNT="/data"
FORCE=0
for a in "$@"; do [ "$a" = "--force" ] && FORCE=1; done

die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
ok()  { printf '\033[32m  ok\033[0m  %s\n' "$*"; }
warn(){ printf '\033[33m  !!\033[0m  %s\n' "$*"; }

[ -b "$DEV" ] || die "$DEV is not a block device"

echo "== target: $DEV =="
lsblk -o NAME,SIZE,TYPE,FSTYPE,LABEL,MOUNTPOINT,MODEL "$DEV"
echo

# ---- safety audit (runs in BOTH modes) --------------------------------------
PROBLEMS=0
BASE="$(basename "$DEV")"

if findmnt -S "$DEV" >/dev/null 2>&1; then warn "$DEV is MOUNTED"; PROBLEMS=1; else ok "not mounted"; fi

HOLDERS="$(ls /sys/block/$BASE/holders 2>/dev/null | wc -l)"
if [ "$HOLDERS" -gt 0 ]; then warn "$DEV has device-mapper/RAID holders"; PROBLEMS=1; else ok "no holders (not in RAID/LVM/LUKS)"; fi

PARTS="$(lsblk -no NAME "$DEV" | tail -n +2 | wc -l)"
if [ "$PARTS" -gt 0 ]; then warn "$DEV has $PARTS existing partition(s)"; PROBLEMS=1; else ok "no partitions"; fi

if command -v wipefs >/dev/null && [ "$(id -u)" = 0 ]; then
  SIGS="$(wipefs -n "$DEV" 2>/dev/null | tail -n +2 | wc -l)"
  if [ "$SIGS" -gt 0 ]; then
    warn "$DEV carries $SIGS filesystem/partition signature(s):"
    wipefs -n "$DEV" | sed 's/^/      /'
    PROBLEMS=1
  else ok "no filesystem signatures"; fi
else
  warn "run as root to check filesystem signatures (wipefs -n)"
fi

if command -v nvme >/dev/null && [ "$(id -u)" = 0 ]; then
  echo "  -- SMART --"
  nvme smart-log "$DEV" 2>/dev/null | grep -iE "critical_warning|percentage_used|data_units_written|power_on_hours|media_errors" | sed 's/^/      /'
fi

echo
if [ "$MODE" = "--check" ]; then
  [ "$PROBLEMS" -eq 0 ] && echo "VERDICT: $DEV looks unused. Re-run with: sudo $0 --format $DEV" \
                        || echo "VERDICT: $DEV is NOT clearly unused — do not format without checking the warnings above."
  exit 0
fi

[ "$MODE" = "--format" ] || die "unknown mode '$MODE' (use --check or --format)"
[ "$(id -u)" = 0 ] || die "--format must run as root"
if [ "$PROBLEMS" -ne 0 ] && [ "$FORCE" -ne 1 ]; then
  die "refusing to format: the audit found existing data. Re-read the warnings; pass --force only if you accept erasing $DEV."
fi

echo "About to ERASE $DEV ($(lsblk -dno SIZE "$DEV")) and mount it at $MNT."
read -rp "Type the device name to confirm [$DEV]: " CONFIRM
[ "$CONFIRM" = "$DEV" ] || die "confirmation did not match; nothing changed"

# ---- do it ------------------------------------------------------------------
wipefs -a "$DEV"
# sfdisk, not sgdisk: gptfdisk is NOT installed on this box and the original
# sgdisk call died mid-run (after wipefs) with "command not found". sfdisk ships
# with util-linux, so it is always present.
printf 'label: gpt\nname=devdata, type=linux\n' | sfdisk "$DEV"
partprobe "$DEV"; sleep 2
PART="${DEV}p1"; [ -b "$PART" ] || PART="${DEV}1"
# -m 0: no root reserve. This is a data/build disk, not a system disk, so the
# default 5% (=50 GB here) would be pure waste.
mkfs.ext4 -m 0 -L devdata "$PART"

mkdir -p "$MNT"
UUID="$(blkid -s UUID -o value "$PART")"
if ! grep -q "$UUID" /etc/fstab; then
  # noatime: build trees are read constantly; atime writes are pure overhead.
  # nofail: a missing data disk must never block boot.
  echo "UUID=$UUID  $MNT  ext4  defaults,noatime,nofail  0 2" >> /etc/fstab
fi
systemctl daemon-reload
mount "$MNT"
OWNER="${SUDO_USER:-floofy}"
chown "$OWNER:$OWNER" "$MNT"
mkdir -p "$MNT"/{cargo-target,models,renders,swap}
chown -R "$OWNER:$OWNER" "$MNT"

# TRIM via the timer, not the `discard` mount option (lower latency in-line).
systemctl enable --now fstrim.timer 2>/dev/null || true

echo
df -h "$MNT"
cat <<EOF

Mounted $PART at $MNT (ext4, noatime, nofail), UUID=$UUID

Next, point the heavy things at it:

  # source trees + model weights: COPY, verify, then replace the original with a
  # symlink. Do NOT use --remove-source-files: it deletes as it goes, so a failed
  # run leaves the tree split across two disks with no intact copy to fall back on.
  rsync -a ~/development/ $MNT/development/ && mv ~/development ~/development.old
  ln -s $MNT/development ~/development
  rsync -a ~/models/ $MNT/models/

  # verify BEFORE deleting the original — file list, sizes, then content:
  #   diff <(cd ~/models.old && find . -type f -printf '%s %p\\n' | sort) \\
  #        <(cd $MNT/models && find . -type f -printf '%s %p\\n' | sort)
  #   cd DIR && find . -type f -print0 | sort -z | xargs -0 xxh128sum   # both sides
  # For a git tree also compare HEAD + `status --porcelain | wc -l` and run
  # `git fsck` on the copy. Expect .git/index to differ on both sides — it is a
  # stat cache rewritten by any `git status`, not repository data.
  rm -rf ~/models.old && ln -s $MNT/models ~/models

  # The symlink matters: ~50 references to ~/models/... survive in scripts, run
  # scripts under renders/, caps.toml provenance and CONTROL_SURVEY.md, and the
  # frozen tests/control_survey_gate.rs asserts on the '~/models/' spelling.

  # a real swapfile here (your current 4 GB is zram = compressed RAM, which
  # cannot absorb the 19 GB real_fish_s2_e2e run)
  fallocate -l 32G $MNT/swap/swapfile && chmod 600 $MNT/swap/swapfile
  mkswap $MNT/swap/swapfile && swapon $MNT/swap/swapfile
  echo '$MNT/swap/swapfile none swap defaults,pri=10 0 0' >> /etc/fstab

  # CARGO_TARGET_DIR is deliberately NOT set: once the source tree lives on $MNT
  # its target/ is already there, and a shared target dir across projects just
  # adds cargo lock contention. $MNT/cargo-target stays empty unless a tree that
  # is NOT on $MNT needs it.
EOF

# NOTE (2026-09-03): a Flatpak file manager cannot see a new top-level directory
# at all — Dolphin reported "/data does not exist" while the host had it mounted
# with shared propagation. That is bubblewrap confinement, not a mount problem:
#   flatpak override --user --filesystem=/data org.kde.dolphin
# then fully restart the app. Check `pgrep -af bwrap` before blaming fstab.

# NOTE (2026-09-03): migration executed on this box. /data = nvme1n1p1 (ext4,
# noatime,nofail), ~/development -> /data/development, ~/models -> /data/models,
# 32 GB swapfile at /data/swap/swapfile (pri=10, above zram's pri=100 so zram is
# still preferred for small pressure). Both originals were verified byte-identical
# with xxh128sum (173 model files; 162,853 source files excluding target/) before
# deletion — 114 GB reclaimed on /home. scripts/test-all.env now points at
# /data/models; the ~/models symlink keeps every other ~/models/... reference valid.
