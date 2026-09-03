# Dev-box storage layout (NovaBox)

Written 2026-09-03, after the two-disk migration and the follow-up audit.
Companion to `scripts/setup-dev-drive.sh`, which is the script that did it and
carries the same facts in its trailing `NOTE` blocks. **If you change the layout,
change both.**

## The disks

Two identical 1 TB Kingston SNV3S1000G:

```
nvme0n1                                    nvme1n1
├─p1  1 G     vfat  /boot                  └─p1  931.5 G  ext4  LABEL=devdata  /data
├─p2   50 G   ext4  /                            UUID=1e0a8d94-030b-4f8a-881e-549ca5055023
└─p3  880.5 G ext4  /home
zram0 4 G swap (zstd, pri=100)
```

`/data` is mounted from `/etc/fstab` with
`defaults,noatime,nofail,x-gvfs-show,x-gvfs-name=devdata`:

- **noatime** — build trees are read constantly; atime writes are pure overhead.
- **nofail** — a missing data disk must never block boot.
- **x-gvfs-show / x-gvfs-name** — without them the mount is invisible in the
  desktop Places sidebar, which reads like "the disk did not mount". (A Flatpak
  file manager needs a separate grant: `flatpak override --user
  --filesystem=/data org.kde.dolphin`, then a full app restart. That is
  bubblewrap confinement, not a mount problem.)

`-m 0` at mkfs time: no root reserve. This is a data/build disk, not a system
disk, so the default 5% (= 50 GB here) would be pure waste.

## What lives where

`/data` contains **exactly four** things, and nothing else should be created there
speculatively:

| Path | Size | Why |
|---|---|---|
| `/data/development/` | 98 G | the whole source tree, incl. `syrinx-build/` (27 G, of which `target/` is 26 G and `renders/` 718 M) |
| `/data/models/` | 36 G | all model weights (s2-pro, s1-mini, the Qwen3-TTS set, whisper-base) |
| `/data/swap/swapfile` | 32 G | real swap — see below |
| `/data/lost+found/` | — | ext4 |

`/home` is back down to 78 G used (from 192 G). The remaining large items there
are **not** migration leftovers and were deliberately left alone: `~/.cache/uv`
(22 G), `~/Downloads` (16 G), `~/.venvs` (13 G), `~/.local` (9.3 G),
`~/.cache/huggingface` (5.5 G), `~/.cache/pip` (4.3 G), `~/cuda-12.8` (3.1 G),
`~/refs` (378 M audio references). The HF hub cache does **not** duplicate
`/data/models` — those models were fetched with `--local-dir`, so the hub entries
for `fishaudio/s2-pro` and `openai/whisper-base` are 12 K of refs each.

## The two symlinks — load-bearing, not cosmetic

```
~/development -> /data/development
~/models      -> /data/models
```

Do **not** remove them and do **not** "clean up" the paths that go through them:

- **15 `~/models/...` references across 10 tracked files** — `caps.toml` and
  `vocab.rs` provenance, the `syrinx-qwen` tokenizer/decoder doc comments,
  `docs/backends/CONTROL_SURVEY.md`, `scripts/convert-fish-tokenizer.py`,
  `scripts/render-qwen.py`, `scripts/gen-qwen-index.py` (which actually resolves
  it: `os.path.expanduser("~/models")`), and `tests/golden/qwen/README.md`.
  **The frozen `tests/control_survey_gate.rs` asserts on the literal
  `~/models/` spelling** (`row.contains("~/models/")` as a valid citation form),
  so rewriting those rows to `/data/models/` would break a frozen test.
- **14 absolute `/home/floofy/{development,models}/...` paths** in the archived
  `renders/2026-08-29-s2-pro-en-de-pl-300/run*.sh` reproduction scripts (12) and
  `.opt-reports/` (2), plus 19 more in that run's `gpu*.log` records. Historical
  provenance: they must keep resolving, and they must not be rewritten.
- **1 outside the repo:**
  `~/.config/pipewire/pipewire.conf.d/nova.conf -> ~/development/NovafoxV2/config/pipewire/nova.conf`.

`scripts/test-all.env`'s `/home/floofy/{refs,cuda-12.8,gcc14}` paths are *not* in
that list — those really do live on `/home` and were never migrated. Its model
paths point at `/data/models` directly.

## Swap

```
/dev/zram0            4 G   pri=100   (zstd, systemd zram-generator)
/data/swap/swapfile  32 G   pri=10    (fstab: defaults,pri=10)
```

Higher priority number wins, so the kernel fills the fast compressed zram first
and only spills to NVMe under real pressure — e.g. `real_fish_s2_e2e`, which needs
~19 GB because the s2-pro weights are BF16 on disk (9.1 G) and the CPU parity path
upcasts to F32 by design. zram alone cannot absorb that; it is compressed RAM.

The file is `root:root 0600`, fully allocated (not sparse), 34359738368 bytes.

## Deliberately absent

`/data/cargo-target` and `/data/renders` were created by an earlier revision of
`setup-dev-drive.sh`, sat empty, were referenced by nothing in the repo or in any
shell/editor/systemd config, and were **removed on 2026-09-03**. The script no
longer creates them.

- **`CARGO_TARGET_DIR` is deliberately unset.** The source tree already lives on
  `/data`, so every `target/` under it is already on this disk; a shared target
  dir across projects only adds cargo lock contention.
- **Renders are versioned in-tree** at `<repo>/renders/`, already on `/data` for
  the same reason. A second renders root would split the corpus across two places.

## Known rough edge (not fixed — needs a human call)

VSCodium's recent-workspace state still points at the pre-migration spelling
`/home/floofy/development/syrinx-build`
(`~/.config/VSCodium/User/globalStorage/storage.json` and three
`workspaceStorage/*/workspace.json`). It *resolves* through the symlink, so
nothing is broken — but opening the folder by that path makes the shell `cwd`
`/home/floofy/...`, and Claude Code keys its session history off `cwd`. History is
consequently split between `~/.claude/projects/-home-floofy-development-syrinx-build`
(16 M, pre-migration) and `~/.claude/projects/-data-development-syrinx-build`
(2.6 M, since). Open `/data/development/syrinx-build` directly to keep new
sessions in one place. Nothing here was edited — it is live editor state.

## Verification — re-runnable, no reboot needed

Every one of these was clean on 2026-09-03.

```bash
# 1. fstab parses and every entry is resolvable
findmnt --verify
#    -> "0 parse errors, 0 errors, 2 warnings"
#    Both warnings are expected and benign:
#      [W] non-bind mount source /data/swap/swapfile is a directory or regular file
#          (correct — it IS a swapfile, not a device)
#      [W] cannot detect on-disk filesystem type (Permission denied)
#          (only because it was run unprivileged)

# 2. the UUID in fstab is the disk that is actually there
lsblk -o NAME,SIZE,FSTYPE,LABEL,UUID,MOUNTPOINT /dev/nvme1n1
ls -l /dev/disk/by-uuid/1e0a8d94-030b-4f8a-881e-549ca5055023

# 3. boot-time behaviour, proven without rebooting: systemd-fstab-generator has
#    ALREADY parsed both entries into units, and both are active. The generated
#    units are what boot will use.
cat /run/systemd/generator/data.mount
cat /run/systemd/generator/data-swap-swapfile.swap
systemctl status data.mount data-swap-swapfile.swap
#    Ordering is correct — the swap unit carries
#      Requires=data.mount   After=data.mount   RequiresMountsFor=/data/swap/swapfile
#    so /data mounts before swapon on every boot:
systemctl show data-swap-swapfile.swap -p Requires -p After -p RequiresMountsFor

# 4. swap is live at the intended priorities
swapon --show

# 5. no unit or timer is failing on a path
systemctl --failed ; systemctl list-timers --all
#    (fstrim.timer must be enabled — TRIM is done by the timer, not by the
#     `discard` mount option, which costs in-line latency.)

# 6. no mount is session-only: everything under / comes from fstab, no bind
#    mounts, no manual mounts
findmnt -o TARGET,SOURCE,FSTYPE,OPTIONS

# 7. no symlink anywhere points at a deleted migration path
find /home/floofy /data -xtype l -printf '%p -> %l\n' \
  | grep -E 'development\.old|models\.old|/home/floofy/(development|models)'
#    -> no output. (There are ~1700 other dangling links under ~ — Steam runtimes,
#       icon themes, browser SingletonLock files. All pre-existing, none ours.)

# 8. the two symlinks resolve
readlink -f ~/development ~/models

# 9. repo-side path audit. NOTE: `grep` in a Claude Code shell is a gitignore-aware
#    ugrep wrapper and will silently skip ignored files such as scripts/test-all.env.
#    Use `command grep` for an audit.
cd /data/development/syrinx-build
command grep -rIo "/home/floofy" --exclude-dir=target --exclude-dir=.git --exclude-dir=.ratchet . \
  | sed 's/:.*//' | sort | uniq -c
command grep -rIo "~/models" --exclude-dir=target --exclude-dir=.git --exclude-dir=.ratchet . \
  | sed 's/:.*//' | sort | uniq -c
#    Compare against the two lists in "The two symlinks" above; anything new is
#    either a legitimate /home path or a regression.
```

`.ratchet/journal/*.jsonl` also contains pre-migration
`/home/floofy/development/...` paths. It is an append-only HMAC-chained record of
past runs — **never rewrite it.**
