#!/usr/bin/env bash
set -u
cd /home/floofy/development/syrinx-build
source scripts/test-all.env >/dev/null 2>&1
GPU="$1"; RUN="$2"
for pair in "rest_en_de:/home/floofy/refs/voice_en_10s.wav" "rest_pl:/home/floofy/refs/voice_pl.wav"; do
  part="${pair%%:*}"; ref="${pair#*:}"
  shard="$RUN/${part}_${GPU}.jsonl"
  [ -s "$shard" ] || continue
  echo "=== gpu$GPU $part ($(wc -l < "$shard") entries) ==="
  SYRINX_FISH_DEVICE="$GPU" ./target/release/syrinx synth --fish s2-pro \
    --fish-dir /home/floofy/models/s2-pro --cuda --ref-wav "$ref" \
    --batch "$shard" --out-dir "$RUN" --batch-size 1
done
echo "=== gpu$GPU DONE ==="
