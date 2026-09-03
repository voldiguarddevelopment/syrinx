#!/usr/bin/env bash
set -u
cd /home/floofy/development/syrinx-build
source scripts/test-all.env >/dev/null 2>&1
RUN="$1"
echo $$ > "$RUN/render.pid"
for pair in "short_en_de:/home/floofy/refs/voice_en_10s.wav" "short_pl:/home/floofy/refs/voice_pl.wav"; do
  part="${pair%%:*}"; ref="${pair#*:}"
  echo "=== $part ($(wc -l < "$RUN/$part.jsonl") entries) ref=$(basename "$ref") ==="
  SYRINX_FISH_DEVICE=1 ./target/release/syrinx synth --fish s2-pro \
    --fish-dir /home/floofy/models/s2-pro --cuda --ref-wav "$ref" \
    --batch "$RUN/$part.jsonl" --out-dir "$RUN" --batch-size 1
done
echo "=== gpu1 DONE ==="
