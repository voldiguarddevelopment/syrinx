#!/usr/bin/env python3
"""Regenerate the weight-free Qwen3-TTS golden fixtures under tests/golden/qwen/.

Two kinds of fixture, both derived from the published Apache-2.0 checkpoints and both
holding NO weight data:

  tests/golden/qwen/config/<ckpt>.json   the checkpoint's config.json, byte-for-byte
  tests/golden/qwen/index/<ckpt>.json    {tensor name: {shape, dtype}} read out of the
                                         safetensors HEADER only (the first 8-byte
                                         length prefix + that many bytes of JSON)

That index is what lets `tests/qwen_tensor_manifest.rs` check `syrinx_qwen::load::
expected_tensors` against the real published checkpoints on a box with no weights and
no GPU — the check that would have caught the sibling Fish s1 loader's wrong names.

    python3 scripts/gen-qwen-index.py [--models DIR] [--out DIR]

Defaults: --models ~/models (a symlink to /data/models on the model box), --out
tests/golden/qwen. Re-running must be a no-op unless the upstream checkpoints changed;
if it is not, the tests are meant to fail first and be re-baselined deliberately.
"""
import argparse
import json
import os
import struct
import sys

PREFIX = "Qwen3-TTS-12Hz-"


def header(path):
    """Read a safetensors file's header without touching the tensor payload."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.expanduser("~/models"))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden", "qwen"))
    a = ap.parse_args()

    names = sorted(d for d in os.listdir(a.models) if d.startswith(PREFIX))
    if not names:
        sys.exit(f"no {PREFIX}* checkpoints under {a.models}")

    cfg_dir = os.path.join(a.out, "config")
    idx_dir = os.path.join(a.out, "index")
    os.makedirs(cfg_dir, exist_ok=True)
    os.makedirs(idx_dir, exist_ok=True)

    for d in names:
        src = os.path.join(a.models, d)
        with open(os.path.join(src, "config.json"), "rb") as fh:
            raw = fh.read()
        with open(os.path.join(cfg_dir, d + ".json"), "wb") as fh:
            fh.write(raw)

        hdr = header(os.path.join(src, "model.safetensors"))
        idx = {k: {"shape": v["shape"], "dtype": v["dtype"]} for k, v in sorted(hdr.items())}
        with open(os.path.join(idx_dir, d + ".json"), "w") as fh:
            fh.write(json.dumps(idx, separators=(",", ":"), sort_keys=True) + "\n")
        print(f"  {d}: config {len(raw)} B, index {len(idx)} tensors")


if __name__ == "__main__":
    main()
