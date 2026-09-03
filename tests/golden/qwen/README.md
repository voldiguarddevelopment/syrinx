# Qwen3-TTS golden fixtures — weight-free, board-runnable

These fixtures exist so `GROUP_qwen` (see `scripts/test-all.sh`) can check the
`syrinx-qwen` port against the **real published checkpoints** on a box with no weights,
no CUDA and no Python — the part of the port that is genuinely deterministic.

| directory | what it holds | size |
|-----------|---------------|------|
| `config/` | each checkpoint's `config.json`, byte-for-byte | ~4.5 KB each |
| `index/`  | `{tensor name: {shape, dtype}}` read from the safetensors **header** only | ~35–41 KB each |

No tensor data is stored here. The index is the first 8-byte length prefix plus that
many bytes of JSON at the head of `model.safetensors`; the payload is never read.

Provenance: the five Apache-2.0 checkpoints `Qwen3-TTS-12Hz-{0.6B,1.7B}-Base`,
`-{0.6B,1.7B}-CustomVoice` and `-1.7B-VoiceDesign`, as downloaded to `~/models`
(`/data/models`) on 2026-09-01. Regenerate with:

    python3 scripts/gen-qwen-index.py            # --models ~/models --out tests/golden/qwen

Re-running must be a no-op. If upstream republishes a checkpoint the tests are supposed
to fail first (`tests/qwen_config_contract.rs`, `tests/qwen_tensor_manifest.rs`), so the
change is re-baselined deliberately rather than absorbed silently.

What these fixtures deliberately do **not** cover, because it needs the real weights:
the numeric content of any tensor, the tokenizer (`vocab.json` / `merges.txt` are part
of the checkpoint, not of this fixture set), the 682 MB `Qwen3-TTS-Tokenizer-12Hz` codec,
and everything downstream of a forward pass. See `docs/backends/QWEN_PORT_STATUS.md` for
the full boundary.
