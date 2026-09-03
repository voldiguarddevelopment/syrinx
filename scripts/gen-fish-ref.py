#!/usr/bin/env python3
# =============================================================================
# gen-fish-ref.py — produce the Fish Audio parity fixtures the Rust parity tests
# read (`tests/real_fish_s1_parity.rs` / `tests/real_fish_s2_parity.rs`).
#
# Runs ON BOX only: it loads the **reference** `fish-speech` Python model for a
# variant, runs a FIXED prompt + a FIXED code grid, and dumps the anchors the Rust
# tests compare against into a `.safetensors`.
#
# Every number in the dump comes out of the reference implementation. This script
# contains NO reimplementation of any Fish module: it calls
# `fish_speech.models.text2semantic.inference.load_codec_model`,
# `fish_speech.models.text2semantic.llama.DualARTransformer.from_pretrained`,
# `DAC.from_indices` and `DualARTransformer.forward_generate` and writes what they
# return. If the reference package cannot be imported, or a checkpoint/tokenizer
# cannot be loaded, the script FAILS — it never falls back to a synthesized value.
#
# The Rust parity tests are tolerant: each anchor is checked only if its keys are
# present, so you can dump just the codec anchor first (cheap, no LM: `--codec-only`)
# and add the slow-AR logits anchor after.
#
# -----------------------------------------------------------------------------
# Output keys (MUST match the Rust parity tests exactly):
#
#   prompt_ids         int64  [T_prompt]          row-0 slow-vocab token ids of the
#                                                 FIXED prompt (text-only; no audio rows)
#   slow_logits_step0  float32[vocab_size]        reference slow-AR logits at the LAST
#                                                 prompt position (pre-sampling)
#   codec_codes        int64  [num_codebooks, T]  a FIXED RVQ code grid (10 x T)
#   codec_wav          float32[N]                 Python codec decode of codec_codes,
#                                                 mono 44.1 kHz
#
# -----------------------------------------------------------------------------
# PREREQUISITE — the reference package (this is the part the box must provide)
#
# `fish-speech` is NOT a dependency of this repo and is not vendored. It is already
# cloned on this box at ~/refs/fish-speech, pinned/verified against commit
# befe4001745417f8c42131739d862b8a6fdbd15a (2026-08-22) — both `dual_ar` (s1-mini)
# and `fish_qwen3_omni` (s2-pro) load there. To re-fetch:
#
#   git clone https://github.com/fishaudio/fish-speech ~/refs/fish-speech
#
# Its `pyproject.toml` pins `torch==2.8.0` / `torchaudio==2.8.0` /
# `transformers<=4.57.3` and needs `descript-audio-codec` for the codec, which drags
# in `descript-audiotools` (pinned `protobuf<3.20`; fish-speech overrides that to
# `>=3.20,<6` in `[tool.uv]`, so a plain `pip install` will NOT resolve — use uv).
# torch 2.8.0 has no cp314 wheels and this box's only system interpreter is 3.14, so
# the environment must be built on a managed 3.12. `uv` IS installed:
#
#   cd ~/refs/fish-speech
#   uv venv --python 3.12 .venv       # uv downloads a managed CPython 3.12, no root
#   uv sync --extra cpu               # CPU torch — the parity path is CPU/f32 anyway
#
# That yields ~/refs/fish-speech/.venv/bin/python with `fish_speech` importable, so
# --fish-speech-root is then optional. Pass it (or $FISH_SPEECH_ROOT) whenever the
# interpreter you use does not have the package on its path.
#
# -----------------------------------------------------------------------------
# Usage (ON BOX):
#   ~/refs/fish-speech/.venv/bin/python scripts/gen-fish-ref.py \
#       --variant s1-mini \
#       --ckpt   ~/models/openaudio-s1-mini \
#       --out    ~/parity-fish/s1/ref.safetensors \
#       --fish-speech-root ~/refs/fish-speech
#
#   ~/refs/fish-speech/.venv/bin/python scripts/gen-fish-ref.py \
#       --variant s2-pro \
#       --ckpt   ~/models/s2-pro \
#       --out    ~/parity-fish/s2/ref.safetensors \
#       --fish-speech-root ~/refs/fish-speech
#
# Then point the Rust tests at the dump:
#   SYRINX_FISH_S1_REF=~/parity-fish/s1/ref.safetensors  (or SYRINX_FISH_S2_REF)
#   and run via scripts/run-fish.sh --parity <variant>  (or test-all.sh).
#
# These paths are the conventional box layout used in scripts/test-all.env.example.
#
# MEMORY: the defaults are CPU/float32 because the Rust parity tests run on
# `Device::Cpu` in f32 — comparing against a bf16 reference would measure the dtype,
# not the port. s2-pro is ~9.1 GB of bf16 weights on disk, so the f32 LM anchor needs
# ~19 GB: run it under `scripts/run-isolated.sh` (see CLAUDE.md) and never alongside
# another `real`-feature suite. `--codec-only` needs ~2 GB and is the cheap first step.
# =============================================================================

import argparse
import json
import os
import sys
from pathlib import Path

# The FIXED prompt + seed that pin a reproducible reference. The Rust tests do NOT
# re-tokenize this string — they read `prompt_ids` — so the exact text only has to be
# stable across the run that produced the dump.
FIXED_PROMPT_TEXT = "The quick brown fox jumps over the lazy dog."
FIXED_SEED = 0
# A small fixed frame count for the codec grid so the dump stays tiny + fast to diff.
FIXED_T_FRAMES = 16


def log(*a):
    print("[gen-fish-ref]", *a, file=sys.stderr)


def die(msg):
    raise SystemExit(f"[gen-fish-ref] FATAL: {msg}")


def import_fish_speech(root):
    """Import the reference package, or die with the exact remedy.

    Never returns a stand-in: a missing reference means no fixture, not a fake one.
    """
    if root:
        root = str(Path(os.path.expanduser(root)).resolve())
        if not Path(root, "fish_speech").is_dir():
            die(f"--fish-speech-root {root} has no `fish_speech/` package directory")
        sys.path.insert(0, root)
    # Import what we will actually call, not just the (empty) top-level package, so a
    # half-installed environment fails here with the remedy instead of mid-run with a
    # bare traceback.
    try:
        import torch  # noqa: F401
        import safetensors.torch  # noqa: F401
        import fish_speech  # noqa: F401
        import fish_speech.content_sequence  # noqa: F401
        import fish_speech.conversation  # noqa: F401
        import fish_speech.models.text2semantic.llama  # noqa: F401
        import fish_speech.models.text2semantic.inference  # noqa: F401
    except ImportError as e:
        die(
            f"cannot import the reference implementation ({e}).\n"
            "  This script dumps the REFERENCE model's own outputs; without the\n"
            "  reference there is nothing honest to dump. Build its environment:\n"
            "    git clone https://github.com/fishaudio/fish-speech ~/refs/fish-speech\n"
            "    cd ~/refs/fish-speech && uv venv --python 3.12 .venv && uv sync --extra cpu\n"
            "  then re-run with ~/refs/fish-speech/.venv/bin/python and\n"
            "  --fish-speech-root ~/refs/fish-speech"
        )


def build_fixed_codes(num_codebooks, residual_size, semantic_size, t_frames):
    """A deterministic `[num_codebooks, T]` RVQ code grid (no model needed).

    Row 0 is the semantic codebook (range < `semantic_size`); rows 1.. are the
    residual codebooks (range < `residual_size`). A simple modular pattern keeps it
    fixed and in-range for both variants — in-range matters because the reference
    `DownsampleResidualVectorQuantize.decode` CLAMPS out-of-range indices, so an
    out-of-range grid would silently compare two different inputs.
    """
    import torch

    codes = torch.zeros((num_codebooks, t_frames), dtype=torch.int64)
    for c in range(num_codebooks):
        hi = semantic_size if c == 0 else residual_size
        for t in range(t_frames):
            codes[c, t] = (c * 7 + t * 13 + 1) % hi
    return codes


def load_prompt_ids_arg(spec):
    """`--prompt-ids` is either an inline JSON array or `@path/to/file.json`."""
    if spec.startswith("@"):
        with open(os.path.expanduser(spec[1:]), encoding="utf-8") as f:
            ids = json.load(f)
    else:
        ids = json.loads(spec)
    if not isinstance(ids, list) or not ids or not all(isinstance(i, int) for i in ids):
        die("--prompt-ids must be a non-empty JSON array of integers")
    return ids


def reference_prompt_ids(variant, tokenizer, num_codebooks, text):
    """Row-0 ids for the fixed prompt, built with the REFERENCE prompt builders.

    Both branches use `fish_speech`'s own `ContentSequence` / `Conversation` classes
    and its own tokenizer — this script never hand-assembles a template.

      * s2-pro mirrors `generate_long(use_prompt=False)`: a system/user/assistant
        `Conversation`, which renders
        `<|im_start|>system\\nconvert the provided text to speech<|im_end|>\\n`
        `<|im_start|>user\\n{text}<|im_end|>\\n<|im_start|>assistant\\n<|voice|>`.
      * s1-mini uses the interleave `ContentSequence` form
        `<|interleave|><|speaker:0|>{text}` left OPEN (`add_end=False`), which is the
        template `S1Mini::encode_prompt_ids` reproduces.

    Returns `values` `[1 + num_codebooks, T]` (row 0 = ids, code rows 0 for a
    text-only prompt) — the same shape the Rust test reconstructs from `prompt_ids`.
    """
    from fish_speech.content_sequence import ContentSequence, TextPart

    if variant == "s2-pro":
        from fish_speech.conversation import Conversation, Message

        conv = Conversation()
        conv.append(
            Message(
                role="system",
                parts=[TextPart(text="convert the provided text to speech", cal_loss=False)],
                cal_loss=False,
                add_im_start=True,
                add_im_end=True,
            )
        )
        conv.append(
            Message(
                role="user",
                parts=[TextPart(text=text, cal_loss=False)],
                cal_loss=False,
                add_im_start=True,
                add_im_end=True,
            )
        )
        conv.append(
            Message(
                role="assistant",
                parts=[],
                cal_loss=False,
                modality="voice",
                add_im_start=True,
                add_im_end=False,
            )
        )
        values, _, _ = conv.encode_for_inference(tokenizer, num_codebooks=num_codebooks)
        return values

    seq = ContentSequence(modality="interleave")
    seq.append([TextPart(text=text, cal_loss=False)], add_end=False, speaker=0)
    values, _, _ = seq.encode_for_inference(tokenizer, num_codebooks=num_codebooks)
    return values


def dump_codec_anchor(args, torch, out):
    """(A) codec decode anchor — the reference DAC's own `from_indices`."""
    from fish_speech.models.text2semantic.inference import load_codec_model

    codec_ckpt = args.codec_ckpt or str(Path(args.ckpt) / "codec.pth")
    if not Path(codec_ckpt).is_file():
        die(f"codec checkpoint not found: {codec_ckpt}")

    dtype = getattr(torch, args.dtype)
    log(f"loading reference codec: {codec_ckpt} (device={args.device} dtype={args.dtype})")
    codec = load_codec_model(codec_ckpt, device=args.device, precision=dtype)

    # Geometry straight off the instantiated reference module — never guessed.
    semantic_size = int(codec.quantizer.semantic_quantizer.codebook_size)
    residual_size = int(codec.quantizer.quantizer.codebook_size)
    n_residual = int(codec.quantizer.quantizer.n_codebooks)
    num_codebooks = 1 + n_residual
    log(
        f"codec geometry: num_codebooks={num_codebooks} "
        f"semantic_size={semantic_size} residual_size={residual_size} "
        f"sample_rate={codec.sample_rate}"
    )

    codes = build_fixed_codes(num_codebooks, residual_size, semantic_size, args.t_frames)
    out["codec_codes"] = codes.contiguous()

    # `DownsampleResidualVectorQuantize.decode` clamps its argument IN PLACE, so hand
    # it a clone and keep the dumped grid pristine.
    with torch.inference_mode():
        wav = codec.from_indices(codes.clone().to(args.device)[None])
    out["codec_wav"] = wav.reshape(-1).to(torch.float32).cpu().contiguous()
    log(f"codec_wav: {tuple(out['codec_wav'].shape)} samples @ {codec.sample_rate} Hz")

    del codec


def dump_lm_anchor(args, torch, out):
    """(B) slow-AR step-0 logits anchor — the reference DualAR backbone's prefill."""
    from fish_speech.models.text2semantic.llama import DualARTransformer

    dtype = getattr(torch, args.dtype)
    log(f"loading reference LM: {args.ckpt} (device={args.device} dtype={args.dtype})")
    # `init_model()` in the reference is exactly this plus a torch.compile wrapper and
    # some sampling constants we do not use. `max_length` only shortens the RoPE table
    # and causal-mask precompute (their values at a given position are independent of
    # the precomputed length) and the KV-cache extent, so it is numerically inert for
    # prompts shorter than it — but it is the difference between a 256 KB mask and a
    # 1 GiB one on s2-pro's 32768-long default.
    model = DualARTransformer.from_pretrained(
        args.ckpt, load_weights=True, max_length=args.max_length
    )
    model = model.to(device=args.device, dtype=dtype).eval()

    n_cb = model.config.num_codebooks
    if model.config.semantic_begin_id == 0 and model.config.semantic_end_id == 0:
        # `BaseTransformer.from_pretrained` only WARNS when the tokenizer fails to
        # load, leaving the semantic range at 0..0 — which silently changes `embed`'s
        # codebook mask. Refuse rather than dump logits from a mis-masked model.
        if args.semantic_begin_id is None or args.semantic_end_id is None:
            die(
                "the reference left semantic_begin_id/semantic_end_id at 0 (its "
                "tokenizer load failed and config.json carries no "
                "semantic_start_token_id). Pass --semantic-begin-id/--semantic-end-id "
                "(read them off the checkpoint's special_tokens.json: the min/max id "
                "of the <|semantic:i|> tokens) — do NOT dump logits from a model whose "
                "codebook mask is wrong."
            )
        model.config.semantic_begin_id = args.semantic_begin_id
        model.config.semantic_end_id = args.semantic_end_id
        log(
            f"semantic range overridden: {args.semantic_begin_id}..{args.semantic_end_id}"
        )
    log(
        f"LM config: vocab_size={model.config.vocab_size} num_codebooks={n_cb} "
        f"semantic={model.config.semantic_begin_id}..{model.config.semantic_end_id}"
    )

    if args.prompt_ids:
        ids = load_prompt_ids_arg(args.prompt_ids)
        values = torch.zeros((1 + n_cb, len(ids)), dtype=torch.long)
        values[0] = torch.tensor(ids, dtype=torch.long)
        log(f"prompt ids supplied verbatim ({len(ids)} tokens)")
    else:
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            die(
                "the reference tokenizer did not load for this checkpoint, so the "
                "prompt cannot be built. Either make it loadable (the reference "
                "`FishTokenizer` wraps `AutoTokenizer`, so the checkpoint dir needs an "
                "HF `tokenizer.json` + `tokenizer_config.json`) or pass the ids "
                "explicitly with --prompt-ids '[...]' / --prompt-ids @ids.json "
                "(e.g. the row-0 ids `SYRINX_FISH_DUMP=1` prints from the Rust side — "
                "the parity check is logits-given-ids, so identical ids on both sides "
                "is exactly what it needs)."
            )
        values = reference_prompt_ids(args.variant, tokenizer, n_cb, args.prompt_text)
        log(f"prompt built by the reference ({values.shape[1]} tokens)")
        try:
            log("prompt decodes to: " + repr(tokenizer.decode(values[0].tolist())))
        except Exception as e:  # decoding is diagnostics only
            log(f"(could not decode prompt for logging: {e})")

    t = int(values.shape[1])
    if t == 0:
        die("the prompt encoded to zero tokens")
    if t > model.config.max_seq_len:
        die(f"prompt length {t} exceeds max_seq_len {model.config.max_seq_len}")

    values = values.to(args.device)
    with torch.device(args.device):
        model.setup_caches(max_batch_size=1, max_seq_len=model.config.max_seq_len, dtype=dtype)

    input_pos = torch.arange(0, t, device=args.device, dtype=torch.long)
    with torch.inference_mode():
        result = model.forward_generate(values[None], input_pos)

    # `forward_generate` already narrows to the last position: logits is [1, 1, vocab].
    logits = result.logits[0, -1]
    if int(logits.shape[-1]) != int(model.config.vocab_size):
        die(
            f"logits width {int(logits.shape[-1])} != vocab_size "
            f"{int(model.config.vocab_size)} — refusing to dump"
        )

    out["prompt_ids"] = values[0].to(torch.int64).cpu().contiguous()
    out["slow_logits_step0"] = logits.to(torch.float32).cpu().contiguous()
    log(f"slow_logits_step0: {tuple(out['slow_logits_step0'].shape)}")

    del model


def main():
    ap = argparse.ArgumentParser(description="Dump Fish Audio parity fixtures (ON BOX).")
    ap.add_argument("--variant", required=True, choices=["s1-mini", "s2-pro"])
    ap.add_argument("--ckpt", required=True, help="checkpoint dir (model + config + codec + tokenizer)")
    ap.add_argument("--out", required=True, help="output .safetensors path")
    ap.add_argument("--codec-ckpt", default=None, help="default: <ckpt>/codec.pth")
    ap.add_argument(
        "--fish-speech-root",
        default=os.environ.get("FISH_SPEECH_ROOT"),
        help="path to a fish-speech checkout (default: $FISH_SPEECH_ROOT, else rely on "
        "the interpreter's installed `fish_speech`)",
    )
    ap.add_argument("--device", default="cpu", help="cpu (default, matches the Rust parity path) or cuda")
    ap.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="float32 (default) — the Rust parity tests run f32 on CPU",
    )
    ap.add_argument("--t-frames", type=int, default=FIXED_T_FRAMES)
    ap.add_argument("--prompt-text", default=FIXED_PROMPT_TEXT)
    ap.add_argument(
        "--prompt-ids",
        default=None,
        help="inline JSON array or @file of row-0 token ids, bypassing the reference "
        "tokenizer (use when the checkpoint has no HF tokenizer the reference can load)",
    )
    ap.add_argument("--semantic-begin-id", type=int, default=None)
    ap.add_argument("--semantic-end-id", type=int, default=None)
    ap.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="override the LM's max_seq_len (RoPE/mask/KV-cache extent only — "
        "numerically inert for shorter prompts, but s2-pro's 32768 default costs a "
        "1 GiB causal mask). 0 = keep the checkpoint's value.",
    )
    ap.add_argument("--codec-only", action="store_true", help="dump only the codec anchor (skip the LM)")
    ap.add_argument("--lm-only", action="store_true", help="dump only the slow-AR anchor (skip the codec)")
    args = ap.parse_args()

    if args.codec_only and args.lm_only:
        die("--codec-only and --lm-only are mutually exclusive")

    args.ckpt = os.path.expanduser(args.ckpt)
    args.out = os.path.expanduser(args.out)
    if args.codec_ckpt:
        args.codec_ckpt = os.path.expanduser(args.codec_ckpt)
    if not Path(args.ckpt).is_dir():
        die(f"checkpoint dir not found: {args.ckpt}")
    if args.max_length == 0:
        args.max_length = None

    import_fish_speech(args.fish_speech_root)

    import torch
    from safetensors.torch import save_file

    torch.manual_seed(FIXED_SEED)

    log(f"variant={args.variant} ckpt={args.ckpt}")
    out = {}

    if not args.lm_only:
        dump_codec_anchor(args, torch, out)
    if not args.codec_only:
        dump_lm_anchor(args, torch, out)

    if not out:
        die("nothing to dump")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(out, args.out)
    log(f"wrote {args.out} with keys: {sorted(out.keys())}")
    if "codec_wav" not in out:
        log("note: no codec_wav — the Rust parity test will SKIP the codec anchor.")
    if "slow_logits_step0" not in out:
        log("note: no slow_logits_step0 — the Rust parity test will SKIP the LM anchor.")


if __name__ == "__main__":
    main()
