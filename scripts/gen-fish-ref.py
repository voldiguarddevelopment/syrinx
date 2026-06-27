#!/usr/bin/env python3
# =============================================================================
# gen-fish-ref.py — produce the Fish Audio parity fixtures the Rust parity tests
# read (`tests/real_fish_s1_parity.rs` / `tests/real_fish_s2_parity.rs`).
#
# TEMPLATE — runs ON BOX only (the dev box is offline and the `real`-feature test
# binaries SIGILL there). It loads the *reference* fish-speech Python model for a
# variant, runs a FIXED prompt + seed, and dumps the anchors the Rust tests compare
# against into a `.safetensors`.
#
# The Rust parity tests are tolerant: each anchor is checked only if its keys are
# present, so you can dump just the codec anchor first (cheap, no LM) and add the
# slow-AR logits anchor once the LM forward is wired below.
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
# Optional / informational (not asserted by the current Rust tests, handy for debugging
# the fast-AR wave later):
#   fast_ar_codes      int64  [num_codebooks, T]  fast-AR expansion of a fixed frame
#
# -----------------------------------------------------------------------------
# Usage (ON BOX):
#   micromamba run -n fish python scripts/gen-fish-ref.py \
#       --variant s1-mini \
#       --ckpt   /root/models/openaudio-s1-mini \
#       --out    /root/parity-fish/s1/ref.safetensors
#
#   micromamba run -n fish python scripts/gen-fish-ref.py \
#       --variant s2-pro \
#       --ckpt   /root/models/s2-pro \
#       --out    /root/parity-fish/s2/ref.safetensors
#
# Then point the Rust tests at the dump:
#   SYRINX_FISH_S1_REF=/root/parity-fish/s1/ref.safetensors  (or SYRINX_FISH_S2_REF)
#   and run via scripts/run-fish.sh --parity <variant>  (or test-all.sh).
#
# These paths are the conventional box layout used in scripts/test-all.env.example.
# =============================================================================

import argparse
import sys

# The FIXED prompt + seed that pin a reproducible reference. The Rust tests do NOT
# re-tokenize this string — they read `prompt_ids` — so the exact text only has to be
# stable across the run that produced the dump.
FIXED_PROMPT_TEXT = "The quick brown fox jumps over the lazy dog."
FIXED_SEED = 0
# A small fixed frame count for the codec grid so the dump stays tiny + fast to diff.
FIXED_T_FRAMES = 16


def log(*a):
    print("[gen-fish-ref]", *a, file=sys.stderr)


def build_fixed_codes(num_codebooks, residual_size, semantic_size, t_frames):
    """A deterministic [num_codebooks, T] RVQ code grid (no model needed).

    Row 0 is the semantic-derived codebook (range < semantic_size); rows 1.. are the
    residual codebooks (range < residual_size). A simple modular pattern keeps it fixed
    and in-range for both variants.
    """
    import torch

    codes = torch.zeros((num_codebooks, t_frames), dtype=torch.int64)
    for c in range(num_codebooks):
        hi = semantic_size if c == 0 else residual_size
        for t in range(t_frames):
            codes[c, t] = (c * 7 + t * 13 + 1) % hi
    return codes


def main():
    ap = argparse.ArgumentParser(description="Dump Fish Audio parity fixtures (ON BOX).")
    ap.add_argument("--variant", required=True, choices=["s1-mini", "s2-pro"])
    ap.add_argument("--ckpt", required=True, help="checkpoint dir (model + config + codec + tokenizer)")
    ap.add_argument("--out", required=True, help="output .safetensors path")
    ap.add_argument("--t-frames", type=int, default=FIXED_T_FRAMES)
    ap.add_argument("--codec-only", action="store_true",
                    help="dump only the codec anchor (skip loading the LM)")
    args = ap.parse_args()

    import torch
    from safetensors.torch import save_file

    torch.manual_seed(FIXED_SEED)

    # The codec geometry — these are the Fish architecture constants the Rust
    # `CodecConfig` also uses (10 codebooks, 4096-way semantic). residual_size is read
    # from the model config below; default to 1024 for the codes-grid builder.
    num_codebooks = 10
    semantic_size = 4096
    residual_size = 1024

    out = {}

    # =====================================================================
    # TODO(on-box): load the reference fish-speech model for `args.variant`.
    #
    # s1-mini: the `fish_speech` repo (openaudio-s1-mini) — DualARTransformer + the
    #          modded-DAC codec (`firefly`/`DAC`). Typically:
    #     from fish_speech.models.text2semantic.llama import BaseTransformer
    #     from fish_speech.models.dac.modded_dac import DAC   # or the firefly codec
    #     lm   = BaseTransformer.from_pretrained(args.ckpt, ...)
    #     codec = DAC.load(args.ckpt + "/codec.pth")
    #
    # s2-pro:  the s2-pro stack — Qwen3-4B (`fish_qwen3_omni`) slow AR + the 4-layer
    #          audio decoder + the EVA-GAN/causal-DAC codec (`codec.pth`).
    #
    # Pull `residual_size`/`num_codebooks` from the loaded config so the codes grid is
    # in-range, e.g.:
    #     residual_size = codec.config.codebook_size
    #     num_codebooks = codec.config.n_codebooks
    # =====================================================================
    codec = None
    lm = None
    tokenizer = None
    log(f"variant={args.variant} ckpt={args.ckpt}")
    log("TODO(on-box): wire the fish-speech model load above; see the comment block.")

    # ---- (A) codec decode anchor -------------------------------------------------
    codes = build_fixed_codes(num_codebooks, residual_size, semantic_size, args.t_frames)
    out["codec_codes"] = codes.contiguous()
    # TODO(on-box): run the REAL codec decode and store the waveform.
    #     with torch.no_grad():
    #         wav = codec.decode(codes.unsqueeze(0))   # -> [1, 1, N] or [1, N]
    #     out["codec_wav"] = wav.reshape(-1).to(torch.float32).contiguous()
    if codec is not None:
        raise SystemExit("remove this guard once the codec decode above is wired")
    else:
        log("SKIP codec_wav: codec not loaded yet (dump carries codec_codes only).")

    # ---- (B) slow-AR step-0 logits anchor ---------------------------------------
    if not args.codec_only:
        # TODO(on-box): tokenize FIXED_PROMPT_TEXT with the variant's tokenizer to the
        # row-0 slow-vocab ids the Rust `build_prompt` produces (the same chat template:
        # <|im_start|>system…<|im_end|><|im_start|>user…<|im_end|><|im_start|>assistant…),
        # store them, then run the slow backbone's prefill/forward and store the logits at
        # the LAST position.
        #     ids = tokenizer.encode(build_prompt(FIXED_PROMPT_TEXT))   # list[int]
        #     out["prompt_ids"] = torch.tensor(ids, dtype=torch.int64)
        #     with torch.no_grad():
        #         logits = lm.forward_prefill(torch.tensor([ids]))      # -> [1, T, vocab]
        #     out["slow_logits_step0"] = logits[0, -1].to(torch.float32).contiguous()
        if lm is not None or tokenizer is not None:
            raise SystemExit("remove this guard once the slow-AR forward above is wired")
        else:
            log("SKIP prompt_ids/slow_logits_step0: LM/tokenizer not loaded yet.")

    if not out:
        raise SystemExit("nothing to dump — wire at least one anchor above (see TODOs).")

    save_file(out, args.out)
    log(f"wrote {args.out} with keys: {sorted(out.keys())}")
    log("note: a codec_codes-only dump makes the Rust parity test SKIP the codec anchor "
        "(it needs codec_wav too). Wire the TODO(on-box) blocks for a real parity check.")


if __name__ == "__main__":
    main()
