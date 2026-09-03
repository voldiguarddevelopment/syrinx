#!/usr/bin/env python3
"""
gen-qwen-ref.py — dump a Qwen3-TTS parity reference for `tests/real_qwen_*`.

The sibling of `gen-fish-ref.py`, and it exists for the same reason: without a reference
dump the Rust port can only be checked against itself. That gap let a real bug ship —
`text_projection` was implemented as `linear_fc2(linear_fc1(x))` with no activation,
which is a plain linear map rather than the reference's
`linear_fc2(act_fn(linear_fc1(x)))`. Plain synthesis still produced intelligible speech,
so nothing caught it; an `instruct` block pushed the talker into repeating its text.

Every number here comes from the REFERENCE's own modules — never a reimplementation:
the prompt is built by `Qwen3TTSModel.generate_custom_voice`, and the tensor captured is
exactly the `inputs_embeds` that `Qwen3TTSTalkerForConditionalGeneration.generate`
receives. If the reference cannot be imported, this script refuses rather than
approximating.

Environment (not vendored — the reference is a separate package):
    uv venv --python 3.12 ~/.venvs/qwen && ~/.venvs/qwen/bin/pip install qwen-tts
  or point at any interpreter that can `import qwen_tts`.

Usage:
    scripts/gen-qwen-ref.py --ckpt /data/models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
        --out ~/parity-qwen/1.7b-customvoice.safetensors [--device cuda:0] [--dtype bfloat16]

The anchor is deliberately the PROMPT, not the generated audio: generation samples, so it
cannot be compared bit-for-bit, while the prompt is a deterministic function of
(checkpoint, text, speaker, language, instruct) and is where this class of bug lives.
"""
import argparse
import os
import sys

# The fixed probe case. Pinned here and mirrored by tests/real_qwen_prompt_parity.rs —
# changing either side without the other makes the test compare two different prompts.
TEXT = "Come closer, I have something to tell you."
SPEAKER = "serena"
LANGUAGE = "English"
INSTRUCT = "Whisper"


def die(msg: str) -> "typing.NoReturn":  # noqa: F821
    raise SystemExit(f"[gen-qwen-ref] FATAL: {msg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="local CustomVoice checkpoint directory")
    ap.add_argument("--out", required=True, help="output .safetensors")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = ap.parse_args()

    if not os.path.isdir(args.ckpt):
        die(f"--ckpt is not a directory: {args.ckpt}")

    try:
        import torch
        from safetensors.torch import save_file
        from qwen_tts import Qwen3TTSModel
        from qwen_tts.core.models.modeling_qwen3_tts import (
            Qwen3TTSTalkerForConditionalGeneration as Talker,
        )
    except Exception as e:  # noqa: BLE001
        die(
            f"cannot import the reference ({type(e).__name__}: {e}). "
            "This script never approximates the reference — install qwen-tts and retry."
        )

    dtype = getattr(torch, args.dtype)
    model = Qwen3TTSModel.from_pretrained(args.ckpt, device_map=args.device, dtype=dtype)

    # Capture `inputs_embeds` on its way into the talker, then abort generation: the
    # prompt is the anchor, and sampling past it would only cost time.
    grabbed = {}
    original = Talker.generate

    def capture(self, *a, **kw):
        grabbed["inputs_embeds"] = kw["inputs_embeds"].detach().float().cpu()
        grabbed["trailing_text_hidden"] = kw["trailing_text_hidden"].detach().float().cpu()
        raise _Captured()

    class _Captured(Exception):
        pass

    Talker.generate = capture
    tensors = {}
    for tag, instruct in (("plain", None), ("instruct", INSTRUCT)):
        kwargs = {} if instruct is None else {"instruct": instruct}
        try:
            model.generate_custom_voice(
                text=TEXT, language=LANGUAGE, speaker=SPEAKER, **kwargs
            )
        except _Captured:
            pass
        else:
            die("the talker was never invoked — the reference API has changed")
        e = grabbed.pop("inputs_embeds")
        t = grabbed.pop("trailing_text_hidden")
        tensors[f"{tag}.inputs_embeds"] = e[0].contiguous()
        tensors[f"{tag}.trailing_text_hidden"] = t[0].contiguous()
        print(f"  {tag:9s} inputs_embeds {tuple(e.shape)}  trailing {tuple(t.shape)}")
    Talker.generate = original

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    save_file(tensors, args.out)
    print(f"[gen-qwen-ref] wrote {args.out}")
    print(f"[gen-qwen-ref] probe case: text={TEXT!r} speaker={SPEAKER!r} "
          f"language={LANGUAGE!r} instruct={INSTRUCT!r}")


if __name__ == "__main__":
    sys.exit(main())
