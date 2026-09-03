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

Four anchors, each chosen because it is DETERMINISTIC — generation samples, so generated
audio can never be compared bit-for-bit, but every stage that feeds it can:

  prompt.*     the talker prompt: a pure function of (ckpt, text, speaker, language,
               instruct). Gates the embedding tables, the projections and the assembly.
  talker.*     logits from the prefill forward. Gates attention, RoPE, norms and the head
               — everything the prompt anchor cannot see.
  predictor.*  the code predictor's input embeds and its first-step logits. Gates the
               talker->predictor handoff and the fast-AR stack.
  codec.*      a fixed code grid decoded to a waveform. Gates RVQ + the decoder, with no
               language model involved at all.

Together they cover the whole chain; a mismatch localizes itself to one stage.
"""
import argparse
import functools
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
    # CPU/float32 by default, and that default is load-bearing. The decoder is a deep
    # conv stack, so accumulation order matters: the SAME reference decoding the SAME
    # codes on CUDA vs CPU disagrees with itself by 0.031 max abs on a [-1,1] waveform.
    # Generating on CUDA and comparing against the CPU parity path would spend that
    # entire budget on a device difference and leave no room to detect a real one — the
    # port matches the CPU reference to 2.2e-5. Same rule as the Fish port: CPU stays
    # f32 for parity.
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["bfloat16", "float32"])
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

    # ---- talker + predictor: hook the real forwards during one real generate ---------
    inner = model.model
    talker = inner.talker
    grabbed_fwd = {}

    orig_fwd = type(talker).forward
    # `functools.wraps` is load-bearing, not cosmetic: HF's `generate` validates its
    # kwargs against `inspect.signature(self.forward)`, and a bare `*a, **kw` wrapper
    # makes every real kwarg look unsupported. `inspect.signature` follows `__wrapped__`.
    @functools.wraps(orig_fwd)
    def fwd_probe(self, *a, **kw):
        out = orig_fwd(self, *a, **kw)
        ie = kw.get("inputs_embeds")
        # The prefill is the only call with more than one position; later steps are
        # single-token. Record the first one and leave the rest alone.
        if "talker.prefill_logits" not in grabbed_fwd and ie is not None and ie.shape[1] > 1:
            grabbed_fwd["talker.prefill_logits"] = out.logits[0, -1].detach().float().cpu()
        return out
    type(talker).forward = fwd_probe

    predictor = talker.code_predictor
    orig_pgen = type(predictor).generate
    @functools.wraps(orig_pgen)
    def pgen_probe(self, *a, **kw):
        if "predictor.inputs_embeds" not in grabbed_fwd:
            ie = kw["inputs_embeds"]
            grabbed_fwd["predictor.inputs_embeds"] = ie[0].detach().float().cpu()
            # Its own first-step logits, computed deterministically. `generate` would
            # sample; the logits it samples FROM are the honest anchor.
            with torch.no_grad():
                grabbed_fwd["predictor.logits"] = self(inputs_embeds=ie).logits[0, -1].detach().float().cpu()
        return orig_pgen(self, *a, **kw)
    type(predictor).generate = pgen_probe

    try:
        model.generate_custom_voice(
            text=TEXT, language=LANGUAGE, speaker=SPEAKER, instruct=INSTRUCT, max_new_tokens=4
        )
    finally:
        type(talker).forward = orig_fwd
        type(predictor).generate = orig_pgen
    for k, v in grabbed_fwd.items():
        tensors[k] = v.contiguous()
        print(f"  {k:28s} {tuple(v.shape)}")

    # ---- codec: a fixed code grid decoded to a waveform ------------------------------
    # No language model in this path, so it is fully deterministic. The grid is SAVED
    # rather than described, so the Rust side decodes the identical input instead of
    # reimplementing a formula that could drift.
    ccfg = inner.speech_tokenizer.model.config
    dcfg = getattr(ccfg, "decoder_config", ccfg)
    if isinstance(dcfg, dict):
        n_groups = int(dcfg["num_quantizers"])
        residual_size = int(dcfg["codebook_size"])
        semantic_size = int(dcfg.get("semantic_codebook_size", residual_size))
    else:
        n_groups = int(dcfg.num_quantizers)
        residual_size = int(dcfg.codebook_size)
        semantic_size = int(getattr(dcfg, "semantic_codebook_size", residual_size))
    frames = 12

    def emit(name: str, grid: torch.Tensor) -> None:
        wavs, fs = inner.speech_tokenizer.decode([{"audio_codes": grid.to(inner.talker.device)}])
        w = torch.as_tensor(wavs[0]).detach().float().cpu().reshape(-1)
        tensors[f"{name}.codes"] = grid.to(torch.int64).contiguous()
        tensors[f"{name}.wav"] = w.contiguous()
        print(f"  {name + '.codes':28s} {tuple(grid.shape)} -> wav {tuple(w.shape)} @ {fs} Hz")

    # Ordinary interior values, safely inside every table.
    emit("codec", torch.tensor(
        [[(t * 7 + q * 13) % residual_size for q in range(n_groups)] for t in range(frames)],
        dtype=torch.long,
    ))

    # The boundary probe: every group at the last valid row of its table.
    #
    # `decoder_config` advertises `codebook_size: 2048` alongside
    # `semantic_codebook_size: 4096`, which invites the Fish-style bug of clamping one
    # stack with the other's width. It does not apply here, and that was checked against
    # the checkpoint rather than the config: EVERY decoder table — `rvq_first` (semantic)
    # and all 15 `rvq_rest` layers — is [2048, 256]. The 4096 belongs to the encoder path.
    # Feeding 4095 in group 0 makes the reference die with a CUDA device-side assert,
    # because `Qwen3TTSTokenizerV2Model.decode` clamps only `min=0` and never bounds the
    # top. So the real contract is "codes must already be in range", and 2047 is the edge.
    assert semantic_size >= residual_size  # keeps the two names honest if a config changes
    edge = torch.full((frames, n_groups), residual_size - 1, dtype=torch.long)
    emit("codec_edge", edge)
    print(f"  (decoder tables uniform at {residual_size}; config's semantic {semantic_size} "
          f"is the encoder's, {n_groups} groups)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    save_file(tensors, args.out)
    print(f"[gen-qwen-ref] wrote {args.out}")
    print(f"[gen-qwen-ref] probe case: text={TEXT!r} speaker={SPEAKER!r} "
          f"language={LANGUAGE!r} instruct={INSTRUCT!r}")


if __name__ == "__main__":
    sys.exit(main())
