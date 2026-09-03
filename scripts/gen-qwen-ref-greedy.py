#!/usr/bin/env python3
"""
gen-qwen-ref-greedy.py — dump the reference's GREEDY code matrix for `tests/real_qwen_greedy_parity.rs`.

`gen-qwen-ref.py` anchors four points of the Qwen3-TTS port — the prompt, the talker's
prefill logits, the code predictor's first-step logits and the codec decode — and every
one of them is a SINGLE step. Nothing there checks that the autoregressive loop stays in
agreement over many frames: the KV cache, position advancement, the trailing-text
schedule, the talker->predictor handoff per frame, the EOS / min-new-tokens guard and
`suppress_tokens` are exercised once or not at all. A drift that first appears at frame 5
is invisible to all four.

The reason that gap was left open is that generation SAMPLES, and two PRNGs are never
bit-comparable. **Greedy decoding closes it**: `do_sample=False` on both heads makes the
whole loop a deterministic function of (weights, prompt), so the generated integer code
matrix can be compared exactly, frame by frame.

## What "greedy" means on the reference side, verified rather than assumed

`Qwen3TTSModel.generate_custom_voice(**kwargs)` funnels through `_merge_generate_kwargs`,
which fills anything the caller leaves `None` from the checkpoint's `generation_config.json`.
Passing `do_sample=False, subtalker_dosample=False` therefore reaches BOTH heads:

* the talker: `Qwen3TTSForConditionalGeneration.generate` forwards `do_sample` into
  `talker_kwargs`, and `transformers.generation.utils._get_logits_processor` installs the
  temperature / top-k / top-p warpers **only** under `if generation_config.do_sample:`
  (utils.py, "Processors previously known as `LogitsWarpers`, only applied with sampling
  strategies"). `_sample` then takes the `else` branch, `torch.argmax(next_token_scores)`.
* the sub-talker: `Qwen3TTSTalkerForConditionalGeneration.forward` passes
  `do_sample=subtalker_dosample` straight into `code_predictor.generate`.

What does NOT switch off is the rest of the processor list, and this is the trap worth
naming: `RepetitionPenaltyLogitsProcessor`, `MinNewTokensLengthLogitsProcessor` and
`SuppressTokensLogitsProcessor` are installed above that `if`, so under greedy the talker
still runs **repetition_penalty = 1.05**, still masks the codec EOS until
`min_new_tokens = 2`, and still suppresses the top-1024 control block. The sub-talker gets
none of those: `forward` forwards only the four `subtalker_*` knobs, so the penalty falls
back to `code_predictor_config.repetition_penalty`, which the checkpoint ships as `1.0`,
and its `suppress_tokens` / `min_length` are `None` / `0`. The Rust side must match that
asymmetry exactly, which is what `real_qwen_greedy_parity.rs` asserts.

## CPU / float32, and why that is not negotiable

Same rule as `gen-qwen-ref.py`, only sharper here. Greedy decoding is an argmax chain: a
numeric difference far below any sane tolerance can flip one code, and from that frame on
the two runs are generating different audio. The reference computing the same conv-heavy
path on CUDA vs CPU already disagrees with ITSELF by ~0.03, which is orders of magnitude
more than enough. So the fixture is generated on CPU/f32 — the port's designated parity
path — and the test refuses to run anywhere else.

Environment (the reference is NOT vendored):
    ~/refs/qwen3-tts  (github.com/QwenLM/Qwen3-TTS, verified at 022e286)
    an interpreter that can `import qwen_tts` (here: ~/.venvs/qwen/bin/python)

Usage:
    scripts/gen-qwen-ref-greedy.py --ckpt "$SYRINX_QWEN_CV_DIR" --out "$SYRINX_QWEN_REF_GREEDY"

Three cases, chosen so that between them they walk every branch of the loop:

  plain      non-streaming, no instruct — the ordinary CustomVoice path.
  instruct   non-streaming, with an instruct block — the path that hid the `silu` bug.
  streaming  `non_streaming_mode=False` — the ONLY configuration in which
             `trailing_text_hidden` is longer than one row, so it is the only case that
             exercises the per-frame trailing-text schedule and its exhaustion into
             `tts_pad_embed`. In non-streaming mode the reference sets
             `trailing_text_hidden = tts_pad_embed`, a single row, and every frame past
             the first takes the pad anyway.
"""
import argparse
import functools
import os
import sys

# The probe case. Mirrors scripts/gen-qwen-ref.py so the two fixtures describe the same
# utterance; tests/real_qwen_greedy_parity.rs pins the identical strings.
TEXT = "Come closer, I have something to tell you."
SPEAKER = "serena"
LANGUAGE = "English"
INSTRUCT = "Whisper"

# Cases: (tag, instruct, non_streaming_mode).
CASES = (
    ("plain", None, True),
    ("instruct", INSTRUCT, True),
    ("streaming", None, False),
)


def die(msg: str):
    raise SystemExit(f"[gen-qwen-ref-greedy] FATAL: {msg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="local CustomVoice checkpoint directory")
    ap.add_argument("--out", required=True, help="output .safetensors")
    # Not a default to be overridden lightly — see the module docstring. The flags exist so
    # a device difference can be MEASURED, not so a fixture can be generated elsewhere.
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["bfloat16", "float32"])
    # A frame is 1/12.5 s, so 200 frames is 16 s of audio — comfortably more than the probe
    # utterance needs (~35 frames sampled) while bounding a slow CPU/f32 run. Recorded in
    # the fixture so the test can tell "ended on EOS" from "ran into the cap".
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--cases", default=",".join(t for t, _, _ in CASES))
    args = ap.parse_args()

    if not os.path.isdir(args.ckpt):
        die(f"--ckpt is not a directory: {args.ckpt}")
    wanted = [c.strip() for c in args.cases.split(",") if c.strip()]
    unknown = [c for c in wanted if c not in {t for t, _, _ in CASES}]
    if unknown:
        die(f"unknown case(s) {unknown}; known: {[t for t, _, _ in CASES]}")

    try:
        import torch
        from safetensors.torch import save_file
        from qwen_tts import Qwen3TTSModel
        from qwen_tts.core.models.modeling_qwen3_tts import (
            Qwen3TTSForConditionalGeneration as Outer,
            Qwen3TTSTalkerForConditionalGeneration as Talker,
        )
    except Exception as e:  # noqa: BLE001
        die(
            f"cannot import the reference ({type(e).__name__}: {e}). "
            "This script never approximates the reference — install qwen-tts and retry."
        )

    dtype = getattr(torch, args.dtype)
    model = Qwen3TTSModel.from_pretrained(args.ckpt, device_map=args.device, dtype=dtype)
    eos = int(model.model.config.talker_config.codec_eos_token_id)
    print(f"[gen-qwen-ref-greedy] device={args.device} dtype={args.dtype} codec_eos={eos}")

    grabbed = {}

    # --- the talker's own generate: the prompt it was handed, and every id it sampled ----
    # `functools.wraps` is load-bearing here for the same reason gen-qwen-ref.py flags it:
    # HF introspects the callables it drives. Wrapping keeps the signature intact.
    orig_talker_generate = Talker.generate

    @functools.wraps(orig_talker_generate)
    def talker_probe(self, *a, **kw):
        grabbed["inputs_embeds"] = kw["inputs_embeds"].detach().float().cpu()
        grabbed["trailing_text_hidden"] = kw["trailing_text_hidden"].detach().float().cpu()
        # Pinned so a fixture generated with different knobs cannot masquerade as greedy.
        for k, want in (("do_sample", False), ("subtalker_dosample", False)):
            if kw.get(k) is not False:
                die(f"{k} reached the talker as {kw.get(k)!r}, not False — this is not a greedy run")
        grabbed["talker_kwargs"] = {
            k: kw.get(k)
            for k in ("min_new_tokens", "max_new_tokens", "repetition_penalty", "eos_token_id")
        }
        grabbed["n_suppress"] = len(kw.get("suppress_tokens") or ())
        out = orig_talker_generate(self, *a, **kw)
        # `sequences` is every group-0 id the talker drew, INCLUDING a terminal EOS. The
        # trimmed code matrix below never contains it, so this is how the fixture records
        # whether the run ended on the EOS or ran into the frame cap.
        grabbed["sequences"] = out.sequences[0].detach().to(torch.int64).cpu()
        return out

    # --- the outer generate: the reference's OWN trimmed [T, num_code_groups] matrix -----
    orig_outer_generate = Outer.generate

    class _Captured(Exception):
        pass

    @functools.wraps(orig_outer_generate)
    def outer_probe(self, *a, **kw):
        codes_list, _ = orig_outer_generate(self, *a, **kw)
        grabbed["codes"] = codes_list[0].detach().to(torch.int64).cpu()
        # Abort before the codec decode: the code matrix is the anchor, and the waveform is
        # already gated by gen-qwen-ref.py's `codec.*`.
        raise _Captured()

    Talker.generate = talker_probe
    Outer.generate = outer_probe

    tensors = {}
    try:
        for tag, instruct, non_streaming in CASES:
            if tag not in wanted:
                continue
            grabbed.clear()
            kwargs = {} if instruct is None else {"instruct": instruct}
            try:
                model.generate_custom_voice(
                    text=TEXT,
                    language=LANGUAGE,
                    speaker=SPEAKER,
                    non_streaming_mode=non_streaming,
                    do_sample=False,
                    subtalker_dosample=False,
                    max_new_tokens=args.max_frames,
                    **kwargs,
                )
            except _Captured:
                pass
            else:
                die("the outer generate never returned — the reference API has changed")

            codes = grabbed["codes"]            # [T, num_code_groups]
            seq = grabbed["sequences"]          # [S]
            stopped_on_eos = bool(seq.numel() and int(seq[-1]) == eos)
            tensors[f"{tag}.codes"] = codes.contiguous()
            tensors[f"{tag}.talker_sequence"] = seq.contiguous()
            tensors[f"{tag}.inputs_embeds"] = grabbed["inputs_embeds"][0].contiguous()
            tensors[f"{tag}.trailing_text_hidden"] = grabbed["trailing_text_hidden"][0].contiguous()
            # A tiny int64 record so the Rust side asserts on measurements, not on prose:
            # [frames, num_code_groups, stopped_on_eos, codec_eos_id, max_frames_cap,
            #  min_new_tokens, len(suppress_tokens), non_streaming, has_instruct]
            tensors[f"{tag}.meta"] = torch.tensor(
                [
                    codes.shape[0],
                    codes.shape[1],
                    int(stopped_on_eos),
                    eos,
                    int(args.max_frames),
                    int(grabbed["talker_kwargs"]["min_new_tokens"]),
                    int(grabbed["n_suppress"]),
                    int(non_streaming),
                    int(instruct is not None),
                ],
                dtype=torch.int64,
            )
            print(
                f"  {tag:9s} frames {codes.shape[0]:4d} x {codes.shape[1]} groups   "
                f"sampled {seq.numel()} ids, stopped_on_eos={stopped_on_eos}   "
                f"prompt {tuple(grabbed['inputs_embeds'].shape)}  "
                f"trailing {tuple(grabbed['trailing_text_hidden'].shape)}  "
                f"rep_penalty={grabbed['talker_kwargs']['repetition_penalty']}  "
                f"min_new={grabbed['talker_kwargs']['min_new_tokens']}  "
                f"suppress={grabbed['n_suppress']}"
            )
            if not stopped_on_eos:
                print(
                    f"  {tag:9s} NOTE: ran into the {args.max_frames}-frame cap without an "
                    "EOS; the test will gate the frames it produced, not the stop condition."
                )
            if codes.shape[0] < 8:
                die(
                    f"{tag}: only {codes.shape[0]} frames — too short to say anything about a "
                    "multi-step loop. Refusing to write a fixture that cannot gate the gap."
                )
    finally:
        Talker.generate = orig_talker_generate
        Outer.generate = orig_outer_generate

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    save_file(tensors, args.out)
    print(f"[gen-qwen-ref-greedy] wrote {args.out}")
    print(
        f"[gen-qwen-ref-greedy] probe case: text={TEXT!r} speaker={SPEAKER!r} "
        f"language={LANGUAGE!r} instruct={INSTRUCT!r}"
    )


if __name__ == "__main__":
    sys.exit(main())
