#!/usr/bin/env python3
"""
gen-qwen-ref-encoder.py — dump a Qwen3-TTS **encode** parity reference.

`gen-qwen-ref.py` anchors everything that runs in the synthesis direction: the prompt,
the talker, the code predictor and the codec DECODER. The analysis direction —
waveform -> RVQ codes, which is what a `*-Base` checkpoint uses to clone a voice from a
reference clip — had no reference anchor at all. `crates/syrinx-qwen/src/codec/encoder.rs`
implements it and could only be checked against itself. This script closes that gap.

Every number here comes from the REFERENCE's own modules, through the REFERENCE's own
call path — never a reimplementation and never a hand-composed sequence of submodule
calls:

    Qwen3TTSTokenizer.encode(audio, sr)                 # qwen_tts/inference/…tokenizer.py
      -> feature_extractor(raw_audio=…)                 # EncodecFeatureExtractor
      -> Qwen3TTSTokenizerV2Model.encode(values, mask)  # …/tokenizer_12hz/modeling_…v2.py
         -> Qwen3TTSTokenizerV2Encoder.encode(…)        # a stock HF MimiModel
            -> encoder / encoder_transformer / downsample / quantizer.encode
         -> [:encoder_valid_num_quantizers], trim to ceil(n_samples / 1920), transpose

That is exactly how `Qwen3TTSModel.create_voice_clone_prompt` obtains `ref_code`
(`self.model.speech_tokenizer.encode(ref_wavs_for_code, sr=…)`). If the reference cannot
be imported, this script refuses rather than approximating.

What is captured, per case:

  <case>.wav      the f32 samples that actually reached `Qwen3TTSTokenizerV2Model.encode`
                  — i.e. AFTER librosa's resample to 24 kHz and after the feature
                  extractor. Stored rather than described, because the Rust side cannot
                  reproduce librosa's resampler and must not have to: the anchor is the
                  encode stack, not the resampler.
  <case>.latent   the pre-quantizer latent `[hidden, frames]` — the reference's own input
                  to `quantizer.encode`. Continuous, so drift shows here even when the
                  argmin happens to land on the same codebook entry.
  <case>.codes    the returned `audio_codes[0]`, `[frames, valid_num_quantizers]` int64.
                  Discrete, so an argmin flip shows here.

Two cases, and the second is not redundant:

  full            the whole reference clip, hop-aligned by construction.
  ragged          a deliberately non-hop-aligned prefix, so the cascade emits an ODD
                  number of 25 Hz steps and `downsample`'s right-side *replicate* pad is
                  exercised. A zero-pad there would pass `full` and fail `ragged`.

CPU / float32 by default, and that default is load-bearing — the same rule
`gen-qwen-ref.py` documents. These conv stacks accumulate differently per device: the
reference decoding identical codes on CUDA vs CPU disagrees with ITSELF by 0.031 max abs
on a [-1,1] waveform. The Rust parity path is CPU/f32, so a CUDA fixture would spend the
entire error budget on the device and leave nothing to detect a real fault with. For
codes the hazard is worse than a loose tolerance: a device-level drift that flips one
argmin turns an exact-equality gate into a false alarm.

Environment (not vendored — the reference is a separate package):
    a python that can `import qwen_tts` (here: ~/.venvs/qwen/bin/python)

Usage:
    scripts/gen-qwen-ref-encoder.py \
        --tok /data/models/Qwen3-TTS-Tokenizer-12Hz \
        --wav ~/refs/voice_en_10s.wav \
        --out ~/parity-qwen/encode.safetensors

Consumed by `tests/real_qwen_encode_parity.rs` via `SYRINX_QWEN_REF_ENCODE`.
"""
import argparse
import json
import os
import sys

# The ragged case's length in samples. Chosen to be a prime-ish non-multiple of both the
# 960-sample cascade hop and the 1920-sample frame hop, so it lands mid-step on both:
# 60297 = 62 * 960 + 777, giving 63 cascade steps (odd) and 32 frames.
RAGGED_SAMPLES = 60297


def die(msg: str) -> "typing.NoReturn":  # noqa: F821
    raise SystemExit(f"[gen-qwen-ref-encoder] FATAL: {msg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True, help="Qwen3-TTS-Tokenizer-12Hz directory")
    ap.add_argument("--wav", required=True, help="a real reference clip (any sr; resampled)")
    ap.add_argument("--out", required=True, help="output .safetensors")
    # See the module docstring: CPU/f32 is not a convenience default, it is the parity
    # path. Overridable only so a future investigation can measure the device gap.
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["bfloat16", "float32"])
    args = ap.parse_args()

    if not os.path.isdir(args.tok):
        die(f"--tok is not a directory: {args.tok}")
    if not os.path.isfile(args.wav):
        die(f"--wav is not a file: {args.wav}")

    try:
        import numpy as np
        import torch
        from safetensors.torch import save_file
        from qwen_tts import Qwen3TTSTokenizer
        from qwen_tts.core import Qwen3TTSTokenizerV2Model
    except Exception as e:  # noqa: BLE001
        die(
            f"cannot import the reference ({type(e).__name__}: {e}). "
            "This script never approximates the reference — install qwen-tts and retry."
        )

    dtype = getattr(torch, args.dtype)
    tok = Qwen3TTSTokenizer.from_pretrained(args.tok, device_map=args.device, dtype=dtype)
    if tok.get_model_type() != "qwen3_tts_tokenizer_12hz":
        die(f"--tok is a {tok.get_model_type()}, not the 12 Hz tokenizer")

    hop = tok.get_encode_downsample_rate()
    sr_in = tok.get_input_sample_rate()

    # ---- probes on the reference's own call path -------------------------------------
    # `Qwen3TTSTokenizerV2Model.encode` is where the waveform enters the model, so its
    # `input_values` is the exact tensor the Rust encoder must be handed. `quantizer.encode`
    # is where the latent enters the codebook search, so its `embeddings` is the latent.
    grabbed = {}

    orig_encode = Qwen3TTSTokenizerV2Model.encode

    def encode_probe(self, input_values, padding_mask=None, return_dict=None):
        grabbed["input_values"] = input_values.detach().float().cpu()
        grabbed["padding_mask"] = None if padding_mask is None else padding_mask.detach().cpu()
        return orig_encode(self, input_values, padding_mask, return_dict)

    quantizer = tok.model.encoder.quantizer
    orig_q = type(quantizer).encode

    def q_probe(self, embeddings, num_quantizers=None):
        grabbed["latent"] = embeddings.detach().float().cpu()
        return orig_q(self, embeddings, num_quantizers)

    Qwen3TTSTokenizerV2Model.encode = encode_probe
    type(quantizer).encode = q_probe

    tensors = {}
    meta = {
        "source_wav": os.path.abspath(args.wav),
        "tokenizer": os.path.abspath(args.tok),
        "device": args.device,
        "dtype": args.dtype,
        "input_sample_rate": str(sr_in),
        "encode_downsample_rate": str(hop),
        "valid_num_quantizers": str(tok.model.encoder_valid_num_quantizers),
    }

    def run(case: str, audio, sr):
        grabbed.clear()
        enc = tok.encode(audio, sr=sr)
        if "input_values" not in grabbed:
            die("Qwen3TTSTokenizerV2Model.encode was never called — the reference API has changed")
        if "latent" not in grabbed:
            die("the quantizer was never called — the reference API has changed")

        iv = grabbed["input_values"]
        if iv.dim() != 2 or iv.shape[0] != 1:
            die(f"{case}: expected a single-item [1, L] input_values, got {tuple(iv.shape)}")
        wav = iv[0].contiguous()

        lat = grabbed["latent"]
        if lat.dim() != 3 or lat.shape[0] != 1:
            die(f"{case}: expected a [1, C, T] latent, got {tuple(lat.shape)}")

        codes = enc.audio_codes[0].detach().to(torch.int64).cpu().contiguous()

        n = wav.numel()
        want_frames = -(-n // hop)
        if codes.shape[0] != want_frames:
            die(
                f"{case}: reference returned {codes.shape[0]} frames for {n} samples; "
                f"ceil({n}/{hop}) = {want_frames}. The frame-count contract the Rust "
                "encoder implements no longer matches the reference."
            )

        tensors[f"{case}.wav"] = wav
        tensors[f"{case}.latent"] = lat[0].contiguous()
        tensors[f"{case}.codes"] = codes
        print(
            f"  {case:8s} wav {tuple(wav.shape)} @ {sr_in} Hz -> latent "
            f"{tuple(lat[0].shape)} -> codes {tuple(codes.shape)} "
            f"[min {int(codes.min())}, max {int(codes.max())}]"
        )
        return wav

    # `full`: the clip itself, by path, so librosa's load+resample runs exactly as it does
    # in `create_voice_clone_prompt`.
    full = run("full", args.wav, None)

    # `ragged`: a prefix of the ALREADY-resampled samples, handed back at 24 kHz so the
    # resampler is a no-op and the prefix is bit-exact. Encoded on its own — batching it
    # with `full` would pad it and change what the conv stack sees.
    if full.numel() <= RAGGED_SAMPLES:
        die(f"--wav is only {full.numel()} samples at {sr_in} Hz; need > {RAGGED_SAMPLES}")
    run("ragged", full[:RAGGED_SAMPLES].numpy().astype(np.float32), sr_in)

    Qwen3TTSTokenizerV2Model.encode = orig_encode
    type(quantizer).encode = orig_q

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    save_file(tensors, args.out, metadata=meta)
    print(f"[gen-qwen-ref-encoder] wrote {args.out}")
    print(f"[gen-qwen-ref-encoder] {json.dumps(meta, indent=2)}")


if __name__ == "__main__":
    sys.exit(main())
