#!/usr/bin/env python3
"""
gen-qwen-ref-speaker.py — dump the Qwen3-TTS **speaker encoder** (x-vector) reference
for `tests/real_qwen_speaker_parity.rs`.

The third sibling of `gen-fish-ref.py` / `gen-qwen-ref.py`, and it exists because the
`-Base` (voice-clone) checkpoints condition on an x-vector extracted from a reference
WAV, and nothing in-tree had ever compared that extraction against the reference. The
port's own in-crate `speaker::tests::real_checkpoint_parity` pins hand-copied numbers on
a synthetic waveform; this dumps the reference's OWN tensors for a REAL clip, on the real
call path, into a file the frozen test reads.

Everything captured here comes from the reference's own modules — never a
reimplementation. The script drives `Qwen3TTSModel.create_voice_clone_prompt`, which is
exactly what `generate_voice_clone` calls, and hooks:

  * `Qwen3TTSForConditionalGeneration.extract_speaker_embedding` — to capture the
    24 kHz waveform the wrapper hands it (i.e. after `librosa.resample`), so the Rust
    side is fed the identical samples and the anchor cannot be spent on a resampler
    difference. Rust has no librosa/soxr, so resampling is deliberately OUTSIDE the gate.
  * `Qwen3TTSSpeakerEncoder.forward` — to capture its input mel and its output vector
    separately. Two anchors instead of one: if the x-vector disagrees, the mel anchor
    says immediately whether the fault is the front end or the encoder.

Environment (not vendored — the reference is a separate package):
    a python that can `import qwen_tts`; on this box /home/floofy/.venvs/qwen/bin/python.

Usage:
    MEMMAX=12G scripts/run-isolated.sh /home/floofy/.venvs/qwen/bin/python \\
        scripts/gen-qwen-ref-speaker.py \\
        --ckpt /data/models/Qwen3-TTS-12Hz-1.7B-Base \\
        --wav  /home/floofy/refs/voice_en_10s.wav \\
        --out  ~/parity-qwen/speaker.safetensors

CPU/float32 by default, and that default is load-bearing — the same rule as
`gen-qwen-ref.py`. The encoder is a deep dilated-conv stack whose accumulation order
matters; the reference computing this on CUDA vs CPU disagrees with itself far more than
a real porting fault would, so a CUDA fixture would spend the whole error budget on the
device. `--device` exists to MEASURE that drift, not to produce the shipped fixture.

Tensors written:

  wav24        [N]           the resampled 24 kHz mono clip, exactly as the reference's
                             `extract_speaker_embedding` received it
  mel          [T, 128]      the reference `mel_spectrogram(...).transpose(1,2)`, i.e.
                             the speaker encoder's own input (batch dim squeezed)
  xvector      [enc_dim]     the reference `Qwen3TTSSpeakerEncoder` output for that mel
  wav_in       [M]           the clip as `librosa.load(path, sr=None, mono=True)` read
                             it, before resampling — for the driver-path diagnostic only
"""
import argparse
import functools
import os
import sys


def die(msg: str) -> "typing.NoReturn":  # noqa: F821
    raise SystemExit(f"[gen-qwen-ref-speaker] FATAL: {msg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="local Qwen3-TTS-12Hz-*-Base checkpoint dir")
    ap.add_argument("--wav", required=True, help="reference clip (any rate; resampled by the reference)")
    ap.add_argument("--out", required=True, help="output .safetensors")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["bfloat16", "float32"])
    args = ap.parse_args()

    if not os.path.isdir(args.ckpt):
        die(f"--ckpt is not a directory: {args.ckpt}")
    if not os.path.isfile(args.wav):
        die(f"--wav is not a file: {args.wav}")

    try:
        import torch
        from safetensors.torch import save_file
        import librosa
        from qwen_tts import Qwen3TTSModel
        from qwen_tts.core.models.modeling_qwen3_tts import (
            Qwen3TTSForConditionalGeneration as Cond,
            Qwen3TTSSpeakerEncoder as SpkEnc,
        )
    except Exception as e:  # noqa: BLE001
        die(
            f"cannot import the reference ({type(e).__name__}: {e}). "
            "This script never approximates the reference — install qwen-tts and retry."
        )

    dtype = getattr(torch, args.dtype)
    model = Qwen3TTSModel.from_pretrained(args.ckpt, device_map=args.device, dtype=dtype)
    if model.model.tts_model_type != "base":
        die(
            f"--ckpt is a '{model.model.tts_model_type}' checkpoint; the speaker encoder "
            "only exists on the -Base (voice-clone) ones"
        )

    grabbed = {}

    # `extract_speaker_embedding(audio=<np.float32 24 kHz>, sr=24000)` — capture its input
    # so the Rust test never has to reproduce librosa's resampler.
    orig_extract = Cond.extract_speaker_embedding

    @functools.wraps(orig_extract)
    def extract_probe(self, audio, sr):
        grabbed["wav24"] = torch.as_tensor(audio).detach().float().cpu().reshape(-1)
        grabbed["sr24"] = int(sr)
        return orig_extract(self, audio=audio, sr=sr)

    Cond.extract_speaker_embedding = extract_probe

    # The encoder's own input and output. `mels` is `[1, T, 128]`; the returned vector is
    # `[1, enc_dim]` and the caller takes `[0]`.
    orig_fwd = SpkEnc.forward

    @functools.wraps(orig_fwd)
    def enc_probe(self, hidden_states):
        out = orig_fwd(self, hidden_states)
        grabbed.setdefault("mel", hidden_states.detach().float().cpu())
        grabbed.setdefault("enc_out", out.detach().float().cpu())
        return out

    SpkEnc.forward = enc_probe

    try:
        items = model.create_voice_clone_prompt(ref_audio=args.wav, x_vector_only_mode=True)
    finally:
        Cond.extract_speaker_embedding = orig_extract
        SpkEnc.forward = orig_fwd

    for key in ("wav24", "mel", "enc_out"):
        if key not in grabbed:
            die(f"the reference never produced {key} — its API has changed")
    if grabbed["sr24"] != 24_000:
        die(f"the reference fed the encoder {grabbed['sr24']} Hz audio, not 24000")

    # `create_voice_clone_prompt` returns the same vector it stored in the prompt item;
    # cross-check the hook against it so a wrong hook cannot pass silently.
    item_vec = items[0].ref_spk_embedding.detach().float().cpu().reshape(-1)
    enc_vec = grabbed["enc_out"].reshape(-1)
    drift = (item_vec - enc_vec).abs().max().item()
    if drift != 0.0:
        die(f"hooked encoder output differs from the prompt item's by {drift} — bad hook")

    wav_in, sr_in = librosa.load(args.wav, sr=None, mono=True)

    tensors = {
        "wav24": grabbed["wav24"].contiguous(),
        "mel": grabbed["mel"][0].contiguous(),
        "xvector": item_vec.contiguous(),
        "wav_in": torch.as_tensor(wav_in).float().reshape(-1).contiguous(),
    }
    meta = {
        "wav": os.path.abspath(args.wav),
        "ckpt": os.path.abspath(args.ckpt),
        "device": args.device,
        "dtype": args.dtype,
        "sr_in": str(int(sr_in)),
        "sr24": str(grabbed["sr24"]),
    }
    for k, v in tensors.items():
        print(f"  {k:10s} {tuple(v.shape)}")
    norm = (item_vec * item_vec).sum().sqrt().item()
    print(f"  x-vector L2 norm {norm:.6f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    save_file(tensors, args.out, metadata=meta)
    print(f"[gen-qwen-ref-speaker] wrote {args.out}")
    print(f"[gen-qwen-ref-speaker] probe case: wav={args.wav} ({sr_in} Hz -> 24000 Hz), "
          f"ckpt={args.ckpt}, {args.device}/{args.dtype}")


if __name__ == "__main__":
    sys.exit(main())
