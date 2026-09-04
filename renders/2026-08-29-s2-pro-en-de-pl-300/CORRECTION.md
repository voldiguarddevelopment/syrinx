# These renders are INVALID above ~5.9 seconds (added 2026-09-04)

On 2026-09-04 the Fish s2-pro codec was checked against the reference implementation for
the first time (`tests/real_fish_s2_parity.rs`, fixture from `scripts/gen-fish-ref.py`)
and two real defects were found and fixed. One of them changes the audio of every render
in this directory that is longer than about six seconds.

**Missing sliding-window attention.** The reference's `pre_module` and `post_module` are
`WindowLimitedTransformer`s with `window_size = 128`; the Rust used a full lower-triangular
causal mask. Below the window the two agree exactly, so nothing shorter than
`128 x 2048 / 44100 = 5.94 s` is affected — and nothing shorter could ever have detected
it. Above it the divergence is gross: measured **0.5377 max abs on a [-1,1] waveform** at
192 frames (8.9 s), which is different audio, not a rounding difference.

**A bf16-rounded RoPE table.** The reference builds its table via `precompute_freqs_cis`
with no `dtype`, which defaults to `torch.bfloat16`, and only then upcasts to f32 — so it
reports f32 while carrying bf16 rounding. The port computed a genuinely-f32 table, i.e. it
was *more* accurate than the model that was trained through that rounding. This one
affects renders of every length, though far more subtly (6.4e-3).

After both fixes the port matches the reference at **1.7e-5** on both a 16-frame and a
192-frame fixture.

**What this means for this corpus:** every sample here longer than ~5.9 s was produced by
the pre-fix codec and should be treated as invalid. The shorter samples are affected only
by the RoPE issue. The manifest, report and FINDINGS in this directory describe a render
that can no longer be reproduced by the current tree; re-render before drawing any
conclusion from the audio. Nothing here has been deleted — the metadata is still the
record of what was run, and the defect is exactly the kind that only a reference could
have caught.
