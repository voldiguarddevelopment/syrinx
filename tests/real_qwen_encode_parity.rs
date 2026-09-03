//! Numerical parity for the Qwen3-TTS codec **encoder**: waveform -> RVQ codes.
//!
//! `real_qwen_stack_parity.rs` anchors everything that runs in the synthesis direction —
//! prompt, talker, code predictor, codec *decoder*. The analysis direction had no
//! reference anchor at all, and it is not a minor corner: it is the whole voice-cloning
//! path, the codes a `*-Base` checkpoint conditions on. Until now
//! `crates/syrinx-qwen/src/codec/encoder.rs` could only be checked against itself, which
//! is exactly the situation that let the `silu` bug in `text_projection` ship (see
//! `renders/2026-09-03-qwen-first/FINDINGS.md` §4): an error can be numerically small,
//! audibly invisible, and still wreck behaviour.
//!
//! Fixture: `scripts/gen-qwen-ref-encoder.py`, which drives the reference's OWN call path
//! (`Qwen3TTSTokenizer.encode` -> `Qwen3TTSTokenizerV2Model.encode` -> the stock
//! `MimiModel` encode stack) and refuses to run if it cannot import the reference. Gated
//! on `SYRINX_QWEN_TOK_DIR` and `SYRINX_QWEN_REF_ENCODE`; skips cleanly without them.
//!
//! # Why the input samples come from the fixture
//!
//! The reference's own preprocessing is `librosa.load(..., sr=None)` followed by
//! `librosa.resample` to 24 kHz. Reproducing that resampler in Rust is a separate problem
//! from the encode stack, and mixing the two would mean a resampler difference of 1e-4
//! showing up here as an encoder fault. So the fixture stores the exact f32 samples that
//! reached `Qwen3TTSTokenizerV2Model.encode`, and this test feeds those. What is gated is
//! the encode stack; nothing else is smuggled in.
//!
//! # Why this test pins CPU
//!
//! Unlike the rest of the `real_qwen_*` suite this test does **not** honour
//! `SYRINX_QWEN_DEVICE`. The fixture is CPU/float32 — the parity path — because these conv
//! stacks accumulate differently per device (the reference decoding identical codes on
//! CUDA vs CPU disagrees with itself by 0.031 max abs on a [-1,1] waveform). For a
//! continuous tensor that only costs tolerance. For **codes** it is worse: the quantizer
//! is an `argmin` over 2048 codebook entries, so a device-level drift that flips a single
//! near-tie turns an exact-equality gate into a false alarm and invites someone to
//! "relax" it. Running both sides on CPU/f32 keeps the codes exactly comparable.

#![cfg(feature = "real")]

use candle_core::{DType, Device, Tensor};

use syrinx_qwen::codec::encoder::{encode_chunk_steps_env, MimiEncoder, MimiEncoderConfig};
use syrinx_qwen::nn::Weights;

/// The two probe cases in the fixture. `ragged` is not redundant: its length is not a
/// multiple of the 960-sample cascade hop, so the cascade emits an odd number of 25 Hz
/// steps and `downsample`'s right-side **replicate** pad is exercised. A zero-pad there
/// passes `full` and fails `ragged`.
const CASES: [&str; 2] = ["full", "ragged"];

/// Cascade chunk lengths, in 960-sample steps, that must all reproduce the one-shot
/// result exactly. 0 is the one-shot path itself; 1 and 2 are far below the derived
/// 4-step receptive field bound, so they prove the left-context machinery rather than
/// merely avoiding chunking; 128 is the shipped default.
const CHUNKS_RAGGED: [usize; 5] = [0, 1, 2, 7, 128];
/// The same for the 10 s clip, which is 250 steps — so 64 and 128 really do split it.
/// 1 is omitted here only for runtime: at 250 chunks x 17 steps of context it re-does the
/// cascade 17 times, and `ragged` already pins that boundary.
const CHUNKS_FULL: [usize; 4] = [0, 7, 64, 128];

/// Pre-quantizer latent tolerance, **measured, not chosen**.
///
/// Both sides are CPU/f32 over the identical samples, so the only legitimate residual is
/// reassociation inside the conv and matmul reductions. Measured against this fixture:
/// `full` 3.206e-04 and `ragged` 2.351e-04, against a reference latent whose max abs is
/// 37.492 — i.e. ~9e-6 relative. The bound is ~6x the worst of those, and matches the
/// one the in-crate `matches_the_python_reference` test already carries.
///
/// This is a diagnostic, not the gate: the gate is exact equality on the codes below. It
/// exists so that a real fault reports "the latent drifted" instead of only "some code
/// changed", and so that drift which has not yet flipped an argmin is still visible.
const TOL_LATENT: f32 = 2e-3;

fn env_path(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|p| std::path::Path::new(p).exists())
}

/// `(fixture, tokenizer dir)`, or `None` with a SKIP printed.
fn setup(name: &str) -> Option<(std::collections::HashMap<String, Tensor>, String)> {
    let Some(tok_dir) = env_path("SYRINX_QWEN_TOK_DIR") else {
        eprintln!("SKIP {name}: set SYRINX_QWEN_TOK_DIR to the Qwen3-TTS tokenizer dir");
        return None;
    };
    let Some(ref_path) = env_path("SYRINX_QWEN_REF_ENCODE") else {
        eprintln!(
            "SKIP {name}: set SYRINX_QWEN_REF_ENCODE to the fixture from \
             scripts/gen-qwen-ref-encoder.py"
        );
        return None;
    };
    let refs = candle_core::safetensors::load(&ref_path, &Device::Cpu).expect("load fixture");
    Some((refs, tok_dir))
}

/// Load the encode stack on CPU/f32 — see the module note on why the device is pinned.
fn encoder(tok_dir: &str) -> MimiEncoder {
    let dev = Device::Cpu;
    let dt = DType::F32;
    let cfg_json = std::fs::read_to_string(std::path::Path::new(tok_dir).join("config.json"))
        .expect("tokenizer config.json");
    let cfg = MimiEncoderConfig::from_json(&cfg_json).expect("encoder config");
    let map = syrinx_qwen::load::load_tensors(tok_dir, &dev, dt).expect("load codec tensors");
    MimiEncoder::new(Weights { map, dev, dt }, cfg).expect("build encoder")
}

fn to_vec(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap()
}

/// The fixture's codes are `[frames, n_q]`; the encoder returns `[n_q][frames]`.
fn want_rows(codes: &Tensor) -> Vec<Vec<u32>> {
    let (frames, n_q) = (codes.dim(0).unwrap(), codes.dim(1).unwrap());
    let host: Vec<i64> = codes.flatten_all().unwrap().to_vec1().unwrap();
    (0..n_q)
        .map(|q| (0..frames).map(|t| host[t * n_q + q] as u32).collect())
        .collect()
}

/// Where two code grids first disagree, as a human-readable report. An `argmin` flip is a
/// single cell; a structural fault (an off-by-one frame, a wrong stack) is a whole row or
/// everything from some frame on. The distinction is the first thing anyone chasing a
/// failure needs, so the assertion carries it instead of just "not equal".
fn describe(got: &[Vec<u32>], want: &[Vec<u32>]) -> String {
    let mut bad = 0usize;
    let mut total = 0usize;
    let mut first = None;
    for (q, (g, w)) in got.iter().zip(want).enumerate() {
        for (t, (a, b)) in g.iter().zip(w).enumerate() {
            total += 1;
            if a != b {
                bad += 1;
                if first.is_none() {
                    first = Some((q, t, *a, *b));
                }
            }
        }
    }
    match first {
        None => format!("{total} codes, all equal"),
        Some((q, t, a, b)) => format!(
            "{bad}/{total} codes differ; first at quantizer {q} frame {t}: got {a}, reference {b}"
        ),
    }
}

/// The pre-quantizer latent, compared continuously. Diagnostic — see [`TOL_LATENT`].
#[test]
fn qwen_encoder_latent_matches_the_reference() {
    let Some((refs, tok_dir)) = setup("real_qwen_encode_parity::latent") else { return };
    let enc = encoder(&tok_dir);

    for case in CASES {
        let wav = refs.get(&format!("{case}.wav")).expect("fixture wav");
        let want = refs.get(&format!("{case}.latent")).expect("fixture latent");

        let got_t = enc.latent_with(wav, 0).expect("latent");
        assert_eq!(
            (got_t.dim(1).unwrap(), got_t.dim(2).unwrap()),
            (want.dim(0).unwrap(), want.dim(1).unwrap()),
            "{case}: latent shape {:?} vs reference {:?}",
            got_t.shape(),
            want.shape()
        );

        let (got, want_v) = (to_vec(&got_t), to_vec(want));
        let worst = got.iter().zip(&want_v).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
        let mag = want_v.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        eprintln!(
            "[qwen-encode] {case}: latent {:?} max abs diff {worst:.3e} (tol {TOL_LATENT:.0e}, \
             reference max abs {mag:.3})",
            want.shape()
        );
        assert!(
            worst <= TOL_LATENT,
            "{case}: pre-quantizer latent differs from the reference by {worst} \
             (tolerance {TOL_LATENT}). Both sides are CPU/f32 over the reference's own \
             samples, so this is the cascade, the transformer bottleneck or downsample — \
             not a device or a resampler difference."
        );
    }
}

/// The gate: **exact** equality on the integer codes, through the one-shot cascade.
///
/// Exact and not approximate because there is nothing to approximate — a code is an index.
/// The measured latent gap (~1.7e-4 against magnitudes of order 10) is four orders below
/// the spacing that would flip an `argmin`, and the run below confirms it: 2000 + 512
/// codes, zero differences.
#[test]
fn qwen_encoder_codes_match_the_reference_exactly() {
    let Some((refs, tok_dir)) = setup("real_qwen_encode_parity::codes") else { return };
    let enc = encoder(&tok_dir);
    let n_q = enc.config().valid_num_quantizers;

    for case in CASES {
        let wav = refs.get(&format!("{case}.wav")).expect("fixture wav");
        let want = want_rows(refs.get(&format!("{case}.codes")).expect("fixture codes"));

        let got = enc.encode_oneshot(wav).expect("encode");
        assert_eq!(got.len(), n_q, "{case}: quantizer count");
        assert_eq!(want.len(), n_q, "{case}: fixture quantizer count");
        for (q, row) in got.iter().enumerate() {
            assert_eq!(row.len(), want[q].len(), "{case}: quantizer {q} frame count");
        }

        let report = describe(&got, &want);
        eprintln!("[qwen-encode] {case}: one-shot codes — {report}");
        assert!(
            got == want,
            "{case}: encoded codes differ from the reference ({report}). No language model \
             and no sampling is in this path and both sides ran CPU/f32 on the same samples, \
             so this is the cascade, the bottleneck, downsample, or the split RVQ search."
        );
    }
}

/// Chunking must be a pure optimisation: every chunk length reproduces the one-shot codes,
/// which the test above has already tied to the reference.
///
/// This is the half of the encoder the reference alone cannot police. `encode()` chunks by
/// default (`SYRINX_QWEN_CODEC_ENCODE_CHUNK`, default 128 steps), so the codes a caller
/// actually gets come from the chunked path — and the reference has no chunked path to
/// compare against. Chunk sizes below the derived 4-step receptive field are included
/// deliberately: they fail loudly if the left-context machinery is wrong, where a
/// generously large chunk would hide it.
#[test]
fn qwen_encoder_chunked_and_oneshot_agree_with_the_reference() {
    let Some((refs, tok_dir)) = setup("real_qwen_encode_parity::chunked") else { return };
    let enc = encoder(&tok_dir);

    for case in CASES {
        let wav = refs.get(&format!("{case}.wav")).expect("fixture wav");
        let want = want_rows(refs.get(&format!("{case}.codes")).expect("fixture codes"));
        let chunks: &[usize] = if case == "full" { &CHUNKS_FULL } else { &CHUNKS_RAGGED };

        for &k in chunks {
            let got = enc.encode_with(wav, k).expect("encode_with");
            let report = describe(&got, &want);
            eprintln!("[qwen-encode] {case}: chunk {k:>3} — {report}");
            assert!(
                got == want,
                "{case}: cascade chunk {k} disagrees with the reference ({report}). \
                 The one-shot path is separately gated, so a failure here alone is the \
                 chunking: too little left context, a wrong keep window, or a boundary \
                 that changes each chunk's trailing `extra_padding`."
            );
        }

        // The default entry point, exactly as a caller reaches it.
        let steps = encode_chunk_steps_env();
        let got = enc.encode(wav).expect("encode");
        eprintln!(
            "[qwen-encode] {case}: encode() default ({steps} steps) — {}",
            describe(&got, &want)
        );
        assert!(
            got == want,
            "{case}: encode() with the default chunk length ({steps} steps) disagrees with \
             the reference. This is the path every caller takes."
        );
    }
}
