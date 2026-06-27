//! End-to-end functional smoke for the **Fish Audio `openaudio-s1-mini`** port.
//!
//! Honest smoke coverage of the FULL s1 pipeline: tiktoken tokenizer + prompt builder →
//! Llama-style dual-AR slow backbone → 4-layer fast AR → modded-DAC RVQ codec → 44.1 kHz
//! waveform (`syrinx_fish::s1::S1Mini::synthesize`). Bit-parity is NOT asserted here (the
//! sampler is stochastic from a seed and the codec is parity-checked in
//! `real_fish_s1_parity`); this proves the wired chain runs and yields finite, non-silent
//! audio of a plausible length.
//!
//! Gated on the `real` feature AND `SYRINX_FISH_S1_DIR` pointing at the on-box checkpoint
//! directory (model.safetensors + codec.safetensors + tokenizer.json + optional
//! config.json). `SYRINX_FISH_REF_WAV` is OPTIONAL — when set, it also round-trips the
//! reference through the codec `encode` (cloning-codes) path and checks the code-matrix
//! shape. Skips cleanly when the dir is absent (device-bound recipe; the maintainer runs
//! it on the model box):
//!
//!   SYRINX_FISH_S1_DIR=/root/models/openaudio-s1-mini \
//!   [SYRINX_FISH_REF_WAV=/root/refs/voice.wav] \
//!   [SYRINX_FISH_MAXFRAMES=48] \
//!   cargo test --features real --test real_fish_s1_e2e -- --nocapture
//!
//! PARITY: numeric output is parity-UNVERIFIED until run on box — this test asserts only
//! the structural invariants (finite / non-silent / plausible length), never an exact value.

#![cfg(feature = "real")]

use candle_core::{Device, Tensor};

use syrinx_fish::common::audio as fish_audio;
use syrinx_fish::common::codec::RvqCodec;
use syrinx_fish::common::dualar::DriveParams;
use syrinx_fish::s1::S1Mini;

/// An env var that names an existing path, or `None`.
fn env_path(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|p| std::path::Path::new(p).exists())
}

#[test]
fn real_fish_s1_e2e_smoke() {
    let dir = match env_path("SYRINX_FISH_S1_DIR") {
        Some(d) => d,
        None => {
            eprintln!(
                "SKIP real_fish_s1_e2e: set SYRINX_FISH_S1_DIR to the on-box \
                 openaudio-s1-mini checkpoint directory"
            );
            return;
        }
    };

    let dev = Device::Cpu;
    let mut model = S1Mini::load(&dir, dev.clone()).expect("load openaudio-s1-mini");

    // A short, emotion-TAGGED text. The tag is Fish-native PLAIN TEXT — it flows through the
    // tokenizer like any other characters (no special emotion module).
    let text = "[happy] Hello from Syrinx, this is the s1 mini smoke test.";

    // Keep the frame budget small so a CPU run is tractable; overridable on box.
    let max_frames = std::env::var("SYRINX_FISH_MAXFRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(48usize);
    let params = DriveParams {
        seed: 0,
        max_new_frames: max_frames,
        ..Default::default()
    };

    let wav = model.synthesize(text, &params).expect("s1 synthesize");

    // ---- structural assertions (no bit-parity on the live path) ----
    assert!(!wav.is_empty(), "s1 synthesis produced no audio");
    assert!(
        wav.iter().all(|x| x.is_finite()),
        "s1 audio has non-finite samples"
    );
    let rms = (wav.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / wav.len() as f64).sqrt();
    let peak = wav.iter().fold(0f32, |m, &x| m.max(x.abs()));

    // Plausible length: the codec emits exactly `frame_hop` samples per generated frame, so
    // the waveform length is bounded above by `(max_new_frames + slack) * frame_hop`, and is
    // at least one frame (the loop ran). `RvqCodec::config` is fully-qualified to disambiguate
    // from the `DualArBackend::config` method of the same name.
    let codec_cfg = <S1Mini as RvqCodec>::config(&model);
    let hop = codec_cfg.frame_hop;
    let sr = codec_cfg.sample_rate;
    let dur = wav.len() as f64 / sr as f64;
    eprintln!(
        "s1 e2e: {} samples ({dur:.3}s @{sr}Hz)  rms={rms:.5}  peak={peak:.4}  \
         frame_hop={hop}  max_frames={max_frames}",
        wav.len()
    );
    assert!(
        wav.len() >= hop,
        "s1 audio shorter than one codec frame ({} < hop {hop})",
        wav.len()
    );
    assert!(
        wav.len() <= (max_frames + 2) * hop,
        "s1 audio longer than the frame budget allows ({} > {})",
        wav.len(),
        (max_frames + 2) * hop
    );
    assert!(rms > 1e-6 && rms.is_finite(), "s1 audio is silent (rms {rms:.3e})");

    // ---- optional: reference-codec encode round-trip (cloning-codes path) ----
    if let Some(ref_wav) = env_path("SYRINX_FISH_REF_WAV") {
        let samples = fish_audio::read_ref_wav_44k(&ref_wav).expect("read SYRINX_FISH_REF_WAV");
        let n = samples.len();
        let wav_t = Tensor::from_vec(samples, n, &dev).expect("ref wav tensor");
        let codes = model.encode_reference(&wav_t).expect("s1 encode_reference");
        let nc = codes.dim(0).expect("codes dim0");
        eprintln!(
            "s1 e2e: reference codec encode -> codes {:?} ({} samples @44.1k)",
            codes.dims(),
            n
        );
        assert_eq!(
            nc, codec_cfg.num_codebooks,
            "encode_reference must yield [num_codebooks, T] (got {nc} rows, want {})",
            codec_cfg.num_codebooks
        );
        assert!(codes.dim(1).expect("codes dim1") > 0, "ref encode produced 0 frames");
    }

    eprintln!("PASS: s1 e2e — wired tokenizer->slow-AR->fast-AR->codec produces plausible audio.");
}
