//! End-to-end functional smoke for the **Fish Audio `s2-pro`** port.
//!
//! Honest smoke coverage of the FULL s2 pipeline: Qwen3 155k BPE tokenizer + prompt
//! builder → Qwen3-4B (`fish_qwen3_omni`) dual-AR slow backbone → 4-layer ~400M fast AR
//! (shared embedding + codebook-axis RoPE + MCF) → 446M EVA-GAN / causal-DAC codec →
//! 44.1 kHz waveform (`syrinx_fish::s2::S2Pro::synthesize` / `synthesize_cloned`). Bit-
//! parity is NOT asserted here (the codec is parity-checked in `real_fish_s2_parity`);
//! this proves the wired chain runs and yields finite, non-silent audio of plausible
//! length, including the voice-cloning path when a reference is supplied.
//!
//! Gated on the `real` feature AND `SYRINX_FISH_S2_DIR` pointing at the on-box checkpoint
//! directory (sharded model-*.safetensors + index + codec.pth + tokenizer.json +
//! config.json). `SYRINX_FISH_REF_WAV` is OPTIONAL — when set, the smoke runs the CLONED
//! path (`encode_reference` → `synthesize_cloned`) instead of plain synthesis. Skips
//! cleanly when the dir is absent (device-bound recipe; the maintainer runs it on box):
//!
//!   SYRINX_FISH_S2_DIR=/root/models/s2-pro \
//!   [SYRINX_FISH_REF_WAV=/root/refs/voice.wav] \
//!   [SYRINX_FISH_MAXFRAMES=48] \
//!   cargo test --features real --test real_fish_s2_e2e -- --nocapture
//!
//! PARITY: numeric output is parity-UNVERIFIED until run on box — this test asserts only
//! the structural invariants (finite / non-silent / plausible length), never an exact value.

#![cfg(feature = "real")]

use candle_core::{Device, Tensor};

use syrinx_fish::common::audio as fish_audio;
use syrinx_fish::common::codec::RvqCodec;
use syrinx_fish::common::dualar::DriveParams;
use syrinx_fish::s2::S2Pro;

/// An env var that names an existing path, or `None`.
fn env_path(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|p| std::path::Path::new(p).exists())
}

#[test]
fn real_fish_s2_e2e_smoke() {
    let dir = match env_path("SYRINX_FISH_S2_DIR") {
        Some(d) => d,
        None => {
            eprintln!(
                "SKIP real_fish_s2_e2e: set SYRINX_FISH_S2_DIR to the on-box s2-pro \
                 checkpoint directory"
            );
            return;
        }
    };

    let dev = Device::Cpu;
    let mut model = S2Pro::load(&dir, dev.clone()).expect("load s2-pro");

    // A short, emotion-TAGGED text. The tag is Fish-native PLAIN TEXT (no emotion module).
    let text = "[whisper] Hello from Syrinx, this is the s2 pro smoke test.";

    let max_frames = std::env::var("SYRINX_FISH_MAXFRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(48usize);
    let params = DriveParams {
        seed: 0,
        max_new_frames: max_frames,
        ..Default::default()
    };

    // The codec geometry, used both for the optional cloning encode and the length bound.
    // `RvqCodec::config` is fully-qualified to disambiguate from `DualArBackend::config`.
    let codec_cfg = <S2Pro as RvqCodec>::config(&model).clone();

    // When a reference wav is supplied, exercise the CLONED voice path end to end
    // (resample → codec encode → `synthesize_cloned`); otherwise plain text synthesis.
    let wav = match env_path("SYRINX_FISH_REF_WAV") {
        Some(ref_wav) => {
            let samples = fish_audio::read_ref_wav_44k(&ref_wav).expect("read SYRINX_FISH_REF_WAV");
            let n = samples.len();
            let wav_t = Tensor::from_vec(samples, n, &dev).expect("ref wav tensor");
            let ref_codes = model.encode_reference(&wav_t).expect("s2 encode_reference");
            assert_eq!(
                ref_codes.dim(0).expect("ref codes dim0"),
                codec_cfg.num_codebooks,
                "encode_reference must yield [num_codebooks, T]"
            );
            eprintln!(
                "s2 e2e: cloning path — ref {:?} ({n} samples @44.1k) -> ref_codes {:?}",
                ref_wav,
                ref_codes.dims()
            );
            // Empty reference transcript still conditions on the audio codes (matches the CLI
            // default when --prompt-text is omitted).
            model
                .synthesize_cloned("", &ref_codes, text, &params)
                .expect("s2 synthesize_cloned")
        }
        None => {
            eprintln!("s2 e2e: plain synthesis path (set SYRINX_FISH_REF_WAV for the clone path)");
            model.synthesize(text, &params).expect("s2 synthesize")
        }
    };

    // ---- structural assertions (no bit-parity on the live path) ----
    assert!(!wav.is_empty(), "s2 synthesis produced no audio");
    assert!(
        wav.iter().all(|x| x.is_finite()),
        "s2 audio has non-finite samples"
    );
    let rms = (wav.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / wav.len() as f64).sqrt();
    let peak = wav.iter().fold(0f32, |m, &x| m.max(x.abs()));
    let hop = codec_cfg.frame_hop;
    let sr = codec_cfg.sample_rate;
    let dur = wav.len() as f64 / sr as f64;
    eprintln!(
        "s2 e2e: {} samples ({dur:.3}s @{sr}Hz)  rms={rms:.5}  peak={peak:.4}  \
         frame_hop={hop}  max_frames={max_frames}",
        wav.len()
    );
    assert!(
        wav.len() >= hop,
        "s2 audio shorter than one codec frame ({} < hop {hop})",
        wav.len()
    );
    assert!(
        wav.len() <= (max_frames + 2) * hop,
        "s2 audio longer than the frame budget allows ({} > {})",
        wav.len(),
        (max_frames + 2) * hop
    );
    assert!(rms > 1e-6 && rms.is_finite(), "s2 audio is silent (rms {rms:.3e})");

    eprintln!("PASS: s2 e2e — wired tokenizer->slow-AR->fast-AR->codec produces plausible audio.");
}
