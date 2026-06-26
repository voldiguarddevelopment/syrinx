//! Functional smoke for the **differentiator control layer** — paralinguistic
//! pass-through + speech-rate control (DESIGN Phase 3+ / Phase 5).
//!
//! This is a *functional* smoke, not a parity test (rigorous perceptual/parity
//! eval is deferred, per the task). It proves two real, audio-affecting knobs on
//! top of the now-complete CosyVoice2 base:
//!
//!   1. **Paralinguistic pass-through** — inserting `[laughter]` into the text
//!      (via `syrinx_prosody::markup`) changes the LM-generated speech-token
//!      sequence vs the unmarked text. The marker is an atomic tokenizer id that
//!      flows text → LM, so the generation diverges — the control reaches the
//!      model, not just the string.
//!   2. **Speech-rate control** — `Synthesizer::synthesize_with_rate` at rates
//!      0.7 / 1.0 / 1.3 produces audio whose duration scales ≈ 1/rate (slower =>
//!      longer, faster => shorter), via the pitch-preserving mel time-scale.
//!
//! Both write sample wavs to `/tmp` (when `SYRINX_SMOKE_OUT_DIR` is set) for a
//! manual listen, and print measured token counts + durations.
//!
//! Gated on the `real` feature AND env vars pointing at the on-box weights +
//! a `feat_ref` carrying the reference-resampled prompt wavs. Skips cleanly when
//! any is absent (device-bound recipe), exactly like `real_synth_e2e`.
//!
//!   SYRINX_LM_WEIGHTS=/root/models/CosyVoice2-0.5B/llm_fp32.safetensors \
//!   SYRINX_SPK_WEIGHTS=/root/parity/speaker/campplus_weights.safetensors \
//!   SYRINX_FLOW_WEIGHTS=/root/models/CosyVoice2-0.5B/flow_fp32.safetensors \
//!   SYRINX_HIFT_WEIGHTS=/root/parity/vocoder/hift_fp32.safetensors \
//!   SYRINX_TOK_JSON=/root/parity/frontend/tokenizer.json \
//!   SYRINX_STOK_ONNX=/root/models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx \
//!   SYRINX_FEAT_REF=/root/parity/frontend/feat_ref.safetensors \
//!   SYRINX_SMOKE_OUT_DIR=/tmp \
//!   SYRINX_SYNTH_MAXSTEPS=200 \
//!   cargo test --features real --test prosody_control_smoke -- --nocapture

#![cfg(feature = "real")]

use std::path::Path;

use candle_core::Device;

use syrinx_prosody::markup::{Markup, Marker};
use syrinx_serve::synth::{SynthConfig, SynthInputs, Synthesizer};

const SR_24K: f32 = 24_000.0;
const PROMPT_TEXT: &str = "希望你以后能够做的比我还好呦。";
/// A short English content line; the markup variant inserts `[laughter]`.
const TTS_TEXT: &str = "I really did not expect that to happen.";

struct Env {
    cfg: SynthConfig,
    feat_ref: String,
}

fn env() -> Option<Env> {
    let v = |k: &str| std::env::var(k).ok().filter(|p| Path::new(p).exists());
    Some(Env {
        cfg: SynthConfig {
            lm_weights: v("SYRINX_LM_WEIGHTS")?,
            spk_weights: v("SYRINX_SPK_WEIGHTS")?,
            flow_weights: v("SYRINX_FLOW_WEIGHTS")?,
            hift_weights: v("SYRINX_HIFT_WEIGHTS")?,
            tokenizer_json: v("SYRINX_TOK_JSON")?,
            speech_tokenizer_onnx: v("SYRINX_STOK_ONNX")?,
        },
        feat_ref: v("SYRINX_FEAT_REF")?,
    })
}

fn wav_vec(r: &std::collections::HashMap<String, candle_core::Tensor>, k: &str) -> Vec<f32> {
    r.get(k)
        .unwrap_or_else(|| panic!("feat_ref missing `{k}`"))
        .to_dtype(candle_core::DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// Minimal 16-bit PCM mono WAV writer (24 kHz) for manual listening.
fn write_wav(path: &Path, samples: &[f32], sr: u32) {
    let n = samples.len();
    let data_bytes = (n * 2) as u32;
    let mut buf: Vec<u8> = Vec::with_capacity(44 + n * 2);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&sr.to_le_bytes());
    buf.extend_from_slice(&(sr * 2).to_le_bytes()); // byte rate
    buf.extend_from_slice(&2u16.to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_bytes.to_le_bytes());
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        buf.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, buf).expect("write wav");
}

#[test]
fn prosody_control_smoke() {
    let env = match env() {
        Some(e) => e,
        None => {
            eprintln!(
                "SKIP prosody_control_smoke: set SYRINX_LM_WEIGHTS, SYRINX_SPK_WEIGHTS, \
                 SYRINX_FLOW_WEIGHTS, SYRINX_HIFT_WEIGHTS, SYRINX_TOK_JSON, \
                 SYRINX_STOK_ONNX, SYRINX_FEAT_REF to the on-box fixtures"
            );
            return;
        }
    };

    let dev = Device::Cpu;
    let feat = candle_core::safetensors::load(&env.feat_ref, &dev).expect("load feat_ref");
    let ref_wav_16k = wav_vec(&feat, "wav16_a");
    let ref_wav_24k = wav_vec(&feat, "wav24_a");

    let mut synth = Synthesizer::load(&env.cfg).expect("load synthesizer");

    // Keep the LM tractable on CPU (no KV cache in some builds). Default 200.
    let max_steps = std::env::var("SYRINX_SYNTH_MAXSTEPS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .or(Some(200));
    let seed = 0u64;

    let out_dir = std::env::var("SYRINX_SMOKE_OUT_DIR").ok();

    // -------------------------------------------------------------------------
    // (1) Paralinguistic pass-through: [laughter] changes the generated tokens.
    // -------------------------------------------------------------------------
    let plain_text = Markup::new().text(TTS_TEXT).render();
    let laugh_text = Markup::new()
        .text(TTS_TEXT)
        .marker(Marker::Laughter)
        .render();
    assert_eq!(plain_text, TTS_TEXT, "plain markup must round-trip the text");
    assert!(
        laugh_text.contains("[laughter]"),
        "laughter markup must carry the marker literal"
    );

    let cond_plain = synth
        .prompt_cond(&plain_text, PROMPT_TEXT, &ref_wav_16k, &ref_wav_24k)
        .expect("prompt_cond plain");
    let cond_laugh = synth
        .prompt_cond(&laugh_text, PROMPT_TEXT, &ref_wav_16k, &ref_wav_24k)
        .expect("prompt_cond laugh");

    // The marker must be a distinct text token (the tokenizer kept it atomic).
    assert!(
        cond_laugh.text_token.len() > cond_plain.text_token.len(),
        "the [laughter] marker must add text token(s): plain={} laugh={}",
        cond_plain.text_token.len(),
        cond_laugh.text_token.len()
    );

    let tok_plain = synth
        .generate_speech_token(&cond_plain, seed, max_steps)
        .expect("gen plain");
    let tok_laugh = synth
        .generate_speech_token(&cond_laugh, seed, max_steps)
        .expect("gen laugh");
    let n_plain = tok_plain.dim(1).unwrap();
    let n_laugh = tok_laugh.dim(1).unwrap();
    let ids_plain: Vec<i64> = tok_plain.flatten_all().unwrap().to_vec1::<i64>().unwrap();
    let ids_laugh: Vec<i64> = tok_laugh.flatten_all().unwrap().to_vec1::<i64>().unwrap();

    println!(
        "[paraling] plain speech-tokens={n_plain}, laughter speech-tokens={n_laugh}"
    );
    // Same seed, same prompt voice — only the text differs. If the marker reached
    // the model, the generated speech-token stream must differ (length and/or ids).
    assert!(
        n_plain != n_laugh || ids_plain != ids_laugh,
        "[laughter] did not change the generated speech tokens — the marker did not \
         reach the model (plain n={n_plain}, laugh n={n_laugh})"
    );
    println!("[paraling] PASS: [laughter] changed the generated speech-token stream");

    // -------------------------------------------------------------------------
    // (2) Speech-rate control: duration scales ~1/rate, pitch preserved by design.
    // -------------------------------------------------------------------------
    let rates = [0.7f64, 1.0, 1.3];
    let mut durations = Vec::new();
    let inputs = SynthInputs {
        lm_seed: seed,
        max_gen_steps: max_steps,
        ..Default::default()
    };
    for &rate in &rates {
        let wav = synth
            .synthesize_with_rate(
                &plain_text,
                PROMPT_TEXT,
                &ref_wav_16k,
                &ref_wav_24k,
                &inputs,
                rate,
            )
            .unwrap_or_else(|e| panic!("synthesize_with_rate({rate}) failed: {e}"));
        let dur_s = wav.len() as f32 / SR_24K;
        assert!(
            wav.iter().all(|x| x.is_finite()) && wav.iter().any(|&x| x.abs() > 1e-4),
            "rate {rate} produced silent/non-finite audio"
        );
        println!(
            "[rate] rate={rate:.2} samples={} duration={dur_s:.3}s",
            wav.len()
        );
        if let Some(dir) = &out_dir {
            let p = Path::new(dir).join(format!("syrinx_rate_{:.0}.wav", rate * 100.0));
            write_wav(&p, &wav, SR_24K as u32);
            println!("[rate] wrote {}", p.display());
        }
        durations.push(dur_s);
    }

    // Monotone: slower (0.7) is longest, faster (1.3) is shortest.
    let (d_slow, d_norm, d_fast) = (durations[0], durations[1], durations[2]);
    assert!(
        d_slow > d_norm && d_norm > d_fast,
        "rate did not scale duration monotonically: 0.7x={d_slow:.3}s 1.0x={d_norm:.3}s \
         1.3x={d_fast:.3}s (expected slow > normal > fast)"
    );
    // Sanity on magnitude: 0.7x should be ~1/0.7 ≈ 1.43x the 1.0x duration, 1.3x
    // ~1/1.3 ≈ 0.77x. Allow generous tolerance (rounding + vocoder edge frames).
    let ratio_slow = d_slow / d_norm;
    let ratio_fast = d_fast / d_norm;
    println!("[rate] duration ratios vs 1.0x: 0.7x={ratio_slow:.3} 1.3x={ratio_fast:.3}");
    assert!(
        ratio_slow > 1.2 && ratio_slow < 1.7,
        "0.7x duration ratio {ratio_slow:.3} not near 1/0.7≈1.43"
    );
    assert!(
        ratio_fast > 0.6 && ratio_fast < 0.9,
        "1.3x duration ratio {ratio_fast:.3} not near 1/1.3≈0.77"
    );
    println!("[rate] PASS: duration scales ~1/rate, pitch-preserving by mel time-scale");
}
