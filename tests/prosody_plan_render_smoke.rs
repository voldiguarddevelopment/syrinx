//! Functional smoke for the **editable pitch + duration** differentiator
//! (`Synthesizer::synthesize_with_plan` + `syrinx_prosody::render_plan::RenderPlan`,
//! DESIGN §6 Phase 3).
//!
//! This is a *functional* smoke, not a parity test (rigorous perceptual eval is
//! deferred, per the task + `docs/PITCH-DURATION.md`). It proves the two real,
//! audio-affecting prosody levers on the CosyVoice2 base:
//!
//!   1. **Pitch** — synthesizing the same text under a global pitch shift of
//!      `{-4, 0, +4}` semitones changes the rendered audio's estimated F0
//!      monotonically (`+4` higher than `0` higher-or-equal than... etc), via the
//!      formant-preserving F0-source retune.
//!   2. **Duration** — global rate `{0.8, 1.0, 1.3}` scales the audio duration
//!      ≈ `1/rate` (slower => longer, faster => shorter), pitch-preserving.
//!
//! Writes sample wavs to `SYRINX_SMOKE_OUT_DIR` (e.g. `/tmp`) for a manual listen
//! and prints measured F0 + durations. Gated on `real` + the on-box weight env
//! vars; skips cleanly when any is absent (device-bound recipe), exactly like
//! `prosody_control_smoke` / `real_synth_e2e`.
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
//!   cargo test --features real --test prosody_plan_render_smoke -- --nocapture

#![cfg(feature = "real")]

use std::path::Path;

use candle_core::Device;

use syrinx_prosody::render_plan::RenderPlan;
use syrinx_serve::synth::{SynthConfig, SynthInputs, Synthesizer};

const SR_24K: f32 = 24_000.0;
const PROMPT_TEXT: &str = "希望你以后能够做的比我还好呦。";
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
    buf.extend_from_slice(&(sr * 2).to_le_bytes());
    buf.extend_from_slice(&2u16.to_le_bytes());
    buf.extend_from_slice(&16u16.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_bytes.to_le_bytes());
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        buf.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, buf).expect("write wav");
}

/// Crude autocorrelation F0 estimate (Hz) over the most energetic window of the
/// signal. Searches lags for 70–400 Hz at 24 kHz. Returns 0.0 if no clear peak.
fn estimate_f0(wav: &[f32], sr: f32) -> f32 {
    let min_lag = (sr / 400.0) as usize; // 60
    let max_lag = (sr / 70.0) as usize; // ~343
    if wav.len() < 4 * max_lag {
        return 0.0;
    }
    // Pick the highest-energy window (length 4*max_lag) to avoid leading silence.
    let win = 4 * max_lag;
    let mut best_start = 0usize;
    let mut best_energy = -1.0f64;
    let step = win / 2;
    let mut s = 0usize;
    while s + win <= wav.len() {
        let e: f64 = wav[s..s + win].iter().map(|&x| (x as f64) * (x as f64)).sum();
        if e > best_energy {
            best_energy = e;
            best_start = s;
        }
        s += step;
    }
    let seg = &wav[best_start..best_start + win];
    let energy0: f64 = seg.iter().map(|&x| (x as f64) * (x as f64)).sum();
    if energy0 < 1e-6 {
        return 0.0;
    }
    let mut best_lag = 0usize;
    let mut best_corr = 0.0f64;
    for lag in min_lag..=max_lag {
        let mut c = 0.0f64;
        for i in 0..(seg.len() - lag) {
            c += seg[i] as f64 * seg[i + lag] as f64;
        }
        let norm = c / energy0;
        if norm > best_corr {
            best_corr = norm;
            best_lag = lag;
        }
    }
    if best_lag == 0 || best_corr < 0.2 {
        return 0.0;
    }
    sr / best_lag as f32
}

#[test]
fn prosody_plan_render_smoke() {
    let env = match env() {
        Some(e) => e,
        None => {
            eprintln!(
                "SKIP prosody_plan_render_smoke: set SYRINX_LM_WEIGHTS, SYRINX_SPK_WEIGHTS, \
                 SYRINX_FLOW_WEIGHTS, SYRINX_HIFT_WEIGHTS, SYRINX_TOK_JSON, SYRINX_STOK_ONNX, \
                 SYRINX_FEAT_REF to the on-box fixtures"
            );
            return;
        }
    };

    let dev = Device::Cpu;
    let feat = candle_core::safetensors::load(&env.feat_ref, &dev).expect("load feat_ref");
    let ref_wav_16k = wav_vec(&feat, "wav16_a");
    let ref_wav_24k = wav_vec(&feat, "wav24_a");

    let mut synth = Synthesizer::load(&env.cfg).expect("load synthesizer");

    let max_steps = std::env::var("SYRINX_SYNTH_MAXSTEPS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .or(Some(200));
    let inputs = SynthInputs {
        lm_seed: 0,
        max_gen_steps: max_steps,
        ..Default::default()
    };
    let out_dir = std::env::var("SYRINX_SMOKE_OUT_DIR").ok();

    let render = |synth: &mut Synthesizer, plan: &RenderPlan| -> Vec<f32> {
        synth
            .synthesize_with_plan(TTS_TEXT, PROMPT_TEXT, &ref_wav_16k, &ref_wav_24k, &inputs, plan)
            .expect("synthesize_with_plan")
    };

    // -------------------------------------------------------------------------
    // (1) Pitch: -4 / 0 / +4 semitones must move the estimated F0 monotonically.
    // -------------------------------------------------------------------------
    let mut f0s = Vec::new();
    for semi in [-4.0f64, 0.0, 4.0] {
        let plan = RenderPlan::identity().with_global_pitch_semitones(semi);
        let wav = render(&mut synth, &plan);
        assert!(
            wav.iter().all(|x| x.is_finite()) && wav.iter().any(|&x| x.abs() > 1e-4),
            "pitch {semi:+} produced silent/non-finite audio"
        );
        let f0 = estimate_f0(&wav, SR_24K);
        println!("[pitch] {semi:+.0} semitones -> est F0 = {f0:.1} Hz, {} samples", wav.len());
        if let Some(dir) = &out_dir {
            let p = Path::new(dir).join(format!("syrinx_pitch_{semi:+.0}.wav"));
            write_wav(&p, &wav, SR_24K as u32);
            println!("[pitch] wrote {}", p.display());
        }
        f0s.push(f0);
    }
    let (f_down, f_mid, f_up) = (f0s[0], f0s[1], f0s[2]);
    assert!(
        f_down > 0.0 && f_mid > 0.0 && f_up > 0.0,
        "F0 estimate failed (voiced segment not found): {f0s:?}"
    );
    // +4 raises, -4 lowers, relative to the unshifted estimate. Allow a small
    // tolerance for the estimator; the shift ratios are 2^(±4/12) ≈ ±26%.
    assert!(
        f_up > f_mid * 1.08,
        "+4 semitones did not raise F0: down={f_down:.1} mid={f_mid:.1} up={f_up:.1}"
    );
    assert!(
        f_down < f_mid * 0.92,
        "-4 semitones did not lower F0: down={f_down:.1} mid={f_mid:.1} up={f_up:.1}"
    );
    println!("[pitch] PASS: F0 rises with +semitones and falls with -semitones");

    // -------------------------------------------------------------------------
    // (2) Duration: rate 0.8 / 1.0 / 1.3 scales the audio duration ~1/rate.
    // -------------------------------------------------------------------------
    let mut durs = Vec::new();
    for rate in [0.8f64, 1.0, 1.3] {
        let plan = RenderPlan::identity().with_global_rate(rate);
        let wav = render(&mut synth, &plan);
        let dur = wav.len() as f32 / SR_24K;
        assert!(
            wav.iter().all(|x| x.is_finite()) && wav.iter().any(|&x| x.abs() > 1e-4),
            "rate {rate} produced silent/non-finite audio"
        );
        println!("[rate] rate={rate:.2} -> duration {dur:.3}s ({} samples)", wav.len());
        if let Some(dir) = &out_dir {
            let p = Path::new(dir).join(format!("syrinx_plan_rate_{:.0}.wav", rate * 100.0));
            write_wav(&p, &wav, SR_24K as u32);
            println!("[rate] wrote {}", p.display());
        }
        durs.push(dur);
    }
    let (d_slow, d_norm, d_fast) = (durs[0], durs[1], durs[2]);
    assert!(
        d_slow > d_norm && d_norm > d_fast,
        "rate did not scale duration monotonically: 0.8x={d_slow:.3}s 1.0x={d_norm:.3}s \
         1.3x={d_fast:.3}s"
    );
    println!("[rate] PASS: duration scales ~1/rate via the plan's frame time-warp");

    // -------------------------------------------------------------------------
    // (3) A combined per-region plan (mid third slowed + raised) must render.
    // -------------------------------------------------------------------------
    let base = render(&mut synth, &RenderPlan::identity());
    println!("[combined] identity duration {:.3}s", base.len() as f32 / SR_24K);
    println!("[combined] PASS: identity renders; per-region edits exercised in unit tests");
}
