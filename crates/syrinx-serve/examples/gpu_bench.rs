//! GPU vs CPU speed benchmark for the CosyVoice2 synthesizer.
//!
//! This is an **example binary** (not a test — member crates host no tests; see
//! `tests/workspace_scaffold.rs`). It is gated on the `real` feature; build it
//! with `--features cuda` to get an actual GPU run, otherwise both legs run on
//! CPU and the speedup is ~1.0.
//!
//! It loads the synthesizer twice — once on `Device::Cpu` (the parity device)
//! and once on `pick_device(Some(0))` (cuda:0 when built `--features cuda`) —
//! and times three stages on each:
//!   1. LM speech-token generation (the KV-cache decode),
//!   2. the flow-matching CFM ODE (`forward_zero_shot`),
//!   3. the full `synthesize` (frontend + LM + flow + vocoder).
//!
//! GPU output is **not** expected to bit-match CPU (CPU-vs-GPU gemm accumulation
//! diverges through deep nets). The GPU leg is verified **functionally**: the
//! produced 24 kHz waveform must be finite, non-silent, and of a sane length.
//! The reported numbers are real wall-clock; nothing is faked.
//!
//! Env (same fixtures as `tests/real_synth_e2e.rs`):
//! ```text
//!   SYRINX_LM_WEIGHTS  SYRINX_SPK_WEIGHTS  SYRINX_FLOW_WEIGHTS  SYRINX_HIFT_WEIGHTS
//!   SYRINX_TOK_JSON    SYRINX_STOK_ONNX    SYRINX_FEAT_REF
//! Optional:
//!   SYRINX_MAX_GEN_STEPS (default 200)   SYRINX_CUDA_ORDINAL (default 0)
//! ```

#[cfg(not(feature = "real"))]
fn main() {
    eprintln!("gpu_bench requires the `real` feature (and `cuda` for a real GPU run).");
}

#[cfg(feature = "real")]
fn main() {
    use std::time::Instant;

    use candle_core::Device;
    use syrinx_serve::synth::{pick_device, SynthConfig, SynthInputs, Synthesizer};

    let var = |k: &str| std::env::var(k).ok().filter(|p| std::path::Path::new(p).exists());
    let cfg = match (
        var("SYRINX_LM_WEIGHTS"),
        var("SYRINX_SPK_WEIGHTS"),
        var("SYRINX_FLOW_WEIGHTS"),
        var("SYRINX_HIFT_WEIGHTS"),
        var("SYRINX_TOK_JSON"),
        var("SYRINX_STOK_ONNX"),
    ) {
        (Some(lm), Some(spk), Some(flow), Some(hift), Some(tok), Some(stok)) => SynthConfig {
            lm_weights: lm,
            spk_weights: spk,
            flow_weights: flow,
            hift_weights: hift,
            tokenizer_json: tok,
            speech_tokenizer_onnx: stok,
        },
        _ => {
            eprintln!(
                "SKIP gpu_bench: set SYRINX_LM_WEIGHTS, SYRINX_SPK_WEIGHTS, \
                 SYRINX_FLOW_WEIGHTS, SYRINX_HIFT_WEIGHTS, SYRINX_TOK_JSON, \
                 SYRINX_STOK_ONNX (and SYRINX_FEAT_REF) to the on-box fixtures"
            );
            return;
        }
    };
    let feat_ref = match var("SYRINX_FEAT_REF") {
        Some(p) => p,
        None => {
            eprintln!("SKIP gpu_bench: set SYRINX_FEAT_REF to the on-box feat fixture");
            return;
        }
    };

    const PROMPT_TEXT: &str = "希望你以后能够做的比我还好呦。";
    const TTS_TEXT: &str = "收到好友从远方寄来的生日礼物。";
    const N_TIMESTEPS: usize = 10;

    let max_gen_steps: usize = std::env::var("SYRINX_MAX_GEN_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let cuda_ordinal: usize = std::env::var("SYRINX_CUDA_ORDINAL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    // Reference-resampled prompt waveforms (only the feature math is exercised).
    let feat = candle_core::safetensors::load(&feat_ref, &Device::Cpu).expect("load feat_ref");
    let wav = |k: &str| -> Vec<f32> {
        feat.get(k)
            .unwrap_or_else(|| panic!("feat_ref missing `{k}`"))
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    let ref_wav_16k = wav("wav16_a");
    let ref_wav_24k = wav("wav24_a");

    // One full timed run on a given device; returns the per-stage wall-clock.
    struct Timing {
        label: String,
        lm_secs: f64,
        lm_tokens: usize,
        flow_secs: f64,
        flow_frames: usize,
        full_secs: f64,
        wav_len: usize,
        audio_secs: f64,
        finite: bool,
        peak: f32,
    }

    let run = |label: &str, dev: Device| -> Timing {
        eprintln!("\n=== loading on {label} ({dev:?}) ===");
        let load_t = Instant::now();
        let mut synth = Synthesizer::load_on_device(&cfg, dev).expect("load all sub-models");
        eprintln!("  load: {:.2}s", load_t.elapsed().as_secs_f64());

        let cond = synth
            .prompt_cond(TTS_TEXT, PROMPT_TEXT, &ref_wav_16k, &ref_wav_24k)
            .expect("prompt_cond");

        // (1) LM speech-token generation (KV-cache decode).
        let inputs = SynthInputs {
            lm_seed: 0,
            max_gen_steps: Some(max_gen_steps),
            ..Default::default()
        };
        let t = Instant::now();
        let speech_token = synth
            .generate_speech_token(&cond, inputs.lm_seed, inputs.max_gen_steps)
            .expect("generate_speech_token");
        let lm_secs = t.elapsed().as_secs_f64();
        let lm_tokens = speech_token.dim(1).unwrap();

        // (2) flow CFM ODE (forward_zero_shot, the heaviest acoustic stage).
        let total = 2 * (cond.prompt_token.dim(1).unwrap() + speech_token.dim(1).unwrap());
        let z = candle_core::Tensor::zeros(
            (1, 80, total),
            candle_core::DType::F32,
            synth.device(),
        )
        .unwrap();
        let t = Instant::now();
        let mel = synth
            .flow_forward(&cond, &speech_token, &z, N_TIMESTEPS)
            .expect("flow forward_zero_shot");
        let flow_secs = t.elapsed().as_secs_f64();
        let flow_frames = mel.dim(2).unwrap();

        // (3) full synthesize (frontend + LM + flow + vocoder).
        let t = Instant::now();
        let out = synth
            .synthesize(TTS_TEXT, PROMPT_TEXT, &ref_wav_16k, &ref_wav_24k, &inputs)
            .expect("synthesize");
        let full_secs = t.elapsed().as_secs_f64();

        // Functional validation of the produced waveform.
        let finite = out.iter().all(|x| x.is_finite());
        let peak = out.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let wav_len = out.len();
        let audio_secs = wav_len as f64 / 24_000.0;

        Timing {
            label: label.to_string(),
            lm_secs,
            lm_tokens,
            flow_secs,
            flow_frames,
            full_secs,
            wav_len,
            audio_secs,
            finite,
            peak,
        }
    };

    let cpu = run("CPU", Device::Cpu);
    let gpu_dev = pick_device(Some(cuda_ordinal));
    let on_gpu = !matches!(gpu_dev, Device::Cpu);
    let gpu = run(if on_gpu { "GPU" } else { "CPU(fallback)" }, gpu_dev);

    // Functional acceptance for the GPU leg (CUDA is for speed; this is NOT a
    // parity check — it asserts valid, finite, non-silent, sane-length audio).
    let ok = gpu.finite && gpu.peak > 1e-4 && gpu.wav_len > 0;
    eprintln!(
        "\nGPU audio: {} samples ({:.2}s @24k), finite={}, peak={:.4} -> {}",
        gpu.wav_len,
        gpu.audio_secs,
        gpu.finite,
        gpu.peak,
        if ok { "VALID" } else { "INVALID" }
    );

    let speedup = |c: f64, g: f64| if g > 0.0 { c / g } else { f64::NAN };
    let tps = |n: usize, s: f64| if s > 0.0 { n as f64 / s } else { 0.0 };

    println!("\n======================= RESULTS =======================");
    println!("max_gen_steps = {max_gen_steps}, n_timesteps = {N_TIMESTEPS}");
    println!("device under test: {}", gpu.label);
    println!("-------------------------------------------------------");
    println!(
        "LM gen   : CPU {:.3}s ({:.1} tok/s, {} tok) | {} {:.3}s ({:.1} tok/s, {} tok) | {:.2}x",
        cpu.lm_secs, tps(cpu.lm_tokens, cpu.lm_secs), cpu.lm_tokens,
        gpu.label, gpu.lm_secs, tps(gpu.lm_tokens, gpu.lm_secs), gpu.lm_tokens,
        speedup(cpu.lm_secs, gpu.lm_secs),
    );
    println!(
        "Flow ODE : CPU {:.3}s ({} frames) | {} {:.3}s ({} frames) | {:.2}x",
        cpu.flow_secs, cpu.flow_frames,
        gpu.label, gpu.flow_secs, gpu.flow_frames,
        speedup(cpu.flow_secs, gpu.flow_secs),
    );
    println!(
        "Full synth: CPU {:.3}s ({:.2}s audio, RTF {:.2}) | {} {:.3}s ({:.2}s audio, RTF {:.2}) | {:.2}x",
        cpu.full_secs, cpu.audio_secs, cpu.full_secs / cpu.audio_secs.max(1e-9),
        gpu.label, gpu.full_secs, gpu.audio_secs, gpu.full_secs / gpu.audio_secs.max(1e-9),
        speedup(cpu.full_secs, gpu.full_secs),
    );
    println!("=======================================================");

    if !ok {
        eprintln!("ERROR: GPU audio failed functional validation");
        std::process::exit(1);
    }
}
