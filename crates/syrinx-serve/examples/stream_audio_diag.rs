//! Streaming-vs-non-streaming **audio** diagnostic: windowed correlation along the
//! utterance, to distinguish a *constant* source-phase offset (fixable) from an
//! *accumulating* phase drift (which means the per-chunk F0 — hence the mel — must
//! match the non-streaming reference; i.e. cause (2), the non-causal flow re-run).
//!
//! For the same pinned tokens it runs `synthesize` (full) and `synthesize_streaming`,
//! then reports: overall corr, per-window zero-lag corr, and per-window best-lag corr
//! (max over a small +/- sample shift). If best-lag >> zero-lag and windows decay,
//! the source phase is drifting (integrated F0 error). Example binary, `real`-gated.

#[cfg(not(feature = "real"))]
fn main() {
    eprintln!("stream_audio_diag requires the `real` feature.");
}

#[cfg(feature = "real")]
fn main() {
    use candle_core::Device;
    use syrinx_serve::synth::{SynthConfig, SynthInputs, Synthesizer};

    let var = |k: &str| {
        std::env::var(k)
            .ok()
            .filter(|p| std::path::Path::new(p).exists())
    };
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
            eprintln!("SKIP stream_audio_diag: set the SYRINX_* fixtures.");
            return;
        }
    };
    let feat_ref = match var("SYRINX_FEAT_REF") {
        Some(p) => p,
        None => {
            eprintln!("SKIP stream_audio_diag: set SYRINX_FEAT_REF.");
            return;
        }
    };

    const PROMPT_TEXT: &str = "希望你以后能够做的比我还好呦。";
    const TTS_TEXT: &str = "收到好友从远方寄来的生日礼物。";
    let max_gen_steps: usize = std::env::var("SYRINX_MAX_GEN_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120);
    let token_hop: usize = std::env::var("SYRINX_TOKEN_HOP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);

    let feat = candle_core::safetensors::load(&feat_ref, &Device::Cpu).expect("load feat_ref");
    let wav = |k: &str| -> Vec<f32> {
        feat.get(k)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    let ref_wav_16k = wav("wav16_a");
    let ref_wav_24k = wav("wav24_a");

    eprintln!("=== loading synthesizer (CPU) ===");
    let mut synth = Synthesizer::load(&cfg).expect("load all sub-models");

    let cond = synth
        .prompt_cond(TTS_TEXT, PROMPT_TEXT, &ref_wav_16k, &ref_wav_24k)
        .expect("cond");
    let speech_tok = synth
        .generate_speech_token(&cond, 0, Some(max_gen_steps))
        .expect("gen");
    let pinned: Vec<i64> = speech_tok.flatten_all().unwrap().to_vec1::<i64>().unwrap();
    let inputs = || SynthInputs {
        pinned_speech_token: Some(pinned.clone()),
        ..Default::default()
    };

    let full = synth
        .synthesize(TTS_TEXT, PROMPT_TEXT, &ref_wav_16k, &ref_wav_24k, &inputs())
        .expect("full");
    let mut chunks: Vec<Vec<f32>> = Vec::new();
    synth
        .synthesize_streaming(
            TTS_TEXT,
            PROMPT_TEXT,
            &ref_wav_16k,
            &ref_wav_24k,
            &inputs(),
            token_hop,
            |c| {
                chunks.push(c);
                Ok(())
            },
        )
        .expect("stream");
    let streamed: Vec<f32> = chunks.iter().flatten().copied().collect();
    let n = streamed.len().min(full.len());
    eprintln!("full={} stream={} compared={n}", full.len(), streamed.len());
    eprintln!("OVERALL corr = {:.4}", pearson(&streamed[..n], &full[..n]));

    // windowed zero-lag and best-lag (over +/- LAG samples) correlation.
    const W: usize = 2400; // 0.1 s
    const LAG: i64 = 240; // search +/- 10 ms for a phase/timing offset
    eprintln!("\n=== windowed corr (W={W} samples ~0.1s) ===");
    let mut w0 = 0usize;
    while w0 + W <= n {
        let a = &streamed[w0..w0 + W];
        let b = &full[w0..w0 + W];
        let z = pearson(a, b);
        // best-lag: shift `a` against `b` within +/-LAG, both clipped to the overlap.
        let mut best = z;
        let mut best_l = 0i64;
        for l in -LAG..=LAG {
            let c = lag_pearson(a, b, l);
            if c > best {
                best = c;
                best_l = l;
            }
        }
        eprintln!(
            "  [{:6}..{:6}]  zero-lag={:+.3}  best-lag={:+.3} @ {:+}",
            w0,
            w0 + W,
            z,
            best,
            best_l
        );
        w0 += W;
    }
}

#[cfg(feature = "real")]
fn lag_pearson(a: &[f32], b: &[f32], lag: i64) -> f64 {
    // correlate a[i] with b[i+lag] over the valid overlap.
    let n = a.len() as i64;
    let (lo, hi) = (0.max(-lag), n.min(n - lag));
    if hi - lo < 16 {
        return -1.0;
    }
    let (mut sa, mut sb) = (0f64, 0f64);
    let cnt = (hi - lo) as f64;
    for i in lo..hi {
        sa += a[i as usize] as f64;
        sb += b[(i + lag) as usize] as f64;
    }
    let (ma, mb) = (sa / cnt, sb / cnt);
    let (mut cov, mut va, mut vb) = (0f64, 0f64, 0f64);
    for i in lo..hi {
        let da = a[i as usize] as f64 - ma;
        let db = b[(i + lag) as usize] as f64 - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va <= 0.0 || vb <= 0.0 {
        return 0.0;
    }
    cov / (va.sqrt() * vb.sqrt())
}

#[cfg(feature = "real")]
fn pearson(a: &[f32], b: &[f32]) -> f64 {
    lag_pearson(a, b, 0)
}
