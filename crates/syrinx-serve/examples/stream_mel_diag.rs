//! Streaming-vs-non-streaming **MEL** diagnostic (Pass: streaming faithfulness).
//!
//! For ONE pinned token sequence this compares the mel the streaming path decodes
//! (the per-chunk full-context flow re-run + slice of the newly-finalized region)
//! against the single non-streaming full-context mel, frame-by-frame. It reports
//! the per-chunk and global max-abs / mean-abs mel diff, plus the diff in the
//! frames straddling each chunk boundary.
//!
//! This decides the scope of the streaming-faithfulness fix:
//!   * mel MATCHES (small diff)  -> the audio divergence is purely source/vocoder
//!     phase; fix the F0 source continuity (cause 1).
//!   * mel DIVERGES (esp. at boundaries) -> the non-causal flow re-run leaks
//!     right-context; a causal cached flow is also needed (cause 2).
//!
//! It is an example binary (member crates host no tests). Gated on `real`, SKIPs
//! cleanly without the on-box fixtures. Env identical to `examples/stream_demo.rs`.

#[cfg(not(feature = "real"))]
fn main() {
    eprintln!("stream_mel_diag requires the `real` feature.");
}

#[cfg(feature = "real")]
fn main() {
    use candle_core::{DType, Device, Tensor};
    use syrinx_serve::synth::{SynthConfig, Synthesizer};

    // Mirrors the private acoustic streaming constants (real.rs).
    const TOKEN_MEL_RATIO: usize = 2;
    const PRE_LOOKAHEAD: usize = 3;
    const N_TIMESTEPS: usize = 10;
    const MEL_NUM_MELS: usize = 80;

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
            eprintln!("SKIP stream_mel_diag: set the SYRINX_* fixtures (see stream_demo).");
            return;
        }
    };
    let feat_ref = match var("SYRINX_FEAT_REF") {
        Some(p) => p,
        None => {
            eprintln!("SKIP stream_mel_diag: set SYRINX_FEAT_REF.");
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
        feat.get(k).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
    };
    let ref_wav_16k = wav("wav16_a");
    let ref_wav_24k = wav("wav24_a");

    eprintln!("=== loading synthesizer (CPU) ===");
    let mut synth = Synthesizer::load(&cfg).expect("load all sub-models");
    let dev = synth.device().clone();

    let cond = synth
        .prompt_cond(TTS_TEXT, PROMPT_TEXT, &ref_wav_16k, &ref_wav_24k)
        .expect("prompt_cond");
    let speech_tok = synth
        .generate_speech_token(&cond, 0, Some(max_gen_steps))
        .expect("generate_speech_token");
    let pinned: Vec<i64> = speech_tok.flatten_all().unwrap().to_vec1::<i64>().unwrap();
    let n_tokens = pinned.len();
    let prompt_len = cond.prompt_token.dim(1).unwrap();
    eprintln!("pinned speech tokens: {n_tokens}, prompt_len: {prompt_len}, token_hop: {token_hop}");

    // Full z = zeros, length 2*(prompt+n).
    let total = TOKEN_MEL_RATIO * (prompt_len + n_tokens);
    let z_full = Tensor::zeros((1, MEL_NUM_MELS, total), DType::F32, &dev).unwrap();

    // ---- (A) non-streaming full mel ----
    let speech_token = Tensor::from_vec(pinned.clone(), (1, n_tokens), &dev).unwrap();
    let mel_ns = synth
        .flow_forward(&cond, &speech_token, &z_full, N_TIMESTEPS)
        .expect("flow_forward full"); // [1,80,2*n]
    let ns_frames = mel_ns.dim(2).unwrap();
    eprintln!("non-streaming mel frames: {ns_frames}");
    let ns: Vec<f32> = mel_ns.flatten_all().unwrap().to_vec1::<f32>().unwrap(); // row-major [80][F]

    // ---- (B) streaming mel: replicate the real.rs per-chunk slice ----
    // Collect (mel_new flat [80*cnt], frame_start, frame_cnt) per chunk.
    let mut stream_frames: Vec<Vec<f32>> = vec![Vec::new(); MEL_NUM_MELS]; // per-mel-row concat
    let mut boundaries: Vec<usize> = Vec::new(); // global frame index at each chunk end
    let mut offset = 0usize;
    let mut chunk_idx = 0usize;
    eprintln!("\n=== per-chunk mel diff (streaming slice vs non-streaming) ===");
    while offset < n_tokens {
        let want_end = (offset + token_hop).min(n_tokens);
        let avail_end = (want_end + PRE_LOOKAHEAD).min(n_tokens);
        let tok_slice = speech_token.narrow(1, 0, avail_end).unwrap();
        let flow_len = TOKEN_MEL_RATIO * (prompt_len + avail_end);
        let z = z_full.narrow(2, 0, flow_len).unwrap().contiguous().unwrap();
        let mel_full = synth
            .flow_forward(&cond, &tok_slice, &z, N_TIMESTEPS)
            .expect("flow_forward chunk"); // [1,80,2*avail_end]
        let mel_start = TOKEN_MEL_RATIO * offset;
        let mel_count = TOKEN_MEL_RATIO * (want_end - offset);
        let mel_new = mel_full.narrow(2, mel_start, mel_count).unwrap().contiguous().unwrap();
        let mn: Vec<f32> = mel_new.flatten_all().unwrap().to_vec1::<f32>().unwrap(); // [80][mel_count]

        // diff this chunk against the corresponding non-streaming frames.
        let mut max_d = 0f32;
        let mut sum_d = 0f64;
        for r in 0..MEL_NUM_MELS {
            for c in 0..mel_count {
                let g = mel_start + c; // global frame
                let a = mn[r * mel_count + c];
                let b = ns[r * ns_frames + g];
                let d = (a - b).abs();
                max_d = max_d.max(d);
                sum_d += d as f64;
                stream_frames[r].push(a);
            }
        }
        let mean_d = sum_d / (MEL_NUM_MELS * mel_count) as f64;
        // also the diff of just the FIRST mel frame of this chunk (the seam) across all rows.
        let seam_max = (0..MEL_NUM_MELS)
            .map(|r| (mn[r * mel_count] - ns[r * ns_frames + mel_start]).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "  chunk {chunk_idx:2}: tokens[{offset:3}..{want_end:3}] mel[{mel_start:4}..{:4}] (avail={avail_end:3}, finalize={})  max|d|={max_d:.4}  mean|d|={mean_d:.5}  seam_max={seam_max:.4}",
            mel_start + mel_count,
            avail_end == n_tokens
        );
        boundaries.push(mel_start + mel_count);
        offset = want_end;
        chunk_idx += 1;
    }

    // ---- (C) global mel diff ----
    let sf = stream_frames[0].len();
    let cmp = sf.min(ns_frames);
    let mut gmax = 0f32;
    let mut gsum = 0f64;
    for r in 0..MEL_NUM_MELS {
        for c in 0..cmp {
            let d = (stream_frames[r][c] - ns[r * ns_frames + c]).abs();
            gmax = gmax.max(d);
            gsum += d as f64;
        }
    }
    let gmean = gsum / (MEL_NUM_MELS * cmp) as f64;
    // reference scale: rms of the non-streaming mel.
    let ns_rms = (ns.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>()
        / ns.len() as f64)
        .sqrt();

    eprintln!("\n=== global mel diff (frames compared: {cmp}, stream={sf} ns={ns_frames}) ===");
    eprintln!("  global max|d| = {gmax:.5}");
    eprintln!("  global mean|d| = {gmean:.6}");
    eprintln!("  non-streaming mel rms = {ns_rms:.5}  (relative mean|d| = {:.4}%)", 100.0 * gmean / ns_rms);

    // boundary-frame diffs: the frame just after each chunk boundary (a causal-leakage tell).
    eprintln!("\n=== boundary-frame max|d| (frame right after each chunk end) ===");
    for (i, &bnd) in boundaries.iter().enumerate() {
        if bnd < cmp {
            let bmax = (0..MEL_NUM_MELS)
                .map(|r| (stream_frames[r][bnd] - ns[r * ns_frames + bnd]).abs())
                .fold(0f32, f32::max);
            eprintln!("  after chunk {i:2} (frame {bnd:4}): max|d|={bmax:.4}");
        }
    }
}
