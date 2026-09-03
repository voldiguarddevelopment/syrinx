//! **Voice-clone driver for the `-Base` checkpoints** — the piece that was missing.
//!
//! `Qwen3-TTS-12Hz-{0.6B,1.7B}-Base` are clone-only: no instruct, no preset speakers, and
//! `talker_config.spk_id` is empty. The one thing they condition on is an x-vector
//! extracted from a reference clip, and until this file existed nothing in-tree turned a
//! WAV into one — `realize_plan` took the speaker vector and the reference frames as
//! arguments and no caller built them. So `-Base` had never synthesized anything, which is
//! why `renders/2026-09-03-qwen-first/FINDINGS.md` could only check its *capability* rows
//! (every cue dropped) and not its audio.
//!
//! The chain, mirroring `Qwen3TTSModel.create_voice_clone_prompt` + `generate_voice_clone`:
//!
//! ```text
//!   ref.wav --read--> mono f32 --resample--> 24 kHz
//!           --SpeakerEncoder::embed--> x-vector [1, enc_dim]
//!           (--MimiEncoder::encode--> reference RVQ frames, ICL mode only)
//!   build_voice_clone -> realize_plan -> generate -> split RVQ -> decoder -> out.wav
//! ```
//!
//! Two modes, exactly the reference's two:
//!
//! * **x-vector only** (default, `x_vector_only_mode=True`): the clip enters solely as the
//!   speaker vector in the codec stream. Cheap, and the only mode that needs no transcript.
//! * **in-context** (`--ref-text "<transcript>"`, the reference's *default*, `icl_mode`):
//!   the reference transcript and its RVQ frames are spliced into the prompt ahead of the
//!   target text. The reference then decodes `cat(ref_code, generated)` and drops the
//!   leading `ref_frames / total_frames` fraction of the waveform; this does the same.
//!
//! Usage:
//!   clone <base-dir> <tokenizer-dir> <ref.wav> <out.wav> "<text>" [language]
//!         [--ref-text "<transcript of ref.wav>"] [--seed N]
//!
//! CPU by default; `SYRINX_QWEN_DEVICE=<N>` selects a CUDA ordinal, as in `synth.rs`.
//! Note the x-vector itself is always computed in f32 (`SpeakerEncoder` casts on load) —
//! it is one pass over a few hundred frames and the pooling is a whole-utterance sum,
//! which is the reduction bf16 handles worst.
use candle_core::{DType, Device, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut a: Vec<String> = Vec::new();
    let mut ref_text: Option<String> = None;
    let mut seed: u64 = 0;
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--ref-text" => { ref_text = Some(argv[i + 1].clone()); i += 2; }
            "--seed" => { seed = argv[i + 1].parse()?; i += 2; }
            _ => { a.push(argv[i].clone()); i += 1; }
        }
    }
    if a.len() < 5 {
        return Err("usage: clone <base-dir> <tokenizer-dir> <ref.wav> <out.wav> \"<text>\" \
                    [language] [--ref-text \"<transcript>\"] [--seed N]"
            .into());
    }
    let (base_dir, tok_dir, ref_wav, out, text) = (&a[0], &a[1], &a[2], &a[3], &a[4]);
    let language = a.get(5).map(|s| s.as_str()).unwrap_or("english");

    let dev = match std::env::var("SYRINX_QWEN_DEVICE").ok().and_then(|v| v.trim().parse::<usize>().ok()) {
        Some(i) => Device::new_cuda(i)?,
        None => Device::Cpu,
    };
    let dt = if dev.is_cuda() { DType::BF16 } else { DType::F32 };
    eprintln!("device: {:?}  dtype: {dt:?}", dev.location());

    let cfg_json = std::fs::read_to_string(std::path::Path::new(base_dir).join("config.json"))?;
    let cfg = syrinx_qwen::Qwen3TtsConfig::from_json(&cfg_json)?;
    if !cfg.variant.supports_voice_clone() {
        return Err(format!(
            "{base_dir} is a {:?} checkpoint — the clone path needs a -Base one (it is the \
             only variant carrying speaker_encoder.*)",
            cfg.variant
        )
        .into());
    }
    let pcfg = syrinx_qwen::prompt::PromptConfig::from_json(&cfg_json)?;
    let tok = syrinx_qwen::tokenizer::QwenTokenizer::from_dir(base_dir)?;

    // ---- reference clip -> 24 kHz mono f32 -------------------------------------------
    let (samples, sr) = read_wav_mono(ref_wav)?;
    let wav24 = syrinx_qwen::speaker::resample(&samples, sr, cfg.sample_rate);
    eprintln!(
        "reference: {ref_wav} — {} samples @ {sr} Hz -> {} @ {} Hz ({:.2} s)",
        samples.len(),
        wav24.len(),
        cfg.sample_rate,
        wav24.len() as f32 / cfg.sample_rate as f32
    );

    // ---- x-vector --------------------------------------------------------------------
    // Only the 76 `speaker_encoder.*` tensors, mmapped: the checkpoint is 3.6 GB and the
    // talker is loaded separately below, so materialising the whole bag twice is pure waste.
    let t = std::time::Instant::now();
    let spk_cfg = syrinx_qwen::speaker::SpeakerEncoderConfig::from_model_config(&cfg);
    let enc_dim = spk_cfg.enc_dim;
    let st = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(
            std::path::Path::new(base_dir).join("model.safetensors"),
        )?
    };
    let mut spk_map = std::collections::HashMap::new();
    for (k, _) in st.tensors() {
        if k.starts_with("speaker_encoder.") {
            spk_map.insert(k.clone(), st.load(&k, &dev)?);
        }
    }
    let spk_w = syrinx_qwen::nn::Weights { map: spk_map, dev: dev.clone(), dt: DType::F32 };
    let spk_enc = syrinx_qwen::speaker::SpeakerEncoder::load(&spk_w, "speaker_encoder", spk_cfg)?;
    let xvector = spk_enc.embed(&wav24, cfg.sample_rate)?;
    drop(spk_w);
    drop(st);
    let norm: f32 = xvector
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    eprintln!(
        "x-vector: {enc_dim} dims, L2 norm {norm:.4}, in {:.1}s",
        t.elapsed().as_secs_f32()
    );
    // The reference `view(1, 1, -1)`s the vector straight into the codec stream, so a
    // checkpoint whose enc_dim differed from the talker width would be a shape error deep
    // inside `realize_plan`. Say so here instead.
    if enc_dim != cfg.talker.hidden_size {
        return Err(format!(
            "speaker_enc_dim {enc_dim} != talker hidden size {} — the x-vector cannot occupy \
             a codec-stream position on this checkpoint",
            cfg.talker.hidden_size
        )
        .into());
    }

    // ---- reference RVQ frames (in-context mode only) ----------------------------------
    let mut ref_frames: Vec<Vec<u32>> = Vec::new();
    if ref_text.is_some() {
        let t = std::time::Instant::now();
        let tk_json = std::fs::read_to_string(std::path::Path::new(tok_dir).join("config.json"))?;
        let ecfg = syrinx_qwen::codec::encoder::MimiEncoderConfig::from_json(&tk_json)?;
        let emap = syrinx_qwen::load::load_tensors(tok_dir, &dev, dt)?;
        let enc = syrinx_qwen::codec::encoder::MimiEncoder::new(
            syrinx_qwen::nn::Weights { map: emap, dev: dev.clone(), dt },
            ecfg,
        )?;
        let wav_t = Tensor::from_vec(wav24.clone(), wav24.len(), &dev)?;
        // `[n_q][frames]` out of the encoder; the prompt wants one row per FRAME.
        let rows = enc.encode(&wav_t)?;
        let frames = rows.first().map(|r| r.len()).unwrap_or(0);
        if rows.len() != cfg.num_code_groups {
            return Err(format!(
                "encoder produced {} quantizer rows, the talker consumes {}",
                rows.len(),
                cfg.num_code_groups
            )
            .into());
        }
        ref_frames = (0..frames)
            .map(|t| rows.iter().map(|r| r[t]).collect())
            .collect();
        eprintln!(
            "reference codes: {frames} frames x {} groups in {:.1}s",
            rows.len(),
            t.elapsed().as_secs_f32()
        );
    }

    // ---- prompt ------------------------------------------------------------------------
    let reference = match ref_text.as_deref() {
        None => syrinx_qwen::prompt::CloneRef::XVectorOnly,
        Some(rt) => syrinx_qwen::prompt::CloneRef::InContext { ref_text: rt, frames: ref_frames.len() },
    };
    let plan = syrinx_qwen::prompt::build_voice_clone(
        &tok,
        &pcfg,
        text,
        reference,
        language,
        syrinx_qwen::prompt::VOICE_CLONE_NON_STREAMING,
    )?;
    eprintln!(
        "prompt: {} steps, {} trailing text, {} reference frames ({})",
        plan.len(),
        plan.trailing_text.len(),
        plan.ref_frames,
        if ref_text.is_some() { "in-context" } else { "x-vector only" }
    );

    // ---- generate -----------------------------------------------------------------------
    let t = std::time::Instant::now();
    let mut m = syrinx_qwen::model::Qwen3Tts::load(base_dir, dev.clone())?;
    eprintln!("talker loaded {:.1}s", t.elapsed().as_secs_f32());

    let prompt = m.realize_plan(&plan, Some(&xvector), &ref_frames)?;
    let params = syrinx_qwen::model::DriveParams { seed, ..Default::default() };
    let t = std::time::Instant::now();
    let gen = m.generate(&prompt, &params)?;
    eprintln!(
        "generated {} frames in {:.1}s (stopped on EOS: {})",
        gen.frames.len(),
        t.elapsed().as_secs_f32(),
        gen.stopped_on_eos
    );
    if gen.frames.is_empty() {
        return Err("no frames generated".into());
    }

    // ---- codec: RVQ -> decoder ------------------------------------------------------------
    // In-context mode decodes `cat(ref_code, generated)` and cuts the reference's share of
    // the waveform back off, exactly as `generate_voice_clone` does — the decoder is
    // convolutional, so decoding the two halves separately is not the same thing.
    let mut all: Vec<Vec<u32>> = ref_frames.clone();
    all.extend(gen.frames.iter().cloned());
    let total = all.len();
    let n_groups = m.config().num_code_groups;
    let rows: Vec<Vec<u32>> = (0..n_groups)
        .map(|g| all.iter().map(|f| f[g]).collect())
        .collect();

    let cw = syrinx_qwen::load::load_tensors(tok_dir, &dev, dt)?;
    let w = syrinx_qwen::nn::Weights { map: cw, dev: dev.clone(), dt };
    let sem = syrinx_qwen::codec::rvq::Rvq::load(&w, "decoder.quantizer.rvq_first", 1)?;
    let ac = syrinx_qwen::codec::rvq::Rvq::load(&w, "decoder.quantizer.rvq_rest", rows.len() - 1)?;
    let z = (sem.decode(&w, &rows[..1], dt)? + ac.decode(&w, &rows[1..], dt)?)?;

    let tk_json = std::fs::read_to_string(std::path::Path::new(tok_dir).join("config.json"))?;
    let dcfg = syrinx_qwen::codec::decoder::DecoderConfig::from_json(&tk_json)?;
    let dec = syrinx_qwen::codec::decoder::Decoder::new("decoder", dcfg);
    let t = std::time::Instant::now();
    let wav_t = dec.decode(&w, &z)?;
    let mut wav: Vec<f32> = wav_t.flatten_all()?.to_vec1()?;
    eprintln!("decoded {} samples in {:.1}s", wav.len(), t.elapsed().as_secs_f32());

    if !ref_frames.is_empty() {
        let cut = (ref_frames.len() as f64 / total.max(1) as f64 * wav.len() as f64) as usize;
        eprintln!("trimming {cut} samples of reference lead-in ({}/{total} frames)", ref_frames.len());
        wav.drain(..cut.min(wav.len()));
    }

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: syrinx_qwen::SAMPLE_RATE_24K,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut wr = hound::WavWriter::create(out, spec)?;
    for s in &wav {
        wr.write_sample((s.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
    }
    wr.finalize()?;
    eprintln!(
        "wrote {out} ({:.2}s @ {} kHz)",
        wav.len() as f32 / syrinx_qwen::SAMPLE_RATE_24K as f32,
        syrinx_qwen::SAMPLE_RATE_24K / 1000
    );
    Ok(())
}

/// Read a WAV to mono `f32` in `[-1, 1]`, returning `(samples, sample_rate)`.
///
/// Channels are averaged, which is what the reference's `_load_audio_to_np` does
/// (`librosa.load(mono=True)`, then `np.mean(axis=-1)` for anything still 2-D).
fn read_wav_mono(path: &str) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let mut r = hound::WavReader::open(path)?;
    let spec = r.spec();
    let ch = spec.channels as usize;
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    };
    if ch <= 1 {
        return Ok((interleaved, spec.sample_rate));
    }
    let mono = interleaved
        .chunks(ch)
        .map(|f| f.iter().sum::<f32>() / ch as f32)
        .collect();
    Ok((mono, spec.sample_rate))
}
