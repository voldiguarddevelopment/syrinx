//! First end-to-end synthesis through the pure-Rust port: text -> talker -> code
//! predictor -> RVQ -> codec decoder -> 24 kHz WAV.
//!
//! Usage: synth <talker-dir> <tokenizer-dir> <out.wav> "<text>" [speaker] [language]
use candle_core::{DType, Device};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (talker_dir, tok_dir, out, text) = (&a[0], &a[1], &a[2], &a[3]);
    let speaker = a.get(4).map(|s| s.as_str()).unwrap_or("serena");
    let language = a.get(5).map(|s| s.as_str()).unwrap_or("english");
    // SYRINX_QWEN_DEVICE=<N> selects a CUDA ordinal; unset = CPU. The model picks its
    // compute dtype from the device (f32 on CPU for parity, bf16 on CUDA to fit), and
    // the codec bag follows it — a hardcoded f32 codec is what OOMed the sibling Fish
    // port on a long reference.
    let dev = match std::env::var("SYRINX_QWEN_DEVICE").ok().and_then(|v| v.trim().parse::<usize>().ok()) {
        Some(i) => Device::new_cuda(i)?,
        None => Device::Cpu,
    };
    let dt = if dev.is_cuda() { DType::BF16 } else { DType::F32 };
    eprintln!("device: {:?}  dtype: {dt:?}", dev.location());

    let cfg_json = std::fs::read_to_string(std::path::Path::new(talker_dir).join("config.json"))?;
    let pcfg = syrinx_qwen::prompt::PromptConfig::from_json(&cfg_json)?;
    let tok = syrinx_qwen::tokenizer::QwenTokenizer::from_dir(talker_dir)?;

    let t = std::time::Instant::now();
    let mut m = syrinx_qwen::model::Qwen3Tts::load(talker_dir, dev.clone())?;
    eprintln!("talker loaded {:.1}s", t.elapsed().as_secs_f32());

    let plan = syrinx_qwen::prompt::build_custom_voice(
        &tok, &pcfg, text, speaker, None, language, true,
    )?;
    eprintln!("prompt: {} steps", plan.len());

    let prompt = m.realize_plan(&plan, None, &[])?;
    let params = syrinx_qwen::model::DriveParams::default();
    let t = std::time::Instant::now();
    let gen = m.generate(&prompt, &params)?;
    eprintln!("generated {} frames in {:.1}s", gen.frames.len(), t.elapsed().as_secs_f32());
    if gen.frames.is_empty() {
        return Err("no frames generated".into());
    }

    // codec: RVQ -> decoder
    let cw = syrinx_qwen::load::load_tensors(tok_dir, &dev, dt)?;
    let w = syrinx_qwen::nn::Weights { map: cw, dev: dev.clone(), dt };
    let rows = gen.group_rows(m.config().num_code_groups);
    // Split RVQ: group 0 through the semantic stack, groups 1.. through the acoustic
    // one. Decode is `rvq_first(...) + rvq_rest(...)` — the two stacks are PARALLEL,
    // not serial (the acoustic stack does not start from the semantic residual).
    let sem = syrinx_qwen::codec::rvq::Rvq::load(&w, "decoder.quantizer.rvq_first", 1)?;
    let ac = syrinx_qwen::codec::rvq::Rvq::load(&w, "decoder.quantizer.rvq_rest", rows.len() - 1)?;
    let z_sem = sem.decode(&w, &rows[..1], dt)?;
    let z_ac = ac.decode(&w, &rows[1..], dt)?;
    let z = (z_sem + z_ac)?;

    let tk_json = std::fs::read_to_string(std::path::Path::new(tok_dir).join("config.json"))?;
    let dcfg = syrinx_qwen::codec::decoder::DecoderConfig::from_json(&tk_json)?;
    let dec = syrinx_qwen::codec::decoder::Decoder::new("decoder", dcfg);
    let t = std::time::Instant::now();
    let wav_t = dec.decode(&w, &z)?;
    let wav: Vec<f32> = wav_t.flatten_all()?.to_vec1()?;
    eprintln!("decoded {} samples in {:.1}s", wav.len(), t.elapsed().as_secs_f32());

    let spec = hound::WavSpec { channels: 1, sample_rate: 24_000, bits_per_sample: 16,
                                sample_format: hound::SampleFormat::Int };
    let mut wr = hound::WavWriter::create(out, spec)?;
    for s in &wav { wr.write_sample((s.clamp(-1.0, 1.0) * 32767.0).round() as i16)?; }
    wr.finalize()?;
    eprintln!("wrote {out} ({:.2}s @ 24 kHz)", wav.len() as f32 / 24_000.0);
    Ok(())
}
