//! SIM-o for Qwen clone renders: cosine in the anchored speaker space.
//!
//! Usage: simo <base-ckpt-dir> <reference.wav> <render.wav>...
//!
//! Loads ONLY the 76 `speaker_encoder.*` tensors (the checkpoint is gigabytes and nothing
//! else is needed), resamples both clips to the encoder's rate, and prints the cosine.
//! See `syrinx_eval::qwen::speaker_similarity` for what this does and does not prove.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candle_core::{DType, Device};
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (base_dir, reference) = (&a[0], &a[1]);
    let dev = Device::Cpu;

    let cfg_json = std::fs::read_to_string(std::path::Path::new(base_dir).join("config.json"))?;
    let cfg = syrinx_qwen::Qwen3TtsConfig::from_json(&cfg_json)?;
    let spk_cfg = syrinx_qwen::speaker::SpeakerEncoderConfig::from_model_config(&cfg);
    let rate = spk_cfg.sample_rate;

    let st = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(
            std::path::Path::new(base_dir).join("model.safetensors"),
        )?
    };
    let mut map = std::collections::HashMap::new();
    for (k, _) in st.tensors() {
        if k.starts_with("speaker_encoder.") {
            map.insert(k.clone(), st.load(&k, &dev)?);
        }
    }
    let w = syrinx_qwen::nn::Weights { map, dev: dev.clone(), dt: DType::F32 };
    let enc = syrinx_qwen::speaker::SpeakerEncoder::load(&w, "speaker_encoder", spk_cfg)?;
    eprintln!("[simo] encoder {}-wide @ {rate} Hz from {base_dir}", enc.config().enc_dim);

    let load24 = |p: &str| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let (mono, sr) = read_wav_mono(p)?;
        Ok(if sr == rate { mono } else { syrinx_qwen::speaker::resample(&mono, sr, rate) })
    };
    let refc = load24(reference)?;
    for r in &a[2..] {
        let render = load24(r)?;
        let cos = syrinx_eval::qwen::speaker_similarity(&enc, &refc, &render, rate)?;
        println!("  {cos:.6}  {r}");
    }
    Ok(())
}

/// Mono f32 from a WAV, mirroring `syrinx-qwen/examples/clone.rs`'s local reader.
fn read_wav_mono(path: &str) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let mut r = hound::WavReader::open(path)?;
    let spec = r.spec();
    let ch = spec.channels as usize;
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|v| v.map(|v| v as f32 / max)).collect::<Result<_, _>>()?
        }
    };
    let mono = if ch == 1 {
        raw
    } else {
        raw.chunks(ch).map(|c| c.iter().sum::<f32>() / ch as f32).collect()
    };
    Ok((mono, spec.sample_rate))
}
