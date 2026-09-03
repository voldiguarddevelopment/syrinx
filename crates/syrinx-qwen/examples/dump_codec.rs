//! Decode a fixture's code grid and write the waveform, for numerical comparison
//! against the reference dump. Usage: dump_codec <tokenizer-dir> <ref.safetensors> <out.safetensors>
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candle_core::{DType, Device};
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (tok_dir, refp, out) = (&a[0], &a[1], &a[2]);
    let dev = match std::env::var("SYRINX_QWEN_DEVICE").ok().and_then(|v| v.trim().parse::<usize>().ok()) {
        Some(i) => Device::new_cuda(i)?,
        None => Device::Cpu,
    };
    let dt = if dev.is_cuda() { DType::BF16 } else { DType::F32 };
    let refs = candle_core::safetensors::load(refp, &Device::Cpu)?;
    let cw = syrinx_qwen::load::load_tensors(tok_dir, &dev, dt)?;
    let w = syrinx_qwen::nn::Weights { map: cw, dev: dev.clone(), dt };
    let tk = std::fs::read_to_string(std::path::Path::new(tok_dir).join("config.json"))?;
    let dcfg = syrinx_qwen::codec::decoder::DecoderConfig::from_json(&tk)?;
    let dec = syrinx_qwen::codec::decoder::Decoder::new("decoder", dcfg);
    let mut outmap = std::collections::HashMap::new();
    for tag in ["codec", "codec_edge"] {
        let codes = refs.get(&format!("{tag}.codes")).unwrap();
        let (frames, groups) = (codes.dim(0)?, codes.dim(1)?);
        let host: Vec<i64> = codes.flatten_all()?.to_vec1()?;
        let rows: Vec<Vec<u32>> = (0..groups)
            .map(|g| (0..frames).map(|t| host[t * groups + g] as u32).collect())
            .collect();
        let sem = syrinx_qwen::codec::rvq::Rvq::load(&w, "decoder.quantizer.rvq_first", 1)?;
        let ac = syrinx_qwen::codec::rvq::Rvq::load(&w, "decoder.quantizer.rvq_rest", rows.len() - 1)?;
        let z = (sem.decode(&w, &rows[..1], dt)? + ac.decode(&w, &rows[1..], dt)?)?;
        let wav = dec.decode(&w, &z)?.flatten_all()?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        eprintln!("{tag}: {} samples", wav.elem_count());
        outmap.insert(format!("{tag}.wav"), wav);
    }
    candle_core::safetensors::save(&outmap, out)?;
    Ok(())
}
