//! End-to-end load check against the REAL published checkpoints.
//!
//! Everything the agents verified was against rebuilt fixtures or the reference's own
//! Python classes. This is the first thing that drives the crate's own loader over the
//! shipped weights — the step where the sibling Fish port's bugs surfaced.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let talker_dir = args.next().expect("usage: loadcheck <talker-dir> <tokenizer-dir>");
    let tok_dir = args.next().expect("usage: loadcheck <talker-dir> <tokenizer-dir>");
    let dev = candle_core::Device::Cpu;

    let t0 = std::time::Instant::now();
    let m = syrinx_qwen::model::Qwen3Tts::load(&talker_dir, dev.clone())?;
    println!("  talker+predictor loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!("  variant: {:?}", m.config().variant);

    let t1 = std::time::Instant::now();
    let w = syrinx_qwen::load::load_tensors(
        &tok_dir,
        &dev,
        candle_core::DType::F32,
    )?;
    println!("  codec bag loaded in {:.1}s ({} tensors)", t1.elapsed().as_secs_f32(), w.len());
    Ok(())
}
