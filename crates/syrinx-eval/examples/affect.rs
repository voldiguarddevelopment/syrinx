//! Acoustic correlates of AROUSAL for a set of renders — a zero-dependency first look.
//!
//! Not an emotion model. Arousal correlates well with pitch height/variability, loudness
//! and speaking rate; valence does not correlate with anything this simple, which is
//! precisely why a real SER model would be needed for the other axis.
//! Usage: affect <wav>...
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{:<26} {:>8} {:>8} {:>8} {:>8} {:>8}", "render", "f0", "f0sd", "rms", "rate", "tilt");
    for p in std::env::args().skip(1) {
        let (mono, sr) = read_wav_mono(&p)?;
        let f = syrinx_eval::acoustic::features(&mono, sr);
        let g = |n: &str| f.get(n).unwrap_or(f64::NAN);
        // "rate" here is voiced frames per second — a crude tempo/activity proxy.
        let rate = g("voiced_ratio") / (g("duration_s") / g("duration_s").max(1e-9)).max(1e-9);
        println!(
            "{:<26} {:>8.1} {:>8.1} {:>8.4} {:>8.3} {:>8.3}",
            std::path::Path::new(&p).file_stem().unwrap().to_string_lossy(),
            g("f0_mean"), g("f0_std"), g("rms_mean"), rate, g("spectral_tilt")
        );
    }
    Ok(())
}

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
    let mono = if ch == 1 { raw } else { raw.chunks(ch).map(|c| c.iter().sum::<f32>() / ch as f32).collect() };
    Ok((mono, spec.sample_rate))
}
