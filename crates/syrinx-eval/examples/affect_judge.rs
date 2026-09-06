//! Score renders with the affect judge and report what it heard.
//!
//! Usage: affect_judge <model.onnx> <wav>...
//!
//! Prints, for every clip, the full 8-class probability vector — never just the argmax,
//! because a cue that lifts the right class from 0.20 to 0.40 without overtaking the leader
//! is a real effect the argmax throws away.
//!
//! Then two things that keep the numbers honest:
//!
//! * **Pairs.** Clips named `<n>-<cue>-plain` / `<n>-<cue>-tagged` are matched up and the
//!   per-class delta printed, with the cued class called out. Reported, never asserted:
//!   `CLAUDE.md` puts "intended emotion" in blocked-on-human territory and this judge is
//!   trained on acted human speech, so a null means "cannot tell", not "no effect".
//! * **A within-render spread.** Each clip is also scored over three overlapping 60 %
//!   windows. The spread across those windows is a floor on how much this measurement
//!   moves for reasons that have nothing to do with the cue — content, length, where the
//!   pauses fall. A pair delta smaller than that floor is not evidence of anything. It is
//!   NOT the sampler's run-to-run noise, which needs several seeds per condition and is
//!   the stronger control; it is the cheap lower bound available from one render.
//!
//! See `syrinx_eval::affect` for what the judge is and what it cannot tell you.
use syrinx_eval::affect::{direction, ravdess8_spec, AffectJudge, OnnxJudge, RAVDESS8_LABELS};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: affect_judge <model.onnx> <wav>...");
        std::process::exit(2);
    }
    let mut judge = OnnxJudge::load(&args[0], ravdess8_spec())?;
    eprintln!("[affect] judge: {}", judge.name());

    let header: String =
        RAVDESS8_LABELS.iter().map(|l| format!("{:>9}", &l[..l.len().min(5)])).collect();
    println!("{:<24}{header}{:>10}", "clip", "spread");

    let mut scored = Vec::new();
    for path in &args[1..] {
        let (mono, sr) = read_wav_mono(path)?;
        let at16 = if sr == judge.sample_rate() {
            mono
        } else {
            syrinx_qwen::speaker::resample(&mono, sr, judge.sample_rate())
        };
        let score = judge.score(&at16)?;

        // Three overlapping 60 % windows: the widest per-class range across them is the
        // within-render floor described in the module docs.
        let mut lo = vec![f32::INFINITY; RAVDESS8_LABELS.len()];
        let mut hi = vec![f32::NEG_INFINITY; RAVDESS8_LABELS.len()];
        let len = (at16.len() as f64 * 0.6) as usize;
        for k in 0..3 {
            let start = (at16.len().saturating_sub(len)) * k / 2;
            let w = judge.score(&at16[start..start + len])?;
            for (i, v) in w.values.iter().enumerate() {
                lo[i] = lo[i].min(*v);
                hi[i] = hi[i].max(*v);
            }
        }
        let spread: Vec<f32> = lo.iter().zip(&hi).map(|(a, b)| b - a).collect();
        let worst_spread = spread.iter().copied().fold(0.0f32, f32::max);

        let stem = std::path::Path::new(path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.clone());
        let row: String = score.values.iter().map(|v| format!("{v:>9.3}")).collect();
        println!("{stem:<24}{row}{worst_spread:>10.3}");
        scored.push((stem, score, spread));
    }

    // ---- pairs -------------------------------------------------------------------
    let mut any = false;
    for (stem, plain, plain_spread) in &scored {
        let Some(base) = stem.strip_suffix("-plain") else { continue };
        let tagged_stem = format!("{base}-tagged");
        let Some((_, tagged, tagged_spread)) = scored.iter().find(|(s, _, _)| *s == tagged_stem)
        else {
            continue;
        };
        if !any {
            println!("\npairs (delta = tagged - plain), REPORTED, NOT ASSERTED");
            any = true;
        }
        // `<n>-<cue>-plain` -> the cue id.
        let cue = base.split('-').nth(1).unwrap_or(base);
        let rep = direction(cue, plain, tagged);
        let deltas: String =
            rep.deltas.iter().map(|d| format!("{:>9.3}", d.delta())).collect();
        // Compare the cued class's movement against THAT class's own window spread, not
        // against the worst class's: a floor borrowed from an unrelated label would
        // dismiss a real effect.
        let ci = rep
            .cued_label
            .as_deref()
            .and_then(|l| RAVDESS8_LABELS.iter().position(|r| *r == l));
        let floor = ci.map(|i| plain_spread[i].max(tagged_spread[i])).unwrap_or(f32::NAN);
        println!("{:<24}{deltas}", format!("[{cue}]"));
        let (gain_label, gain) = rep.largest_gain();
        match rep.cued_delta() {
            Some(d) => {
                let verdict = if d.abs() <= floor {
                    format!("inside this class's {floor:.3} within-render floor — cannot tell")
                } else if d > 0.0 {
                    format!("TOWARD the cued class, above its {floor:.3} floor")
                } else {
                    format!("AWAY from the cued class, beyond its {floor:.3} floor")
                };
                println!(
                    "{:<24}cued `{}` {d:+.3} ({verdict}); largest gain `{gain_label}` {gain:+.3}",
                    "",
                    rep.cued_label.as_deref().unwrap_or("?")
                );
            }
            None => println!(
                "{:<24}cue `{cue}` has no counterpart among the judge's 8 classes",
                ""
            ),
        }
    }
    Ok(())
}

/// Mono f32 from a WAV, mirroring the local reader in `examples/simo.rs`.
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
