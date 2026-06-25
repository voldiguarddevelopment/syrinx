//! Real CosyVoice2 LM-forward **hammer** test — stress the Candle port across many
//! varied references (text / speech-embedding / mixed, lengths 1..2048) and assert
//! every case matches within 1e-3 with full per-position argmax agreement.
//!
//! Runs **parallel across all CPU cores** (work-stealing over cases). Pin Candle to
//! inline single-threaded matmuls so N workers peg N cores without oversubscription:
//! set `RAYON_NUM_THREADS=1`. An optional pass count sustains the load.
//!
//! Gated on the `real` feature + on-disk fixtures (generated on the GPU box).
//!
//!   RAYON_NUM_THREADS=1 \
//!   SYRINX_LM_WEIGHTS=/root/models/CosyVoice2-0.5B/llm_fp32.safetensors \
//!   SYRINX_LM_HAMMER=/root/parity/hammer \
//!   SYRINX_LM_HAMMER_PASSES=3 \
//!   cargo test -p syrinx-lm --features real --release real_lm_forward_hammer -- --nocapture

#![cfg(feature = "real")]

use candle_core::{DType, Device, Tensor, D};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use syrinx_lm::real::Qwen2Lm;

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    (a - b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap()
}

fn argmax_all(t: &Tensor) -> Vec<u32> {
    t.squeeze(0).unwrap().argmax(D::Minus1).unwrap().to_vec1::<u32>().unwrap()
}

#[test]
fn real_lm_forward_hammer() {
    let (weights, dir) = match (
        std::env::var("SYRINX_LM_WEIGHTS").ok(),
        std::env::var("SYRINX_LM_HAMMER").ok(),
    ) {
        (Some(w), Some(d)) if Path::new(&w).exists() && Path::new(&d).is_dir() => (w, d),
        _ => {
            eprintln!("SKIP hammer: set SYRINX_LM_WEIGHTS + SYRINX_LM_HAMMER (dir of case_*.safetensors)");
            return;
        }
    };
    let passes: usize = std::env::var("SYRINX_LM_HAMMER_PASSES").ok().and_then(|s| s.parse().ok()).unwrap_or(1);

    let lm = Arc::new(Qwen2Lm::load(&weights, Device::Cpu).expect("load fp32 weights"));

    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read hammer dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    // Largest-first so the long T=2048 forwards launch immediately and spread across
    // cores (work-stealing backfills with small cases), keeping every core busy to the
    // tail instead of stranding a slow huge case at the end.
    files.sort_by_key(|p| std::cmp::Reverse(std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)));
    assert!(!files.is_empty(), "no hammer cases found in {dir}");
    let cases = Arc::new(files);

    let n_workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let total_work = cases.len() * passes;
    let next = Arc::new(AtomicUsize::new(0));
    let positions = Arc::new(AtomicUsize::new(0));
    let mismatches = Arc::new(AtomicUsize::new(0));
    let worst = Arc::new(Mutex::new((0f32, String::new())));
    let failures: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    eprintln!(
        "HAMMER start: {} cases x {} passes = {} forwards across {} workers (RAYON_NUM_THREADS={})",
        cases.len(), passes, total_work, n_workers,
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into())
    );
    let t0 = Instant::now();

    let handles: Vec<_> = (0..n_workers)
        .map(|_| {
            let (lm, cases, next, positions, mismatches, worst, failures) = (
                lm.clone(), cases.clone(), next.clone(),
                positions.clone(), mismatches.clone(), worst.clone(), failures.clone(),
            );
            std::thread::spawn(move || {
                let dev = Device::Cpu;
                loop {
                    let w = next.fetch_add(1, Ordering::Relaxed);
                    if w >= total_work {
                        break;
                    }
                    let path = &cases[w % cases.len()];
                    let name = path.file_name().unwrap().to_string_lossy().into_owned();
                    let t = candle_core::safetensors::load(path, &dev).expect("load case");
                    let embeds = t.get("input_embeds").unwrap().to_dtype(DType::F32).unwrap();
                    let expected = t.get("logits").unwrap().to_dtype(DType::F32).unwrap();
                    let logits = lm.forward_logits(&embeds).expect("forward");
                    if logits.dims() != expected.dims() {
                        failures.lock().unwrap().push(format!("{name}: shape mismatch"));
                        continue;
                    }
                    let d = max_abs_diff(&logits, &expected);
                    let (ours, refs) = (argmax_all(&logits), argmax_all(&expected));
                    positions.fetch_add(ours.len(), Ordering::Relaxed);
                    let mm = ours.iter().zip(&refs).filter(|(a, b)| a != b).count();
                    mismatches.fetch_add(mm, Ordering::Relaxed);
                    {
                        let mut wr = worst.lock().unwrap();
                        if d > wr.0 {
                            *wr = (d, name.clone());
                        }
                    }
                    if d >= 1e-3 {
                        failures.lock().unwrap().push(format!("{name}: max-abs-diff {d:.3e} >= 1e-3"));
                    }
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("worker panicked");
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let (worst_d, worst_name) = worst.lock().unwrap().clone();
    let positions = positions.load(Ordering::Relaxed);
    let mismatches = mismatches.load(Ordering::Relaxed);
    let failures = failures.lock().unwrap();
    eprintln!(
        "HAMMER done: {} forwards in {:.1}s ({:.1}/s) | worst max-abs-diff = {:.3e} ({}) | argmax {}/{} match | {} failures",
        total_work, elapsed, total_work as f64 / elapsed, worst_d, worst_name,
        positions - mismatches, positions, failures.len()
    );

    assert!(failures.is_empty(), "tolerance failures:\n  {}", failures.join("\n  "));
    assert!(worst_d < 1e-3, "worst-case logit diff {worst_d:.3e} exceeds 1e-3");
    assert_eq!(mismatches, 0, "{mismatches} argmax mismatches across {positions} positions");
}
