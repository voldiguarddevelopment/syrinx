fn main() {
    for d in std::env::args().skip(1) {
        let j = match std::fs::read_to_string(std::path::Path::new(&d).join("config.json")) {
            Ok(j) => j,
            Err(_) => continue,
        };
        let cfg = match syrinx_qwen::Qwen3TtsConfig::from_json(&j) {
            Ok(c) => c,
            Err(e) => { println!("  ERR {d}: {e}"); continue; }
        };
        let name = std::path::Path::new(&d).file_name().unwrap().to_string_lossy().to_string();
        match syrinx_qwen::load::verify_checkpoint(&d, &cfg) {
            Ok(r) => {
                println!("  {:<34} checked {:3}  missing {}  mismatched {}  unaccounted {}  => {}",
                    name, r.checked, r.missing.len(), r.mismatched.len(), r.unaccounted.len(),
                    if r.ok() { "OK" } else { "FAIL" });
                for m in r.missing.iter().take(3) { println!("      MISSING {m}"); }
                for (n,w,g) in r.mismatched.iter().take(3) { println!("      SHAPE {n}: want {w:?} got {g:?}"); }
                let mut pre: std::collections::BTreeSet<&str> = Default::default();
                for u in &r.unaccounted { pre.insert(u.split('.').next().unwrap_or(u)); }
                if !pre.is_empty() { println!("      unaccounted prefixes: {pre:?}"); }
                for u in r.unaccounted.iter().filter(|u| !u.starts_with("speaker_encoder")) {
                    println!("      unaccounted: {u}");
                }
            }
            Err(e) => println!("  ERR {name}: {e}"),
        }
    }
}
