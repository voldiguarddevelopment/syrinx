//! Text-normalization parity — the Rust `syrinx-frontend::textnorm::normalize_text`
//! vs. CosyVoice2's wetext-based `frontend.text_normalize(s, split=False)`.
//!
//! This is a **text-level** (fast — no synthesis) parity check. The reference is a
//! `{ input -> normalized }` JSON map produced on the model box by `gen_ref.py`,
//! which replicates `text_normalize` exactly using the real wetext `Normalizer` +
//! the CosyVoice `frontend_utils` helpers (the wetext frontend the model actually
//! runs). The test runs the Rust normalizer on each input and reports the exact
//! match rate, listing every miss (input / expected / got) so coverage is honest.
//!
//! Gated on the `tn` feature (lightweight — pure Rust, no model deps) AND the env
//! var pointing at the on-box reference; it skips cleanly when the file is absent:
//!
//!   SYRINX_TEXTNORM_REF=/root/parity/textnorm/ref.json \
//!   cargo test --features tn --test real_textnorm_parity -- --nocapture

#![cfg(feature = "tn")]

use std::path::Path;

use syrinx_frontend::textnorm::normalize_text;

/// Minimum exact-match rate the Rust normalizer must hold against the wetext
/// reference on the common-case corpus. Honestly scoped: full WFST parity is not
/// claimed (see the module docs in `textnorm.rs`), but the common cases must match.
const MIN_MATCH_RATE: f64 = 0.80;

#[test]
fn rust_normalizer_matches_wetext_reference() {
    let ref_path = match std::env::var("SYRINX_TEXTNORM_REF").ok() {
        Some(p) if Path::new(&p).exists() => p,
        _ => {
            eprintln!(
                "skipping real_textnorm_parity: set SYRINX_TEXTNORM_REF to the \
                 on-box ref.json ({{input -> normalized}}) to run it"
            );
            return;
        }
    };

    let raw = std::fs::read_to_string(&ref_path).expect("read ref.json");
    let map: serde_json::Value = serde_json::from_str(&raw).expect("parse ref.json");
    let obj = map.as_object().expect("ref.json is a JSON object");

    let mut total = 0usize;
    let mut hits = 0usize;
    let mut misses: Vec<(String, String, String)> = Vec::new();

    for (input, expected_v) in obj {
        let expected = expected_v.as_str().expect("ref value is a string");
        let got = normalize_text(input);
        total += 1;
        if got == expected {
            hits += 1;
        } else {
            misses.push((input.clone(), expected.to_string(), got));
        }
    }

    let rate = hits as f64 / total.max(1) as f64;
    println!("\n=== textnorm parity: {hits}/{total} exact ({:.1}%) ===", rate * 100.0);
    if misses.is_empty() {
        println!("(no misses)");
    } else {
        println!("misses ({}):", misses.len());
        for (input, expected, got) in &misses {
            println!("  input:    {input:?}");
            println!("    expect: {expected:?}");
            println!("    got:    {got:?}");
        }
    }

    assert!(
        rate >= MIN_MATCH_RATE,
        "textnorm match rate {:.1}% below floor {:.0}% ({hits}/{total})",
        rate * 100.0,
        MIN_MATCH_RATE * 100.0
    );
}
