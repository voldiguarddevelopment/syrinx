//! Real CosyVoice2 text-tokenizer parity — the `text -> token ids` front-half of
//! the pipeline, buildable on the model box. Mirrors the LM/speaker parity recipe.
//!
//! Asserts the Rust `syrinx-frontend::tokenizer::TextTokenizer` (the HF
//! `tokenizers` crate loading the model's serialized Qwen2 BPE `tokenizer.json`)
//! reproduces, EXACTLY, the token-id lists that `CosyVoice2Tokenizer.encode`
//! produces for a fixed corpus of already-normalized strings.
//!
//! Gated on the `real` feature AND env vars pointing at the on-box fixtures
//! (produced by `dump_tok.py`): the serialized tokenizer + the reference id dump.
//! Both live on the model box; the test skips cleanly when absent.
//!
//!   SYRINX_TOK_JSON=/root/parity/frontend/tokenizer.json \
//!   SYRINX_TOK_REF=/root/parity/frontend/tok_ref.json \
//!   cargo test --features real --test real_tokenizer_parity -- --nocapture

#![cfg(feature = "real")]

use std::path::Path;

use syrinx_frontend::tokenizer::TextTokenizer;

/// One reference case parsed out of `tok_ref.json` (kept to the root's existing
/// `serde_json` dev-dep — no `serde` derive — so the shared root Cargo.toml needs
/// no extra dependency).
struct RefCase {
    text: String,
    ids: Vec<u32>,
}

fn load_cases(path: &str) -> Vec<RefCase> {
    let raw = std::fs::read_to_string(path).expect("read tok_ref.json");
    let v: serde_json::Value = serde_json::from_str(&raw).expect("parse tok_ref.json");
    let cases = v["cases"].as_array().expect("tok_ref.json has a `cases` array");
    cases
        .iter()
        .map(|c| RefCase {
            text: c["text"].as_str().expect("case.text is a string").to_string(),
            ids: c["ids"]
                .as_array()
                .expect("case.ids is an array")
                .iter()
                .map(|x| x.as_u64().expect("id is a u64") as u32)
                .collect(),
        })
        .collect()
}

#[test]
fn real_tokenizer_reproduces_reference_ids_exactly() {
    let (tok_json, tok_ref) = match (
        std::env::var("SYRINX_TOK_JSON").ok(),
        std::env::var("SYRINX_TOK_REF").ok(),
    ) {
        (Some(j), Some(r)) if Path::new(&j).exists() && Path::new(&r).exists() => (j, r),
        _ => {
            eprintln!(
                "SKIP real_tokenizer parity: set SYRINX_TOK_JSON + SYRINX_TOK_REF to the \
                 on-disk fixtures (serialized tokenizer.json + Python id reference dump)"
            );
            return;
        }
    };

    let cases = load_cases(&tok_ref);
    assert!(!cases.is_empty(), "reference dump has no cases");

    let tok = TextTokenizer::from_file(&tok_json).expect("load tokenizer.json");

    let mut mismatches = 0usize;
    for case in &cases {
        let got = tok.encode(&case.text).expect("encode reference string");
        if got != case.ids {
            mismatches += 1;
            eprintln!(
                "MISMATCH for {:?}\n  expected = {:?}\n  got      = {:?}",
                case.text, case.ids, got
            );
        }
    }

    // C1: every fixed reference string round-trips to the identical id list — a
    // single off-by-one id, a missing/added BOS/EOS, or a mis-tokenized special
    // marker would fail this.
    assert_eq!(
        mismatches, 0,
        "{mismatches}/{} reference cases did not reproduce the exact CosyVoice2 token ids",
        cases.len()
    );

    // C2: the special-marker cases must encode the CosyVoice2 added special tokens
    // as their atomic ids (e.g. `[breath]` = 151647), not BPE-split — this is the
    // behavior that distinguishes the CosyVoice2-configured tokenizer from a bare
    // Qwen2 BPE. We assert the `[breath]` id appears for a marker case.
    let breath_case = cases
        .iter()
        .find(|c| c.text.contains("[breath]"))
        .expect("reference corpus must include a `[breath]` marker case");
    let got = tok.encode(&breath_case.text).expect("encode breath case");
    assert!(
        got.contains(&151647),
        "`[breath]` must tokenize to its atomic added-special-token id 151647; got {got:?}"
    );

    eprintln!(
        "tokenizer parity OK: {} / {} reference cases reproduced exactly",
        cases.len(),
        cases.len()
    );
}
