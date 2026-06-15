//! T-02.01c — deterministic weight generation by name, frozen RED tests.
//!
//! Weights are not stored in a file: every named tensor's values are a pure
//! function of its name via `reference.py` §2 (= PARITY.md §2). These pin that
//! port byte-for-byte:
//!   * C1 — `fnv1a_64(name)` is unsalted FNV-1a-64 (offset basis
//!     `0xCBF29CE484222325`, prime `0x00000100000001B3`, XOR-then-multiply mod
//!     2^64). The seed for `"tok_embeddings"` is pinned to its reference value
//!     and flipping a single byte of the name changes the seed.
//!   * C2 — `xorshift64_next` advances the stream as `x ^= x<<13; x ^= x>>7;
//!     x ^= x<<17` (wrapping mod 2^64) and emits the post-update state, with the
//!     `0` seed substituted by `0x9E3779B97F4A7C15`. The first emitted `u64` for
//!     the `tok_embeddings` seed is pinned (a swapped shift order would diverge)
//!     and the seed-`0` substitution is pinned against the magic constant.
//!   * C3 — `weights("tok_embeddings", 8)` matches `weights_sample.json`'s `data`
//!     within `tol` (1e-4) max-abs; corrupting any one of the 8 values fails.
//!   * C4 — `weights(name, count)` returns exactly `count` values that are the
//!     length-`count` prefix of the stream, so `weights(name, 8)` equals the
//!     first 8 of `weights(name, 16)`; a different `name` yields a different
//!     first value (the stream is name-seeded, not constant).
//!
//! RED: `syrinx-core` exposes none of `fnv1a_64`/`xorshift64_next`/`weights`, so
//! this target fails to build. GREEN adds them as a direct transcription of
//! `reference.py` §2. This file is frozen at red-pass; do not edit it in GREEN.

use syrinx_core::{fnv1a_64, weights, xorshift64_next};

// ----- golden plumbing --------------------------------------------------------

fn load(name: &str) -> serde_json::Value {
    let path = format!("{}/tests/golden/parity/{}", env!("CARGO_MANIFEST_DIR"), name);
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

/// A flat JSON number array → `Vec<f32>` (the golden `data` field).
fn flat(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .expect("data is an array")
        .iter()
        .map(|x| x.as_f64().expect("data cell is a number") as f32)
        .collect()
}

fn tol(g: &serde_json::Value) -> f32 {
    g["tol"].as_f64().expect("tol is a number") as f32
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "compared vectors must be equal length");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// The reference FNV-1a-64 seed of the literal string `"tok_embeddings"`, computed
// directly from `reference.py` §2.1 (offset basis × prime over the UTF-8 bytes).
const TOK_EMBEDDINGS_SEED: u64 = 0xA082_3965_E393_A4B0;
// The first post-update state emitted by the xorshift64 stream seeded with the
// `tok_embeddings` seed (`reference.py` §2.2).
const TOK_EMBEDDINGS_FIRST_EMIT: u64 = 0x565A_A84A_E7D9_AFF9;
// The seed-`0` substitution constant (`reference.py` §2.2).
const MAGIC_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

// ----- C1: FNV-1a-64 name hash -----------------------------------------------

#[test]
fn test_fnv1a_64_tok_embeddings_seed_pinned() {
    // The seed for the literal name is pinned to its reference value.
    assert_eq!(
        fnv1a_64("tok_embeddings"),
        TOK_EMBEDDINGS_SEED,
        "FNV-1a-64 of \"tok_embeddings\" must equal the reference seed"
    );

    // A degenerate input pins that the loop actually folds bytes: the empty
    // string hashes to the bare offset basis (no XOR/multiply steps run), so a
    // non-empty name must differ from it.
    assert_eq!(
        fnv1a_64(""),
        0xCBF2_9CE4_8422_2325,
        "FNV-1a-64 of the empty string is exactly the offset basis"
    );
    assert_ne!(
        fnv1a_64("tok_embeddings"),
        fnv1a_64(""),
        "a non-empty name must hash differently from the offset basis"
    );
}

#[test]
fn test_fnv1a_64_single_flipped_byte_changes_seed() {
    // Flipping exactly one byte of the name ('g' -> 'h' at index 13) changes the
    // seed — the hash depends on every byte, not just length or a prefix.
    let base = fnv1a_64("tok_embeddings");
    let flipped = fnv1a_64("tok_embeddinhs");
    assert_ne!(
        base, flipped,
        "a single flipped name byte must change the FNV-1a-64 seed"
    );

    // A different single-byte change (last byte 's' -> 't') also diverges, and
    // from the first flip too — pinning byte-position sensitivity, not a fluke.
    let flipped_last = fnv1a_64("tok_embeddingt");
    assert_ne!(base, flipped_last);
    assert_ne!(flipped, flipped_last);
}

// ----- C2: xorshift64 stream advance + emit ----------------------------------

#[test]
fn test_xorshift64_first_emit_pinned() {
    // The first emitted u64 (post-update state) for the tok_embeddings seed is
    // pinned. The exact triple-shift order `<<13, >>7, <<17` produces this value;
    // any swapped shift order or wrong shift amount diverges and fails here.
    assert_eq!(
        xorshift64_next(TOK_EMBEDDINGS_SEED),
        TOK_EMBEDDINGS_FIRST_EMIT,
        "first xorshift64 emit for the tok_embeddings seed must match the reference"
    );

    // The emit is the post-update state, never the seed itself.
    assert_ne!(
        xorshift64_next(TOK_EMBEDDINGS_SEED),
        TOK_EMBEDDINGS_SEED,
        "the emitted value is the advanced state, not the input seed"
    );
}

#[test]
fn test_xorshift64_zero_seed_substitution() {
    // A `0` seed is substituted by the magic constant before advancing, so the
    // emit for seed `0` equals the emit for the magic constant exactly...
    assert_eq!(
        xorshift64_next(0),
        xorshift64_next(MAGIC_SEED),
        "seed 0 must be substituted by 0x9E3779B97F4A7C15 before advancing"
    );

    // ...and a non-zero seed is NOT substituted: it advances to its own value,
    // distinct from the substituted path (pins the guard fires only on `0`).
    assert_ne!(
        xorshift64_next(1),
        xorshift64_next(MAGIC_SEED),
        "a non-zero seed must not take the substitution branch"
    );
    // The emit for a zero seed is non-zero (the magic state advances to it).
    assert_ne!(xorshift64_next(0), 0, "the substituted stream emits a non-zero state");
}

// ----- C3: weights golden parity + corruption sensitivity --------------------

#[test]
fn test_weights_tok_embeddings_golden_parity() {
    let g = load("weights_sample.json");
    let got = weights("tok_embeddings", 8);

    // Exactly 8 values, matching the golden `shape`.
    assert_eq!(got.len(), 8, "weights(name, 8) returns exactly 8 values");

    // Max-abs elementwise difference to the golden `data` is within tol (1e-4).
    let t = tol(&g);
    let d = max_abs_diff(&got, &flat(&g["data"]));
    assert!(d <= t, "weights max-abs diff {d} exceeds tol {t}");
}

#[test]
fn test_weights_golden_corruption_fails() {
    let g = load("weights_sample.json");
    let got = weights("tok_embeddings", 8);
    let t = tol(&g);

    // Sanity: the true golden is within tol.
    let golden = flat(&g["data"]);
    assert!(max_abs_diff(&got, &golden) <= t, "true golden must be within tol");

    // Corrupting ANY one of the 8 values pushes that case past tol — the parity
    // check is sensitive at every element, not just the first.
    for i in 0..8 {
        let mut corrupt = golden.clone();
        corrupt[i] += 1.0;
        let d = max_abs_diff(&got, &corrupt);
        assert!(
            d > t,
            "corrupting value {i} (diff {d}) should exceed tol {t}"
        );
    }
}

// ----- C4: count = stream prefix; name-seeded, not constant -------------------

#[test]
fn test_weights_count_is_stream_prefix() {
    // `weights(name, 8)` is exactly the length-8 prefix of `weights(name, 16)` —
    // the same name-seeded stream, just read further. Both also return exactly
    // `count` values.
    let eight = weights("tok_embeddings", 8);
    let sixteen = weights("tok_embeddings", 16);

    assert_eq!(eight.len(), 8, "weights(name, 8) returns exactly 8 values");
    assert_eq!(sixteen.len(), 16, "weights(name, 16) returns exactly 16 values");

    assert_eq!(
        eight.as_slice(),
        &sixteen[..8],
        "weights(name, 8) must equal the first 8 draws of weights(name, 16)"
    );

    // A length-0 request yields no values (the prefix is empty).
    assert_eq!(weights("tok_embeddings", 0).len(), 0);
}

#[test]
fn test_weights_first_value_is_name_seeded() {
    // The stream is seeded by `fnv1a_64(name)`, so a different name yields a
    // different first value — the generator is not a name-independent constant.
    let a = weights("tok_embeddings", 1);
    let b = weights("output.weight", 1);
    assert_eq!(a.len(), 1);
    assert_eq!(b.len(), 1);
    assert_ne!(
        a[0], b[0],
        "distinct names must seed distinct streams (different first value)"
    );

    // The first value matches the first golden sample (ties C4 back to the
    // name-seeded stream that C3 pins).
    let g = load("weights_sample.json");
    let first_golden = flat(&g["data"])[0];
    assert!(
        (a[0] - first_golden).abs() <= tol(&g),
        "the first weight equals the first golden draw"
    );
}
