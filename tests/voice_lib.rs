//! Model-free unit tests for the voice **manipulation math** + the **voice library**.
//!
//! These exercise `syrinx_serve::voice` (the [`Voice`] bundle, [`Voice::blend`] /
//! [`Voice::interpolate`] / [`Voice::slerp`] / [`VoiceArithmetic`] / [`Voice::with_attribute`],
//! and [`VoiceLibrary`]) on **synthetic** speaker embeddings / prompt mels — no model
//! weights, no frontend run — so they run on box and off. They build under the default
//! `real` feature (the `voice` module is Candle-backed) but load nothing from disk.
//!
//! Pinned behaviour:
//!   * blend = weight-normalized average of L2-normalized embeddings, re-normalized
//!     (weight-scale invariant; single voice ≈ its own normalized embedding);
//!   * interpolate/slerp hit their endpoints and stay on the unit sphere;
//!   * arithmetic `add`/`sub` cancel; the builder normalizes;
//!   * `with_attribute` is deterministic, name-dependent, and actually moves the embedding;
//!   * every op carries the **base** voice's clip-tied conditioning (prompt_feat/token/text);
//!   * `VoiceLibrary` save→load→list→remove round-trips a voice byte-exactly.

#![cfg(feature = "real")]

use candle_core::{Device, Tensor};
use syrinx_serve::voice::{Voice, VoiceArithmetic, VoiceLibrary};

const DIM: usize = 192;

/// Build a synthetic [`Voice`] with a given flat embedding (length `DIM`) and a small
/// deterministic prompt mel / token sequence keyed off `name`, on the CPU device.
fn make_voice(name: &str, emb: &[f32]) -> Voice {
    assert_eq!(emb.len(), DIM);
    let dev = Device::Cpu;
    let speaker_embedding = Tensor::from_vec(emb.to_vec(), (1, DIM), &dev).unwrap();
    // A tiny deterministic prompt mel [1, 4, 80] + 2 prompt tokens (clip-tied data).
    let feat: Vec<f32> = (0..4 * 80)
        .map(|i| (i as f32 * 0.001) + name.len() as f32)
        .collect();
    let prompt_feat = Tensor::from_vec(feat, (1, 4, 80), &dev).unwrap();
    Voice {
        name: name.to_string(),
        speaker_embedding,
        prompt_feat,
        prompt_token: vec![name.len() as i64, name.len() as i64 + 1],
        prompt_text: format!("transcript for {name}"),
        source: Some(format!("/clips/{name}.wav")),
    }
}

/// A deterministic, distinct embedding from a small seed (not normalized — exercises the
/// ops' own normalization).
fn emb_from(seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..DIM)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            // map to a spread of magnitudes incl. some scale so normalization matters
            ((s >> 33) as f32 / u32::MAX as f32 - 0.5) * 3.0
        })
        .collect()
}

fn norm(v: &[f32]) -> f64 {
    v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt()
}

fn l2(v: &[f32]) -> Vec<f32> {
    let n = norm(v);
    v.iter().map(|&x| (x as f64 / n) as f32).collect()
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
    dot / (norm(a) * norm(b))
}

fn max_abs(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .fold(0.0, f64::max)
}

#[test]
fn blend_is_weight_normalized_average_of_normalized_embeddings() {
    let a = make_voice("alice", &emb_from(1));
    let b = make_voice("bob", &emb_from(2));

    let out = Voice::blend(&[(&a, 3.0), (&b, 1.0)]).unwrap();
    let oe = out.embedding_vec().unwrap();

    // Expected: normalize(0.75*norm(a) + 0.25*norm(b)).
    let na = l2(&a.embedding_vec().unwrap());
    let nb = l2(&b.embedding_vec().unwrap());
    let mixed: Vec<f32> = na
        .iter()
        .zip(&nb)
        .map(|(&x, &y)| 0.75 * x + 0.25 * y)
        .collect();
    let expected = l2(&mixed);
    assert!(max_abs(&oe, &expected) < 1e-5, "blend != weighted normalized avg");

    // Result lives on the unit sphere.
    assert!((norm(&oe) - 1.0).abs() < 1e-5, "blend result not L2-normalized");

    // Weight-scale invariant: (2,2) == (1,1).
    let e22 = Voice::blend(&[(&a, 2.0), (&b, 2.0)]).unwrap().embedding_vec().unwrap();
    let e11 = Voice::blend(&[(&a, 1.0), (&b, 1.0)]).unwrap().embedding_vec().unwrap();
    assert!(max_abs(&e22, &e11) < 1e-6, "blend not weight-scale invariant");

    // Single voice ≈ its own normalized embedding.
    let solo = Voice::blend(&[(&a, 5.0)]).unwrap().embedding_vec().unwrap();
    assert!(max_abs(&solo, &na) < 1e-5, "single-voice blend != normalized self");

    // Carries the BASE (first) voice's clip-tied conditioning.
    assert_eq!(out.prompt_token, a.prompt_token);
    assert_eq!(out.prompt_text, a.prompt_text);
    let of: Vec<f32> = out.prompt_feat.flatten_all().unwrap().to_vec1().unwrap();
    let af: Vec<f32> = a.prompt_feat.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(of, af, "blend must carry base voice prompt_feat");
}

#[test]
fn blend_rejects_empty_and_zero_weights() {
    let a = make_voice("alice", &emb_from(1));
    let b = make_voice("bob", &emb_from(2));
    assert!(Voice::blend(&[]).is_err(), "empty blend must error");
    assert!(
        Voice::blend(&[(&a, 1.0), (&b, -1.0)]).is_err(),
        "weights summing to ~0 must error"
    );
}

#[test]
fn interpolate_and_slerp_hit_endpoints_and_stay_normalized() {
    let a = make_voice("alice", &emb_from(7));
    let b = make_voice("bob", &emb_from(8));
    let na = l2(&a.embedding_vec().unwrap());
    let nb = l2(&b.embedding_vec().unwrap());

    for (label, lo, hi) in [
        (
            "lerp",
            Voice::interpolate(&a, &b, 0.0).unwrap(),
            Voice::interpolate(&a, &b, 1.0).unwrap(),
        ),
        (
            "slerp",
            Voice::slerp(&a, &b, 0.0).unwrap(),
            Voice::slerp(&a, &b, 1.0).unwrap(),
        ),
    ] {
        let le = lo.embedding_vec().unwrap();
        let he = hi.embedding_vec().unwrap();
        assert!(max_abs(&le, &na) < 1e-5, "{label} t=0 != normalize(a)");
        assert!(max_abs(&he, &nb) < 1e-5, "{label} t=1 != normalize(b)");
    }

    // Midpoints stay on the unit sphere.
    let mid_l = Voice::interpolate(&a, &b, 0.5).unwrap().embedding_vec().unwrap();
    let mid_s = Voice::slerp(&a, &b, 0.5).unwrap().embedding_vec().unwrap();
    assert!((norm(&mid_l) - 1.0).abs() < 1e-5, "lerp midpoint not normalized");
    assert!((norm(&mid_s) - 1.0).abs() < 1e-5, "slerp midpoint not normalized");

    // SLERP midpoint is equidistant (cosine) from both endpoints — the spherical mean.
    let ca = cos(&mid_s, &na);
    let cb = cos(&mid_s, &nb);
    assert!((ca - cb).abs() < 1e-5, "slerp midpoint not equidistant ({ca} vs {cb})");

    // Interpolate carries a's clip-tied conditioning.
    let out = Voice::interpolate(&a, &b, 0.3).unwrap();
    assert_eq!(out.prompt_text, a.prompt_text);
}

#[test]
fn arithmetic_add_sub_cancel_and_normalize() {
    let a = make_voice("alice", &emb_from(11));
    let b = make_voice("bob", &emb_from(12));
    let na = l2(&a.embedding_vec().unwrap());

    // add(b) then sub(b) cancels back to normalize(a).
    let cancelled = VoiceArithmetic::base(&a)
        .add(&b, 1.0)
        .sub(&b, 1.0)
        .build()
        .unwrap()
        .embedding_vec()
        .unwrap();
    assert!(max_abs(&cancelled, &na) < 1e-5, "add then sub did not cancel");

    // base + b == normalize(norm(a)+norm(b)).
    let summed = VoiceArithmetic::base(&a)
        .add(&b, 1.0)
        .build()
        .unwrap()
        .embedding_vec()
        .unwrap();
    let nb = l2(&b.embedding_vec().unwrap());
    let expected = l2(&na.iter().zip(&nb).map(|(&x, &y)| x + y).collect::<Vec<_>>());
    assert!(max_abs(&summed, &expected) < 1e-5, "arithmetic add wrong");
    assert!((norm(&summed) - 1.0).abs() < 1e-5, "arithmetic result not normalized");

    // sub(b, w) == add(b, -w).
    let via_sub = VoiceArithmetic::base(&a).sub(&b, 0.4).build().unwrap().embedding_vec().unwrap();
    let via_add = VoiceArithmetic::base(&a).add(&b, -0.4).build().unwrap().embedding_vec().unwrap();
    assert!(max_abs(&via_sub, &via_add) < 1e-6, "sub != add(-w)");

    // Carries base conditioning.
    let out = VoiceArithmetic::base(&a).add(&b, 0.5).build().unwrap();
    assert_eq!(out.prompt_token, a.prompt_token);
}

#[test]
fn with_attribute_is_deterministic_name_dependent_and_moves_embedding() {
    let a = make_voice("alice", &emb_from(21));
    let base = l2(&a.embedding_vec().unwrap());

    let w1 = a.with_attribute("warmth", 0.2).unwrap().embedding_vec().unwrap();
    let w2 = a.with_attribute("warmth", 0.2).unwrap().embedding_vec().unwrap();
    assert_eq!(w1, w2, "with_attribute must be bit-deterministic");

    // It actually moves the embedding...
    assert!(max_abs(&w1, &base) > 1e-3, "with_attribute did not move the embedding");
    // ...but stays on the unit sphere...
    assert!((norm(&w1) - 1.0).abs() < 1e-5, "with_attribute result not normalized");
    // ...and a different axis name gives a different direction.
    let b1 = a.with_attribute("brightness", 0.2).unwrap().embedding_vec().unwrap();
    assert!(max_abs(&b1, &w1) > 1e-3, "distinct attribute names should differ");

    // amount=0 is a no-op (just a re-normalization of an already-normalized vec).
    let z = a.with_attribute("warmth", 0.0).unwrap().embedding_vec().unwrap();
    assert!(max_abs(&z, &base) < 1e-5, "with_attribute amount=0 should be ~identity");
}

#[test]
fn voice_library_round_trips_byte_exactly() {
    let dir = std::env::temp_dir().join(format!(
        "syrinx_voicelib_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let lib = VoiceLibrary::open(&dir).unwrap();

    let a = make_voice("alice", &emb_from(31));
    let b = make_voice("bob", &emb_from(32));

    // Empty to start.
    assert!(lib.list().unwrap().is_empty());
    assert!(lib.load("alice").is_err(), "loading absent voice must error");
    assert!(lib.remove("alice").is_err(), "removing absent voice must error");

    lib.save(&a).unwrap();
    lib.save(&b).unwrap();
    assert_eq!(lib.list().unwrap(), vec!["alice".to_string(), "bob".to_string()]);

    // Load alice and compare every field byte-exactly.
    let la = lib.load("alice").unwrap();
    assert_eq!(la.name, a.name);
    assert_eq!(la.prompt_token, a.prompt_token);
    assert_eq!(la.prompt_text, a.prompt_text);
    assert_eq!(la.source, a.source);
    let (oe, ne): (Vec<f32>, Vec<f32>) = (
        a.speaker_embedding.flatten_all().unwrap().to_vec1().unwrap(),
        la.speaker_embedding.flatten_all().unwrap().to_vec1().unwrap(),
    );
    assert_eq!(oe, ne, "embedding must round-trip byte-exactly");
    let (of, nf): (Vec<f32>, Vec<f32>) = (
        a.prompt_feat.flatten_all().unwrap().to_vec1().unwrap(),
        la.prompt_feat.flatten_all().unwrap().to_vec1().unwrap(),
    );
    assert_eq!(of, nf, "prompt_feat must round-trip byte-exactly");
    assert_eq!(la.prompt_feat.dims(), a.prompt_feat.dims(), "prompt_feat shape preserved");

    // Remove bob; list shrinks; bob no longer loadable.
    lib.remove("bob").unwrap();
    assert_eq!(lib.list().unwrap(), vec!["alice".to_string()]);
    assert!(lib.load("bob").is_err());

    // Path-bearing names are rejected (no directory escape).
    assert!(lib.save(&make_voice("../evil", &emb_from(1))).is_err());

    std::fs::remove_dir_all(&dir).ok();
}
