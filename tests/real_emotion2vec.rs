//! Anchor the Rust emotion2vec+ judge against funasr's own forward pass.
//!
//! The judge decides whether a cue moved a render toward its named emotion, so a silent
//! disagreement with the reference would corrupt every measurement downstream while
//! looking perfectly plausible. This is the same discipline that caught the missing `silu`
//! in the Qwen talker and both Fish codec defects: the reference is the upstream package's
//! OWN forward pass, dumped by `scripts/gen-emotion2vec-ref.py`, never a reimplementation.
//!
//! **The anchor is on LOGITS, not probabilities.** This model saturates — logit spreads of
//! 12 to 51 on the clips in the fixture — so `softmax` returns a one-hot vector to float
//! precision and two clearly different clips compare equal. `probs/*` is in the dump for
//! completeness and is deliberately not what is asserted; asserting it would be a test
//! that cannot fail.
//!
//! Opt-in. Set both:
//!
//!     export SYRINX_EMOTION2VEC_ONNX=/data/models/emotion2vec-plus-large/model.onnx
//!     export SYRINX_EMOTION2VEC_REF=$HOME/parity-affect/emotion2vec.safetensors
//!
//! Build with `--features affect`.

#![cfg(feature = "affect")]

use std::collections::BTreeSet;

use candle_core::{Device, Tensor};
use syrinx_eval::affect::{
    emotion2vec9_spec, emotion2vec_label_for_cue, AffectJudge, OnnxJudge, ScoreKind,
    EMOTION2VEC9_LABELS,
};

/// Max |onnx − torch| on a logit, over 24 transformer blocks in f32.
///
/// Measured at export time across three unseen lengths: worst 3.5e-5. This is ~30x that,
/// which leaves room for ORT graph-optimisation differences across machines while staying
/// far below the smallest thing the judge is ever asked to resolve — the sham-arm deltas
/// in `renders/2026-09-06-instruct-lang/` are order 1.0, and the cued deltas order 2-4.
const TOL_LOGIT: f32 = 1e-3;

fn env_path(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|p| std::path::Path::new(p).exists())
}

struct Fixture {
    tensors: std::collections::HashMap<String, Tensor>,
    stems: Vec<String>,
}

impl Fixture {
    fn open(path: &str) -> Fixture {
        let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }
            .unwrap_or_else(|e| panic!("opening {path}: {e}"));
        let mut tensors = std::collections::HashMap::new();
        let mut stems = BTreeSet::new();
        for (k, _) in st.tensors() {
            let t = st.load(&k, &Device::Cpu).unwrap_or_else(|e| panic!("loading {k}: {e}"));
            if let Some(stem) = k.strip_prefix("logits/") {
                stems.insert(stem.to_string());
            }
            tensors.insert(k, t);
        }
        assert!(!stems.is_empty(), "{path} has no `logits/*` entries — wrong fixture?");
        Fixture { tensors, stems: stems.into_iter().collect() }
    }

    fn vec1(&self, key: &str) -> Vec<f32> {
        self.tensors
            .get(key)
            .unwrap_or_else(|| panic!("fixture has no {key}"))
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .unwrap_or_else(|e| panic!("reading {key}: {e}"))
    }
}

fn setup(name: &str) -> Option<(OnnxJudge, Fixture)> {
    let Some(onnx) = env_path("SYRINX_EMOTION2VEC_ONNX") else {
        eprintln!(
            "SKIP {name}: set SYRINX_EMOTION2VEC_ONNX \
             (scripts/export-emotion2vec-onnx.py)"
        );
        return None;
    };
    let Some(refp) = env_path("SYRINX_EMOTION2VEC_REF") else {
        eprintln!(
            "SKIP {name}: set SYRINX_EMOTION2VEC_REF \
             (scripts/gen-emotion2vec-ref.py)"
        );
        return None;
    };
    let judge =
        OnnxJudge::load(&onnx, emotion2vec9_spec()).unwrap_or_else(|e| panic!("loading {onnx}: {e}"));
    Some((judge, Fixture::open(&refp)))
}

/// THE anchor: every clip's logit vector must match funasr's, elementwise.
#[test]
fn matches_the_funasr_reference_on_every_clip() {
    let Some((mut judge, fx)) = setup("matches_the_funasr_reference_on_every_clip") else {
        return;
    };
    let mut worst = 0.0f32;
    let mut worst_at = String::new();
    for stem in &fx.stems {
        let wav = fx.vec1(&format!("wav16/{stem}"));
        let want = fx.vec1(&format!("logits/{stem}"));
        let got = judge.read(&wav).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let got = &got.score.values;

        assert_eq!(got.len(), want.len(), "{stem}: label count");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let d = (g - w).abs();
            if d > worst {
                worst = d;
                worst_at = format!("{stem}/{}", EMOTION2VEC9_LABELS[i]);
            }
        }
    }
    assert!(
        worst <= TOL_LOGIT,
        "worst logit disagreement {worst:.3e} at {worst_at} exceeds {TOL_LOGIT:.1e} \
         — the Rust judge is not computing what funasr computes"
    );
    eprintln!("[emotion2vec] {} clips, worst |onnx-torch| = {worst:.3e}", fx.stems.len());
}

/// The fixture must contain clips the model reads *differently*, or the test above would
/// pass on a judge that returns a constant. Guards against a fixture that has quietly
/// become degenerate.
#[test]
fn the_fixture_discriminates_between_clips() {
    let Some((_, fx)) = setup("the_fixture_discriminates_between_clips") else {
        return;
    };
    assert!(fx.stems.len() >= 4, "too few clips to be a meaningful anchor");
    let tops: BTreeSet<usize> = fx
        .stems
        .iter()
        .map(|s| {
            let v = fx.vec1(&format!("logits/{s}"));
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        })
        .collect();
    assert!(
        tops.len() >= 2,
        "every clip in the fixture reads as the same class — the anchor cannot detect a \
         judge that ignores its input"
    );
}

/// Saturation, asserted rather than assumed — it is the whole reason this judge is scored
/// on logits. If a future checkpoint stops saturating, this fails and the decision to skip
/// the softmax should be revisited rather than silently inherited.
#[test]
fn the_model_saturates_which_is_why_logits_are_the_anchor() {
    let Some((_, fx)) = setup("the_model_saturates_which_is_why_logits_are_the_anchor") else {
        return;
    };
    for stem in &fx.stems {
        let lg = fx.vec1(&format!("logits/{stem}"));
        let (lo, hi) = lg.iter().fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
        assert!(
            hi - lo > 10.0,
            "{stem}: logit spread {:.2} — if the model no longer saturates, softmax would \
             carry signal and `emotion2vec9_spec` should be reconsidered",
            hi - lo
        );
        // And the corresponding probs really are one-hot, which is the failure being avoided.
        let pr = fx.vec1(&format!("probs/{stem}"));
        let top = pr.iter().cloned().fold(f32::MIN, f32::max);
        assert!(top > 0.99, "{stem}: softmax top is {top:.4}, expected saturation");
    }
}

/// The 10.005 s ceiling: the graph must accept a clip at the limit and must FAIL LOUDLY
/// past it. A silent truncation would be far worse than an error — it would score the
/// first 10 s of a longer clip and report it as the whole thing.
#[test]
fn the_length_ceiling_errors_rather_than_silently_truncating() {
    let Some((mut judge, _)) = setup("the_length_ceiling_errors_rather_than_silently_truncating")
    else {
        return;
    };
    const MAX: usize = 160_079;
    let at = vec![0.0f32; MAX];
    assert!(judge.read(&at).is_ok(), "the documented ceiling of {MAX} samples must run");

    let past = vec![0.0f32; MAX + 1];
    let err = judge.read(&past);
    assert!(
        err.is_err(),
        "{} samples returned Ok — the graph is silently truncating, which would score the \
         first 10 s of a longer clip and report it as the whole clip",
        MAX + 1
    );
}

/// The spec must declare what it actually emits. `Dimensional` because unbounded logits do
/// not sum to 1 and a rise in one class does not imply a fall in another — treating them as
/// a simplex is the mistake this guards.
#[test]
fn the_spec_declares_logits_not_a_probability_simplex() {
    let s = emotion2vec9_spec();
    assert!(matches!(s.kind, ScoreKind::Dimensional));
    assert!(matches!(s.activation, syrinx_eval::affect::OutputActivation::None));
    assert_eq!(s.labels.len(), 9);
    assert_eq!(s.sample_rate, 16_000);
}

/// The cue→class map covers the classes it should and refuses the ones it cannot.
#[test]
fn the_cue_map_covers_the_reachable_classes_and_refuses_the_rest() {
    for (cue, class) in [
        ("happy", "happy"),
        ("sad", "sad"),
        ("angry", "angry"),
        ("afraid", "fearful"),
        ("surprised", "surprised"),
        ("disgusted", "disgusted"),
    ] {
        assert_eq!(emotion2vec_label_for_cue(cue), Some(class), "cue {cue}");
        assert!(EMOTION2VEC9_LABELS.contains(&class), "{class} must be in the vocabulary");
    }
    // No counterpart: answered with None, never mapped onto the nearest class.
    for cue in ["sarcastic", "curious", "whisper", "narrator", "calm"] {
        assert_eq!(emotion2vec_label_for_cue(cue), None, "cue {cue} must have no reading");
    }
}
