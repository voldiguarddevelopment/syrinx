//! Numerical anchor for the **affect judge**, plus a reported (never asserted) direction
//! check on the cue A/B renders.
//!
//! # What is anchored, and against what
//!
//! `syrinx_eval::affect` runs `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`
//! through `ort`. The reference is the same checkpoint run through torch + transformers by
//! `scripts/gen-affect-ref.py`, which dumps, per clip, the 16 kHz waveform it fed, the raw
//! logits, the softmax probabilities and the mean-pooled hidden state. Same discipline as
//! `tests/real_qwen_stack_parity.rs`: capture the real upstream implementation, then anchor
//! the Rust against it, and derive every tolerance from a measurement.
//!
//! Three anchors rather than one, so a disagreement localises itself:
//!
//! * **hidden** — the mean-pooled last hidden state. Covers the entire wav2vec2 trunk (the
//!   conv front end, 24 transformer layers, the in-graph normalisation) and nothing else.
//! * **logits** — the head's raw output. If `hidden` matches and this does not, the fault
//!   is in the four numbers of `classifier.dense` / `classifier.output`, which is exactly
//!   the part of this checkpoint that is easy to get wrong (see `scripts/export-affect-onnx.py`).
//! * **probs** — what a caller actually reads.
//!
//! A fourth check covers the **driver path**: the fixture's clips are 24 kHz and the model
//! wants 16 kHz, so something must resample. The reference uses `librosa` (soxr `HQ`) and
//! Rust uses `syrinx_qwen::speaker::resample` (64-lobe Lanczos). Those are two different
//! band-limited resamplers and will never agree numerically, so that test asserts a
//! *measured* behavioural bound instead — the same arrangement, and for the same reason, as
//! `real_qwen_speaker_parity`'s `driver_path_resample` case. It is not decorative: a
//! resampler that leaks imaging above the input Nyquist has already corrupted a speaker
//! embedding once in this tree.
//!
//! # The direction check is REPORTED, NOT ASSERTED
//!
//! `CLAUDE.md` names "intended emotion" as blocked-on-human, and this judge is a proxy
//! trained on *acted human* speech being applied to *synthetic* speech. It is a screen, not
//! a certificate. The direction section prints and never fails. Turning it into a gate
//! would be inventing a green.
//!
//! # Running it
//!
//!     scripts/fetch-affect-model.sh                    # weights + ONNX export + fixture
//!     export SYRINX_AFFECT_ONNX=/data/models/w2v2-lg-xlsr-en-ser/model.onnx
//!     export SYRINX_AFFECT_REF=$HOME/parity-affect/affect.safetensors
//!     MEMMAX=10G scripts/run-isolated.sh \
//!         cargo test --features affect --release --test real_qwen_affect -- --nocapture
//!
//! CPU only. The judge is 315 M parameters in f32 and takes a couple of seconds per clip.

// Without the off-by-default `affect` feature there is no judge to test. This prints a
// SKIP rather than compiling to an empty file: a test binary with zero tests reports `ok`,
// which is indistinguishable from a real pass on the board.
#[cfg(not(feature = "affect"))]
#[test]
fn qwen_affect_requires_the_affect_feature() {
    eprintln!(
        "SKIP real_qwen_affect: built without `--features affect`, so the ONNX judge is not \
         compiled in. See scripts/fetch-affect-model.sh."
    );
}

#[cfg(feature = "affect")]
mod judged {
    use std::collections::BTreeSet;

    use candle_core::{DType, Device, Tensor};
    use syrinx_eval::affect::{
        direction, ravdess8_spec, AffectJudge, OnnxJudge, ScoreKind, RAVDESS8_LABELS,
    };

    /// Tolerances, measured rather than guessed.
    ///
    /// Both sides are CPU/float32 running the same weights, but they are two different
    /// runtimes (PyTorch eager vs ONNX Runtime) over a 24-layer transformer, so the
    /// accumulation order differs and bit-equality is not on offer. The export script
    /// already measures ORT-vs-torch on random input inside one process and reports
    /// logits `4.2e-5` / hidden `1.6e-5`; this test measures the same comparison across
    /// the process boundary on real clips.
    ///
    /// MEASURED on NovaBox, 2026-09-06, worst over all 11 fixture clips:
    ///   hidden  max|d| 2.52e-05
    ///   logits  max|d| 6.75e-05
    ///   probs   max|d| 2.06e-05
    /// (worst clip for all three: `1001_DFA_NEU_XX`, the shortest at 1.9 s.) The bounds
    /// below sit 5-8x above those — loose enough that a different CPU's BLAS cannot trip
    /// them, tight enough that any real porting fault (a dropped activation, a wrong
    /// pooling axis, a transposed head) is orders away.
    const TOL_HIDDEN: f32 = 2e-4;
    const TOL_LOGITS: f32 = 5e-4;
    const TOL_PROBS: f32 = 1e-4;

    /// The driver-path bound: our Lanczos resampler feeding the judge, against the
    /// reference's soxr resampler feeding it. NOT a parity bound — two different filters,
    /// so the waveforms genuinely differ.
    ///
    /// MEASURED on the six 24 kHz renders, 2026-09-06: worst per-class probability
    /// difference **0.0017**, worst total-variation distance **0.0021**. The bound is
    /// 0.005 — 3x the measurement, and still half the SMALLEST pair delta this test
    /// reports (0.010) and 1/50 of the largest (0.260), so a resampler fault cannot hide
    /// inside a direction verdict. Headroom is deliberately modest because this
    /// difference is deterministic (two fixed filters), not machine-dependent. For scale:
    /// the first (16-lobe, unshaded) version of that resampler leaked imaging above the
    /// input Nyquist and moved a speaker cosine by ~0.01 on a metric whose whole useful
    /// range was 0.07; the equivalent fault here would blow this bound wide open.
    const TOL_DRIVER_PATH: f32 = 5e-3;

    fn env_path(k: &str) -> Option<String> {
        std::env::var(k).ok().filter(|p| std::path::Path::new(p).exists())
    }

    /// `(judge, fixture)`, or `None` with a SKIP printed.
    fn setup(name: &str) -> Option<(OnnxJudge, Fixture)> {
        let Some(onnx) = env_path("SYRINX_AFFECT_ONNX") else {
            eprintln!(
                "SKIP {name}: set SYRINX_AFFECT_ONNX to the exported judge \
                 (scripts/fetch-affect-model.sh)"
            );
            return None;
        };
        let Some(refp) = env_path("SYRINX_AFFECT_REF") else {
            eprintln!(
                "SKIP {name}: set SYRINX_AFFECT_REF to the dump from scripts/gen-affect-ref.py"
            );
            return None;
        };
        let judge = OnnxJudge::load(&onnx, ravdess8_spec())
            .unwrap_or_else(|e| panic!("loading {onnx}: {e}"));
        Some((judge, Fixture::open(&refp)))
    }

    /// The reference dump, keyed `<group>/<stem>`.
    struct Fixture {
        tensors: std::collections::HashMap<String, Tensor>,
        stems: Vec<String>,
    }

    impl Fixture {
        fn open(path: &str) -> Fixture {
            let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }
                .unwrap_or_else(|e| panic!("opening {path}: {e}"));
            let dev = Device::Cpu;
            let mut tensors = std::collections::HashMap::new();
            let mut stems = BTreeSet::new();
            for (k, _) in st.tensors() {
                let t = st.load(&k, &dev).unwrap_or_else(|e| panic!("loading {k}: {e}"));
                if let Some(stem) = k.strip_prefix("probs/") {
                    stems.insert(stem.to_string());
                }
                tensors.insert(k, t);
            }
            assert!(!stems.is_empty(), "{path} has no `probs/*` entries — wrong fixture?");
            Fixture { tensors, stems: stems.into_iter().collect() }
        }

        fn vec(&self, key: &str) -> Vec<f32> {
            self.tensors
                .get(key)
                .unwrap_or_else(|| panic!("fixture has no {key}"))
                .to_dtype(DType::F32)
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>())
                .unwrap_or_else(|e| panic!("reading {key}: {e}"))
        }

        fn sr_in(&self, stem: &str) -> u32 {
            self.tensors
                .get(&format!("sr_in/{stem}"))
                .unwrap_or_else(|| panic!("fixture has no sr_in/{stem}"))
                .flatten_all()
                .and_then(|t| t.to_vec1::<i64>())
                .unwrap_or_else(|e| panic!("reading sr_in/{stem}: {e}"))[0] as u32
        }
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    /// The graph's label vocabulary is part of the contract: the fixture's probability
    /// vectors are in `config.id2label` order, and reading them in any other order would
    /// silently relabel every emotion.
    #[test]
    fn qwen_affect_label_contract() {
        let Some((judge, fx)) = setup("qwen_affect_label_contract") else { return };
        assert_eq!(judge.labels(), RAVDESS8_LABELS.map(String::from));
        assert_eq!(judge.kind(), ScoreKind::Categorical);
        assert_eq!(judge.sample_rate(), 16_000);
        for stem in &fx.stems {
            assert_eq!(
                fx.vec(&format!("probs/{stem}")).len(),
                RAVDESS8_LABELS.len(),
                "fixture probs/{stem} is not 8-wide"
            );
        }
    }

    /// The anchor: Rust `ort` against the reference torch run, on the reference's OWN
    /// 16 kHz samples, so no resampler is inside the comparison.
    #[test]
    fn qwen_affect_matches_the_python_reference() {
        let Some((mut judge, fx)) = setup("qwen_affect_matches_the_python_reference") else {
            return;
        };
        let (mut worst_h, mut worst_l, mut worst_p) = (0f32, 0f32, 0f32);
        for stem in &fx.stems {
            let wav = fx.vec(&format!("wav16/{stem}"));
            let got = judge.read(&wav).unwrap_or_else(|e| panic!("scoring {stem}: {e}"));

            let dh = max_abs(
                got.embedding.as_ref().expect("spec asks for hidden_states"),
                &fx.vec(&format!("hidden/{stem}")),
            );
            let dl = max_abs(&got.raw, &fx.vec(&format!("logits/{stem}")));
            let dp = max_abs(&got.score.values, &fx.vec(&format!("probs/{stem}")));
            eprintln!(
                "[affect-parity] {stem:<20} hidden {dh:.3e}  logits {dl:.3e}  probs {dp:.3e}"
            );
            worst_h = worst_h.max(dh);
            worst_l = worst_l.max(dl);
            worst_p = worst_p.max(dp);
        }
        eprintln!(
            "[affect-parity] WORST over {} clips: hidden {worst_h:.3e} logits {worst_l:.3e} \
             probs {worst_p:.3e}",
            fx.stems.len()
        );
        assert!(worst_h <= TOL_HIDDEN, "hidden {worst_h:.3e} > {TOL_HIDDEN:.0e}");
        assert!(worst_l <= TOL_LOGITS, "logits {worst_l:.3e} > {TOL_LOGITS:.0e}");
        assert!(worst_p <= TOL_PROBS, "probs {worst_p:.3e} > {TOL_PROBS:.0e}");
    }

    /// The driver path: our own resampler, 24 kHz -> 16 kHz, feeding the judge.
    ///
    /// A behavioural bound, not a numeric one. What is being defended is the claim that
    /// scoring a 24 kHz render through `syrinx_qwen::speaker::resample` gives the same
    /// verdict as scoring it through the reference's soxr — because every number this
    /// module reports for a real render comes through that path.
    #[test]
    fn qwen_affect_driver_path_resample_preserves_the_verdict() {
        let Some((mut judge, fx)) =
            setup("qwen_affect_driver_path_resample_preserves_the_verdict")
        else {
            return;
        };
        let mut worst_class = 0f32;
        let mut worst_tv = 0f32;
        let mut checked = 0usize;
        for stem in &fx.stems {
            let sr = fx.sr_in(stem);
            if sr == judge.sample_rate() {
                continue; // nothing to resample; the anchor above already covers it
            }
            let raw = fx.vec(&format!("wav_in/{stem}"));
            let ours = syrinx_qwen::speaker::resample(&raw, sr, judge.sample_rate());
            let got = judge.score(&ours).unwrap_or_else(|e| panic!("scoring {stem}: {e}"));
            let want = fx.vec(&format!("probs/{stem}"));

            let dc = max_abs(&got.values, &want);
            // Total variation between the two distributions: half the L1 distance. The
            // per-class max can look small while the whole distribution has shifted.
            let tv: f32 =
                got.values.iter().zip(&want).map(|(a, b)| (a - b).abs()).sum::<f32>() / 2.0;
            eprintln!(
                "[affect-driver] {stem:<20} {sr} Hz -> {} Hz  max class {dc:.4}  TV {tv:.4}",
                judge.sample_rate()
            );
            worst_class = worst_class.max(dc);
            worst_tv = worst_tv.max(tv);
            checked += 1;
        }
        assert!(checked > 0, "no fixture clip needed resampling — the bound tested nothing");
        eprintln!(
            "[affect-driver] WORST over {checked} clips: max class {worst_class:.4}  \
             TV {worst_tv:.4}"
        );
        assert!(worst_class <= TOL_DRIVER_PATH, "class {worst_class:.4} > {TOL_DRIVER_PATH}");
        assert!(worst_tv <= TOL_DRIVER_PATH, "TV {worst_tv:.4} > {TOL_DRIVER_PATH}");
    }

    /// The direction check. **Prints. Never fails.** See the module docs.
    #[test]
    fn qwen_affect_direction_report() {
        let Some((mut judge, fx)) = setup("qwen_affect_direction_report") else { return };

        // The A/B set is three pairs, named `<n>-<cue>-plain` / `<n>-<cue>-tagged`.
        let pairs: Vec<(String, String, String)> = fx
            .stems
            .iter()
            .filter_map(|s| s.strip_suffix("-plain"))
            .filter_map(|base| {
                let tagged = format!("{base}-tagged");
                fx.stems.iter().any(|s| s == &tagged).then(|| {
                    let cue = base.split('-').nth(1).unwrap_or(base).to_string();
                    (cue, format!("{base}-plain"), tagged)
                })
            })
            .collect();
        if pairs.is_empty() {
            eprintln!("[affect-direction] fixture has no *-plain / *-tagged pairs; nothing to say");
            return;
        }

        let score = |j: &mut OnnxJudge, stem: &str| {
            j.score(&fx.vec(&format!("wav16/{stem}"))).expect("scoring")
        };

        eprintln!(
            "\n[affect-direction] {:<8}{:<9}{}",
            "cue",
            "class",
            RAVDESS8_LABELS.map(|l| format!("{:>9}", &l[..l.len().min(5)])).join("")
        );
        for (cue, plain_stem, tagged_stem) in &pairs {
            let plain = score(&mut judge, plain_stem);
            let tagged = score(&mut judge, tagged_stem);
            let rep = direction(cue, &plain, &tagged);
            let row = |vals: &[f32]| {
                vals.iter().map(|v| format!("{v:>9.3}")).collect::<Vec<_>>().join("")
            };
            eprintln!("[affect-direction] {cue:<8}{:<9}{}", "plain", row(&plain.values));
            eprintln!("[affect-direction] {:<8}{:<9}{}", "", "tagged", row(&tagged.values));
            let deltas: Vec<f32> = rep.deltas.iter().map(|d| d.delta()).collect();
            eprintln!("[affect-direction] {:<8}{:<9}{}", "", "delta", row(&deltas));
            let (gain_label, gain) = rep.largest_gain();
            match rep.cued_delta() {
                Some(d) => eprintln!(
                    "[affect-direction] {:<8}-> cued class `{}` moved {d:+.3}; largest gain was \
                     `{gain_label}` {gain:+.3}",
                    "",
                    rep.cued_label.as_deref().unwrap_or("?")
                ),
                None => eprintln!(
                    "[affect-direction] {:<8}-> cue `{cue}` has no counterpart in this judge's \
                     8 classes; no direction available",
                    ""
                ),
            }
        }
        eprintln!(
            "[affect-direction] REPORTED, NOT ASSERTED. n = 1 per condition, so a single \
             pair's delta cannot be separated from the sampler's own run-to-run spread; and \
             this judge is acted-human-speech trained, applied to synthetic speech. Read a \
             null as \"cannot tell\", never as \"the cue did nothing\"."
        );
    }
}
