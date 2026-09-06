//! **Per-render seed control on the Qwen3-TTS engine**, proved on real weights.
//!
//! Until now `syrinx_serve::qwen::QwenRequest` carried no draw selector and
//! `synth_qwen::QwenModelEngine` rendered with whatever `DriveParams` it was configured
//! with, so every call for the same request produced the same audio. That is fine for a
//! corpus render and fatal for anything that has to separate a real prosody change from a
//! lucky draw: `syrinx_eval::qwen::measure_activation` refuses outright, because a
//! permutation test over two groups of *identical* vectors calls any difference
//! significant and would report ~100 % cue activation for a backend that ignores cues.
//!
//! [`QwenEngine::render_seeded`] closes that. This file is the gate on it, and it needs
//! weights: "seed 7 twice is the same audio, seed 8 is not" is a statement about the
//! model's sampler, and no stand-in engine can make it true or false.
//!
//! # The four properties
//!
//! 1. **A seed reproduces.** Two `render_seeded(req, 7)` calls are bit-identical —
//!    `syrinx_qwen::model::DriveParams` promises `(seed, weights, prompt)` reproduces
//!    bit-for-bit, and this is that promise reaching through `syrinx-serve`.
//! 2. **Two seeds differ.** `render_seeded(req, 8)` is a *different* draw. Without this
//!    the first property would be satisfiable by an engine that ignores the seed
//!    entirely — which is exactly the failure mode being fixed, so asserting reproduction
//!    alone would be worse than asserting nothing.
//! 3. **The clobber trap is closed.** `with_drive_params` replaces the whole
//!    `DriveParams`, seed included, so `.with_seed(s).with_drive_params(p)` silently loses
//!    `s` — the hazard the module already documents. The per-render seed is an *argument*,
//!    not engine state, so no builder can reach it: property 1 is re-checked **after** a
//!    deliberately hostile builder chain that sets two decoy seeds, and must return the
//!    same bytes as before it.
//! 4. **The default path is unchanged.** `render()` still takes the configured seed and
//!    nothing else: with the engine configured for seed 8, `render()` reproduces the
//!    `render_seeded(req, 8)` bytes exactly.
//!
//! Plus two free (no-render) checks that the *advertisement* is honest:
//! `honors_seed()` is true for a sampling engine and false once `greedy` is configured,
//! because greedy decode makes every draw an argmax and, in the port's own words, "the run
//! stops depending on `seed`".
//!
//! # Running it
//!
//! CPU only; never `--features cuda`. It self-skips cleanly without its env:
//!
//! ```text
//! source scripts/test-all.env
//! MEMMAX=10G scripts/run-isolated.sh \
//!   cargo test --features real --release --test real_qwen_seed -- --nocapture
//! ```
//!
//! The **0.6B** CustomVoice checkpoint is deliberate (2.4 GB on disk against the 1.7B's
//! 4.3, and the CPU path upcasts to f32); `SYRINX_QWEN_CV_DIR` (the 1.7B) is not a
//! fallback. Four renders are the minimum that can carry the four properties — one
//! reference draw, one after the hostile builder chain, one at a second seed, and one
//! through the unseeded path — and [`MAX_FRAMES`] caps each of them, because these
//! assertions compare sample buffers and do not care whether the sentence finished.
//!
//! **Cost, measured on NovaBox 2026-09-06** (CPU/f32, `--features real --release`, warm
//! build, `MEMMAX=10G`): **372 s** in total — 94.9 s + 92.0 s + 92.0 s + 92.4 s for the
//! four renders, and 1.0 s for the engine load, which is an mmap. Peak RSS stayed around
//! 4.5 GB, well inside the cap. Roughly a third of `real_qwen_serve`'s ~12 min, entirely
//! because of [`MAX_FRAMES`]. It is nevertheless **opt-in** — it belongs in
//! `OPT_IN_TESTS`, not in `GROUP_qwen_ckpt`, whose four tests take 23 s together.
//!
//! **Result of that run** (all four properties green): seed 7 rendered
//! `digest 1ac3c598aeaacb9e` both before and after the hostile builder chain; seed 8
//! rendered `digest 8ca38cb7cde9ef3d`, differing in **30720 of 30720** samples with a
//! largest gap of 0.5002; and `render()` at configured seed 8 reproduced seed 8's digest
//! exactly.

#![cfg(feature = "real")]

use std::path::Path;
use std::time::Instant;

use candle_core::Device;
use syrinx_cue::BackendId;
use syrinx_qwen::model::DriveParams;
use syrinx_serve::qwen::{plan, QwenEngine, QwenRequest};
use syrinx_serve::synth_qwen::{QwenModelEngine, QwenVoice};

/// The probe. The sentence `tests/real_qwen_serve.rs` renders at WER 0.000 on this
/// checkpoint, so it is known to drive the model into ordinary speech rather than into an
/// immediate EOS — which is what makes a capped render worth comparing at all.
const TEXT: &str = "Come closer, I have something to tell you.";

/// Frames per render, capped hard.
///
/// This test compares sample buffers; it never transcribes, so it has no reason to pay
/// for a complete sentence. At 12.5 Hz, 16 frames is ~1.3 s of audio and long enough that
/// two seeds diverge across 16 x 16 = 256 draws, while cutting each render to roughly a
/// third of what the uncapped sentence costs. `real_qwen_serve.rs` is the test that
/// checks a *whole* render is intelligible; this one checks *which* render you get.
const MAX_FRAMES: usize = 16;

/// The draw under test.
const SEED_A: u64 = 7;
/// A second draw, which must not be the same audio.
const SEED_B: u64 = 8;
/// A decoy pushed into engine state with `with_seed`, to be clobbered.
const DECOY_STORED: u64 = 4_242;
/// A second decoy, carried in the whole-`DriveParams` replacement that does the
/// clobbering.
const DECOY_REPLACED: u64 = 9_999;

/// The engine's configuration for this test: the frame cap, everything else stock.
fn capped(seed: u64) -> DriveParams {
    DriveParams { seed, max_new_frames: MAX_FRAMES, ..DriveParams::default() }
}

fn env_path(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.trim().is_empty()).filter(|p| Path::new(p).exists())
}

/// The two directories this file needs, or `None` (with the reason printed) so it
/// self-skips.
fn env() -> Option<(String, String)> {
    let Some(cv_dir) = env_path("SYRINX_QWEN_CV_DIR_0_6B") else {
        eprintln!(
            "SKIP real_qwen_seed: set SYRINX_QWEN_CV_DIR_0_6B to the \
             Qwen3-TTS-12Hz-0.6B-CustomVoice checkpoint dir (the 1.7B is deliberately NOT \
             a fallback — it is nearly twice the weights for the same assertion)"
        );
        return None;
    };
    let Some(tok_dir) = env_path("SYRINX_QWEN_TOK_DIR") else {
        eprintln!("SKIP real_qwen_seed: set SYRINX_QWEN_TOK_DIR to the Qwen3-TTS-Tokenizer-12Hz dir");
        return None;
    };
    Some((cv_dir, tok_dir))
}

/// Bit-exact sample comparison. `==` on `f32` is not the question being asked — two draws
/// could differ by a hair and `-0.0 == 0.0` — so the bit patterns are compared directly.
fn bit_identical(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// How far apart two renders are, for the log: differing samples and the largest gap.
fn divergence(a: &[f32], b: &[f32]) -> (usize, usize, f32) {
    let n = a.len().min(b.len());
    let differing = (0..n).filter(|&i| a[i].to_bits() != b[i].to_bits()).count();
    let worst = (0..n).map(|i| (a[i] - b[i]).abs()).fold(0.0f32, f32::max);
    (differing, n, worst)
}

/// A short, stable fingerprint of a render, so the log shows *which* draw each line is.
fn digest(samples: &[f32]) -> u64 {
    samples.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, s| {
        (h ^ u64::from(s.to_bits())).wrapping_mul(0x1000_0000_01b3)
    })
}

/// Peak amplitude — a render of digital silence would satisfy every equality below while
/// proving nothing, so each one is checked for actual signal.
fn peak(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0f32, |m, s| m.max(s.abs()))
}

/// Render, time it, and refuse a result that is not audio.
fn render_seeded(engine: &QwenModelEngine, req: &QwenRequest<'_>, seed: u64, label: &str) -> Vec<f32> {
    let t = Instant::now();
    let wav = engine
        .render_seeded(req, seed)
        .unwrap_or_else(|e| panic!("{label}: render_seeded(seed={seed}) failed: {e}"));
    report(label, &wav, t);
    wav
}

fn report(label: &str, wav: &[f32], t: Instant) {
    let secs = t.elapsed().as_secs_f64();
    assert!(!wav.is_empty(), "{label}: the engine returned no samples");
    let p = peak(wav);
    assert!(p > 0.001, "{label}: the render is silent (peak {p:e}) — nothing was compared");
    eprintln!(
        "[qwen-seed] {label}: {} samples ({:.2}s audio), peak {p:.4}, digest {:016x}, rendered in {secs:.1}s",
        wav.len(),
        wav.len() as f64 / 24_000.0,
        digest(wav)
    );
}

/// The whole gate, in one test function: it holds one multi-gigabyte f32 model and four
/// renders of it, and splitting the properties across `#[test]`s would mean loading the
/// checkpoint (and paying the reference render) once per property.
#[test]
fn a_per_render_seed_picks_the_draw_and_no_builder_can_clobber_it() {
    let Some((cv_dir, tok_dir)) = env() else {
        return;
    };
    let backend = BackendId::Qwen06bCustomVoice;

    let t0 = Instant::now();
    let engine = QwenModelEngine::load(
        backend,
        &cv_dir,
        &tok_dir,
        Device::Cpu,
        QwenVoice::Preset("serena".to_string()),
    )
    .expect("load the 0.6B CustomVoice checkpoint")
    .with_drive_params(capped(0));
    eprintln!("[qwen-seed] engine loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // A sampling engine says so, before anyone spends four renders finding out.
    assert!(
        engine.honors_seed(),
        "a stock (non-greedy) engine must advertise that render_seeded selects the draw"
    );

    // The request comes from the planner, not from a literal, so the mode and the
    // instruct-effect are the checkpoint's real ones rather than this file's guess.
    let planned = plan(backend, TEXT).expect("plan the probe");
    assert_eq!(planned.segments.len(), 1, "the probe carries no cue and must not split");
    let seg = &planned.segments[0];
    let req = QwenRequest {
        mode: planned.mode,
        text: &seg.text,
        instruct: seg.instruct.as_deref(),
        instruct_effect: planned.instruct_effect,
    };

    // ---- 1. the reference draw ---------------------------------------------
    let a = render_seeded(&engine, &req, SEED_A, "seed A (stored seed 0)");

    // ---- 3. the clobber trap, sprung on purpose -----------------------------
    // `with_seed` then `with_drive_params` is precisely the order the module warns about:
    // the second call replaces the whole `DriveParams`, so DECOY_STORED is gone and the
    // engine's configured seed is now DECOY_REPLACED. Neither decoy may be able to reach
    // a render whose seed was passed as an argument.
    let engine = engine.with_seed(DECOY_STORED).with_drive_params(capped(DECOY_REPLACED));

    let b = render_seeded(&engine, &req, SEED_A, "seed A again (stored seed clobbered)");
    assert!(
        bit_identical(&a, &b),
        "the same seed did not reproduce across a hostile builder chain: {} vs {} samples, \
         digests {:016x} / {:016x} — a per-render seed that a `with_seed` / \
         `with_drive_params` pair can influence is exactly the trap this design exists to \
         close",
        a.len(),
        b.len(),
        digest(&a),
        digest(&b)
    );
    eprintln!("[qwen-seed] seed {SEED_A} reproduced bit-for-bit after the clobber chain");

    // ---- 2. a different seed is a different draw ----------------------------
    let c = render_seeded(&engine, &req, SEED_B, "seed B");
    assert!(
        !bit_identical(&a, &c),
        "seeds {SEED_A} and {SEED_B} produced bit-identical audio ({} samples, digest \
         {:016x}). Either the seed is being discarded — in which case reproduction above \
         proves nothing — or the sampler is not sampling.",
        a.len(),
        digest(&a)
    );
    let (differing, compared, worst) = divergence(&a, &c);
    eprintln!(
        "[qwen-seed] seed {SEED_A} vs {SEED_B}: {} vs {} samples, {differing}/{compared} \
         compared samples differ, largest gap {worst:.4}",
        a.len(),
        c.len()
    );

    // ---- 4. the unseeded path is unchanged ----------------------------------
    // `render()` still means "the configured draw, and nothing else": configure SEED_B and
    // it must reproduce the render_seeded(SEED_B) bytes exactly. This also pins that
    // `render_seeded` does not disturb the engine's own configuration — the four renders
    // above all left it intact.
    let engine = engine.with_seed(SEED_B);
    let t = Instant::now();
    let d = engine.render(&req).expect("the unseeded path must still render");
    report("render() at configured seed B", &d, t);
    assert!(
        bit_identical(&c, &d),
        "render() and render_seeded(seed={SEED_B}) disagree ({:016x} vs {:016x}) — the \
         default path changed behaviour",
        digest(&c),
        digest(&d)
    );

    // ---- the advertisement stays honest under greedy decode -----------------
    // Greedy makes every draw an argmax, so no seed can change the output; an engine that
    // still claimed to honour seeds would promise variance it cannot produce.
    let engine = engine.with_drive_params(DriveParams { greedy: true, ..capped(SEED_A) });
    assert!(
        !engine.honors_seed(),
        "a greedy engine must not advertise that render_seeded selects the draw"
    );
}
