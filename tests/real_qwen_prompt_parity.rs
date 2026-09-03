//! Numerical parity of the Qwen3-TTS **prompt** against the reference implementation.
//!
//! ## Why the prompt is the anchor
//!
//! Generation samples, so it cannot be compared bit-for-bit. The prompt can: it is a
//! deterministic function of `(checkpoint, text, speaker, language, instruct)`, and it is
//! where a whole class of porting bug lives — a wrong embedding table, a missing bias, a
//! misplaced addend, a dropped activation.
//!
//! ## The bug this gate exists for
//!
//! `Talker::embed_text` implemented `text_projection` as `linear_fc2(linear_fc1(x))`. The
//! reference's `Qwen3TTSTalkerResizeMLP.forward` is `linear_fc2(act_fn(linear_fc1(x)))`
//! with `hidden_act = silu`. Without the activation the projection collapses to a
//! composition of two linear maps — a plain linear map — and every projected text
//! embedding came out inflated (norms ran roughly 1.3x-1.9x the reference's, max abs
//! deviation 0.78).
//!
//! Nothing caught it, because plain synthesis still produced perfectly intelligible
//! speech (WER 0.000 against the Whisper oracle). It only became visible as audible
//! damage when an `instruct` block widened the prompt: the talker then repeated its
//! target text two to five times. That is the shape of the problem this file addresses —
//! **a numerical error large enough to corrupt behaviour, small enough to sound fine** —
//! and no amount of self-consistent Rust testing can find it. Only the reference can.
//!
//! With the activation restored the two agree to 0.0078 max abs difference, which is one
//! bf16 ulp at these magnitudes.
//!
//! ## Running it
//!
//! The fixture is produced by `scripts/gen-qwen-ref.py` (which calls the reference's own
//! modules and refuses to run if it cannot import them):
//!
//! ```text
//! scripts/run-isolated.sh ~/.venvs/qwen/bin/python scripts/gen-qwen-ref.py \
//!   --ckpt /data/models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
//!   --out ~/parity-qwen/1.7b-customvoice.safetensors
//! ```
//!
//! Then set `SYRINX_QWEN_CV_DIR` and `SYRINX_QWEN_REF`. Skips cleanly without them.

#![cfg(feature = "real")]

use candle_core::{DType, Device, Tensor};

/// Must mirror the probe case pinned in `scripts/gen-qwen-ref.py`. If these drift, the
/// test compares two different prompts and its result means nothing.
const TEXT: &str = "Come closer, I have something to tell you.";
const SPEAKER: &str = "serena";
const LANGUAGE: &str = "english";
const INSTRUCT: &str = "Whisper";

/// The fixture is dumped in **float32**, so it is the reference's true values rather than
/// a quantized copy, and the tolerance belongs to whatever dtype WE compute in.
///
/// * f32 (CPU, the parity path): measured **0.00000** — bit-exact against the reference.
///   1e-4 leaves room for a different matmul order on other hardware without admitting
///   anything real.
/// * bf16 (CUDA, the fits-in-12GB path): measured **0.0201**, which is bf16's own
///   rounding at these magnitudes and not a disagreement about the maths.
///
/// Both sit far below the missing-activation bug's 0.78 — 39x the bf16 bound — so neither
/// tolerance can hide it.
const TOL_F32: f32 = 1e-4;
const TOL_BF16: f32 = 0.05;

fn env_path(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|p| std::path::Path::new(p).exists())
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> candle_core::Result<f32> {
    let d = (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?.abs()?;
    d.flatten_all()?.max(0)?.to_scalar::<f32>()
}

#[test]
fn qwen_prompt_matches_the_reference() {
    let Some(dir) = env_path("SYRINX_QWEN_CV_DIR") else {
        eprintln!(
            "SKIP real_qwen_prompt_parity: set SYRINX_QWEN_CV_DIR to a Qwen3-TTS \
             CustomVoice checkpoint directory"
        );
        return;
    };
    let Some(ref_path) = env_path("SYRINX_QWEN_REF") else {
        eprintln!(
            "SKIP real_qwen_prompt_parity: set SYRINX_QWEN_REF to the fixture from \
             scripts/gen-qwen-ref.py"
        );
        return;
    };

    // CUDA when the binary was built with it, CPU otherwise. The board compiles
    // `--features real` WITHOUT `cuda`, so honouring SYRINX_QWEN_DEVICE unconditionally
    // turned a runnable test into a FAIL there. CPU is the more accurate comparison
    // anyway (f32 against an f32 fixture); the device only changes the tolerance headroom.
    let dev = match std::env::var("SYRINX_QWEN_DEVICE").ok().and_then(|v| v.trim().parse::<usize>().ok())
    {
        Some(i) => Device::new_cuda(i).unwrap_or_else(|e| {
            eprintln!("[qwen-parity] cuda:{i} unavailable ({e}); using CPU");
            Device::Cpu
        }),
        None => Device::Cpu,
    };

    // The port picks its compute dtype from the device (f32 on CPU for parity, bf16 on
    // CUDA to fit), so the tolerance has to follow the same rule.
    let (tol, dt_name) = if dev.is_cuda() { (TOL_BF16, "bf16") } else { (TOL_F32, "f32") };

    let refs = candle_core::safetensors::load(&ref_path, &Device::Cpu).expect("load fixture");

    let cfg_json = std::fs::read_to_string(std::path::Path::new(&dir).join("config.json"))
        .expect("read config.json");
    let pcfg = syrinx_qwen::prompt::PromptConfig::from_json(&cfg_json).expect("prompt config");
    let tok = syrinx_qwen::tokenizer::QwenTokenizer::from_dir(&dir).expect("tokenizer");
    let model = syrinx_qwen::model::Qwen3Tts::load(&dir, dev.clone()).expect("load talker");

    for (tag, instruct) in [("plain", None), ("instruct", Some(INSTRUCT))] {
        let want = refs
            .get(&format!("{tag}.inputs_embeds"))
            .unwrap_or_else(|| panic!("fixture has no {tag}.inputs_embeds"));

        let plan = syrinx_qwen::prompt::build_custom_voice(
            &tok, &pcfg, TEXT, SPEAKER, instruct, LANGUAGE, true,
        )
        .expect("build prompt");

        // Geometry first: a length mismatch means the two sides are not even describing
        // the same prompt, and a value diff would be meaningless noise on top of that.
        let (want_steps, hidden) = (want.dim(0).unwrap(), want.dim(1).unwrap());
        assert_eq!(
            plan.len(),
            want_steps,
            "{tag}: prompt has {} steps, reference has {want_steps}",
            plan.len()
        );

        let got = model
            .realize_plan(&plan, None, &[])
            .expect("realize plan")
            .inputs_embeds
            .reshape((want_steps, hidden))
            .expect("reshape")
            .to_device(&Device::Cpu)
            .expect("to cpu");

        let diff = max_abs_diff(&got, want).expect("diff");
        assert!(
            diff <= tol,
            "{tag}: prompt embeddings differ from the reference by {diff} (tolerance {tol} \
             for {dt_name}). This is the signature of a wrong table, a dropped bias or a \
             missing activation in the text/codec embedding path — not of sampling, which \
             does not enter here."
        );

        // Per-step, so a single bad position cannot average away behind 27 good ones.
        for s in 0..want_steps {
            let g = got.get(s).expect("row");
            let w = want.get(s).expect("row");
            let d = max_abs_diff(&g, &w).expect("row diff");
            assert!(d <= tol, "{tag}: step {s} differs by {d} (tolerance {tol} for {dt_name})");
        }

        eprintln!("[qwen-parity] {tag}: {want_steps} steps, {dt_name}, max abs diff {diff:.5} (tol {tol})");
    }
}
