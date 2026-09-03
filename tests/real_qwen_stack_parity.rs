//! Numerical parity for the rest of the Qwen3-TTS stack: talker, code predictor, codec.
//!
//! `real_qwen_prompt_parity.rs` gates the prompt. It caught nothing else, and could not:
//! the prompt is only the input to three more stages, and the bug that motivated all of
//! this (a dropped `silu` in `text_projection`) proved that an error can be numerically
//! small, audibly invisible, and still wreck behaviour. Each stage below therefore gets
//! its own anchor, so a future mismatch localizes itself instead of showing up as
//! "the audio sounds a bit off".
//!
//! Every anchor is deterministic, which is what makes it comparable at all:
//!
//! * **talker** — logits from the prefill forward over the instruct prompt. Gates
//!   attention, RoPE, the norms and the codec head; none of that is visible in the prompt.
//! * **code predictor** — its first-step logits, computed from the reference's OWN
//!   captured input embeddings. Feeding the reference's input rather than one we derive
//!   is deliberate: the predictor's real input depends on the sampled group-0 code, so
//!   deriving it would make this test depend on sampling and stop being a parity check.
//! * **codec** — a fixed code grid decoded to a waveform, with no language model in the
//!   path at all, plus a boundary grid at the last valid row of every table.
//!
//! Fixture: `scripts/gen-qwen-ref.py` (calls the reference's own modules, refuses to run
//! if it cannot import them). Gated on `SYRINX_QWEN_CV_DIR`, `SYRINX_QWEN_TOK_DIR` and
//! `SYRINX_QWEN_REF`; skips cleanly without them.

#![cfg(feature = "real")]

use candle_core::{DType, Device, Tensor};

const TEXT: &str = "Come closer, I have something to tell you.";
const SPEAKER: &str = "serena";
const LANGUAGE: &str = "english";
const INSTRUCT: &str = "Whisper";

/// Tolerances, measured rather than guessed.
///
/// The fixture is generated on **CPU/float32**, which is the parity path, and that choice
/// is what makes these numbers meaningful. The decoder is a deep conv stack where
/// accumulation order matters: the reference decoding the same codes on CUDA vs CPU
/// disagrees with ITSELF by 0.031 max abs on a [-1,1] waveform. A CUDA-generated fixture
/// would spend the entire error budget on that and leave nothing to detect a real fault
/// with.
///
/// Measured against the CPU fixture: talker logits 0.00003, predictor logits 0.00002,
/// codec waveform 0.000022 (correlation 1.0000000). The f32 bounds sit an order or two
/// above those, and the bf16 bounds accommodate running the same test on CUDA.
const TOL_LOGITS_F32: f32 = 2e-2;
const TOL_LOGITS_BF16: f32 = 0.6;
/// Waveform samples live in [-1, 1].
const TOL_WAV_F32: f32 = 1e-3;
const TOL_WAV_BF16: f32 = 5e-2;

fn env_path(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|p| std::path::Path::new(p).exists())
}

fn device() -> Device {
    match std::env::var("SYRINX_QWEN_DEVICE").ok().and_then(|v| v.trim().parse::<usize>().ok()) {
        Some(i) => Device::new_cuda(i).unwrap_or_else(|e| {
            eprintln!("[qwen-stack] cuda:{i} unavailable ({e}); using CPU");
            Device::Cpu
        }),
        None => Device::Cpu,
    }
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> candle_core::Result<f32> {
    let d = (a.to_dtype(DType::F32)?.flatten_all()? - b.to_dtype(DType::F32)?.flatten_all()?)?
        .abs()?;
    d.max(0)?.to_scalar::<f32>()
}

/// `(fixture, checkpoint dir, tokenizer dir, device)`, or `None` with a SKIP printed.
fn setup(name: &str) -> Option<(std::collections::HashMap<String, Tensor>, String, String, Device)> {
    let Some(dir) = env_path("SYRINX_QWEN_CV_DIR") else {
        eprintln!("SKIP {name}: set SYRINX_QWEN_CV_DIR to a CustomVoice checkpoint dir");
        return None;
    };
    let Some(tok_dir) = env_path("SYRINX_QWEN_TOK_DIR") else {
        eprintln!("SKIP {name}: set SYRINX_QWEN_TOK_DIR to the Qwen3-TTS tokenizer dir");
        return None;
    };
    let Some(ref_path) = env_path("SYRINX_QWEN_REF") else {
        eprintln!("SKIP {name}: set SYRINX_QWEN_REF to the fixture from scripts/gen-qwen-ref.py");
        return None;
    };
    let refs = candle_core::safetensors::load(&ref_path, &Device::Cpu).expect("load fixture");
    Some((refs, dir, tok_dir, device()))
}

#[test]
fn qwen_talker_prefill_logits_match_the_reference() {
    let Some((refs, dir, _tok, dev)) = setup("real_qwen_stack_parity::talker") else { return };
    let want = refs.get("talker.prefill_logits").expect("fixture: talker.prefill_logits");
    let tol = if dev.is_cuda() { TOL_LOGITS_BF16 } else { TOL_LOGITS_F32 };

    let cfg_json = std::fs::read_to_string(std::path::Path::new(&dir).join("config.json"))
        .expect("config.json");
    let pcfg = syrinx_qwen::prompt::PromptConfig::from_json(&cfg_json).expect("prompt config");
    let tok = syrinx_qwen::tokenizer::QwenTokenizer::from_dir(&dir).expect("tokenizer");
    let mut model = syrinx_qwen::model::Qwen3Tts::load(&dir, dev.clone()).expect("load talker");

    let plan = syrinx_qwen::prompt::build_custom_voice(
        &tok, &pcfg, TEXT, SPEAKER, Some(INSTRUCT), LANGUAGE, true,
    )
    .expect("build prompt");
    let prompt = model.realize_plan(&plan, None, &[]).expect("realize");

    model.talker_mut().reset();
    let h = model.talker_mut().forward(&prompt.inputs_embeds).expect("talker prefill");
    let got = model.talker().codec_logits(&h).expect("codec logits");

    let diff = max_abs_diff(&got, want).expect("diff");
    eprintln!(
        "[qwen-stack] talker prefill logits: {} values, max abs diff {diff:.5} (tol {tol})",
        want.elem_count()
    );
    assert!(
        diff <= tol,
        "talker prefill logits differ from the reference by {diff} (tolerance {tol}). \
         The prompt gate passing while this fails would place the fault in attention, \
         RoPE, the norms or the codec head."
    );
}

#[test]
fn qwen_code_predictor_logits_match_the_reference() {
    let Some((refs, dir, _tok, dev)) = setup("real_qwen_stack_parity::predictor") else { return };
    let want = refs.get("predictor.logits").expect("fixture: predictor.logits");
    // The reference's own captured input, NOT one derived here: the real input depends on
    // the sampled group-0 code, and deriving it would make this depend on sampling.
    let input = refs
        .get("predictor.inputs_embeds")
        .expect("fixture: predictor.inputs_embeds")
        .to_device(&dev)
        .expect("to device");
    let tol = if dev.is_cuda() { TOL_LOGITS_BF16 } else { TOL_LOGITS_F32 };

    let mut model = syrinx_qwen::model::Qwen3Tts::load(&dir, dev.clone()).expect("load");
    let dt = model.dtype();
    let x = input.unsqueeze(0).expect("batch").to_dtype(dt).expect("dtype");

    let bridged = model.code_predictor().bridge(&x).expect("bridge to predictor width");
    let cp = model.code_predictor_mut();
    cp.reset();
    let h = cp.forward(&bridged).expect("predictor forward");
    let got = cp.group_logits(0, &h).expect("group 0 logits");

    let diff = max_abs_diff(&got, want).expect("diff");
    eprintln!(
        "[qwen-stack] code predictor logits: {} values, max abs diff {diff:.5} (tol {tol})",
        want.elem_count()
    );
    assert!(
        diff <= tol,
        "code predictor logits differ from the reference by {diff} (tolerance {tol}). \
         Its input came from the fixture, so this is the fast-AR stack itself, not the \
         talker handoff."
    );
}

#[test]
fn qwen_codec_decode_matches_the_reference() {
    let Some((refs, _dir, tok_dir, dev)) = setup("real_qwen_stack_parity::codec") else { return };
    let dt = if dev.is_cuda() { DType::BF16 } else { DType::F32 };
    let tol = if dev.is_cuda() { TOL_WAV_BF16 } else { TOL_WAV_F32 };

    let cw = syrinx_qwen::load::load_tensors(&tok_dir, &dev, dt).expect("load codec tensors");
    let w = syrinx_qwen::nn::Weights { map: cw, dev: dev.clone(), dt };
    let tk_json = std::fs::read_to_string(std::path::Path::new(&tok_dir).join("config.json"))
        .expect("tokenizer config.json");
    let dcfg = syrinx_qwen::codec::decoder::DecoderConfig::from_json(&tk_json).expect("decoder cfg");
    let dec = syrinx_qwen::codec::decoder::Decoder::new("decoder", dcfg);

    // Both grids: ordinary interior values, and every group at the last valid row. The
    // second exists because this codec advertises `codebook_size: 2048` next to
    // `semantic_codebook_size: 4096`, which is exactly the ambiguity that produced a real
    // out-of-range bug in the sibling Fish codec. Here the decoder tables are uniformly
    // 2048 (checked against the checkpoint, not the config), so 2047 is the true edge —
    // and the reference bounds nothing above, it only clamps `min=0`.
    for tag in ["codec", "codec_edge"] {
        let codes_t = refs.get(&format!("{tag}.codes")).expect("fixture codes");
        let want = refs.get(&format!("{tag}.wav")).expect("fixture wav");

        let (frames, groups) = (codes_t.dim(0).unwrap(), codes_t.dim(1).unwrap());
        let host: Vec<i64> = codes_t.flatten_all().unwrap().to_vec1().unwrap();
        // The fixture is [frames, groups]; the RVQ stacks want one row per group.
        let rows: Vec<Vec<u32>> = (0..groups)
            .map(|g| (0..frames).map(|t| host[t * groups + g] as u32).collect())
            .collect();

        let sem = syrinx_qwen::codec::rvq::Rvq::load(&w, "decoder.quantizer.rvq_first", 1)
            .expect("rvq_first");
        let ac = syrinx_qwen::codec::rvq::Rvq::load(&w, "decoder.quantizer.rvq_rest", rows.len() - 1)
            .expect("rvq_rest");
        let z_sem = sem.decode(&w, &rows[..1], dt).expect("semantic decode");
        let z_ac = ac.decode(&w, &rows[1..], dt).expect("acoustic decode");
        let z = (z_sem + z_ac).expect("sum");
        let got = dec.decode(&w, &z).expect("decoder");

        assert_eq!(
            got.elem_count(),
            want.elem_count(),
            "{tag}: decoded {} samples, reference produced {}",
            got.elem_count(),
            want.elem_count()
        );
        let diff = max_abs_diff(&got, want).expect("diff");
        eprintln!(
            "[qwen-stack] codec {tag}: {frames}x{groups} codes -> {} samples, max abs diff \
             {diff:.6} (tol {tol})",
            want.elem_count()
        );
        assert!(
            diff <= tol,
            "{tag}: decoded waveform differs from the reference by {diff} (tolerance {tol}). \
             No language model is in this path, so this is RVQ or the decoder stack."
        );
    }
}
