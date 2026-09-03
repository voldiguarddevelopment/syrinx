//! Multi-step parity for the Qwen3-TTS **generation loop**, by greedy decoding.
//!
//! `real_qwen_prompt_parity.rs` and `real_qwen_stack_parity.rs` anchor four points of this
//! port — the prompt, the talker's prefill logits, the code predictor's first-step logits
//! and the codec decode. Every one of them is a SINGLE step, and between them they leave
//! the loop itself ungated: the talker's KV cache and position advancement across frames,
//! the per-frame reset and RoPE walk of the code predictor, the talker->predictor handoff
//! (`past_hidden` + group 0's talker-side embedding), the feedback sum of all sixteen code
//! embeddings, the trailing-text schedule and its exhaustion into `tts_pad_embed`, the
//! `min_new_tokens` EOS guard and the `suppress_tokens` block are exercised once each at
//! most. A drift that first shows up at frame 5 is invisible to all four anchors, and the
//! `silu` that `text_projection` was missing until 2026-09-03 is the standing proof that a
//! defect here can be numerically small, audibly invisible, and still wreck behaviour.
//!
//! The reason that gap stayed open is that generation SAMPLES: two PRNGs are never
//! bit-comparable, so a sampled run cannot be diffed against the reference at all.
//! **Greedy decoding closes it.** With `do_sample=False` on both heads the whole loop is a
//! deterministic function of `(weights, prompt)` on both sides, and the generated integer
//! code matrix can be compared exactly — no tolerance, no correlation, just equality.
//!
//! ## What greedy does and does not switch off
//!
//! `transformers.generation.utils._get_logits_processor` installs the temperature / top-k /
//! top-p warpers only under `if generation_config.do_sample:`, and `_sample` then takes
//! `torch.argmax(next_token_scores)`. Everything ABOVE that switch still runs, which is the
//! part worth stating out loud: the talker keeps `repetition_penalty = 1.05`, keeps masking
//! the codec EOS until `min_new_tokens = 2`, and keeps suppressing the top-1024 control
//! block; the code predictor gets none of the three, because
//! `Qwen3TTSTalkerForConditionalGeneration.forward` forwards only the four `subtalker_*`
//! knobs and `code_predictor_config` ships `repetition_penalty: 1.0`, `suppress_tokens:
//! null`, `min_length: 0`. So this test is not "the loop with the sampler removed" — the
//! whole processor list is still in the path, and an asymmetry between the two heads is
//! one of the things it can catch.
//!
//! On the Rust side that is [`syrinx_qwen::sampling::Sampler::greedy`], reached through
//! `DriveParams::greedy`. It is deliberately NOT `top_k = 1`: `top_k = 1` keeps every tied
//! maximum and lets the PRNG pick between them, where `torch.argmax` always takes the
//! first. That difference is rare, not impossible, and a parity fixture must not rest on
//! "rare".
//!
//! ## Three cases, because one does not reach the whole loop
//!
//! | case | prompt | trailing rows | what only it reaches |
//! |---|---|---|---|
//! | `plain` | 21 | 1 | the ordinary CustomVoice path |
//! | `instruct` | 28 | 1 | the instruct block — the path the `silu` bug destroyed |
//! | `streaming` | 10 | 10 | the per-frame trailing-text schedule |
//!
//! `streaming` is not a nicety. In non-streaming mode the reference sets
//! `trailing_text_hidden = tts_pad_embed`, exactly one row, so every frame after the first
//! takes the pad and a loop that read row `min(step, len-1)`, or row `0` forever, or never
//! advanced at all would pass. `non_streaming_mode=False` moves the text tail into
//! `trailing_text_hidden`, giving ten rows consumed over forty-four frames — both sides of
//! `step < trailing_len`, in order, forty-three times.
//!
//! ## CPU / float32, enforced rather than preferred
//!
//! This test **ignores `SYRINX_QWEN_DEVICE` and always runs on CPU/f32**, which is the one
//! place it deliberately departs from its sibling parity tests. Greedy decoding is an
//! argmax chain: a numeric difference far below any tolerance those tests would accept can
//! flip a single code, and from that frame on the two runs are generating different audio.
//! The reference computing the same conv-heavy path on CUDA versus CPU already disagrees
//! with ITSELF by ~0.03, which is orders of magnitude more than enough to do it. CPU/f32 on
//! both sides is the only fair comparison and is this port's designated parity path.
//!
//! Fixture: `scripts/gen-qwen-ref-greedy.py` (drives the reference's own
//! `generate_custom_voice`, refuses to write a fixture that is not greedy). Gated on
//! `SYRINX_QWEN_CV_DIR` and `SYRINX_QWEN_REF_GREEDY`; skips cleanly without them.

#![cfg(feature = "real")]

use candle_core::{DType, Device, Tensor};

/// The probe case. Identical strings in `scripts/gen-qwen-ref-greedy.py` — changing either
/// side alone makes this compare two different utterances.
const TEXT: &str = "Come closer, I have something to tell you.";
const SPEAKER: &str = "serena";
/// The reference is handed `"English"`; the Rust prompt builder lowercases its lookup.
const LANGUAGE: &str = "english";
const INSTRUCT: &str = "Whisper";

/// `(tag, instruct, non_streaming)` — must match `CASES` in the generator.
const CASES: [(&str, Option<&str>, bool); 3] = [
    ("plain", None, true),
    ("instruct", Some(INSTRUCT), true),
    ("streaming", None, false),
];

/// Tolerance for the realized prompt, which is a precondition of the loop comparison rather
/// than its subject: if the prompt already differs, a divergent code matrix says nothing
/// about the loop. Derived from measurement, not chosen — `real_qwen_prompt_parity.rs`
/// records 0.00000 (bit-exact) for the f32 path, and this test measures the same on all
/// three cases; the bound sits well above that so an ordinary f32 reassociation does not
/// trip it while the 0.781 of the `silu` bug would.
const TOL_PROMPT_F32: f32 = 1e-3;

/// Frames of headroom over the reference's own frame count.
///
/// Load-bearing: driving the loop with `max_new_frames = frames` would truncate a port that
/// never emits the EOS to exactly the right length and let it pass. With headroom, an
/// over-generating port produces more frames than the fixture and the length assertion
/// catches it.
const FRAME_HEADROOM: usize = 8;

fn env_path(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|p| std::path::Path::new(p).exists())
}

/// `(fixture, checkpoint dir)`, or `None` with a SKIP printed.
fn setup(name: &str) -> Option<(std::collections::HashMap<String, Tensor>, String)> {
    let Some(dir) = env_path("SYRINX_QWEN_CV_DIR") else {
        eprintln!("SKIP {name}: set SYRINX_QWEN_CV_DIR to a CustomVoice checkpoint dir");
        return None;
    };
    let Some(ref_path) = env_path("SYRINX_QWEN_REF_GREEDY") else {
        eprintln!(
            "SKIP {name}: set SYRINX_QWEN_REF_GREEDY to the fixture from \
             scripts/gen-qwen-ref-greedy.py"
        );
        return None;
    };
    if std::env::var("SYRINX_QWEN_DEVICE").is_ok() {
        eprintln!(
            "[qwen-greedy] NOTE: SYRINX_QWEN_DEVICE is set and is deliberately ignored — an \
             argmax chain cannot survive cross-device drift, so this gate is CPU/f32 only."
        );
    }
    let refs = candle_core::safetensors::load(&ref_path, &Device::Cpu).expect("load fixture");
    Some((refs, dir))
}

/// The nine int64 fields `gen-qwen-ref-greedy.py` records alongside each case.
struct Meta {
    frames: usize,
    groups: usize,
    stopped_on_eos: bool,
    eos_id: u32,
    cap: usize,
    min_new_tokens: usize,
    n_suppress: usize,
    non_streaming: bool,
    has_instruct: bool,
}

fn meta(refs: &std::collections::HashMap<String, Tensor>, tag: &str) -> Meta {
    let v: Vec<i64> = refs
        .get(&format!("{tag}.meta"))
        .unwrap_or_else(|| panic!("fixture: {tag}.meta"))
        .to_vec1()
        .expect("meta to_vec1");
    assert_eq!(v.len(), 9, "{tag}.meta has {} fields, expected 9", v.len());
    Meta {
        frames: v[0] as usize,
        groups: v[1] as usize,
        stopped_on_eos: v[2] == 1,
        eos_id: v[3] as u32,
        cap: v[4] as usize,
        min_new_tokens: v[5] as usize,
        n_suppress: v[6] as usize,
        non_streaming: v[7] == 1,
        has_instruct: v[8] == 1,
    }
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    let d = (a.to_dtype(DType::F32).unwrap().flatten_all().unwrap()
        - b.to_dtype(DType::F32).unwrap().flatten_all().unwrap())
    .unwrap()
    .abs()
    .unwrap();
    d.max(0).unwrap().to_scalar::<f32>().unwrap()
}

fn build(
    tok: &syrinx_qwen::tokenizer::QwenTokenizer,
    pcfg: &syrinx_qwen::prompt::PromptConfig,
    instruct: Option<&str>,
    non_streaming: bool,
) -> syrinx_qwen::prompt::PromptPlan {
    syrinx_qwen::prompt::build_custom_voice(
        tok,
        pcfg,
        TEXT,
        SPEAKER,
        instruct,
        LANGUAGE,
        non_streaming,
    )
    .expect("build prompt")
}

/// The precondition: the prompt handed to the loop is the reference's, in all three cases.
///
/// `plain` and `instruct` overlap `real_qwen_prompt_parity.rs` on purpose — if this file's
/// loop comparison fails, the first question is whether the input was already wrong, and
/// answering it here rather than in another test file is the difference between "the loop
/// diverges at frame 7" and "something is off somewhere". `streaming` is new coverage: no
/// other test builds a prompt with a multi-row `trailing_text_hidden`.
#[test]
fn qwen_greedy_prompts_match_the_reference() {
    let Some((refs, dir)) = setup("real_qwen_greedy_parity::prompts") else { return };
    let dev = Device::Cpu;

    let cfg_json = std::fs::read_to_string(std::path::Path::new(&dir).join("config.json"))
        .expect("config.json");
    let pcfg = syrinx_qwen::prompt::PromptConfig::from_json(&cfg_json).expect("prompt config");
    let tok = syrinx_qwen::tokenizer::QwenTokenizer::from_dir(&dir).expect("tokenizer");
    let model = syrinx_qwen::model::Qwen3Tts::load(&dir, dev).expect("load talker");

    for (tag, instruct, non_streaming) in CASES {
        let m = meta(&refs, tag);
        assert_eq!(m.non_streaming, non_streaming, "{tag}: fixture disagrees on the mode");
        assert_eq!(m.has_instruct, instruct.is_some(), "{tag}: fixture disagrees on the instruct");

        let plan = build(&tok, &pcfg, instruct, non_streaming);
        let got = model.realize_plan(&plan, None, &[]).expect("realize");

        for (what, got_t, key) in [
            ("inputs_embeds", &got.inputs_embeds, format!("{tag}.inputs_embeds")),
            (
                "trailing_text_hidden",
                &got.trailing_text_hidden,
                format!("{tag}.trailing_text_hidden"),
            ),
        ] {
            let want = refs.get(&key).unwrap_or_else(|| panic!("fixture: {key}"));
            let got_t = got_t.squeeze(0).expect("drop batch");
            assert_eq!(
                got_t.dims(),
                want.dims(),
                "{tag} {what}: shape {:?}, reference {:?}",
                got_t.dims(),
                want.dims()
            );
            let diff = max_abs_diff(&got_t, want);
            eprintln!(
                "[qwen-greedy] {tag:9} {what:20} {:?}  max abs diff {diff:.6} \
                 (tol {TOL_PROMPT_F32})",
                want.dims()
            );
            assert!(
                diff <= TOL_PROMPT_F32,
                "{tag} {what} differs from the reference by {diff} (tolerance \
                 {TOL_PROMPT_F32}). The loop comparison downstream would be meaningless \
                 until this is fixed — the fault is in the prompt, not the loop."
            );
        }
    }
}

/// The gate: run the loop greedily and require the generated code matrix to equal the
/// reference's, frame for frame and group for group.
#[test]
fn qwen_greedy_generation_matches_the_reference() {
    let Some((refs, dir)) = setup("real_qwen_greedy_parity::generation") else { return };
    let dev = Device::Cpu;

    let cfg_json = std::fs::read_to_string(std::path::Path::new(&dir).join("config.json"))
        .expect("config.json");
    let pcfg = syrinx_qwen::prompt::PromptConfig::from_json(&cfg_json).expect("prompt config");
    let tok = syrinx_qwen::tokenizer::QwenTokenizer::from_dir(&dir).expect("tokenizer");
    let mut model = syrinx_qwen::model::Qwen3Tts::load(&dir, dev).expect("load talker");

    let mut failures: Vec<String> = Vec::new();

    for (tag, instruct, non_streaming) in CASES {
        let m = meta(&refs, tag);

        // The reference's own knobs, cross-checked rather than assumed. If the checkpoint
        // or the reference ever changes one of these, the run below is no longer the run
        // the fixture describes and the diff would be blamed on the loop.
        assert_eq!(
            model.config().codec_eos_token_id,
            m.eos_id,
            "{tag}: this port's codec EOS is not the reference's"
        );
        assert_eq!(
            model.suppressed_ids().len(),
            m.n_suppress,
            "{tag}: the suppress block is {} ids here and {} in the reference",
            model.suppressed_ids().len(),
            m.n_suppress
        );
        assert!(
            !model.suppressed_ids().contains(&m.eos_id),
            "{tag}: the EOS is inside the suppress block, so the run could never stop"
        );

        let want_t = refs
            .get(&format!("{tag}.codes"))
            .unwrap_or_else(|| panic!("fixture: {tag}.codes"));
        let want_flat: Vec<i64> = want_t.flatten_all().unwrap().to_vec1().unwrap();
        let groups = model.config().num_code_groups;
        assert_eq!(m.groups, groups, "{tag}: fixture has {} groups, port has {groups}", m.groups);
        assert_eq!(want_flat.len(), m.frames * groups);
        let want: Vec<Vec<u32>> = (0..m.frames)
            .map(|t| (0..groups).map(|g| want_flat[t * groups + g] as u32).collect())
            .collect();

        let plan = build(&tok, &pcfg, instruct, non_streaming);
        let prompt = model.realize_plan(&plan, None, &[]).expect("realize");
        let trailing_rows = prompt.trailing_text_hidden.dim(1).unwrap();

        let params = syrinx_qwen::model::DriveParams {
            greedy: true,
            // Headroom, so an over-generating port is caught by the length check instead of
            // being trimmed into agreement.
            max_new_frames: m.frames + FRAME_HEADROOM,
            min_new_frames: m.min_new_tokens,
            // Under greedy the warper fields are ignored; `repetition_penalty` is NOT, and
            // these are the shipped values the reference runs (talker 1.05, predictor 1.0).
            ..Default::default()
        };
        assert!(
            params.max_new_frames <= m.cap,
            "{tag}: the reference was capped at {} frames, so a run allowed {} could not be \
             compared past the cap",
            m.cap,
            params.max_new_frames
        );

        let t0 = std::time::Instant::now();
        let got = model.generate(&prompt, &params).expect("generate");
        let secs = t0.elapsed().as_secs_f32();

        // ---- report before asserting: "diverges at frame 7" beats a bare pass/fail ------
        let common = got.frames.len().min(want.len());
        let mut first: Option<(usize, usize, u32, u32)> = None;
        let mut bad_codes = 0usize;
        let mut agreeing_frames = 0usize;
        for t in 0..common {
            let mut ok = true;
            for g in 0..groups {
                if got.frames[t][g] != want[t][g] {
                    bad_codes += 1;
                    ok = false;
                    if first.is_none() {
                        first = Some((t, g, got.frames[t][g], want[t][g]));
                    }
                }
            }
            if ok {
                agreeing_frames += 1;
            }
        }
        let leading = first.map(|(t, _, _, _)| t).unwrap_or(common);

        eprintln!(
            "[qwen-greedy] {tag:9} prompt {} steps, trailing {trailing_rows} rows -> \
             {} frames in {secs:.1}s (reference {} frames, stopped_on_eos \
             port={} ref={})",
            plan.len(),
            got.frames.len(),
            m.frames,
            got.stopped_on_eos,
            m.stopped_on_eos,
        );
        eprintln!(
            "[qwen-greedy] {tag:9} {leading}/{} leading frames identical, \
             {agreeing_frames}/{common} frames identical overall, {bad_codes}/{} codes differ",
            m.frames,
            common * groups,
        );
        match first {
            None => eprintln!("[qwen-greedy] {tag:9} no divergence in the common prefix"),
            Some((t, g, a, b)) => eprintln!(
                "[qwen-greedy] {tag:9} FIRST DIVERGENCE frame {t} group {g}: port {a}, \
                 reference {b}\n                        port  frame {t} = {:?}\n\
                 \x20                       ref   frame {t} = {:?}",
                got.frames[t], want[t]
            ),
        }

        // ---- and now the gate: exact integer agreement, no tolerance to loosen ----------
        if let Some((t, g, a, b)) = first {
            failures.push(format!(
                "{tag}: diverges at frame {t} group {g} (port {a}, reference {b}); \
                 {leading} leading frames were identical, {bad_codes} codes differ in the \
                 common {common}-frame prefix"
            ));
        }
        if got.frames.len() != m.frames {
            failures.push(format!(
                "{tag}: produced {} frames, the reference produced {}",
                got.frames.len(),
                m.frames
            ));
        }
        if got.stopped_on_eos != m.stopped_on_eos {
            failures.push(format!(
                "{tag}: stopped_on_eos = {}, the reference {}",
                got.stopped_on_eos, m.stopped_on_eos
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "the greedy loop does not reproduce the reference:\n  {}\n\nBoth sides ran CPU/f32 \
         with do_sample=False on both heads, so there is no sampling, no device drift and \
         no tolerance in this comparison — a difference is a real difference in the loop \
         (KV cache, position advance, the talker->predictor handoff, the feedback sum, the \
         trailing-text schedule, the EOS guard or suppress_tokens).",
        failures.join("\n  ")
    );
}
