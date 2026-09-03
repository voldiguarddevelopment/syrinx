//! `syrinx-qwen` — the geometry contract, checked against the **real published**
//! Qwen3-TTS `config.json` files.
//!
//! **What is certified here:** that `Qwen3TtsConfig::from_json` reads the five shipped
//! Apache-2.0 checkpoints correctly, that every derived quantity (`code_predictor_heads`,
//! the attention widths, the talker/predictor width split) matches what those files
//! actually say, and that every silent-default and every rejection path fires on the
//! right side of its boundary.
//!
//! **What is not:** anything numeric. No tensor is read, no forward pass runs, nothing
//! here says the port *synthesises* correctly. See `docs/backends/QWEN_PORT_STATUS.md`
//! for the weights/CPU/GPU boundary.
//!
//! The fixtures under `tests/golden/qwen/config/` are the published files byte-for-byte
//! (`scripts/gen-qwen-index.py`), so this runs on any box — no weights, no GPU, no
//! Python. Model-free by construction: it must never SKIP.
//!
//! This exists because `crates/syrinx-qwen/src/config.rs` says the sibling Fish s1 port
//! hand-wrote this geometry and got seven of nine fields wrong. A unit test inside the
//! crate does not appear on the `scripts/verify.sh` board — repo-root `tests/*.rs` is
//! the only path the harness runs.

use syrinx_qwen::config::{Qwen3TtsConfig, QwenVariant};

// ---------------------------------------------------------------------------- fixtures

const BASE_0B6: &str = include_str!("golden/qwen/config/Qwen3-TTS-12Hz-0.6B-Base.json");
const CUSTOM_0B6: &str = include_str!("golden/qwen/config/Qwen3-TTS-12Hz-0.6B-CustomVoice.json");
const BASE_1B7: &str = include_str!("golden/qwen/config/Qwen3-TTS-12Hz-1.7B-Base.json");
const CUSTOM_1B7: &str = include_str!("golden/qwen/config/Qwen3-TTS-12Hz-1.7B-CustomVoice.json");
const DESIGN_1B7: &str = include_str!("golden/qwen/config/Qwen3-TTS-12Hz-1.7B-VoiceDesign.json");

/// Every published checkpoint: `(name, config.json, variant)`.
pub const ALL: &[(&str, &str, QwenVariant)] = &[
    ("0.6B-Base", BASE_0B6, QwenVariant::Base),
    ("0.6B-CustomVoice", CUSTOM_0B6, QwenVariant::CustomVoice),
    ("1.7B-Base", BASE_1B7, QwenVariant::Base),
    ("1.7B-CustomVoice", CUSTOM_1B7, QwenVariant::CustomVoice),
    ("1.7B-VoiceDesign", DESIGN_1B7, QwenVariant::VoiceDesign),
];

fn parse(name: &str, json: &str) -> Qwen3TtsConfig {
    Qwen3TtsConfig::from_json(json).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// Parse a fixture into a mutable `serde_json::Value` so a test can remove or change one
/// key and re-serialize. String surgery on the raw file would be silently order- and
/// whitespace-dependent.
fn edited(json: &str, f: impl FnOnce(&mut serde_json::Value)) -> String {
    let mut v: serde_json::Value = serde_json::from_str(json).expect("fixture is valid JSON");
    f(&mut v);
    v.to_string()
}

fn talker_of(v: &mut serde_json::Value) -> &mut serde_json::Map<String, serde_json::Value> {
    v.get_mut("talker_config")
        .and_then(|t| t.as_object_mut())
        .expect("fixture has talker_config")
}

// ------------------------------------------------------------------ the published files

/// Every shipped checkpoint parses, and the variant is DERIVED from the file rather than
/// assumed from the directory name.
#[test]
fn every_published_checkpoint_parses_with_the_right_variant() {
    for (name, json, want) in ALL {
        let c = parse(name, json);
        assert_eq!(c.variant, *want, "{name}: variant");
        // Parsing is a pure function of the bytes.
        assert_eq!(c, parse(name, json), "{name}: parse is not deterministic");
    }
}

/// The 0.6B talker: 28 layers at width 1024, and a code predictor that is exactly as wide
/// as the talker — the case where `small_to_mtp_projection` must NOT be expected.
#[test]
fn the_0b6_geometry_is_the_published_one() {
    let c = parse("0.6B-Base", BASE_0B6);
    let t = &c.talker;
    assert_eq!(t.hidden_size, 1024);
    assert_eq!(t.num_hidden_layers, 28);
    assert_eq!(t.num_attention_heads, 16);
    assert_eq!(t.num_key_value_heads, 8);
    assert_eq!(t.head_dim, 128);
    assert_eq!(t.intermediate_size, 3072);
    assert_eq!(t.vocab_size, 3072);
    assert_eq!(t.rope_theta, 1_000_000.0);
    assert_eq!(t.rms_norm_eps, 1e-6);
    assert_eq!(t.max_position_embeddings, 32_768);

    let p = &c.code_predictor;
    assert_eq!(p.hidden_size, 1024);
    assert_eq!(p.num_hidden_layers, 5);
    assert_eq!(p.intermediate_size, 3072);
    assert_eq!(p.vocab_size, 2048);
    assert_eq!(
        t.hidden_size, p.hidden_size,
        "the 0.6B is the equal-width case; the width bridge must be absent"
    );
    // The 0.6B CustomVoice shares the talker geometry exactly — only the prompt-side
    // tables differ — so a geometry regression cannot hide in one of the two.
    assert_eq!(parse("0.6B-CustomVoice", CUSTOM_0B6).talker, *t);
}

/// The 1.7B talker: same depth, twice the width, and a predictor left at 1024 — the case
/// where the width bridge MUST be expected. Same assertions as the 0.6B, opposite answer
/// on the one comparison that matters.
#[test]
fn the_1b7_geometry_is_the_published_one() {
    for (name, json) in [
        ("1.7B-Base", BASE_1B7),
        ("1.7B-CustomVoice", CUSTOM_1B7),
        ("1.7B-VoiceDesign", DESIGN_1B7),
    ] {
        let c = parse(name, json);
        let t = &c.talker;
        assert_eq!(t.hidden_size, 2048, "{name}");
        assert_eq!(t.num_hidden_layers, 28, "{name}");
        assert_eq!(t.num_attention_heads, 16, "{name}");
        assert_eq!(t.num_key_value_heads, 8, "{name}");
        assert_eq!(t.head_dim, 128, "{name}");
        assert_eq!(t.intermediate_size, 6144, "{name}");
        assert_eq!(t.vocab_size, 3072, "{name}");

        let p = &c.code_predictor;
        assert_eq!(p.hidden_size, 1024, "{name}: predictor stays narrow");
        assert_eq!(p.num_hidden_layers, 5, "{name}");
        assert_eq!(p.intermediate_size, 3072, "{name}");
        assert_eq!(p.vocab_size, 2048, "{name}");
        assert_ne!(
            t.hidden_size, p.hidden_size,
            "{name}: the 1.7B is the UNEQUAL-width case; the width bridge must be present"
        );
    }
}

/// The attention projection widths implied by the config, on both sizes. `q_proj` is
/// `heads * head_dim` (2048 / 2048), `k`/`v` are `kv_heads * head_dim` (1024 / 1024), and
/// `o_proj` lands back on `hidden_size` (1024 / 2048) — so on the 0.6B the query width is
/// TWICE the model width, and on the 1.7B it is equal. A head-count slip breaks one or
/// the other.
#[test]
fn attention_widths_follow_from_the_head_counts() {
    let small = parse("0.6B-Base", BASE_0B6).talker;
    assert_eq!(small.num_attention_heads * small.head_dim, 2048);
    assert_eq!(small.num_key_value_heads * small.head_dim, 1024);
    assert_ne!(
        small.num_attention_heads * small.head_dim,
        small.hidden_size,
        "0.6B: q_proj is wider than the residual stream"
    );

    let big = parse("1.7B-Base", BASE_1B7).talker;
    assert_eq!(big.num_attention_heads * big.head_dim, 2048);
    assert_eq!(big.num_key_value_heads * big.head_dim, 1024);
    assert_eq!(
        big.num_attention_heads * big.head_dim,
        big.hidden_size,
        "1.7B: q_proj is exactly the residual stream"
    );

    // Grouped-query attention on both: strictly fewer KV heads than Q heads, and the
    // repeat factor is exact.
    for (name, json) in [("0.6B", BASE_0B6), ("1.7B", BASE_1B7)] {
        let t = parse(name, json).talker;
        assert!(t.num_key_value_heads < t.num_attention_heads, "{name}: GQA");
        assert_eq!(t.num_attention_heads % t.num_key_value_heads, 0, "{name}: repeat_kv");
        assert_eq!(t.num_attention_heads / t.num_key_value_heads, 2, "{name}");
    }
}

/// Group 0 comes from the talker's `codec_head`, so the predictor carries exactly
/// `num_code_groups - 1` heads. Pinned on every checkpoint, then on both sides of the
/// `saturating_sub(1)` boundary.
#[test]
fn code_predictor_heads_is_groups_minus_one() {
    for (name, json, _) in ALL {
        let c = parse(name, json);
        assert_eq!(c.num_code_groups, 16, "{name}");
        assert_eq!(c.code_predictor_heads(), 15, "{name}");
    }

    let two = parse("g=2", &edited(BASE_0B6, |v| {
        talker_of(v).insert("num_code_groups".into(), serde_json::json!(2));
    }));
    assert_eq!(two.code_predictor_heads(), 1, "2 groups -> 1 residual head");

    // One group means the talker's own head is the only one: zero residual heads, not -1.
    let one = parse("g=1", &edited(BASE_0B6, |v| {
        talker_of(v).insert("num_code_groups".into(), serde_json::json!(1));
    }));
    assert_eq!(one.code_predictor_heads(), 0);

    // …and the saturating subtraction must not wrap at zero.
    let zero = parse("g=0", &edited(BASE_0B6, |v| {
        talker_of(v).insert("num_code_groups".into(), serde_json::json!(0));
    }));
    assert_eq!(zero.code_predictor_heads(), 0, "saturating, not wrapping");
}

/// `num_code_groups` is looked up on the talker FIRST and only then on the code predictor.
/// Both sides: present on the talker (used), absent on the talker but present on the
/// predictor (fallback used), absent from both (hard error).
#[test]
fn num_code_groups_falls_back_to_the_code_predictor_then_fails() {
    // The published files carry it in BOTH places, so the fallback is invisible there;
    // make the two disagree to prove which one wins.
    let talker_wins = parse("disagreeing", &edited(BASE_0B6, |v| {
        let t = talker_of(v);
        t.insert("num_code_groups".into(), serde_json::json!(8));
        t.get_mut("code_predictor_config")
            .and_then(|c| c.as_object_mut())
            .unwrap()
            .insert("num_code_groups".into(), serde_json::json!(4));
    }));
    assert_eq!(talker_wins.num_code_groups, 8, "the talker's value takes precedence");

    let fallback = parse("cp-only", &edited(BASE_0B6, |v| {
        let t = talker_of(v);
        t.remove("num_code_groups");
        t.get_mut("code_predictor_config")
            .and_then(|c| c.as_object_mut())
            .unwrap()
            .insert("num_code_groups".into(), serde_json::json!(4));
    }));
    assert_eq!(fallback.num_code_groups, 4, "falls back to code_predictor_config");

    let err = Qwen3TtsConfig::from_json(&edited(BASE_0B6, |v| {
        let t = talker_of(v);
        t.remove("num_code_groups");
        t.get_mut("code_predictor_config")
            .and_then(|c| c.as_object_mut())
            .unwrap()
            .remove("num_code_groups");
    }))
    .unwrap_err();
    assert!(err.contains("no `num_code_groups`"), "got {err}");
}

/// The text stream is 2048 wide and is projected DOWN to the talker width — on the 0.6B
/// those are different numbers (2048 vs 1024), which is the whole reason the field exists
/// separately. Read from `text_hidden_size`, the key the published files actually carry.
#[test]
fn the_text_width_comes_from_text_hidden_size() {
    let small = parse("0.6B-Base", BASE_0B6);
    assert_eq!(small.text_embed_dim, 2048);
    assert_eq!(small.text_vocab_size, 151_936);
    assert_ne!(
        small.text_embed_dim, small.talker.hidden_size,
        "0.6B: text_projection genuinely narrows"
    );
    let big = parse("1.7B-Base", BASE_1B7);
    assert_eq!(big.text_embed_dim, 2048);
    assert_eq!(
        big.text_embed_dim, big.talker.hidden_size,
        "1.7B: text_projection is width-preserving"
    );

    // The file's value is used, not a constant that happens to agree.
    let wide = parse("wide", &edited(BASE_0B6, |v| {
        talker_of(v).insert("text_hidden_size".into(), serde_json::json!(4096));
    }));
    assert_eq!(wide.text_embed_dim, 4096);

    // No published config carries `text_embed_dim`; it is a last-resort alias only, so
    // when both keys are present the file's real key must win. (An earlier revision of
    // the parser read only the alias and silently fell back to a default that happened to
    // be correct — the exact silent-default failure the s1 port shipped.)
    let both = parse("both", &edited(BASE_0B6, |v| {
        talker_of(v).insert("text_embed_dim".into(), serde_json::json!(999));
    }));
    assert_eq!(both.text_embed_dim, 2048, "`text_hidden_size` outranks the alias");
    let alias = parse("alias", &edited(BASE_0B6, |v| {
        let t = talker_of(v);
        t.remove("text_hidden_size");
        t.insert("text_embed_dim".into(), serde_json::json!(1536));
    }));
    assert_eq!(alias.text_embed_dim, 1536);

    // Neither key present: the hard-coded fallback, which is a silent default and is
    // pinned here precisely so a future upstream rename is visible.
    let defaulted = parse("defaulted", &edited(BASE_0B6, |v| {
        talker_of(v).remove("text_hidden_size");
    }));
    assert_eq!(defaulted.text_embed_dim, 2048);
    assert_eq!(
        parse("vocab-defaulted", &edited(BASE_0B6, |v| { talker_of(v).remove("text_vocab_size"); }))
            .text_vocab_size,
        151_936
    );
}

/// The special-token ids every checkpoint agrees on. An id off by one here is the exact
/// Fish failure mode (`tokenizer.rs`: the model emitted its stop token after three
/// frames), so they are pinned literally on all five files.
#[test]
fn the_special_ids_are_identical_across_every_checkpoint() {
    for (name, json, _) in ALL {
        let c = parse(name, json);
        assert_eq!(c.im_start_token_id, 151_644, "{name}");
        assert_eq!(c.im_end_token_id, 151_645, "{name}");
        assert_eq!(c.tts_pad_token_id, 151_671, "{name}");
        assert_eq!(c.tts_bos_token_id, 151_672, "{name}");
        assert_eq!(c.tts_eos_token_id, 151_673, "{name}");
        // The codec-side control ids come out of `talker_config`, and none of them is the
        // 0 the parser would fall back to.
        assert_eq!(c.codec_pad_id, 2148, "{name}");
        assert_eq!(c.codec_bos_id, 2149, "{name}");
        assert_eq!(c.codec_eos_token_id, 2150, "{name}");
        assert_ne!(c.codec_bos_id, 0, "{name}: not the fallback");
        // The EOS is inside the talker's vocabulary but outside the code predictor's, so
        // group 0 can stop and the residual groups cannot.
        assert!(c.codec_eos_token_id < c.talker.vocab_size as u32, "{name}");
        assert!(c.codec_eos_token_id >= c.code_predictor.vocab_size as u32, "{name}");
    }
}

/// `speaker_encoder_config` is present only on the two clone-capable checkpoints, and the
/// two differ in width — hardcoding 1024 loads the 0.6B and mis-shapes the 1.7B.
#[test]
fn the_speaker_encoder_width_differs_between_the_two_base_checkpoints() {
    let small = parse("0.6B-Base", BASE_0B6);
    assert_eq!(small.speaker_enc_dim, 1024);
    assert_eq!(small.sample_rate, 24_000);
    let big = parse("1.7B-Base", BASE_1B7);
    assert_eq!(big.speaker_enc_dim, 2048);
    assert_eq!(big.sample_rate, 24_000);
    assert_ne!(small.speaker_enc_dim, big.speaker_enc_dim);

    // The x-vector is written straight into the codec stream, so it must be exactly the
    // talker's width on the checkpoint that owns it.
    assert_eq!(small.speaker_enc_dim, small.talker.hidden_size);
    assert_eq!(big.speaker_enc_dim, big.talker.hidden_size);

    // On the non-clone checkpoints the block is absent and the field is a meaningless
    // default. Pinned so nobody reads it as a real width for those.
    for (name, json) in [("0.6B-CustomVoice", CUSTOM_0B6), ("1.7B-VoiceDesign", DESIGN_1B7)] {
        let c = parse(name, json);
        assert!(!c.variant.supports_voice_clone(), "{name}");
        assert_eq!(c.speaker_enc_dim, 1024, "{name}: the default, not a real width");
        assert_eq!(c.sample_rate, 24_000, "{name}: the default");
    }
}

/// The optional numeric defaults are fallbacks, not the source of truth: present in every
/// published file, and used only when absent.
#[test]
fn the_optional_numerics_are_read_before_they_are_defaulted() {
    let odd = parse("odd", &edited(BASE_0B6, |v| {
        let t = talker_of(v);
        t.insert("rope_theta".into(), serde_json::json!(500_000.0));
        t.insert("rms_norm_eps".into(), serde_json::json!(1e-5));
        t.insert("max_position_embeddings".into(), serde_json::json!(4096));
    }));
    assert_eq!(odd.talker.rope_theta, 500_000.0);
    assert_eq!(odd.talker.rms_norm_eps, 1e-5);
    assert_eq!(odd.talker.max_position_embeddings, 4096);

    let bare = parse("bare", &edited(BASE_0B6, |v| {
        let t = talker_of(v);
        t.remove("rope_theta");
        t.remove("rms_norm_eps");
        t.remove("max_position_embeddings");
    }));
    assert_eq!(bare.talker.rope_theta, 1_000_000.0);
    assert_eq!(bare.talker.rms_norm_eps, 1e-6);
    assert_eq!(bare.talker.max_position_embeddings, 32_768);
}

// ------------------------------------------------------------------ variant capabilities

/// The capability split that decides what a tagged corpus can be ported to: `Base` clones
/// but takes no instruction; `CustomVoice`/`VoiceDesign` take an instruction but do not
/// clone. Both predicates asserted true AND false for every variant.
#[test]
fn the_variant_capability_matrix_is_exclusive() {
    let table = [
        (QwenVariant::Base, true, false),
        (QwenVariant::CustomVoice, false, true),
        (QwenVariant::VoiceDesign, false, true),
    ];
    for (variant, clone, instruct) in table {
        assert_eq!(variant.supports_voice_clone(), clone, "{variant:?}: clone");
        assert_eq!(variant.supports_instruct(), instruct, "{variant:?}: instruct");
        // No variant does both, and none does neither.
        assert!(
            variant.supports_voice_clone() != variant.supports_instruct(),
            "{variant:?}: the two capabilities must be exclusive"
        );
    }
    // …and the published files land on the right row.
    for (name, json, want) in ALL {
        let c = parse(name, json);
        let (_, clone, instruct) = table.iter().find(|(v, _, _)| v == want).unwrap();
        assert_eq!(c.variant.supports_voice_clone(), *clone, "{name}");
        assert_eq!(c.variant.supports_instruct(), *instruct, "{name}");
    }
}

/// `tts_model_type` parsing: the three published spellings, the normalisations the parser
/// promises (trim, case, `-` → `_`), and the strings just outside each of them.
#[test]
fn model_type_parsing_normalises_only_what_it_claims_to() {
    for (s, want) in [
        ("base", QwenVariant::Base),
        ("  base\n", QwenVariant::Base),
        ("BASE", QwenVariant::Base),
        ("custom_voice", QwenVariant::CustomVoice),
        ("custom-voice", QwenVariant::CustomVoice),
        ("customvoice", QwenVariant::CustomVoice),
        ("CustomVoice", QwenVariant::CustomVoice),
        ("voice_design", QwenVariant::VoiceDesign),
        ("voice-design", QwenVariant::VoiceDesign),
        ("voicedesign", QwenVariant::VoiceDesign),
    ] {
        assert_eq!(QwenVariant::from_model_type(s), Some(want), "{s:?}");
    }
    // Just outside: a space is not a hyphen, an empty string is not a variant, and a
    // near-miss spelling is rejected rather than guessed at.
    for s in ["", "  ", "custom voice", "voice design", "bases", "clone", "base_voice"] {
        assert_eq!(QwenVariant::from_model_type(s), None, "{s:?} must not parse");
    }
}

// -------------------------------------------------------------------------- rejections

/// Everything `from_json` must refuse, each with the message that names the problem.
#[test]
fn a_config_that_is_not_a_qwen3_tts_config_is_rejected() {
    // The sibling backend's own config must not be mistaken for this one.
    let err = Qwen3TtsConfig::from_json(r#"{"model_type":"dual_ar"}"#).unwrap_err();
    assert!(err.contains("not a Qwen3-TTS config"), "got {err}");

    // Absent `model_type` is refused too — an empty string is not a pass.
    let err = Qwen3TtsConfig::from_json(r#"{"tts_model_type":"base"}"#).unwrap_err();
    assert!(err.contains("not a Qwen3-TTS config"), "got {err}");

    // …and the exact string is what admits it.
    let ok = edited(BASE_0B6, |v| {
        v.as_object_mut().unwrap().insert("model_type".into(), serde_json::json!("qwen3_tts"));
    });
    assert!(Qwen3TtsConfig::from_json(&ok).is_ok());
    let near = edited(BASE_0B6, |v| {
        v.as_object_mut().unwrap().insert("model_type".into(), serde_json::json!("qwen3-tts"));
    });
    assert!(Qwen3TtsConfig::from_json(&near).is_err(), "`model_type` is not normalised");

    assert!(Qwen3TtsConfig::from_json("not json at all").unwrap_err().contains("parse config.json"));
}

#[test]
fn a_missing_or_unknown_structural_field_is_named_in_the_error() {
    let cases: Vec<(&str, String)> = vec![
        (
            "unrecognised `tts_model_type`",
            edited(BASE_0B6, |v| {
                v.as_object_mut().unwrap().insert("tts_model_type".into(), serde_json::json!("clone"));
            }),
        ),
        (
            "unrecognised `tts_model_type`",
            edited(BASE_0B6, |v| {
                v.as_object_mut().unwrap().remove("tts_model_type");
            }),
        ),
        (
            "no `talker_config`",
            edited(BASE_0B6, |v| {
                v.as_object_mut().unwrap().remove("talker_config");
            }),
        ),
        (
            "no `code_predictor_config`",
            edited(BASE_0B6, |v| {
                talker_of(v).remove("code_predictor_config");
            }),
        ),
        (
            "talker_config: missing `head_dim`",
            edited(BASE_0B6, |v| {
                talker_of(v).remove("head_dim");
            }),
        ),
        (
            "talker_config: missing `num_hidden_layers`",
            edited(BASE_0B6, |v| {
                talker_of(v).remove("num_hidden_layers");
            }),
        ),
        (
            "code_predictor_config: missing `vocab_size`",
            edited(BASE_0B6, |v| {
                talker_of(v)
                    .get_mut("code_predictor_config")
                    .and_then(|c| c.as_object_mut())
                    .unwrap()
                    .remove("vocab_size");
            }),
        ),
    ];
    for (want, json) in cases {
        let err = Qwen3TtsConfig::from_json(&json)
            .map(|c| format!("unexpectedly parsed as {:?}", c.variant))
            .unwrap_err();
        assert!(err.contains(want), "expected an error naming {want:?}, got {err:?}");
    }
    // The control: the unedited fixture parses, so every failure above is caused by the
    // edit and not by the fixture.
    assert!(Qwen3TtsConfig::from_json(BASE_0B6).is_ok());
}
