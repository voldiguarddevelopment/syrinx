//! `syrinx-chatterbox` — the Chatterbox Turbo config and tokenizer contract, checked
//! against the **files the model repository actually ships**, with no weights and no GPU.
//!
//! # Provenance of the fixtures
//!
//! `tests/golden/chatterbox/config/` holds four files copied byte-for-byte from
//! <https://huggingface.co/ResembleAI/chatterbox-turbo> (`license: mit`):
//! `t3_turbo_v1.yaml` (8.5 KB), `added_tokens.json`, `tokenizer_config.json` and
//! `special_tokens_map.json`. Together they are under 14 KB and contain no weight data.
//! They are committed rather than downloaded so this suite never depends on the network.
//!
//! # Why the negative cases are edits of the real file
//!
//! Every rejection test below starts from the shipped YAML/JSON and changes exactly one
//! value. A hand-written minimal fixture would prove the parser rejects a fixture; an
//! edit of the real file proves it rejects *the real file with one thing wrong*, which is
//! the failure that actually happens when a download is stale or a revision is mixed.
//! Each edit is asserted to have applied, so a typo in the substitution cannot turn a
//! rejection test into a vacuous pass.
//!
//! # Companion
//!
//! `tests/chatterbox_tensor_manifest.rs` checks what these files imply about the shipped
//! tensors.

use syrinx_chatterbox::config::{T3Config, BACKBONE_POS_EMB, VOICE_ENCODER};
use syrinx_chatterbox::contract::ModelContract;
use syrinx_chatterbox::tokenizer::{TokenizerContract, EOT, SPECIAL_ROLES};

const YAML: &str = include_str!("golden/chatterbox/config/t3_turbo_v1.yaml");
const ADDED: &str = include_str!("golden/chatterbox/config/added_tokens.json");
const TOKCFG: &str = include_str!("golden/chatterbox/config/tokenizer_config.json");
const SMAP: &str = include_str!("golden/chatterbox/config/special_tokens_map.json");

/// One-value edit of a shipped fixture, with a guard that the edit applied.
fn edited(src: &str, from: &str, to: &str) -> String {
    assert!(src.contains(from), "fixture does not contain {from:?}; the edit would be a no-op");
    src.replacen(from, to, 1)
}

fn cfg() -> T3Config {
    T3Config::from_yaml(YAML).expect("the shipped t3_turbo_v1.yaml must parse")
}

fn tok() -> TokenizerContract {
    TokenizerContract::from_files(ADDED, TOKCFG, SMAP)
        .expect("the shipped tokenizer files must parse")
}

// ============================================================== the shipped file parses

/// Every load-bearing value, pinned. If a future revision of the checkpoint changes one
/// of these, this is where it is noticed.
#[test]
fn the_shipped_yaml_yields_the_documented_geometry() {
    let c = cfg();

    // Backbone: resolved from `gpt_transformer_type`, which is the field that is true.
    assert_eq!(c.backbone.preset, "gpt2-medium");
    assert_eq!(c.backbone.n_layers, 24);
    assert_eq!(c.backbone.n_heads, 16);
    assert_eq!(c.backbone.n_channels, 1024);
    assert_eq!(c.backbone.ffn_dim, 4096);
    assert_eq!(c.max_total_tokens, 8196);
    assert_eq!(c.input_pos_emb, BACKBONE_POS_EMB);

    // Text side.
    assert_eq!(c.text.dict_size, 50276);
    assert_eq!(c.text.start_token, 255);
    assert_eq!(c.text.stop_token, 0);
    assert_eq!(c.text.max_tokens, 402);

    // Speech side.
    assert_eq!(c.speech.dict_size, 6563);
    assert_eq!(c.speech.start_token, 6561);
    assert_eq!(c.speech.stop_token, 6562);
    assert_eq!(c.speech.max_tokens, 604);
    assert_eq!(c.speech.token_type, "tortoise");
    assert_eq!(c.speech_codebook_size(), 6561);

    // Audio + speaker.
    assert_eq!(c.audio.sample_rate, 32_000);
    assert_eq!(c.audio.hop_size, 320);
    assert_eq!(c.mel_frame_rate_hz(), 100);
    assert_eq!(c.speaker.encoder_type, VOICE_ENCODER);
    assert_eq!(c.speaker.embed_size, 256);
    assert_eq!(c.speaker.cond_prompt_len, 250);
}

/// **The trap this crate exists to record.** `t3_turbo_v1.yaml` is a training-config
/// superset: it declares a 30-layer Llama backbone, and the checkpoint it ships beside is
/// a 24-block GPT-2. The parser must take the truth from `gpt_transformer_type` and must
/// NOT take it from `n_transformer_layers` — so the declared value is read, kept, and
/// reported as disagreeing.
#[test]
fn the_declared_layer_count_contradicts_the_backbone_and_is_not_used() {
    let c = cfg();
    assert_eq!(c.declared_transformer_layers, 30, "the file really does say 30");
    assert_eq!(c.declared_llama_config_name, "Llama_520M", "and really does say Llama");
    assert_eq!(c.backbone.n_layers, 24, "and the checkpoint really has 24 blocks");
    assert!(
        !c.declared_layers_match_backbone(),
        "the shipped file's declared layer count must be reported as disagreeing"
    );

    // The other side of that predicate: a file whose declaration agrees reports agreement.
    let fixed = T3Config::from_yaml(&edited(YAML, "n_transformer_layers: 30", "n_transformer_layers: 24"))
        .unwrap();
    assert!(fixed.declared_layers_match_backbone());
    assert_eq!(fixed.backbone.n_layers, 24, "the backbone is unchanged either way");
}

// ==================================================================== YAML reader shape

/// The reader takes top-level scalars only. An indented key of the same name — which is
/// what a nested block in a superset config looks like — must not shadow the real one.
#[test]
fn indented_keys_do_not_shadow_top_level_ones() {
    let shadowed = format!("{YAML}\nsome_block:\n  hop_size: 999\n  sample_rate: 1\n");
    let c = T3Config::from_yaml(&shadowed).expect("trailing block must not break the parse");
    assert_eq!(c.audio.hop_size, 320, "an indented `hop_size` must be ignored");
    assert_eq!(c.audio.sample_rate, 32_000);

    // …and a top-level key at column zero IS read, so the test above is about the
    // indentation and not about the reader ignoring everything appended.
    let appended = format!("{YAML}\nhop_size: 640\n");
    let c2 = T3Config::from_yaml(&appended).unwrap();
    assert_eq!(c2.audio.hop_size, 640);
}

/// A missing load-bearing key is an error naming it, never a default.
#[test]
fn a_missing_key_is_an_error_not_a_default() {
    let without = YAML.replace("speaker_embed_size: 256", "");
    let err = T3Config::from_yaml(&without).unwrap_err();
    assert!(err.contains("speaker_embed_size"), "got {err}");
}

/// A non-numeric value where an integer is required is an error, not a zero.
#[test]
fn a_non_integer_value_is_an_error() {
    let err = T3Config::from_yaml(&edited(YAML, "hop_size: 320", "hop_size: many")).unwrap_err();
    assert!(err.contains("hop_size"), "got {err}");
    assert!(err.contains("not an integer"), "got {err}");
}

// ==================================================================== the config rejects

#[test]
fn an_unknown_backbone_preset_is_rejected() {
    let err = T3Config::from_yaml(&edited(
        YAML,
        "gpt_transformer_type: gpt2-medium",
        "gpt_transformer_type: gpt2-large",
    ))
    .unwrap_err();
    assert!(err.contains("gpt_transformer_type"), "got {err}");
}

/// The two declared numbers that DO agree with the preset are cross-checked, on both
/// sides: the shipped values pass, a changed one fails.
#[test]
fn a_declared_head_count_or_width_that_contradicts_the_preset_is_rejected() {
    assert!(T3Config::from_yaml(YAML).is_ok());

    let err = T3Config::from_yaml(&edited(YAML, "n_transformer_heads: 16", "n_transformer_heads: 8"))
        .unwrap_err();
    assert!(err.contains("heads"), "got {err}");

    let err = T3Config::from_yaml(&edited(YAML, "n_gpt_channels: 1024", "n_gpt_channels: 2048"))
        .unwrap_err();
    assert!(err.contains("n_gpt_channels"), "got {err}");

    let err = T3Config::from_yaml(&edited(
        YAML,
        "legacy_gpt_hidden_size: 1024",
        "legacy_gpt_hidden_size: 768",
    ))
    .unwrap_err();
    assert!(err.contains("legacy_gpt_hidden_size"), "got {err}");
}

/// The speech specials are the top two ids of the speech dictionary. Both sides of both
/// boundaries: 6561/6562 against 6563 is accepted; one off either way is rejected.
#[test]
fn speech_specials_must_be_the_top_two_ids() {
    assert!(T3Config::from_yaml(YAML).is_ok());

    for (from, to) in [
        ("start_speech_token: 6561", "start_speech_token: 6560"),
        ("start_speech_token: 6561", "start_speech_token: 6562"),
    ] {
        let err = T3Config::from_yaml(&edited(YAML, from, to)).unwrap_err();
        assert!(err.contains("start_speech_token"), "{to}: got {err}");
    }
    for (from, to) in [
        ("stop_speech_token: 6562", "stop_speech_token: 6561"),
        ("stop_speech_token: 6562", "stop_speech_token: 6563"),
    ] {
        let err = T3Config::from_yaml(&edited(YAML, from, to)).unwrap_err();
        assert!(err.contains("stop_speech_token"), "{to}: got {err}");
    }

    // Shifting the dictionary and both specials together stays consistent — the check is
    // relational, not a hardcoded 6561/6562.
    let shifted = edited(YAML, "speech_tokens_dict_size: 6563", "speech_tokens_dict_size: 6663");
    let shifted = edited(&shifted, "start_speech_token: 6561", "start_speech_token: 6661");
    let shifted = edited(&shifted, "stop_speech_token: 6562", "stop_speech_token: 6662");
    let c = T3Config::from_yaml(&shifted).unwrap();
    assert_eq!(c.speech_codebook_size(), 6661);
}

/// The position table must hold prompt + text + speech. Exactly enough is accepted; one
/// short is rejected.
#[test]
fn the_position_table_must_hold_the_whole_budget() {
    // 250 prompt + 402 text + 604 speech = 1256.
    let exact = T3Config::from_yaml(&edited(YAML, "max_total_tokens: 8196", "max_total_tokens: 1256"));
    assert!(exact.is_ok(), "exactly the budget must be accepted: {exact:?}");

    let err = T3Config::from_yaml(&edited(YAML, "max_total_tokens: 8196", "max_total_tokens: 1255"))
        .unwrap_err();
    assert!(err.contains("max_total_tokens"), "got {err}");
}

/// The sample rate must be a whole number of hops, or `mel_frame_rate_hz` would lie.
#[test]
fn a_sample_rate_that_is_not_a_whole_number_of_hops_is_rejected() {
    let ok = T3Config::from_yaml(&edited(YAML, "hop_size: 320", "hop_size: 640")).unwrap();
    assert_eq!(ok.mel_frame_rate_hz(), 50);

    let err = T3Config::from_yaml(&edited(YAML, "hop_size: 320", "hop_size: 300")).unwrap_err();
    assert!(err.contains("hop"), "got {err}");
}

/// The manifest is written for the LSTM voice encoder and for backbone-internal
/// positions. A file declaring anything else describes a model this crate does not.
#[test]
fn an_unexpected_speaker_encoder_or_position_scheme_is_rejected() {
    let err = T3Config::from_yaml(&edited(YAML, "encoder_type: voice_encoder", "encoder_type: campplus"))
        .unwrap_err();
    assert!(err.contains("encoder_type"), "got {err}");

    let err = T3Config::from_yaml(&edited(
        YAML,
        "input_pos_emb: handled_internally_by_backbone",
        "input_pos_emb: learned",
    ))
    .unwrap_err();
    assert!(err.contains("input_pos_emb"), "got {err}");
}

// ============================================================== the tokenizer contract

/// All nineteen tags, their exact spellings and their exact ids. This is the enumeration
/// `docs/CONTROL_SURVEY.md` recorded as unverifiable upstream; it is verified here.
#[test]
fn the_nineteen_tags_ship_as_one_contiguous_run_above_the_base_vocabulary() {
    let t = tok();

    assert_eq!(t.tokenizer_class, "GPT2Tokenizer");
    assert_eq!(t.eot_text, EOT);
    assert_eq!(t.eot_id, 50_256, "the last id of the GPT-2 base vocabulary");
    assert_eq!(t.base_vocab_size, 50_257, "eot_id + 1");
    assert_eq!(t.tags.len(), 19);
    assert_eq!(t.implied_text_vocab_size(), 50_276);

    // In id order, exactly as `added_tokens.json` numbers them.
    let expected: [(&str, u32); 19] = [
        ("[angry]", 50257),
        ("[fear]", 50258),
        ("[surprised]", 50259),
        ("[whispering]", 50260),
        ("[advertisement]", 50261),
        ("[dramatic]", 50262),
        ("[narration]", 50263),
        ("[crying]", 50264),
        ("[happy]", 50265),
        ("[sarcastic]", 50266),
        ("[clear throat]", 50267),
        ("[sigh]", 50268),
        ("[shush]", 50269),
        ("[cough]", 50270),
        ("[groan]", 50271),
        ("[sniff]", 50272),
        ("[gasp]", 50273),
        ("[chuckle]", 50274),
        ("[laugh]", 50275),
    ];
    let got: Vec<(&str, u32)> = t.tags.iter().map(|x| (x.text.as_str(), x.id)).collect();
    assert_eq!(got, expected.to_vec());

    // The run starts exactly at the base vocabulary and has no gaps, so `id -
    // base_vocab_size` is a tag index.
    assert_eq!(t.tags[0].id as usize, t.base_vocab_size);
    assert_eq!(t.tags[18].id as usize, t.base_vocab_size + 18);

    // Spellings a canonical vocabulary must map ONTO, not assume.
    assert_eq!(t.tag_id("[whispering]"), Some(50_260));
    assert_eq!(t.tag_id("[whisper]"), None, "the backend does not spell it this way");
    assert_eq!(t.tag_id("[gasp]"), Some(50_273));
    assert_eq!(t.tag_id("[laugh]"), Some(50_275));
    assert_eq!(t.tag_id("[laughs]"), None, "nor this way");
}

/// The rest of the tokenizer contract: one special filling all four roles, and the two
/// preprocessing switches a port must match.
#[test]
fn one_special_token_fills_every_role_and_preprocessing_is_off() {
    let t = tok();
    assert_eq!(SPECIAL_ROLES, ["bos_token", "eos_token", "pad_token", "unk_token"]);
    assert_eq!(t.eot_text, EOT);
    assert_eq!(t.model_max_length, 1024, "the tokenizer's stock default, not the model's bound");
    assert!(!t.add_bos_token, "T3 supplies start_text_token itself");
    assert!(!t.add_prefix_space);
}

/// Both encodings of a `special_tokens_map.json` entry are handled: `pad_token` ships as
/// a bare string and the other three as objects with a `content` field. Changing either
/// form to a different token is rejected — so neither arm can be silently skipped.
#[test]
fn a_special_token_role_pointing_elsewhere_is_rejected() {
    assert!(TokenizerContract::from_files(ADDED, TOKCFG, SMAP).is_ok());

    // the bare-string form
    let err = TokenizerContract::from_files(
        ADDED,
        TOKCFG,
        &edited(SMAP, "\"pad_token\": \"<|endoftext|>\"", "\"pad_token\": \"<|pad|>\""),
    )
    .unwrap_err();
    assert!(err.contains("pad_token"), "got {err}");

    // the object form
    let err = TokenizerContract::from_files(
        ADDED,
        TOKCFG,
        &edited(SMAP, "\"bos_token\": {\n    \"content\": \"<|endoftext|>\"", "\"bos_token\": {\n    \"content\": \"<|bos|>\""),
    )
    .unwrap_err();
    assert!(err.contains("bos_token"), "got {err}");
}

/// The two tag listings must describe the same set. A tag present in one file and not
/// the other — the signature of a half-updated download — is rejected either way round.
#[test]
fn the_two_tag_listings_must_agree() {
    // renamed in added_tokens.json only
    let err = TokenizerContract::from_files(&edited(ADDED, "\"[laugh]\"", "\"[laughs]\""), TOKCFG, SMAP)
        .unwrap_err();
    assert!(err.contains("same tag set"), "got {err}");

    // renamed in tokenizer_config.json only
    let err = TokenizerContract::from_files(ADDED, &edited(TOKCFG, "\"[laugh]\"", "\"[laughs]\""), SMAP)
        .unwrap_err();
    assert!(err.contains("same tag set"), "got {err}");
}

/// Contiguity, on both sides. Moving one tag off the run is rejected; shifting the whole
/// run together with the base vocabulary is accepted, because the check is relational.
#[test]
fn a_gap_in_the_tag_run_is_rejected() {
    let gapped_added = edited(ADDED, "\"[laugh]\": 50275", "\"[laugh]\": 50276");
    let gapped_cfg = edited(TOKCFG, "\"50275\": {", "\"50276\": {");
    let err = TokenizerContract::from_files(&gapped_added, &gapped_cfg, SMAP).unwrap_err();
    assert!(err.contains("unbroken block"), "got {err}");

    // The run must start AT the base vocabulary size, not above it.
    let shifted_added = edited(ADDED, "\"[angry]\": 50257", "\"[angry]\": 50276");
    let shifted_cfg = edited(TOKCFG, "\"50257\": {", "\"50276\": {");
    let err = TokenizerContract::from_files(&shifted_added, &shifted_cfg, SMAP).unwrap_err();
    assert!(err.contains("unbroken block"), "got {err}");

    // …and moving the base vocabulary boundary moves the whole expectation with it.
    let moved = edited(TOKCFG, "\"50256\": {", "\"50255\": {");
    let err = TokenizerContract::from_files(ADDED, &moved, SMAP).unwrap_err();
    assert!(err.contains("unbroken block"), "got {err}");
}

/// Exactly one entry may be marked `special`. Two, or none, is a file this parser must
/// not guess its way through.
#[test]
fn there_must_be_exactly_one_special_token() {
    // none
    let err = TokenizerContract::from_files(ADDED, &edited(TOKCFG, "\"special\": true", "\"special\": false"), SMAP)
        .unwrap_err();
    assert!(err.contains("exactly one special"), "got {err}");

    // two
    let two = edited(TOKCFG, "\"content\": \"[angry]\",\n      \"lstrip\": false,\n      \"normalized\": true,\n      \"rstrip\": false,\n      \"single_word\": false,\n      \"special\": false", "\"content\": \"[angry]\",\n      \"lstrip\": false,\n      \"normalized\": true,\n      \"rstrip\": false,\n      \"single_word\": false,\n      \"special\": true");
    let err = TokenizerContract::from_files(ADDED, &two, SMAP).unwrap_err();
    assert!(err.contains("exactly one special"), "got {err}");
}

/// The one special must be `<|endoftext|>`; the base vocabulary size is read from its id
/// and everything downstream depends on that.
#[test]
fn the_special_token_must_be_endoftext() {
    let err = TokenizerContract::from_files(
        ADDED,
        &edited(TOKCFG, "\"content\": \"<|endoftext|>\"", "\"content\": \"<|eot|>\""),
        SMAP,
    )
    .unwrap_err();
    assert!(err.contains("special token is"), "got {err}");
}

// ====================================================================== the cross-check

/// `text_tokens_dict_size` must be exactly base vocabulary + tags: 50257 + 19 = 50276.
/// Both sides — the shipped pair agrees, and either file moved on its own is rejected.
#[test]
fn the_text_vocabulary_must_be_base_plus_tags() {
    let c = ModelContract::new(cfg(), tok()).expect("the shipped files must agree");
    assert_eq!(c.tag_count(), 19);
    assert_eq!(c.config.text.dict_size, c.tokenizer.base_vocab_size + c.tag_count());

    // the YAML moved
    let err = ModelContract::new(
        T3Config::from_yaml(&edited(YAML, "text_tokens_dict_size: 50276", "text_tokens_dict_size: 50275")).unwrap(),
        tok(),
    )
    .unwrap_err();
    assert!(err.contains("text_tokens_dict_size"), "got {err}");

    // one tag short: base + 18 no longer matches the YAML's 50276
    let short_added = edited(ADDED, "  \"[laugh]\": 50275,\n", "");
    let short_cfg = edited(
        TOKCFG,
        ",\n    \"50275\": {\n      \"content\": \"[laugh]\",\n      \"lstrip\": false,\n      \"normalized\": true,\n      \"rstrip\": false,\n      \"single_word\": false,\n      \"special\": false\n    }",
        "",
    );
    let t = TokenizerContract::from_files(&short_added, &short_cfg, SMAP).unwrap();
    assert_eq!(t.tags.len(), 18, "the edit must remove exactly one tag");
    let err = ModelContract::new(cfg(), t).unwrap_err();
    assert!(err.contains("18 tags"), "got {err}");
}

/// The text sequence markers reuse ordinary base BPE ids (255 and 0). One that strayed
/// into the tag block would collide with a paralinguistic tag. Boundary on both sides:
/// the last base id is accepted, the first tag id is not.
#[test]
fn a_text_marker_may_not_stray_into_the_tag_block() {
    let at_last_base = T3Config::from_yaml(&edited(YAML, "start_text_token: 255", "start_text_token: 50256")).unwrap();
    assert!(ModelContract::new(at_last_base, tok()).is_ok(), "50256 is still a base BPE id");

    let at_first_tag = T3Config::from_yaml(&edited(YAML, "start_text_token: 255", "start_text_token: 50257")).unwrap();
    let err = ModelContract::new(at_first_tag, tok()).unwrap_err();
    assert!(err.contains("start_text_token"), "got {err}");

    let stop_in_tags = T3Config::from_yaml(&edited(YAML, "stop_text_token: 0", "stop_text_token: 50275")).unwrap();
    let err = ModelContract::new(stop_in_tags, tok()).unwrap_err();
    assert!(err.contains("stop_text_token"), "got {err}");
}
