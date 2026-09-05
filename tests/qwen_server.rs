//! Qwen3-TTS on the OpenAI-compatible server — the model-FREE gate.
//!
//! `syrinx-qwen` is a verified port that nothing could reach: `syrinx-serve` did not
//! depend on it, while `syrinx-cue`'s `caps.toml` already carried all five Qwen
//! `[[backend]]` rows (`docs/backends/QWEN_PORT_STATUS.md` §5 gap 6). `syrinx_serve::qwen`
//! closes that, and this file gates the half that needs no weights:
//!
//!   * **backend selection** — which of the five checkpoints maps to which generation
//!     mode, and that a non-Qwen backend id is refused rather than guessed at;
//!   * **the per-checkpoint capability truth**, which is the whole reason `caps.toml` is
//!     per-checkpoint rather than per-crate: `-Base` is clone-only and instruct-less, the
//!     0.6B CustomVoice *accepts* an instruction and silently discards it, the 1.7B
//!     CustomVoice honours it, and VoiceDesign's instruction describes the **voice**, so
//!     its utterance is never split;
//!   * **the route and wire contract** — 200 + `audio/wav`, 422 for a malformed body, 400
//!     for a blank input, a typed 500 when synthesis fails;
//!   * **the hard invariant** (CLAUDE.md): no bracket cue text, no SSML tag and no speaker
//!     token ever reaches a backend as literal spoken text, in any dialect, however
//!     malformed.
//!
//! Everything here drives the real Axum router through `tower::oneshot` — no port is
//! bound — with a recording [`QwenEngine`] in place of the Candle one. That substitution
//! is exactly what makes the gate honest: the engine is the only part that needs a 3.9 GB
//! checkpoint, and every decision this file checks is made **before** it is called.
//! Anything needing real weights belongs in a `real_qwen_*` test, not here.
//!
//! What this file therefore does NOT prove: that the port renders audible, correct audio
//! (see `tests/real_qwen_*.rs` and the WER runs in `docs/backends/QWEN_PORT_STATUS.md`),
//! or that `synth_qwen::QwenModelEngine` loads a checkpoint — that needs the weights.

use std::sync::{Arc, Mutex};

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, Response, StatusCode};
use tower::ServiceExt; // brings `oneshot` into scope

use syrinx_cue::BackendId;
use syrinx_serve::qwen::{
    plan, router_with_qwen_synth, InstructEffect, QwenEngine, QwenMode, QwenPlanError,
    QwenRequest, QwenSynth, QWEN_SAMPLE_RATE,
};
use syrinx_serve::ApiError;

const SPEECH_ROUTE: &str = "/v1/audio/speech";

/// Samples the recording engine returns per rendered segment — small, and not a round
/// number, so a body length can only match by actually concatenating the right count.
const SAMPLES_PER_SEGMENT: usize = 3;

/// One call the planner made into the engine.
#[derive(Debug, Clone, PartialEq)]
struct Seen {
    mode: QwenMode,
    text: String,
    instruct: Option<String>,
    effect: InstructEffect,
}

/// A [`QwenEngine`] that renders nothing and records everything: the stand-in for
/// `synth_qwen::QwenModelEngine`, which needs the weights.
#[derive(Clone)]
struct Recorder {
    log: Arc<Mutex<Vec<Seen>>>,
    /// When set, every render fails — the load/synth-failure path.
    fail: bool,
}

impl Recorder {
    fn new() -> Self {
        Self { log: Arc::new(Mutex::new(Vec::new())), fail: false }
    }

    fn failing() -> Self {
        Self { log: Arc::new(Mutex::new(Vec::new())), fail: true }
    }

    fn seen(&self) -> Vec<Seen> {
        self.log.lock().expect("the recorder lock must not be poisoned").clone()
    }
}

impl QwenEngine for Recorder {
    fn render(&self, req: &QwenRequest<'_>) -> Result<Vec<f32>, String> {
        self.log.lock().expect("the recorder lock must not be poisoned").push(Seen {
            mode: req.mode,
            text: req.text.to_string(),
            instruct: req.instruct.map(str::to_string),
            effect: req.instruct_effect,
        });
        if self.fail {
            return Err("engine refused".to_string());
        }
        Ok(vec![0.5; SAMPLES_PER_SEGMENT])
    }
}

/// A synth for `backend` plus the recorder behind it.
fn synth_for(backend: BackendId) -> (QwenSynth<Recorder>, Recorder) {
    let rec = Recorder::new();
    let synth = QwenSynth::new(backend, rec.clone()).expect("a Qwen backend must bind");
    (synth, rec)
}

/// A well-formed speech body for `input`.
fn speech_body(input: &str) -> String {
    format!(
        r#"{{"model":"qwen3-tts","input":{},"voice":"serena"}}"#,
        serde_json::to_string(input).expect("the input must serialize")
    )
}

/// Drive a router with one POST and return the response.
async fn post(app: axum::Router, path: &str, json: &str) -> Response<Body> {
    let request = Request::builder()
        .method("POST")
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json.to_string()))
        .expect("the request must build");
    app.oneshot(request).await.expect("the router must answer")
}

async fn collect(response: Response<Body>) -> Vec<u8> {
    to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the response body must collect")
        .to_vec()
}

fn content_type(response: &Response<Body>) -> String {
    response
        .headers()
        .get(header::CONTENT_TYPE)
        .expect("a response must carry a content-type")
        .to_str()
        .expect("the content-type must be UTF-8")
        .to_string()
}

fn le_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}

fn le_u16(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(bytes[at..at + 2].try_into().expect("two bytes"))
}

/// Send `input` through a router for `backend` and return what the engine saw.
async fn drive(backend: BackendId, input: &str) -> Vec<Seen> {
    let (synth, rec) = synth_for(backend);
    let response = post(router_with_qwen_synth(synth), SPEECH_ROUTE, &speech_body(input)).await;
    assert_eq!(response.status(), StatusCode::OK, "{} / {input:?}", backend.as_str());
    rec.seen()
}

// ---------------------------------------------------------------- backend selection

/// Every Qwen checkpoint resolves to its generation mode, and every non-Qwen backend
/// resolves to none — the routing decision, made from the backend id alone.
#[test]
fn every_backend_id_maps_to_its_qwen_mode_or_to_none() {
    assert_eq!(QwenMode::of(BackendId::Qwen06bBase), Some(QwenMode::VoiceClone));
    assert_eq!(QwenMode::of(BackendId::Qwen17bBase), Some(QwenMode::VoiceClone));
    assert_eq!(QwenMode::of(BackendId::Qwen06bCustomVoice), Some(QwenMode::CustomVoice));
    assert_eq!(QwenMode::of(BackendId::Qwen17bCustomVoice), Some(QwenMode::CustomVoice));
    assert_eq!(QwenMode::of(BackendId::Qwen17bVoiceDesign), Some(QwenMode::VoiceDesign));

    for other in [
        BackendId::FishS1Mini,
        BackendId::FishS2Pro,
        BackendId::CosyVoice2,
        BackendId::CosyVoice3,
    ] {
        assert_eq!(QwenMode::of(other), None, "{} is not a Qwen backend", other.as_str());
    }

    // And the five are exactly the `krate = "syrinx-qwen"` rows of caps.toml — the mode
    // table cannot silently miss a checkpoint the capability table knows about.
    let qwen: Vec<&str> = BackendId::ALL
        .iter()
        .filter(|b| b.caps().expect("caps must load").krate == "syrinx-qwen")
        .map(|b| b.as_str())
        .collect();
    assert_eq!(
        qwen,
        vec![
            "qwen3-0.6b-base",
            "qwen3-0.6b-customvoice",
            "qwen3-1.7b-base",
            "qwen3-1.7b-customvoice",
            "qwen3-1.7b-voicedesign",
        ]
    );
    for id in &qwen {
        let backend = BackendId::from_str(id).expect("the id must resolve");
        assert!(QwenMode::of(backend).is_some(), "{id} has no mode");
    }
}

/// A backend that is not Qwen is refused, at planning and at construction — never
/// silently driven as if it were one.
#[test]
fn a_non_qwen_backend_is_refused() {
    assert_eq!(
        plan(BackendId::FishS2Pro, "hello").unwrap_err(),
        QwenPlanError::NotQwen(BackendId::FishS2Pro)
    );
    let err = QwenSynth::new(BackendId::CosyVoice2, Recorder::new())
        .err()
        .expect("a non-Qwen backend must not bind");
    assert_eq!(err, QwenPlanError::NotQwen(BackendId::CosyVoice2));
    assert!(err.to_string().contains("cosyvoice2"), "{err}");
}

// ------------------------------------------------- the five checkpoints are not alike

/// `-Base` is clone-only: no instruction channel of any kind, so a cue is dropped and
/// reported and the engine is handed a bare text request.
#[tokio::test]
async fn the_base_checkpoints_have_no_expressive_channel() {
    for backend in [BackendId::Qwen06bBase, BackendId::Qwen17bBase] {
        let planned = plan(backend, "[happy] hello").expect("planning must succeed");
        assert_eq!(planned.mode, QwenMode::VoiceClone);
        assert_eq!(planned.instruct_effect, InstructEffect::Unsupported);
        assert!(!planned.instruct_effect.reaches_model());
        assert_eq!(planned.segments.len(), 1, "a clone-only backend is never split");
        assert_eq!(planned.segments[0].instruct, None);
        assert!(
            !planned.report.entries.is_empty(),
            "a dropped cue must be reported, not silently lost"
        );

        let seen = drive(backend, "[happy] hello").await;
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].mode, QwenMode::VoiceClone);
        assert_eq!(seen[0].instruct, None);
        assert_eq!(seen[0].effect, InstructEffect::Unsupported);
        assert_eq!(seen[0].text.trim(), "hello");
    }
}

/// The reason `caps.toml` is per **checkpoint**: the same request, the same API, two
/// different truths. The 0.6B CustomVoice takes an instruction and throws it away; the
/// 1.7B obeys it. Flattening the two would make one of them a lie.
#[tokio::test]
async fn the_06b_customvoice_discards_the_instruction_the_17b_honors() {
    let small = plan(BackendId::Qwen06bCustomVoice, "[happy] hello").expect("plan");
    let big = plan(BackendId::Qwen17bCustomVoice, "[happy] hello").expect("plan");

    assert_eq!(small.mode, QwenMode::CustomVoice);
    assert_eq!(big.mode, QwenMode::CustomVoice);
    assert_eq!(small.instruct_effect, InstructEffect::AcceptedAndDiscarded);
    assert_eq!(big.instruct_effect, InstructEffect::Honored);

    // The cue layer already refuses to build a delivery instruction the 0.6B would
    // swallow (it reports the cue as accepted-but-ignored instead)…
    assert_eq!(small.segments[0].instruct, None);
    // … while the 1.7B gets the utterance-scoped instruction the cue lowers to.
    assert_eq!(
        big.segments[0].instruct.as_deref(),
        Some("Speak in a happy, cheerful tone")
    );

    // A *configured* instruction is a different matter: the upstream API accepts one on
    // the 0.6B and discards it inside `generate_custom_voice`, so it is passed through
    // and the effect says what will become of it — "accepted" is not "honoured", and a
    // caller must be able to tell the difference.
    for (backend, effect) in [
        (BackendId::Qwen06bCustomVoice, InstructEffect::AcceptedAndDiscarded),
        (BackendId::Qwen17bCustomVoice, InstructEffect::Honored),
    ] {
        let rec = Recorder::new();
        let synth = QwenSynth::new(backend, rec.clone())
            .expect("bind")
            .with_instruct("Speak like a newsreader");
        let response =
            post(router_with_qwen_synth(synth), SPEECH_ROUTE, &speech_body("hello")).await;
        assert_eq!(response.status(), StatusCode::OK);
        let seen = rec.seen();
        assert_eq!(seen[0].instruct.as_deref(), Some("Speak like a newsreader"));
        assert_eq!(seen[0].effect, effect);
    }
}

/// A configured instruction must NOT leak into a checkpoint that has no slot for it: on
/// `-Base` there is no instruct argument at all, so the engine is handed `None`.
#[tokio::test]
async fn a_configured_instruction_never_reaches_a_clone_only_checkpoint() {
    let rec = Recorder::new();
    let synth = QwenSynth::new(BackendId::Qwen06bBase, rec.clone())
        .expect("bind")
        .with_instruct("Speak like a newsreader");
    let response = post(router_with_qwen_synth(synth), SPEECH_ROUTE, &speech_body("hello")).await;
    assert_eq!(response.status(), StatusCode::OK);
    let seen = rec.seen();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].instruct, None, "a -Base checkpoint has no instruction channel");
}

/// On a delivery-instruct backend a conflicting second cue means "say the rest
/// differently", and the utterance becomes two requests, each with its own instruction.
#[tokio::test]
async fn a_conflicting_cue_splits_a_customvoice_utterance() {
    let planned = plan(BackendId::Qwen17bCustomVoice, "[happy] hi [sad] bye").expect("plan");
    assert_eq!(planned.segments.len(), 2, "two deliveries, two requests");
    assert_eq!(
        planned.segments[0].instruct.as_deref(),
        Some("Speak in a happy, cheerful tone")
    );
    assert_eq!(
        planned.segments[1].instruct.as_deref(),
        Some("Speak in a sad, sorrowful tone")
    );
    // The split partitions the text: nothing is duplicated and nothing is lost.
    let rejoined: String = planned.segments.iter().map(|s| s.text.as_str()).collect();
    assert_eq!(rejoined, planned.text);

    // A cue that does not conflict does not split.
    let one = plan(BackendId::Qwen17bCustomVoice, "[happy] hi there").expect("plan");
    assert_eq!(one.segments.len(), 1);

    let seen = drive(BackendId::Qwen17bCustomVoice, "[happy] hi [sad] bye").await;
    assert_eq!(seen.len(), 2, "each segment is its own synthesis call");
    assert_eq!(seen[0].text.trim(), "hi");
    assert_eq!(seen[1].text.trim(), "bye");
}

/// VoiceDesign honours its instruction, but that instruction describes the **voice**, not
/// the delivery: splitting would re-design the timbre mid-line, i.e. change who is
/// speaking. So the identical input that splits on CustomVoice stays one request here,
/// and the cue it could not honour is reported.
#[tokio::test]
async fn voicedesign_honors_the_instruction_and_is_never_split() {
    let planned = plan(BackendId::Qwen17bVoiceDesign, "[happy] hi [sad] bye").expect("plan");
    assert_eq!(planned.mode, QwenMode::VoiceDesign);
    assert_eq!(planned.instruct_effect, InstructEffect::Honored);
    assert_eq!(planned.segments.len(), 1, "one voice for the whole utterance");
    assert_eq!(
        planned.segments[0].instruct.as_deref(),
        Some("Speak in a happy, cheerful tone")
    );
    assert_eq!(planned.segments[0].text, planned.text, "the text is not carved up");
    assert!(
        !planned.report.entries.is_empty(),
        "the cue that could not be honoured must be reported"
    );

    // A free-text cue is a voice description and reaches the engine verbatim.
    let described = plan(BackendId::Qwen17bVoiceDesign, "[a gravelly old sailor] ahoy")
        .expect("plan");
    assert_eq!(
        described.segments[0].instruct.as_deref(),
        Some("a gravelly old sailor")
    );

    let seen = drive(BackendId::Qwen17bVoiceDesign, "[happy] hi [sad] bye").await;
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].mode, QwenMode::VoiceDesign);
}

/// Both authoring syntaxes reach the backend through the cue layer, and `syrinx-serve`
/// parses neither: an SSML document lowers to the same utterance-scoped instruction a
/// bracket cue would.
#[tokio::test]
async fn ssml_reaches_the_backend_through_the_same_cue_path() {
    let seen = drive(
        BackendId::Qwen17bCustomVoice,
        "<speak><emphasis level=\"strong\">hi</emphasis> there</speak>",
    )
    .await;
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].text, "hi there");
    assert_eq!(seen[0].instruct.as_deref(), Some("Emphasise this part strongly"));
}

// ---------------------------------------------------------------- the hard invariant

/// CLAUDE.md, non-negotiable: **no bracket cue text and no speaker token may ever reach a
/// backend as literal text. No SSML tag either — no cue markup of any kind — ever, in any
/// dialect, however malformed.**
///
/// The corollary (ADR-0001 §9.1 / D5) is that every unescaped `[...]` is cue syntax, so
/// `array[0]` lowers to `array` — the accepted price of an invariant that is total rather
/// than best-effort. Each input below is driven through the whole route, for every one of
/// the five checkpoints, and the text handed to the engine is checked for every markup
/// character of either dialect.
#[tokio::test]
async fn no_cue_markup_ever_reaches_the_backend_as_spoken_text() {
    const INPUTS: &[&str] = &[
        "[happy] hello",
        "[happy] hi [sad] bye",
        "he paused [laughs] and went on",
        "[wibble] an unknown label",
        "[happy hello",                 // never closed
        "[[nested]] x",                 // doubled delimiters
        "]stray[ brackets",             // reversed
        "[speaker 2] hi",               // speaker turn, bracket spelling
        "<|speaker:2|> hi",             // speaker turn, token spelling
        "array[0] = 1",                 // D5: this is cue syntax, and lowers away
        "[prosody rate=\"slow\"] slow down",
        "<speak>plain ssml</speak>",
        "<speak><emphasis level=\"strong\">loud</emphasis></speak>",
        "\\[escaped\\] text",
        "trailing [",
    ];
    const BACKENDS: &[BackendId] = &[
        BackendId::Qwen06bBase,
        BackendId::Qwen06bCustomVoice,
        BackendId::Qwen17bBase,
        BackendId::Qwen17bCustomVoice,
        BackendId::Qwen17bVoiceDesign,
    ];

    for &backend in BACKENDS {
        for &input in INPUTS {
            let (synth, rec) = synth_for(backend);
            let response =
                post(router_with_qwen_synth(synth), SPEECH_ROUTE, &speech_body(input)).await;
            // Every input here is *plannable* — malformed markup is lowered away, not
            // rejected — so the engine really is called and the check below is never
            // vacuous. (A document that cannot be planned is covered separately.)
            assert_eq!(
                response.status(),
                StatusCode::OK,
                "{} / {input:?} did not reach the engine",
                backend.as_str()
            );
            let seen = rec.seen();
            assert!(!seen.is_empty(), "{} / {input:?}: nothing rendered", backend.as_str());
            // An ESCAPED bracket is the one legitimate way a bracket may be spoken
            // (CLAUDE.md: "`\[` is the only way to speak a literal bracket"), so the
            // invariant is not "no brackets" — it is "no brackets the source did not
            // escape". This mirrors the headline property in
            // crates/syrinx-cue/tests/projection_strip.rs, which asserts the same bound.
            // Asserting the stricter form here happened to hold only while adr/0002's
            // defect was live and escapes were being destroyed; it would now forbid the
            // fixed behaviour.
            let escaped = input.matches(r"\[").count() + input.matches(r"\]").count();
            for seen in seen {
                let text = &seen.text;
                let brackets = text.matches('[').count() + text.matches(']').count();
                assert!(
                    brackets <= escaped,
                    "{} / {input:?}: {brackets} bracket(s) reached the backend but the \
                     source escaped only {escaped}: {text:?}",
                    backend.as_str()
                );
                for bad in ['<', '>', '|'] {
                    assert!(
                        !text.contains(bad),
                        "{} / {input:?}: cue markup `{bad}` reached the backend as text: \
                         {text:?}",
                        backend.as_str()
                    );
                }
                assert!(
                    !text.contains("speaker:"),
                    "{} / {input:?}: a speaker token reached the backend: {text:?}",
                    backend.as_str()
                );
            }
        }
    }
}

/// A document that mixes the two syntaxes is a hard error in `syrinx-cue`, and
/// `syrinx-serve` surfaces it as a typed 500 rather than guessing which dialect was meant
/// — and never calls the engine.
#[tokio::test]
async fn a_mixed_syntax_document_fails_without_reaching_the_backend() {
    let (synth, rec) = synth_for(BackendId::Qwen17bCustomVoice);
    let response = post(
        router_with_qwen_synth(synth),
        SPEECH_ROUTE,
        &speech_body("[happy] a <speak>b</speak>"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(content_type(&response), "application/json");
    assert!(rec.seen().is_empty(), "an unplannable request must not reach the engine");
}

/// An utterance that is *entirely* cue markup leaves nothing to say: lowering strips it
/// to the empty string, `pass_hoist` yields no request, and the synth produces no audio —
/// which the handler answers with its typed synthesis-failure 500 (the `Synth` trait has
/// no other error channel). Pinned because "no audio" must never be a 200 with an empty
/// body.
#[tokio::test]
async fn an_utterance_that_is_all_markup_renders_nothing() {
    let (synth, rec) = synth_for(BackendId::Qwen06bCustomVoice);
    let response = post(router_with_qwen_synth(synth), SPEECH_ROUTE, &speech_body("[]")).await;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let err: ApiError = serde_json::from_slice(&collect(response).await).expect("typed error");
    assert_eq!(err.error.kind, "synthesis_error");
    assert!(rec.seen().is_empty(), "there was nothing to render");
}

// ---------------------------------------------------------------- the wire contract

/// A well-formed request returns 200 with a complete, well-formed 24 kHz mono 16-bit WAV
/// whose payload is every rendered segment, concatenated.
#[tokio::test]
async fn a_speech_request_returns_a_24khz_wav_body() {
    // Two segments, so the body can only be right if both were rendered and joined.
    let (synth, rec) = synth_for(BackendId::Qwen17bCustomVoice);
    let response = post(
        router_with_qwen_synth(synth),
        SPEECH_ROUTE,
        &speech_body("[happy] hi [sad] bye"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(content_type(&response), "audio/wav");

    let segments = rec.seen().len();
    assert_eq!(segments, 2);
    let body = collect(response).await;
    let data_len = (segments * SAMPLES_PER_SEGMENT * 2) as u32;

    assert_eq!(&body[0..4], b"RIFF");
    assert_eq!(le_u32(&body, 4), 36 + data_len, "RIFF size = 44 - 8 + data");
    assert_eq!(&body[8..12], b"WAVE");
    assert_eq!(&body[12..16], b"fmt ");
    assert_eq!(le_u32(&body, 16), 16, "PCM fmt chunk size");
    assert_eq!(le_u16(&body, 20), 1, "format = PCM");
    assert_eq!(le_u16(&body, 22), 1, "mono");
    assert_eq!(le_u32(&body, 24), QWEN_SAMPLE_RATE, "24 kHz");
    assert_eq!(le_u32(&body, 28), QWEN_SAMPLE_RATE * 2, "byte rate");
    assert_eq!(le_u16(&body, 32), 2, "block align");
    assert_eq!(le_u16(&body, 34), 16, "16-bit");
    assert_eq!(&body[36..40], b"data");
    assert_eq!(le_u32(&body, 40), data_len);
    assert_eq!(body.len(), 44 + data_len as usize, "header + every segment's samples");
    // 0.5 -> round(0.5 * 32767) = 16384, in every frame.
    for frame in body[44..].chunks(2) {
        assert_eq!(i16::from_le_bytes([frame[0], frame[1]]), 16384);
    }

    // One segment gives exactly half the payload — the concatenation is real, not a
    // fixed-size buffer.
    let (synth, _) = synth_for(BackendId::Qwen17bCustomVoice);
    let single = post(router_with_qwen_synth(synth), SPEECH_ROUTE, &speech_body("hi")).await;
    let single = collect(single).await;
    assert_eq!(single.len(), 44 + SAMPLES_PER_SEGMENT * 2);
}

/// `response_format: "stream"` is answered from the buffered body: the port exposes no
/// chunk-streaming path, so `QwenSynth` does not override the streaming hook and the
/// handler's fallback applies.
#[tokio::test]
async fn a_streaming_request_falls_back_to_the_buffered_body() {
    let (synth, rec) = synth_for(BackendId::Qwen17bCustomVoice);
    let body = r#"{"model":"qwen3-tts","input":"hi","voice":"serena","response_format":"stream"}"#;
    let response = post(router_with_qwen_synth(synth), SPEECH_ROUTE, body).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(content_type(&response), "audio/wav");
    assert_eq!(rec.seen().len(), 1);
    assert_eq!(collect(response).await.len(), 44 + SAMPLES_PER_SEGMENT * 2);
}

/// The error contract, unchanged by the Qwen backend: a malformed body is a typed 422, a
/// blank input a typed 400, and neither reaches the engine.
#[tokio::test]
async fn malformed_and_blank_requests_are_typed_errors() {
    let (synth, rec) = synth_for(BackendId::Qwen17bCustomVoice);
    let response = post(
        router_with_qwen_synth(synth),
        SPEECH_ROUTE,
        r#"{"model":"qwen3-tts","input":"hi"}"#, // no `voice`
    )
    .await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let err: ApiError = serde_json::from_slice(&collect(response).await).expect("typed error");
    assert_eq!(err.error.kind, "invalid_request_error");
    assert!(rec.seen().is_empty());

    let (synth, rec) = synth_for(BackendId::Qwen17bCustomVoice);
    let response = post(router_with_qwen_synth(synth), SPEECH_ROUTE, &speech_body("   ")).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let err: ApiError = serde_json::from_slice(&collect(response).await).expect("typed error");
    assert_eq!(err.error.kind, "invalid_request_error");
    assert!(rec.seen().is_empty());
}

/// A failing engine is a typed 500 — never a 200 carrying an empty body.
#[tokio::test]
async fn an_engine_failure_is_a_typed_500() {
    let rec = Recorder::failing();
    let synth = QwenSynth::new(BackendId::Qwen17bCustomVoice, rec.clone()).expect("bind");
    let response = post(router_with_qwen_synth(synth), SPEECH_ROUTE, &speech_body("hi")).await;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(content_type(&response), "application/json");
    let err: ApiError = serde_json::from_slice(&collect(response).await).expect("typed error");
    assert_eq!(err.error.kind, "synthesis_error");
    assert_eq!(rec.seen().len(), 1, "the engine was called and refused");
}
