//! Default-feature hardening tests for the OpenAI-compatible audio server.
//!
//! These pin the additive hardening of `crates/syrinx-serve` *without the model*
//! (no `real` feature): the new `GET /v1/version` endpoint and its typed body, the
//! robust typed-error responses (400 for a blank `input`, 500 when the synth yields
//! an empty body, 422 still for a malformed request), and the **true chunked**
//! streaming path driven through the trait's optional `synthesize_stream` hook by a
//! model-free stub synth.
//!
//! They use `tower::ServiceExt::oneshot` exactly like the frozen `audio_server.rs`
//! scaffold test and add only *new* test functions — the frozen tests are untouched.

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, Response, StatusCode};
use http_body::Body as _; // brings `size_hint` into scope for the streaming-shape check
use tower::ServiceExt; // brings `oneshot` into scope

use syrinx_serve::{
    router, router_with_synth, ApiError, SpeechRequest, Synth, Version,
};

const SPEECH_ROUTE: &str = "/v1/audio/speech";
const VERSION_ROUTE: &str = "/v1/version";

/// A well-formed buffered request (all four required fields, `wav` format).
const BUFFERED_JSON: &str =
    r#"{"model":"syrinx-1","input":"hello world","voice":"alloy","response_format":"wav"}"#;

/// A well-formed streaming request (`stream` format).
const STREAMING_JSON: &str =
    r#"{"model":"syrinx-1","input":"hello world","voice":"alloy","response_format":"stream"}"#;

/// A request with a present-but-blank `input` (whitespace) — valid JSON, invalid value.
const BLANK_INPUT_JSON: &str =
    r#"{"model":"syrinx-1","input":"   ","voice":"alloy","response_format":"wav"}"#;

/// A malformed request: the required `voice` field is omitted (deserialization fails).
const MISSING_VOICE_JSON: &str =
    r#"{"model":"syrinx-1","input":"hello world","response_format":"wav"}"#;

/// A stub synth whose buffered output is non-empty marker bytes and which also
/// streams: `synthesize_stream` yields three distinct chunks in order, exercising
/// the true chunked path with no model.
struct ChunkStreamSynth;

const CHUNK_A: &[u8] = &[1, 2, 3];
const CHUNK_B: &[u8] = &[4, 5];
const CHUNK_C: &[u8] = &[6, 7, 8, 9];

impl Synth for ChunkStreamSynth {
    fn synthesize(&self, _req: &SpeechRequest) -> Vec<u8> {
        // The buffered body is the concatenation of the streamed chunks.
        [CHUNK_A, CHUNK_B, CHUNK_C].concat()
    }

    fn synthesize_stream(
        &self,
        _req: &SpeechRequest,
    ) -> Option<Box<dyn Iterator<Item = Vec<u8>> + Send>> {
        let chunks = vec![CHUNK_A.to_vec(), CHUNK_B.to_vec(), CHUNK_C.to_vec()];
        Some(Box::new(chunks.into_iter()))
    }
}

/// A stub synth that always fails: it returns an empty buffered body and provides no
/// streaming hook, modelling a load/synth failure.
struct EmptySynth;

impl Synth for EmptySynth {
    fn synthesize(&self, _req: &SpeechRequest) -> Vec<u8> {
        Vec::new()
    }
}

/// Drive a router with one HTTP request and return the response.
async fn send(app: axum::Router, method: &str, path: &str, json: Option<&str>) -> Response<Body> {
    let builder = Request::builder().method(method).uri(path);
    let request = match json {
        Some(body) => builder
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .expect("request with a JSON body must build"),
        None => builder
            .body(Body::empty())
            .expect("request with an empty body must build"),
    };
    app.oneshot(request)
        .await
        .expect("the router must answer the request")
}

fn content_type(response: &Response<Body>) -> String {
    response
        .headers()
        .get(header::CONTENT_TYPE)
        .expect("a response must carry a content-type header")
        .to_str()
        .expect("the content-type header must be valid UTF-8")
        .to_string()
}

async fn collect(response: Response<Body>) -> Vec<u8> {
    to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the response body must collect")
        .to_vec()
}

// ----------------------------------------------------------------------------
// GET /v1/version — typed {name, version}.
// ----------------------------------------------------------------------------

#[tokio::test]
async fn test_version_returns_typed_json_name_and_version() {
    let response = send(router(), "GET", VERSION_ROUTE, None).await;

    assert_eq!(response.status(), StatusCode::OK, "GET /v1/version is 200");
    assert_eq!(
        content_type(&response),
        "application/json",
        "the version body must be typed JSON"
    );

    let body = collect(response).await;
    let version: Version =
        serde_json::from_slice(&body).expect("the body must deserialize into the typed Version");
    assert_eq!(
        version.name, "syrinx-serve",
        "the version body must report the crate name"
    );
    assert!(
        !version.version.is_empty(),
        "the version body must carry a non-empty version"
    );
}

#[tokio::test]
async fn test_version_route_rejects_non_get() {
    let post = send(router(), "POST", VERSION_ROUTE, Some(BUFFERED_JSON)).await;
    assert_eq!(
        post.status(),
        StatusCode::METHOD_NOT_ALLOWED,
        "a POST to the GET-only version route must be 405"
    );
}

// ----------------------------------------------------------------------------
// Robust error responses — 400 blank input, 500 empty synth body, 422 malformed.
// ----------------------------------------------------------------------------

#[tokio::test]
async fn test_blank_input_returns_400_typed_error() {
    let response = send(router(), "POST", SPEECH_ROUTE, Some(BLANK_INPUT_JSON)).await;

    assert_eq!(
        response.status(),
        StatusCode::BAD_REQUEST,
        "a blank input must be 400 Bad Request"
    );
    assert_eq!(content_type(&response), "application/json");

    let body = collect(response).await;
    let err: ApiError = serde_json::from_slice(&body).expect("the 400 body must be a typed ApiError");
    assert_eq!(err.error.kind, "invalid_request_error");
    assert!(!err.error.message.is_empty());
}

#[tokio::test]
async fn test_empty_synth_body_returns_500_typed_error() {
    let app = router_with_synth(Arc::new(EmptySynth));
    let response = send(app, "POST", SPEECH_ROUTE, Some(BUFFERED_JSON)).await;

    assert_eq!(
        response.status(),
        StatusCode::INTERNAL_SERVER_ERROR,
        "an empty synth body (synth failure) must be 500, not a 200 with empty bytes"
    );
    assert_eq!(content_type(&response), "application/json");

    let body = collect(response).await;
    let err: ApiError = serde_json::from_slice(&body).expect("the 500 body must be a typed ApiError");
    assert_eq!(err.error.kind, "synthesis_error");
    assert!(!err.error.message.is_empty());
}

#[tokio::test]
async fn test_empty_synth_body_streaming_also_returns_500() {
    // The fallback streaming path (no native stream hook) must also 500 on empty.
    let app = router_with_synth(Arc::new(EmptySynth));
    let response = send(app, "POST", SPEECH_ROUTE, Some(STREAMING_JSON)).await;
    assert_eq!(
        response.status(),
        StatusCode::INTERNAL_SERVER_ERROR,
        "an empty buffered fallback in streaming mode must also be 500"
    );
}

#[tokio::test]
async fn test_malformed_request_still_returns_422() {
    let response = send(router(), "POST", SPEECH_ROUTE, Some(MISSING_VOICE_JSON)).await;
    assert_eq!(
        response.status(),
        StatusCode::UNPROCESSABLE_ENTITY,
        "a missing required field must still be 422"
    );
    let body = collect(response).await;
    let err: ApiError =
        serde_json::from_slice(&body).expect("the 422 body must still be a typed ApiError");
    assert_eq!(err.error.kind, "invalid_request_error");
}

// ----------------------------------------------------------------------------
// True chunked streaming via the synthesize_stream hook.
// ----------------------------------------------------------------------------

#[tokio::test]
async fn test_native_stream_hook_emits_chunked_body() {
    let app = router_with_synth(Arc::new(ChunkStreamSynth));
    let response = send(app, "POST", SPEECH_ROUTE, Some(STREAMING_JSON)).await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(content_type(&response), "audio/wav");

    // Streaming shape: the size is not known up front.
    let response = {
        // Re-issue because size_hint consumes the body below; build a fresh one.
        let app = router_with_synth(Arc::new(ChunkStreamSynth));
        send(app, "POST", SPEECH_ROUTE, Some(STREAMING_JSON)).await
    };
    assert_eq!(
        response.into_body().size_hint().exact(),
        None,
        "the native streamed shape must not advertise a known exact size"
    );

    // The concatenated chunks must equal the in-order chunk bytes.
    let app = router_with_synth(Arc::new(ChunkStreamSynth));
    let response = send(app, "POST", SPEECH_ROUTE, Some(STREAMING_JSON)).await;
    let body = collect(response).await;
    let expected: Vec<u8> = [CHUNK_A, CHUNK_B, CHUNK_C].concat();
    assert_eq!(
        body, expected,
        "the streamed body must be the in-order concatenation of the synth's chunks"
    );
}

#[tokio::test]
async fn test_wav_format_uses_buffered_path_not_stream_hook() {
    // With response_format=wav, the buffered path is used even when a stream hook
    // exists: the body has a known exact size.
    let app = router_with_synth(Arc::new(ChunkStreamSynth));
    let response = send(app, "POST", SPEECH_ROUTE, Some(BUFFERED_JSON)).await;
    assert_eq!(response.status(), StatusCode::OK);
    let expected_len = [CHUNK_A, CHUNK_B, CHUNK_C].concat().len() as u64;
    assert_eq!(
        response.into_body().size_hint().exact(),
        Some(expected_len),
        "the buffered (wav) shape must advertise a known exact size"
    );
}

// ----------------------------------------------------------------------------
// The default-synth streaming fallback keeps the buffered single-chunk shape.
// ----------------------------------------------------------------------------

#[tokio::test]
async fn test_default_synth_streaming_falls_back_to_single_chunk() {
    // SilentSynth provides no stream hook -> the handler falls back to the buffered
    // body wrapped in a single-chunk stream: 200, audio content-type, streaming shape.
    let response = send(router(), "POST", SPEECH_ROUTE, Some(STREAMING_JSON)).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(content_type(&response), "audio/wav");
    assert_eq!(
        response.into_body().size_hint().exact(),
        None,
        "the fallback streaming shape must not advertise a known exact size"
    );
}
