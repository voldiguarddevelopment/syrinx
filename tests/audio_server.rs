//! Frozen RED tests for T-08.01 — scaffold the audio server.
//!
//! These pin the four acceptance criteria against the real public API the green
//! phase must build in `syrinx-serve`:
//!
//!   * `router() -> axum::Router` — the OpenAI-compatible audio router, wired
//!     with the default silent-buffer synth. It registers the speech route at
//!     `/v1/audio/speech` for `POST` exactly once.
//!   * `router_with_synth(Arc<dyn Synth>) -> axum::Router` — the same router with
//!     a caller-supplied synth, proving the synth is a *pluggable* trait.
//!   * `SpeechRequest { model, input, voice, response_format }` — the typed
//!     OpenAI-style speech request. All four fields are required; a missing field
//!     fails deserialization. `response_format` is a `ResponseFormat`.
//!   * `ResponseFormat::{Wav, Stream}` — buffered vs. streaming selector. JSON
//!     wire form is snake_case (`"wav"`, `"stream"`).
//!   * `Synth` — the pluggable synth trait: `fn synthesize(&self, &SpeechRequest)
//!     -> Vec<u8>`. `SilentSynth` is the default stub; it returns a *fixed*
//!     silent buffer (`SILENT_BUFFER_LEN` zero bytes) regardless of the request.
//!   * `ApiError { error: ApiErrorBody { message, kind } }` — the typed JSON
//!     error body returned with a 422 for a malformed request.
//!
//! Contract (list.md / DESIGN §T8.1): a well-formed POST to `/v1/audio/speech`
//! returns 200 with an audio content-type whose body is the stub synth's output;
//! a request missing a required field returns 422 with a typed JSON error body; a
//! request whose `response_format` selects streaming returns the streaming
//! response shape (a body of unknown/streamed size) rather than a single buffered
//! body (a body of known exact size); and the speech route is registered exactly
//! once (building the router does not panic on a duplicate, the path answers
//! `POST`, rejects other methods with 405, and unknown paths 404).
//!
//! RED: `syrinx-serve` exposes no router/request/synth types yet, so none of
//! these symbols resolve and the test target fails to build — every criterion is
//! unmet. GREEN implements the scaffold so each assertion below holds.

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, Response, StatusCode};
use http_body::Body as _; // brings `size_hint` into scope for the streaming-shape check
use tower::ServiceExt; // brings `oneshot` into scope

use syrinx_serve::{
    router, router_with_synth, ApiError, ResponseFormat, SilentSynth, SpeechRequest, Synth,
    SILENT_BUFFER_LEN,
};

/// The audited speech route — registered exactly once for `POST`.
const SPEECH_ROUTE: &str = "/v1/audio/speech";

/// A well-formed buffered request body (all four required fields, `wav` format).
const BUFFERED_JSON: &str =
    r#"{"model":"syrinx-1","input":"hello world","voice":"alloy","response_format":"wav"}"#;

/// A well-formed streaming request body (`stream` format selects the stream shape).
const STREAMING_JSON: &str =
    r#"{"model":"syrinx-1","input":"hello world","voice":"alloy","response_format":"stream"}"#;

/// A malformed request body: the required `voice` field is omitted.
const MISSING_VOICE_JSON: &str =
    r#"{"model":"syrinx-1","input":"hello world","response_format":"wav"}"#;

/// A test-only synth proving the trait is pluggable: it returns a fixed, *non*-
/// silent marker buffer distinct from `SilentSynth`'s output so a response that
/// carries these bytes can only have come from this injected synth.
struct MarkerSynth;

/// The bytes `MarkerSynth` emits — non-zero and a different length than the stub.
const MARKER_BYTES: &[u8] = &[7, 7, 7, 7, 7];

impl Synth for MarkerSynth {
    fn synthesize(&self, _req: &SpeechRequest) -> Vec<u8> {
        MARKER_BYTES.to_vec()
    }
}

/// Build a `SpeechRequest` value for the in-process synth checks.
fn sample_request() -> SpeechRequest {
    SpeechRequest {
        model: "syrinx-1".to_string(),
        input: "hello world".to_string(),
        voice: "alloy".to_string(),
        response_format: ResponseFormat::Wav,
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

/// Read the response's content-type header as a string.
fn content_type(response: &Response<Body>) -> String {
    response
        .headers()
        .get(header::CONTENT_TYPE)
        .expect("a response must carry a content-type header")
        .to_str()
        .expect("the content-type header must be valid UTF-8")
        .to_string()
}

/// Collect a response body fully into bytes.
async fn collect(response: Response<Body>) -> Vec<u8> {
    to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the response body must collect")
        .to_vec()
}

// ----------------------------------------------------------------------------
// C1 — typed request + pluggable synth trait whose default stub is a fixed
//      silent buffer.
// ----------------------------------------------------------------------------

/// The default `SilentSynth` stub returns a *fixed* silent buffer: exactly
/// `SILENT_BUFFER_LEN` (> 0) zero bytes, independent of the request it is given.
#[test]
fn test_default_stub_returns_fixed_silent_buffer() {
    assert!(
        SILENT_BUFFER_LEN > 0,
        "the silent buffer must have a positive fixed length"
    );

    let stub = SilentSynth;
    let out = stub.synthesize(&sample_request());

    assert_eq!(
        out.len(),
        SILENT_BUFFER_LEN,
        "the stub buffer length must equal SILENT_BUFFER_LEN"
    );
    assert!(
        out.iter().all(|&b| b == 0),
        "every byte of the silent buffer must be zero"
    );

    // "Fixed": a different request must yield byte-identical output.
    let mut other = sample_request();
    other.model = "syrinx-2".to_string();
    other.input = "a completely different utterance".to_string();
    other.voice = "echo".to_string();
    other.response_format = ResponseFormat::Stream;
    assert_eq!(
        stub.synthesize(&other),
        out,
        "the stub buffer must be fixed regardless of the request"
    );
}

/// The handler calls a *pluggable* synth trait: injecting a custom synth makes
/// the 200 response body carry that synth's bytes rather than the silent stub's.
#[tokio::test]
async fn test_handler_calls_pluggable_synth_trait() {
    let app = router_with_synth(Arc::new(MarkerSynth));
    let response = send(app, "POST", SPEECH_ROUTE, Some(BUFFERED_JSON)).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = collect(response).await;
    assert_eq!(
        body, MARKER_BYTES,
        "the response body must be the injected synth's output, proving the trait is pluggable"
    );
}

// ----------------------------------------------------------------------------
// C2 — well-formed POST -> 200 + audio content-type; missing field -> 422 typed
//      error body.
// ----------------------------------------------------------------------------

/// A well-formed POST returns 200 with an audio content-type, and the body is the
/// default stub's silent buffer.
#[tokio::test]
async fn test_well_formed_post_returns_200_audio() {
    let response = send(router(), "POST", SPEECH_ROUTE, Some(BUFFERED_JSON)).await;

    assert_eq!(response.status(), StatusCode::OK, "a well-formed POST is 200");
    assert_eq!(
        content_type(&response),
        "audio/wav",
        "the success response must carry the audio content-type"
    );

    let body = collect(response).await;
    assert_eq!(
        body.len(),
        SILENT_BUFFER_LEN,
        "the body must be the stub's silent buffer"
    );
    assert!(
        body.iter().all(|&b| b == 0),
        "the default stub body must be silent (all zero bytes)"
    );
}

/// A request missing a required field returns 422 with a typed JSON error body
/// that deserializes into `ApiError`.
#[tokio::test]
async fn test_missing_required_field_returns_422_typed_error() {
    let response = send(router(), "POST", SPEECH_ROUTE, Some(MISSING_VOICE_JSON)).await;

    assert_eq!(
        response.status(),
        StatusCode::UNPROCESSABLE_ENTITY,
        "a missing required field must be 422 Unprocessable Entity"
    );
    assert_eq!(
        content_type(&response),
        "application/json",
        "the error body must be typed JSON"
    );

    let body = collect(response).await;
    let err: ApiError =
        serde_json::from_slice(&body).expect("the 422 body must deserialize into the typed ApiError");
    assert_eq!(
        err.error.kind, "invalid_request_error",
        "the typed error must classify a malformed request"
    );
    assert!(
        !err.error.message.is_empty(),
        "the typed error must carry a non-empty message"
    );
}

// ----------------------------------------------------------------------------
// C3 — streaming response_format -> streaming shape, not a single buffered body.
// ----------------------------------------------------------------------------

/// When `response_format` selects streaming, the handler returns the streaming
/// response shape — a body of unknown/streamed size — whereas the buffered format
/// returns a single body of known exact size. Both succeed with 200; the size
/// hint distinguishes the two shapes.
#[tokio::test]
async fn test_streaming_format_returns_streaming_shape() {
    // Buffered: a single body whose exact size is known up front.
    let buffered = send(router(), "POST", SPEECH_ROUTE, Some(BUFFERED_JSON)).await;
    assert_eq!(buffered.status(), StatusCode::OK);
    assert_eq!(
        buffered.into_body().size_hint().exact(),
        Some(SILENT_BUFFER_LEN as u64),
        "the buffered shape must be a single body of known exact size"
    );

    // Streaming: the body size is not known up front — it is a stream.
    let streaming = send(router(), "POST", SPEECH_ROUTE, Some(STREAMING_JSON)).await;
    assert_eq!(streaming.status(), StatusCode::OK);
    assert_eq!(
        streaming.into_body().size_hint().exact(),
        None,
        "the streaming shape must not be a single buffered body of known size"
    );
}

// ----------------------------------------------------------------------------
// C4 — the speech route is registered exactly once.
// ----------------------------------------------------------------------------

/// The speech route is registered exactly once: `router()` builds without
/// panicking (a duplicate registration would panic), the path answers `POST`
/// (200, not 404), the same path rejects a non-POST method with 405 (it exists
/// for exactly that one method), and an unrelated path is 404.
#[tokio::test]
async fn test_speech_route_registered_exactly_once() {
    // Building the router must not panic — a duplicate route registration would.
    let post = send(router(), "POST", SPEECH_ROUTE, Some(BUFFERED_JSON)).await;
    assert_eq!(
        post.status(),
        StatusCode::OK,
        "the speech route must be registered and answer POST"
    );

    // The path exists for exactly one method: a GET to it is 405, not 404.
    let get = send(router(), "GET", SPEECH_ROUTE, None).await;
    assert_eq!(
        get.status(),
        StatusCode::METHOD_NOT_ALLOWED,
        "the registered path must reject non-POST methods with 405"
    );

    // An unrelated path under /v1/audio is not registered at all -> 404.
    let missing = send(router(), "POST", "/v1/audio/not-a-route", Some(BUFFERED_JSON)).await;
    assert_eq!(
        missing.status(),
        StatusCode::NOT_FOUND,
        "an unregistered path must be 404, proving the speech route is the only one"
    );
}
