//! syrinx-serve — Axum server, OpenAI-compatible `/v1/audio` (scaffold; T-08.01).
//!
//! This is the audio-server *scaffold*: the typed OpenAI-style speech request, a
//! pluggable [`Synth`] trait whose default stub ([`SilentSynth`]) returns a fixed
//! silent buffer, and the Axum [`router`] that exposes `POST /v1/audio/speech`.
//! There is no real synthesis behind the trait and no health/version endpoint —
//! those are out of scope for this task (see CLAUDE.md / list.md T-08.01).

use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::Response;
use axum::routing::post;
use axum::Router;
use serde::{Deserialize, Serialize};

/// Fixed length (in bytes) of the silent buffer the default stub synth emits.
pub const SILENT_BUFFER_LEN: usize = 1024;

/// Buffered vs. streaming selector for a speech request. JSON wire form is
/// snake_case: `"wav"` and `"stream"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    /// A single buffered WAV body.
    Wav,
    /// A streamed body.
    Stream,
}

/// The typed OpenAI-style speech request. All four fields are required; a
/// missing field fails deserialization.
#[derive(Debug, Clone, Deserialize)]
pub struct SpeechRequest {
    pub model: String,
    pub input: String,
    pub voice: String,
    pub response_format: ResponseFormat,
}

/// The pluggable synth: turns a request into raw audio bytes.
pub trait Synth: Send + Sync {
    fn synthesize(&self, req: &SpeechRequest) -> Vec<u8>;
}

/// The default stub synth: returns a fixed silent buffer of [`SILENT_BUFFER_LEN`]
/// zero bytes regardless of the request.
pub struct SilentSynth;

impl Synth for SilentSynth {
    fn synthesize(&self, _req: &SpeechRequest) -> Vec<u8> {
        vec![0u8; SILENT_BUFFER_LEN]
    }
}

/// The typed JSON error body returned for a malformed request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    pub error: ApiErrorBody,
}

/// The inner error payload of an [`ApiError`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiErrorBody {
    pub message: String,
    pub kind: String,
}

/// Build the OpenAI-compatible audio router wired with the default silent-buffer
/// synth. Registers `POST /v1/audio/speech` exactly once.
pub fn router() -> Router {
    router_with_synth(Arc::new(SilentSynth))
}

/// Build the audio router with a caller-supplied synth, proving the synth is a
/// pluggable trait.
pub fn router_with_synth(synth: Arc<dyn Synth>) -> Router {
    Router::new()
        .route("/v1/audio/speech", post(speech))
        .with_state(synth)
}

/// Handle `POST /v1/audio/speech`: deserialize the typed request, run the
/// pluggable synth, and return either a buffered or a streaming audio response.
async fn speech(State(synth): State<Arc<dyn Synth>>, body: Bytes) -> Response {
    let request: SpeechRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(err) => return error_response(err.to_string()),
    };

    let audio = synth.synthesize(&request);

    match request.response_format {
        ResponseFormat::Wav => buffered_response(audio),
        ResponseFormat::Stream => streaming_response(audio),
    }
}

/// A single buffered body of known exact size, with the audio content-type.
fn buffered_response(audio: Vec<u8>) -> Response {
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "audio/wav")
        .body(Body::from(audio))
        .expect("a buffered audio response must build")
}

/// A streamed body whose size is not known up front (the streaming shape).
fn streaming_response(audio: Vec<u8>) -> Response {
    let stream = futures_util::stream::once(async move { Ok::<Vec<u8>, std::io::Error>(audio) });
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "audio/wav")
        .body(Body::from_stream(stream))
        .expect("a streaming audio response must build")
}

/// A 422 with a typed JSON error body for a malformed request.
fn error_response(message: String) -> Response {
    let error = ApiError {
        error: ApiErrorBody {
            message,
            kind: "invalid_request_error".to_string(),
        },
    };
    let json = serde_json::to_vec(&error).expect("the typed error must serialize");
    Response::builder()
        .status(StatusCode::UNPROCESSABLE_ENTITY)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json))
        .expect("an error response must build")
}
