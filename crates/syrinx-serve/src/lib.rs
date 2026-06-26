//! syrinx-serve — Axum server, OpenAI-compatible `/v1/audio` (scaffold; T-08.01).
//!
//! This is the audio-server *scaffold*: the typed OpenAI-style speech request, a
//! pluggable [`Synth`] trait whose default stub ([`SilentSynth`]) returns a fixed
//! silent buffer, and the Axum [`router`] that exposes `POST /v1/audio/speech`.
//! There is no real synthesis behind the trait and no health/version endpoint —
//! those are out of scope for this task (see CLAUDE.md / list.md T-08.01).

use std::sync::Arc;

// The real CosyVoice2 end-to-end synthesizer (the `real`-feature capstone): wires
// the five parity-verified sub-models into one `text + voice -> 24 kHz audio`
// pipeline. Compiled only under the crate's `real` feature, on the model box, where
// the weights + fixtures live; the default Axum scaffold build stays Candle-free.
#[cfg(feature = "real")]
pub mod synth;

// WAV read/resample/encode helpers for the `real` surfaces (shared with the CLI).
// Candle-free in shape but `hound`-backed, so it rides the same `real` gate.
#[cfg(feature = "real")]
pub mod wavio;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::Response;
use axum::routing::{get, post};
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

/// The **real** synth: drives the CosyVoice2 [`Synthesizer`](crate::synth::Synthesizer)
/// for every request and returns the 24 kHz audio encoded as WAV bytes.
///
/// The OpenAI-style [`SpeechRequest`] carries only `input` text and a named
/// `voice` — there is no per-request reference clip — so a `RealSynth` is
/// constructed **bound to one reference voice** (the prompt transcript + its
/// already-resampled 16 kHz/24 kHz waveforms). Each request synthesizes
/// `req.input` *in that voice*; `req.model`/`req.voice` are currently advisory
/// (single configured voice). Use [`wavio::read_ref_wav`](crate::wavio::read_ref_wav)
/// to load the reference clip.
///
/// [`Synthesizer::synthesize`](crate::synth::Synthesizer::synthesize) takes
/// `&mut self`, so the loaded models live behind a [`std::sync::Mutex`]; requests
/// to a single `RealSynth` are therefore serialized (the sensible default for a
/// single-GPU/CPU TTS box).
#[cfg(feature = "real")]
pub struct RealSynth {
    synth: std::sync::Mutex<crate::synth::Synthesizer>,
    prompt_text: String,
    ref_wav_16k: Vec<f32>,
    ref_wav_24k: Vec<f32>,
    seed: u64,
    max_gen_steps: Option<usize>,
}

#[cfg(feature = "real")]
impl RealSynth {
    /// Build a `RealSynth` from a loaded [`Synthesizer`](crate::synth::Synthesizer)
    /// and a reference voice (`prompt_text` + its resampled 16 kHz/24 kHz mono
    /// waveforms). Defaults: `seed = 0`, no live-LM step cap (the real ratio).
    pub fn new(
        synth: crate::synth::Synthesizer,
        prompt_text: impl Into<String>,
        ref_wav_16k: Vec<f32>,
        ref_wav_24k: Vec<f32>,
    ) -> Self {
        Self {
            synth: std::sync::Mutex::new(synth),
            prompt_text: prompt_text.into(),
            ref_wav_16k,
            ref_wav_24k,
            seed: 0,
            max_gen_steps: None,
        }
    }

    /// Set the LM sampling seed (default `0`).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Cap the live LM generation steps (default `None` = the real `(text)*20`
    /// ratio). A cap keeps CPU runs tractable; see `synth::SynthInputs`.
    pub fn with_max_gen_steps(mut self, max_gen_steps: Option<usize>) -> Self {
        self.max_gen_steps = max_gen_steps;
        self
    }
}

#[cfg(feature = "real")]
impl Synth for RealSynth {
    fn synthesize(&self, req: &SpeechRequest) -> Vec<u8> {
        use crate::synth::SynthInputs;
        let inputs = SynthInputs {
            lm_seed: self.seed,
            max_gen_steps: self.max_gen_steps,
            ..Default::default()
        };
        // The Synth trait has no error channel; on a poisoned lock or a synth
        // failure we log and return an empty body rather than panic the handler.
        let mut synth = match self.synth.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        let wav = match synth.synthesize(
            &req.input,
            &self.prompt_text,
            &self.ref_wav_16k,
            &self.ref_wav_24k,
            &inputs,
        ) {
            Ok(wav) => wav,
            Err(e) => {
                eprintln!("syrinx-serve: synthesis failed: {e}");
                return Vec::new();
            }
        };
        match crate::wavio::encode_wav_24k(&wav) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!("syrinx-serve: wav encode failed: {e}");
                Vec::new()
            }
        }
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

/// The documented typed JSON health body. `status` is the literal health marker
/// `"ok"`; `version` is a non-empty version string identifying the build.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Health {
    pub status: String,
    pub version: String,
}

/// Build the OpenAI-compatible audio router wired with the default silent-buffer
/// synth. Registers `POST /v1/audio/speech` and `GET /v1/health` exactly once each.
pub fn router() -> Router {
    router_with_synth(Arc::new(SilentSynth))
}

/// Build the audio router with a caller-supplied synth, proving the synth is a
/// pluggable trait.
pub fn router_with_synth(synth: Arc<dyn Synth>) -> Router {
    Router::new()
        .route("/v1/audio/speech", post(speech))
        .route("/v1/health", get(health))
        .with_state(synth)
}

/// Build the audio router wired to a [`RealSynth`], so `POST /v1/audio/speech`
/// returns **real** CosyVoice2 audio (24 kHz WAV) in the configured reference
/// voice instead of the silent stub. Convenience over [`router_with_synth`].
#[cfg(feature = "real")]
pub fn router_with_real_synth(synth: RealSynth) -> Router {
    router_with_synth(Arc::new(synth))
}

/// Boot the OpenAI-compatible audio server around `synth` and serve it **blocking**
/// on `addr` until the process is killed. Runs a self-contained multi-thread Tokio
/// runtime internally so a synchronous caller (e.g. the `syrinx serve` CLI command)
/// can launch the server without being async itself. Returns only on bind/serve
/// error.
#[cfg(feature = "real")]
pub fn serve_blocking(synth: RealSynth, addr: std::net::SocketAddr) -> std::io::Result<()> {
    let app = router_with_real_synth(synth);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await
    })
}

/// Handle `GET /v1/health`: return 200 with the documented typed JSON health body
/// reporting status `"ok"` and the crate's build version.
async fn health() -> Response {
    let body = Health {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    };
    let json = serde_json::to_vec(&body).expect("the typed health body must serialize");
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json))
        .expect("a health response must build")
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
