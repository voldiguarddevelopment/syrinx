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

// The real CosyVoice3 end-to-end synthesizer: ties the four parity-verified CV3
// component ports (v3 speech tokenizer, Cv3Lm, Cv3Flow, Cv3Hift) plus the reused CV2
// frontend pieces into one `text + voice -> 24 kHz audio` pipeline. Additive to (and
// mirrors the structure of) `synth`; same `real` gate, on the model box.
#[cfg(feature = "real")]
pub mod synth_cv3;

// WAV read/resample/encode helpers for the `real` surfaces (shared with the CLI).
// Candle-free in shape but `hound`-backed, so it rides the same `real` gate.
#[cfg(feature = "real")]
pub mod wavio;

// Spread-spectrum output watermark (the README's "post-edit-detectable watermark
// on every synthesized output"). Pure-Rust, training-free, model-free — so it is
// NON-optional (no `real` gate): embed/detect work on any 24 kHz mono `f32` buffer
// and are unit-testable at the repo root without the model. The `real` synth path
// uses it via `Synthesizer::synthesize_watermarked`.
pub mod watermark;

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
    /// Buffered synthesis: produce the **complete** audio body for `req` in one
    /// `Vec`. An **empty** return value signals a load/synth failure — the speech
    /// handler maps it to an HTTP 500 with a typed [`ApiError`] rather than a 200
    /// with an empty body.
    fn synthesize(&self, req: &SpeechRequest) -> Vec<u8>;

    /// Optional **streaming** hook: produce the audio body incrementally as an
    /// iterator of byte chunks (emitted in order; concatenating them yields the
    /// full body). Returning `None` (the default) means "no native streaming" — the
    /// handler falls back to the buffered [`synthesize`](Synth::synthesize) wrapped
    /// in a single-chunk stream.
    ///
    /// Implementors that stream (e.g. [`RealSynth`] under the `real` feature) emit a
    /// WAV header chunk first, then PCM frame chunks (see the wire-format docs on
    /// the implementing type), giving low first-byte latency for
    /// `response_format == "stream"`.
    fn synthesize_stream(
        &self,
        _req: &SpeechRequest,
    ) -> Option<Box<dyn Iterator<Item = Vec<u8>> + Send>> {
        None
    }
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
    // `Arc<Mutex<…>>` (not a bare `Mutex`) so the streaming hook can hand a clone of
    // the loaded models to a producer thread that drives `synthesize_streaming` and
    // pushes chunks back over a channel; the buffered path locks it exactly as before.
    synth: Arc<std::sync::Mutex<crate::synth::Synthesizer>>,
    prompt_text: String,
    ref_wav_16k: Vec<f32>,
    ref_wav_24k: Vec<f32>,
    seed: u64,
    max_gen_steps: Option<usize>,
}

/// Number of finalized speech tokens per streamed chunk for [`RealSynth`]'s
/// streaming hook (the `token_hop` fed to
/// [`Synthesizer::synthesize_streaming`](crate::synth::Synthesizer::synthesize_streaming)).
/// Smaller = lower first-byte latency, more boundary cross-fades; this is the
/// CosyVoice2 default chunk size.
#[cfg(feature = "real")]
pub const STREAM_TOKEN_HOP: usize = 25;

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
            synth: Arc::new(std::sync::Mutex::new(synth)),
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

    /// Native streaming: drive [`Synthesizer::synthesize_streaming`] on a producer
    /// thread and surface each emitted audio chunk over a channel, so the HTTP layer
    /// can begin sending bytes long before synthesis finishes (low first-byte latency).
    ///
    /// ## Wire format (streamed body)
    ///
    /// 1. **One 44-byte canonical WAV header** (`RIFF`/`WAVE`, PCM, **24 kHz, mono,
    ///    16-bit**) with the RIFF and `data` chunk sizes set to `0xFFFFFFFF` — the
    ///    standard "unknown/streaming length" sentinel, since the total length is not
    ///    known when the first byte is sent.
    /// 2. **Then raw little-endian signed 16-bit PCM frames**, one channel, in order.
    ///    Each subsequent chunk is the PCM encoding of one
    ///    [`AudioChunk`](syrinx_acoustic::real::AudioChunk) (samples clamped to
    ///    `[-1, 1]`, scaled by `32767`). Concatenating the header + every PCM chunk
    ///    yields a well-formed (if length-sentinel) 24 kHz mono WAV stream.
    ///
    /// The models are serialized through the same lock as the buffered path: the
    /// producer thread holds the lock for the whole streamed synthesis.
    fn synthesize_stream(
        &self,
        req: &SpeechRequest,
    ) -> Option<Box<dyn Iterator<Item = Vec<u8>> + Send>> {
        use crate::synth::{SynthError, SynthInputs};

        let (tx, rx) = std::sync::mpsc::channel::<Vec<u8>>();
        let synth = Arc::clone(&self.synth);
        let prompt_text = self.prompt_text.clone();
        let ref_wav_16k = self.ref_wav_16k.clone();
        let ref_wav_24k = self.ref_wav_24k.clone();
        let input = req.input.clone();
        let seed = self.seed;
        let max_gen_steps = self.max_gen_steps;

        std::thread::spawn(move || {
            let inputs = SynthInputs {
                lm_seed: seed,
                max_gen_steps,
                ..Default::default()
            };
            let mut guard = match synth.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            // Header first: a streaming-length 24 kHz/mono/16-bit WAV header.
            if tx.send(wav_stream_header_24k_mono()).is_err() {
                return; // receiver dropped (client gone) — nothing to do.
            }
            let result = guard.synthesize_streaming(
                &input,
                &prompt_text,
                &ref_wav_16k,
                &ref_wav_24k,
                &inputs,
                STREAM_TOKEN_HOP,
                |chunk: Vec<f32>| {
                    // Emit raw little-endian PCM16 frames for this chunk.
                    tx.send(pcm16le_bytes(&chunk))
                        .map_err(|_| SynthError::Candle("stream receiver dropped".into()))
                },
            );
            if let Err(e) = result {
                eprintln!("syrinx-serve: streaming synthesis failed: {e}");
            }
        });

        Some(Box::new(rx.into_iter()))
    }
}

/// Build the 44-byte canonical WAV header for a **24 kHz, mono, 16-bit PCM** stream
/// whose total length is not yet known: the RIFF and `data` chunk sizes use the
/// `0xFFFFFFFF` streaming sentinel.
#[cfg(feature = "real")]
fn wav_stream_header_24k_mono() -> Vec<u8> {
    const SAMPLE_RATE: u32 = 24_000;
    const CHANNELS: u16 = 1;
    const BITS: u16 = 16;
    let byte_rate = SAMPLE_RATE * CHANNELS as u32 * (BITS as u32 / 8);
    let block_align = CHANNELS * (BITS / 8);

    let mut h = Vec::with_capacity(44);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // streaming-unknown RIFF size
    h.extend_from_slice(b"WAVE");
    h.extend_from_slice(b"fmt ");
    h.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    h.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
    h.extend_from_slice(&CHANNELS.to_le_bytes());
    h.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    h.extend_from_slice(&byte_rate.to_le_bytes());
    h.extend_from_slice(&block_align.to_le_bytes());
    h.extend_from_slice(&BITS.to_le_bytes());
    h.extend_from_slice(b"data");
    h.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // streaming-unknown data size
    h
}

/// Encode a chunk of 24 kHz mono `f32` samples (in `[-1, 1]`) as raw little-endian
/// signed 16-bit PCM bytes — the per-chunk payload of the streamed body.
#[cfg(feature = "real")]
fn pcm16le_bytes(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// A **multi-voice** synth: routes each request to one of several configured
/// [`RealSynth`] reference voices by `req.voice`, falling back to the default voice
/// when `req.voice` names no configured voice.
///
/// `router_with_synth(Arc::new(multi))` therefore serves several reference voices
/// from one server. Each named voice is an independent [`RealSynth`] (its own loaded
/// models + reference clip), so requests to different voices are independent and
/// requests to the *same* voice serialize on that voice's lock (as for a single
/// `RealSynth`). [`RealSynth`] itself (single voice) is unchanged.
#[cfg(feature = "real")]
pub struct MultiVoiceSynth {
    voices: std::collections::HashMap<String, RealSynth>,
    default_voice: String,
}

#[cfg(feature = "real")]
impl MultiVoiceSynth {
    /// Create a registry seeded with its **default** voice: requests whose `voice`
    /// matches no registered name are served by this one. The default is registered
    /// under `default_voice`, so a request naming it routes here too.
    pub fn new(default_voice: impl Into<String>, default_synth: RealSynth) -> Self {
        let default_voice = default_voice.into();
        let mut voices = std::collections::HashMap::new();
        voices.insert(default_voice.clone(), default_synth);
        Self {
            voices,
            default_voice,
        }
    }

    /// Register an additional named voice (builder style). A name already present
    /// (including the default) is replaced.
    pub fn with_voice(mut self, name: impl Into<String>, synth: RealSynth) -> Self {
        self.voices.insert(name.into(), synth);
        self
    }

    /// The number of distinct registered voices (including the default).
    pub fn voice_count(&self) -> usize {
        self.voices.len()
    }

    /// Resolve a request's `voice` to a configured [`RealSynth`], falling back to the
    /// default voice when the name is unknown. The default is always present, so the
    /// lookup cannot fail.
    fn pick(&self, voice: &str) -> &RealSynth {
        self.voices
            .get(voice)
            .unwrap_or_else(|| &self.voices[&self.default_voice])
    }
}

#[cfg(feature = "real")]
impl Synth for MultiVoiceSynth {
    fn synthesize(&self, req: &SpeechRequest) -> Vec<u8> {
        self.pick(&req.voice).synthesize(req)
    }

    fn synthesize_stream(
        &self,
        req: &SpeechRequest,
    ) -> Option<Box<dyn Iterator<Item = Vec<u8>> + Send>> {
        self.pick(&req.voice).synthesize_stream(req)
    }
}

/// The **real CosyVoice3** synth: drives the CV3 [`Cv3Synthesizer`](crate::synth_cv3::Cv3Synthesizer)
/// for every request and returns the 24 kHz audio encoded as WAV bytes — the CV3
/// counterpart of [`RealSynth`], implementing the same [`Synth`] trait so it slots into
/// the identical Axum routes (`router_with_synth` / `MultiVoiceSynth` route by `req.model`/
/// `req.voice`; this is additive, the CV2 paths are untouched).
///
/// Constructed **bound to one reference voice** (prompt transcript + its already-resampled
/// 16 kHz/24 kHz waveforms), exactly like [`RealSynth`]. Each request synthesizes
/// `req.input` in that voice; `req.model`/`req.voice` are advisory (single configured voice).
///
/// The CV3 synthesizer has **no native chunk-streaming path** (only the buffered
/// `synthesize`), so [`Cv3RealSynth`] does not override
/// [`Synth::synthesize_stream`] — a `response_format == "stream"` request therefore
/// uses the handler's buffered-fallback (the whole body wrapped in one chunk).
/// [`Cv3Synthesizer::synthesize`](crate::synth_cv3::Cv3Synthesizer::synthesize) takes
/// `&mut self`, so the loaded models live behind a [`std::sync::Mutex`]; requests to a
/// single `Cv3RealSynth` are serialized (the single-GPU/CPU default).
#[cfg(feature = "real")]
pub struct Cv3RealSynth {
    synth: Arc<std::sync::Mutex<crate::synth_cv3::Cv3Synthesizer>>,
    prompt_text: String,
    ref_wav_16k: Vec<f32>,
    ref_wav_24k: Vec<f32>,
    seed: u64,
    max_gen_steps: Option<usize>,
}

#[cfg(feature = "real")]
impl Cv3RealSynth {
    /// Build a `Cv3RealSynth` from a loaded [`Cv3Synthesizer`](crate::synth_cv3::Cv3Synthesizer)
    /// and a reference voice (`prompt_text` + its resampled 16 kHz/24 kHz mono
    /// waveforms). Defaults: `seed = 0`, no live-LM step cap (the real ratio).
    pub fn new(
        synth: crate::synth_cv3::Cv3Synthesizer,
        prompt_text: impl Into<String>,
        ref_wav_16k: Vec<f32>,
        ref_wav_24k: Vec<f32>,
    ) -> Self {
        Self {
            synth: Arc::new(std::sync::Mutex::new(synth)),
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

    /// Cap the live LM generation steps (default `None` = the real `|tok(tts_text)|*20`
    /// ratio). A cap keeps CPU runs tractable; see `synth_cv3::Cv3SynthInputs`.
    pub fn with_max_gen_steps(mut self, max_gen_steps: Option<usize>) -> Self {
        self.max_gen_steps = max_gen_steps;
        self
    }
}

#[cfg(feature = "real")]
impl Synth for Cv3RealSynth {
    fn synthesize(&self, req: &SpeechRequest) -> Vec<u8> {
        let inputs = crate::synth_cv3::Cv3SynthInputs {
            lm_seed: self.seed,
            max_gen_steps: self.max_gen_steps,
            ..Default::default()
        };
        // The Synth trait has no error channel; on a poisoned lock or a synth failure we
        // log and return an empty body (the handler maps it to a typed 500) rather than panic.
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
                eprintln!("syrinx-serve: cv3 synthesis failed: {e}");
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

/// The typed JSON body of `GET /v1/version`: the crate `name` (`CARGO_PKG_NAME`)
/// and its build `version` (`CARGO_PKG_VERSION`). Both are non-empty.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Version {
    pub name: String,
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
        .route("/v1/version", get(version))
        .with_state(synth)
}

/// Build the audio router wired to a [`RealSynth`], so `POST /v1/audio/speech`
/// returns **real** CosyVoice2 audio (24 kHz WAV) in the configured reference
/// voice instead of the silent stub. Convenience over [`router_with_synth`].
#[cfg(feature = "real")]
pub fn router_with_real_synth(synth: RealSynth) -> Router {
    router_with_synth(Arc::new(synth))
}

/// Build the audio router wired to a [`Cv3RealSynth`], so `POST /v1/audio/speech`
/// returns **real CosyVoice3** audio (24 kHz WAV) in the configured reference voice.
/// The CV3 counterpart of [`router_with_real_synth`]; `Cv3RealSynth` is just a
/// [`Synth`], so this is a thin convenience over [`router_with_synth`].
#[cfg(feature = "real")]
pub fn router_with_cv3_synth(synth: Cv3RealSynth) -> Router {
    router_with_synth(Arc::new(synth))
}

/// Boot the OpenAI-compatible audio server around `synth` and serve it **blocking**
/// on `addr` until the process is killed. Runs a self-contained multi-thread Tokio
/// runtime internally so a synchronous caller (e.g. the `syrinx serve` CLI command)
/// can launch the server without being async itself. Returns only on bind/serve
/// error.
#[cfg(feature = "real")]
pub fn serve_blocking(synth: RealSynth, addr: std::net::SocketAddr) -> std::io::Result<()> {
    serve_router_blocking(router_with_real_synth(synth), addr)
}

/// CV3 counterpart of [`serve_blocking`]: boot the OpenAI-compatible audio server
/// backed by a [`Cv3RealSynth`] and serve it **blocking** on `addr`. Used by the
/// `syrinx serve --cv3` CLI command. Same routes + runtime as the CV2 path.
#[cfg(feature = "real")]
pub fn serve_blocking_cv3(synth: Cv3RealSynth, addr: std::net::SocketAddr) -> std::io::Result<()> {
    serve_router_blocking(router_with_cv3_synth(synth), addr)
}

/// Shared blocking-serve core: run `app` on a self-contained multi-thread Tokio
/// runtime, binding `addr`. The CV2 ([`serve_blocking`]) and CV3
/// ([`serve_blocking_cv3`]) entry points differ only in which router they pass.
#[cfg(feature = "real")]
fn serve_router_blocking(app: Router, addr: std::net::SocketAddr) -> std::io::Result<()> {
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

/// Handle `GET /v1/version`: return 200 with the typed JSON `{name, version}` body
/// reporting the crate name and its build version.
async fn version() -> Response {
    let body = Version {
        name: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    };
    let json = serde_json::to_vec(&body).expect("the typed version body must serialize");
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json))
        .expect("a version response must build")
}

/// Handle `POST /v1/audio/speech`: deserialize + validate the typed request, run the
/// pluggable synth, and return either a buffered or a (true-)streaming audio
/// response — with typed [`ApiError`] bodies for malformed, invalid, and failed
/// requests:
///   * deserialization failure (e.g. a missing required field) -> **422**,
///   * a blank `input` -> **400**,
///   * the synth returning an empty buffered body (load/synth failure) -> **500**.
async fn speech(State(synth): State<Arc<dyn Synth>>, body: Bytes) -> Response {
    let request: SpeechRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        // A malformed/incomplete body is a typed 422 (unchanged contract).
        Err(err) => {
            return api_error_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                err.to_string(),
                "invalid_request_error",
            )
        }
    };

    // Validation: a present-but-blank `input` is a typed 400 (a missing `input`
    // already failed deserialization above as a 422).
    if request.input.trim().is_empty() {
        return api_error_response(
            StatusCode::BAD_REQUEST,
            "input must not be empty".to_string(),
            "invalid_request_error",
        );
    }

    match request.response_format {
        ResponseFormat::Wav => {
            let audio = synth.synthesize(&request);
            if audio.is_empty() {
                return synth_failure_response();
            }
            buffered_response(audio)
        }
        ResponseFormat::Stream => {
            // Prefer a native streaming source (emits chunks as produced); otherwise
            // fall back to the buffered body wrapped in a single-chunk stream.
            if let Some(chunks) = synth.synthesize_stream(&request) {
                return chunked_response(chunks);
            }
            let audio = synth.synthesize(&request);
            if audio.is_empty() {
                return synth_failure_response();
            }
            streaming_response(audio)
        }
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

/// A streamed body whose size is not known up front (the streaming shape): the
/// buffered-fallback path, where the whole body is one already-produced chunk.
fn streaming_response(audio: Vec<u8>) -> Response {
    let stream = futures_util::stream::once(async move { Ok::<Vec<u8>, std::io::Error>(audio) });
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "audio/wav")
        .body(Body::from_stream(stream))
        .expect("a streaming audio response must build")
}

/// A truly **chunked** streamed body: the audio is emitted chunk-by-chunk from the
/// synth's streaming iterator (header chunk, then PCM chunks) as it is produced, so
/// the size is not known up front.
fn chunked_response(chunks: Box<dyn Iterator<Item = Vec<u8>> + Send>) -> Response {
    let stream = futures_util::stream::iter(chunks.map(Ok::<Vec<u8>, std::io::Error>));
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "audio/wav")
        .body(Body::from_stream(stream))
        .expect("a chunked audio response must build")
}

/// A 500 with a typed JSON error body when the synth produces no audio (a load or
/// synthesis failure), instead of a 200 carrying an empty body.
fn synth_failure_response() -> Response {
    api_error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        "audio synthesis failed".to_string(),
        "synthesis_error",
    )
}

/// A typed JSON [`ApiError`] body at an explicit `status` with an explicit error
/// `kind`.
fn api_error_response(status: StatusCode, message: String, kind: &str) -> Response {
    let error = ApiError {
        error: ApiErrorBody {
            message,
            kind: kind.to_string(),
        },
    };
    let json = serde_json::to_vec(&error).expect("the typed error must serialize");
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json))
        .expect("an error response must build")
}
