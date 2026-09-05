//! **First execution of `syrinx_serve::synth_qwen`** — the Candle engine behind the
//! OpenAI-compatible route, driven against real Qwen3-TTS weights.
//!
//! `tests/qwen_server.rs` gates the model-free half: which checkpoint maps to which
//! generation mode, what each one does with an instruction, the wire contract, and the
//! hard no-cue-markup invariant — all with a recording [`QwenEngine`] in place of the
//! Candle one. That substitution is what makes it honest, and it is also its limit:
//! `synth_qwen.rs` was compile-verified and had **never loaded a checkpoint**
//! (`docs/backends/QWEN_PORT_STATUS.md` §"Still unexecuted"). This file is that run.
//!
//! It proves four things that only weights can prove:
//!
//! 1. **`QwenModelEngine::load` cross-checks before it commits.** The module claims a
//!    backend/checkpoint/voice mismatch "would otherwise surface as a shape error deep
//!    inside `realize_plan`", so it is caught at load. Verified by handing `load` a
//!    deliberately wrong pairing *and a nonexistent tokenizer directory*: the mismatch
//!    error must come back, which is only possible if the check runs before the talker
//!    weights and the codec bag are touched.
//! 2. **The route returns real audio.** A `POST /v1/audio/speech` through the actual Axum
//!    router (`tower::oneshot`, no port bound, exactly as `tests/qwen_server.rs` and
//!    `tests/audio_server.rs` drive it) answers 200 + `audio/wav` with a well-formed
//!    24 kHz mono PCM16 body of a plausible length.
//! 3. **The audio is speech.** Byte counts prove nothing about a vocoder, so the body is
//!    transcribed by the native Whisper oracle (`syrinx-stt`, no Python at inference) and
//!    scored by WER against the text that was requested.
//! 4. **No cue markup is spoken.** A cued request renders through the same path, and the
//!    transcript is held to `brackets <= escaped` — the same bound
//!    `crates/syrinx-cue/tests/projection_strip.rs` enforces on the lowered text, carried
//!    all the way through to what a listener actually hears. The source escapes nothing,
//!    so the bound here is the strictest form: zero brackets, and the style label must
//!    not be spoken either.
//!
//! The voice-clone branch (`QwenVoice::clone_from_wav` → in-context reference frames →
//! the `decode(cat(ref, generated))`-then-cut in `render_inner`) is the least-exercised
//! code in the module, so it gets its own render when a `-Base` checkpoint is configured.
//!
//! ## Running it
//!
//! CPU only — the board compiles `--features real` without `cuda`, and this test never
//! opens a device other than [`Device::Cpu`]. It self-skips cleanly without its env:
//!
//! ```text
//! source scripts/test-all.env
//! SYRINX_QWEN_CV_DIR_0_6B=/data/models/Qwen3-TTS-12Hz-0.6B-CustomVoice \
//! SYRINX_QWEN_BASE_DIR_0_6B=/data/models/Qwen3-TTS-12Hz-0.6B-Base \
//!   MEMMAX=10G scripts/run-isolated.sh \
//!   cargo test --features real --release --test real_qwen_serve -- --nocapture
//! ```
//!
//! The 0.6B checkpoints are deliberate: they are ~1.8 GB on disk against the 1.7B's
//! ~3.9 GB, and the CPU parity path upcasts to f32. `SYRINX_QWEN_CV_DIR` (the 1.7B) is
//! **not** used as a fallback — it would silently make a run three times the size, and
//! its known instruct-repeat defect would confuse the cued case.
//!
//! **Cost, measured on NovaBox 2026-09-05** (CPU/f32, `--features real --release`, warm
//! build, nothing else on the CPU): ~12.3 min in total — the three renders are 259 s,
//! 257 s and 214 s, everything else (two engine loads, the Mimi encode, three Whisper
//! passes) is under 10 s. That is why the test is **opt-in** (`scripts/test-groups.sh`,
//! `OPT_IN_TESTS`) rather than a member of `GROUP_qwen_ckpt`, whose four tests take 23 s
//! together. The weight-backed cases hold a process-wide mutex, so `cargo test`'s default
//! thread fan-out cannot put two multi-gigabyte f32 models in memory at once; the run
//! completes under a `MEMMAX=10G` cap — which is a bound the run stayed inside, not a
//! measured peak.

#![cfg(feature = "real")]

use std::path::Path;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, Response, StatusCode};
use candle_core::Device;
use tower::ServiceExt; // brings `oneshot` into scope

use syrinx_cue::BackendId;
use syrinx_qwen::model::DriveParams;
use syrinx_serve::qwen::{plan, router_with_qwen_synth, QWEN_SAMPLE_RATE};
use syrinx_serve::synth_qwen::{QwenModelEngine, QwenVoice};
use syrinx_stt::{wer, Stt};

const SPEECH_ROUTE: &str = "/v1/audio/speech";

/// The plain probe. Same sentence the first end-to-end renders used
/// (`renders/2026-09-03-qwen-first/`), which transcribed at WER 0.000 on this checkpoint
/// — so a WER regression here is the port, not the sentence.
const PLAIN_TEXT: &str = "Come closer, I have something to tell you.";

/// The cued probe. `[whisper]` is a `style` label in `syrinx-cue`'s vocab, so it lowers
/// to an utterance-scoped instruction and leaves the sentence behind; the sentence itself
/// shares no word with the label, so "the label was spoken" is distinguishable from "the
/// sentence was spoken". Eight words once lowered, like [`PLAIN_TEXT`] — see [`WER_MAX`],
/// whose bound is derived from that count.
const CUED_TEXT: &str = "[whisper] The garden gate was open again this morning.";

/// Frames per render, capped. A short English sentence lands around 40 frames at 12.5 Hz;
/// 200 is ~16 s of audio — far above anything these probes should produce, and low enough
/// that a runaway loop fails the duration assertion in minutes instead of hanging a CPU
/// box for an hour. This bounds the test, not the model: a render that needs more frames
/// than this is already broken.
const MAX_FRAMES: usize = 200;

/// Codec frame rate of the 12 Hz tokenizer, used only to turn [`MAX_FRAMES`] into the
/// duration ceiling. (`12.5`, not `12` — the checkpoint family is named for the rounded
/// figure.)
const FRAME_RATE_HZ: f64 = 12.5;

/// Shortest believable render, as a floor under a truncated generation.
///
/// **Measured** on the 0.6B checkpoints (CPU/f32, seed 0, 2026-09-05): plain **3.60 s**,
/// cued **3.52 s**, clone **2.80 s**. The floor is set at 36% of the shortest of those —
/// loose enough that a differently-sampled render on another box is not a false alarm,
/// tight enough to reject the two failures that matter: a generation that stopped at the
/// `min_new_frames = 2` EOS guard (0.16 s) or one that emitted a handful of frames.
const MIN_SECONDS: f64 = 1.0;

/// WER ceiling for a render against the text that was requested.
///
/// **Measured** on the 0.6B checkpoints (CPU/f32, seed 0, whisper-base oracle,
/// 2026-09-05): **0.0000** for all three renders — plain, cued and clone. The port was
/// already recorded at WER 0.000 for the 2026-09-03 renders in
/// `docs/backends/QWEN_PORT_STATUS.md`, so this is the expected figure, not a lucky draw.
///
/// The bound is therefore derived, not chosen: every probe is exactly **eight words**, so
/// `1/8` is the smallest non-zero score any of them can produce. The ceiling admits
/// exactly one wrong word and nothing more. That is not a flaky bound — `(seed, weights,
/// prompt)` reproduces the talker bit-for-bit and Whisper's greedy decode is
/// deterministic, so a run either reproduces 0.0000 or something changed. A silent or
/// babbling render scores ~1.0, eight times the bound.
const WER_MAX: f32 = 0.125;

// --------------------------------------------------------------------------- env

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.trim().is_empty())
}

fn env_path(k: &str) -> Option<String> {
    env(k).filter(|p| Path::new(p).exists())
}

/// The three directories every case in this file needs, or `None` (with the reason
/// printed) so the test self-skips.
fn common_env(test: &str) -> Option<(String, String, String)> {
    let Some(cv_dir) = env_path("SYRINX_QWEN_CV_DIR_0_6B") else {
        eprintln!(
            "SKIP {test}: set SYRINX_QWEN_CV_DIR_0_6B to the Qwen3-TTS-12Hz-0.6B-CustomVoice \
             checkpoint dir (the 1.7B is deliberately NOT a fallback — see the module docs)"
        );
        return None;
    };
    let Some(tok_dir) = env_path("SYRINX_QWEN_TOK_DIR") else {
        eprintln!("SKIP {test}: set SYRINX_QWEN_TOK_DIR to the Qwen3-TTS-Tokenizer-12Hz dir");
        return None;
    };
    let Some(stt_dir) = env_path("SYRINX_STT_MODEL_DIR") else {
        eprintln!(
            "SKIP {test}: set SYRINX_STT_MODEL_DIR to the Whisper model dir — the render is \
             scored by the native oracle, never by its byte count"
        );
        return None;
    };
    Some((cv_dir, tok_dir, stt_dir))
}

/// Serializes the weight-backed cases. `cargo test` runs test functions on as many
/// threads as the box has cores, and each case here holds a multi-gigabyte f32 model:
/// two of them alive at once is an OOM, not a slow run.
fn heavy() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

// --------------------------------------------------------------------------- wav

/// A decoded WAV body: everything the header claimed, plus the samples.
struct Wav {
    channels: u16,
    sample_rate: u32,
    bits: u16,
    samples: Vec<f32>,
}

impl Wav {
    fn seconds(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }
}

fn le_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}

fn le_u16(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(bytes[at..at + 2].try_into().expect("two bytes"))
}

/// Parse a canonical 44-byte-header PCM WAV, asserting every field of the header rather
/// than trusting the four magic bytes. A body whose `data` length disagrees with the
/// bytes present is exactly the shape of a truncated stream, so it is checked too.
fn parse_wav(body: &[u8]) -> Wav {
    assert!(body.len() > 44, "a WAV body must carry more than its header ({} bytes)", body.len());
    assert_eq!(&body[0..4], b"RIFF", "not a RIFF container");
    assert_eq!(
        le_u32(body, 4) as usize,
        body.len() - 8,
        "RIFF size disagrees with the body length"
    );
    assert_eq!(&body[8..12], b"WAVE");
    assert_eq!(&body[12..16], b"fmt ");
    assert_eq!(le_u32(body, 16), 16, "PCM fmt chunk is 16 bytes");
    assert_eq!(le_u16(body, 20), 1, "audio format must be uncompressed PCM");

    let channels = le_u16(body, 22);
    let sample_rate = le_u32(body, 24);
    let byte_rate = le_u32(body, 28);
    let block_align = le_u16(body, 32);
    let bits = le_u16(body, 34);
    assert_eq!(block_align, channels * (bits / 8), "block align disagrees with the format");
    assert_eq!(byte_rate, sample_rate * block_align as u32, "byte rate disagrees");

    assert_eq!(&body[36..40], b"data");
    let data_len = le_u32(body, 40) as usize;
    assert_eq!(data_len, body.len() - 44, "data chunk length disagrees with the body");
    assert_eq!(data_len % 2, 0, "a PCM16 data chunk has an even length");

    let samples = body[44..]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect();
    Wav { channels, sample_rate, bits, samples }
}

/// The wire contract every render must satisfy, before anything is said about its
/// content: the status, the media type, and a 24 kHz mono PCM16 body of plausible length.
fn accept_render(label: &str, response: Response<Body>, body: Vec<u8>) -> Wav {
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "{label}: the route must answer 200 (body: {})",
        String::from_utf8_lossy(&body)
    );
    let ct = response
        .headers()
        .get(header::CONTENT_TYPE)
        .expect("a response must carry a content-type")
        .to_str()
        .expect("the content-type must be UTF-8");
    assert_eq!(ct, "audio/wav", "{label}: wrong media type");

    let wav = parse_wav(&body);
    assert_eq!(wav.channels, 1, "{label}: the Qwen decoder is mono");
    assert_eq!(
        wav.sample_rate, QWEN_SAMPLE_RATE,
        "{label}: the 12 Hz tokenizer decodes at 24 kHz"
    );
    assert_eq!(wav.bits, 16, "{label}: the body is PCM16");

    let seconds = wav.seconds();
    let ceiling = MAX_FRAMES as f64 / FRAME_RATE_HZ;
    assert!(
        seconds >= MIN_SECONDS,
        "{label}: {seconds:.2}s of audio is too short to be this sentence (floor {MIN_SECONDS:.2}s)"
    );
    assert!(
        seconds <= ceiling,
        "{label}: {seconds:.2}s exceeds the {MAX_FRAMES}-frame cap ({ceiling:.2}s) — the \
         generation ran away"
    );
    eprintln!("[qwen-serve] {label}: {} samples, {seconds:.2}s @ {} Hz", wav.samples.len(), wav.sample_rate);
    wav
}

// --------------------------------------------------------------------------- http

fn speech_body(input: &str) -> String {
    format!(
        r#"{{"model":"qwen3-tts","input":{},"voice":"serena"}}"#,
        serde_json::to_string(input).expect("the input must serialize")
    )
}

async fn post(app: axum::Router, json: &str) -> (Response<Body>, Vec<u8>) {
    let request = Request::builder()
        .method("POST")
        .uri(SPEECH_ROUTE)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json.to_string()))
        .expect("the request must build");
    let response = app.oneshot(request).await.expect("the router must answer");
    let (parts, body) = response.into_parts();
    let bytes = to_bytes(body, usize::MAX).await.expect("the body must collect").to_vec();
    (Response::from_parts(parts, Body::empty()), bytes)
}

// --------------------------------------------------------------------------- oracle

/// Collapse runs of whitespace — stripping a cue leaves the gap it occupied, and the WER
/// reference must not carry that artifact.
fn squeeze(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Transcribe a render and score it against what was asked for.
fn score(stt: &Stt, label: &str, wav: &Wav, reference: &str) -> String {
    let transcript = stt
        .transcribe_lang(&wav.samples, wav.sample_rate, Some("en"))
        .unwrap_or_else(|e| panic!("{label}: transcribe: {e}"));
    let text = transcript.text.trim().to_string();
    let score = wer(reference, &text);
    eprintln!("[qwen-serve] {label}: WER {score:.4} (max {WER_MAX:.4})\n  want: {reference:?}\n  got:  {text:?}");
    assert!(
        score <= WER_MAX,
        "{label}: WER {score:.4} exceeds {WER_MAX:.4} — the render is not intelligible \
         speech for {reference:?} (transcript {text:?})"
    );
    text
}

// --------------------------------------------------------------------------- cases

/// A mismatch between the backend row, the checkpoint's own `tts_model_type` and the
/// configured voice is caught by `load` — the claim `synth_qwen`'s doc comment makes.
///
/// The tokenizer directory is deliberately a path that does not exist. If either check
/// ran after the weights were loaded, this test would report an io error about that path
/// (or spend a minute mapping 1.8 GB first); getting the mismatch message back is what
/// proves the ordering.
#[test]
fn load_refuses_a_mismatched_backend_checkpoint_or_voice() {
    let Some((cv_dir, _tok_dir, _stt_dir)) = common_env("real_qwen_serve") else {
        return;
    };
    let nowhere = "/nonexistent/qwen-tokenizer-dir";
    assert!(!Path::new(nowhere).exists(), "the negative-control path must not exist");

    // (a) The backend row and the voice disagree. Checked first, before any file is read.
    let err = QwenModelEngine::load(
        BackendId::Qwen06bCustomVoice,
        &cv_dir,
        nowhere,
        Device::Cpu,
        QwenVoice::Design,
    )
    .err()
    .expect("a CustomVoice backend must refuse a VoiceDesign voice");
    assert!(
        err.0.contains("configured voice") && err.0.contains("VoiceDesign"),
        "the error must name the disagreement, not a shape: {err}"
    );

    // (b) The backend row and the CHECKPOINT disagree. Reading `config.json` is enough to
    // know; the weights must not be touched.
    let err = QwenModelEngine::load(
        BackendId::Qwen17bVoiceDesign,
        &cv_dir,
        nowhere,
        Device::Cpu,
        QwenVoice::Design,
    )
    .err()
    .expect("a VoiceDesign backend must refuse a CustomVoice checkpoint");
    assert!(
        err.0.contains("CustomVoice checkpoint"),
        "the error must name the checkpoint's own variant: {err}"
    );

    // (c) A backend that is not Qwen at all never reaches the loader.
    let err = QwenModelEngine::load(
        BackendId::FishS2Pro,
        &cv_dir,
        nowhere,
        Device::Cpu,
        QwenVoice::Preset("serena".to_string()),
    )
    .err()
    .expect("a non-Qwen backend must not load a Qwen checkpoint");
    assert!(err.0.contains("not a Qwen3-TTS backend"), "{err}");

    // (d) Cloning needs a `-Base` checkpoint, and the CustomVoice one carries no
    // `speaker_encoder.*` to build an x-vector from. Refused before the mmap, too.
    let err = QwenVoice::clone_from_wav(&cv_dir, nowhere, &Device::Cpu, &[0.0f32; 16], 24_000, None)
        .err()
        .expect("cloning must refuse a CustomVoice checkpoint");
    assert!(
        err.0.contains("-Base"),
        "the error must say which variant cloning needs: {err}"
    );
}

/// The headline: a real checkpoint, a real request through the real router, real audio,
/// and the oracle's verdict on it — plain first, then cued through the same engine.
#[tokio::test]
async fn qwen_custom_voice_renders_intelligible_speech_over_the_router() {
    let Some((cv_dir, tok_dir, stt_dir)) = common_env("real_qwen_serve") else {
        return;
    };
    let _guard = heavy().lock().unwrap_or_else(|p| p.into_inner());

    let t0 = Instant::now();
    let engine = QwenModelEngine::load(
        BackendId::Qwen06bCustomVoice,
        &cv_dir,
        &tok_dir,
        Device::Cpu,
        QwenVoice::Preset("serena".to_string()),
    )
    .expect("load the 0.6B CustomVoice checkpoint")
    // `with_drive_params` replaces the WHOLE `DriveParams`, seed included, so it must come
    // before `with_seed` — the other order silently discards the seed.
    .with_drive_params(DriveParams { max_new_frames: MAX_FRAMES, ..DriveParams::default() })
    .with_seed(0);
    eprintln!("[qwen-serve] engine loaded in {:.1}s", t0.elapsed().as_secs_f64());

    assert_eq!(engine.backend(), BackendId::Qwen06bCustomVoice);
    assert_eq!(engine.mode(), syrinx_serve::qwen::QwenMode::CustomVoice);

    let stt = Stt::load(&stt_dir, Device::Cpu).expect("load the Whisper oracle");
    let synth = engine.into_synth().expect("the engine's own backend must bind");
    let app = router_with_qwen_synth(synth);

    // ---- plain -------------------------------------------------------------
    let t = Instant::now();
    let (response, body) = post(app.clone(), &speech_body(PLAIN_TEXT)).await;
    eprintln!("[qwen-serve] plain rendered in {:.1}s", t.elapsed().as_secs_f64());
    let wav = accept_render("plain", response, body);
    score(&stt, "plain", &wav, PLAIN_TEXT);

    // ---- cued --------------------------------------------------------------
    // The WER reference is the lowered text — `syrinx-cue`'s own answer to "what should
    // be spoken", the same control `tests/real_cue_activation.rs` uses. Re-deriving it
    // here would only risk disagreeing with the crate that owns the syntax.
    let lowered = squeeze(&plan(BackendId::Qwen06bCustomVoice, CUED_TEXT).expect("plan the cue").text);
    assert_ne!(lowered, CUED_TEXT, "the cue must have been lowered away");

    let t = Instant::now();
    let (response, body) = post(app, &speech_body(CUED_TEXT)).await;
    eprintln!("[qwen-serve] cued rendered in {:.1}s", t.elapsed().as_secs_f64());
    let wav = accept_render("cued", response, body);
    let spoken = score(&stt, "cued", &wav, &lowered);

    // The hard invariant, carried through to what is actually heard. The source escapes
    // nothing, so `escaped` is 0 and the bound is "no bracket at all" — the same
    // `brackets <= escaped` that `crates/syrinx-cue/tests/projection_strip.rs` holds the
    // lowered text to (`\[` legitimately speaks a literal bracket, per adr/0002).
    let escaped = CUED_TEXT.matches(r"\[").count() + CUED_TEXT.matches(r"\]").count();
    let brackets = spoken.matches('[').count() + spoken.matches(']').count();
    assert!(
        brackets <= escaped,
        "cue markup was SPOKEN: {brackets} brackets from {escaped} escapes in {CUED_TEXT:?} \
         -> {spoken:?}"
    );
    assert!(!spoken.contains("<|speaker"), "a speaker token was spoken: {spoken:?}");
    assert!(
        !spoken.to_lowercase().contains("whisper"),
        "the style label itself was spoken: {spoken:?}"
    );
}

/// The clone branch: `clone_from_wav` (x-vector + Mimi-encoded reference frames) into
/// `render_inner`'s in-context path, which decodes `cat(ref_frames, generated)` and cuts
/// the reference's share of the waveform back off. That cut is the least-exercised
/// arithmetic in the module, and a wrong one is audible as a clipped or doubled render —
/// which is why this case is scored by the oracle like the others rather than by length.
#[tokio::test]
async fn qwen_voice_clone_renders_intelligible_speech_over_the_router() {
    let Some((_cv_dir, tok_dir, stt_dir)) = common_env("real_qwen_serve clone") else {
        return;
    };
    let Some(base_dir) = env_path("SYRINX_QWEN_BASE_DIR_0_6B").or_else(|| env_path("SYRINX_QWEN_BASE_DIR"))
    else {
        eprintln!(
            "SKIP real_qwen_serve clone: set SYRINX_QWEN_BASE_DIR_0_6B (or SYRINX_QWEN_BASE_DIR) \
             to a Qwen3-TTS `-Base` checkpoint — it is the only variant carrying speaker_encoder.*"
        );
        return;
    };
    let Some(ref_wav) = env_path("SYRINX_QWEN_REF_WAV") else {
        eprintln!("SKIP real_qwen_serve clone: set SYRINX_QWEN_REF_WAV to a reference clip");
        return;
    };
    let _guard = heavy().lock().unwrap_or_else(|p| p.into_inner());

    // The reference clip at the model's own rate; `clone_from_wav` resamples anyway, but
    // handing it 24 kHz keeps the resample a no-op rather than an upsample from 16 kHz.
    let (_w16, w24) = syrinx_serve::wavio::read_ref_wav(Path::new(&ref_wav)).expect("read the reference clip");

    let stt = Stt::load(&stt_dir, Device::Cpu).expect("load the Whisper oracle");

    // In-context mode needs the reference clip's OWN transcript. `SYRINX_STT_REF` is the
    // ground truth for `SYRINX_STT_WAV`, which is a different (longer) clip, so using it
    // here would condition the prompt on a lie. The oracle that scores the render is also
    // the honest way to read the clip it is handed, so the transcript is taken from it —
    // and printed, so a bad read is visible rather than silently degrading the clone.
    let t0 = Instant::now();
    let ref_text = stt
        .transcribe_lang(&w24, 24_000, Some("en"))
        .expect("transcribe the reference clip")
        .text
        .trim()
        .to_string();
    assert!(!ref_text.is_empty(), "the reference clip transcribed to nothing");
    eprintln!(
        "[qwen-serve] reference clip ({:.2}s) transcribed in {:.1}s: {ref_text:?}",
        w24.len() as f64 / 24_000.0,
        t0.elapsed().as_secs_f64()
    );

    let t0 = Instant::now();
    let voice = QwenVoice::clone_from_wav(
        &base_dir,
        &tok_dir,
        &Device::Cpu,
        &w24,
        24_000,
        Some(ref_text.as_str()),
    )
    .expect("reduce the reference clip to a clone voice");
    eprintln!("[qwen-serve] clone voice built in {:.1}s", t0.elapsed().as_secs_f64());

    // Which of the two `-Base` rows to claim. The path is the only hint available, and it
    // is a weak one — but both Base rows carry identical caps (clone-only, no instruct
    // channel) and both map to `QwenMode::VoiceClone`, so a wrong guess changes the id in
    // the plan and nothing else. The load-time variant check still has to agree.
    let backend = if base_dir.contains("0.6B") {
        BackendId::Qwen06bBase
    } else {
        BackendId::Qwen17bBase
    };

    let t0 = Instant::now();
    let engine = QwenModelEngine::load(backend, &base_dir, &tok_dir, Device::Cpu, voice)
        .expect("load the -Base checkpoint")
        .with_drive_params(DriveParams { max_new_frames: MAX_FRAMES, ..DriveParams::default() })
        .with_seed(0);
    eprintln!("[qwen-serve] clone engine loaded in {:.1}s", t0.elapsed().as_secs_f64());
    assert_eq!(engine.mode(), syrinx_serve::qwen::QwenMode::VoiceClone);

    let app = router_with_qwen_synth(engine.into_synth().expect("the engine's backend must bind"));

    let t = Instant::now();
    let (response, body) = post(app, &speech_body(PLAIN_TEXT)).await;
    eprintln!("[qwen-serve] clone rendered in {:.1}s", t.elapsed().as_secs_f64());
    let wav = accept_render("clone", response, body);
    score(&stt, "clone", &wav, PLAIN_TEXT);
}
