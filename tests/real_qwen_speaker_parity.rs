//! Numerical parity for the Qwen3-TTS **speaker encoder** — the x-vector the `-Base`
//! (voice-clone) checkpoints condition on.
//!
//! This is the last unanchored stage of the port. `real_qwen_prompt_parity` gates the
//! prompt, `real_qwen_stack_parity` gates the talker / code predictor / codec, and all
//! of them run on `CustomVoice`, where a preset speaker id is a table lookup. The clone
//! path replaces that lookup with a 2048-wide vector computed by an ECAPA-TDNN over a
//! reference clip — 76 tensors and a hand-written mel front end that, until now, had
//! only ever been checked against the port's own reading of the reference.
//!
//! That is precisely the shape of the `text_projection` bug (a dropped `silu`) this
//! project already paid for: self-consistent Rust tests all passed, the audio sounded
//! fine, and only the reference settled it. So the encoder gets the reference's own
//! tensors, for a real WAV, on the real call path.
//!
//! ## Three anchors, so a mismatch localizes itself
//!
//! | test | input | what it gates |
//! |---|---|---|
//! | `…mel_front_end…` | the reference's resampled 24 kHz clip | STFT, reflect pre-pad, Hann window, slaney filterbank, log floor |
//! | `…encoder_from_the_reference_mel` | the reference's OWN mel | the 76-tensor ECAPA-TDNN alone — reflect-`same` convs, cascaded Res2Net, SE gates, attentive pooling |
//! | `…xvector_end_to_end` | the reference's resampled clip | both together, i.e. what `SpeakerEncoder::embed` actually does |
//!
//! If the x-vector disagrees, the first two say immediately which half is at fault.
//!
//! ## Resampling is deliberately outside the numeric gate
//!
//! The reference reaches its encoder through `librosa.resample` (soxr `HQ`); there is no
//! soxr in this workspace and reproducing it bit-for-bit is not a porting question. So
//! `gen-qwen-ref-speaker.py` captures the **resampled** clip that
//! `extract_speaker_embedding` received, and the three anchors above start from it — a
//! resampler difference can never be mistaken for an encoder fault.
//!
//! The driver path still has to resample, so it gets its own test
//! (`…driver_path_resample…`), which asserts a *measured* cosine bound rather than a
//! numeric one, because that is the honest question: does the clip our resampler
//! produces still name the same voice?
//!
//! ## Verified to fail
//!
//! Both halves were confirmed by perturbing the implementation and watching the right
//! anchors go red — including the localization claim, which is the whole reason there are
//! three of them:
//!
//! | perturbation | mel | encoder-from-ref-mel | end to end | driver |
//! |---|---|---|---|---|
//! | `tdnn` loses its `ReLU` (the `text_projection` bug's exact shape) | **ok** 0.00025 | FAIL 556.68 | FAIL 556.68 | FAIL cos 0.2678 |
//! | Hann window made symmetric instead of periodic (one-sample off-by-one) | FAIL 0.0629 | **ok** 0.0000010 | FAIL 0.00034 | ok |
//!
//! The second row is the interesting one: an off-by-one in the window is a 12x-tolerance
//! failure at the mel and only a 3.4x one at the x-vector, so the 1e-4 end-to-end bound is
//! doing real work — a looser one would have let it through.
//!
//! Fixture: `scripts/gen-qwen-ref-speaker.py` (calls the reference's own modules and
//! refuses to run if it cannot import them). Gated on `SYRINX_QWEN_BASE_DIR` and
//! `SYRINX_QWEN_REF_SPEAKER`; skips cleanly without them.

#![cfg(feature = "real")]

use candle_core::{DType, Device, Tensor};
use syrinx_qwen::speaker::{log_mel_spectrogram, MelConfig, SpeakerEncoder, SpeakerEncoderConfig};

/// Tolerances, every one of them derived from a run rather than picked.
///
/// The fixture is generated on **CPU/float32**, which is the parity path — the same rule
/// `gen-qwen-ref.py` already follows, for the same reason (this encoder is a deep
/// dilated-conv stack whose accumulation order matters, so a CUDA fixture would spend
/// the whole budget on cross-device drift).
///
/// Measured 2026-09-03 on `Qwen3-TTS-12Hz-1.7B-Base` against `voice_en_10s.wav` (10.0 s,
/// 16 kHz -> 24 kHz = 240000 samples, 937 mel frames, 2048-wide x-vector, reference L2
/// norm 17.028715):
///
/// | anchor | max abs diff | bound below | margin |
/// |---|---|---|---|
/// | mel `[937, 128]` | 0.0002508 | 5e-3 | 20x |
/// | x-vector from the reference's own mel | 0.0000010 | 1e-4 | 100x |
/// | x-vector end to end | 0.0000010 | 1e-4 | 100x |
///
/// The mel is the loose one and that is expected: [`log_mel_spectrogram`] evaluates the
/// DFT as a dense f32 matmul while the reference calls `torch.stft`, so the two differ by
/// f32 summation order over 1024 taps. It is also the harmless one — the same 2.5e-4 mel
/// difference moves the x-vector by 1.0e-6, which is why the end-to-end anchor lands on
/// exactly the number the reference-mel anchor does.
///
/// The CUDA bounds are **not measured** (this port was anchored CPU-only, and the encoder
/// runs f32 on both devices, so CUDA differs only by accumulation order). They are slack,
/// not evidence; the CPU/f32 run is the gate.
const TOL_MEL_F32: f32 = 5e-3;
const TOL_MEL_CUDA: f32 = 6e-2;
/// x-vector components on this checkpoint run to roughly +/- 1.0 (L2 norm 17.0 over 2048).
const TOL_XVEC_F32: f32 = 1e-4;
const TOL_XVEC_CUDA: f32 = 3e-1;

/// The driver bound: `resample(16 kHz clip) -> embed` against the reference's
/// `librosa.resample` (soxr `HQ`) -> encoder. NOT a numeric parity bound — two different
/// band-limited resamplers, so the components differ far more than the anchors above.
/// What must hold is that the vector still names the same voice.
///
/// Measured with the shipped `speaker::resample` (64 lobes, 0.96 passband): **cosine
/// 0.999977**, relative L2 0.0070, max abs component difference 0.0303. The bound is set
/// at 0.9995, ~20x the measured 2.3e-5 gap.
///
/// That bound is not decorative: the first version of this resampler (16 lobes, cutoff
/// exactly Nyquist, copied from `syrinx_serve::wavio`) scored **0.996886** here and this
/// assertion is what caught it. Chasing it down found real image leakage above the input
/// Nyquist landing in the top mel bands — see [`syrinx_qwen::speaker::resample`] for the
/// spectrum and the sweep that picked 64/0.96.
const MIN_DRIVER_COSINE: f32 = 0.9995;

fn env_path(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|p| std::path::Path::new(p).exists())
}

fn device() -> Device {
    match std::env::var("SYRINX_QWEN_DEVICE").ok().and_then(|v| v.trim().parse::<usize>().ok()) {
        Some(i) => Device::new_cuda(i).unwrap_or_else(|e| {
            eprintln!("[qwen-speaker] cuda:{i} unavailable ({e}); using CPU");
            Device::Cpu
        }),
        None => Device::Cpu,
    }
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> candle_core::Result<f32> {
    let d = (a.to_dtype(DType::F32)?.flatten_all()? - b.to_dtype(DType::F32)?.flatten_all()?)?
        .abs()?;
    d.max(0)?.to_scalar::<f32>()
}

fn to_vec(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap()
}

/// `(fixture, encoder, device)`, or `None` with a SKIP printed.
///
/// Only the 76 `speaker_encoder.*` tensors are pulled out of the checkpoint (it is
/// 3.6 GB, and the talker is not in this path at all), through the same
/// `SpeakerEncoder::load` shape check the driver uses.
fn setup(name: &str) -> Option<(std::collections::HashMap<String, Tensor>, SpeakerEncoder, Device)> {
    let Some(dir) = env_path("SYRINX_QWEN_BASE_DIR") else {
        eprintln!("SKIP {name}: set SYRINX_QWEN_BASE_DIR to a Qwen3-TTS-12Hz-*-Base checkpoint dir");
        return None;
    };
    let Some(ref_path) = env_path("SYRINX_QWEN_REF_SPEAKER") else {
        eprintln!(
            "SKIP {name}: set SYRINX_QWEN_REF_SPEAKER to the fixture from \
             scripts/gen-qwen-ref-speaker.py"
        );
        return None;
    };
    let refs = candle_core::safetensors::load(&ref_path, &Device::Cpu).expect("load fixture");
    let dev = device();

    let base = std::path::Path::new(&dir);
    let json = std::fs::read_to_string(base.join("config.json")).expect("config.json");
    let cfg = SpeakerEncoderConfig::from_model_config(
        &syrinx_qwen::Qwen3TtsConfig::from_json(&json).expect("parse config"),
    );
    // mmap and take only `speaker_encoder.*`; loading all 3.6 GB to reach 76 tensors
    // would make this test the most expensive one in the suite for no reason.
    let st = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(base.join("model.safetensors"))
            .expect("mmap model.safetensors")
    };
    let mut map = std::collections::HashMap::new();
    for (k, _) in st.tensors() {
        if k.starts_with("speaker_encoder.") {
            map.insert(k.clone(), st.load(&k, &dev).expect("load tensor"));
        }
    }
    assert_eq!(
        map.len(),
        76,
        "{dir} carries {} speaker_encoder.* tensors, not the 76 the -Base checkpoints ship \
         (is this really a -Base checkpoint?)",
        map.len()
    );
    let w = syrinx_qwen::nn::Weights { map, dev: dev.clone(), dt: DType::F32 };
    let enc = SpeakerEncoder::load(&w, "speaker_encoder", cfg).expect("load speaker encoder");
    Some((refs, enc, dev))
}

#[test]
fn qwen_speaker_mel_front_end_matches_the_reference() {
    let Some((refs, _enc, dev)) = setup("real_qwen_speaker_parity::mel") else { return };
    let want = refs.get("mel").expect("fixture: mel").to_device(&dev).expect("to device");
    let wav: Vec<f32> = to_vec(refs.get("wav24").expect("fixture: wav24"));
    let tol = if dev.is_cuda() { TOL_MEL_CUDA } else { TOL_MEL_F32 };

    let got = log_mel_spectrogram(&wav, &MelConfig::QWEN3_TTS, &dev).expect("mel");
    let got = got.squeeze(0).expect("drop batch");

    assert_eq!(
        got.dims(),
        want.dims(),
        "mel geometry: got {:?}, the reference produced {:?} for the same {} samples",
        got.dims(),
        want.dims(),
        wav.len()
    );
    let diff = max_abs_diff(&got, &want).expect("diff");
    eprintln!(
        "[qwen-speaker] mel {:?} from {} samples: max abs diff {diff:.7} (tol {tol})",
        want.dims(),
        wav.len()
    );
    assert!(
        diff <= tol,
        "mel differs from the reference by {diff} (tolerance {tol}). The encoder anchor \
         passing while this fails places the fault in the STFT, the reflect pre-pad, the \
         Hann window, the slaney filterbank or the log floor — not in the 76 weights."
    );
}

#[test]
fn qwen_speaker_encoder_from_the_reference_mel_matches() {
    let Some((refs, enc, dev)) = setup("real_qwen_speaker_parity::encoder") else { return };
    let want = refs.get("xvector").expect("fixture: xvector");
    // The reference's OWN mel, not one derived here: this isolates the 76-tensor stack
    // from the front end, so the two halves of `embed` fail independently.
    let mel = refs
        .get("mel")
        .expect("fixture: mel")
        .to_device(&dev)
        .expect("to device")
        .unsqueeze(0)
        .expect("batch");
    let tol = if dev.is_cuda() { TOL_XVEC_CUDA } else { TOL_XVEC_F32 };

    let got = enc.forward(&mel).expect("encoder forward");
    assert_eq!(
        got.elem_count(),
        want.elem_count(),
        "x-vector width: got {}, reference {}",
        got.elem_count(),
        want.elem_count()
    );
    let diff = max_abs_diff(&got.flatten_all().unwrap(), want).expect("diff");
    eprintln!(
        "[qwen-speaker] encoder from the reference mel: {} components, max abs diff \
         {diff:.7} (tol {tol})",
        want.elem_count()
    );
    assert!(
        diff <= tol,
        "the x-vector differs from the reference by {diff} (tolerance {tol}) even though \
         the mel came from the reference itself. The fault is inside the ECAPA-TDNN: \
         reflect-`same` padding, the cascaded Res2Net sum, an SE gate, the attentive \
         pooling statistics, or the MFA skipping blocks.0."
    );
}

#[test]
fn qwen_speaker_xvector_end_to_end_matches_the_reference() {
    let Some((refs, enc, dev)) = setup("real_qwen_speaker_parity::xvector") else { return };
    let want = refs.get("xvector").expect("fixture: xvector");
    let wav: Vec<f32> = to_vec(refs.get("wav24").expect("fixture: wav24"));
    let tol = if dev.is_cuda() { TOL_XVEC_CUDA } else { TOL_XVEC_F32 };

    let got = enc.embed(&wav, 24_000).expect("embed");
    assert_eq!(
        got.dims(),
        &[1, want.elem_count()],
        "embed must return [1, enc_dim]; got {:?} against a {}-wide reference",
        got.dims(),
        want.elem_count()
    );
    let diff = max_abs_diff(&got.flatten_all().unwrap(), want).expect("diff");

    // The norm is reported too: a systematic scale error (a missing normalisation, a
    // filterbank normalised the wrong way) shows up here more legibly than in a max-abs.
    let g = to_vec(&got);
    let w = to_vec(want);
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    eprintln!(
        "[qwen-speaker] x-vector end to end: {} components, max abs diff {diff:.7} \
         (tol {tol}); L2 norm {:.6} vs reference {:.6}",
        want.elem_count(),
        norm(&g),
        norm(&w)
    );
    assert!(
        diff <= tol,
        "the end-to-end x-vector differs from the reference by {diff} (tolerance {tol}). \
         This is what the -Base clone path actually conditions on."
    );
}

#[test]
fn qwen_speaker_driver_path_resample_preserves_the_voice() {
    let Some((refs, enc, _dev)) = setup("real_qwen_speaker_parity::driver") else { return };
    let want = to_vec(refs.get("xvector").expect("fixture: xvector"));
    // The clip at its ORIGINAL rate, as the reference's `librosa.load(sr=None)` read it —
    // i.e. what the driver is handed. `wav24` is what the reference's soxr made of it.
    let src = to_vec(refs.get("wav_in").expect("fixture: wav_in"));
    let ref24 = to_vec(refs.get("wav24").expect("fixture: wav24"));
    // The source rate is whatever produced `wav24` at 24 kHz over the same duration; the
    // fixture's two lengths pin it exactly, so the test needs no second env var.
    let sr_in = ((src.len() as f64 / ref24.len() as f64) * 24_000.0).round() as u32;
    assert!(
        sr_in > 0 && sr_in != 24_000,
        "fixture clip is already 24 kHz ({} vs {} samples) — this test needs a resampling \
         case to mean anything",
        src.len(),
        ref24.len()
    );

    let wav = syrinx_qwen::speaker::resample(&src, sr_in, 24_000);
    assert_eq!(
        wav.len(),
        ref24.len(),
        "resample({sr_in} -> 24000) produced {} samples, the reference's {}",
        wav.len(),
        ref24.len()
    );
    let got = to_vec(&enc.embed(&wav, 24_000).expect("embed"));

    let dot: f32 = got.iter().zip(&want).map(|(a, b)| a * b).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos = dot / (norm(&got) * norm(&want));
    let rel = norm(&got.iter().zip(&want).map(|(a, b)| a - b).collect::<Vec<_>>()) / norm(&want);
    let max_abs = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!(
        "[qwen-speaker] driver path ({sr_in} Hz -> 24 kHz, Lanczos-64/0.96 vs soxr HQ): \
         cosine {cos:.6} (min {MIN_DRIVER_COSINE}), relative L2 {rel:.4}, max abs {max_abs:.4}"
    );
    assert!(
        cos >= MIN_DRIVER_COSINE,
        "the driver's resampled x-vector points {cos} with the reference's (minimum \
         {MIN_DRIVER_COSINE}). Two different band-limited resamplers never agree \
         numerically, but they must agree on the speaker; this far off means aliasing, a \
         missing anti-alias cutoff, or a wrong rate ratio."
    );
}
