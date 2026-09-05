//! Numerical parity for the Qwen3-TTS **speaker encoder** — the x-vector the `-Base`
//! (voice-clone) checkpoints condition on.
//!
//! This is the last unanchored stage of the port. `real_qwen_prompt_parity` gates the
//! prompt, `real_qwen_stack_parity` gates the talker / code predictor / codec, and all
//! of them run on `CustomVoice`, where a preset speaker id is a table lookup. The clone
//! path replaces that lookup with a wide vector computed by an ECAPA-TDNN over a
//! reference clip — 76 tensors and a hand-written mel front end that, until now, had
//! only ever been checked against the port's own reading of the reference.
//!
//! That is precisely the shape of the `text_projection` bug (a dropped `silu`) this
//! project already paid for: self-consistent Rust tests all passed, the audio sounded
//! fine, and only the reference settled it. So the encoder gets the reference's own
//! tensors, for a real WAV, on the real call path.
//!
//! ## Both `-Base` checkpoints, not just one
//!
//! The two `-Base` sizes differ in exactly one thing: `speaker_encoder_config.enc_dim`,
//! 1024 on the 0.6B and 2048 on the 1.7B. Every other tensor in the 76 is the same shape.
//! That single difference is enough to make a checkpoint-specific fault possible —
//! `SpeakerEncoderConfig::from_model_config` reads `enc_dim` from `config.json` rather
//! than hardcoding it, and a port that hardcoded either width would load one checkpoint
//! and fail the other — so this test runs **every configured checkpoint**, not the first
//! one it finds. Until 2026-09-05 only the 1.7B had a fixture and the 0.6B was anchored
//! against nothing but the port's own reading of the reference.
//!
//! A checkpoint is configured by pointing one env var at its fixture:
//!
//! | checkpoint | fixture var | checkpoint-dir override |
//! |---|---|---|
//! | first  | `SYRINX_QWEN_REF_SPEAKER`        | `SYRINX_QWEN_BASE_DIR` |
//! | second | `SYRINX_QWEN_REF_SPEAKER_0_6B`   | `SYRINX_QWEN_BASE_DIR_0_6B` |
//!
//! The dir is an **override**, not a requirement: `gen-qwen-ref-speaker.py` records the
//! absolute `ckpt` it ran against in the fixture's safetensors metadata, so a fixture
//! already knows which checkpoint produced it and the test reads it from there when the
//! override is unset. That is deliberately stricter than a second env var would be — a
//! fixture can never be silently paired with the wrong checkpoint — and it means adding a
//! checkpoint to the board costs exactly one line of `test-all.env`. If neither the
//! override nor the recorded path exists on disk, the test fails loudly rather than
//! quietly dropping that checkpoint; a half-configured anchor is a hole, not a skip.
//!
//! With no fixture var set at all the whole file SKIPs, as every weight-backed test does.
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
//! The mel anchor is checkpoint-INDEPENDENT by construction (`MelConfig::QWEN3_TTS` is a
//! constant and both checkpoints declare `sample_rate: 24000`), and the two shipped
//! fixtures were dumped from the same clip, so their `wav24` and `mel` tensors are
//! bit-identical — verified, not assumed. Running it per checkpoint is therefore a
//! cross-check that the fixtures really do share a clip, not new coverage of the mel.
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
//! refuses to run if it cannot import them), once per checkpoint:
//!
//! ```text
//! CUDA_VISIBLE_DEVICES= MEMMAX=10G scripts/run-isolated.sh \
//!   /home/floofy/.venvs/qwen/bin/python scripts/gen-qwen-ref-speaker.py \
//!   --ckpt /data/models/Qwen3-TTS-12Hz-0.6B-Base \
//!   --wav  /home/floofy/refs/voice_en_10s.wav \
//!   --out  /home/floofy/parity-qwen/speaker-0.6b.safetensors
//! ```

#![cfg(feature = "real")]

use candle_core::{DType, Device, Tensor};
use syrinx_qwen::speaker::{log_mel_spectrogram, MelConfig, SpeakerEncoder, SpeakerEncoderConfig};

/// Tolerances, every one of them derived from a run rather than picked.
///
/// The fixtures are generated on **CPU/float32**, which is the parity path — the same rule
/// `gen-qwen-ref.py` already follows, for the same reason (this encoder is a deep
/// dilated-conv stack whose accumulation order matters, so a CUDA fixture would spend
/// the whole budget on cross-device drift).
///
/// Measured against `voice_en_10s.wav` (10.0 s, 16 kHz -> 24 kHz = 240000 samples,
/// 937 mel frames):
///
/// | anchor | 1.7B-Base (2048, L2 17.028715) | 0.6B-Base (1024, L2 10.409612) | bound | margin |
/// |---|---|---|---|---|
/// | mel `[937, 128]` | 0.0002508 | 0.0002508 | 5e-3 | 20x |
/// | x-vector from the reference's own mel | 0.0000010 | 0.0000006 | 1e-4 | 100x / 167x |
/// | x-vector end to end | 0.0000010 | 0.0000006 | 1e-4 | 100x / 167x |
///
/// (1.7B measured 2026-09-03, 0.6B added 2026-09-05. The two mel figures are the same
/// number because the mel front end does not depend on the checkpoint and both fixtures
/// carry the identical clip.)
///
/// **The x-vector agreement is at the reference's own noise floor**, which is the only way
/// to read a figure like 6e-7 honestly. Re-dumping the 0.6B fixture from the same
/// reference, same device, same dtype, differing only in `OMP_NUM_THREADS` (32 vs 1),
/// moves its x-vector by **4.8e-7** — `wav24` and `mel` come back bit-identical, so all of
/// it is the encoder's own reduction order. The port's 6e-7 is that same magnitude: there
/// is no residue left for a porting fault to hide in.
///
/// The mel is the loose one and that is expected: [`log_mel_spectrogram`] evaluates the
/// DFT as a dense f32 matmul while the reference calls `torch.stft`, so the two differ by
/// f32 summation order over 1024 taps. It is also the harmless one — the same 2.5e-4 mel
/// difference moves the x-vector by 1.0e-6, which is why the end-to-end anchor lands on
/// exactly the number the reference-mel anchor does, on both checkpoints.
///
/// The CUDA bounds are **not measured** (this port was anchored CPU-only, and the encoder
/// runs f32 on both devices, so CUDA differs only by accumulation order). They are slack,
/// not evidence; the CPU/f32 run is the gate.
const TOL_MEL_F32: f32 = 5e-3;
const TOL_MEL_CUDA: f32 = 6e-2;
/// x-vector components run to roughly +/- 1.0 (L2 norm 17.0 over 2048 on the 1.7B, 10.4
/// over 1024 on the 0.6B).
const TOL_XVEC_F32: f32 = 1e-4;
const TOL_XVEC_CUDA: f32 = 3e-1;

/// The driver bound: `resample(16 kHz clip) -> embed` against the reference's
/// `librosa.resample` (soxr `HQ`) -> encoder. NOT a numeric parity bound — two different
/// band-limited resamplers, so the components differ far more than the anchors above.
/// What must hold is that the vector still names the same voice.
///
/// Measured with the shipped `speaker::resample` (64 lobes, 0.96 passband): **cosine
/// 0.999977** on the 1.7B (relative L2 0.0070, max abs component difference 0.0303) and
/// **0.999980** on the 0.6B (relative L2 0.0064, max abs 0.0073). The bound is set at
/// 0.9995, ~20x the measured 2.3e-5 gap.
///
/// That bound is not decorative: the first version of this resampler (16 lobes, cutoff
/// exactly Nyquist, copied from `syrinx_serve::wavio`) scored **0.996886** here and this
/// assertion is what caught it. Chasing it down found real image leakage above the input
/// Nyquist landing in the top mel bands — see [`syrinx_qwen::speaker::resample`] for the
/// spectrum and the sweep that picked 64/0.96.
const MIN_DRIVER_COSINE: f32 = 0.9995;

/// The checkpoints this gate covers, in board order: `(fixture var, checkpoint-dir
/// override var)`. Adding a third `-Base` checkpoint means adding a row here and one
/// `export` in `scripts/test-all.env` — nothing else.
const CHECKPOINTS: [(&str, &str); 2] = [
    ("SYRINX_QWEN_REF_SPEAKER", "SYRINX_QWEN_BASE_DIR"),
    ("SYRINX_QWEN_REF_SPEAKER_0_6B", "SYRINX_QWEN_BASE_DIR_0_6B"),
];

fn env_path(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|p| std::path::Path::new(p).exists())
}

/// CPU/f32, always — `SYRINX_QWEN_DEVICE` is deliberately ignored here.
///
/// Every number this file asserts was measured on CPU/f32, and the fixtures are dumped
/// there too. The CUDA bounds that used to apply on this path were slack placeholders
/// (3e-1 on an x-vector that agrees to 1e-6 on CPU), so honouring `SYRINX_QWEN_DEVICE` —
/// which `scripts/test-all.env` sets to 0 — quietly turned the board row into a smoke
/// test: it would have passed with a defect two orders of magnitude larger than anything
/// measured. `real_qwen_encode_parity` pins CPU for the same reason. A CUDA gate is a
/// legitimate thing to want, but it needs its own measured budget, not this one's.
fn device() -> Device {
    Device::Cpu
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> candle_core::Result<f32> {
    let d = (a.to_dtype(DType::F32)?.flatten_all()? - b.to_dtype(DType::F32)?.flatten_all()?)?
        .abs()?;
    d.max(0)?.to_scalar::<f32>()
}

fn to_vec(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap()
}

/// The `ckpt` path `gen-qwen-ref-speaker.py` recorded in the fixture's safetensors
/// `__metadata__`, read straight out of the 8-byte length prefix + JSON header. Candle's
/// loaders drop the metadata, and pulling in the `safetensors` crate just for one string
/// would be a heavier dependency than the twelve lines it saves.
fn fixture_ckpt(path: &str) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    let (len, rest) = bytes.split_at_checked(8)?;
    let n = u64::from_le_bytes(len.try_into().ok()?) as usize;
    let header: serde_json::Value = serde_json::from_slice(rest.get(..n)?).ok()?;
    header.get("__metadata__")?.get("ckpt")?.as_str().map(str::to_owned)
}

/// One configured checkpoint: its fixture tensors and its loaded 76-tensor encoder.
struct Case {
    label: String,
    refs: std::collections::HashMap<String, Tensor>,
    enc: SpeakerEncoder,
    dev: Device,
}

/// Every configured checkpoint, or an empty vec with a SKIP printed.
///
/// Only the 76 `speaker_encoder.*` tensors are pulled out of each checkpoint (they are
/// 1.8-3.6 GB, and the talker is not in this path at all), through the same
/// `SpeakerEncoder::load` shape check the driver uses.
fn cases(name: &str) -> Vec<Case> {
    let dev = device();
    let mut out = Vec::new();
    for (ref_var, dir_var) in CHECKPOINTS {
        let Some(ref_path) = env_path(ref_var) else {
            // NOT the word SKIP: the runners grep for it and one unconfigured checkpoint
            // must not paint the whole row yellow while another checkpoint really ran.
            eprintln!("[qwen-speaker] {ref_var} unset or missing — that checkpoint is not configured");
            continue;
        };
        // The dir override wins; otherwise the fixture names its own checkpoint. Anything
        // else is a hard failure, never a silent drop: a fixture with no reachable
        // checkpoint is a hole in the gate.
        let recorded = fixture_ckpt(&ref_path);
        let (dir, from) = match env_path(dir_var) {
            Some(d) => (d, format!("${dir_var}")),
            None => match recorded.as_deref().filter(|p| std::path::Path::new(p).exists()) {
                Some(d) => (d.to_string(), format!("{ref_var}'s recorded ckpt")),
                None => panic!(
                    "{ref_var}={ref_path} is configured but its checkpoint is unreachable: \
                     ${dir_var} is unset (or missing) and the fixture's recorded ckpt is \
                     {recorded:?}. Set ${dir_var} to the -Base checkpoint dir that produced \
                     this fixture, or unset {ref_var}."
                ),
            },
        };

        let refs = candle_core::safetensors::load(&ref_path, &Device::Cpu).expect("load fixture");
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
        // `enc_dim` comes from that checkpoint's own config.json, which is the whole
        // reason both sizes are on this gate; pin it against the fixture the reference
        // actually produced, so a fixture/checkpoint mispairing says so in one line
        // instead of surfacing as a mysterious width error three anchors later.
        let want_dim = refs.get("xvector").expect("fixture: xvector").elem_count();
        assert_eq!(
            cfg.enc_dim, want_dim,
            "{dir} declares speaker_encoder_config.enc_dim = {}, but {ref_path} holds a \
             {want_dim}-wide x-vector — the fixture and the checkpoint do not belong \
             together (dir resolved from {from})",
            cfg.enc_dim
        );
        let w = syrinx_qwen::nn::Weights { map, dev: dev.clone(), dt: DType::F32 };
        let enc = SpeakerEncoder::load(&w, "speaker_encoder", cfg).expect("load speaker encoder");

        let label = format!(
            "{} ({want_dim}-wide)",
            base.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or(dir.clone())
        );
        eprintln!("[qwen-speaker] case {label}: fixture ${ref_var}, checkpoint from {from}");
        out.push(Case { label, refs, enc, dev: dev.clone() });
    }
    if out.is_empty() {
        eprintln!(
            "SKIP {name}: set SYRINX_QWEN_REF_SPEAKER (and/or SYRINX_QWEN_REF_SPEAKER_0_6B) to \
             a fixture from scripts/gen-qwen-ref-speaker.py"
        );
    }
    out
}

#[test]
fn qwen_speaker_mel_front_end_matches_the_reference() {
    for c in cases("real_qwen_speaker_parity::mel") {
        let want = c.refs.get("mel").expect("fixture: mel").to_device(&c.dev).expect("to device");
        let wav: Vec<f32> = to_vec(c.refs.get("wav24").expect("fixture: wav24"));
        let tol = if c.dev.is_cuda() { TOL_MEL_CUDA } else { TOL_MEL_F32 };

        let got = log_mel_spectrogram(&wav, &MelConfig::QWEN3_TTS, &c.dev).expect("mel");
        let got = got.squeeze(0).expect("drop batch");

        assert_eq!(
            got.dims(),
            want.dims(),
            "[{}] mel geometry: got {:?}, the reference produced {:?} for the same {} samples",
            c.label,
            got.dims(),
            want.dims(),
            wav.len()
        );
        let diff = max_abs_diff(&got, &want).expect("diff");
        eprintln!(
            "[qwen-speaker] {} mel {:?} from {} samples: max abs diff {diff:.7} (tol {tol})",
            c.label,
            want.dims(),
            wav.len()
        );
        assert!(
            diff <= tol,
            "[{}] mel differs from the reference by {diff} (tolerance {tol}). The encoder \
             anchor passing while this fails places the fault in the STFT, the reflect \
             pre-pad, the Hann window, the slaney filterbank or the log floor — not in the \
             76 weights.",
            c.label
        );
    }
}

#[test]
fn qwen_speaker_encoder_from_the_reference_mel_matches() {
    for c in cases("real_qwen_speaker_parity::encoder") {
        let want = c.refs.get("xvector").expect("fixture: xvector");
        // The reference's OWN mel, not one derived here: this isolates the 76-tensor stack
        // from the front end, so the two halves of `embed` fail independently.
        let mel = c
            .refs
            .get("mel")
            .expect("fixture: mel")
            .to_device(&c.dev)
            .expect("to device")
            .unsqueeze(0)
            .expect("batch");
        let tol = if c.dev.is_cuda() { TOL_XVEC_CUDA } else { TOL_XVEC_F32 };

        let got = c.enc.forward(&mel).expect("encoder forward");
        assert_eq!(
            got.elem_count(),
            want.elem_count(),
            "[{}] x-vector width: got {}, reference {}",
            c.label,
            got.elem_count(),
            want.elem_count()
        );
        let diff = max_abs_diff(&got.flatten_all().unwrap(), want).expect("diff");
        eprintln!(
            "[qwen-speaker] {} encoder from the reference mel: {} components, max abs diff \
             {diff:.7} (tol {tol})",
            c.label,
            want.elem_count()
        );
        assert!(
            diff <= tol,
            "[{}] the x-vector differs from the reference by {diff} (tolerance {tol}) even \
             though the mel came from the reference itself. The fault is inside the \
             ECAPA-TDNN: reflect-`same` padding, the cascaded Res2Net sum, an SE gate, the \
             attentive pooling statistics, or the MFA skipping blocks.0.",
            c.label
        );
    }
}

#[test]
fn qwen_speaker_xvector_end_to_end_matches_the_reference() {
    for c in cases("real_qwen_speaker_parity::xvector") {
        let want = c.refs.get("xvector").expect("fixture: xvector");
        let wav: Vec<f32> = to_vec(c.refs.get("wav24").expect("fixture: wav24"));
        let tol = if c.dev.is_cuda() { TOL_XVEC_CUDA } else { TOL_XVEC_F32 };

        let got = c.enc.embed(&wav, 24_000).expect("embed");
        assert_eq!(
            got.dims(),
            &[1, want.elem_count()],
            "[{}] embed must return [1, enc_dim]; got {:?} against a {}-wide reference",
            c.label,
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
            "[qwen-speaker] {} x-vector end to end: {} components, max abs diff {diff:.7} \
             (tol {tol}); L2 norm {:.6} vs reference {:.6}",
            c.label,
            want.elem_count(),
            norm(&g),
            norm(&w)
        );
        assert!(
            diff <= tol,
            "[{}] the end-to-end x-vector differs from the reference by {diff} (tolerance \
             {tol}). This is what the -Base clone path actually conditions on.",
            c.label
        );
    }
}

#[test]
fn qwen_speaker_driver_path_resample_preserves_the_voice() {
    for c in cases("real_qwen_speaker_parity::driver") {
        let want = to_vec(c.refs.get("xvector").expect("fixture: xvector"));
        // The clip at its ORIGINAL rate, as the reference's `librosa.load(sr=None)` read it —
        // i.e. what the driver is handed. `wav24` is what the reference's soxr made of it.
        let src = to_vec(c.refs.get("wav_in").expect("fixture: wav_in"));
        let ref24 = to_vec(c.refs.get("wav24").expect("fixture: wav24"));
        // The source rate is whatever produced `wav24` at 24 kHz over the same duration; the
        // fixture's two lengths pin it exactly, so the test needs no second env var.
        let sr_in = ((src.len() as f64 / ref24.len() as f64) * 24_000.0).round() as u32;
        assert!(
            sr_in > 0 && sr_in != 24_000,
            "[{}] fixture clip is already 24 kHz ({} vs {} samples) — this test needs a \
             resampling case to mean anything",
            c.label,
            src.len(),
            ref24.len()
        );

        let wav = syrinx_qwen::speaker::resample(&src, sr_in, 24_000);
        assert_eq!(
            wav.len(),
            ref24.len(),
            "[{}] resample({sr_in} -> 24000) produced {} samples, the reference's {}",
            c.label,
            wav.len(),
            ref24.len()
        );
        let got = to_vec(&c.enc.embed(&wav, 24_000).expect("embed"));

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
            "[qwen-speaker] {} driver path ({sr_in} Hz -> 24 kHz, Lanczos-64/0.96 vs soxr HQ): \
             cosine {cos:.6} (min {MIN_DRIVER_COSINE}), relative L2 {rel:.4}, max abs {max_abs:.4}",
            c.label
        );
        assert!(
            cos >= MIN_DRIVER_COSINE,
            "[{}] the driver's resampled x-vector points {cos} with the reference's (minimum \
             {MIN_DRIVER_COSINE}). Two different band-limited resamplers never agree \
             numerically, but they must agree on the speaker; this far off means aliasing, a \
             missing anti-alias cutoff, or a wrong rate ratio.",
            c.label
        );
    }
}
