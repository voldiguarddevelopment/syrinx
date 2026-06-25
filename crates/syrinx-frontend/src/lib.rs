//! syrinx-frontend — deterministic text frontend (T-00.01 scaffold; T-01.01
//! normalize; T-01.02 numeric expansion; T-01.04 G2P phonemization; T-01.05
//! custom pronunciation overrides; T-01.06 heteronym resolution; T-01.07 SSML
//! subset parsing).

pub mod context;
pub mod contract;
pub mod expand;
pub mod g2p;
pub mod hetero;
pub mod lexicon;
pub mod normalize;
pub mod pacing;
pub mod punct;
pub mod ssml;

// Real CosyVoice2 text tokenizer (Qwen2 BPE via the HF `tokenizers` crate).
// Compiled only under the crate's `real` feature, on the model box, where the
// serialized `tokenizer.json` and the parity fixtures are available.
#[cfg(feature = "real")]
pub mod tokenizer;

// Audio feature extraction (kaldi fbank for the CAM++ speaker encoder + the flow
// decoder prompt mel). Compiled only under the crate's `real` feature, on the
// model box, where the parity fixtures (reference waveforms + features) live.
#[cfg(feature = "real")]
pub mod feat;

/// Prompt **speech-token** tokenizer (real-weights track): 16 kHz reference wav
/// -> `prompt_speech_token` ids via a Rust whisper log-mel + `speech_tokenizer_v2.onnx`
/// run through ONNX Runtime. Compiled only under the crate's `real` feature.
#[cfg(feature = "real")]
pub mod speech_token;

/// Dispatch a single named frontend stage over raw `input`, returning the stage's
/// textual output (T-01.11). This is the one entry point the aggregating
/// golden-file suite drives, one stage per fixture sub-tree:
///
///   * `"normalize"` -> [`normalize::normalize`]
///   * `"numbers"`   -> [`expand::expand_numbers`]
///   * `"ssml"`      -> the `Debug` rendering of [`ssml::parse`]
///
/// An unknown stage name is a programming error in the fixture tree and panics.
pub fn render_stage(stage: &str, input: &str) -> String {
    match stage {
        "normalize" => normalize::normalize(input),
        "numbers" => expand::expand_numbers(input),
        "ssml" => format!("{:?}", ssml::parse(input)),
        other => panic!("unknown frontend stage `{other}`"),
    }
}
