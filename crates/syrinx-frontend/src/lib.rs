//! syrinx-frontend — the real CosyVoice2 text frontend (real-weights track):
//! wetext-style text normalization (`textnorm`), the Qwen2 BPE text tokenizer
//! (`tokenizer`), kaldi-fbank / prompt-mel feature extraction (`feat`), and the
//! prompt speech-token tokenizer (`speech_token`).

// Faithful (common-case) wetext-style zh+en text normalizer — the pre-tokenization
// normalization the real CosyVoice2 frontend runs (`frontend.text_normalize`).
// Pure Rust; gated behind `tn` (implied by `real`) so the default build is unchanged.
#[cfg(feature = "tn")]
pub mod textnorm;

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
