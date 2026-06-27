//! `openaudio-s1-mini` (0.5B) backend — **filled by the s1 backend wave**.
//!
//! This wave (foundation) only fixes the contracts. The s1 wave implements here:
//!
//! * the Llama-style `DualARTransformer` slow AR ([`crate::common::dualar::DualArBackend`])
//!   — per-codebook embedding tables, no QK-norm, no qkv bias, tied output head;
//! * the 4-layer fast AR (per-codebook embeddings + RoPE over the codebook axis);
//! * the modded-DAC RVQ codec ([`crate::common::codec::RvqCodec`]) — Snake1d / WN causal
//!   convs, residual VQ, 44.1 kHz;
//! * the tiktoken tokenizer + `.pth` weight loading + the content-sequence prompt builder.
//!
//! The next wave will add the submodules (e.g. `transformer`, `fast`, `codec`, `tokenizer`,
//! `backend`) and re-export an `S1Backend` implementing the two traits. Nothing is built
//! here yet so the crate compiles cleanly from the foundation alone.
