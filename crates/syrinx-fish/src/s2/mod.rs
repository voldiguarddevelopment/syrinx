//! `s2-pro` (5B) backend — **filled by the s2 backend wave**.
//!
//! This wave (foundation) only fixes the contracts. The s2 wave implements here:
//!
//! * the Qwen3-4B slow AR (`fish_qwen3_omni`) implementing
//!   [`crate::common::dualar::DualArBackend`] — QK-RMSNorm, GQA (n_local_heads 8),
//!   qkv/o bias, dim 2560 × 36 layers, sharded-safetensors loading;
//! * the 4-layer ~400M fast AR — a **single shared embedding table** with the codebook
//!   identity carried by RoPE position, plus MCF fusion of the slow hidden;
//! * the 446M EVA-GAN / causal-DAC RVQ codec ([`crate::common::codec::RvqCodec`]),
//!   `codec.pth`, 44.1 kHz;
//! * the Qwen3 155k BPE tokenizer + the prompt builder.
//!
//! The next wave will add the submodules (e.g. `qwen3`, `fast`, `codec`, `tokenizer`,
//! `backend`) and re-export an `S2Backend` implementing the two traits. Nothing is built
//! here yet so the crate compiles cleanly from the foundation alone.
