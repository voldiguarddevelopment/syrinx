//! Crate root for `syrinx-core`.
//!
//! `lib.rs` holds the tensor ops (T-02.01a/b) and is kept byte-identical to its
//! freeze base so the mutation gate never re-targets it; this root only wires
//! the modules together. `weights.rs` (T-02.01c) is isolated in its own file so
//! the gate scopes its mutants to the frozen `weights_parity.rs` tests.
#[path = "lib.rs"]
mod ops;
mod weights;

pub use ops::*;
pub use weights::{fnv1a_64, weights, xorshift64_next};
