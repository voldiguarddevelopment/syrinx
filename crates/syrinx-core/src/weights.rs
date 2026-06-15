//! T-02.01c — deterministic weight generation by name (`reference.py` §2).
//!
//! Weights are never read from a file: every named tensor's values are a pure
//! function of its `name`. The name is hashed to a seed (unsalted FNV-1a-64),
//! the seed drives an xorshift64 stream, and each emitted `u64` is mapped to an
//! `f32` in `[-0.02, 0.02)`. This is a byte-exact port of the reference PRNG —
//! the documented `tok_embeddings` seed and first draw reproduce exactly.

/// Unsalted FNV-1a-64 over the name's UTF-8 bytes (`reference.py` §2.1): offset
/// basis `0xCBF29CE484222325`, prime `0x00000100000001B3`, XOR-then-multiply mod
/// 2^64. The empty name hashes to the bare offset basis.
pub fn fnv1a_64(name: &str) -> u64 {
    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    for byte in name.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

/// Advance an xorshift64 stream one step and emit the post-update state
/// (`reference.py` §2.2): `x ^= x<<13; x ^= x>>7; x ^= x<<17` (all wrapping mod
/// 2^64). A `0` seed is substituted by `0x9E3779B97F4A7C15` before advancing.
pub fn xorshift64_next(seed: u64) -> u64 {
    let mut x = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

/// Map an emitted `u64` to an `f32` weight (`reference.py` §2.3):
/// `((x >> 11) as f64 * (1/2^53)) * 2 - 1` (a unit value scaled to `[-1, 1)`)
/// then `* 0.02`, cast to `f32`.
fn draw_to_f32(state: u64) -> f32 {
    let unit = (state >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0);
    ((unit * 2.0 - 1.0) * 0.02) as f32
}

/// The first `count` draws of the xorshift64 stream seeded by `fnv1a_64(name)`,
/// each mapped to an `f32` (`reference.py` §2). The result has length `count`,
/// so `weights(name, 8)` is the length-8 prefix of `weights(name, 16)`.
pub fn weights(name: &str, count: usize) -> Vec<f32> {
    let mut state = fnv1a_64(name);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        state = xorshift64_next(state);
        out.push(draw_to_f32(state));
    }
    out
}
