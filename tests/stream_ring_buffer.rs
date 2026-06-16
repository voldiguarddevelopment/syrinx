//! Frozen RED tests for T-07.01 — buffer streaming audio packets.
//!
//! These pin the three acceptance criteria against the real public API the green
//! phase must build in `syrinx-stream`:
//!
//!   * `RingBuffer` — a deterministic FIFO ring buffer over `f32` audio samples
//!     with a fixed capacity. Constructed with `RingBuffer::new(capacity: usize)`.
//!   * `RingBuffer::push(&mut self, sample: f32) -> Result<(), BufferError>` —
//!     enqueue one sample at the tail. When the live count already equals the
//!     capacity the buffer is full and the push returns `Err(BufferError::
//!     Backpressure)` WITHOUT overwriting any live sample. Never panics.
//!   * `RingBuffer::pop(&mut self) -> Option<f32>` — dequeue the oldest sample
//!     from the head in FIFO order. On an empty buffer it returns `None`. Never
//!     panics.
//!   * `BufferError::Backpressure` — the typed full-buffer signal.
//!
//! Contract (list.md / DESIGN §T7.01): a deterministic in-memory ring buffer plus
//! packetizer over a synthetic `f32` sample stream. Samples come out in exactly
//! the order they went in (FIFO, element-for-element — order, not just
//! membership). The ring wraps at the capacity boundary: pushing and popping more
//! than `C` total samples (kept interleaved so the live count never exceeds `C`)
//! returns every sample in order rather than overwriting live data. The empty
//! signal is `None`; the full signal is `Err(BufferError::Backpressure)` with no
//! overwrite; the full boundary is pinned at `C-1` (still `Ok`) versus `C` (the
//! first `Err`). Nothing panics on any path.
//!
//! RED: `syrinx-stream` exposes no `RingBuffer`/`BufferError` yet, so none of
//! these symbols resolve and the test target fails to build — every criterion is
//! unmet. GREEN implements them so each assertion below holds.

use syrinx_stream::{BufferError, RingBuffer};

/// Push every sample in `samples` into a fresh buffer of `capacity`, asserting
/// each push succeeds, then drain the whole buffer and return the popped order.
fn fill_then_drain(capacity: usize, samples: &[f32]) -> Vec<f32> {
    let mut rb = RingBuffer::new(capacity);
    for &s in samples {
        assert!(
            rb.push(s).is_ok(),
            "push of {s} within capacity {capacity} must succeed"
        );
    }
    let mut out = Vec::new();
    while let Some(v) = rb.pop() {
        out.push(v);
    }
    out
}

// ----------------------------------------------------------------------------
// C1 — N pushes (N <= C) then N pops yield FIFO order, element-for-element; a
//      reordered expectation must FAIL (order is pinned, not just membership).
// ----------------------------------------------------------------------------

/// Pushing N distinct samples (N == C here) then popping N returns them in the
/// exact order they were pushed.
#[test]
fn test_fifo_order_element_for_element() {
    let samples = [11.0_f32, 22.0, 33.0, 44.0, 55.0];
    let out = fill_then_drain(5, &samples);

    // Exact, element-for-element FIFO order.
    assert_eq!(out, samples.to_vec());
    // Each position individually, so a single transposed element is caught.
    for (i, &s) in samples.iter().enumerate() {
        assert_eq!(out[i], s, "sample at FIFO position {i} must match input");
    }
}

/// The same drained output, compared against REORDERED expectations, must NOT be
/// equal — pinning order rather than mere membership. The reversed and a single
/// adjacent swap are both genuine reorderings of this non-palindromic input.
#[test]
fn test_reordered_expectation_fails() {
    let samples = [11.0_f32, 22.0, 33.0, 44.0, 55.0];
    let out = fill_then_drain(5, &samples);

    let mut reversed = samples.to_vec();
    reversed.reverse();
    assert_ne!(out, reversed, "reversed order must not match a FIFO drain");

    // A single adjacent swap is also a reordering and must not match.
    let mut swapped = samples.to_vec();
    swapped.swap(0, 1);
    assert_ne!(out, swapped, "a single transposition must not match a FIFO drain");

    // Sanity: the reorderings really do differ from the input (the buffer is not
    // being credited for order just because the expectations were degenerate).
    assert_ne!(reversed, samples.to_vec());
    assert_ne!(swapped, samples.to_vec());
}

// ----------------------------------------------------------------------------
// C2 — pushing/popping more than C total samples, interleaved so the live count
//      never exceeds C, returns every sample in order: the ring wraps at the
//      capacity boundary instead of overwriting live data.
// ----------------------------------------------------------------------------

/// With capacity 3 we stream 8 distinct samples (8 > 3) through the buffer:
/// prime it to full, then for each remaining sample pop one and push one (so the
/// live count holds at 3 and the head/tail indices wrap the ring multiple times),
/// then drain the tail. Every sample must emerge exactly once, in push order.
#[test]
fn test_ring_wraps_without_overwriting_live_data() {
    let capacity = 3;
    let inputs = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut rb = RingBuffer::new(capacity);

    let mut out = Vec::new();

    // Prime: fill the ring to exactly capacity.
    for &s in &inputs[..capacity] {
        assert!(rb.push(s).is_ok(), "priming push must succeed");
    }

    // Steady state: pop one, push one, for every remaining input. The live count
    // returns to `capacity` after each push, so no push ever hits backpressure,
    // yet head and tail both advance past the end of the backing store and wrap.
    for &s in &inputs[capacity..] {
        out.push(rb.pop().expect("a primed/steady ring is never empty here"));
        assert!(
            rb.push(s).is_ok(),
            "push after a pop keeps live count <= capacity, so must succeed"
        );
    }

    // Drain whatever remains.
    while let Some(v) = rb.pop() {
        out.push(v);
    }

    // Every sample, exactly once, in the order it was pushed — proof the wrap did
    // not overwrite live data.
    assert_eq!(out, inputs.to_vec());
}

// ----------------------------------------------------------------------------
// C3 — empty pop is None; full push is Err(Backpressure) with no overwrite; the
//      full boundary is pinned at C-1 (Ok) vs C (Err); nothing panics.
// ----------------------------------------------------------------------------

/// Popping a freshly constructed (empty) buffer returns `None`, and popping a
/// buffer that has been emptied again returns `None` — the empty signal, not a
/// panic and not a stale value.
#[test]
fn test_pop_empty_returns_none() {
    let mut rb = RingBuffer::new(4);
    assert_eq!(rb.pop(), None, "popping a fresh buffer must yield None");

    // Push one then pop it (a non-empty pop yields Some), then the buffer is
    // empty again and the next pop is None.
    rb.push(7.0).expect("push into an empty buffer must succeed");
    assert_eq!(rb.pop(), Some(7.0), "a non-empty pop returns the live sample");
    assert_eq!(rb.pop(), None, "popping an emptied buffer must yield None again");
}

/// The full boundary: with capacity 4, the pushes at live counts 0..=3 all
/// succeed (the push at live count == C-1 == 3 is the last `Ok`), and the push at
/// live count == C == 4 reports backpressure. Both sides of the boundary are
/// pinned.
#[test]
fn test_full_boundary_pushes_then_backpressure() {
    let capacity = 4;
    let mut rb = RingBuffer::new(capacity);

    // Live counts 0, 1, 2 -> still room, all Ok.
    for &s in &[10.0_f32, 20.0, 30.0] {
        assert!(rb.push(s).is_ok(), "push below the full boundary must succeed");
    }

    // Live count == C-1 == 3: the LAST successful push (pins the C-1 side).
    assert!(
        rb.push(40.0).is_ok(),
        "push at live count == C-1 must still succeed"
    );

    // Live count == C == 4: the buffer is full, push reports backpressure.
    assert!(
        matches!(rb.push(50.0), Err(BufferError::Backpressure)),
        "push at live count == C must return Err(BufferError::Backpressure)"
    );
}

/// A full-buffer push must NOT overwrite live data: after backpressure is
/// reported, draining the buffer yields exactly the originally accepted samples,
/// in order — the rejected sample never entered the ring.
#[test]
fn test_backpressure_does_not_overwrite() {
    let capacity = 4;
    let accepted = [10.0_f32, 20.0, 30.0, 40.0];
    let mut rb = RingBuffer::new(capacity);

    for &s in &accepted {
        assert!(rb.push(s).is_ok(), "filling to capacity must succeed");
    }

    // Two rejected pushes while full — each reports backpressure.
    assert!(matches!(rb.push(99.0), Err(BufferError::Backpressure)));
    assert!(matches!(rb.push(88.0), Err(BufferError::Backpressure)));

    // Draining returns exactly the accepted samples, in order: the rejected
    // values (99.0, 88.0) never overwrote anything.
    let mut out = Vec::new();
    while let Some(v) = rb.pop() {
        out.push(v);
    }
    assert_eq!(out, accepted.to_vec());
}
