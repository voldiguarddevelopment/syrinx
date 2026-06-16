//! syrinx-stream — packet streaming, ring buffer, cpal out, TTFB path.
//!
//! T-07.01 implements the deterministic in-memory buffer/packetizer: a
//! fixed-capacity FIFO [`RingBuffer`] over `f32` audio samples. Samples come out
//! in exactly the order they went in; the empty signal is `None`, the full signal
//! is [`BufferError::Backpressure`] (no overwrite of live data), and the ring
//! wraps at the capacity boundary. No `cpal` device output here — that and the
//! TTFB path are separate tasks.
//!
//! T-07.04 adds the [`resample`] module: a deterministic 48kHz→8kHz telephony
//! downsampler (anti-alias band-limit via `syrinx-vocoder`, then 6:1 decimation).

pub mod resample;

/// Error returned by [`RingBuffer::push`] when the buffer cannot accept a sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferError {
    /// The buffer is full (live count equals capacity): the producer must wait
    /// rather than overwrite live data.
    Backpressure,
}

/// A deterministic fixed-capacity FIFO ring buffer over `f32` samples.
///
/// The backing store has exactly `capacity` slots. `head` is the index of the
/// oldest live sample; `len` is the number of live samples. The tail (next write
/// slot) is derived as `(head + len) % capacity`, so both indices wrap the store.
pub struct RingBuffer {
    buf: Vec<f32>,
    head: usize,
    len: usize,
}

impl RingBuffer {
    /// Create an empty ring buffer that can hold `capacity` live samples.
    pub fn new(capacity: usize) -> Self {
        RingBuffer {
            buf: vec![0.0; capacity],
            head: 0,
            len: 0,
        }
    }

    /// Enqueue one sample at the tail in FIFO order.
    ///
    /// Returns `Err(BufferError::Backpressure)` without overwriting any live
    /// sample when the live count already equals the capacity. Never panics.
    pub fn push(&mut self, sample: f32) -> Result<(), BufferError> {
        if self.len == self.buf.len() {
            return Err(BufferError::Backpressure);
        }
        let tail = (self.head + self.len) % self.buf.len();
        self.buf[tail] = sample;
        self.len += 1;
        Ok(())
    }

    /// Dequeue the oldest sample from the head in FIFO order.
    ///
    /// Returns `None` on an empty buffer. Never panics.
    pub fn pop(&mut self) -> Option<f32> {
        if self.len == 0 {
            return None;
        }
        let sample = self.buf[self.head];
        self.head = (self.head + 1) % self.buf.len();
        self.len -= 1;
        Some(sample)
    }
}
