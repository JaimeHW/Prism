//! Generator state management.
//!
//! This module provides the `GeneratorHeader` which uses tagged encoding to
//! pack both generator state AND resume index into a single u32 for maximum
//! performance on state checks and dispatch.
//!
//! # Encoding
//!
//! ```text
//! Bits 0-1:  State (Created=0, Running=1, Suspended=2, Exhausted=3)
//! Bits 2-31: Resume index (yield point ID, max 2^30 = 1 billion yield points)
//! ```
//!
//! # Performance
//!
//! - Single atomic load for both state check AND dispatch target
//! - State comparison is a 2-bit mask operation
//! - Resume index extraction is a single right-shift

use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};

// ============================================================================
// Generator State
// ============================================================================

/// Generator execution state.
///
/// Packed into 2 bits for single-instruction comparison.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GeneratorState {
    /// Generator created but never started (before first `next()`).
    Created = 0,
    /// Generator is currently executing (reentry check).
    Running = 1,
    /// Generator is suspended at a yield point.
    Suspended = 2,
    /// Generator has completed or been closed.
    Exhausted = 3,
}

impl GeneratorState {
    /// Number of bits used to encode state.
    pub const BITS: u32 = 2;

    /// Mask for extracting state from header.
    pub const MASK: u32 = (1 << Self::BITS) - 1; // 0b11

    /// Creates state from raw 2-bit value.
    #[inline(always)]
    pub const fn from_bits(bits: u32) -> Self {
        match bits & Self::MASK {
            0 => Self::Created,
            1 => Self::Running,
            2 => Self::Suspended,
            3 => Self::Exhausted,
            _ => unreachable!(), // Compiler can prove this is impossible
        }
    }

    /// Returns true if the generator can be resumed.
    #[inline(always)]
    pub const fn is_resumable(self) -> bool {
        matches!(self, Self::Created | Self::Suspended)
    }

    /// Returns true if the generator is finished.
    #[inline(always)]
    pub const fn is_finished(self) -> bool {
        matches!(self, Self::Exhausted)
    }

    /// Returns true if yielding is valid (generator is running).
    #[inline(always)]
    pub const fn can_yield(self) -> bool {
        matches!(self, Self::Running)
    }

    /// Returns the Python name for this state.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Created => "GEN_CREATED",
            Self::Running => "GEN_RUNNING",
            Self::Suspended => "GEN_SUSPENDED",
            Self::Exhausted => "GEN_CLOSED",
        }
    }
}

impl fmt::Display for GeneratorState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl Default for GeneratorState {
    #[inline]
    fn default() -> Self {
        Self::Created
    }
}

// ============================================================================
// Generator Header
// ============================================================================

/// Tagged header combining state and resume index.
///
/// This is the core optimization: a single u32 encodes everything needed
/// to check generator state AND dispatch to the correct resume point.
///
/// # Memory Layout
///
/// ```text
/// +-------------------+-------+
/// | Resume Index (30) | State |
/// |                   | (2)   |
/// +-------------------+-------+
/// MSB                       LSB
/// ```
///
/// # Thread Safety
///
/// Uses atomic operations for safe concurrent access (though generators
/// themselves are not thread-safe, the state must be atomically updated).
#[repr(transparent)]
pub struct GeneratorHeader {
    bits: AtomicU32,
}

impl GeneratorHeader {
    /// Maximum resume index (2^30 - 1).
    pub const MAX_RESUME_INDEX: u32 = (1 << 30) - 1;

    /// Shift amount to access resume index.
    const RESUME_SHIFT: u32 = GeneratorState::BITS;

    /// Creates a new header in Created state with resume index 0.
    #[inline]
    pub fn new() -> Self {
        Self {
            bits: AtomicU32::new(GeneratorState::Created as u32),
        }
    }

    /// Creates a header with specific state and resume index.
    #[inline]
    pub fn with_state_and_index(state: GeneratorState, resume_index: u32) -> Self {
        debug_assert!(resume_index <= Self::MAX_RESUME_INDEX);
        let bits = (resume_index << Self::RESUME_SHIFT) | (state as u32);
        Self {
            bits: AtomicU32::new(bits),
        }
    }

    /// Gets the current state.
    #[inline(always)]
    pub fn state(&self) -> GeneratorState {
        GeneratorState::from_bits(self.bits.load(Ordering::Acquire))
    }

    /// Gets the current resume index.
    #[inline(always)]
    pub fn resume_index(&self) -> u32 {
        self.bits.load(Ordering::Acquire) >> Self::RESUME_SHIFT
    }

    /// Gets both state and resume index in a single atomic load.
    #[inline(always)]
    pub fn state_and_index(&self) -> (GeneratorState, u32) {
        let bits = self.bits.load(Ordering::Acquire);
        (GeneratorState::from_bits(bits), bits >> Self::RESUME_SHIFT)
    }

    /// Sets the state (preserving resume index).
    #[inline]
    pub fn set_state(&self, state: GeneratorState) {
        let old = self.bits.load(Ordering::Relaxed);
        let new = (old & !GeneratorState::MASK) | (state as u32);
        self.bits.store(new, Ordering::Release);
    }

    /// Sets both state and resume index atomically.
    #[inline]
    pub fn set_state_and_index(&self, state: GeneratorState, resume_index: u32) {
        debug_assert!(resume_index <= Self::MAX_RESUME_INDEX);
        let bits = (resume_index << Self::RESUME_SHIFT) | (state as u32);
        self.bits.store(bits, Ordering::Release);
    }

    /// Transitions to Running state if currently resumable.
    /// Returns previous state on success, None if not resumable.
    #[inline]
    pub fn try_start(&self) -> Option<GeneratorState> {
        let old = self.bits.load(Ordering::Relaxed);
        let old_state = GeneratorState::from_bits(old);

        if !old_state.is_resumable() {
            return None;
        }

        let new = (old & !GeneratorState::MASK) | (GeneratorState::Running as u32);
        self.bits.store(new, Ordering::Release);
        Some(old_state)
    }

    /// Transitions to Suspended state with new resume index.
    /// Only valid when Running.
    #[inline]
    pub fn suspend(&self, resume_index: u32) {
        debug_assert_eq!(self.state(), GeneratorState::Running);
        debug_assert!(resume_index <= Self::MAX_RESUME_INDEX);
        let bits = (resume_index << Self::RESUME_SHIFT) | (GeneratorState::Suspended as u32);
        self.bits.store(bits, Ordering::Release);
    }

    /// Transitions to Exhausted state.
    #[inline]
    pub fn exhaust(&self) {
        // Preserve resume index for debugging
        let old = self.bits.load(Ordering::Relaxed);
        let new = (old & !GeneratorState::MASK) | (GeneratorState::Exhausted as u32);
        self.bits.store(new, Ordering::Release);
    }

    /// Returns true if generator is in Running state (reentry check).
    #[inline(always)]
    pub fn is_running(&self) -> bool {
        self.state() == GeneratorState::Running
    }

    /// Returns true if generator can be resumed.
    #[inline(always)]
    pub fn is_resumable(&self) -> bool {
        self.state().is_resumable()
    }

    /// Returns true if generator is exhausted.
    #[inline(always)]
    pub fn is_exhausted(&self) -> bool {
        self.state().is_finished()
    }

    /// Returns the raw bits (for debugging).
    #[inline]
    pub fn raw(&self) -> u32 {
        self.bits.load(Ordering::Relaxed)
    }
}

impl Default for GeneratorHeader {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for GeneratorHeader {
    fn clone(&self) -> Self {
        Self {
            bits: AtomicU32::new(self.bits.load(Ordering::Relaxed)),
        }
    }
}

impl fmt::Debug for GeneratorHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (state, index) = self.state_and_index();
        f.debug_struct("GeneratorHeader")
            .field("state", &state)
            .field("resume_index", &index)
            .finish()
    }
}
