//! Generator suspension logic.
//!
//! This module handles capturing frame state when a generator yields,
//! including register values and the resume point.
//!
//! # Overview
//!
//! When a generator executes a `yield` statement:
//!
//! 1. **Liveness Analysis**: Determine which registers contain live values
//! 2. **Register Capture**: Save only live registers (sparse capture)
//! 3. **Resume Point**: Record the PC offset to resume from
//! 4. **State Transition**: Move generator to Suspended state
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────────┐
//! │                       Suspension Flow                                   │
//! ├────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │  yield value                                                           │
//! │     │                                                                   │
//! │     ▼                                                                   │
//! │  ┌──────────────────┐                                                  │
//! │  │ Liveness Bitmap  │ ── From code object or computed                  │
//! │  └────────┬─────────┘                                                  │
//! │           │                                                             │
//! │           ▼                                                             │
//! │  ┌──────────────────┐                                                  │
//! │  │ save_live_regs   │ ── Only copy marked registers                    │
//! │  └────────┬─────────┘                                                  │
//! │           │                                                             │
//! │           ▼                                                             │
//! │  ┌──────────────────┐                                                  │
//! │  │ SuspendResult    │ ── Captured state for resumption                 │
//! │  └──────────────────┘                                                  │
//! │                                                                         │
//! └────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance
//!
//! The suspension process is optimized for:
//!
//! - **Sparse Capture**: Only live registers are saved (popcnt for count)
//! - **Liveness Bitmap**: 64-bit bitmap covers up to 64 registers
//! - **Zero Allocation**: Uses pooled frame storage
//! - **Cache Friendly**: Sequential register iteration

use crate::ops::generator::frame_pool::PooledFrame;
use prism_core::Value;

// =============================================================================
// Suspend Result
// =============================================================================

/// Result of suspending a generator.
///
/// Contains all state needed to resume the generator later.
#[derive(Debug)]
pub struct SuspendResult {
    /// The resume index (which yield point to return to).
    pub resume_index: u32,
    /// Number of live registers captured.
    pub live_count: usize,
    /// Liveness bitmap for register restoration.
    pub liveness: u64,
    /// Captured register values (sparse).
    pub registers: Vec<Value>,
}

impl SuspendResult {
    /// Create a new suspend result.
    #[inline]
    pub fn new(resume_index: u32, liveness: u64) -> Self {
        let live_count = liveness.count_ones() as usize;
        Self {
            resume_index,
            live_count,
            liveness,
            registers: Vec::with_capacity(live_count),
        }
    }

    /// Check if there are any captured registers.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.registers.is_empty()
    }

    /// Get the number of captured registers.
    #[inline]
    pub fn len(&self) -> usize {
        self.registers.len()
    }
}

// =============================================================================
// Core Suspension Functions
// =============================================================================

/// Capture the frame state for a generator yield.
///
/// This function creates a `SuspendResult` containing only the live registers,
/// using the liveness bitmap to determine which registers to save.
///
/// # Arguments
///
/// * `registers` - The frame's register array
/// * `liveness` - Bitmap indicating which registers are live (bit N = register N)
/// * `resume_index` - The yield point index to resume from
///
/// # Returns
///
/// A `SuspendResult` containing the captured state.
///
/// # Example
///
/// ```ignore
/// let registers = [Value::int(1), Value::int(2), Value::none(), Value::int(4)];
/// let liveness = 0b1011; // Registers 0, 1, 3 are live
/// let result = capture_generator_frame(&registers, liveness, 0)?;
/// assert_eq!(result.live_count, 3);
/// ```
#[inline]
pub fn capture_generator_frame(
    registers: &[Value],
    liveness: u64,
    resume_index: u32,
) -> Result<SuspendResult, SuspendError> {
    let mut result = SuspendResult::new(resume_index, liveness);

    // Capture only live registers
    for (i, reg) in registers.iter().enumerate() {
        if i >= 64 {
            break; // Liveness bitmap only covers 64 registers
        }
        if (liveness >> i) & 1 == 1 {
            result.registers.push(reg.clone());
        }
    }

    Ok(result)
}

/// Capture live registers directly into a pooled frame.
///
/// This is the optimized path that avoids intermediate allocation
/// by writing directly to the pooled frame storage.
///
/// # Arguments
///
/// * `registers` - The frame's register array
/// * `liveness` - Bitmap indicating which registers are live
/// * `resume_index` - The yield point index to resume from
/// * `frame` - Pooled frame to store into
///
/// # Returns
///
/// The number of registers captured.
#[inline]
pub fn capture_to_pooled_frame(
    registers: &[Value],
    liveness: u64,
    resume_index: u32,
    frame: &mut PooledFrame,
) -> usize {
    frame.set_metadata(liveness, resume_index);
    frame.store_registers(registers, liveness);
    liveness.count_ones() as usize
}

/// Save only live registers based on liveness bitmap.
///
/// This function performs the core sparse copy operation.
///
/// # Arguments
///
/// * `src` - Source register array
/// * `dst` - Destination buffer (must have capacity for live_count registers)
/// * `liveness` - Bitmap indicating which registers are live
///
/// # Returns
///
/// The number of registers saved.
#[inline]
pub fn save_live_registers(src: &[Value], dst: &mut Vec<Value>, liveness: u64) -> usize {
    let mut count = 0;

    for (i, reg) in src.iter().enumerate() {
        if i >= 64 {
            break;
        }
        if (liveness >> i) & 1 == 1 {
            dst.push(reg.clone());
            count += 1;
        }
    }

    count
}

/// Compute liveness bitmap from a list of live register indices.
///
/// # Arguments
///
/// * `live_indices` - Iterator of register indices that are live
///
/// # Returns
///
/// A 64-bit bitmap where bit N is set if register N is live.
#[inline]
pub fn compute_liveness_bitmap<I>(live_indices: I) -> u64
where
    I: IntoIterator<Item = usize>,
{
    let mut bitmap = 0u64;
    for idx in live_indices {
        if idx < 64 {
            bitmap |= 1 << idx;
        }
    }
    bitmap
}

/// Count the number of live registers from a liveness bitmap.
#[inline(always)]
pub fn count_live_registers(liveness: u64) -> usize {
    liveness.count_ones() as usize
}

/// Check if a specific register is live.
#[inline(always)]
pub fn is_register_live(liveness: u64, register: usize) -> bool {
    register < 64 && (liveness >> register) & 1 == 1
}

/// Get the Nth live register index.
///
/// # Arguments
///
/// * `liveness` - The liveness bitmap
/// * `n` - Which live register to get (0-indexed)
///
/// # Returns
///
/// The register index of the Nth live register, or None if n >= live_count.
#[inline]
pub fn nth_live_register(liveness: u64, n: usize) -> Option<usize> {
    let mut remaining = n;
    let mut bitmap = liveness;

    while bitmap != 0 {
        let trailing = bitmap.trailing_zeros() as usize;
        if remaining == 0 {
            return Some(trailing);
        }
        remaining -= 1;
        bitmap &= bitmap - 1; // Clear lowest set bit
    }

    None
}

// =============================================================================
// Suspend Error
// =============================================================================

/// Error during generator suspension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SuspendError {
    /// Generator is not in a suspendable state.
    NotSuspendable,
    /// No active generator to suspend.
    NoActiveGenerator,
    /// Resume index out of range.
    InvalidResumeIndex(u32),
    /// Too many live registers (exceeds 64).
    TooManyRegisters(usize),
}

impl std::fmt::Display for SuspendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SuspendError::NotSuspendable => write!(f, "Generator is not in a suspendable state"),
            SuspendError::NoActiveGenerator => write!(f, "No active generator to suspend"),
            SuspendError::InvalidResumeIndex(idx) => write!(f, "Invalid resume index: {}", idx),
            SuspendError::TooManyRegisters(count) => {
                write!(f, "Too many live registers: {} (max 64)", count)
            }
        }
    }
}

impl std::error::Error for SuspendError {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
