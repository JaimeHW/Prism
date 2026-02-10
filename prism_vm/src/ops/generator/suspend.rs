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
mod tests {
    use super::*;

    // ════════════════════════════════════════════════════════════════════════
    // SuspendResult Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_suspend_result_new() {
        let result = SuspendResult::new(42, 0b1010);
        assert_eq!(result.resume_index, 42);
        assert_eq!(result.liveness, 0b1010);
        assert_eq!(result.live_count, 2);
        assert!(result.is_empty()); // No registers captured yet
    }

    #[test]
    fn test_suspend_result_empty_liveness() {
        let result = SuspendResult::new(0, 0);
        assert_eq!(result.live_count, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_suspend_result_full_liveness() {
        let result = SuspendResult::new(0, u64::MAX);
        assert_eq!(result.live_count, 64);
    }

    // ════════════════════════════════════════════════════════════════════════
    // capture_generator_frame Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_capture_empty() {
        let registers: Vec<Value> = vec![];
        let result = capture_generator_frame(&registers, 0, 0).unwrap();

        assert_eq!(result.resume_index, 0);
        assert_eq!(result.live_count, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_capture_all_live() {
        let registers = vec![
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ];
        let result = capture_generator_frame(&registers, 0b111, 5).unwrap();

        assert_eq!(result.resume_index, 5);
        assert_eq!(result.live_count, 3);
        assert_eq!(result.len(), 3);
        assert_eq!(result.registers[0].as_int(), Some(1));
        assert_eq!(result.registers[1].as_int(), Some(2));
        assert_eq!(result.registers[2].as_int(), Some(3));
    }

    #[test]
    fn test_capture_sparse() {
        let registers = vec![
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
            Value::int(4).unwrap(),
        ];
        // Only registers 0 and 3 are live
        let result = capture_generator_frame(&registers, 0b1001, 10).unwrap();

        assert_eq!(result.live_count, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result.registers[0].as_int(), Some(1)); // Register 0
        assert_eq!(result.registers[1].as_int(), Some(4)); // Register 3
    }

    #[test]
    fn test_capture_with_none_values() {
        let registers = vec![
            Value::none(),
            Value::int(42).unwrap(),
            Value::none(),
            Value::bool(true),
        ];
        let result = capture_generator_frame(&registers, 0b1111, 0).unwrap();

        assert_eq!(result.len(), 4);
        assert!(result.registers[0].is_none());
        assert_eq!(result.registers[1].as_int(), Some(42));
        assert!(result.registers[2].is_none());
        assert_eq!(result.registers[3].as_bool(), Some(true));
    }

    #[test]
    fn test_capture_preserves_resume_index() {
        let registers = vec![Value::int(1).unwrap()];

        for idx in [0, 1, 100, u32::MAX] {
            let result = capture_generator_frame(&registers, 0b1, idx).unwrap();
            assert_eq!(result.resume_index, idx);
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // save_live_registers Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_save_live_registers_empty() {
        let src: Vec<Value> = vec![Value::int(1).unwrap(), Value::int(2).unwrap()];
        let mut dst = Vec::new();

        let count = save_live_registers(&src, &mut dst, 0);

        assert_eq!(count, 0);
        assert!(dst.is_empty());
    }

    #[test]
    fn test_save_live_registers_all() {
        let src = vec![Value::int(1).unwrap(), Value::int(2).unwrap()];
        let mut dst = Vec::new();

        let count = save_live_registers(&src, &mut dst, 0b11);

        assert_eq!(count, 2);
        assert_eq!(dst[0].as_int(), Some(1));
        assert_eq!(dst[1].as_int(), Some(2));
    }

    #[test]
    fn test_save_live_registers_sparse() {
        let src: Vec<Value> = (0..8).map(|i| Value::int(i).unwrap()).collect();
        let mut dst = Vec::new();

        // Even registers only
        let count = save_live_registers(&src, &mut dst, 0b01010101);

        assert_eq!(count, 4);
        assert_eq!(dst[0].as_int(), Some(0));
        assert_eq!(dst[1].as_int(), Some(2));
        assert_eq!(dst[2].as_int(), Some(4));
        assert_eq!(dst[3].as_int(), Some(6));
    }

    // ════════════════════════════════════════════════════════════════════════
    // compute_liveness_bitmap Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_compute_liveness_empty() {
        let bitmap = compute_liveness_bitmap(std::iter::empty());
        assert_eq!(bitmap, 0);
    }

    #[test]
    fn test_compute_liveness_single() {
        let bitmap = compute_liveness_bitmap([5]);
        assert_eq!(bitmap, 0b100000);
    }

    #[test]
    fn test_compute_liveness_multiple() {
        let bitmap = compute_liveness_bitmap([0, 2, 4, 6]);
        assert_eq!(bitmap, 0b01010101);
    }

    #[test]
    fn test_compute_liveness_large_index_ignored() {
        let bitmap = compute_liveness_bitmap([0, 64, 65, 100]);
        assert_eq!(bitmap, 0b1); // Only index 0 counted
    }

    #[test]
    fn test_compute_liveness_duplicates() {
        let bitmap = compute_liveness_bitmap([1, 1, 1, 1]);
        assert_eq!(bitmap, 0b10);
    }

    // ════════════════════════════════════════════════════════════════════════
    // count_live_registers Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_count_live_zero() {
        assert_eq!(count_live_registers(0), 0);
    }

    #[test]
    fn test_count_live_one() {
        assert_eq!(count_live_registers(0b1), 1);
        assert_eq!(count_live_registers(0b100), 1);
    }

    #[test]
    fn test_count_live_multiple() {
        assert_eq!(count_live_registers(0b1010), 2);
        assert_eq!(count_live_registers(0b1111), 4);
        assert_eq!(count_live_registers(0xFF), 8);
    }

    #[test]
    fn test_count_live_max() {
        assert_eq!(count_live_registers(u64::MAX), 64);
    }

    // ════════════════════════════════════════════════════════════════════════
    // is_register_live Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_is_register_live_empty() {
        assert!(!is_register_live(0, 0));
        assert!(!is_register_live(0, 63));
    }

    #[test]
    fn test_is_register_live_full() {
        for i in 0..64 {
            assert!(is_register_live(u64::MAX, i));
        }
    }

    #[test]
    fn test_is_register_live_sparse() {
        let liveness = 0b1010;
        assert!(!is_register_live(liveness, 0));
        assert!(is_register_live(liveness, 1));
        assert!(!is_register_live(liveness, 2));
        assert!(is_register_live(liveness, 3));
    }

    #[test]
    fn test_is_register_live_out_of_range() {
        assert!(!is_register_live(u64::MAX, 64));
        assert!(!is_register_live(u64::MAX, 100));
    }

    // ════════════════════════════════════════════════════════════════════════
    // nth_live_register Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_nth_live_empty() {
        assert_eq!(nth_live_register(0, 0), None);
    }

    #[test]
    fn test_nth_live_first() {
        assert_eq!(nth_live_register(0b1, 0), Some(0));
        assert_eq!(nth_live_register(0b100, 0), Some(2));
    }

    #[test]
    fn test_nth_live_sequential() {
        let liveness = 0b1010_1010;
        assert_eq!(nth_live_register(liveness, 0), Some(1));
        assert_eq!(nth_live_register(liveness, 1), Some(3));
        assert_eq!(nth_live_register(liveness, 2), Some(5));
        assert_eq!(nth_live_register(liveness, 3), Some(7));
        assert_eq!(nth_live_register(liveness, 4), None);
    }

    #[test]
    fn test_nth_live_out_of_range() {
        assert_eq!(nth_live_register(0b1, 1), None);
        assert_eq!(nth_live_register(0b11, 2), None);
    }

    // ════════════════════════════════════════════════════════════════════════
    // SuspendError Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_suspend_error_display() {
        assert_eq!(
            SuspendError::NotSuspendable.to_string(),
            "Generator is not in a suspendable state"
        );
        assert_eq!(
            SuspendError::NoActiveGenerator.to_string(),
            "No active generator to suspend"
        );
        assert_eq!(
            SuspendError::InvalidResumeIndex(42).to_string(),
            "Invalid resume index: 42"
        );
        assert_eq!(
            SuspendError::TooManyRegisters(100).to_string(),
            "Too many live registers: 100 (max 64)"
        );
    }

    // ════════════════════════════════════════════════════════════════════════
    // Edge Case Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_capture_more_registers_than_liveness_bits() {
        // Registers beyond bit 63 are ignored
        let registers: Vec<Value> = (0..100).map(|i| Value::int(i).unwrap()).collect();
        let result = capture_generator_frame(&registers, u64::MAX, 0).unwrap();

        // Only 64 registers captured
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_capture_fewer_registers_than_liveness_bits() {
        // Liveness marks more registers than exist
        let registers = vec![Value::int(1).unwrap(), Value::int(2).unwrap()];
        let result = capture_generator_frame(&registers, 0b1111, 0).unwrap();

        // Only 2 actually captured
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_liveness_bitmap_boundary() {
        // Test bit 63
        let bitmap = compute_liveness_bitmap([63]);
        assert_eq!(bitmap, 1 << 63);
        assert!(is_register_live(bitmap, 63));
        assert_eq!(nth_live_register(bitmap, 0), Some(63));
    }

    // ════════════════════════════════════════════════════════════════════════
    // capture_to_pooled_frame Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_capture_to_pooled_frame() {
        use crate::ops::generator::frame_pool::{GeneratorFramePool, PooledFrame};

        let mut pool = GeneratorFramePool::new();
        let mut frame = pool.allocate(4);

        let registers = vec![
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ];
        let liveness = 0b101; // Registers 0 and 2

        let count = capture_to_pooled_frame(&registers, liveness, 42, &mut frame);

        assert_eq!(count, 2);
        assert_eq!(frame.resume_index(), 42);
        assert_eq!(frame.liveness(), 0b101);
    }
}
