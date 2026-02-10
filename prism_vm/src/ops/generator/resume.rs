//! Generator resumption logic.
//!
//! This module handles restoring frame state when a generator is resumed,
//! including register restoration and dispatch to the correct yield point.
//!
//! # Overview
//!
//! When a generator is resumed via `next()`, `send()`, or `throw()`:
//!
//! 1. **State Check**: Verify generator is in Suspended state
//! 2. **Register Restore**: Copy saved registers back to frame
//! 3. **Send Value**: Place sent value in appropriate register (if any)
//! 4. **Exception**: Inject exception if `throw()` was called
//! 5. **Dispatch**: Jump to the resume point PC
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────────┐
//! │                       Resumption Flow                                   │
//! ├────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │  generator.send(value)                                                 │
//! │     │                                                                   │
//! │     ▼                                                                   │
//! │  ┌──────────────────┐                                                  │
//! │  │ Validate State   │ ── Must be Suspended                             │
//! │  └────────┬─────────┘                                                  │
//! │           │                                                             │
//! │           ▼                                                             │
//! │  ┌──────────────────┐                                                  │
//! │  │ Restore Registers│ ── From saved frame                              │
//! │  └────────┬─────────┘                                                  │
//! │           │                                                             │
//! │           ▼                                                             │
//! │  ┌──────────────────┐                                                  │
//! │  │ Inject SendValue │ ── Into YIELD_RESULT register                    │
//! │  └────────┬─────────┘                                                  │
//! │           │                                                             │
//! │           ▼                                                             │
//! │  ┌──────────────────┐                                                  │
//! │  │ dispatch_to_pc   │ ── Jump to resume point                          │
//! │  └──────────────────┘                                                  │
//! │                                                                         │
//! └────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance
//!
//! - **Sparse Restore**: Only restores live registers
//! - **Computed Goto**: Uses resume table for O(1) dispatch
//! - **Zero Allocation**: All operations are in-place

use crate::ops::generator::frame_pool::PooledFrame;
use crate::ops::generator::resume_cache::ResumeTable;
use crate::ops::generator::suspend::SuspendResult;
use prism_core::Value;

// =============================================================================
// Resume Result
// =============================================================================

/// Result of resuming a generator.
#[derive(Debug)]
pub struct ResumeResult {
    /// The PC to jump to.
    pub target_pc: u32,
    /// Number of registers restored.
    pub registers_restored: usize,
    /// Whether a send value was injected.
    pub had_send_value: bool,
}

impl ResumeResult {
    /// Create a new resume result.
    #[inline]
    pub fn new(target_pc: u32, registers_restored: usize) -> Self {
        Self {
            target_pc,
            registers_restored,
            had_send_value: false,
        }
    }

    /// Mark that a send value was injected.
    #[inline]
    pub fn with_send_value(mut self) -> Self {
        self.had_send_value = true;
        self
    }
}

// =============================================================================
// Resume Dispatcher
// =============================================================================

/// High-performance dispatcher for generator resumption.
///
/// Uses cached resume tables for O(1) dispatch to yield points.
pub struct ResumeDispatcher {
    /// Default target when no table is available.
    fallback_pc: u32,
}

impl ResumeDispatcher {
    /// Create a new dispatcher.
    #[inline]
    pub fn new() -> Self {
        Self { fallback_pc: 0 }
    }

    /// Create a dispatcher with a fallback PC.
    #[inline]
    pub fn with_fallback(fallback_pc: u32) -> Self {
        Self { fallback_pc }
    }

    /// Dispatch to a resume point.
    ///
    /// # Arguments
    ///
    /// * `resume_table` - The resume table for this code object
    /// * `resume_index` - The yield point to resume to
    ///
    /// # Returns
    ///
    /// The PC to jump to.
    #[inline]
    pub fn dispatch(&self, resume_table: Option<&ResumeTable>, resume_index: u32) -> u32 {
        if let Some(table) = resume_table {
            table.get_pc(resume_index).unwrap_or(self.fallback_pc)
        } else {
            self.fallback_pc
        }
    }

    /// Dispatch using a direct PC lookup.
    ///
    /// This is used when the resume table is known to be dense
    /// and the resume index is the direct PC offset.
    #[inline]
    pub fn dispatch_direct(&self, resume_index: u32) -> u32 {
        resume_index
    }
}

impl Default for ResumeDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Core Resumption Functions
// =============================================================================

/// Restore a generator's frame from a suspend result.
///
/// This function restores register values based on the liveness bitmap,
/// placing values back in their original register positions.
///
/// # Arguments
///
/// * `suspended` - The suspend result containing saved state
/// * `registers` - The frame's register array to restore into
///
/// # Example
///
/// ```ignore
/// let suspended = capture_generator_frame(&old_registers, 0b1011, 0)?;
/// restore_generator_frame(&suspended, &mut new_registers);
/// // new_registers[0], [1], [3] are now restored
/// ```
#[inline]
pub fn restore_generator_frame(suspended: &SuspendResult, registers: &mut [Value]) {
    let mut saved_idx = 0;

    for (reg_idx, reg) in registers.iter_mut().enumerate() {
        if reg_idx >= 64 {
            break;
        }
        if (suspended.liveness >> reg_idx) & 1 == 1 {
            if saved_idx < suspended.registers.len() {
                *reg = suspended.registers[saved_idx].clone();
                saved_idx += 1;
            }
        }
    }
}

/// Restore registers from a pooled frame.
///
/// # Arguments
///
/// * `frame` - The pooled frame containing saved state
/// * `registers` - The frame's register array to restore into
#[inline]
pub fn restore_from_pooled_frame(frame: &PooledFrame, registers: &mut [Value]) {
    let liveness = frame.liveness();
    frame.restore_registers(registers, liveness);
}

/// Dispatch to a resume point using the resume table.
///
/// # Arguments
///
/// * `resume_table` - The resume table for this code object
/// * `resume_index` - The yield point to resume to
///
/// # Returns
///
/// The PC to jump to, or None if the resume index is invalid.
#[inline]
pub fn dispatch_to_resume_point(
    resume_table: Option<&ResumeTable>,
    resume_index: u32,
) -> Option<u32> {
    resume_table.and_then(|table| table.get_pc(resume_index))
}

/// Inject a send value into the yield result register.
///
/// In Python generators, the value sent via `generator.send(value)`
/// becomes the result of the `yield` expression.
///
/// # Arguments
///
/// * `registers` - The frame's register array
/// * `yield_result_reg` - The register that receives the yield result
/// * `send_value` - The value to inject
#[inline]
pub fn inject_send_value(registers: &mut [Value], yield_result_reg: usize, send_value: Value) {
    if yield_result_reg < registers.len() {
        registers[yield_result_reg] = send_value;
    }
}

/// Prepare registers for throw operation.
///
/// This sets up the frame state so that when execution resumes,
/// the exception will be raised at the yield point.
///
/// # Arguments
///
/// * `registers` - The frame's register array
/// * `exception_reg` - The register to store the exception in
/// * `exception` - The exception value to throw
#[inline]
pub fn prepare_throw(registers: &mut [Value], exception_reg: usize, exception: Value) {
    if exception_reg < registers.len() {
        registers[exception_reg] = exception;
    }
}

// =============================================================================
// Resume Error
// =============================================================================

/// Error during generator resumption.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResumeError {
    /// Generator is not suspended.
    NotSuspended,
    /// Generator has completed.
    GeneratorExhausted,
    /// Invalid resume index.
    InvalidResumeIndex(u32),
    /// Cannot resume with a value on first iteration.
    CantSendNonNone,
    /// No saved state to restore.
    NoSavedState,
}

impl std::fmt::Display for ResumeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResumeError::NotSuspended => write!(f, "Generator is not suspended"),
            ResumeError::GeneratorExhausted => write!(f, "Generator has already completed"),
            ResumeError::InvalidResumeIndex(idx) => write!(f, "Invalid resume index: {}", idx),
            ResumeError::CantSendNonNone => {
                write!(f, "Can't send non-None value to a just-started generator")
            }
            ResumeError::NoSavedState => write!(f, "No saved state to restore"),
        }
    }
}

impl std::error::Error for ResumeError {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::generator::suspend::capture_generator_frame;

    // ════════════════════════════════════════════════════════════════════════
    // ResumeResult Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_resume_result_new() {
        let result = ResumeResult::new(100, 5);
        assert_eq!(result.target_pc, 100);
        assert_eq!(result.registers_restored, 5);
        assert!(!result.had_send_value);
    }

    #[test]
    fn test_resume_result_with_send_value() {
        let result = ResumeResult::new(100, 5).with_send_value();
        assert!(result.had_send_value);
    }

    // ════════════════════════════════════════════════════════════════════════
    // ResumeDispatcher Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_dispatcher_new() {
        let dispatcher = ResumeDispatcher::new();
        assert_eq!(dispatcher.fallback_pc, 0);
    }

    #[test]
    fn test_dispatcher_with_fallback() {
        let dispatcher = ResumeDispatcher::with_fallback(999);
        assert_eq!(dispatcher.fallback_pc, 999);
    }

    #[test]
    fn test_dispatcher_no_table() {
        let dispatcher = ResumeDispatcher::with_fallback(42);
        let pc = dispatcher.dispatch(None, 0);
        assert_eq!(pc, 42);
    }

    #[test]
    fn test_dispatcher_with_table() {
        use crate::ops::generator::resume_cache::ResumeTable;

        let mut table = ResumeTable::new();
        table.insert(0, 100);
        table.insert(1, 200);
        table.insert(2, 300);

        let dispatcher = ResumeDispatcher::new();

        assert_eq!(dispatcher.dispatch(Some(&table), 0), 100);
        assert_eq!(dispatcher.dispatch(Some(&table), 1), 200);
        assert_eq!(dispatcher.dispatch(Some(&table), 2), 300);
    }

    #[test]
    fn test_dispatcher_invalid_index_uses_fallback() {
        use crate::ops::generator::resume_cache::ResumeTable;

        let table = ResumeTable::new();
        let dispatcher = ResumeDispatcher::with_fallback(999);

        assert_eq!(dispatcher.dispatch(Some(&table), 100), 999);
    }

    #[test]
    fn test_dispatcher_direct() {
        let dispatcher = ResumeDispatcher::new();
        assert_eq!(dispatcher.dispatch_direct(42), 42);
        assert_eq!(dispatcher.dispatch_direct(0), 0);
    }

    // ════════════════════════════════════════════════════════════════════════
    // restore_generator_frame Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_restore_empty() {
        let suspended = SuspendResult::new(0, 0);
        let mut registers = vec![Value::int(99).unwrap(); 4];

        restore_generator_frame(&suspended, &mut registers);

        // Nothing restored, values unchanged
        for reg in &registers {
            assert_eq!(reg.as_int(), Some(99));
        }
    }

    #[test]
    fn test_restore_all() {
        let original = vec![
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ];
        let suspended = capture_generator_frame(&original, 0b111, 0).unwrap();

        let mut restored = vec![Value::none(); 3];
        restore_generator_frame(&suspended, &mut restored);

        assert_eq!(restored[0].as_int(), Some(1));
        assert_eq!(restored[1].as_int(), Some(2));
        assert_eq!(restored[2].as_int(), Some(3));
    }

    #[test]
    fn test_restore_sparse() {
        let original = vec![
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
            Value::int(4).unwrap(),
        ];
        // Only registers 0 and 3 live
        let suspended = capture_generator_frame(&original, 0b1001, 0).unwrap();

        let mut restored = vec![Value::none(); 4];
        restore_generator_frame(&suspended, &mut restored);

        assert_eq!(restored[0].as_int(), Some(1));
        assert!(restored[1].is_none()); // Not live
        assert!(restored[2].is_none()); // Not live
        assert_eq!(restored[3].as_int(), Some(4));
    }

    #[test]
    fn test_restore_preserves_non_live() {
        let original = vec![
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ];
        // Only register 1 live
        let suspended = capture_generator_frame(&original, 0b010, 0).unwrap();

        // Pre-fill with different values
        let mut restored = vec![
            Value::int(100).unwrap(),
            Value::int(200).unwrap(),
            Value::int(300).unwrap(),
        ];
        restore_generator_frame(&suspended, &mut restored);

        assert_eq!(restored[0].as_int(), Some(100)); // Unchanged
        assert_eq!(restored[1].as_int(), Some(2)); // Restored
        assert_eq!(restored[2].as_int(), Some(300)); // Unchanged
    }

    // ════════════════════════════════════════════════════════════════════════
    // dispatch_to_resume_point Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_dispatch_no_table() {
        let result = dispatch_to_resume_point(None, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_dispatch_valid_index() {
        use crate::ops::generator::resume_cache::ResumeTable;

        let mut table = ResumeTable::new();
        table.insert(0, 100);

        let result = dispatch_to_resume_point(Some(&table), 0);
        assert_eq!(result, Some(100));
    }

    #[test]
    fn test_dispatch_invalid_index() {
        use crate::ops::generator::resume_cache::ResumeTable;

        let table = ResumeTable::new();

        let result = dispatch_to_resume_point(Some(&table), 99);
        assert!(result.is_none());
    }

    // ════════════════════════════════════════════════════════════════════════
    // inject_send_value Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_inject_send_value() {
        let mut registers = vec![Value::none(); 4];

        inject_send_value(&mut registers, 2, Value::int(42).unwrap());

        assert!(registers[0].is_none());
        assert!(registers[1].is_none());
        assert_eq!(registers[2].as_int(), Some(42));
        assert!(registers[3].is_none());
    }

    #[test]
    fn test_inject_send_value_first_register() {
        let mut registers = vec![Value::none(); 4];

        inject_send_value(&mut registers, 0, Value::bool(true));

        assert_eq!(registers[0].as_bool(), Some(true));
    }

    #[test]
    fn test_inject_send_value_out_of_bounds() {
        let mut registers = vec![Value::none(); 4];

        // Should not panic, just do nothing
        inject_send_value(&mut registers, 100, Value::int(42).unwrap());

        // All still None
        for reg in &registers {
            assert!(reg.is_none());
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // prepare_throw Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_prepare_throw() {
        let mut registers = vec![Value::none(); 4];

        prepare_throw(&mut registers, 1, Value::int(999).unwrap());

        assert!(registers[0].is_none());
        assert_eq!(registers[1].as_int(), Some(999));
        assert!(registers[2].is_none());
    }

    #[test]
    fn test_prepare_throw_out_of_bounds() {
        let mut registers = vec![Value::none(); 4];

        // Should not panic
        prepare_throw(&mut registers, 100, Value::int(999).unwrap());
    }

    // ════════════════════════════════════════════════════════════════════════
    // ResumeError Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_resume_error_display() {
        assert_eq!(
            ResumeError::NotSuspended.to_string(),
            "Generator is not suspended"
        );
        assert_eq!(
            ResumeError::GeneratorExhausted.to_string(),
            "Generator has already completed"
        );
        assert_eq!(
            ResumeError::InvalidResumeIndex(42).to_string(),
            "Invalid resume index: 42"
        );
        assert_eq!(
            ResumeError::CantSendNonNone.to_string(),
            "Can't send non-None value to a just-started generator"
        );
        assert_eq!(
            ResumeError::NoSavedState.to_string(),
            "No saved state to restore"
        );
    }

    // ════════════════════════════════════════════════════════════════════════
    // Round-trip Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_capture_restore_round_trip() {
        let original = vec![
            Value::int(10).unwrap(),
            Value::int(20).unwrap(),
            Value::int(30).unwrap(),
            Value::int(40).unwrap(),
            Value::none(),
            Value::bool(true),
        ];
        let liveness = 0b111011; // All except register 2

        let suspended = capture_generator_frame(&original, liveness, 42).unwrap();
        assert_eq!(suspended.resume_index, 42);

        let mut restored = vec![Value::none(); 6];
        restore_generator_frame(&suspended, &mut restored);

        assert_eq!(restored[0].as_int(), Some(10));
        assert_eq!(restored[1].as_int(), Some(20));
        assert!(restored[2].is_none()); // Not live
        assert_eq!(restored[3].as_int(), Some(40));
        assert!(restored[4].is_none());
        assert_eq!(restored[5].as_bool(), Some(true));
    }

    #[test]
    fn test_full_resume_workflow() {
        use crate::ops::generator::resume_cache::ResumeTable;

        // Setup
        let original = vec![
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ];

        // Simulate yield (suspend)
        let liveness = 0b111;
        let resume_index = 1;
        let suspended = capture_generator_frame(&original, liveness, resume_index).unwrap();

        // Build resume table
        let mut table = ResumeTable::new();
        table.insert(0, 100);
        table.insert(1, 200); // Our resume point
        table.insert(2, 300);

        // Simulate resume (restore)
        let mut registers = vec![Value::none(); 3];
        restore_generator_frame(&suspended, &mut registers);

        // Inject send value
        inject_send_value(&mut registers, 0, Value::int(42).unwrap());

        // Dispatch
        let target_pc = dispatch_to_resume_point(Some(&table), resume_index);

        // Verify
        assert_eq!(target_pc, Some(200));
        assert_eq!(registers[0].as_int(), Some(42)); // Send value overwrote
        assert_eq!(registers[1].as_int(), Some(2));
        assert_eq!(registers[2].as_int(), Some(3));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Edge Case Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_restore_to_smaller_array() {
        let original = vec![
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
            Value::int(4).unwrap(),
        ];
        let suspended = capture_generator_frame(&original, 0b1111, 0).unwrap();

        // Restore to smaller array
        let mut restored = vec![Value::none(); 2];
        restore_generator_frame(&suspended, &mut restored);

        // Only first 2 restored
        assert_eq!(restored[0].as_int(), Some(1));
        assert_eq!(restored[1].as_int(), Some(2));
    }

    #[test]
    fn test_restore_to_larger_array() {
        let original = vec![Value::int(1).unwrap(), Value::int(2).unwrap()];
        let suspended = capture_generator_frame(&original, 0b11, 0).unwrap();

        // Restore to larger array
        let mut restored = vec![Value::int(99).unwrap(); 4];
        restore_generator_frame(&suspended, &mut restored);

        assert_eq!(restored[0].as_int(), Some(1));
        assert_eq!(restored[1].as_int(), Some(2));
        assert_eq!(restored[2].as_int(), Some(99)); // Unchanged
        assert_eq!(restored[3].as_int(), Some(99)); // Unchanged
    }
}
