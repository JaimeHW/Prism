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
