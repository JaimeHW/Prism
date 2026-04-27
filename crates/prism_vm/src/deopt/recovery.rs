//! Deoptimization Recovery.
//!
//! Reconstructs interpreter frames from JIT state using deopt deltas.
//! Handles the transition back to interpreter execution.

use super::state::{DeoptDelta, DeoptReason, DeoptState};
use prism_core::Value;

// =============================================================================
// Recovery Result
// =============================================================================

/// Result of deoptimization recovery.
#[derive(Debug, Clone)]
pub enum RecoveryResult {
    /// Successfully recovered, resume at bytecode offset.
    Resume {
        /// Bytecode offset to resume at.
        bc_offset: u32,
        /// Whether to recompile this function.
        should_recompile: bool,
    },
    /// Recovery failed, propagate error.
    Error(RecoveryError),
}

/// Errors that can occur during recovery.
#[derive(Debug, Clone)]
pub enum RecoveryError {
    /// Invalid bytecode offset.
    InvalidOffset(u32),
    /// Frame reconstruction failed.
    FrameCorruption,
    /// Stack overflow during recovery.
    StackOverflow,
}

impl std::fmt::Display for RecoveryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOffset(offset) => write!(f, "Invalid bytecode offset: {}", offset),
            Self::FrameCorruption => write!(f, "Frame corruption during recovery"),
            Self::StackOverflow => write!(f, "Stack overflow during recovery"),
        }
    }
}

impl std::error::Error for RecoveryError {}

// =============================================================================
// Deopt Recovery
// =============================================================================

/// Handles deoptimization recovery.
///
/// Reconstructs interpreter state from JIT state using the deopt delta.
#[derive(Debug, Default)]
pub struct DeoptRecovery {
    /// Recompilation threshold - number of deopts before recompile.
    recompile_threshold: u32,
    /// Patch threshold - number of deopts before patching guard.
    patch_threshold: u32,
}

impl DeoptRecovery {
    /// Create a new recovery handler.
    #[inline]
    pub fn new() -> Self {
        Self {
            recompile_threshold: 10,
            patch_threshold: 100,
        }
    }

    /// Create with custom thresholds.
    #[inline]
    pub fn with_thresholds(recompile: u32, patch: u32) -> Self {
        Self {
            recompile_threshold: recompile,
            patch_threshold: patch,
        }
    }

    /// Recover from a deoptimization.
    ///
    /// # Arguments
    /// * `state` - The captured deopt state
    /// * `jit_registers` - Snapshot of JIT register values
    /// * `frame_registers` - Target interpreter frame registers (output)
    ///
    /// # Returns
    /// Recovery result with resume information.
    pub fn recover(
        &self,
        state: &DeoptState,
        jit_registers: &[Value],
        frame_registers: &mut [Value],
    ) -> RecoveryResult {
        // Apply delta to reconstruct full register state
        self.apply_delta(&state.delta, jit_registers, frame_registers);

        // Determine if recompilation is warranted
        let should_recompile = state.reason.triggers_recompile();

        RecoveryResult::Resume {
            bc_offset: state.bc_offset,
            should_recompile,
        }
    }

    /// Apply delta to reconstruct register state.
    fn apply_delta(
        &self,
        delta: &DeoptDelta,
        jit_registers: &[Value],
        frame_registers: &mut [Value],
    ) {
        // Start with JIT register values
        let copy_len = jit_registers.len().min(frame_registers.len());
        frame_registers[..copy_len].copy_from_slice(&jit_registers[..copy_len]);

        // Apply delta overrides
        for entry in delta.iter() {
            if (entry.slot as usize) < frame_registers.len() {
                frame_registers[entry.slot as usize] = entry.value;
            }
        }
    }

    /// Materialize a value from JIT representation.
    ///
    /// Some values may be in machine-specific formats and need conversion.
    #[inline]
    pub fn materialize_value(&self, raw_bits: u64, type_hint: ValueTypeHint) -> Value {
        match type_hint {
            ValueTypeHint::Int => Value::int_unchecked(raw_bits as i64),
            ValueTypeHint::Float => Value::from(f64::from_bits(raw_bits)),
            ValueTypeHint::Bool => Value::from(raw_bits != 0),
            ValueTypeHint::Object => {
                // Object pointer - needs to be wrapped
                // This is a simplified version; real impl would handle GC roots
                Value::none()
            }
            ValueTypeHint::Unknown => {
                // Assume tagged value format
                Value::from_bits(raw_bits)
            }
        }
    }

    /// Check if guard should be patched based on deopt count.
    #[inline]
    pub fn should_patch_guard(&self, deopt_count: u32, reason: DeoptReason) -> bool {
        // Always patch for certain reasons
        if reason.should_patch_guard() {
            return true;
        }

        // Patch after threshold
        deopt_count >= self.patch_threshold
    }

    /// Check if function should be recompiled.
    #[inline]
    pub fn should_recompile(&self, deopt_count: u32, reason: DeoptReason) -> bool {
        if !reason.triggers_recompile() {
            return false;
        }
        deopt_count >= self.recompile_threshold
    }
}

// =============================================================================
// Value Type Hints
// =============================================================================

/// Type hint for value materialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueTypeHint {
    /// Signed integer.
    Int,
    /// Floating point.
    Float,
    /// Boolean.
    Bool,
    /// Object pointer.
    Object,
    /// Unknown - use tagged format.
    Unknown,
}

// =============================================================================
// Frame Reconstructor
// =============================================================================

/// Low-level frame reconstruction utilities.
pub struct FrameReconstructor;

impl FrameReconstructor {
    /// Reconstruct stack slots from spill info.
    ///
    /// # Arguments
    /// * `stack_base` - Base address of stack frame
    /// * `spill_slots` - Spill slot descriptors
    /// * `output` - Output buffer for reconstructed values
    ///
    /// # Safety
    /// stack_base must point to a valid stack frame.
    #[inline]
    pub unsafe fn reconstruct_from_spills(
        stack_base: *const u8,
        spill_slots: &[(i32, ValueTypeHint)],
        output: &mut [Value],
    ) {
        for (i, &(offset, type_hint)) in spill_slots.iter().enumerate() {
            if i >= output.len() {
                break;
            }

            // SAFETY: Caller guarantees stack_base is valid
            let slot_addr = unsafe { stack_base.offset(offset as isize) as *const u64 };
            let raw_bits = unsafe { *slot_addr };

            output[i] = match type_hint {
                ValueTypeHint::Int => Value::int_unchecked(raw_bits as i64),
                ValueTypeHint::Float => Value::from(f64::from_bits(raw_bits)),
                ValueTypeHint::Bool => Value::from(raw_bits != 0),
                ValueTypeHint::Object | ValueTypeHint::Unknown => Value::from_bits(raw_bits),
            };
        }
    }
}
