//! Generator execution context for VM integration.
//!
//! This module provides the `GeneratorContext` which tracks the currently
//! active generator during execution, enabling proper yield/resume semantics.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        GeneratorContext                                  │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │  Execution State:                                                        │
//! │  ┌─────────────────────────────────────────────────────────────────┐    │
//! │  │ active_generator: Option<NonNull<GeneratorObject>>              │    │
//! │  │ nesting_depth: u32  (for nested generator calls)                │    │
//! │  │ state: GeneratorExecutionState                                   │    │
//! │  └─────────────────────────────────────────────────────────────────┘    │
//! │                                                                          │
//! │  Statistics:                                                            │
//! │  ┌─────────────────────────────────────────────────────────────────┐    │
//! │  │ total_yields: u64                                               │    │
//! │  │ total_resumes: u64                                              │    │
//! │  │ total_closes: u64                                               │    │
//! │  └─────────────────────────────────────────────────────────────────┘    │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Safety
//!
//! The `GeneratorContext` uses `NonNull` pointers to generator objects.
//! The caller is responsible for ensuring the generator remains valid
//! for the duration of the execution.
//!
//! # Thread Safety
//!
//! `GeneratorContext` is NOT thread-safe. Each VM owns its own context.

use crate::stdlib::generators::GeneratorObject;
use std::ptr::NonNull;

// =============================================================================
// Generator Execution State
// =============================================================================

/// The current state of generator execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratorExecutionState {
    /// No generator is currently executing.
    Idle,
    /// Generator is executing normally.
    Running,
    /// Generator is suspended at a yield point.
    Suspended,
    /// Generator is being closed (running finally blocks).
    Closing,
    /// Generator threw an exception.
    Throwing,
    /// Generator has completed (returned or exhausted).
    Completed,
}

impl GeneratorExecutionState {
    /// Check if execution can continue.
    #[inline]
    pub fn can_continue(&self) -> bool {
        matches!(self, Self::Running | Self::Throwing)
    }

    /// Check if generator is active (not idle or completed).
    #[inline]
    pub fn is_active(&self) -> bool {
        !matches!(self, Self::Idle | Self::Completed)
    }

    /// Check if generator can be resumed.
    #[inline]
    pub fn can_resume(&self) -> bool {
        matches!(self, Self::Suspended)
    }

    /// Check if generator can be closed.
    #[inline]
    pub fn can_close(&self) -> bool {
        matches!(self, Self::Suspended | Self::Running)
    }
}

impl Default for GeneratorExecutionState {
    fn default() -> Self {
        Self::Idle
    }
}

// =============================================================================
// Generator Context Statistics
// =============================================================================

/// Statistics for generator execution.
#[derive(Debug, Clone, Copy, Default)]
pub struct GeneratorContextStats {
    /// Total number of yields across all generators.
    pub total_yields: u64,
    /// Total number of resumes across all generators.
    pub total_resumes: u64,
    /// Total number of generator closes.
    pub total_closes: u64,
    /// Total number of throw operations.
    pub total_throws: u64,
    /// Maximum nesting depth reached.
    pub max_nesting_depth: u32,
    /// Number of generator activations.
    pub activations: u64,
}

// =============================================================================
// Generator Context
// =============================================================================

/// Context for tracking the currently executing generator.
///
/// This is stored in the VirtualMachine and updated during generator
/// yield/resume operations.
pub struct GeneratorContext {
    /// Currently active generator (if any).
    active_generator: Option<NonNull<GeneratorObject>>,
    /// Current execution state.
    state: GeneratorExecutionState,
    /// Nesting depth for nested generator calls.
    nesting_depth: u32,
    /// Saved value for send operations.
    send_value: Option<prism_core::Value>,
    /// Exception being thrown into generator.
    thrown_exception: Option<prism_core::Value>,
    /// Statistics.
    stats: GeneratorContextStats,
}

impl GeneratorContext {
    /// Create a new idle generator context.
    #[inline]
    pub fn new() -> Self {
        Self {
            active_generator: None,
            state: GeneratorExecutionState::Idle,
            nesting_depth: 0,
            send_value: None,
            thrown_exception: None,
            stats: GeneratorContextStats::default(),
        }
    }

    /// Check if any generator is currently active.
    #[inline(always)]
    pub fn is_active(&self) -> bool {
        self.active_generator.is_some()
    }

    /// Get the currently active generator.
    #[inline(always)]
    pub fn current_generator(&self) -> Option<NonNull<GeneratorObject>> {
        self.active_generator
    }

    /// Get the current execution state.
    #[inline(always)]
    pub fn state(&self) -> GeneratorExecutionState {
        self.state
    }

    /// Get the current nesting depth.
    #[inline(always)]
    pub fn nesting_depth(&self) -> u32 {
        self.nesting_depth
    }

    /// Enter a generator for execution.
    ///
    /// # Safety
    ///
    /// The caller must ensure the generator pointer remains valid until
    /// `exit()` is called.
    #[inline]
    pub fn enter(&mut self, generator: NonNull<GeneratorObject>) {
        // If there's already an active generator, we're entering a nested call
        if self.active_generator.is_some() {
            self.nesting_depth += 1;
            if self.nesting_depth > self.stats.max_nesting_depth {
                self.stats.max_nesting_depth = self.nesting_depth;
            }
        }

        self.active_generator = Some(generator);
        self.state = GeneratorExecutionState::Running;
        self.stats.activations += 1;
    }

    /// Exit the current generator.
    ///
    /// Returns the previous generator if this was a nested call.
    #[inline]
    pub fn exit(&mut self) {
        if self.nesting_depth > 0 {
            self.nesting_depth -= 1;
            // Parent generator is still active
            self.state = GeneratorExecutionState::Running;
        } else {
            self.active_generator = None;
            self.state = GeneratorExecutionState::Idle;
        }

        self.send_value = None;
        self.thrown_exception = None;
    }

    /// Suspend the current generator at a yield point.
    #[inline]
    pub fn suspend(&mut self) {
        debug_assert!(self.is_active(), "Cannot suspend without active generator");
        self.state = GeneratorExecutionState::Suspended;
        self.stats.total_yields += 1;
    }

    /// Resume a suspended generator.
    ///
    /// # Arguments
    ///
    /// * `send_value` - Optional value to send into the generator.
    #[inline]
    pub fn resume(&mut self, send_value: Option<prism_core::Value>) {
        debug_assert!(
            self.state.can_resume(),
            "Cannot resume generator in state {:?}",
            self.state
        );
        self.send_value = send_value;
        self.state = GeneratorExecutionState::Running;
        self.stats.total_resumes += 1;
    }

    /// Begin closing the generator.
    #[inline]
    pub fn begin_close(&mut self) {
        debug_assert!(self.is_active(), "Cannot close without active generator");
        self.state = GeneratorExecutionState::Closing;
        self.stats.total_closes += 1;
    }

    /// Throw an exception into the generator.
    #[inline]
    pub fn throw(&mut self, exception: prism_core::Value) {
        debug_assert!(self.is_active(), "Cannot throw without active generator");
        self.thrown_exception = Some(exception);
        self.state = GeneratorExecutionState::Throwing;
        self.stats.total_throws += 1;
    }

    /// Mark the generator as completed.
    #[inline]
    pub fn complete(&mut self) {
        self.state = GeneratorExecutionState::Completed;
    }

    /// Take the send value (if any).
    ///
    /// Returns the value sent via `generator.send()` and clears it.
    #[inline]
    pub fn take_send_value(&mut self) -> Option<prism_core::Value> {
        self.send_value.take()
    }

    /// Take the thrown exception (if any).
    #[inline]
    pub fn take_thrown_exception(&mut self) -> Option<prism_core::Value> {
        self.thrown_exception.take()
    }

    /// Get the pending send value without taking it.
    #[inline]
    pub fn peek_send_value(&self) -> Option<&prism_core::Value> {
        self.send_value.as_ref()
    }

    /// Get the thrown exception without taking it.
    #[inline]
    pub fn peek_thrown_exception(&self) -> Option<&prism_core::Value> {
        self.thrown_exception.as_ref()
    }

    /// Check if there's a pending send value.
    #[inline]
    pub fn has_send_value(&self) -> bool {
        self.send_value.is_some()
    }

    /// Check if there's a pending thrown exception.
    #[inline]
    pub fn has_thrown_exception(&self) -> bool {
        self.thrown_exception.is_some()
    }

    /// Get statistics.
    #[inline]
    pub fn stats(&self) -> GeneratorContextStats {
        self.stats
    }

    /// Reset the context to idle state.
    #[inline]
    pub fn reset(&mut self) {
        self.active_generator = None;
        self.state = GeneratorExecutionState::Idle;
        self.nesting_depth = 0;
        self.send_value = None;
        self.thrown_exception = None;
    }
}

impl Default for GeneratorContext {
    fn default() -> Self {
        Self::new()
    }
}

// Safety: GeneratorContext does not contain any thread-local state
// and the NonNull pointer is just a marker for the active generator.
// Actual synchronization must be handled by the VM.
unsafe impl Send for GeneratorContext {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
