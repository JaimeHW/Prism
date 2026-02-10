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
mod tests {
    use super::*;
    use prism_core::Value;

    // Helper to create a dangling generator pointer for tests
    fn dangling_generator() -> NonNull<GeneratorObject> {
        NonNull::dangling()
    }

    // ════════════════════════════════════════════════════════════════════════
    // GeneratorExecutionState Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_state_default() {
        let state = GeneratorExecutionState::default();
        assert_eq!(state, GeneratorExecutionState::Idle);
    }

    #[test]
    fn test_state_can_continue() {
        assert!(!GeneratorExecutionState::Idle.can_continue());
        assert!(GeneratorExecutionState::Running.can_continue());
        assert!(!GeneratorExecutionState::Suspended.can_continue());
        assert!(!GeneratorExecutionState::Closing.can_continue());
        assert!(GeneratorExecutionState::Throwing.can_continue());
        assert!(!GeneratorExecutionState::Completed.can_continue());
    }

    #[test]
    fn test_state_is_active() {
        assert!(!GeneratorExecutionState::Idle.is_active());
        assert!(GeneratorExecutionState::Running.is_active());
        assert!(GeneratorExecutionState::Suspended.is_active());
        assert!(GeneratorExecutionState::Closing.is_active());
        assert!(GeneratorExecutionState::Throwing.is_active());
        assert!(!GeneratorExecutionState::Completed.is_active());
    }

    #[test]
    fn test_state_can_resume() {
        assert!(!GeneratorExecutionState::Idle.can_resume());
        assert!(!GeneratorExecutionState::Running.can_resume());
        assert!(GeneratorExecutionState::Suspended.can_resume());
        assert!(!GeneratorExecutionState::Closing.can_resume());
        assert!(!GeneratorExecutionState::Throwing.can_resume());
        assert!(!GeneratorExecutionState::Completed.can_resume());
    }

    #[test]
    fn test_state_can_close() {
        assert!(!GeneratorExecutionState::Idle.can_close());
        assert!(GeneratorExecutionState::Running.can_close());
        assert!(GeneratorExecutionState::Suspended.can_close());
        assert!(!GeneratorExecutionState::Closing.can_close());
        assert!(!GeneratorExecutionState::Throwing.can_close());
        assert!(!GeneratorExecutionState::Completed.can_close());
    }

    // ════════════════════════════════════════════════════════════════════════
    // GeneratorContext Construction Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_context_new() {
        let ctx = GeneratorContext::new();
        assert!(!ctx.is_active());
        assert!(ctx.current_generator().is_none());
        assert_eq!(ctx.state(), GeneratorExecutionState::Idle);
        assert_eq!(ctx.nesting_depth(), 0);
    }

    #[test]
    fn test_context_default() {
        let ctx = GeneratorContext::default();
        assert!(!ctx.is_active());
    }

    // ════════════════════════════════════════════════════════════════════════
    // GeneratorContext Enter/Exit Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_context_enter() {
        let mut ctx = GeneratorContext::new();
        let generator = dangling_generator();

        ctx.enter(generator);

        assert!(ctx.is_active());
        assert!(ctx.current_generator().is_some());
        assert_eq!(ctx.state(), GeneratorExecutionState::Running);
    }

    #[test]
    fn test_context_exit() {
        let mut ctx = GeneratorContext::new();
        let generator = dangling_generator();

        ctx.enter(generator);
        ctx.exit();

        assert!(!ctx.is_active());
        assert!(ctx.current_generator().is_none());
        assert_eq!(ctx.state(), GeneratorExecutionState::Idle);
    }

    #[test]
    fn test_context_enter_exit_preserves_stats() {
        let mut ctx = GeneratorContext::new();
        let generator = dangling_generator();

        ctx.enter(generator);
        ctx.exit();

        let stats = ctx.stats();
        assert_eq!(stats.activations, 1);
    }

    // ════════════════════════════════════════════════════════════════════════
    // GeneratorContext Suspend/Resume Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_context_suspend() {
        let mut ctx = GeneratorContext::new();
        let generator = dangling_generator();

        ctx.enter(generator);
        ctx.suspend();

        assert_eq!(ctx.state(), GeneratorExecutionState::Suspended);

        let stats = ctx.stats();
        assert_eq!(stats.total_yields, 1);
    }

    #[test]
    fn test_context_resume_without_value() {
        let mut ctx = GeneratorContext::new();
        let generator = dangling_generator();

        ctx.enter(generator);
        ctx.suspend();
        ctx.resume(None);

        assert_eq!(ctx.state(), GeneratorExecutionState::Running);
        assert!(ctx.take_send_value().is_none());

        let stats = ctx.stats();
        assert_eq!(stats.total_resumes, 1);
    }

    #[test]
    fn test_context_resume_with_value() {
        let mut ctx = GeneratorContext::new();
        let generator = dangling_generator();

        ctx.enter(generator);
        ctx.suspend();
        ctx.resume(Some(Value::int(42).unwrap()));

        assert_eq!(ctx.state(), GeneratorExecutionState::Running);
        assert!(ctx.has_send_value());

        let sent = ctx.take_send_value();
        assert_eq!(sent.unwrap().as_int(), Some(42));

        // Value should be taken
        assert!(!ctx.has_send_value());
    }

    #[test]
    fn test_context_resume_clears_on_exit() {
        let mut ctx = GeneratorContext::new();
        let generator = dangling_generator();

        ctx.enter(generator);
        ctx.suspend();
        ctx.resume(Some(Value::int(42).unwrap()));
        ctx.exit();

        assert!(!ctx.has_send_value());
    }

    // ════════════════════════════════════════════════════════════════════════
    // GeneratorContext Close Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_context_begin_close() {
        let mut ctx = GeneratorContext::new();
        let generator = dangling_generator();

        ctx.enter(generator);
        ctx.begin_close();

        assert_eq!(ctx.state(), GeneratorExecutionState::Closing);

        let stats = ctx.stats();
        assert_eq!(stats.total_closes, 1);
    }

    #[test]
    fn test_context_complete() {
        let mut ctx = GeneratorContext::new();
        let generator = dangling_generator();

        ctx.enter(generator);
        ctx.complete();

        assert_eq!(ctx.state(), GeneratorExecutionState::Completed);
        assert!(ctx.is_active()); // Still has generator ref
    }

    // ════════════════════════════════════════════════════════════════════════
    // GeneratorContext Throw Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_context_throw() {
        let mut ctx = GeneratorContext::new();
        let generator = dangling_generator();

        ctx.enter(generator);
        ctx.throw(Value::int(999).unwrap());

        assert_eq!(ctx.state(), GeneratorExecutionState::Throwing);
        assert!(ctx.has_thrown_exception());

        let stats = ctx.stats();
        assert_eq!(stats.total_throws, 1);
    }

    #[test]
    fn test_context_take_thrown_exception() {
        let mut ctx = GeneratorContext::new();
        let generator = dangling_generator();

        ctx.enter(generator);
        ctx.throw(Value::int(999).unwrap());

        let exc = ctx.take_thrown_exception();
        assert_eq!(exc.unwrap().as_int(), Some(999));
        assert!(!ctx.has_thrown_exception());
    }

    #[test]
    fn test_context_throw_clears_on_exit() {
        let mut ctx = GeneratorContext::new();
        let generator = dangling_generator();

        ctx.enter(generator);
        ctx.throw(Value::int(999).unwrap());
        ctx.exit();

        assert!(!ctx.has_thrown_exception());
    }

    // ════════════════════════════════════════════════════════════════════════
    // GeneratorContext Nesting Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_context_nesting() {
        let mut ctx = GeneratorContext::new();
        let gen1 = dangling_generator();
        let gen2 = dangling_generator();

        ctx.enter(gen1);
        assert_eq!(ctx.nesting_depth(), 0);

        ctx.enter(gen2);
        assert_eq!(ctx.nesting_depth(), 1);

        ctx.exit();
        assert_eq!(ctx.nesting_depth(), 0);
        assert!(ctx.is_active()); // gen1 still active

        ctx.exit();
        assert!(!ctx.is_active());
    }

    #[test]
    fn test_context_deep_nesting() {
        let mut ctx = GeneratorContext::new();

        for i in 0..10 {
            ctx.enter(dangling_generator());
            assert_eq!(ctx.nesting_depth(), i.min(9)); // First enter doesn't increment
        }

        let stats = ctx.stats();
        assert_eq!(stats.max_nesting_depth, 9);
    }

    // ════════════════════════════════════════════════════════════════════════
    // GeneratorContext Reset Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_context_reset() {
        let mut ctx = GeneratorContext::new();
        let generator = dangling_generator();

        ctx.enter(generator);
        ctx.suspend();
        ctx.resume(Some(Value::int(42).unwrap()));
        ctx.throw(Value::int(999).unwrap());

        ctx.reset();

        assert!(!ctx.is_active());
        assert_eq!(ctx.state(), GeneratorExecutionState::Idle);
        assert_eq!(ctx.nesting_depth(), 0);
        assert!(!ctx.has_send_value());
        assert!(!ctx.has_thrown_exception());
    }

    // ════════════════════════════════════════════════════════════════════════
    // GeneratorContext Edge Case Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_context_multiple_yield_resume_cycles() {
        let mut ctx = GeneratorContext::new();
        let generator = dangling_generator();

        ctx.enter(generator);

        for i in 0..5 {
            ctx.suspend();
            assert_eq!(ctx.state(), GeneratorExecutionState::Suspended);

            ctx.resume(Some(Value::int(i).unwrap()));
            assert_eq!(ctx.state(), GeneratorExecutionState::Running);

            let val = ctx.take_send_value();
            assert_eq!(val.unwrap().as_int(), Some(i));
        }

        let stats = ctx.stats();
        assert_eq!(stats.total_yields, 5);
        assert_eq!(stats.total_resumes, 5);
    }

    #[test]
    fn test_context_peek_does_not_consume() {
        let mut ctx = GeneratorContext::new();
        let generator = dangling_generator();

        ctx.enter(generator);
        ctx.suspend();
        ctx.resume(Some(Value::int(42).unwrap()));

        // Peek should not consume
        assert_eq!(ctx.peek_send_value().unwrap().as_int(), Some(42));
        assert_eq!(ctx.peek_send_value().unwrap().as_int(), Some(42));

        // Take consumes
        assert_eq!(ctx.take_send_value().unwrap().as_int(), Some(42));
        assert!(ctx.peek_send_value().is_none());
    }

    #[test]
    fn test_context_stats_accumulate() {
        let mut ctx = GeneratorContext::new();

        for _ in 0..3 {
            ctx.enter(dangling_generator());
            ctx.suspend();
            ctx.resume(None);
            ctx.begin_close();
            ctx.exit();
        }

        let stats = ctx.stats();
        assert_eq!(stats.activations, 3);
        assert_eq!(stats.total_yields, 3);
        assert_eq!(stats.total_resumes, 3);
        assert_eq!(stats.total_closes, 3);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Size and Performance Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_context_size() {
        let size = std::mem::size_of::<GeneratorContext>();
        // Should be reasonably compact
        assert!(size <= 128, "GeneratorContext too large: {} bytes", size);
    }

    #[test]
    fn test_state_size() {
        let size = std::mem::size_of::<GeneratorExecutionState>();
        assert_eq!(size, 1, "State should be 1 byte enum");
    }

    #[test]
    fn test_stats_size() {
        let size = std::mem::size_of::<GeneratorContextStats>();
        assert!(size <= 48, "Stats too large: {} bytes", size);
    }
}
