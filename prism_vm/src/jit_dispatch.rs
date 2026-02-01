//! JIT Dispatch - Zero-overhead dispatch to compiled code.
//!
//! This module provides the critical hot path for JIT execution dispatch.
//! It is designed for maximum performance with:
//!
//! - **Zero allocations** on the hot path
//! - **Inline lookup** with O(1) code cache access
//! - **Cold path separation** for deoptimization handling
//! - **Branch prediction friendly** - common case (JIT hit) is predictable
//!
//! # Architecture
//!
//! ```text
//!  ┌─────────────────────────────────────────────────────────────────┐
//!  │                        dispatch_call()                          │
//!  │  ┌──────────┐                                                   │
//!  │  │ Profile  │                                                   │
//!  │  │  Call    │                                                   │
//!  │  └────┬─────┘                                                   │
//!  │       │                                                         │
//!  │       ▼                                                         │
//!  │  ┌──────────┐     ┌─────────────┐                              │
//!  │  │ JIT      │────▶│  Code Cache │◀─── O(1) hash lookup         │
//!  │  │ Enabled? │     │   Lookup    │                              │
//!  │  └────┬─────┘     └──────┬──────┘                              │
//!  │       │                  │                                      │
//!  │       │ No         Hit   │   Miss                              │
//!  │       │           ┌──────┴──────┐                              │
//!  │       ▼           ▼             ▼                              │
//!  │  ┌──────────┐ ┌──────────┐ ┌──────────┐                        │
//!  │  │Interpreter│ │ Execute  │ │ Tier-Up  │                        │
//!  │  │  Frame   │ │   JIT    │ │  Check   │                        │
//!  │  │  Push    │ │  Code    │ │          │                        │
//!  │  └──────────┘ └────┬─────┘ └────┬─────┘                        │
//!  │                    │            │                              │
//!  │             ┌──────┴──────┐     │                              │
//!  │             ▼      ▼      ▼     ▼                              │
//!  │         Return  Deopt  Exception  Interpreter                  │
//!  └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance Characteristics
//!
//! | Path | Cost | Notes |
//! |------|------|-------|
//! | JIT Disabled | 1 branch | `Option::is_some()` check |
//! | Cache Miss | ~50ns | Hash lookup + miss recording |
//! | Cache Hit | ~20ns | Hash lookup + function call setup |
//! | JIT Execution | Variable | Depends on compiled code |
//! | Deopt | ~100ns | Frame restore + interpreter resume |

use std::sync::Arc;

use prism_compiler::bytecode::CodeObject;
use prism_core::Value;

use crate::error::RuntimeError;
use crate::frame::Frame;
use crate::jit_context::JitContext;
use crate::jit_executor::{DeoptReason, ExecutionResult};

// =============================================================================
// Dispatch Result
// =============================================================================

/// Result of a JIT dispatch attempt.
///
/// This enum captures all possible outcomes from attempting to execute
/// compiled code, allowing the VM to take appropriate action.
#[derive(Debug)]
pub enum DispatchResult {
    /// JIT execution completed successfully with a return value.
    ///
    /// The value should be propagated to the caller as the function result.
    Executed(Value),

    /// Deoptimization occurred - resume interpreter at bytecode offset.
    ///
    /// The frame has been restored to the deopt point and the interpreter
    /// should continue execution from `bc_offset`.
    Deopt {
        /// Bytecode offset to resume at.
        bc_offset: u32,
        /// Reason for deoptimization (for statistics).
        reason: DeoptReason,
    },

    /// Exception occurred during JIT execution.
    ///
    /// The error should be propagated up the call stack.
    Exception(RuntimeError),

    /// Tail call was requested.
    ///
    /// The caller should set up the new call without pushing a frame.
    TailCall {
        /// Target function code ID.
        target_code_id: u64,
        /// Number of arguments already on stack.
        arg_count: u8,
    },

    /// No compiled code available - fall through to interpreter.
    ///
    /// This is the "miss" case - the interpreter should handle this call.
    NotCompiled,
}

impl DispatchResult {
    /// Check if this result indicates JIT execution happened.
    #[inline]
    pub fn was_executed(&self) -> bool {
        !matches!(self, DispatchResult::NotCompiled)
    }

    /// Check if this result requires interpreter fallback.
    #[inline]
    pub fn needs_interpreter(&self) -> bool {
        matches!(
            self,
            DispatchResult::NotCompiled | DispatchResult::Deopt { .. }
        )
    }
}

// =============================================================================
// Dispatch Statistics
// =============================================================================

/// Statistics for JIT dispatch (for profiling and tuning).
#[derive(Debug, Default, Clone)]
pub struct DispatchStats {
    /// Number of dispatch attempts.
    pub attempts: u64,
    /// Number of cache hits (JIT executed).
    pub hits: u64,
    /// Number of cache misses (interpreter fallback).
    pub misses: u64,
    /// Number of deoptimizations.
    pub deopts: u64,
    /// Number of exceptions during JIT.
    pub exceptions: u64,
    /// Number of tail calls.
    pub tail_calls: u64,
}

impl DispatchStats {
    /// Calculate hit rate (0.0 to 1.0).
    #[inline]
    pub fn hit_rate(&self) -> f64 {
        if self.attempts == 0 {
            0.0
        } else {
            self.hits as f64 / self.attempts as f64
        }
    }

    /// Calculate deopt rate (0.0 to 1.0).
    #[inline]
    pub fn deopt_rate(&self) -> f64 {
        if self.hits == 0 {
            0.0
        } else {
            self.deopts as f64 / self.hits as f64
        }
    }
}

// =============================================================================
// JIT Dispatch - Hot Path
// =============================================================================

/// Try to dispatch a call to JIT-compiled code.
///
/// This is the primary entry point for JIT execution. It handles:
/// 1. Looking up compiled code in the cache
/// 2. Setting up frame state for JIT execution
/// 3. Invoking compiled code
/// 4. Processing the result
///
/// # Arguments
///
/// * `jit` - The JIT context (must be Some)
/// * `code` - The code object to execute
/// * `frame` - A temporary frame for JIT execution
///
/// # Returns
///
/// A `DispatchResult` indicating what happened:
/// - `Executed(value)` - JIT ran and returned a value
/// - `Deopt` - JIT ran but deoptimized
/// - `Exception` - JIT ran but threw an exception
/// - `NotCompiled` - No compiled code, use interpreter
///
/// # Performance
///
/// This function is marked `#[inline]` because the lookup is the
/// critical hot path. The actual execution and result processing
/// are cold paths that will not be inlined.
#[inline]
pub fn try_dispatch(
    jit: &mut JitContext,
    code: &Arc<CodeObject>,
    frame: &mut Frame,
) -> DispatchResult {
    // Get code ID from Arc pointer (stable across calls)
    let code_id = Arc::as_ptr(code) as u64;

    // Hot path: lookup compiled code
    // This is O(1) hash lookup, cache-aligned for performance
    let _entry = match jit.lookup(code_id) {
        Some(entry) => entry,
        None => {
            // Record miss for statistics
            jit.record_miss();
            return DispatchResult::NotCompiled;
        }
    };

    // Execute compiled code
    // Note: try_execute handles frame setup and result decoding
    match jit.try_execute(code_id, frame) {
        Some(result) => {
            // Process execution result (cold path)
            process_execution_result(jit, code_id, result)
        }
        None => {
            // Shouldn't happen if lookup succeeded, but handle gracefully
            jit.record_miss();
            DispatchResult::NotCompiled
        }
    }
}

/// Process the result of JIT execution.
///
/// This is a cold path - separated from the hot dispatch path
/// to improve branch prediction on the common case.
///
/// # Arguments
///
/// * `jit` - The JIT context
/// * `code_id` - The code object ID (for deopt tracking)
/// * `result` - The execution result from JIT
///
/// # Returns
///
/// A `DispatchResult` that can be handled by the VM.
#[inline(never)] // Cold path - don't inline
fn process_execution_result(
    jit: &mut JitContext,
    code_id: u64,
    result: ExecutionResult,
) -> DispatchResult {
    match result {
        ExecutionResult::Return(value) => {
            // Success - return the value
            DispatchResult::Executed(value)
        }

        ExecutionResult::Deopt { bc_offset, reason } => {
            // Handle deoptimization
            jit.handle_deopt(code_id, reason);
            DispatchResult::Deopt { bc_offset, reason }
        }

        ExecutionResult::Exception(err) => {
            // Propagate exception
            DispatchResult::Exception(err)
        }

        ExecutionResult::TailCall { target, arg_count } => {
            // Tail call optimization
            DispatchResult::TailCall {
                target_code_id: target,
                arg_count,
            }
        }
    }
}

// =============================================================================
// Tier-Up Integration
// =============================================================================

/// Check if a function should be compiled and trigger compilation if needed.
///
/// This is called before attempting JIT dispatch to ensure hot functions
/// get compiled. It's separated from dispatch to allow the profiler
/// decision to happen without the lookup overhead.
///
/// # Arguments
///
/// * `jit` - The JIT context
/// * `code` - The code object to potentially compile
/// * `call_count` - Current call count for this function
///
/// # Returns
///
/// `true` if compilation was triggered (code may now be available),
/// `false` if no compilation was needed.
#[inline(never)] // Cold path
pub fn maybe_tier_up(
    jit: &mut JitContext,
    code: &Arc<CodeObject>,
    tier_decision: crate::profiler::TierUpDecision,
) -> bool {
    use crate::profiler::TierUpDecision;

    match tier_decision {
        TierUpDecision::None => false,
        TierUpDecision::Tier1 | TierUpDecision::Tier2 => jit.handle_tier_up(code, tier_decision),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_result_was_executed() {
        assert!(DispatchResult::Executed(Value::none()).was_executed());
        assert!(
            DispatchResult::Deopt {
                bc_offset: 0,
                reason: DeoptReason::TypeGuard
            }
            .was_executed()
        );
        assert!(DispatchResult::Exception(RuntimeError::internal("test")).was_executed());
        assert!(!DispatchResult::NotCompiled.was_executed());
    }

    #[test]
    fn test_dispatch_result_needs_interpreter() {
        assert!(!DispatchResult::Executed(Value::none()).needs_interpreter());
        assert!(
            DispatchResult::Deopt {
                bc_offset: 0,
                reason: DeoptReason::TypeGuard
            }
            .needs_interpreter()
        );
        assert!(!DispatchResult::Exception(RuntimeError::internal("test")).needs_interpreter());
        assert!(DispatchResult::NotCompiled.needs_interpreter());
    }

    #[test]
    fn test_dispatch_stats_default() {
        let stats = DispatchStats::default();
        assert_eq!(stats.attempts, 0);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.hit_rate(), 0.0);
        assert_eq!(stats.deopt_rate(), 0.0);
    }

    #[test]
    fn test_dispatch_stats_rates() {
        let stats = DispatchStats {
            attempts: 100,
            hits: 80,
            misses: 20,
            deopts: 8,
            exceptions: 0,
            tail_calls: 0,
        };
        assert!((stats.hit_rate() - 0.8).abs() < 0.001);
        assert!((stats.deopt_rate() - 0.1).abs() < 0.001);
    }
}
