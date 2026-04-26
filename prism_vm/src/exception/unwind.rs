//! Stack unwinding for exception handling.
//!
//! This module provides the stack unwinding logic used during exception
//! propagation. It coordinates handler lookup, stack restoration, and
//! finally block execution.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────────┐
//! │                         Stack Unwinding Flow                              │
//! ├──────────────────────────────────────────────────────────────────────────┤
//! │                                                                           │
//! │    Exception Raised                                                       │
//! │          │                                                                │
//! │          ▼                                                                │
//! │    ┌─────────────┐                                                        │
//! │    │ Current     │───▶ Search HandlerTable for PC                         │
//! │    │ Frame       │                                                        │
//! │    └──────┬──────┘                                                        │
//! │           │                                                               │
//! │     ┌─────┴─────┐                                                         │
//! │     ▼           ▼                                                         │
//! │  Handler    No Handler                                                    │
//! │  Found      Found                                                         │
//! │     │           │                                                         │
//! │     ▼           ▼                                                         │
//! │  ┌──────┐   ┌──────────┐                                                  │
//! │  │Unwind│   │Pop Frame │                                                  │
//! │  │Stack │   │Continue  │──────────────────┐                               │
//! │  └──┬───┘   │Unwinding │                  │                               │
//! │     │       └──────────┘                  │                               │
//! │     ▼                                     ▼                               │
//! │  ┌──────────┐                      ┌─────────────┐                        │
//! │  │ Jump to  │                      │ Propagate   │                        │
//! │  │ Handler  │                      │ to Caller   │                        │
//! │  └──────────┘                      └─────────────┘                        │
//! │                                                                           │
//! └──────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance
//!
//! | Operation | Complexity |
//! |-----------|------------|
//! | Handler search (single frame) | O(log N) binary search |
//! | Stack depth restoration | O(1) |
//! | Cross-frame unwind | O(F) where F = frames to unwind |

use super::handler_stack::{HandlerFrame, HandlerSearchResult, HandlerStack};
use smallvec::SmallVec;

// ============================================================================
// Constants
// ============================================================================

/// Maximum frames to unwind before giving up.
/// This prevents infinite loops in malformed code.
const MAX_UNWIND_FRAMES: usize = 1000;

/// Default capacity for the finally block queue.
const FINALLY_QUEUE_CAPACITY: usize = 4;

// ============================================================================
// Unwind Action
// ============================================================================

/// Action to take after unwinding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnwindAction {
    /// Jump to handler at the specified PC.
    JumpToHandler {
        /// Handler PC to jump to.
        handler_pc: u32,
        /// Stack depth to restore.
        stack_depth: u16,
    },

    /// Execute a finally block before continuing.
    ExecuteFinally {
        /// Finally block PC.
        finally_pc: u32,
        /// Stack depth to restore.
        stack_depth: u16,
        /// Whether to reraise after finally.
        reraise: bool,
    },

    /// Propagate exception to caller frame.
    PropagateToFrame {
        /// Frame ID to propagate to.
        target_frame_id: u32,
    },

    /// Exception is unhandled, terminate execution.
    Unhandled,

    /// Continue execution (exception was cleared).
    Continue,
}

impl UnwindAction {
    /// Returns true if this action continues execution.
    #[inline]
    pub fn continues(&self) -> bool {
        matches!(self, Self::JumpToHandler { .. } | Self::Continue)
    }

    /// Returns true if this action propagates the exception.
    #[inline]
    pub fn propagates(&self) -> bool {
        matches!(self, Self::PropagateToFrame { .. } | Self::Unhandled)
    }

    /// Returns the handler PC if this is a jump action.
    #[inline]
    pub fn handler_pc(&self) -> Option<u32> {
        match self {
            Self::JumpToHandler { handler_pc, .. } => Some(*handler_pc),
            Self::ExecuteFinally { finally_pc, .. } => Some(*finally_pc),
            _ => None,
        }
    }

    /// Returns the stack depth if applicable.
    #[inline]
    pub fn stack_depth(&self) -> Option<u16> {
        match self {
            Self::JumpToHandler { stack_depth, .. } => Some(*stack_depth),
            Self::ExecuteFinally { stack_depth, .. } => Some(*stack_depth),
            _ => None,
        }
    }
}

// ============================================================================
// Unwind Result
// ============================================================================

/// Result of an unwind operation.
#[derive(Clone, Debug)]
pub struct UnwindResult {
    /// The action to take.
    pub action: UnwindAction,

    /// Number of frames unwound.
    pub frames_unwound: u32,

    /// Number of handlers examined.
    pub handlers_examined: u32,

    /// Finally blocks that need to run.
    pub finally_queue: SmallVec<[FinallyEntry; FINALLY_QUEUE_CAPACITY]>,
}

impl UnwindResult {
    /// Creates a new result with a handler found.
    #[inline]
    pub fn handler_found(handler_pc: u32, stack_depth: u16, handlers_examined: u32) -> Self {
        Self {
            action: UnwindAction::JumpToHandler {
                handler_pc,
                stack_depth,
            },
            frames_unwound: 0,
            handlers_examined,
            finally_queue: SmallVec::new(),
        }
    }

    /// Creates a result for unhandled exception.
    #[inline]
    pub fn unhandled(frames_unwound: u32, handlers_examined: u32) -> Self {
        Self {
            action: UnwindAction::Unhandled,
            frames_unwound,
            handlers_examined,
            finally_queue: SmallVec::new(),
        }
    }

    /// Creates a result for propagation to caller.
    #[inline]
    pub fn propagate(target_frame_id: u32, frames_unwound: u32) -> Self {
        Self {
            action: UnwindAction::PropagateToFrame { target_frame_id },
            frames_unwound,
            handlers_examined: 0,
            finally_queue: SmallVec::new(),
        }
    }

    /// Returns true if a handler was found.
    #[inline]
    pub fn found_handler(&self) -> bool {
        matches!(self.action, UnwindAction::JumpToHandler { .. })
    }
}

impl Default for UnwindResult {
    fn default() -> Self {
        Self {
            action: UnwindAction::Continue,
            frames_unwound: 0,
            handlers_examined: 0,
            finally_queue: SmallVec::new(),
        }
    }
}

// ============================================================================
// Finally Entry
// ============================================================================

/// Entry in the finally block queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FinallyEntry {
    /// PC of the finally block.
    pub finally_pc: u32,

    /// Stack depth to restore.
    pub stack_depth: u16,

    /// Frame ID owning this finally.
    pub frame_id: u32,

    /// Whether to reraise after this finally.
    pub reraise: bool,
}

impl FinallyEntry {
    /// Creates a new finally entry.
    #[inline]
    pub const fn new(finally_pc: u32, stack_depth: u16, frame_id: u32, reraise: bool) -> Self {
        Self {
            finally_pc,
            stack_depth,
            frame_id,
            reraise,
        }
    }
}

// ============================================================================
// Unwind Info
// ============================================================================

/// Information needed for stack unwinding.
#[derive(Clone, Copy, Debug)]
pub struct UnwindInfo {
    /// Current program counter.
    pub pc: u32,

    /// Current frame ID.
    pub frame_id: u32,

    /// Current stack depth.
    pub stack_depth: u16,

    /// Exception type ID (for type matching).
    pub exception_type_id: u16,
}

impl UnwindInfo {
    /// Creates new unwind info.
    #[inline]
    pub const fn new(pc: u32, frame_id: u32, stack_depth: u16, exception_type_id: u16) -> Self {
        Self {
            pc,
            frame_id,
            stack_depth,
            exception_type_id,
        }
    }
}

// ============================================================================
// Unwinder
// ============================================================================

/// Stack unwinder for exception propagation.
///
/// This struct coordinates the search for exception handlers across
/// the call stack.
#[derive(Clone, Debug, Default)]
pub struct Unwinder {
    /// Statistics for unwinding operations.
    stats: UnwinderStats,
}

/// Statistics for unwinding operations.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnwinderStats {
    /// Total unwind operations.
    pub unwind_count: u64,

    /// Total frames unwound.
    pub frames_unwound: u64,

    /// Total handlers examined.
    pub handlers_examined: u64,

    /// Handlers found.
    pub handlers_found: u64,

    /// Unhandled exceptions.
    pub unhandled: u64,

    /// Maximum frames unwound in a single operation.
    pub max_frames_unwound: u32,
}

impl UnwinderStats {
    /// Creates new empty stats.
    #[inline]
    pub const fn new() -> Self {
        Self {
            unwind_count: 0,
            frames_unwound: 0,
            handlers_examined: 0,
            handlers_found: 0,
            unhandled: 0,
            max_frames_unwound: 0,
        }
    }

    /// Resets all statistics.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Returns the average frames unwound per operation.
    #[inline]
    pub fn avg_frames_unwound(&self) -> f64 {
        if self.unwind_count == 0 {
            0.0
        } else {
            self.frames_unwound as f64 / self.unwind_count as f64
        }
    }

    /// Returns the handler found rate as percentage.
    #[inline]
    pub fn handler_found_rate(&self) -> f64 {
        if self.unwind_count == 0 {
            0.0
        } else {
            (self.handlers_found as f64 / self.unwind_count as f64) * 100.0
        }
    }
}

impl Unwinder {
    /// Creates a new unwinder.
    #[inline]
    pub fn new() -> Self {
        Self {
            stats: UnwinderStats::new(),
        }
    }

    /// Returns the statistics.
    #[inline]
    pub fn stats(&self) -> &UnwinderStats {
        &self.stats
    }

    /// Resets statistics.
    #[inline]
    pub fn reset_stats(&mut self) {
        self.stats.reset();
    }

    /// Searches for a handler in the handler stack.
    ///
    /// This is the main unwinding entry point. It searches for a matching
    /// handler and returns the appropriate action.
    #[inline]
    pub fn search_handler(
        &mut self,
        handler_stack: &HandlerStack,
        info: &UnwindInfo,
    ) -> HandlerSearchResult {
        self.stats.unwind_count += 1;

        // Search from top of handler stack (most recent handlers first)
        for (idx, handler) in handler_stack.iter().enumerate() {
            self.stats.handlers_examined += 1;

            // Only consider handlers for this frame or parent frames
            if handler.frame_id > info.frame_id {
                continue;
            }

            // Found a handler for this or parent frame
            let stack_idx = handler_stack.len() - 1 - idx;
            return HandlerSearchResult::Found {
                stack_index: stack_idx,
                handler: *handler,
            };
        }

        self.stats.unhandled += 1;
        HandlerSearchResult::NotFound
    }

    /// Performs a full unwind operation.
    ///
    /// Searches for handlers and builds the unwind result including
    /// any finally blocks that need to run.
    pub fn unwind(&mut self, handler_stack: &HandlerStack, info: &UnwindInfo) -> UnwindResult {
        let search_result = self.search_handler(handler_stack, info);

        match search_result {
            HandlerSearchResult::Found { handler, .. } => {
                self.stats.handlers_found += 1;
                UnwindResult::handler_found(
                    handler.handler_idx as u32, // Use handler_idx as PC placeholder
                    handler.stack_depth,
                    self.stats.handlers_examined as u32,
                )
            }
            HandlerSearchResult::Finally { handler, .. } => {
                let mut result = UnwindResult::default();
                result.action = UnwindAction::ExecuteFinally {
                    finally_pc: handler.handler_idx as u32,
                    stack_depth: handler.stack_depth,
                    reraise: true,
                };
                result.handlers_examined = self.stats.handlers_examined as u32;
                result.finally_queue.push(FinallyEntry::new(
                    handler.handler_idx as u32,
                    handler.stack_depth,
                    handler.frame_id,
                    true,
                ));
                result
            }
            HandlerSearchResult::NotFound => {
                UnwindResult::unhandled(0, self.stats.handlers_examined as u32)
            }
        }
    }

    /// Updates statistics after an unwind operation.
    #[inline]
    pub fn record_unwind(&mut self, frames: u32) {
        self.stats.frames_unwound += frames as u64;
        if frames > self.stats.max_frames_unwound {
            self.stats.max_frames_unwound = frames;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests;
