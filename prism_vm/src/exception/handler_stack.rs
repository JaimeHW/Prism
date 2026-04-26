//! Runtime exception handler stack.
//!
//! This module provides the runtime handler stack that tracks active try
//! blocks during execution. It complements the static `HandlerTable` by
//! maintaining dynamic execution state.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Handler Stack                                 │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  ┌─────────────────┐   push     ┌─────────────────────────────┐│
//! │  │ try block entry │ ────────▶  │ HandlerFrame {              ││
//! │  └─────────────────┘             │   handler_idx: u16          ││
//! │                                   │   frame_id: u32             ││
//! │  ┌─────────────────┐             │   stack_depth: u16          ││
//! │  │ exception raise │ ────────▶  │   flags: HandlerFlags       ││
//! │  └─────────────────┘   pop      │ }                            ││
//! │                                  └─────────────────────────────┘│
//! │                                                                  │
//! │  Benefits:                                                       │
//! │  • O(1) push/pop for try block entry/exit                       │
//! │  • O(1) handler lookup for most exceptions                       │
//! │  • Proper finally block ordering                                 │
//! │  • Efficient stack unwinding across frames                       │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance
//!
//! | Operation | Complexity |
//! |-----------|------------|
//! | push_handler | O(1) |
//! | pop_handler | O(1) |
//! | find_handler | O(N) worst case, O(1) typical |
//! | unwind_to | O(N) where N = frames to unwind |

use smallvec::SmallVec;
use std::fmt;

// ============================================================================
// Constants
// ============================================================================

/// Default inline capacity for the handler stack.
/// Most exception handling involves small nesting depths.
const INLINE_HANDLER_CAPACITY: usize = 8;

/// Maximum handler stack depth to prevent unbounded growth.
const MAX_HANDLER_DEPTH: usize = 1024;

/// Sentinel value for "no handler".
pub const NO_HANDLER: u16 = u16::MAX;

/// Sentinel value for "no frame".
pub const NO_FRAME: u32 = u32::MAX;

// ============================================================================
// Handler Frame
// ============================================================================

/// A runtime handler frame representing an active try block.
///
/// # Memory Layout (8 bytes)
///
/// ```text
/// ┌─────────────────────────────────────────────────────────┐
/// │ handler_idx (2b) │ stack_depth (2b) │ frame_id (4b)     │
/// └─────────────────────────────────────────────────────────┘
/// ```
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct HandlerFrame {
    /// Index into the code object's HandlerTable.
    pub handler_idx: u16,

    /// Stack depth when the try block was entered.
    /// Used for stack unwinding.
    pub stack_depth: u16,

    /// Frame ID that owns this handler.
    /// Used for cross-frame unwinding.
    pub frame_id: u32,
}

impl HandlerFrame {
    /// Creates a new handler frame.
    #[inline]
    pub const fn new(handler_idx: u16, stack_depth: u16, frame_id: u32) -> Self {
        Self {
            handler_idx,
            stack_depth,
            frame_id,
        }
    }

    /// Returns true if this is a valid handler (not sentinel).
    #[inline]
    pub const fn is_valid(&self) -> bool {
        self.handler_idx != NO_HANDLER
    }

    /// Creates an invalid/sentinel handler frame.
    #[inline]
    pub const fn invalid() -> Self {
        Self {
            handler_idx: NO_HANDLER,
            stack_depth: 0,
            frame_id: NO_FRAME,
        }
    }
}

impl fmt::Debug for HandlerFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            f.debug_struct("HandlerFrame")
                .field("handler_idx", &self.handler_idx)
                .field("stack_depth", &self.stack_depth)
                .field("frame_id", &self.frame_id)
                .finish()
        } else {
            write!(f, "HandlerFrame(invalid)")
        }
    }
}

impl Default for HandlerFrame {
    fn default() -> Self {
        Self::invalid()
    }
}

// ============================================================================
// Handler Stack
// ============================================================================

/// Runtime stack of active exception handlers.
///
/// This stack tracks all active try blocks across the call stack.
/// When an exception is raised, the stack is unwound to find a
/// matching handler.
///
/// # Usage
///
/// ```ignore
/// let mut stack = HandlerStack::new();
///
/// // Enter a try block
/// stack.push(HandlerFrame::new(0, 5, 0));
///
/// // ... execute try block ...
///
/// // Exit normally
/// stack.pop();
/// ```
#[derive(Clone)]
pub struct HandlerStack {
    /// Stack of active handlers (newest on top).
    frames: SmallVec<[HandlerFrame; INLINE_HANDLER_CAPACITY]>,
}

impl HandlerStack {
    /// Creates a new empty handler stack.
    #[inline]
    pub fn new() -> Self {
        Self {
            frames: SmallVec::new(),
        }
    }

    /// Creates a handler stack with the specified capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            frames: SmallVec::with_capacity(capacity),
        }
    }

    /// Returns true if the stack is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Returns the number of active handlers.
    #[inline]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Pushes a new handler frame onto the stack.
    ///
    /// Returns `true` if successful, `false` if the stack is full.
    #[inline]
    pub fn push(&mut self, frame: HandlerFrame) -> bool {
        if self.frames.len() >= MAX_HANDLER_DEPTH {
            return false;
        }
        self.frames.push(frame);
        true
    }

    /// Pops the top handler frame from the stack.
    ///
    /// Returns `None` if the stack is empty.
    #[inline]
    pub fn pop(&mut self) -> Option<HandlerFrame> {
        self.frames.pop()
    }

    /// Returns a reference to the top handler frame without removing it.
    #[inline]
    pub fn peek(&self) -> Option<&HandlerFrame> {
        self.frames.last()
    }

    /// Returns a mutable reference to the top handler frame.
    #[inline]
    pub fn peek_mut(&mut self) -> Option<&mut HandlerFrame> {
        self.frames.last_mut()
    }

    /// Clears all handlers from the stack.
    #[inline]
    pub fn clear(&mut self) {
        self.frames.clear();
    }

    /// Returns an iterator over handler frames from top to bottom.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &HandlerFrame> + DoubleEndedIterator {
        self.frames.iter().rev()
    }

    /// Returns the depth of the stack.
    #[inline]
    pub fn depth(&self) -> usize {
        self.frames.len()
    }

    /// Pops all handlers belonging to a specific frame.
    ///
    /// This is used during frame exit to clean up handlers.
    #[inline]
    pub fn pop_frame_handlers(&mut self, frame_id: u32) {
        while let Some(handler) = self.frames.last() {
            if handler.frame_id == frame_id {
                self.frames.pop();
            } else {
                break;
            }
        }
    }

    /// Finds the first handler matching the given frame ID.
    ///
    /// Returns the handler frame if found, or `None` if no handler
    /// exists for the specified frame.
    #[inline]
    pub fn find_in_frame(&self, frame_id: u32) -> Option<&HandlerFrame> {
        self.frames.iter().rev().find(|h| h.frame_id == frame_id)
    }

    /// Returns the number of handlers for a specific frame.
    #[inline]
    pub fn count_frame_handlers(&self, frame_id: u32) -> usize {
        self.frames
            .iter()
            .filter(|h| h.frame_id == frame_id)
            .count()
    }

    /// Truncates the stack to the given depth.
    ///
    /// This is used during exception handling to unwind to a
    /// specific point.
    #[inline]
    pub fn truncate(&mut self, depth: usize) {
        self.frames.truncate(depth);
    }

    /// Returns the handler at the given index (0 = bottom).
    #[inline]
    pub fn get(&self, index: usize) -> Option<&HandlerFrame> {
        self.frames.get(index)
    }
}

impl Default for HandlerStack {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for HandlerStack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HandlerStack")
            .field("depth", &self.frames.len())
            .field("frames", &self.frames.as_slice())
            .finish()
    }
}

// ============================================================================
// Handler Search Result
// ============================================================================

/// Result of searching for an exception handler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandlerSearchResult {
    /// Found a handler at the given stack index.
    Found {
        /// Index in the handler stack.
        stack_index: usize,
        /// The handler frame.
        handler: HandlerFrame,
    },

    /// No handler found, exception should propagate to caller.
    NotFound,

    /// Found a finally block that must run before continuing search.
    Finally {
        /// Index in the handler stack.
        stack_index: usize,
        /// The finally handler frame.
        handler: HandlerFrame,
    },
}

impl HandlerSearchResult {
    /// Returns true if a handler was found.
    #[inline]
    pub fn found(&self) -> bool {
        matches!(self, Self::Found { .. } | Self::Finally { .. })
    }

    /// Returns the handler frame if found.
    #[inline]
    pub fn handler(&self) -> Option<HandlerFrame> {
        match self {
            Self::Found { handler, .. } => Some(*handler),
            Self::Finally { handler, .. } => Some(*handler),
            Self::NotFound => None,
        }
    }

    /// Returns the stack index if found.
    #[inline]
    pub fn stack_index(&self) -> Option<usize> {
        match self {
            Self::Found { stack_index, .. } => Some(*stack_index),
            Self::Finally { stack_index, .. } => Some(*stack_index),
            Self::NotFound => None,
        }
    }
}

// ============================================================================
// Handler Stack Stats
// ============================================================================

/// Statistics for handler stack operations.
#[derive(Clone, Copy, Debug, Default)]
pub struct HandlerStackStats {
    /// Total number of push operations.
    pub push_count: u64,

    /// Total number of pop operations.
    pub pop_count: u64,

    /// Total number of search operations.
    pub search_count: u64,

    /// Total handlers examined during searches.
    pub handlers_examined: u64,

    /// Maximum stack depth observed.
    pub max_depth: usize,

    /// Number of searches that found a handler.
    pub hits: u64,

    /// Number of searches that found no handler.
    pub misses: u64,
}

impl HandlerStackStats {
    /// Creates new empty stats.
    #[inline]
    pub const fn new() -> Self {
        Self {
            push_count: 0,
            pop_count: 0,
            search_count: 0,
            handlers_examined: 0,
            max_depth: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Returns the hit rate as a percentage.
    #[inline]
    pub fn hit_rate(&self) -> f64 {
        if self.search_count == 0 {
            0.0
        } else {
            (self.hits as f64 / self.search_count as f64) * 100.0
        }
    }

    /// Returns the average handlers examined per search.
    #[inline]
    pub fn avg_handlers_examined(&self) -> f64 {
        if self.search_count == 0 {
            0.0
        } else {
            self.handlers_examined as f64 / self.search_count as f64
        }
    }

    /// Resets all statistics.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests;
