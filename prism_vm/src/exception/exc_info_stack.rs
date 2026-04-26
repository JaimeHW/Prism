//! Exception info stack for CPython 3.11+ semantics.
//!
//! This module provides `ExcInfoStack`, which maintains a stack of exception
//! contexts for proper handling of nested try/except/finally blocks.
//!
//! # CPython 3.11+ Semantics
//!
//! In Python 3.11+, exception contexts are preserved on a stack:
//! - `PushExcInfo`: Saves current exception state before entering handler
//! - `PopExcInfo`: Restores previous exception state after handler exits
//!
//! This enables proper exception chaining and `sys.exc_info()` behavior.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      ExcInfoStack                                │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  ┌─────────────────┐                                            │
//! │  │  ExcInfoEntry   │ ◄── Top (most recent)                      │
//! │  │  ├─ type_id     │                                            │
//! │  │  ├─ value       │                                            │
//! │  │  └─ tb_ref      │                                            │
//! │  ├─────────────────┤                                            │
//! │  │  ExcInfoEntry   │                                            │
//! │  ├─────────────────┤                                            │
//! │  │       ...       │                                            │
//! │  └─────────────────┘ ◄── Bottom (oldest)                        │
//! │                                                                  │
//! │  Inline capacity: 4 entries (typical nesting depth)             │
//! │  Max depth: 255 (enforced)                                      │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance
//!
//! | Operation | Complexity | Notes |
//! |-----------|------------|-------|
//! | push | O(1) amortized | Inline for depth ≤ 4 |
//! | pop | O(1) | Direct stack pop |
//! | peek | O(1) | No allocation |
//! | clear | O(N) | N = current depth |

use prism_core::Value;
use smallvec::SmallVec;
use std::fmt;

// =============================================================================
// Constants
// =============================================================================

/// Inline capacity for exception info stack.
/// Most programs have shallow exception nesting (try { try { } } is depth 2).
const INLINE_CAPACITY: usize = 4;

/// Maximum exception info stack depth (prevent unbounded growth).
const MAX_DEPTH: usize = 255;

// =============================================================================
// ExcInfoEntry
// =============================================================================

/// A single exception info entry preserving exception context.
///
/// # Memory Layout (48 bytes on 64-bit)
///
/// ```text
/// ┌────────────────────────────────────────────┐
/// │ type_id: u16                    (2 bytes)  │
/// │ flags: EntryFlags               (1 byte)   │
/// │ _pad: [u8; 5]                   (5 bytes)  │
/// │ value: Option<Value>            (16 bytes) │
/// │ traceback_id: u32               (4 bytes)  │
/// │ frame_id: u32                   (4 bytes)  │
/// │ pc: u32                         (4 bytes)  │
/// │ _reserved: [u8; 12]             (12 bytes) │
/// └────────────────────────────────────────────┘
/// ```
#[derive(Clone)]
#[repr(C)]
pub struct ExcInfoEntry {
    /// Exception type ID (ExceptionTypeId discriminant).
    type_id: u16,

    /// Entry flags.
    flags: EntryFlags,

    /// Padding for alignment.
    _pad: [u8; 5],

    /// Exception value (cloned for preservation).
    value: Option<Value>,

    /// Traceback reference ID (index into traceback table).
    traceback_id: u32,

    /// Frame ID where exception was raised.
    frame_id: u32,

    /// Program counter where exception was raised.
    pc: u32,

    /// Reserved for future use.
    _reserved: [u8; 12],
}

/// Entry flags for ExcInfoEntry.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct EntryFlags(u8);

impl EntryFlags {
    /// No flags set.
    pub const EMPTY: Self = Self(0);

    /// Exception was explicitly raised (vs. implicit from runtime).
    pub const EXPLICIT: u8 = 1 << 0;

    /// Exception has a chained cause (__cause__ is set).
    pub const HAS_CAUSE: u8 = 1 << 1;

    /// Context was suppressed (__suppress_context__ = True).
    pub const SUPPRESS_CONTEXT: u8 = 1 << 2;

    /// Exception is currently being handled.
    pub const HANDLING: u8 = 1 << 3;

    /// Exception came from a `raise ... from ...` statement.
    pub const FROM_RAISE_FROM: u8 = 1 << 4;

    /// Exception was saved while running a finally cleanup.
    pub const FINALLY: u8 = 1 << 5;

    /// Creates flags from raw value.
    #[inline]
    pub const fn from_raw(value: u8) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[inline]
    pub const fn as_raw(self) -> u8 {
        self.0
    }

    /// Sets a flag.
    #[inline]
    pub fn set(&mut self, flag: u8) {
        self.0 |= flag;
    }

    /// Clears a flag.
    #[inline]
    pub fn clear(&mut self, flag: u8) {
        self.0 &= !flag;
    }

    /// Checks if a flag is set.
    #[inline]
    pub const fn has(self, flag: u8) -> bool {
        (self.0 & flag) != 0
    }

    /// Returns true if exception was explicitly raised.
    #[inline]
    pub const fn is_explicit(self) -> bool {
        self.has(Self::EXPLICIT)
    }

    /// Returns true if exception has a chained cause.
    #[inline]
    pub const fn has_cause(self) -> bool {
        self.has(Self::HAS_CAUSE)
    }

    /// Returns true if context is suppressed.
    #[inline]
    pub const fn is_context_suppressed(self) -> bool {
        self.has(Self::SUPPRESS_CONTEXT)
    }
}

impl fmt::Debug for EntryFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut flags = Vec::new();
        if self.has(Self::EXPLICIT) {
            flags.push("EXPLICIT");
        }
        if self.has(Self::HAS_CAUSE) {
            flags.push("HAS_CAUSE");
        }
        if self.has(Self::SUPPRESS_CONTEXT) {
            flags.push("SUPPRESS_CONTEXT");
        }
        if self.has(Self::HANDLING) {
            flags.push("HANDLING");
        }
        if self.has(Self::FROM_RAISE_FROM) {
            flags.push("FROM_RAISE_FROM");
        }
        if self.has(Self::FINALLY) {
            flags.push("FINALLY");
        }
        write!(f, "EntryFlags({})", flags.join("|"))
    }
}

impl ExcInfoEntry {
    /// Creates a new exception info entry.
    #[inline]
    pub fn new(type_id: u16, value: Option<Value>) -> Self {
        Self {
            type_id,
            flags: EntryFlags::EMPTY,
            _pad: [0; 5],
            value,
            traceback_id: 0,
            frame_id: 0,
            pc: 0,
            _reserved: [0; 12],
        }
    }

    /// Creates an entry with full context.
    #[inline]
    pub fn with_context(
        type_id: u16,
        value: Option<Value>,
        traceback_id: u32,
        frame_id: u32,
        pc: u32,
    ) -> Self {
        Self {
            type_id,
            flags: EntryFlags::EMPTY,
            _pad: [0; 5],
            value,
            traceback_id,
            frame_id,
            pc,
            _reserved: [0; 12],
        }
    }

    /// Creates an empty/sentinel entry.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            type_id: 0,
            flags: EntryFlags::EMPTY,
            _pad: [0; 5],
            value: None,
            traceback_id: 0,
            frame_id: 0,
            pc: 0,
            _reserved: [0; 12],
        }
    }

    /// Returns true if this entry represents an active exception.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.type_id != 0 || self.value.is_some()
    }

    /// Returns the exception type ID.
    #[inline]
    pub const fn type_id(&self) -> u16 {
        self.type_id
    }

    /// Returns a reference to the exception value.
    #[inline]
    pub fn value(&self) -> Option<&Value> {
        self.value.as_ref()
    }

    /// Returns a cloned exception value.
    #[inline]
    pub fn value_cloned(&self) -> Option<Value> {
        self.value.clone()
    }

    /// Sets the exception value.
    #[inline]
    pub fn set_value(&mut self, value: Option<Value>) {
        self.value = value;
    }

    /// Returns the traceback ID.
    #[inline]
    pub const fn traceback_id(&self) -> u32 {
        self.traceback_id
    }

    /// Sets the traceback ID.
    #[inline]
    pub fn set_traceback_id(&mut self, id: u32) {
        self.traceback_id = id;
    }

    /// Returns the frame ID where exception was raised.
    #[inline]
    pub const fn frame_id(&self) -> u32 {
        self.frame_id
    }

    /// Returns the program counter where exception was raised.
    #[inline]
    pub const fn pc(&self) -> u32 {
        self.pc
    }

    /// Returns the entry flags.
    #[inline]
    pub const fn flags(&self) -> EntryFlags {
        self.flags
    }

    /// Returns a mutable reference to the entry flags.
    #[inline]
    pub fn flags_mut(&mut self) -> &mut EntryFlags {
        &mut self.flags
    }

    /// Marks this entry as having a chained cause.
    #[inline]
    pub fn set_has_cause(&mut self) {
        self.flags.set(EntryFlags::HAS_CAUSE);
    }

    /// Marks this entry as suppressing context.
    #[inline]
    pub fn set_suppress_context(&mut self) {
        self.flags.set(EntryFlags::SUPPRESS_CONTEXT);
    }

    /// Marks this entry as from a `raise ... from ...` statement.
    #[inline]
    pub fn set_from_raise_from(&mut self) {
        self.flags.set(EntryFlags::FROM_RAISE_FROM);
        self.flags.set(EntryFlags::HAS_CAUSE);
        self.flags.set(EntryFlags::SUPPRESS_CONTEXT);
    }
}

impl Default for ExcInfoEntry {
    fn default() -> Self {
        Self::empty()
    }
}

impl fmt::Debug for ExcInfoEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExcInfoEntry")
            .field("type_id", &self.type_id)
            .field("flags", &self.flags)
            .field("has_value", &self.value.is_some())
            .field("traceback_id", &self.traceback_id)
            .field("frame_id", &self.frame_id)
            .field("pc", &self.pc)
            .finish()
    }
}

// =============================================================================
// ExcInfoStack
// =============================================================================

/// Stack of exception info entries for nested exception handling.
///
/// This implements CPython 3.11+ semantics where exception contexts are
/// preserved on a stack during nested try/except/finally blocks.
///
/// # Usage
///
/// ```ignore
/// let mut stack = ExcInfoStack::new();
///
/// // Entering an except handler - preserve current exception
/// stack.push(ExcInfoEntry::new(exc_type_id, Some(exc_value)));
///
/// // Handle exception...
///
/// // Exiting handler - restore previous exception state
/// if let Some(entry) = stack.pop() {
///     // Restore previous exception context
/// }
/// ```
#[derive(Clone)]
pub struct ExcInfoStack {
    /// Stack entries (inline for common case).
    entries: SmallVec<[ExcInfoEntry; INLINE_CAPACITY]>,

    /// Statistics for monitoring.
    stats: ExcInfoStackStats,
}

/// Statistics for ExcInfoStack operations.
#[derive(Clone, Copy, Default)]
pub struct ExcInfoStackStats {
    /// Total push operations.
    pub pushes: u32,

    /// Total pop operations.
    pub pops: u32,

    /// Peak stack depth reached.
    pub peak_depth: u32,

    /// Number of overflow rejections.
    pub overflows: u32,
}

impl ExcInfoStack {
    /// Creates a new empty exception info stack.
    #[inline]
    pub fn new() -> Self {
        Self {
            entries: SmallVec::new(),
            stats: ExcInfoStackStats::default(),
        }
    }

    /// Creates a new stack with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: SmallVec::with_capacity(capacity),
            stats: ExcInfoStackStats::default(),
        }
    }

    /// Returns true if the stack is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the current stack depth.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns the maximum allowed depth.
    #[inline]
    pub const fn max_depth() -> usize {
        MAX_DEPTH
    }

    /// Pushes an exception info entry onto the stack.
    ///
    /// Returns `true` if successful, `false` if stack overflow.
    #[inline]
    pub fn push(&mut self, entry: ExcInfoEntry) -> bool {
        if self.entries.len() >= MAX_DEPTH {
            self.stats.overflows += 1;
            return false;
        }

        self.entries.push(entry);
        self.stats.pushes += 1;

        let depth = self.entries.len() as u32;
        if depth > self.stats.peak_depth {
            self.stats.peak_depth = depth;
        }

        true
    }

    /// Pops the top exception info entry from the stack.
    ///
    /// Returns `None` if the stack is empty.
    #[inline]
    pub fn pop(&mut self) -> Option<ExcInfoEntry> {
        let entry = self.entries.pop();
        if entry.is_some() {
            self.stats.pops += 1;
        }
        entry
    }

    /// Returns a reference to the top entry without removing it.
    #[inline]
    pub fn peek(&self) -> Option<&ExcInfoEntry> {
        self.entries.last()
    }

    /// Returns a mutable reference to the top entry.
    #[inline]
    pub fn peek_mut(&mut self) -> Option<&mut ExcInfoEntry> {
        self.entries.last_mut()
    }

    /// Returns a reference to the entry at the given index (0 = bottom).
    #[inline]
    pub fn get(&self, index: usize) -> Option<&ExcInfoEntry> {
        self.entries.get(index)
    }

    /// Clears all entries from the stack.
    #[inline]
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Returns an iterator over entries from top to bottom.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &ExcInfoEntry> + DoubleEndedIterator {
        self.entries.iter().rev()
    }

    /// Returns an iterator over entries from bottom to top.
    #[inline]
    pub fn iter_bottom_up(&self) -> impl Iterator<Item = &ExcInfoEntry> {
        self.entries.iter()
    }

    /// Returns the stack statistics.
    #[inline]
    pub const fn stats(&self) -> &ExcInfoStackStats {
        &self.stats
    }

    /// Resets the statistics.
    #[inline]
    pub fn reset_stats(&mut self) {
        self.stats = ExcInfoStackStats::default();
    }

    /// Returns the current exception info (top of stack).
    ///
    /// This implements `sys.exc_info()` semantics.
    #[inline]
    pub fn current_exc_info(&self) -> (Option<u16>, Option<Value>, Option<u32>) {
        match self.peek() {
            Some(entry) => (
                Some(entry.type_id),
                entry.value_cloned(),
                if entry.traceback_id != 0 {
                    Some(entry.traceback_id)
                } else {
                    None
                },
            ),
            None => (None, None, None),
        }
    }

    /// Finds the first active exception entry from the top.
    #[inline]
    pub fn find_active(&self) -> Option<&ExcInfoEntry> {
        self.entries.iter().rev().find(|e| e.is_active())
    }

    /// Truncates the stack to the given depth.
    #[inline]
    pub fn truncate(&mut self, depth: usize) {
        self.entries.truncate(depth);
    }
}

impl Default for ExcInfoStack {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ExcInfoStack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExcInfoStack")
            .field("depth", &self.entries.len())
            .field("entries", &self.entries)
            .field("stats", &self.stats)
            .finish()
    }
}

impl fmt::Debug for ExcInfoStackStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExcInfoStackStats")
            .field("pushes", &self.pushes)
            .field("pops", &self.pops)
            .field("peak_depth", &self.peak_depth)
            .field("overflows", &self.overflows)
            .finish()
    }
}
