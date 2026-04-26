//! Exception handler table.
//!
//! This module provides the `HandlerTable` which maps bytecode ranges to
//! exception handlers. This is the core data structure for zero-cost
//! exception handling - the happy path has no overhead.
//!
//! # Performance Design
//!
//! - **Compile-time generated**: Tables are built during bytecode compilation
//! - **Binary search**: O(log N) handler lookup by PC
//! - **Stack depth tracking**: Enables efficient stack unwinding
//! - **JIT-friendly**: Static metadata, no runtime checks on happy path

use super::types::ExceptionTypeId;
use std::fmt;

// ============================================================================
// Handler Entry
// ============================================================================

/// A single exception handler entry.
///
/// Maps a range of bytecode offsets to a handler address.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct HandlerEntry {
    /// Start of the try block (inclusive).
    pub pc_start: u32,

    /// End of the try block (exclusive).
    pub pc_end: u32,

    /// Bytecode offset of the handler (except block).
    pub handler_pc: u32,

    /// Exception type filter.
    /// - Specific type: filters by type ID
    /// - CATCH_ALL: bare `except:` clause
    pub type_filter: u16,

    /// Operand stack depth at try entry (for stack unwinding).
    pub stack_depth: u8,

    /// Handler flags.
    pub flags: HandlerFlags,
}

/// Special value meaning "catch all exceptions" (bare except).
pub const CATCH_ALL: u16 = 0xFFFF;

/// Special value meaning "catch Exception and subclasses".
pub const CATCH_EXCEPTION: u16 = 0xFFFE;

/// Handler flags.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct HandlerFlags(u8);

impl HandlerFlags {
    /// Empty flags.
    pub const EMPTY: Self = Self(0);

    /// This is a finally block (always runs).
    pub const FINALLY: u8 = 1 << 0;

    /// This handler has a named target (`except E as e:`).
    pub const NAMED: u8 = 1 << 1;

    /// This is a with block __exit__ handler.
    pub const WITH_EXIT: u8 = 1 << 2;

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

    /// Returns true if this is a finally block.
    #[inline]
    pub const fn is_finally(self) -> bool {
        self.0 & Self::FINALLY != 0
    }

    /// Returns true if the handler has a named target.
    #[inline]
    pub const fn is_named(self) -> bool {
        self.0 & Self::NAMED != 0
    }

    /// Returns true if this is a with block handler.
    #[inline]
    pub const fn is_with_exit(self) -> bool {
        self.0 & Self::WITH_EXIT != 0
    }
}

impl fmt::Debug for HandlerFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut flags = Vec::new();
        if self.is_finally() {
            flags.push("FINALLY");
        }
        if self.is_named() {
            flags.push("NAMED");
        }
        if self.is_with_exit() {
            flags.push("WITH_EXIT");
        }

        if flags.is_empty() {
            write!(f, "HandlerFlags(empty)")
        } else {
            write!(f, "HandlerFlags({})", flags.join(" | "))
        }
    }
}

impl HandlerEntry {
    /// Creates a new handler entry.
    #[inline]
    pub const fn new(
        pc_start: u32,
        pc_end: u32,
        handler_pc: u32,
        type_filter: u16,
        stack_depth: u8,
        flags: HandlerFlags,
    ) -> Self {
        Self {
            pc_start,
            pc_end,
            handler_pc,
            type_filter,
            stack_depth,
            flags,
        }
    }

    /// Creates a handler for a specific exception type.
    #[inline]
    pub const fn for_type(
        pc_start: u32,
        pc_end: u32,
        handler_pc: u32,
        type_id: ExceptionTypeId,
        stack_depth: u8,
    ) -> Self {
        Self {
            pc_start,
            pc_end,
            handler_pc,
            type_filter: type_id.as_u8() as u16,
            stack_depth,
            flags: HandlerFlags::EMPTY,
        }
    }

    /// Creates a bare except handler (catches all).
    #[inline]
    pub const fn catch_all(pc_start: u32, pc_end: u32, handler_pc: u32, stack_depth: u8) -> Self {
        Self {
            pc_start,
            pc_end,
            handler_pc,
            type_filter: CATCH_ALL,
            stack_depth,
            flags: HandlerFlags::EMPTY,
        }
    }

    /// Creates an `except Exception:` handler.
    #[inline]
    pub const fn catch_exception(
        pc_start: u32,
        pc_end: u32,
        handler_pc: u32,
        stack_depth: u8,
    ) -> Self {
        Self {
            pc_start,
            pc_end,
            handler_pc,
            type_filter: CATCH_EXCEPTION,
            stack_depth,
            flags: HandlerFlags::EMPTY,
        }
    }

    /// Creates a finally handler.
    #[inline]
    pub const fn finally(pc_start: u32, pc_end: u32, handler_pc: u32, stack_depth: u8) -> Self {
        Self {
            pc_start,
            pc_end,
            handler_pc,
            type_filter: CATCH_ALL,
            stack_depth,
            flags: HandlerFlags::from_raw(HandlerFlags::FINALLY),
        }
    }

    /// Returns true if this handler covers the given PC.
    #[inline]
    pub const fn contains(&self, pc: u32) -> bool {
        pc >= self.pc_start && pc < self.pc_end
    }

    /// Returns true if this handler matches the given exception type.
    pub fn matches(&self, type_id: ExceptionTypeId) -> bool {
        match self.type_filter {
            CATCH_ALL => true,
            CATCH_EXCEPTION => type_id.is_subclass_of(ExceptionTypeId::Exception),
            filter => {
                if let Some(filter_type) = ExceptionTypeId::from_u8(filter as u8) {
                    type_id.is_subclass_of(filter_type)
                } else {
                    false
                }
            }
        }
    }

    /// Returns true if this is a finally handler.
    #[inline]
    pub const fn is_finally(&self) -> bool {
        self.flags.is_finally()
    }

    /// Returns the range covered by this handler.
    #[inline]
    pub const fn range(&self) -> (u32, u32) {
        (self.pc_start, self.pc_end)
    }
}

impl fmt::Debug for HandlerEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HandlerEntry")
            .field("range", &(self.pc_start..self.pc_end))
            .field("handler_pc", &self.handler_pc)
            .field("type_filter", &self.type_filter)
            .field("stack_depth", &self.stack_depth)
            .field("flags", &self.flags)
            .finish()
    }
}

// ============================================================================
// Handler Table
// ============================================================================

/// Exception handler table for a code object.
///
/// Contains all exception handlers in a function, sorted by start PC
/// for efficient binary search lookup.
#[derive(Clone, Default)]
pub struct HandlerTable {
    /// Handler entries, sorted by pc_start then by nesting depth (outer first).
    entries: Box<[HandlerEntry]>,
}

impl HandlerTable {
    /// Creates an empty handler table.
    #[inline]
    pub fn empty() -> Self {
        Self {
            entries: Box::new([]),
        }
    }

    /// Creates a handler table from entries.
    ///
    /// The entries should be pre-sorted by pc_start.
    pub fn from_entries(entries: impl IntoIterator<Item = HandlerEntry>) -> Self {
        let mut entries: Vec<_> = entries.into_iter().collect();
        // Sort by start PC, then by handler PC (for nested handlers)
        entries.sort_by_key(|e| (e.pc_start, e.handler_pc));
        Self {
            entries: entries.into_boxed_slice(),
        }
    }

    /// Returns true if the table has no handlers.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the number of handlers.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns all entries.
    #[inline]
    pub fn entries(&self) -> &[HandlerEntry] {
        &self.entries
    }

    /// Finds the handler for a given PC and exception type.
    ///
    /// Returns the first matching handler entry, or None if no handler matches.
    /// Uses linear scan since handler tables are typically small (< 10 entries)
    /// and the entries are sorted by pc_start, not pc_end.
    ///
    /// For innermost-first matching, we scan through all handlers that contain
    /// the PC and return the first one that matches the exception type.
    pub fn find_handler(&self, pc: u32, type_id: ExceptionTypeId) -> Option<&HandlerEntry> {
        // Linear scan to find all handlers containing this PC
        // Since entries are sorted by pc_start, we can stop when pc_start > pc
        for entry in self.entries.iter() {
            if entry.pc_start > pc {
                // Past the PC, no more matches possible (entries sorted by pc_start)
                break;
            }

            if entry.contains(pc) && entry.matches(type_id) {
                return Some(entry);
            }
        }

        None
    }

    /// Finds all handlers for a given PC (for finally clause processing).
    ///
    /// Returns handlers in inner-to-outer order.
    pub fn find_all_handlers(&self, pc: u32) -> impl Iterator<Item = &HandlerEntry> {
        self.entries.iter().filter(move |e| e.contains(pc))
    }

    /// Finds the finally handler for a given PC, if any.
    pub fn find_finally(&self, pc: u32) -> Option<&HandlerEntry> {
        self.entries
            .iter()
            .filter(|e| e.contains(pc) && e.is_finally())
            .next()
    }
}

impl fmt::Debug for HandlerTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HandlerTable")
            .field("entries", &self.entries.len())
            .finish()
    }
}

// ============================================================================
// Builder
// ============================================================================

/// Builder for constructing handler tables.
pub struct HandlerTableBuilder {
    entries: Vec<HandlerEntry>,
}

impl HandlerTableBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Creates a builder with capacity hint.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    /// Adds a handler entry.
    pub fn add(&mut self, entry: HandlerEntry) -> &mut Self {
        self.entries.push(entry);
        self
    }

    /// Adds a handler for a specific exception type.
    pub fn add_type_handler(
        &mut self,
        pc_start: u32,
        pc_end: u32,
        handler_pc: u32,
        type_id: ExceptionTypeId,
        stack_depth: u8,
    ) -> &mut Self {
        self.add(HandlerEntry::for_type(
            pc_start,
            pc_end,
            handler_pc,
            type_id,
            stack_depth,
        ))
    }

    /// Adds a catch-all handler.
    pub fn add_catch_all(
        &mut self,
        pc_start: u32,
        pc_end: u32,
        handler_pc: u32,
        stack_depth: u8,
    ) -> &mut Self {
        self.add(HandlerEntry::catch_all(
            pc_start,
            pc_end,
            handler_pc,
            stack_depth,
        ))
    }

    /// Adds a finally handler.
    pub fn add_finally(
        &mut self,
        pc_start: u32,
        pc_end: u32,
        handler_pc: u32,
        stack_depth: u8,
    ) -> &mut Self {
        self.add(HandlerEntry::finally(
            pc_start,
            pc_end,
            handler_pc,
            stack_depth,
        ))
    }

    /// Builds the handler table.
    pub fn build(self) -> HandlerTable {
        HandlerTable::from_entries(self.entries)
    }
}

impl Default for HandlerTableBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests;
