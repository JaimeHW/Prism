//! Exception flags bitfield.
//!
//! This module defines `ExceptionFlags`, a compact bitfield that tracks
//! the state of an exception object without wasting memory on separate booleans.
//!
//! # Performance Design
//!
//! - **Single byte**: All flags packed into 1 byte
//! - **Branchless checks**: Flag testing is a single AND instruction
//! - **Lazy initialization tracking**: Enables deferred allocation of args/traceback

use std::fmt;

/// Compact flags for exception state.
///
/// Packed into a single byte for optimal memory layout.
/// Each flag can be tested with a single AND instruction.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ExceptionFlags(u8);

impl ExceptionFlags {
    // ════════════════════════════════════════════════════════════════════════
    // Flag Bit Positions
    // ════════════════════════════════════════════════════════════════════════

    /// Exception has been normalized (args set).
    const NORMALIZED: u8 = 1 << 0;

    /// Exception has args tuple allocated.
    const HAS_ARGS: u8 = 1 << 1;

    /// Exception has traceback attached.
    const HAS_TRACEBACK: u8 = 1 << 2;

    /// Exception has __cause__ set (explicit chaining).
    const HAS_CAUSE: u8 = 1 << 3;

    /// Exception has __context__ set (implicit chaining).
    const HAS_CONTEXT: u8 = 1 << 4;

    /// Exception's __suppress_context__ is True.
    const SUPPRESS_CONTEXT: u8 = 1 << 5;

    /// Exception is a flyweight (singleton, not heap-allocated).
    const FLYWEIGHT: u8 = 1 << 6;

    /// Exception was re-raised (traceback was extended).
    const RERAISED: u8 = 1 << 7;

    // ════════════════════════════════════════════════════════════════════════
    // Constructors
    // ════════════════════════════════════════════════════════════════════════

    /// Creates empty flags (no bits set).
    #[inline(always)]
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Creates flags with all bits set.
    #[inline(always)]
    pub const fn all() -> Self {
        Self(0xFF)
    }

    /// Creates flags from raw byte value.
    #[inline(always)]
    pub const fn from_raw(value: u8) -> Self {
        Self(value)
    }

    /// Creates flags for a flyweight exception.
    #[inline(always)]
    pub const fn flyweight() -> Self {
        Self(Self::FLYWEIGHT | Self::NORMALIZED)
    }

    /// Creates flags for a newly created exception.
    #[inline(always)]
    pub const fn new_exception() -> Self {
        Self(0)
    }

    /// Creates flags for an exception with args.
    #[inline(always)]
    pub const fn with_args() -> Self {
        Self(Self::HAS_ARGS | Self::NORMALIZED)
    }

    // ════════════════════════════════════════════════════════════════════════
    // Flag Getters (Branchless)
    // ════════════════════════════════════════════════════════════════════════

    /// Returns the raw byte value.
    #[inline(always)]
    pub const fn as_raw(self) -> u8 {
        self.0
    }

    /// Returns true if no flags are set.
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns true if the exception has been normalized.
    #[inline(always)]
    pub const fn is_normalized(self) -> bool {
        self.0 & Self::NORMALIZED != 0
    }

    /// Returns true if the exception has args allocated.
    #[inline(always)]
    pub const fn has_args(self) -> bool {
        self.0 & Self::HAS_ARGS != 0
    }

    /// Returns true if the exception has a traceback.
    #[inline(always)]
    pub const fn has_traceback(self) -> bool {
        self.0 & Self::HAS_TRACEBACK != 0
    }

    /// Returns true if the exception has __cause__ set.
    #[inline(always)]
    pub const fn has_cause(self) -> bool {
        self.0 & Self::HAS_CAUSE != 0
    }

    /// Returns true if the exception has __context__ set.
    #[inline(always)]
    pub const fn has_context(self) -> bool {
        self.0 & Self::HAS_CONTEXT != 0
    }

    /// Returns true if __suppress_context__ is True.
    #[inline(always)]
    pub const fn suppress_context(self) -> bool {
        self.0 & Self::SUPPRESS_CONTEXT != 0
    }

    /// Returns true if this is a flyweight exception.
    #[inline(always)]
    pub const fn is_flyweight(self) -> bool {
        self.0 & Self::FLYWEIGHT != 0
    }

    /// Returns true if the exception was re-raised.
    #[inline(always)]
    pub const fn was_reraised(self) -> bool {
        self.0 & Self::RERAISED != 0
    }

    // ════════════════════════════════════════════════════════════════════════
    // Flag Setters (Return new value, don't mutate)
    // ════════════════════════════════════════════════════════════════════════

    /// Returns flags with normalized bit set.
    #[inline(always)]
    pub const fn set_normalized(self) -> Self {
        Self(self.0 | Self::NORMALIZED)
    }

    /// Returns flags with has_args bit set.
    #[inline(always)]
    pub const fn set_has_args(self) -> Self {
        Self(self.0 | Self::HAS_ARGS)
    }

    /// Returns flags with has_traceback bit set.
    #[inline(always)]
    pub const fn set_has_traceback(self) -> Self {
        Self(self.0 | Self::HAS_TRACEBACK)
    }

    /// Returns flags with has_cause bit set.
    #[inline(always)]
    pub const fn set_has_cause(self) -> Self {
        Self(self.0 | Self::HAS_CAUSE)
    }

    /// Returns flags with has_context bit set.
    #[inline(always)]
    pub const fn set_has_context(self) -> Self {
        Self(self.0 | Self::HAS_CONTEXT)
    }

    /// Returns flags with suppress_context bit set.
    #[inline(always)]
    pub const fn set_suppress_context(self) -> Self {
        Self(self.0 | Self::SUPPRESS_CONTEXT)
    }

    /// Returns flags with flyweight bit set.
    #[inline(always)]
    pub const fn set_flyweight(self) -> Self {
        Self(self.0 | Self::FLYWEIGHT)
    }

    /// Returns flags with reraised bit set.
    #[inline(always)]
    pub const fn set_reraised(self) -> Self {
        Self(self.0 | Self::RERAISED)
    }

    // ════════════════════════════════════════════════════════════════════════
    // Flag Clearers (Return new value, don't mutate)
    // ════════════════════════════════════════════════════════════════════════

    /// Returns flags with has_traceback bit cleared.
    #[inline(always)]
    pub const fn clear_has_traceback(self) -> Self {
        Self(self.0 & !Self::HAS_TRACEBACK)
    }

    /// Returns flags with has_cause bit cleared.
    #[inline(always)]
    pub const fn clear_has_cause(self) -> Self {
        Self(self.0 & !Self::HAS_CAUSE)
    }

    /// Returns flags with has_context bit cleared.
    #[inline(always)]
    pub const fn clear_has_context(self) -> Self {
        Self(self.0 & !Self::HAS_CONTEXT)
    }

    /// Returns flags with suppress_context bit cleared.
    #[inline(always)]
    pub const fn clear_suppress_context(self) -> Self {
        Self(self.0 & !Self::SUPPRESS_CONTEXT)
    }

    /// Returns flags with reraised bit cleared.
    #[inline(always)]
    pub const fn clear_reraised(self) -> Self {
        Self(self.0 & !Self::RERAISED)
    }

    // ════════════════════════════════════════════════════════════════════════
    // Mutation Methods (For when you have &mut self)
    // ════════════════════════════════════════════════════════════════════════

    /// Sets the normalized bit in place.
    #[inline(always)]
    pub fn mark_normalized(&mut self) {
        self.0 |= Self::NORMALIZED;
    }

    /// Sets the has_args bit in place.
    #[inline(always)]
    pub fn mark_has_args(&mut self) {
        self.0 |= Self::HAS_ARGS;
    }

    /// Sets the has_traceback bit in place.
    #[inline(always)]
    pub fn mark_has_traceback(&mut self) {
        self.0 |= Self::HAS_TRACEBACK;
    }

    /// Sets the has_cause bit in place.
    #[inline(always)]
    pub fn mark_has_cause(&mut self) {
        self.0 |= Self::HAS_CAUSE;
    }

    /// Sets the has_context bit in place.
    #[inline(always)]
    pub fn mark_has_context(&mut self) {
        self.0 |= Self::HAS_CONTEXT;
    }

    /// Sets the suppress_context bit in place.
    #[inline(always)]
    pub fn mark_suppress_context(&mut self) {
        self.0 |= Self::SUPPRESS_CONTEXT;
    }

    /// Sets the reraised bit in place.
    #[inline(always)]
    pub fn mark_reraised(&mut self) {
        self.0 |= Self::RERAISED;
    }

    // ════════════════════════════════════════════════════════════════════════
    // Bitwise Operations
    // ════════════════════════════════════════════════════════════════════════

    /// Returns flags with specified bits ORed in.
    #[inline(always)]
    pub const fn with(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns flags with specified bits ANDed out.
    #[inline(always)]
    pub const fn without(self, mask: Self) -> Self {
        Self(self.0 & !mask.0)
    }

    /// Returns true if all bits in mask are set.
    #[inline(always)]
    pub const fn contains(self, mask: Self) -> bool {
        self.0 & mask.0 == mask.0
    }

    /// Returns true if any bits in mask are set.
    #[inline(always)]
    pub const fn intersects(self, mask: Self) -> bool {
        self.0 & mask.0 != 0
    }

    /// Counts the number of set bits.
    #[inline(always)]
    pub const fn count_ones(self) -> u32 {
        self.0.count_ones()
    }
}

impl fmt::Debug for ExceptionFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut flags = Vec::new();

        if self.is_normalized() {
            flags.push("NORMALIZED");
        }
        if self.has_args() {
            flags.push("HAS_ARGS");
        }
        if self.has_traceback() {
            flags.push("HAS_TRACEBACK");
        }
        if self.has_cause() {
            flags.push("HAS_CAUSE");
        }
        if self.has_context() {
            flags.push("HAS_CONTEXT");
        }
        if self.suppress_context() {
            flags.push("SUPPRESS_CONTEXT");
        }
        if self.is_flyweight() {
            flags.push("FLYWEIGHT");
        }
        if self.was_reraised() {
            flags.push("RERAISED");
        }

        if flags.is_empty() {
            write!(f, "ExceptionFlags(empty)")
        } else {
            write!(f, "ExceptionFlags({})", flags.join(" | "))
        }
    }
}

impl std::ops::BitOr for ExceptionFlags {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for ExceptionFlags {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl std::ops::BitAnd for ExceptionFlags {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl std::ops::BitAndAssign for ExceptionFlags {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl std::ops::Not for ExceptionFlags {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self {
        Self(!self.0)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests;
