//! Regex compilation flags.
//!
//! Bitflags for regex compilation matching Python's `re` module flags.

use std::fmt;

// =============================================================================
// Flag Constants
// =============================================================================

/// Regex compilation flags matching Python's `re` module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RegexFlags(u32);

impl RegexFlags {
    /// No flags set.
    pub const NONE: u32 = 0;

    /// Case-insensitive matching (re.I, re.IGNORECASE).
    pub const IGNORECASE: u32 = 1 << 0;

    /// Multi-line mode: ^ and $ match at line boundaries (re.M, re.MULTILINE).
    pub const MULTILINE: u32 = 1 << 1;

    /// Dot matches newline (re.S, re.DOTALL).
    pub const DOTALL: u32 = 1 << 2;

    /// Verbose mode: whitespace and comments ignored (re.X, re.VERBOSE).
    pub const VERBOSE: u32 = 1 << 3;

    /// ASCII-only matching for \w, \b, \s, \d (re.A, re.ASCII).
    pub const ASCII: u32 = 1 << 4;

    /// Unicode matching (default in Python 3) (re.U, re.UNICODE).
    pub const UNICODE: u32 = 1 << 5;

    /// Locale-dependent matching (re.L, re.LOCALE).
    pub const LOCALE: u32 = 1 << 6;

    /// Create new flags from raw value.
    #[inline]
    pub const fn new(bits: u32) -> Self {
        Self(bits)
    }

    /// Get raw bits.
    #[inline]
    pub const fn bits(&self) -> u32 {
        self.0
    }

    /// Check if a flag is set.
    #[inline]
    pub const fn contains(&self, flag: u32) -> bool {
        (self.0 & flag) != 0
    }

    /// Check if case-insensitive.
    #[inline]
    pub const fn is_case_insensitive(&self) -> bool {
        self.contains(Self::IGNORECASE)
    }

    /// Check if multi-line mode.
    #[inline]
    pub const fn is_multiline(&self) -> bool {
        self.contains(Self::MULTILINE)
    }

    /// Check if dot-all mode.
    #[inline]
    pub const fn is_dotall(&self) -> bool {
        self.contains(Self::DOTALL)
    }

    /// Check if verbose mode.
    #[inline]
    pub const fn is_verbose(&self) -> bool {
        self.contains(Self::VERBOSE)
    }

    /// Check if ASCII-only mode.
    #[inline]
    pub const fn is_ascii(&self) -> bool {
        self.contains(Self::ASCII)
    }

    /// Combine flags.
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl From<u32> for RegexFlags {
    #[inline]
    fn from(bits: u32) -> Self {
        Self(bits)
    }
}

impl From<RegexFlags> for u32 {
    #[inline]
    fn from(flags: RegexFlags) -> Self {
        flags.0
    }
}

impl fmt::Display for RegexFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if self.is_case_insensitive() {
            parts.push("IGNORECASE");
        }
        if self.is_multiline() {
            parts.push("MULTILINE");
        }
        if self.is_dotall() {
            parts.push("DOTALL");
        }
        if self.is_verbose() {
            parts.push("VERBOSE");
        }
        if self.is_ascii() {
            parts.push("ASCII");
        }
        if parts.is_empty() {
            write!(f, "0")
        } else {
            write!(f, "re.{}", parts.join("|re."))
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flag_constants() {
        assert_eq!(RegexFlags::IGNORECASE, 1);
        assert_eq!(RegexFlags::MULTILINE, 2);
        assert_eq!(RegexFlags::DOTALL, 4);
        assert_eq!(RegexFlags::VERBOSE, 8);
        assert_eq!(RegexFlags::ASCII, 16);
    }

    #[test]
    fn test_flag_contains() {
        let flags = RegexFlags::new(RegexFlags::IGNORECASE | RegexFlags::MULTILINE);
        assert!(flags.is_case_insensitive());
        assert!(flags.is_multiline());
        assert!(!flags.is_dotall());
    }

    #[test]
    fn test_flag_union() {
        let a = RegexFlags::new(RegexFlags::IGNORECASE);
        let b = RegexFlags::new(RegexFlags::MULTILINE);
        let c = a.union(b);
        assert!(c.is_case_insensitive());
        assert!(c.is_multiline());
    }

    #[test]
    fn test_flag_display() {
        let flags = RegexFlags::new(RegexFlags::IGNORECASE | RegexFlags::DOTALL);
        let s = flags.to_string();
        assert!(s.contains("IGNORECASE"));
        assert!(s.contains("DOTALL"));
    }

    #[test]
    fn test_default_flags() {
        let flags = RegexFlags::default();
        assert_eq!(flags.bits(), 0);
        assert!(!flags.is_case_insensitive());
    }
}
