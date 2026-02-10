//! File mode parsing with validation.
//!
//! Implements Python's file mode string parsing with all valid combinations.
//! The mode string determines read/write access, binary/text mode, and behavior.
//!
//! # Mode String Format
//!
//! ```text
//! mode ::= [rwa][b|t]?[+]?[x]?
//! ```
//!
//! - `r`: Open for reading (default)
//! - `w`: Open for writing, truncating the file
//! - `a`: Open for writing, appending to end
//! - `x`: Exclusive creation, fail if file exists  
//! - `b`: Binary mode
//! - `t`: Text mode (default)
//! - `+`: Update mode (read and write)

use std::fmt;

/// Parsed file mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileMode {
    /// Read access enabled
    pub read: bool,
    /// Write access enabled  
    pub write: bool,
    /// Append mode (write at end)
    pub append: bool,
    /// Exclusive creation mode
    pub exclusive: bool,
    /// Binary mode (false = text mode)
    pub binary: bool,
    /// Truncate file on open
    pub truncate: bool,
}

impl Default for FileMode {
    #[inline]
    fn default() -> Self {
        Self {
            read: true,
            write: false,
            append: false,
            exclusive: false,
            binary: false,
            truncate: false,
        }
    }
}

impl FileMode {
    /// Parse a Python mode string into a `FileMode`.
    ///
    /// # Valid Mode Strings
    ///
    /// | Mode | Description |
    /// |------|-------------|
    /// | `r`  | Read text (default) |
    /// | `rb` | Read binary |
    /// | `r+` | Read/write text |
    /// | `rb+`| Read/write binary |
    /// | `w`  | Write text, truncate |
    /// | `wb` | Write binary, truncate |
    /// | `w+` | Read/write text, truncate |
    /// | `a`  | Append text |
    /// | `ab` | Append binary |
    /// | `a+` | Read/append text |
    /// | `x`  | Exclusive create text |
    /// | `xb` | Exclusive create binary |
    ///
    /// # Performance
    ///
    /// This parser uses a single-pass state machine with no allocations.
    /// Typical mode strings are 1-3 characters.
    #[inline]
    pub fn parse(mode: &str) -> Result<Self, ParseModeError> {
        if mode.is_empty() {
            return Err(ParseModeError::Empty);
        }

        let bytes = mode.as_bytes();
        let mut result = FileMode::default();

        // Track what we've seen to detect duplicates
        let mut seen_rwa = false;
        let mut seen_bt = false;
        let mut seen_plus = false;
        let mut seen_x = false;

        // Primary mode character (r/w/a/x)
        let mut primary = None;

        for &b in bytes {
            match b {
                b'r' => {
                    if seen_rwa {
                        return Err(ParseModeError::InvalidMode(mode.to_string()));
                    }
                    seen_rwa = true;
                    primary = Some(b'r');
                    result.read = true;
                    result.write = false;
                    result.append = false;
                    result.truncate = false;
                }
                b'w' => {
                    if seen_rwa {
                        return Err(ParseModeError::InvalidMode(mode.to_string()));
                    }
                    seen_rwa = true;
                    primary = Some(b'w');
                    result.read = false;
                    result.write = true;
                    result.truncate = true;
                }
                b'a' => {
                    if seen_rwa {
                        return Err(ParseModeError::InvalidMode(mode.to_string()));
                    }
                    seen_rwa = true;
                    primary = Some(b'a');
                    result.read = false;
                    result.write = true;
                    result.append = true;
                }
                b'x' => {
                    if seen_x || seen_rwa {
                        return Err(ParseModeError::InvalidMode(mode.to_string()));
                    }
                    seen_x = true;
                    seen_rwa = true;
                    primary = Some(b'x');
                    result.read = false;
                    result.write = true;
                    result.exclusive = true;
                }
                b'b' => {
                    if seen_bt {
                        return Err(ParseModeError::InvalidMode(mode.to_string()));
                    }
                    seen_bt = true;
                    result.binary = true;
                }
                b't' => {
                    if seen_bt {
                        return Err(ParseModeError::InvalidMode(mode.to_string()));
                    }
                    seen_bt = true;
                    result.binary = false;
                }
                b'+' => {
                    if seen_plus {
                        return Err(ParseModeError::InvalidMode(mode.to_string()));
                    }
                    seen_plus = true;
                    result.read = true;
                    result.write = true;
                }
                _ => {
                    return Err(ParseModeError::InvalidCharacter(b as char));
                }
            }
        }

        // Must have a primary mode
        if primary.is_none() && !seen_plus {
            return Err(ParseModeError::InvalidMode(mode.to_string()));
        }

        Ok(result)
    }

    /// Check if mode requires the file to exist.
    #[inline]
    pub const fn requires_existing(&self) -> bool {
        self.read && !self.write && !self.append
    }

    /// Check if mode creates file if not exists.
    #[inline]
    pub const fn creates_file(&self) -> bool {
        self.write || self.append
    }

    /// Convert to std::fs::OpenOptions flags.
    #[inline]
    pub fn to_open_options(&self) -> std::fs::OpenOptions {
        let mut opts = std::fs::OpenOptions::new();

        opts.read(self.read)
            .write(self.write)
            .append(self.append)
            .truncate(self.truncate)
            .create(self.creates_file() && !self.exclusive)
            .create_new(self.exclusive);

        opts
    }
}

impl fmt::Display for FileMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Reconstruct canonical mode string
        if self.exclusive {
            write!(f, "x")?;
        } else if self.append {
            write!(f, "a")?;
        } else if self.truncate {
            write!(f, "w")?;
        } else {
            write!(f, "r")?;
        }

        if self.binary {
            write!(f, "b")?;
        }

        if self.read && self.write && !self.append {
            write!(f, "+")?;
        }

        Ok(())
    }
}

/// Error parsing a file mode string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseModeError {
    /// Mode string was empty.
    Empty,
    /// Invalid character in mode string.
    InvalidCharacter(char),
    /// Invalid mode combination.
    InvalidMode(String),
}

impl fmt::Display for ParseModeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseModeError::Empty => write!(f, "mode string cannot be empty"),
            ParseModeError::InvalidCharacter(c) => {
                write!(f, "invalid mode character: '{}'", c)
            }
            ParseModeError::InvalidMode(m) => {
                write!(f, "invalid mode: '{}'", m)
            }
        }
    }
}

impl std::error::Error for ParseModeError {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Basic Mode Parsing
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_read_mode() {
        let mode = FileMode::parse("r").unwrap();
        assert!(mode.read);
        assert!(!mode.write);
        assert!(!mode.binary);
        assert!(!mode.append);
        assert!(!mode.truncate);
    }

    #[test]
    fn test_parse_write_mode() {
        let mode = FileMode::parse("w").unwrap();
        assert!(!mode.read);
        assert!(mode.write);
        assert!(!mode.binary);
        assert!(mode.truncate);
    }

    #[test]
    fn test_parse_append_mode() {
        let mode = FileMode::parse("a").unwrap();
        assert!(!mode.read);
        assert!(mode.write);
        assert!(mode.append);
        assert!(!mode.truncate);
    }

    #[test]
    fn test_parse_exclusive_mode() {
        let mode = FileMode::parse("x").unwrap();
        assert!(!mode.read);
        assert!(mode.write);
        assert!(mode.exclusive);
    }

    // -------------------------------------------------------------------------
    // Binary Mode
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_read_binary() {
        let mode = FileMode::parse("rb").unwrap();
        assert!(mode.read);
        assert!(mode.binary);
    }

    #[test]
    fn test_parse_write_binary() {
        let mode = FileMode::parse("wb").unwrap();
        assert!(mode.write);
        assert!(mode.binary);
        assert!(mode.truncate);
    }

    #[test]
    fn test_parse_append_binary() {
        let mode = FileMode::parse("ab").unwrap();
        assert!(mode.write);
        assert!(mode.append);
        assert!(mode.binary);
    }

    // -------------------------------------------------------------------------
    // Update Mode (+)
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_read_update() {
        let mode = FileMode::parse("r+").unwrap();
        assert!(mode.read);
        assert!(mode.write);
        assert!(!mode.truncate);
    }

    #[test]
    fn test_parse_write_update() {
        let mode = FileMode::parse("w+").unwrap();
        assert!(mode.read);
        assert!(mode.write);
        assert!(mode.truncate);
    }

    #[test]
    fn test_parse_append_update() {
        let mode = FileMode::parse("a+").unwrap();
        assert!(mode.read);
        assert!(mode.write);
        assert!(mode.append);
    }

    #[test]
    fn test_parse_read_binary_update() {
        let mode = FileMode::parse("rb+").unwrap();
        assert!(mode.read);
        assert!(mode.write);
        assert!(mode.binary);
    }

    #[test]
    fn test_parse_binary_read_update_order() {
        // Order shouldn't matter
        let mode1 = FileMode::parse("r+b").unwrap();
        let mode2 = FileMode::parse("rb+").unwrap();
        assert_eq!(mode1.read, mode2.read);
        assert_eq!(mode1.write, mode2.write);
        assert_eq!(mode1.binary, mode2.binary);
    }

    // -------------------------------------------------------------------------
    // Text Mode Explicit
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_text_mode_explicit() {
        let mode = FileMode::parse("rt").unwrap();
        assert!(mode.read);
        assert!(!mode.binary);
    }

    #[test]
    fn test_parse_write_text_explicit() {
        let mode = FileMode::parse("wt").unwrap();
        assert!(mode.write);
        assert!(!mode.binary);
    }

    // -------------------------------------------------------------------------
    // Error Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_empty_mode() {
        assert!(matches!(FileMode::parse(""), Err(ParseModeError::Empty)));
    }

    #[test]
    fn test_parse_invalid_character() {
        assert!(matches!(
            FileMode::parse("rz"),
            Err(ParseModeError::InvalidCharacter('z'))
        ));
    }

    #[test]
    fn test_parse_duplicate_read() {
        assert!(FileMode::parse("rr").is_err());
    }

    #[test]
    fn test_parse_conflicting_modes() {
        assert!(FileMode::parse("rw").is_err());
        assert!(FileMode::parse("ra").is_err());
        assert!(FileMode::parse("wa").is_err());
    }

    #[test]
    fn test_parse_duplicate_binary() {
        assert!(FileMode::parse("rbb").is_err());
    }

    #[test]
    fn test_parse_conflicting_bt() {
        assert!(FileMode::parse("rbt").is_err());
        assert!(FileMode::parse("rtb").is_err());
    }

    #[test]
    fn test_parse_duplicate_plus() {
        assert!(FileMode::parse("r++").is_err());
    }

    // -------------------------------------------------------------------------
    // Display Round-Trip
    // -------------------------------------------------------------------------

    #[test]
    fn test_display_read() {
        let mode = FileMode::parse("r").unwrap();
        assert_eq!(mode.to_string(), "r");
    }

    #[test]
    fn test_display_write_binary() {
        let mode = FileMode::parse("wb").unwrap();
        assert_eq!(mode.to_string(), "wb");
    }

    #[test]
    fn test_display_append() {
        let mode = FileMode::parse("a").unwrap();
        assert_eq!(mode.to_string(), "a");
    }

    // -------------------------------------------------------------------------
    // Helper Methods
    // -------------------------------------------------------------------------

    #[test]
    fn test_requires_existing() {
        assert!(FileMode::parse("r").unwrap().requires_existing());
        assert!(!FileMode::parse("w").unwrap().requires_existing());
        assert!(!FileMode::parse("r+").unwrap().requires_existing());
    }

    #[test]
    fn test_creates_file() {
        assert!(!FileMode::parse("r").unwrap().creates_file());
        assert!(FileMode::parse("w").unwrap().creates_file());
        assert!(FileMode::parse("a").unwrap().creates_file());
        assert!(FileMode::parse("x").unwrap().creates_file());
    }

    // -------------------------------------------------------------------------
    // OpenOptions Conversion
    // -------------------------------------------------------------------------

    #[test]
    fn test_to_open_options_read() {
        let mode = FileMode::parse("r").unwrap();
        let _opts = mode.to_open_options();
        // OpenOptions doesn't expose getters, so we verify it compiles
    }

    #[test]
    fn test_to_open_options_write() {
        let mode = FileMode::parse("w").unwrap();
        let _opts = mode.to_open_options();
    }

    #[test]
    fn test_to_open_options_exclusive() {
        let mode = FileMode::parse("x").unwrap();
        let _opts = mode.to_open_options();
    }

    // -------------------------------------------------------------------------
    // Edge Cases and Stress Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_all_valid_single_char_modes() {
        for mode in ["r", "w", "a", "x"] {
            assert!(FileMode::parse(mode).is_ok(), "Failed for mode: {}", mode);
        }
    }

    #[test]
    fn test_all_valid_binary_modes() {
        for mode in ["rb", "wb", "ab", "xb"] {
            assert!(FileMode::parse(mode).is_ok(), "Failed for mode: {}", mode);
        }
    }

    #[test]
    fn test_all_update_modes() {
        for mode in ["r+", "w+", "a+", "r+b", "w+b", "a+b", "rb+", "wb+", "ab+"] {
            assert!(FileMode::parse(mode).is_ok(), "Failed for mode: {}", mode);
        }
    }
}
