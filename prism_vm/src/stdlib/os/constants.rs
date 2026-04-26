//! OS module constants.
//!
//! Platform-specific constants for path separators, flags, and identifiers.
//! All constants are compile-time evaluated for zero runtime overhead.

// =============================================================================
// Platform Identification
// =============================================================================

/// Operating system name ("nt" for Windows, "posix" for Unix-like).
#[cfg(windows)]
pub const OS_NAME: &str = "nt";
#[cfg(not(windows))]
pub const OS_NAME: &str = "posix";

// =============================================================================
// Path Separators
// =============================================================================

/// Path component separator ("/" on Unix, "\\" on Windows).
#[cfg(windows)]
pub const SEP: char = '\\';
#[cfg(not(windows))]
pub const SEP: char = '/';

/// Path component separator as string.
#[cfg(windows)]
pub const SEP_STR: &str = "\\";
#[cfg(not(windows))]
pub const SEP_STR: &str = "/";

/// Alternate path separator (Some("/") on Windows, None on Unix).
#[cfg(windows)]
pub const ALTSEP: Option<char> = Some('/');
#[cfg(not(windows))]
pub const ALTSEP: Option<char> = None;

/// Path list separator (":" on Unix, ";" on Windows).
#[cfg(windows)]
pub const PATHSEP: char = ';';
#[cfg(not(windows))]
pub const PATHSEP: char = ':';

/// Path list separator as string.
#[cfg(windows)]
pub const PATHSEP_STR: &str = ";";
#[cfg(not(windows))]
pub const PATHSEP_STR: &str = ":";

/// Line separator ("\r\n" on Windows, "\n" on Unix).
#[cfg(windows)]
pub const LINESEP: &str = "\r\n";
#[cfg(not(windows))]
pub const LINESEP: &str = "\n";

/// Current directory reference.
pub const CURDIR: &str = ".";

/// Parent directory reference.
pub const PARDIR: &str = "..";

/// File extension separator.
pub const EXTSEP: char = '.';

/// Null device path.
#[cfg(windows)]
pub const DEVNULL: &str = "nul";
#[cfg(not(windows))]
pub const DEVNULL: &str = "/dev/null";

// =============================================================================
// File Open Flags (O_*)
// =============================================================================

/// Open for reading only.
pub const O_RDONLY: u32 = 0x0000;

/// Open for writing only.
pub const O_WRONLY: u32 = 0x0001;

/// Open for reading and writing.
pub const O_RDWR: u32 = 0x0002;

/// Create file if it doesn't exist.
#[cfg(windows)]
pub const O_CREAT: u32 = 0x0100;
#[cfg(not(windows))]
pub const O_CREAT: u32 = 0x0040;

/// Truncate file to zero length.
#[cfg(windows)]
pub const O_TRUNC: u32 = 0x0200;
#[cfg(not(windows))]
pub const O_TRUNC: u32 = 0x0200;

/// Append to file.
#[cfg(windows)]
pub const O_APPEND: u32 = 0x0008;
#[cfg(not(windows))]
pub const O_APPEND: u32 = 0x0400;

/// Exclusive create (fail if exists).
#[cfg(windows)]
pub const O_EXCL: u32 = 0x0400;
#[cfg(not(windows))]
pub const O_EXCL: u32 = 0x0080;

/// Binary mode (Windows only, no-op on Unix).
#[cfg(windows)]
pub const O_BINARY: u32 = 0x8000;
#[cfg(not(windows))]
pub const O_BINARY: u32 = 0x0000;

/// Text mode (Windows only, no-op on Unix).
#[cfg(windows)]
pub const O_TEXT: u32 = 0x4000;
#[cfg(not(windows))]
pub const O_TEXT: u32 = 0x0000;

// =============================================================================
// Access Mode Flags
// =============================================================================

/// Test for existence.
pub const F_OK: u32 = 0;

/// Test for read permission.
pub const R_OK: u32 = 4;

/// Test for write permission.
pub const W_OK: u32 = 2;

/// Test for execute permission.
pub const X_OK: u32 = 1;

// =============================================================================
// Seek Whence Constants
// =============================================================================

/// Seek from beginning of file.
pub const SEEK_SET: u32 = 0;

/// Seek from current position.
pub const SEEK_CUR: u32 = 1;

/// Seek from end of file.
pub const SEEK_END: u32 = 2;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
