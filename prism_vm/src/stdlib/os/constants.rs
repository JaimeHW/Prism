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
mod tests {
    use super::*;

    // =========================================================================
    // Platform Constants Tests
    // =========================================================================

    #[test]
    fn test_os_name_valid() {
        assert!(OS_NAME == "nt" || OS_NAME == "posix");
    }

    #[cfg(windows)]
    #[test]
    fn test_os_name_windows() {
        assert_eq!(OS_NAME, "nt");
    }

    #[cfg(not(windows))]
    #[test]
    fn test_os_name_posix() {
        assert_eq!(OS_NAME, "posix");
    }

    // =========================================================================
    // Path Separator Tests
    // =========================================================================

    #[cfg(windows)]
    #[test]
    fn test_sep_windows() {
        assert_eq!(SEP, '\\');
        assert_eq!(SEP_STR, "\\");
    }

    #[cfg(not(windows))]
    #[test]
    fn test_sep_unix() {
        assert_eq!(SEP, '/');
        assert_eq!(SEP_STR, "/");
    }

    #[cfg(windows)]
    #[test]
    fn test_altsep_windows() {
        assert_eq!(ALTSEP, Some('/'));
    }

    #[cfg(not(windows))]
    #[test]
    fn test_altsep_unix() {
        assert_eq!(ALTSEP, None);
    }

    #[cfg(windows)]
    #[test]
    fn test_pathsep_windows() {
        assert_eq!(PATHSEP, ';');
        assert_eq!(PATHSEP_STR, ";");
    }

    #[cfg(not(windows))]
    #[test]
    fn test_pathsep_unix() {
        assert_eq!(PATHSEP, ':');
        assert_eq!(PATHSEP_STR, ":");
    }

    #[cfg(windows)]
    #[test]
    fn test_linesep_windows() {
        assert_eq!(LINESEP, "\r\n");
    }

    #[cfg(not(windows))]
    #[test]
    fn test_linesep_unix() {
        assert_eq!(LINESEP, "\n");
    }

    // =========================================================================
    // Directory Reference Tests
    // =========================================================================

    #[test]
    fn test_curdir() {
        assert_eq!(CURDIR, ".");
    }

    #[test]
    fn test_pardir() {
        assert_eq!(PARDIR, "..");
    }

    #[test]
    fn test_extsep() {
        assert_eq!(EXTSEP, '.');
    }

    // =========================================================================
    // Devnull Tests
    // =========================================================================

    #[cfg(windows)]
    #[test]
    fn test_devnull_windows() {
        assert_eq!(DEVNULL, "nul");
    }

    #[cfg(not(windows))]
    #[test]
    fn test_devnull_unix() {
        assert_eq!(DEVNULL, "/dev/null");
    }

    // =========================================================================
    // O_* Flag Tests
    // =========================================================================

    #[test]
    fn test_o_rdonly() {
        assert_eq!(O_RDONLY, 0);
    }

    #[test]
    fn test_o_wronly() {
        assert_eq!(O_WRONLY, 1);
    }

    #[test]
    fn test_o_rdwr() {
        assert_eq!(O_RDWR, 2);
    }

    #[test]
    fn test_o_creat_nonzero() {
        assert!(O_CREAT > 0);
    }

    #[test]
    fn test_o_trunc_nonzero() {
        assert!(O_TRUNC > 0);
    }

    #[test]
    fn test_o_append_nonzero() {
        assert!(O_APPEND > 0);
    }

    #[test]
    fn test_o_excl_nonzero() {
        assert!(O_EXCL > 0);
    }

    #[test]
    fn test_o_flags_distinct() {
        // Ensure common flags don't overlap (except RDONLY which is 0)
        assert_ne!(O_WRONLY, O_RDWR);
        assert_ne!(O_CREAT, O_TRUNC);
        assert_ne!(O_APPEND, O_EXCL);
    }

    // =========================================================================
    // Access Mode Tests
    // =========================================================================

    #[test]
    fn test_f_ok() {
        assert_eq!(F_OK, 0);
    }

    #[test]
    fn test_r_ok() {
        assert_eq!(R_OK, 4);
    }

    #[test]
    fn test_w_ok() {
        assert_eq!(W_OK, 2);
    }

    #[test]
    fn test_x_ok() {
        assert_eq!(X_OK, 1);
    }

    #[test]
    fn test_access_flags_can_combine() {
        let combined = R_OK | W_OK | X_OK;
        assert_eq!(combined, 7);
    }

    // =========================================================================
    // Seek Whence Tests
    // =========================================================================

    #[test]
    fn test_seek_set() {
        assert_eq!(SEEK_SET, 0);
    }

    #[test]
    fn test_seek_cur() {
        assert_eq!(SEEK_CUR, 1);
    }

    #[test]
    fn test_seek_end() {
        assert_eq!(SEEK_END, 2);
    }

    #[test]
    fn test_seek_whence_distinct() {
        assert_ne!(SEEK_SET, SEEK_CUR);
        assert_ne!(SEEK_CUR, SEEK_END);
        assert_ne!(SEEK_SET, SEEK_END);
    }
}
