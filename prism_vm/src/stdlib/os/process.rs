//! Process-related operations.

use std::process;

/// Get the current process ID.
#[inline]
pub fn getpid() -> u32 {
    process::id()
}

/// Get the parent process ID.
#[cfg(windows)]
pub fn getppid() -> u32 {
    // Windows doesn't have direct getppid, use 0 as placeholder
    0
}

#[cfg(not(windows))]
pub fn getppid() -> u32 {
    unsafe { libc::getppid() as u32 }
}

/// Exit the process with given status code.
pub fn exit(code: i32) -> ! {
    process::exit(code)
}

/// Get the process name/title.
pub fn getprocessname() -> Option<String> {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.file_name().map(|s| s.to_string_lossy().into_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_getpid_nonzero() {
        assert!(getpid() > 0);
    }

    #[test]
    fn test_getpid_consistent() {
        let p1 = getpid();
        let p2 = getpid();
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_getppid() {
        // Just ensure it doesn't panic
        let _ = getppid();
    }

    #[test]
    fn test_getprocessname() {
        let name = getprocessname();
        // Should have some name
        assert!(name.is_some());
    }
}
