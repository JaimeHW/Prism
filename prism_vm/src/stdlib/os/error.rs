//! OS error handling with errno/GetLastError mapping.

use std::fmt;
use std::io;

/// OS-level error with errno information.
#[derive(Debug, Clone)]
pub struct OsError {
    pub code: i32,
    pub message: String,
    pub path: Option<String>,
}

impl OsError {
    /// Create from raw error code and message.
    #[inline]
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            path: None,
        }
    }

    /// Create with associated path.
    #[inline]
    pub fn with_path(code: i32, message: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            path: Some(path.into()),
        }
    }

    /// Create from std::io::Error.
    pub fn from_io_error(e: &io::Error, path: &str) -> Self {
        let code = e.raw_os_error().unwrap_or(-1);
        Self {
            code,
            message: e.to_string(),
            path: if path.is_empty() {
                None
            } else {
                Some(path.to_string())
            },
        }
    }

    /// Create FileNotFoundError.
    #[inline]
    pub fn file_not_found(path: &str) -> Self {
        Self::with_path(2, "No such file or directory", path)
    }

    /// Create PermissionError.
    #[inline]
    pub fn permission_denied(path: &str) -> Self {
        Self::with_path(13, "Permission denied", path)
    }

    /// Create FileExistsError.
    #[inline]
    pub fn file_exists(path: &str) -> Self {
        Self::with_path(17, "File exists", path)
    }

    /// Create IsADirectoryError.
    #[inline]
    pub fn is_a_directory(path: &str) -> Self {
        Self::with_path(21, "Is a directory", path)
    }

    /// Create NotADirectoryError.
    #[inline]
    pub fn not_a_directory(path: &str) -> Self {
        Self::with_path(20, "Not a directory", path)
    }
}

impl fmt::Display for OsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref path) = self.path {
            write!(f, "[Errno {}] {}: '{}'", self.code, self.message, path)
        } else {
            write!(f, "[Errno {}] {}", self.code, self.message)
        }
    }
}

impl std::error::Error for OsError {}

impl From<io::Error> for OsError {
    fn from(e: io::Error) -> Self {
        Self::from_io_error(&e, "")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let e = OsError::new(2, "Not found");
        assert_eq!(e.code, 2);
        assert_eq!(e.message, "Not found");
        assert!(e.path.is_none());
    }

    #[test]
    fn test_with_path() {
        let e = OsError::with_path(2, "Not found", "/foo");
        assert_eq!(e.path, Some("/foo".to_string()));
    }

    #[test]
    fn test_file_not_found() {
        let e = OsError::file_not_found("/missing");
        assert_eq!(e.code, 2);
        assert!(e.to_string().contains("/missing"));
    }

    #[test]
    fn test_permission_denied() {
        let e = OsError::permission_denied("/secret");
        assert_eq!(e.code, 13);
    }

    #[test]
    fn test_display_with_path() {
        let e = OsError::with_path(2, "Not found", "/foo");
        let s = e.to_string();
        assert!(s.contains("Errno 2"));
        assert!(s.contains("/foo"));
    }

    #[test]
    fn test_display_without_path() {
        let e = OsError::new(1, "Error");
        let s = e.to_string();
        assert!(s.contains("Errno 1"));
        assert!(!s.contains("'"));
    }

    #[test]
    fn test_from_io_error() {
        let io_e = io::Error::new(io::ErrorKind::NotFound, "not found");
        let e = OsError::from_io_error(&io_e, "/path");
        assert!(e.path.is_some());
    }
}
