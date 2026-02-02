//! Path query operations (exists, isfile, isdir, islink).

use std::fs;
use std::path::Path;

/// Check if path exists.
#[inline]
pub fn exists<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().exists()
}

/// Check if path is a file.
#[inline]
pub fn isfile<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().is_file()
}

/// Check if path is a directory.
#[inline]
pub fn isdir<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().is_dir()
}

/// Check if path is a symbolic link.
#[inline]
pub fn islink<P: AsRef<Path>>(path: P) -> bool {
    fs::symlink_metadata(path.as_ref())
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false)
}

/// Check if path is absolute.
#[inline]
pub fn isabs<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().is_absolute()
}

/// Get file size, or None if not accessible.
#[inline]
pub fn getsize<P: AsRef<Path>>(path: P) -> Option<u64> {
    fs::metadata(path.as_ref()).ok().map(|m| m.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_exists_curdir() {
        assert!(exists("."));
    }

    #[test]
    fn test_exists_nonexistent() {
        assert!(!exists("/nonexistent_12345"));
    }

    #[test]
    fn test_isfile_on_dir() {
        assert!(!isfile(env::temp_dir()));
    }

    #[test]
    fn test_isdir_temp() {
        assert!(isdir(env::temp_dir()));
    }

    #[test]
    fn test_isdir_on_file() {
        // Current exe is a file
        let exe = env::current_exe().unwrap();
        assert!(!isdir(exe));
    }

    #[test]
    fn test_isabs_absolute() {
        #[cfg(windows)]
        assert!(isabs("C:\\Windows"));
        #[cfg(not(windows))]
        assert!(isabs("/usr"));
    }

    #[test]
    fn test_isabs_relative() {
        assert!(!isabs("foo/bar"));
    }

    #[test]
    fn test_getsize_some() {
        let exe = env::current_exe().unwrap();
        let size = getsize(exe);
        assert!(size.is_some());
        assert!(size.unwrap() > 0);
    }

    #[test]
    fn test_getsize_none() {
        assert!(getsize("/nonexistent_12345").is_none());
    }
}
