//! File system operations.

use super::error::OsError;
use std::fs;
use std::os::windows::fs::MetadataExt;
use std::path::Path;

/// File/directory stat result.
#[derive(Debug, Clone)]
pub struct StatResult {
    pub st_mode: u32,
    pub st_ino: u64,
    pub st_dev: u64,
    pub st_nlink: u64,
    pub st_uid: u32,
    pub st_gid: u32,
    pub st_size: u64,
    pub st_atime: i64,
    pub st_mtime: i64,
    pub st_ctime: i64,
}

impl StatResult {
    /// Create from fs::Metadata.
    #[cfg(windows)]
    pub fn from_metadata(m: &fs::Metadata) -> Self {
        let atime = m.last_access_time() as i64 / 10_000_000 - 11_644_473_600;
        let mtime = m.last_write_time() as i64 / 10_000_000 - 11_644_473_600;
        let ctime = m.creation_time() as i64 / 10_000_000 - 11_644_473_600;
        Self {
            st_mode: if m.is_dir() { 0o40755 } else { 0o100644 },
            st_ino: 0,
            st_dev: 0,
            st_nlink: 1,
            st_uid: 0,
            st_gid: 0,
            st_size: m.len(),
            st_atime: atime,
            st_mtime: mtime,
            st_ctime: ctime,
        }
    }

    #[cfg(not(windows))]
    pub fn from_metadata(m: &fs::Metadata) -> Self {
        use std::os::unix::fs::MetadataExt;
        Self {
            st_mode: m.mode(),
            st_ino: m.ino(),
            st_dev: m.dev(),
            st_nlink: m.nlink(),
            st_uid: m.uid(),
            st_gid: m.gid(),
            st_size: m.len(),
            st_atime: m.atime(),
            st_mtime: m.mtime(),
            st_ctime: m.ctime(),
        }
    }

    #[inline]
    pub fn is_dir(&self) -> bool {
        (self.st_mode & 0o170000) == 0o40000
    }
    #[inline]
    pub fn is_file(&self) -> bool {
        (self.st_mode & 0o170000) == 0o100000
    }
    #[inline]
    pub fn is_link(&self) -> bool {
        (self.st_mode & 0o170000) == 0o120000
    }
}

/// Get file/directory status.
pub fn stat<P: AsRef<Path>>(path: P) -> Result<StatResult, OsError> {
    let path = path.as_ref();
    fs::metadata(path)
        .map(|m| StatResult::from_metadata(&m))
        .map_err(|e| OsError::from_io_error(&e, path.to_string_lossy().as_ref()))
}

/// Get symlink status (don't follow links).
pub fn lstat<P: AsRef<Path>>(path: P) -> Result<StatResult, OsError> {
    let path = path.as_ref();
    fs::symlink_metadata(path)
        .map(|m| StatResult::from_metadata(&m))
        .map_err(|e| OsError::from_io_error(&e, path.to_string_lossy().as_ref()))
}

/// Create a directory.
pub fn mkdir<P: AsRef<Path>>(path: P) -> Result<(), OsError> {
    let path = path.as_ref();
    fs::create_dir(path).map_err(|e| OsError::from_io_error(&e, path.to_string_lossy().as_ref()))
}

/// Create directories recursively.
pub fn makedirs<P: AsRef<Path>>(path: P) -> Result<(), OsError> {
    let path = path.as_ref();
    fs::create_dir_all(path)
        .map_err(|e| OsError::from_io_error(&e, path.to_string_lossy().as_ref()))
}

/// Remove a directory.
pub fn rmdir<P: AsRef<Path>>(path: P) -> Result<(), OsError> {
    let path = path.as_ref();
    fs::remove_dir(path).map_err(|e| OsError::from_io_error(&e, path.to_string_lossy().as_ref()))
}

/// Remove a file.
pub fn remove<P: AsRef<Path>>(path: P) -> Result<(), OsError> {
    let path = path.as_ref();
    fs::remove_file(path).map_err(|e| OsError::from_io_error(&e, path.to_string_lossy().as_ref()))
}

/// Rename a file or directory.
pub fn rename<P: AsRef<Path>, Q: AsRef<Path>>(src: P, dst: Q) -> Result<(), OsError> {
    let src = src.as_ref();
    let dst = dst.as_ref();
    fs::rename(src, dst).map_err(|e| OsError::from_io_error(&e, src.to_string_lossy().as_ref()))
}

/// List directory contents.
pub fn listdir<P: AsRef<Path>>(path: P) -> Result<Vec<String>, OsError> {
    let path = path.as_ref();
    let entries = fs::read_dir(path)
        .map_err(|e| OsError::from_io_error(&e, path.to_string_lossy().as_ref()))?;
    let mut names = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| OsError::from_io_error(&e, ""))?;
        names.push(entry.file_name().to_string_lossy().into_owned());
    }
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_stat_temp_dir() {
        let s = stat(env::temp_dir()).unwrap();
        assert!(s.is_dir());
    }

    #[test]
    fn test_stat_nonexistent() {
        assert!(stat("/nonexistent_12345").is_err());
    }

    #[test]
    fn test_mkdir_rmdir() {
        let dir = env::temp_dir().join("_test_mkdir_12345");
        let _ = rmdir(&dir);
        mkdir(&dir).unwrap();
        assert!(stat(&dir).unwrap().is_dir());
        rmdir(&dir).unwrap();
    }

    #[test]
    fn test_listdir() {
        let entries = listdir(env::temp_dir()).unwrap();
        assert!(entries.len() >= 0); // May be empty but shouldn't error
    }
}
