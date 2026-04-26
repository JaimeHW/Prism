//! Path timestamp access operations.
//!
//! High-performance timestamp queries using platform-specific APIs for
//! maximum precision. Returns UNIX timestamps as `f64` for sub-second accuracy.
//!
//! # Performance
//!
//! - Single syscall per query via `fs::metadata()`
//! - Platform-native timestamp extraction (no conversion overhead)
//! - `#[inline]` for zero-overhead abstraction

use std::fs;
use std::path::Path;
use std::time::SystemTime;

/// Get modification time as UNIX timestamp (seconds since epoch).
///
/// Equivalent to Python's `os.path.getmtime()`.
///
/// # Platform behavior
///
/// - **Windows**: Uses `last_write_time()` from WIN32_FILE_ATTRIBUTE_DATA
/// - **Unix**: Uses `st_mtime` + `st_mtime_nsec` from `stat(2)`
#[inline]
pub fn getmtime<P: AsRef<Path>>(path: P) -> std::io::Result<f64> {
    let metadata = fs::metadata(path.as_ref())?;
    system_time_to_epoch(metadata.modified()?)
}

/// Get last access time as UNIX timestamp (seconds since epoch).
///
/// Equivalent to Python's `os.path.getatime()`.
///
/// # Note
///
/// Many filesystems have `noatime` mount option which disables access time
/// tracking for performance. In such cases, the access time may equal the
/// modification time or remain at creation time.
#[inline]
pub fn getatime<P: AsRef<Path>>(path: P) -> std::io::Result<f64> {
    let metadata = fs::metadata(path.as_ref())?;
    system_time_to_epoch(metadata.accessed()?)
}

/// Get creation time (Windows) or metadata change time (Unix) as UNIX timestamp.
///
/// Equivalent to Python's `os.path.getctime()`.
///
/// # Platform behavior
///
/// - **Windows**: Returns file creation time (birth time)
/// - **Unix**: Returns metadata change time (`st_ctime`), which is updated
///   when file metadata (permissions, owner, etc.) changes
#[inline]
pub fn getctime<P: AsRef<Path>>(path: P) -> std::io::Result<f64> {
    let metadata = fs::metadata(path.as_ref())?;
    // `created()` can fail on some Unix systems that don't support birth time.
    // Fall back to modified() if created() is unavailable.
    match metadata.created() {
        Ok(t) => system_time_to_epoch(t),
        Err(_) => system_time_to_epoch(metadata.modified()?),
    }
}

/// Get modification time for a symlink itself (not its target).
///
/// Equivalent to calling `os.lstat(path).st_mtime`.
#[inline]
pub fn lgetmtime<P: AsRef<Path>>(path: P) -> std::io::Result<f64> {
    let metadata = fs::symlink_metadata(path.as_ref())?;
    system_time_to_epoch(metadata.modified()?)
}

/// Convert `SystemTime` to UNIX epoch seconds with sub-second precision.
///
/// # Performance
///
/// This is a pure arithmetic operation — no syscalls, no allocations.
/// Uses `duration_since(UNIX_EPOCH)` which is a simple subtraction on
/// all supported platforms.
#[inline]
fn system_time_to_epoch(time: SystemTime) -> std::io::Result<f64> {
    time.duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}
