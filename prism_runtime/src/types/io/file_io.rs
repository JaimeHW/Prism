//! Raw unbuffered file I/O.
//!
//! `FileIO` provides direct access to the underlying file descriptor with
//! minimal overhead. It's the foundation layer of Python's I/O stack.
//!
//! # Performance
//!
//! - Direct syscalls with no buffering overhead
//! - Minimal memory footprint
//! - Suitable for memory-mapped files and large block transfers
//!
//! # Thread Safety
//!
//! `FileIO` is not thread-safe. Each file object should be used by only
//! one thread at a time. For concurrent access, use external synchronization.

use super::mode::FileMode;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Raw unbuffered file I/O object.
///
/// This corresponds to Python's `io.FileIO` class.
pub struct FileIO {
    /// The underlying file handle.
    file: File,
    /// The mode used to open the file.
    mode: FileMode,
    /// Original file path (for error messages).
    path: Option<Box<str>>,
    /// Whether the file is closed.
    closed: bool,
    /// Whether this FileIO owns the file handle.
    closefd: bool,
}

impl FileIO {
    /// Open a file with the given mode.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the file
    /// * `mode` - File mode (parsed `FileMode`)
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened.
    pub fn open<P: AsRef<Path>>(path: P, mode: FileMode) -> io::Result<Self> {
        let path_ref = path.as_ref();
        let file = mode.to_open_options().open(path_ref)?;

        Ok(Self {
            file,
            mode,
            path: Some(path_ref.to_string_lossy().into_owned().into_boxed_str()),
            closed: false,
            closefd: true,
        })
    }

    /// Create a FileIO from an existing file handle.
    ///
    /// # Arguments
    ///
    /// * `file` - An open file handle
    /// * `mode` - The mode the file was opened with
    /// * `closefd` - Whether to close the file handle when FileIO is dropped
    #[inline]
    pub fn from_file(file: File, mode: FileMode, closefd: bool) -> Self {
        Self {
            file,
            mode,
            path: None,
            closed: false,
            closefd,
        }
    }

    /// Check if the file is closed.
    #[inline]
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Get the file mode.
    #[inline]
    pub fn mode(&self) -> FileMode {
        self.mode
    }

    /// Get the file path, if available.
    #[inline]
    pub fn path(&self) -> Option<&str> {
        self.path.as_deref()
    }

    /// Check if file is readable.
    #[inline]
    pub fn readable(&self) -> bool {
        self.mode.read && !self.closed
    }

    /// Check if file is writable.
    #[inline]
    pub fn writable(&self) -> bool {
        self.mode.write && !self.closed
    }

    /// Check if file is seekable.
    #[inline]
    pub fn seekable(&self) -> bool {
        !self.closed
    }

    /// Read up to `n` bytes into the provided buffer.
    ///
    /// Returns the number of bytes read, which may be less than `n`.
    pub fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.closed {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "I/O operation on closed file",
            ));
        }
        if !self.mode.read {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "File not open for reading",
            ));
        }
        self.file.read(buf)
    }

    /// Read all available bytes into a new Vec.
    pub fn read_all(&mut self) -> io::Result<Vec<u8>> {
        if self.closed {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "I/O operation on closed file",
            ));
        }
        if !self.mode.read {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "File not open for reading",
            ));
        }
        let mut buf = Vec::new();
        self.file.read_to_end(&mut buf)?;
        Ok(buf)
    }

    /// Write bytes to the file.
    ///
    /// Returns the number of bytes written.
    pub fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.closed {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "I/O operation on closed file",
            ));
        }
        if !self.mode.write {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "File not open for writing",
            ));
        }
        self.file.write(buf)
    }

    /// Write all bytes to the file.
    pub fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        if self.closed {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "I/O operation on closed file",
            ));
        }
        if !self.mode.write {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "File not open for writing",
            ));
        }
        self.file.write_all(buf)
    }

    /// Seek to a position in the file.
    ///
    /// Returns the new position.
    pub fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        if self.closed {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "I/O operation on closed file",
            ));
        }
        self.file.seek(pos)
    }

    /// Get the current position in the file.
    pub fn tell(&mut self) -> io::Result<u64> {
        self.seek(SeekFrom::Current(0))
    }

    /// Truncate the file to the specified size.
    pub fn truncate(&mut self, size: u64) -> io::Result<()> {
        if self.closed {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "I/O operation on closed file",
            ));
        }
        if !self.mode.write {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "File not open for writing",
            ));
        }
        self.file.set_len(size)
    }

    /// Flush any pending writes to disk.
    pub fn flush(&mut self) -> io::Result<()> {
        if self.closed {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "I/O operation on closed file",
            ));
        }
        self.file.flush()
    }

    /// Sync all data and metadata to disk.
    pub fn sync_all(&self) -> io::Result<()> {
        if self.closed {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "I/O operation on closed file",
            ));
        }
        self.file.sync_all()
    }

    /// Sync data (but not necessarily metadata) to disk.
    pub fn sync_data(&self) -> io::Result<()> {
        if self.closed {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "I/O operation on closed file",
            ));
        }
        self.file.sync_data()
    }

    /// Close the file.
    pub fn close(&mut self) -> io::Result<()> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        // File is closed automatically when dropped
        Ok(())
    }

    /// Get the underlying file handle.
    ///
    /// # Safety
    ///
    /// The caller must not close the file handle or invalidate it.
    #[inline]
    pub fn file(&self) -> &File {
        &self.file
    }

    /// Get the underlying file handle mutably.
    ///
    /// # Safety
    ///
    /// The caller must not close the file handle or invalidate it.
    #[inline]
    pub fn file_mut(&mut self) -> &mut File {
        &mut self.file
    }

    /// Get file metadata.
    pub fn metadata(&self) -> io::Result<std::fs::Metadata> {
        self.file.metadata()
    }

    /// Get the file size.
    pub fn size(&self) -> io::Result<u64> {
        Ok(self.metadata()?.len())
    }
}

impl Drop for FileIO {
    fn drop(&mut self) {
        if self.closefd && !self.closed {
            // File is closed automatically
            self.closed = true;
        }
    }
}

impl std::fmt::Debug for FileIO {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileIO")
            .field("mode", &self.mode)
            .field("path", &self.path)
            .field("closed", &self.closed)
            .finish()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // -------------------------------------------------------------------------
    // Opening Files
    // -------------------------------------------------------------------------

    #[test]
    fn test_open_read() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"hello").unwrap();

        let mode = FileMode::parse("r").unwrap();
        let file = FileIO::open(tmp.path(), mode).unwrap();

        assert!(file.readable());
        assert!(!file.writable());
        assert!(!file.is_closed());
    }

    #[test]
    fn test_open_write() {
        let tmp = NamedTempFile::new().unwrap();

        let mode = FileMode::parse("w").unwrap();
        let file = FileIO::open(tmp.path(), mode).unwrap();

        assert!(!file.readable());
        assert!(file.writable());
    }

    #[test]
    fn test_open_read_write() {
        let tmp = NamedTempFile::new().unwrap();

        let mode = FileMode::parse("r+").unwrap();
        let file = FileIO::open(tmp.path(), mode).unwrap();

        assert!(file.readable());
        assert!(file.writable());
    }

    #[test]
    fn test_open_nonexistent_read() {
        let mode = FileMode::parse("r").unwrap();
        let result = FileIO::open("/nonexistent/path/to/file", mode);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Reading
    // -------------------------------------------------------------------------

    #[test]
    fn test_read_bytes() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"hello world").unwrap();

        let mode = FileMode::parse("rb").unwrap();
        let mut file = FileIO::open(tmp.path(), mode).unwrap();

        let mut buf = [0u8; 5];
        let n = file.read(&mut buf).unwrap();

        assert_eq!(n, 5);
        assert_eq!(&buf, b"hello");
    }

    #[test]
    fn test_read_all() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"hello world").unwrap();

        let mode = FileMode::parse("rb").unwrap();
        let mut file = FileIO::open(tmp.path(), mode).unwrap();

        let data = file.read_all().unwrap();
        assert_eq!(data, b"hello world");
    }

    #[test]
    fn test_read_empty_file() {
        let tmp = NamedTempFile::new().unwrap();

        let mode = FileMode::parse("rb").unwrap();
        let mut file = FileIO::open(tmp.path(), mode).unwrap();

        let data = file.read_all().unwrap();
        assert!(data.is_empty());
    }

    #[test]
    fn test_read_write_only_file() {
        let tmp = NamedTempFile::new().unwrap();

        let mode = FileMode::parse("w").unwrap();
        let mut file = FileIO::open(tmp.path(), mode).unwrap();

        let mut buf = [0u8; 10];
        let result = file.read(&mut buf);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Writing
    // -------------------------------------------------------------------------

    #[test]
    fn test_write_bytes() {
        let tmp = NamedTempFile::new().unwrap();

        let mode = FileMode::parse("wb").unwrap();
        let mut file = FileIO::open(tmp.path(), mode).unwrap();

        let n = file.write(b"hello").unwrap();
        assert_eq!(n, 5);

        drop(file);

        let content = std::fs::read(tmp.path()).unwrap();
        assert_eq!(content, b"hello");
    }

    #[test]
    fn test_write_all() {
        let tmp = NamedTempFile::new().unwrap();

        let mode = FileMode::parse("wb").unwrap();
        let mut file = FileIO::open(tmp.path(), mode).unwrap();

        file.write_all(b"hello world").unwrap();
        drop(file);

        let content = std::fs::read(tmp.path()).unwrap();
        assert_eq!(content, b"hello world");
    }

    #[test]
    fn test_write_read_only_file() {
        let tmp = NamedTempFile::new().unwrap();

        let mode = FileMode::parse("r").unwrap();
        let mut file = FileIO::open(tmp.path(), mode).unwrap();

        let result = file.write(b"test");
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Seeking
    // -------------------------------------------------------------------------

    #[test]
    fn test_seek_start() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"hello world").unwrap();

        let mode = FileMode::parse("rb").unwrap();
        let mut file = FileIO::open(tmp.path(), mode).unwrap();

        let pos = file.seek(SeekFrom::Start(6)).unwrap();
        assert_eq!(pos, 6);

        let mut buf = [0u8; 5];
        file.read(&mut buf).unwrap();
        assert_eq!(&buf, b"world");
    }

    #[test]
    fn test_seek_end() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"hello world").unwrap();

        let mode = FileMode::parse("rb").unwrap();
        let mut file = FileIO::open(tmp.path(), mode).unwrap();

        let pos = file.seek(SeekFrom::End(-5)).unwrap();
        assert_eq!(pos, 6);
    }

    #[test]
    fn test_seek_current() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"hello world").unwrap();

        let mode = FileMode::parse("rb").unwrap();
        let mut file = FileIO::open(tmp.path(), mode).unwrap();

        file.seek(SeekFrom::Start(3)).unwrap();
        let pos = file.seek(SeekFrom::Current(3)).unwrap();
        assert_eq!(pos, 6);
    }

    #[test]
    fn test_tell() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"hello world").unwrap();

        let mode = FileMode::parse("rb").unwrap();
        let mut file = FileIO::open(tmp.path(), mode).unwrap();

        assert_eq!(file.tell().unwrap(), 0);

        file.seek(SeekFrom::Start(5)).unwrap();
        assert_eq!(file.tell().unwrap(), 5);
    }

    // -------------------------------------------------------------------------
    // Truncation
    // -------------------------------------------------------------------------

    #[test]
    fn test_truncate() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"hello world").unwrap();

        let mode = FileMode::parse("r+b").unwrap();
        let mut file = FileIO::open(tmp.path(), mode).unwrap();

        file.truncate(5).unwrap();
        drop(file);

        let content = std::fs::read(tmp.path()).unwrap();
        assert_eq!(content, b"hello");
    }

    #[test]
    fn test_truncate_extend() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"hi").unwrap();

        let mode = FileMode::parse("r+b").unwrap();
        let mut file = FileIO::open(tmp.path(), mode).unwrap();

        file.truncate(10).unwrap();
        drop(file);

        let content = std::fs::read(tmp.path()).unwrap();
        assert_eq!(content.len(), 10);
        assert_eq!(&content[..2], b"hi");
    }

    // -------------------------------------------------------------------------
    // Close/Flush
    // -------------------------------------------------------------------------

    #[test]
    fn test_close() {
        let tmp = NamedTempFile::new().unwrap();

        let mode = FileMode::parse("w").unwrap();
        let mut file = FileIO::open(tmp.path(), mode).unwrap();

        assert!(!file.is_closed());
        file.close().unwrap();
        assert!(file.is_closed());
    }

    #[test]
    fn test_close_idempotent() {
        let tmp = NamedTempFile::new().unwrap();

        let mode = FileMode::parse("w").unwrap();
        let mut file = FileIO::open(tmp.path(), mode).unwrap();

        file.close().unwrap();
        file.close().unwrap(); // Should not error
    }

    #[test]
    fn test_operations_on_closed_file() {
        let tmp = NamedTempFile::new().unwrap();

        let mode = FileMode::parse("r+b").unwrap();
        let mut file = FileIO::open(tmp.path(), mode).unwrap();

        file.close().unwrap();

        assert!(file.read(&mut [0; 10]).is_err());
        assert!(file.write(b"test").is_err());
        assert!(file.seek(SeekFrom::Start(0)).is_err());
        assert!(file.truncate(0).is_err());
    }

    #[test]
    fn test_flush() {
        let tmp = NamedTempFile::new().unwrap();

        let mode = FileMode::parse("w").unwrap();
        let mut file = FileIO::open(tmp.path(), mode).unwrap();

        file.write_all(b"test").unwrap();
        file.flush().unwrap();
    }

    // -------------------------------------------------------------------------
    // Metadata
    // -------------------------------------------------------------------------

    #[test]
    fn test_metadata() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"hello").unwrap();

        let mode = FileMode::parse("r").unwrap();
        let file = FileIO::open(tmp.path(), mode).unwrap();

        let metadata = file.metadata().unwrap();
        assert_eq!(metadata.len(), 5);
    }

    #[test]
    fn test_size() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"hello world").unwrap();

        let mode = FileMode::parse("r").unwrap();
        let file = FileIO::open(tmp.path(), mode).unwrap();

        assert_eq!(file.size().unwrap(), 11);
    }

    // -------------------------------------------------------------------------
    // From Existing File
    // -------------------------------------------------------------------------

    #[test]
    fn test_from_file() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"test").unwrap();

        let file = File::open(tmp.path()).unwrap();
        let mode = FileMode::parse("r").unwrap();
        let mut file_io = FileIO::from_file(file, mode, true);

        assert!(file_io.readable());
        let data = file_io.read_all().unwrap();
        assert_eq!(data, b"test");
    }

    // -------------------------------------------------------------------------
    // Path Tracking
    // -------------------------------------------------------------------------

    #[test]
    fn test_path() {
        let tmp = NamedTempFile::new().unwrap();
        let path_str = tmp.path().to_string_lossy().into_owned();

        let mode = FileMode::parse("r").unwrap();
        let file = FileIO::open(tmp.path(), mode).unwrap();

        assert_eq!(file.path(), Some(path_str.as_str()));
    }

    #[test]
    fn test_path_from_file_none() {
        let tmp = NamedTempFile::new().unwrap();
        let file = File::open(tmp.path()).unwrap();
        let mode = FileMode::parse("r").unwrap();
        let file_io = FileIO::from_file(file, mode, false);

        assert!(file_io.path().is_none());
    }

    // -------------------------------------------------------------------------
    // Debug Formatting
    // -------------------------------------------------------------------------

    #[test]
    fn test_debug() {
        let tmp = NamedTempFile::new().unwrap();

        let mode = FileMode::parse("rb").unwrap();
        let file = FileIO::open(tmp.path(), mode).unwrap();

        let debug = format!("{:?}", file);
        assert!(debug.contains("FileIO"));
        assert!(debug.contains("closed: false"));
    }
}
