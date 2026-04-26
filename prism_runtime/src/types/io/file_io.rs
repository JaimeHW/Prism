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
