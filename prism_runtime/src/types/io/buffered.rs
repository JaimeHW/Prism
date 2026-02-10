//! Buffered I/O wrappers.
//!
//! This module provides buffered read and write wrappers over raw file I/O.
//! Buffering reduces syscall overhead by batching reads and writes.
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────┐
//! │                   BufferedReader                       │
//! │  ┌─────────────────────────────────────────────────┐   │
//! │  │              Read Buffer (8KB)                  │   │
//! │  │  [data read ahead from file but not consumed]   │   │
//! │  └─────────────────────────────────────────────────┘   │
//! │                         │                              │
//! │                         ▼                              │
//! │                     FileIO                             │
//! └────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance Features
//!
//! - **Read-ahead**: Prefetches data in large blocks
//! - **Write coalescing**: Batches small writes into larger ones
//! - **Buffer pooling**: Uses thread-local buffer pools (see `buffer_pool`)

use super::buffer_pool::{BufferSizeClass, PooledBuffer, acquire_buffer};
use super::file_io::FileIO;
use super::mode::FileMode;
use std::io::{self, Read, SeekFrom};
use std::path::Path;

/// Buffered reader wrapper.
///
/// Provides efficient read-ahead buffering for sequential file access.
pub struct BufferedReader {
    /// Underlying file I/O.
    inner: FileIO,
    /// Read buffer.
    buffer: PooledBuffer,
    /// Current position in buffer (start of unread data).
    pos: usize,
    /// Amount of valid data in buffer.
    filled: usize,
}

impl BufferedReader {
    /// Create a new buffered reader.
    pub fn new(inner: FileIO) -> Self {
        Self {
            inner,
            buffer: acquire_buffer(BufferSizeClass::Medium),
            pos: 0,
            filled: 0,
        }
    }

    /// Create a buffered reader with a specific buffer size class.
    pub fn with_capacity(inner: FileIO, size_class: BufferSizeClass) -> Self {
        Self {
            inner,
            buffer: acquire_buffer(size_class),
            pos: 0,
            filled: 0,
        }
    }

    /// Open a file for buffered reading.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mode = FileMode::parse("rb").expect("valid mode");
        let inner = FileIO::open(path, mode)?;
        Ok(Self::new(inner))
    }

    /// Get a reference to the underlying FileIO.
    #[inline]
    pub fn inner(&self) -> &FileIO {
        &self.inner
    }

    /// Get a mutable reference to the underlying FileIO.
    #[inline]
    pub fn inner_mut(&mut self) -> &mut FileIO {
        &mut self.inner
    }

    /// Consume this reader and return the underlying FileIO.
    pub fn into_inner(self) -> FileIO {
        self.inner
    }

    /// Check if the buffer has data available.
    #[inline]
    fn buffer_available(&self) -> usize {
        self.filled - self.pos
    }

    /// Fill the buffer from the underlying file.
    fn fill_buffer(&mut self) -> io::Result<usize> {
        // Compact: move remaining data to front
        if self.pos > 0 && self.filled > self.pos {
            self.buffer
                .as_mut_slice()
                .copy_within(self.pos..self.filled, 0);
            self.filled -= self.pos;
            self.pos = 0;
        } else {
            self.pos = 0;
            self.filled = 0;
        }

        // Fill remaining space
        let buf = &mut self.buffer.as_mut_slice()[self.filled..];
        let n = self.inner.read(buf)?;
        self.filled += n;
        Ok(n)
    }

    /// Read up to `limit` bytes.
    pub fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        // If buffer has data, serve from it
        if self.buffer_available() > 0 {
            let to_copy = buf.len().min(self.buffer_available());
            buf[..to_copy].copy_from_slice(&self.buffer[self.pos..self.pos + to_copy]);
            self.pos += to_copy;
            return Ok(to_copy);
        }

        // Buffer empty - for large reads, bypass buffer
        if buf.len() >= self.buffer.capacity() {
            return self.inner.read(buf);
        }

        // Fill buffer and retry
        if self.fill_buffer()? == 0 {
            return Ok(0); // EOF
        }

        let to_copy = buf.len().min(self.buffer_available());
        buf[..to_copy].copy_from_slice(&self.buffer[self.pos..self.pos + to_copy]);
        self.pos += to_copy;
        Ok(to_copy)
    }

    /// Read exactly `buf.len()` bytes.
    pub fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        let mut offset = 0;
        while offset < buf.len() {
            match self.read(&mut buf[offset..])? {
                0 => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "failed to fill whole buffer",
                    ));
                }
                n => offset += n,
            }
        }
        Ok(())
    }

    /// Read all remaining bytes.
    pub fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let mut total = 0;

        // First consume any buffered data
        if self.buffer_available() > 0 {
            buf.extend_from_slice(&self.buffer[self.pos..self.filled]);
            total += self.buffer_available();
            self.pos = self.filled;
        }

        // Then read directly from file
        let start_len = buf.len();
        self.inner.file_mut().read_to_end(buf)?;
        total += buf.len() - start_len;
        Ok(total)
    }

    /// Read a line (up to and including newline).
    ///
    /// Returns the line as bytes, including the newline if present.
    pub fn read_line(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let start_len = buf.len();

        loop {
            // Search for newline in buffer
            if self.buffer_available() > 0 {
                let search_slice = &self.buffer[self.pos..self.filled];

                // SIMD-friendly newline search (scalar for now)
                if let Some(newline_offset) = memchr::memchr(b'\n', search_slice) {
                    // Found newline - copy up to and including it
                    let end = self.pos + newline_offset + 1;
                    buf.extend_from_slice(&self.buffer[self.pos..end]);
                    self.pos = end;
                    return Ok(buf.len() - start_len);
                }

                // No newline - consume entire buffer
                buf.extend_from_slice(search_slice);
                self.pos = self.filled;
            }

            // Need more data
            if self.fill_buffer()? == 0 {
                // EOF
                return Ok(buf.len() - start_len);
            }
        }
    }

    /// Seek to a position.
    pub fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        // Discard buffer on seek
        self.pos = 0;
        self.filled = 0;
        self.inner.seek(pos)
    }

    /// Check if readable.
    #[inline]
    pub fn readable(&self) -> bool {
        self.inner.readable()
    }

    /// Check if closed.
    #[inline]
    pub fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }

    /// Close the reader.
    pub fn close(&mut self) -> io::Result<()> {
        self.inner.close()
    }
}

/// Buffered writer wrapper.
///
/// Provides efficient write coalescing for sequential file output.
pub struct BufferedWriter {
    /// Underlying file I/O (Option for safe extraction).
    inner: Option<FileIO>,
    /// Write buffer.
    buffer: PooledBuffer,
    /// Amount of data in buffer waiting to be flushed.
    filled: usize,
}

impl BufferedWriter {
    /// Create a new buffered writer.
    pub fn new(inner: FileIO) -> Self {
        Self {
            inner: Some(inner),
            buffer: acquire_buffer(BufferSizeClass::Medium),
            filled: 0,
        }
    }

    /// Create a buffered writer with a specific buffer size class.
    pub fn with_capacity(inner: FileIO, size_class: BufferSizeClass) -> Self {
        Self {
            inner: Some(inner),
            buffer: acquire_buffer(size_class),
            filled: 0,
        }
    }

    /// Open a file for buffered writing.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mode = FileMode::parse("wb").expect("valid mode");
        let inner = FileIO::open(path, mode)?;
        Ok(Self::new(inner))
    }

    /// Get a reference to the underlying FileIO.
    #[inline]
    pub fn inner(&self) -> Option<&FileIO> {
        self.inner.as_ref()
    }

    /// Get a mutable reference to the underlying FileIO.
    #[inline]
    pub fn inner_mut(&mut self) -> Option<&mut FileIO> {
        self.inner.as_mut()
    }

    /// Consume this writer and return the underlying FileIO.
    ///
    /// Flushes any pending data before returning.
    pub fn into_inner(mut self) -> io::Result<FileIO> {
        self.flush_buffer()?;
        self.inner
            .take()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "FileIO already taken"))
    }

    /// Available space in the buffer.
    #[inline]
    fn buffer_space(&self) -> usize {
        self.buffer.capacity() - self.filled
    }

    /// Flush the buffer to the underlying file.
    fn flush_buffer(&mut self) -> io::Result<()> {
        if self.filled > 0 {
            if let Some(ref mut inner) = self.inner {
                inner.write_all(&self.buffer[..self.filled])?;
            }
            self.filled = 0;
        }
        Ok(())
    }

    /// Write bytes to the buffer.
    pub fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // Check if we have a valid inner file
        if self.inner.is_none() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "I/O operation on closed file",
            ));
        }

        if buf.is_empty() {
            return Ok(0);
        }

        // For large writes, flush buffer and write directly
        if buf.len() >= self.buffer.capacity() {
            self.flush_buffer()?;
            // Re-acquire the borrow after flush
            return self.inner.as_mut().unwrap().write(buf);
        }

        // Check if buffer has space
        if buf.len() > self.buffer_space() {
            self.flush_buffer()?;
        }

        // Copy to buffer
        let to_copy = buf.len().min(self.buffer_space());
        self.buffer.as_mut_slice()[self.filled..self.filled + to_copy]
            .copy_from_slice(&buf[..to_copy]);
        self.filled += to_copy;
        Ok(to_copy)
    }

    /// Write all bytes.
    pub fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        let mut offset = 0;
        while offset < buf.len() {
            offset += self.write(&buf[offset..])?;
        }
        Ok(())
    }

    /// Flush all buffered data to the underlying file.
    pub fn flush(&mut self) -> io::Result<()> {
        self.flush_buffer()?;
        if let Some(ref mut inner) = self.inner {
            inner.flush()?;
        }
        Ok(())
    }

    /// Seek to a position.
    ///
    /// Flushes the buffer first.
    pub fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.flush_buffer()?;
        self.inner
            .as_mut()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "I/O operation on closed file"))?
            .seek(pos)
    }

    /// Check if writable.
    #[inline]
    pub fn writable(&self) -> bool {
        self.inner.as_ref().map_or(false, |i| i.writable())
    }

    /// Check if closed.
    #[inline]
    pub fn is_closed(&self) -> bool {
        self.inner.as_ref().map_or(true, |i| i.is_closed())
    }

    /// Close the writer.
    ///
    /// Flushes any pending data first.
    pub fn close(&mut self) -> io::Result<()> {
        self.flush_buffer()?;
        if let Some(ref mut inner) = self.inner {
            inner.close()?;
        }
        Ok(())
    }
}

impl Drop for BufferedWriter {
    fn drop(&mut self) {
        if !self.is_closed() {
            // Best-effort flush on drop
            let _ = self.flush_buffer();
        }
    }
}

/// Buffered random-access I/O.
///
/// Supports both reading and writing with separate buffers.
pub struct BufferedRandom {
    /// Underlying file I/O.
    inner: FileIO,
    /// Read buffer.
    read_buffer: PooledBuffer,
    /// Write buffer.
    write_buffer: PooledBuffer,
    /// Read buffer position.
    read_pos: usize,
    /// Read buffer fill level.
    read_filled: usize,
    /// Write buffer fill level.
    write_filled: usize,
}

impl BufferedRandom {
    /// Create a new buffered random-access wrapper.
    pub fn new(inner: FileIO) -> Self {
        Self {
            inner,
            read_buffer: acquire_buffer(BufferSizeClass::Medium),
            write_buffer: acquire_buffer(BufferSizeClass::Medium),
            read_pos: 0,
            read_filled: 0,
            write_filled: 0,
        }
    }

    /// Open a file for buffered random access.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mode = FileMode::parse("r+b").expect("valid mode");
        let inner = FileIO::open(path, mode)?;
        Ok(Self::new(inner))
    }

    /// Get a reference to the underlying FileIO.
    #[inline]
    pub fn inner(&self) -> &FileIO {
        &self.inner
    }

    /// Flush write buffer.
    fn flush_write(&mut self) -> io::Result<()> {
        if self.write_filled > 0 {
            self.inner
                .write_all(&self.write_buffer[..self.write_filled])?;
            self.write_filled = 0;
        }
        Ok(())
    }

    /// Invalidate read buffer.
    fn invalidate_read(&mut self) {
        self.read_pos = 0;
        self.read_filled = 0;
    }

    /// Read bytes.
    pub fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        // Must flush writes before read
        self.flush_write()?;

        if buf.is_empty() {
            return Ok(0);
        }

        // Serve from read buffer if available
        let available = self.read_filled - self.read_pos;
        if available > 0 {
            let to_copy = buf.len().min(available);
            buf[..to_copy]
                .copy_from_slice(&self.read_buffer[self.read_pos..self.read_pos + to_copy]);
            self.read_pos += to_copy;
            return Ok(to_copy);
        }

        // Buffer empty - fill it
        self.read_pos = 0;
        self.read_filled = self.inner.read(self.read_buffer.as_mut_slice())?;

        if self.read_filled == 0 {
            return Ok(0); // EOF
        }

        let to_copy = buf.len().min(self.read_filled);
        buf[..to_copy].copy_from_slice(&self.read_buffer[..to_copy]);
        self.read_pos = to_copy;
        Ok(to_copy)
    }

    /// Write bytes.
    pub fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // Invalidate read buffer on write
        self.invalidate_read();

        if buf.is_empty() {
            return Ok(0);
        }

        // For large writes, flush and write directly
        if buf.len() >= self.write_buffer.capacity() {
            self.flush_write()?;
            return self.inner.write(buf);
        }

        // Check space
        let space = self.write_buffer.capacity() - self.write_filled;
        if buf.len() > space {
            self.flush_write()?;
        }

        let to_copy = buf
            .len()
            .min(self.write_buffer.capacity() - self.write_filled);
        self.write_buffer.as_mut_slice()[self.write_filled..self.write_filled + to_copy]
            .copy_from_slice(&buf[..to_copy]);
        self.write_filled += to_copy;
        Ok(to_copy)
    }

    /// Seek to a position.
    pub fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.flush_write()?;
        self.invalidate_read();
        self.inner.seek(pos)
    }

    /// Flush all buffers.
    pub fn flush(&mut self) -> io::Result<()> {
        self.flush_write()?;
        self.inner.flush()
    }

    /// Close the file.
    pub fn close(&mut self) -> io::Result<()> {
        self.flush_write()?;
        self.inner.close()
    }
}

impl Drop for BufferedRandom {
    fn drop(&mut self) {
        if !self.inner.is_closed() {
            let _ = self.flush_write();
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    // -------------------------------------------------------------------------
    // BufferedReader Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_buffered_reader_basic() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"hello world").unwrap();

        let mut reader = BufferedReader::open(tmp.path()).unwrap();

        let mut buf = [0u8; 5];
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(n, 5);
        assert_eq!(&buf, b"hello");
    }

    #[test]
    fn test_buffered_reader_read_all() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"hello world").unwrap();

        let mut reader = BufferedReader::open(tmp.path()).unwrap();

        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"hello world");
    }

    #[test]
    fn test_buffered_reader_read_line() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"line1\nline2\nline3").unwrap();

        let mut reader = BufferedReader::open(tmp.path()).unwrap();

        let mut line = Vec::new();
        reader.read_line(&mut line).unwrap();
        assert_eq!(line, b"line1\n");

        line.clear();
        reader.read_line(&mut line).unwrap();
        assert_eq!(line, b"line2\n");
    }

    #[test]
    fn test_buffered_reader_read_line_no_newline() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"no newline").unwrap();

        let mut reader = BufferedReader::open(tmp.path()).unwrap();

        let mut line = Vec::new();
        reader.read_line(&mut line).unwrap();
        assert_eq!(line, b"no newline");
    }

    #[test]
    fn test_buffered_reader_seek() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"hello world").unwrap();

        let mut reader = BufferedReader::open(tmp.path()).unwrap();

        // Read some data to fill buffer
        let mut buf = [0u8; 3];
        reader.read(&mut buf).unwrap();

        // Seek should invalidate buffer
        reader.seek(SeekFrom::Start(6)).unwrap();

        reader.read(&mut buf).unwrap();
        assert_eq!(&buf, b"wor");
    }

    #[test]
    fn test_buffered_reader_large_read() {
        let tmp = NamedTempFile::new().unwrap();
        let data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        std::fs::write(tmp.path(), &data).unwrap();

        let mut reader = BufferedReader::open(tmp.path()).unwrap();

        let mut result = Vec::new();
        reader.read_to_end(&mut result).unwrap();
        assert_eq!(result, data);
    }

    // -------------------------------------------------------------------------
    // BufferedWriter Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_buffered_writer_basic() {
        let tmp = NamedTempFile::new().unwrap();

        {
            let mut writer = BufferedWriter::open(tmp.path()).unwrap();
            writer.write_all(b"hello world").unwrap();
            writer.flush().unwrap();
        }

        let content = std::fs::read(tmp.path()).unwrap();
        assert_eq!(content, b"hello world");
    }

    #[test]
    fn test_buffered_writer_many_small_writes() {
        let tmp = NamedTempFile::new().unwrap();

        {
            let mut writer = BufferedWriter::open(tmp.path()).unwrap();
            for _ in 0..100 {
                writer.write_all(b"x").unwrap();
            }
        }

        let content = std::fs::read(tmp.path()).unwrap();
        assert_eq!(content.len(), 100);
        assert!(content.iter().all(|&b| b == b'x'));
    }

    #[test]
    fn test_buffered_writer_large_write() {
        let tmp = NamedTempFile::new().unwrap();
        let data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();

        {
            let mut writer = BufferedWriter::open(tmp.path()).unwrap();
            writer.write_all(&data).unwrap();
        }

        let content = std::fs::read(tmp.path()).unwrap();
        assert_eq!(content, data);
    }

    #[test]
    fn test_buffered_writer_flush_on_drop() {
        let tmp = NamedTempFile::new().unwrap();

        {
            let mut writer = BufferedWriter::open(tmp.path()).unwrap();
            writer.write_all(b"flushed").unwrap();
            // No explicit flush - should flush on drop
        }

        let content = std::fs::read(tmp.path()).unwrap();
        assert_eq!(content, b"flushed");
    }

    // -------------------------------------------------------------------------
    // BufferedRandom Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_buffered_random_read_write() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"initial").unwrap();

        {
            let mut file = BufferedRandom::open(tmp.path()).unwrap();

            // Read initial content
            let mut buf = [0u8; 7];
            file.read(&mut buf).unwrap();
            assert_eq!(&buf, b"initial");

            // Seek and write
            file.seek(SeekFrom::Start(0)).unwrap();
            file.write(b"updated").unwrap();
            file.flush().unwrap();
        }

        let content = std::fs::read(tmp.path()).unwrap();
        assert_eq!(content, b"updated");
    }

    #[test]
    fn test_buffered_random_seek() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"hello world").unwrap();

        let mut file = BufferedRandom::open(tmp.path()).unwrap();

        file.seek(SeekFrom::Start(6)).unwrap();
        let mut buf = [0u8; 5];
        file.read(&mut buf).unwrap();
        assert_eq!(&buf, b"world");
    }
}
