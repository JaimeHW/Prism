//! Standard stream management.
//!
//! Provides access to stdin, stdout, stderr with buffering
//! and encoding support.

use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::sync::{Arc, Mutex};

// =============================================================================
// Stream Types
// =============================================================================

/// Stream access mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamMode {
    /// Read-only text stream.
    ReadText,
    /// Write-only text stream.
    WriteText,
    /// Read-only binary stream.
    ReadBinary,
    /// Write-only binary stream.
    WriteBinary,
}

impl StreamMode {
    /// Check if readable.
    #[inline]
    pub fn is_readable(&self) -> bool {
        matches!(self, StreamMode::ReadText | StreamMode::ReadBinary)
    }

    /// Check if writable.
    #[inline]
    pub fn is_writable(&self) -> bool {
        matches!(self, StreamMode::WriteText | StreamMode::WriteBinary)
    }

    /// Check if binary.
    #[inline]
    pub fn is_binary(&self) -> bool {
        matches!(self, StreamMode::ReadBinary | StreamMode::WriteBinary)
    }

    /// Check if text.
    #[inline]
    pub fn is_text(&self) -> bool {
        matches!(self, StreamMode::ReadText | StreamMode::WriteText)
    }
}

// =============================================================================
// Standard Streams Container
// =============================================================================

/// Container for standard streams.
#[derive(Debug)]
pub struct StandardStreams {
    /// Standard input (buffered).
    stdin: Arc<Mutex<BufReader<io::Stdin>>>,
    /// Standard output (buffered).
    stdout: Arc<Mutex<BufWriter<io::Stdout>>>,
    /// Standard error (line-buffered).
    stderr: Arc<Mutex<io::Stderr>>,
    /// Original stdin (for __stdin__).
    original_stdin: Arc<Mutex<BufReader<io::Stdin>>>,
    /// Original stdout (for __stdout__).
    original_stdout: Arc<Mutex<BufWriter<io::Stdout>>>,
    /// Original stderr (for __stderr__).
    original_stderr: Arc<Mutex<io::Stderr>>,
}

impl StandardStreams {
    /// Create new standard streams.
    pub fn new() -> Self {
        Self {
            stdin: Arc::new(Mutex::new(BufReader::new(io::stdin()))),
            stdout: Arc::new(Mutex::new(BufWriter::new(io::stdout()))),
            stderr: Arc::new(Mutex::new(io::stderr())),
            original_stdin: Arc::new(Mutex::new(BufReader::new(io::stdin()))),
            original_stdout: Arc::new(Mutex::new(BufWriter::new(io::stdout()))),
            original_stderr: Arc::new(Mutex::new(io::stderr())),
        }
    }

    /// Read a line from stdin.
    pub fn read_line(&self) -> io::Result<String> {
        let mut guard = self
            .stdin
            .lock()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "stdin lock poisoned"))?;
        let mut line = String::new();
        guard.read_line(&mut line)?;
        Ok(line)
    }

    /// Write to stdout.
    pub fn write_stdout(&self, data: &str) -> io::Result<()> {
        let mut guard = self
            .stdout
            .lock()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "stdout lock poisoned"))?;
        guard.write_all(data.as_bytes())?;
        Ok(())
    }

    /// Write line to stdout.
    pub fn writeln_stdout(&self, data: &str) -> io::Result<()> {
        let mut guard = self
            .stdout
            .lock()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "stdout lock poisoned"))?;
        guard.write_all(data.as_bytes())?;
        guard.write_all(b"\n")?;
        Ok(())
    }

    /// Flush stdout.
    pub fn flush_stdout(&self) -> io::Result<()> {
        let mut guard = self
            .stdout
            .lock()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "stdout lock poisoned"))?;
        guard.flush()
    }

    /// Write to stderr.
    pub fn write_stderr(&self, data: &str) -> io::Result<()> {
        let mut guard = self
            .stderr
            .lock()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "stderr lock poisoned"))?;
        guard.write_all(data.as_bytes())?;
        Ok(())
    }

    /// Write line to stderr.
    pub fn writeln_stderr(&self, data: &str) -> io::Result<()> {
        let mut guard = self
            .stderr
            .lock()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "stderr lock poisoned"))?;
        guard.write_all(data.as_bytes())?;
        guard.write_all(b"\n")?;
        Ok(())
    }

    /// Flush stderr.
    pub fn flush_stderr(&self) -> io::Result<()> {
        let mut guard = self
            .stderr
            .lock()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "stderr lock poisoned"))?;
        guard.flush()
    }
}

impl Default for StandardStreams {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// String Writer for Capturing Output
// =============================================================================

/// A writer that captures output to a string.
#[derive(Debug, Clone, Default)]
pub struct StringWriter {
    buffer: String,
}

impl StringWriter {
    /// Create new empty string writer.
    #[inline]
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    /// Create with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: String::with_capacity(capacity),
        }
    }

    /// Get the captured string.
    #[inline]
    pub fn get(&self) -> &str {
        &self.buffer
    }

    /// Take the captured string, consuming the writer.
    #[inline]
    pub fn into_string(self) -> String {
        self.buffer
    }

    /// Clear the buffer.
    #[inline]
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Get the length.
    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

impl Write for StringWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let s =
            std::str::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        self.buffer.push_str(s);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // StreamMode Tests
    // =========================================================================

    #[test]
    fn test_stream_mode_is_readable() {
        assert!(StreamMode::ReadText.is_readable());
        assert!(StreamMode::ReadBinary.is_readable());
        assert!(!StreamMode::WriteText.is_readable());
        assert!(!StreamMode::WriteBinary.is_readable());
    }

    #[test]
    fn test_stream_mode_is_writable() {
        assert!(!StreamMode::ReadText.is_writable());
        assert!(!StreamMode::ReadBinary.is_writable());
        assert!(StreamMode::WriteText.is_writable());
        assert!(StreamMode::WriteBinary.is_writable());
    }

    #[test]
    fn test_stream_mode_is_binary() {
        assert!(!StreamMode::ReadText.is_binary());
        assert!(StreamMode::ReadBinary.is_binary());
        assert!(!StreamMode::WriteText.is_binary());
        assert!(StreamMode::WriteBinary.is_binary());
    }

    #[test]
    fn test_stream_mode_is_text() {
        assert!(StreamMode::ReadText.is_text());
        assert!(!StreamMode::ReadBinary.is_text());
        assert!(StreamMode::WriteText.is_text());
        assert!(!StreamMode::WriteBinary.is_text());
    }

    #[test]
    fn test_stream_mode_clone() {
        let mode = StreamMode::ReadText;
        let cloned = mode.clone();
        assert_eq!(mode, cloned);
    }

    #[test]
    fn test_stream_mode_copy() {
        let mode = StreamMode::WriteText;
        let _copied: StreamMode = mode;
        let _again: StreamMode = mode; // Can use again because Copy
    }

    // =========================================================================
    // StringWriter Tests
    // =========================================================================

    #[test]
    fn test_string_writer_new() {
        let writer = StringWriter::new();
        assert!(writer.is_empty());
        assert_eq!(writer.len(), 0);
    }

    #[test]
    fn test_string_writer_with_capacity() {
        let writer = StringWriter::with_capacity(100);
        assert!(writer.is_empty());
    }

    #[test]
    fn test_string_writer_write() {
        let mut writer = StringWriter::new();
        writer.write_all(b"Hello").unwrap();
        assert_eq!(writer.get(), "Hello");
    }

    #[test]
    fn test_string_writer_write_multiple() {
        let mut writer = StringWriter::new();
        writer.write_all(b"Hello, ").unwrap();
        writer.write_all(b"World!").unwrap();
        assert_eq!(writer.get(), "Hello, World!");
    }

    #[test]
    fn test_string_writer_into_string() {
        let mut writer = StringWriter::new();
        writer.write_all(b"test").unwrap();
        let s = writer.into_string();
        assert_eq!(s, "test");
    }

    #[test]
    fn test_string_writer_clear() {
        let mut writer = StringWriter::new();
        writer.write_all(b"data").unwrap();
        writer.clear();
        assert!(writer.is_empty());
    }

    #[test]
    fn test_string_writer_len() {
        let mut writer = StringWriter::new();
        writer.write_all(b"1234567890").unwrap();
        assert_eq!(writer.len(), 10);
    }

    #[test]
    fn test_string_writer_flush() {
        let mut writer = StringWriter::new();
        // Flush should do nothing but succeed
        writer.flush().unwrap();
    }

    #[test]
    fn test_string_writer_invalid_utf8() {
        let mut writer = StringWriter::new();
        let result = writer.write_all(&[0xFF, 0xFE]);
        assert!(result.is_err());
    }

    #[test]
    fn test_string_writer_unicode() {
        let mut writer = StringWriter::new();
        writer.write_all("日本語".as_bytes()).unwrap();
        assert_eq!(writer.get(), "日本語");
    }

    #[test]
    fn test_string_writer_clone() {
        let mut writer = StringWriter::new();
        writer.write_all(b"original").unwrap();
        let cloned = writer.clone();
        assert_eq!(cloned.get(), "original");
    }

    #[test]
    fn test_string_writer_default() {
        let writer = StringWriter::default();
        assert!(writer.is_empty());
    }

    // =========================================================================
    // StandardStreams Tests
    // =========================================================================

    #[test]
    fn test_standard_streams_new() {
        // Just verify it doesn't panic
        let _streams = StandardStreams::new();
    }

    #[test]
    fn test_standard_streams_default() {
        let _streams = StandardStreams::default();
    }
}
