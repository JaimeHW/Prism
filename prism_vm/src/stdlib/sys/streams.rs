//! Standard stream management.
//!
//! Provides access to stdin, stdout, stderr with buffering
//! and encoding support.

use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::sync::{Arc, LazyLock, Mutex};

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

    /// Read bytes from stdin.
    pub fn read_bytes(&self, count: Option<usize>) -> io::Result<Vec<u8>> {
        let mut guard = self
            .stdin
            .lock()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "stdin lock poisoned"))?;
        let mut buffer = Vec::new();
        match count {
            Some(limit) => {
                buffer.resize(limit, 0);
                let read = guard.read(&mut buffer)?;
                buffer.truncate(read);
            }
            None => {
                guard.read_to_end(&mut buffer)?;
            }
        }
        Ok(buffer)
    }

    /// Read a binary line from stdin.
    pub fn read_line_bytes(&self) -> io::Result<Vec<u8>> {
        let mut guard = self
            .stdin
            .lock()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "stdin lock poisoned"))?;
        let mut line = Vec::new();
        guard.read_until(b'\n', &mut line)?;
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

    /// Write raw bytes to stdout.
    pub fn write_stdout_bytes(&self, data: &[u8]) -> io::Result<()> {
        let mut guard = self
            .stdout
            .lock()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "stdout lock poisoned"))?;
        guard.write_all(data)?;
        guard.flush()?;
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

    /// Write raw bytes to stderr.
    pub fn write_stderr_bytes(&self, data: &[u8]) -> io::Result<()> {
        let mut guard = self
            .stderr
            .lock()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "stderr lock poisoned"))?;
        guard.write_all(data)?;
        guard.flush()?;
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

static STANDARD_STREAMS: LazyLock<StandardStreams> = LazyLock::new(StandardStreams::new);

/// Return the process-global standard stream adapters used by Python-visible
/// `sys.std*` objects.
#[inline]
pub fn standard_streams() -> &'static StandardStreams {
    &STANDARD_STREAMS
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
mod tests;
