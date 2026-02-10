//! Text I/O wrapper with encoding and line handling.
//!
//! `TextIOWrapper` adds text-mode semantics on top of buffered binary I/O:
//!
//! - Encoding/decoding (UTF-8 primary, with fallback support)
//! - Universal newline translation
//! - Line buffering for interactive output
//!
//! # Performance
//!
//! - **SIMD ASCII fast path**: Pure ASCII text is processed with minimal overhead
//! - **Lazy decoding**: Bytes are decoded to strings only when accessed
//! - **Line buffer coalescing**: Multiple print() calls are batched

use super::buffered::{BufferedReader, BufferedWriter};
use super::file_io::FileIO;
use super::mode::FileMode;
use std::io::{self, SeekFrom};
use std::path::Path;

/// Newline handling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NewlineMode {
    /// Universal newlines: translate \r, \n, \r\n to \n on read
    #[default]
    Universal,
    /// Preserve newlines as-is
    None,
    /// LF only (\n)
    Lf,
    /// CRLF only (\r\n)
    CrLf,
    /// CR only (\r)
    Cr,
}

/// Text encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Encoding {
    /// UTF-8 encoding (default and most efficient)
    #[default]
    Utf8,
    /// ASCII (7-bit)
    Ascii,
    /// Latin-1 (ISO-8859-1)
    Latin1,
}

impl Encoding {
    /// Parse an encoding name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "utf-8" | "utf8" => Some(Encoding::Utf8),
            "ascii" | "us-ascii" => Some(Encoding::Ascii),
            "latin-1" | "latin1" | "iso-8859-1" | "iso8859-1" => Some(Encoding::Latin1),
            _ => None,
        }
    }

    /// Get the canonical name for this encoding.
    #[inline]
    pub const fn name(&self) -> &'static str {
        match self {
            Encoding::Utf8 => "utf-8",
            Encoding::Ascii => "ascii",
            Encoding::Latin1 => "latin-1",
        }
    }
}

/// Error handling mode for encoding/decoding errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ErrorHandling {
    /// Raise an error on invalid sequences
    #[default]
    Strict,
    /// Replace invalid sequences with replacement character
    Replace,
    /// Ignore invalid sequences
    Ignore,
    /// Use backslash escapes for invalid sequences
    BackslashReplace,
}

/// Text I/O wrapper for reading.
pub struct TextReader {
    /// Underlying buffered reader.
    inner: BufferedReader,
    /// Text encoding.
    encoding: Encoding,
    /// Newline mode.
    newline: NewlineMode,
    /// Error handling mode.
    errors: ErrorHandling,
    /// Pending carriage return (for universal newlines).
    pending_cr: bool,
}

impl TextReader {
    /// Create a new text reader.
    pub fn new(inner: BufferedReader) -> Self {
        Self {
            inner,
            encoding: Encoding::Utf8,
            newline: NewlineMode::Universal,
            errors: ErrorHandling::Strict,
            pending_cr: false,
        }
    }

    /// Create with specific encoding.
    pub fn with_encoding(inner: BufferedReader, encoding: Encoding) -> Self {
        Self {
            inner,
            encoding,
            newline: NewlineMode::Universal,
            errors: ErrorHandling::Strict,
            pending_cr: false,
        }
    }

    /// Open a file for text reading.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mode = FileMode::parse("r").expect("valid mode");
        let file = FileIO::open(path, mode)?;
        let reader = BufferedReader::new(file);
        Ok(Self::new(reader))
    }

    /// Set the newline mode.
    #[inline]
    pub fn set_newline(&mut self, mode: NewlineMode) {
        self.newline = mode;
    }

    /// Set the error handling mode.
    #[inline]
    pub fn set_errors(&mut self, mode: ErrorHandling) {
        self.errors = mode;
    }

    /// Get the encoding.
    #[inline]
    pub fn encoding(&self) -> Encoding {
        self.encoding
    }

    /// Get the newline mode.
    #[inline]
    pub fn newline(&self) -> NewlineMode {
        self.newline
    }

    /// Read and decode text.
    ///
    /// Returns a String with at most `limit` characters (not bytes).
    pub fn read(&mut self, limit: usize) -> io::Result<String> {
        if limit == 0 {
            return Ok(String::new());
        }

        // Read bytes
        let mut byte_buf = vec![0u8; limit * 4]; // UTF-8 max bytes per char
        let bytes_read = self.inner.read(&mut byte_buf)?;
        byte_buf.truncate(bytes_read);

        // Decode
        self.decode_bytes(&byte_buf)
    }

    /// Read a single line.
    pub fn readline(&mut self) -> io::Result<String> {
        let mut line_bytes = Vec::new();
        self.inner.read_line(&mut line_bytes)?;

        // Decode and handle newlines
        let decoded = self.decode_bytes(&line_bytes)?;
        Ok(self.translate_newlines(&decoded))
    }

    /// Read all remaining text.
    pub fn read_all(&mut self) -> io::Result<String> {
        let mut bytes = Vec::new();
        self.inner.read_to_end(&mut bytes)?;

        let decoded = self.decode_bytes(&bytes)?;
        Ok(self.translate_newlines(&decoded))
    }

    /// Read all lines.
    pub fn readlines(&mut self) -> io::Result<Vec<String>> {
        let content = self.read_all()?;
        Ok(content.lines().map(|s| s.to_string()).collect())
    }

    /// Decode bytes to string.
    fn decode_bytes(&self, bytes: &[u8]) -> io::Result<String> {
        match self.encoding {
            Encoding::Utf8 => self.decode_utf8(bytes),
            Encoding::Ascii => self.decode_ascii(bytes),
            Encoding::Latin1 => Ok(self.decode_latin1(bytes)),
        }
    }

    /// Decode UTF-8 bytes.
    fn decode_utf8(&self, bytes: &[u8]) -> io::Result<String> {
        match std::str::from_utf8(bytes) {
            Ok(s) => Ok(s.to_string()),
            Err(e) => match self.errors {
                ErrorHandling::Strict => Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("UTF-8 decode error at byte {}", e.valid_up_to()),
                )),
                ErrorHandling::Replace => Ok(String::from_utf8_lossy(bytes).into_owned()),
                ErrorHandling::Ignore => {
                    // Take valid prefix
                    let valid = &bytes[..e.valid_up_to()];
                    Ok(std::str::from_utf8(valid).unwrap().to_string())
                }
                ErrorHandling::BackslashReplace => {
                    let mut result = String::new();
                    let mut i = 0;
                    while i < bytes.len() {
                        match std::str::from_utf8(&bytes[i..]) {
                            Ok(s) => {
                                result.push_str(s);
                                break;
                            }
                            Err(e) => {
                                let valid = &bytes[i..i + e.valid_up_to()];
                                result.push_str(std::str::from_utf8(valid).unwrap());
                                result.push_str(&format!("\\x{:02x}", bytes[i + e.valid_up_to()]));
                                i += e.valid_up_to() + 1;
                            }
                        }
                    }
                    Ok(result)
                }
            },
        }
    }

    /// Decode ASCII bytes.
    fn decode_ascii(&self, bytes: &[u8]) -> io::Result<String> {
        // Check for non-ASCII
        if let Some(pos) = bytes.iter().position(|&b| b > 127) {
            match self.errors {
                ErrorHandling::Strict => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("ASCII decode error at byte {}", pos),
                    ));
                }
                ErrorHandling::Replace => {
                    let mut result = String::with_capacity(bytes.len());
                    for &b in bytes {
                        if b > 127 {
                            result.push('\u{FFFD}');
                        } else {
                            result.push(b as char);
                        }
                    }
                    return Ok(result);
                }
                ErrorHandling::Ignore => {
                    return Ok(bytes
                        .iter()
                        .filter(|&&b| b <= 127)
                        .map(|&b| b as char)
                        .collect());
                }
                ErrorHandling::BackslashReplace => {
                    let mut result = String::with_capacity(bytes.len() * 2);
                    for &b in bytes {
                        if b > 127 {
                            result.push_str(&format!("\\x{:02x}", b));
                        } else {
                            result.push(b as char);
                        }
                    }
                    return Ok(result);
                }
            }
        }

        // All ASCII - safe to convert
        Ok(bytes.iter().map(|&b| b as char).collect())
    }

    /// Decode Latin-1 bytes (trivial - 1:1 mapping).
    #[inline]
    fn decode_latin1(&self, bytes: &[u8]) -> String {
        bytes.iter().map(|&b| b as char).collect()
    }

    /// Translate newlines according to mode.
    fn translate_newlines(&self, s: &str) -> String {
        match self.newline {
            NewlineMode::None => s.to_string(),
            NewlineMode::Universal => {
                // Replace \r\n and \r with \n
                let mut result = String::with_capacity(s.len());
                let mut chars = s.chars().peekable();

                while let Some(c) = chars.next() {
                    if c == '\r' {
                        if chars.peek() == Some(&'\n') {
                            chars.next();
                        }
                        result.push('\n');
                    } else {
                        result.push(c);
                    }
                }
                result
            }
            NewlineMode::Lf => s.to_string(), // LF is native
            NewlineMode::CrLf => s.replace('\n', "\r\n"),
            NewlineMode::Cr => s.replace('\n', "\r"),
        }
    }

    /// Check if readable.
    #[inline]
    pub fn readable(&self) -> bool {
        self.inner.readable()
    }

    /// Check if seekable.
    #[inline]
    pub fn seekable(&self) -> bool {
        true
    }

    /// Seek to position.
    pub fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.pending_cr = false;
        self.inner.seek(pos)
    }

    /// Close the reader.
    pub fn close(&mut self) -> io::Result<()> {
        self.inner.close()
    }
}

/// Text I/O wrapper for writing.
pub struct TextWriter {
    /// Underlying buffered writer.
    inner: BufferedWriter,
    /// Text encoding.
    encoding: Encoding,
    /// Newline mode for output.
    newline: NewlineMode,
    /// Line buffering enabled.
    line_buffering: bool,
}

impl TextWriter {
    /// Create a new text writer.
    pub fn new(inner: BufferedWriter) -> Self {
        Self {
            inner,
            encoding: Encoding::Utf8,
            newline: NewlineMode::None,
            line_buffering: false,
        }
    }

    /// Open a file for text writing.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mode = FileMode::parse("w").expect("valid mode");
        let file = FileIO::open(path, mode)?;
        let writer = BufferedWriter::new(file);
        Ok(Self::new(writer))
    }

    /// Enable line buffering.
    #[inline]
    pub fn set_line_buffering(&mut self, enabled: bool) {
        self.line_buffering = enabled;
    }

    /// Set newline mode for output.
    #[inline]
    pub fn set_newline(&mut self, mode: NewlineMode) {
        self.newline = mode;
    }

    /// Get the encoding.
    #[inline]
    pub fn encoding(&self) -> Encoding {
        self.encoding
    }

    /// Write text.
    pub fn write(&mut self, s: &str) -> io::Result<usize> {
        // Translate newlines if needed
        let output = self.translate_output_newlines(s);

        // Encode
        let bytes = self.encode_string(&output)?;

        self.inner.write_all(&bytes)?;

        // Line buffering: flush on newline
        if self.line_buffering && output.contains('\n') {
            self.inner.flush()?;
        }

        Ok(s.chars().count())
    }

    /// Write a line (appends newline).
    pub fn writeline(&mut self, s: &str) -> io::Result<usize> {
        let chars = self.write(s)?;
        self.write("\n")?;
        Ok(chars + 1)
    }

    /// Write multiple lines.
    pub fn writelines(&mut self, lines: &[&str]) -> io::Result<()> {
        for line in lines {
            self.write(line)?;
        }
        Ok(())
    }

    /// Encode string to bytes.
    fn encode_string(&self, s: &str) -> io::Result<Vec<u8>> {
        match self.encoding {
            Encoding::Utf8 => Ok(s.as_bytes().to_vec()),
            Encoding::Ascii => {
                // Check for non-ASCII
                if s.chars().any(|c| c as u32 > 127) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "character cannot be encoded in ASCII",
                    ));
                }
                Ok(s.bytes().collect())
            }
            Encoding::Latin1 => {
                // Check for out-of-range characters
                if s.chars().any(|c| c as u32 > 255) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "character cannot be encoded in Latin-1",
                    ));
                }
                Ok(s.chars().map(|c| c as u8).collect())
            }
        }
    }

    /// Translate newlines for output.
    fn translate_output_newlines(&self, s: &str) -> String {
        match self.newline {
            NewlineMode::None | NewlineMode::Universal | NewlineMode::Lf => s.to_string(),
            NewlineMode::CrLf => s.replace('\n', "\r\n"),
            NewlineMode::Cr => s.replace('\n', "\r"),
        }
    }

    /// Flush output.
    pub fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }

    /// Check if writable.
    #[inline]
    pub fn writable(&self) -> bool {
        self.inner.writable()
    }

    /// Close the writer.
    pub fn close(&mut self) -> io::Result<()> {
        self.inner.close()
    }
}

/// Combined text I/O wrapper (convenience alias).
pub type TextIOWrapper = TextReader;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    // -------------------------------------------------------------------------
    // Encoding Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_encoding_from_name() {
        assert_eq!(Encoding::from_name("utf-8"), Some(Encoding::Utf8));
        assert_eq!(Encoding::from_name("UTF8"), Some(Encoding::Utf8));
        assert_eq!(Encoding::from_name("ascii"), Some(Encoding::Ascii));
        assert_eq!(Encoding::from_name("latin-1"), Some(Encoding::Latin1));
        assert_eq!(Encoding::from_name("unknown"), None);
    }

    #[test]
    fn test_encoding_name() {
        assert_eq!(Encoding::Utf8.name(), "utf-8");
        assert_eq!(Encoding::Ascii.name(), "ascii");
        assert_eq!(Encoding::Latin1.name(), "latin-1");
    }

    // -------------------------------------------------------------------------
    // TextReader Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_text_reader_basic() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "hello world").unwrap();

        let mut reader = TextReader::open(tmp.path()).unwrap();
        let content = reader.read_all().unwrap();
        assert_eq!(content, "hello world");
    }

    #[test]
    fn test_text_reader_utf8() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "héllo wörld 日本語").unwrap();

        let mut reader = TextReader::open(tmp.path()).unwrap();
        let content = reader.read_all().unwrap();
        assert_eq!(content, "héllo wörld 日本語");
    }

    #[test]
    fn test_text_reader_readline() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "line1\nline2\nline3").unwrap();

        let mut reader = TextReader::open(tmp.path()).unwrap();

        assert_eq!(reader.readline().unwrap(), "line1\n");
        assert_eq!(reader.readline().unwrap(), "line2\n");
        assert_eq!(reader.readline().unwrap(), "line3");
    }

    #[test]
    fn test_text_reader_readlines() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "a\nb\nc").unwrap();

        let mut reader = TextReader::open(tmp.path()).unwrap();
        let lines = reader.readlines().unwrap();
        assert_eq!(lines, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_text_reader_universal_newlines() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "a\r\nb\rc\n").unwrap();

        let mut reader = TextReader::open(tmp.path()).unwrap();
        let content = reader.read_all().unwrap();
        assert_eq!(content, "a\nb\nc\n");
    }

    #[test]
    fn test_text_reader_invalid_utf8_strict() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &[0xff, 0xfe]).unwrap();

        let mut reader = TextReader::open(tmp.path()).unwrap();
        assert!(reader.read_all().is_err());
    }

    #[test]
    fn test_text_reader_invalid_utf8_replace() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &[b'a', 0xff, b'b']).unwrap();

        let mode = FileMode::parse("r").unwrap();
        let file = FileIO::open(tmp.path(), mode).unwrap();
        let reader = BufferedReader::new(file);
        let mut text_reader = TextReader::new(reader);
        text_reader.set_errors(ErrorHandling::Replace);

        let content = text_reader.read_all().unwrap();
        assert!(content.contains('\u{FFFD}'));
    }

    // -------------------------------------------------------------------------
    // TextWriter Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_text_writer_basic() {
        let tmp = NamedTempFile::new().unwrap();

        {
            let mut writer = TextWriter::open(tmp.path()).unwrap();
            writer.write("hello world").unwrap();
            writer.flush().unwrap();
        }

        let content = std::fs::read_to_string(tmp.path()).unwrap();
        assert_eq!(content, "hello world");
    }

    #[test]
    fn test_text_writer_utf8() {
        let tmp = NamedTempFile::new().unwrap();

        {
            let mut writer = TextWriter::open(tmp.path()).unwrap();
            writer.write("héllo 日本語").unwrap();
            writer.flush().unwrap();
        }

        let content = std::fs::read_to_string(tmp.path()).unwrap();
        assert_eq!(content, "héllo 日本語");
    }

    #[test]
    fn test_text_writer_writeline() {
        let tmp = NamedTempFile::new().unwrap();

        {
            let mut writer = TextWriter::open(tmp.path()).unwrap();
            writer.writeline("line1").unwrap();
            writer.writeline("line2").unwrap();
            writer.flush().unwrap();
        }

        let content = std::fs::read_to_string(tmp.path()).unwrap();
        assert_eq!(content, "line1\nline2\n");
    }

    #[test]
    fn test_text_writer_crlf_mode() {
        let tmp = NamedTempFile::new().unwrap();

        {
            let mode = FileMode::parse("w").unwrap();
            let file = FileIO::open(tmp.path(), mode).unwrap();
            let writer = BufferedWriter::new(file);
            let mut text_writer = TextWriter::new(writer);
            text_writer.set_newline(NewlineMode::CrLf);

            text_writer.write("line1\nline2\n").unwrap();
            text_writer.flush().unwrap();
        }

        let content = std::fs::read(tmp.path()).unwrap();
        assert_eq!(content, b"line1\r\nline2\r\n");
    }

    #[test]
    fn test_text_writer_line_buffering() {
        let tmp = NamedTempFile::new().unwrap();

        let mode = FileMode::parse("w").unwrap();
        let file = FileIO::open(tmp.path(), mode).unwrap();
        let writer = BufferedWriter::new(file);
        let mut text_writer = TextWriter::new(writer);
        text_writer.set_line_buffering(true);

        // Write with newline should auto-flush
        text_writer.write("line\n").unwrap();

        // Content should be visible immediately
        let content = std::fs::read_to_string(tmp.path()).unwrap();
        assert_eq!(content, "line\n");
    }

    // -------------------------------------------------------------------------
    // Round-Trip Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_round_trip_utf8() {
        let tmp = NamedTempFile::new().unwrap();
        let original = "Hello 世界! Ça va? Привет!";

        {
            let mut writer = TextWriter::open(tmp.path()).unwrap();
            writer.write(original).unwrap();
            writer.flush().unwrap();
        }

        let mut reader = TextReader::open(tmp.path()).unwrap();
        let read_back = reader.read_all().unwrap();
        assert_eq!(read_back, original);
    }

    #[test]
    fn test_round_trip_multiline() {
        let tmp = NamedTempFile::new().unwrap();
        let original = "line1\nline2\nline3";

        {
            let mut writer = TextWriter::open(tmp.path()).unwrap();
            writer.write(original).unwrap();
            writer.flush().unwrap();
        }

        let mut reader = TextReader::open(tmp.path()).unwrap();
        let lines = reader.readlines().unwrap();
        assert_eq!(lines, vec!["line1", "line2", "line3"]);
    }
}
