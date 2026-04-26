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
