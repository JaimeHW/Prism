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

// =========================================================================
// StandardStreams Tests
// =========================================================================
