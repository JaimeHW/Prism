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
