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
