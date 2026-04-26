//! Exhaustive tests for the `io` module.
//!
//! Tests are organized by component:
//! 1. StringIO tests
//! 2. BytesIO tests
//! 3. FileMode tests
//! 4. IoModule tests
//! 5. Cross-cutting tests

use super::*;

// =============================================================================
// StringIO Tests
// =============================================================================

mod string_io_tests {
    use super::*;

    // =========================================================================
    // Construction
    // =========================================================================

    #[test]
    fn test_new_empty() {
        let sio = StringIO::new();
        assert_eq!(sio.getvalue(), "");
        assert_eq!(sio.len(), 0);
        assert!(sio.is_empty());
        assert!(!sio.is_closed());
    }

    #[test]
    fn test_with_initial() {
        let sio = StringIO::with_initial("hello");
        assert_eq!(sio.getvalue(), "hello");
        assert_eq!(sio.len(), 5);
    }

    #[test]
    fn test_with_capacity() {
        let sio = StringIO::with_capacity(1024);
        assert_eq!(sio.len(), 0);
        assert!(sio.is_empty());
    }

    // =========================================================================
    // Write
    // =========================================================================

    #[test]
    fn test_write_to_empty() {
        let mut sio = StringIO::new();
        let n = sio.write("hello").unwrap();
        assert_eq!(n, 5);
        assert_eq!(sio.getvalue(), "hello");
    }

    #[test]
    fn test_write_multiple() {
        let mut sio = StringIO::new();
        sio.write("hello").unwrap();
        sio.write(" world").unwrap();
        assert_eq!(sio.getvalue(), "hello world");
    }

    #[test]
    fn test_write_empty_string() {
        let mut sio = StringIO::new();
        let n = sio.write("").unwrap();
        assert_eq!(n, 0);
        assert_eq!(sio.getvalue(), "");
    }

    #[test]
    fn test_write_overwrite() {
        let mut sio = StringIO::with_initial("hello");
        sio.write("HEL").unwrap();
        assert_eq!(sio.getvalue(), "HELlo");
    }

    #[test]
    fn test_write_beyond_end() {
        let mut sio = StringIO::with_initial("hi");
        sio.seek(5, 0).unwrap();
        sio.write("!").unwrap();
        assert_eq!(sio.len(), 6);
    }

    #[test]
    fn test_write_closed_errors() {
        let mut sio = StringIO::new();
        sio.close();
        assert!(sio.write("test").is_err());
    }

    #[test]
    fn test_writelines() {
        let mut sio = StringIO::new();
        sio.writelines(&["hello\n", "world\n"]).unwrap();
        assert_eq!(sio.getvalue(), "hello\nworld\n");
    }

    #[test]
    fn test_writelines_empty() {
        let mut sio = StringIO::new();
        sio.writelines(&[]).unwrap();
        assert_eq!(sio.getvalue(), "");
    }

    // =========================================================================
    // Read
    // =========================================================================

    #[test]
    fn test_read_all() {
        let mut sio = StringIO::with_initial("hello world");
        let result = sio.read(None).unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_read_n() {
        let mut sio = StringIO::with_initial("hello world");
        let result = sio.read(Some(5)).unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_read_sequential() {
        let mut sio = StringIO::with_initial("hello world");
        let r1 = sio.read(Some(5)).unwrap().to_string();
        let r2 = sio.read(Some(6)).unwrap().to_string();
        assert_eq!(r1, "hello");
        assert_eq!(r2, " world");
    }

    #[test]
    fn test_read_empty() {
        let mut sio = StringIO::new();
        let result = sio.read(None).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_read_at_end() {
        let mut sio = StringIO::with_initial("hi");
        sio.read(None).unwrap();
        let result = sio.read(None).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_read_zero() {
        let mut sio = StringIO::with_initial("hello");
        let result = sio.read(Some(0)).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_read_more_than_available() {
        let mut sio = StringIO::with_initial("hi");
        let result = sio.read(Some(100)).unwrap();
        assert_eq!(result, "hi");
    }

    #[test]
    fn test_read_closed_errors() {
        let mut sio = StringIO::with_initial("hello");
        sio.close();
        assert!(sio.read(None).is_err());
    }

    // =========================================================================
    // Readline
    // =========================================================================

    #[test]
    fn test_readline_single_line() {
        let mut sio = StringIO::with_initial("hello\nworld\n");
        let line = sio.readline().unwrap().to_string();
        assert_eq!(line, "hello\n");
    }

    #[test]
    fn test_readline_last_line_no_newline() {
        let mut sio = StringIO::with_initial("hello");
        let line = sio.readline().unwrap().to_string();
        assert_eq!(line, "hello");
    }

    #[test]
    fn test_readline_sequential() {
        let mut sio = StringIO::with_initial("line1\nline2\nline3");
        let l1 = sio.readline().unwrap().to_string();
        let l2 = sio.readline().unwrap().to_string();
        let l3 = sio.readline().unwrap().to_string();
        assert_eq!(l1, "line1\n");
        assert_eq!(l2, "line2\n");
        assert_eq!(l3, "line3");
    }

    #[test]
    fn test_readline_empty() {
        let mut sio = StringIO::new();
        let line = sio.readline().unwrap();
        assert_eq!(line, "");
    }

    #[test]
    fn test_readline_at_end() {
        let mut sio = StringIO::with_initial("hi\n");
        let _ = sio.readline().unwrap();
        let line = sio.readline().unwrap();
        assert_eq!(line, "");
    }

    // =========================================================================
    // Seek and Tell
    // =========================================================================

    #[test]
    fn test_tell_initial() {
        let sio = StringIO::new();
        assert_eq!(sio.tell().unwrap(), 0);
    }

    #[test]
    fn test_tell_after_write() {
        let mut sio = StringIO::new();
        sio.write("hello").unwrap();
        assert_eq!(sio.tell().unwrap(), 5);
    }

    #[test]
    fn test_seek_set() {
        let mut sio = StringIO::with_initial("hello world");
        sio.read(None).unwrap();
        sio.seek(0, 0).unwrap();
        assert_eq!(sio.tell().unwrap(), 0);
    }

    #[test]
    fn test_seek_cur() {
        let mut sio = StringIO::with_initial("hello");
        sio.seek(2, 0).unwrap();
        sio.seek(1, 1).unwrap();
        assert_eq!(sio.tell().unwrap(), 3);
    }

    #[test]
    fn test_seek_end() {
        let mut sio = StringIO::with_initial("hello");
        sio.seek(0, 2).unwrap();
        assert_eq!(sio.tell().unwrap(), 5);
    }

    #[test]
    fn test_seek_end_negative() {
        let mut sio = StringIO::with_initial("hello");
        sio.seek(-2, 2).unwrap();
        assert_eq!(sio.tell().unwrap(), 3);
    }

    #[test]
    fn test_seek_negative_errors() {
        let mut sio = StringIO::with_initial("hello");
        assert!(sio.seek(-1, 0).is_err());
    }

    #[test]
    fn test_seek_invalid_whence() {
        let mut sio = StringIO::with_initial("hello");
        assert!(sio.seek(0, 3).is_err());
    }

    #[test]
    fn test_seek_then_read() {
        let mut sio = StringIO::with_initial("hello world");
        sio.seek(6, 0).unwrap();
        let result = sio.read(None).unwrap();
        assert_eq!(result, "world");
    }

    #[test]
    fn test_seek_then_write() {
        let mut sio = StringIO::with_initial("aaaaa");
        sio.seek(2, 0).unwrap();
        sio.write("BB").unwrap();
        assert_eq!(sio.getvalue(), "aaBBa");
    }

    // =========================================================================
    // Truncate
    // =========================================================================

    #[test]
    fn test_truncate_at_position() {
        let mut sio = StringIO::with_initial("hello world");
        sio.seek(5, 0).unwrap();
        sio.truncate(None).unwrap();
        assert_eq!(sio.getvalue(), "hello");
    }

    #[test]
    fn test_truncate_explicit_size() {
        let mut sio = StringIO::with_initial("hello world");
        sio.truncate(Some(5)).unwrap();
        assert_eq!(sio.getvalue(), "hello");
    }

    #[test]
    fn test_truncate_to_zero() {
        let mut sio = StringIO::with_initial("hello");
        sio.truncate(Some(0)).unwrap();
        assert_eq!(sio.getvalue(), "");
    }

    #[test]
    fn test_truncate_beyond_length() {
        let mut sio = StringIO::with_initial("hello");
        let result = sio.truncate(Some(100)).unwrap();
        assert_eq!(result, 100);
        assert_eq!(sio.getvalue(), "hello"); // No extension
    }

    // =========================================================================
    // Close
    // =========================================================================

    #[test]
    fn test_close() {
        let mut sio = StringIO::new();
        assert!(!sio.is_closed());
        sio.close();
        assert!(sio.is_closed());
    }

    #[test]
    fn test_operations_after_close() {
        let mut sio = StringIO::new();
        sio.close();
        assert!(sio.read(None).is_err());
        assert!(sio.write("test").is_err());
        assert!(sio.seek(0, 0).is_err());
        assert!(sio.tell().is_err());
        assert!(sio.truncate(None).is_err());
        assert!(sio.readline().is_err());
    }

    // =========================================================================
    // Capabilities
    // =========================================================================

    #[test]
    fn test_readable() {
        let sio = StringIO::new();
        assert!(sio.readable());
    }

    #[test]
    fn test_writable() {
        let sio = StringIO::new();
        assert!(sio.writable());
    }

    #[test]
    fn test_seekable() {
        let sio = StringIO::new();
        assert!(sio.seekable());
    }

    // =========================================================================
    // Display and Debug
    // =========================================================================

    #[test]
    fn test_display() {
        let sio = StringIO::new();
        assert_eq!(format!("{}", sio), "<_io.StringIO object>");
    }

    #[test]
    fn test_debug() {
        let sio = StringIO::new();
        let debug = format!("{:?}", sio);
        assert!(debug.contains("StringIO"));
        assert!(debug.contains("len"));
    }

    // =========================================================================
    // Unicode / Multi-byte
    // =========================================================================

    #[test]
    fn test_unicode_write_read() {
        let mut sio = StringIO::new();
        sio.write("héllo 世界 🌍").unwrap();
        sio.seek(0, 0).unwrap();
        let result = sio.read(None).unwrap();
        assert_eq!(result, "héllo 世界 🌍");
    }

    #[test]
    fn test_unicode_getvalue() {
        let sio = StringIO::with_initial("日本語テスト");
        assert_eq!(sio.getvalue(), "日本語テスト");
    }

    // =========================================================================
    // Edge cases
    // =========================================================================

    #[test]
    fn test_large_write() {
        let mut sio = StringIO::new();
        let data = "x".repeat(100_000);
        sio.write(&data).unwrap();
        assert_eq!(sio.len(), 100_000);
    }

    #[test]
    fn test_write_read_cycle() {
        let mut sio = StringIO::new();
        for i in 0..100 {
            sio.write(&format!("line{}\n", i)).unwrap();
        }
        sio.seek(0, 0).unwrap();
        let mut count = 0;
        loop {
            let line = sio.readline().unwrap().to_string();
            if line.is_empty() {
                break;
            }
            count += 1;
        }
        assert_eq!(count, 100);
    }
}

// =============================================================================
// BytesIO Tests
// =============================================================================

mod bytes_io_tests {
    use super::*;

    // =========================================================================
    // Construction
    // =========================================================================

    #[test]
    fn test_new_empty() {
        let bio = BytesIO::new();
        assert_eq!(bio.getvalue(), b"");
        assert_eq!(bio.len(), 0);
        assert!(bio.is_empty());
    }

    #[test]
    fn test_with_initial() {
        let bio = BytesIO::with_initial(b"hello");
        assert_eq!(bio.getvalue(), b"hello");
        assert_eq!(bio.len(), 5);
    }

    #[test]
    fn test_with_capacity() {
        let bio = BytesIO::with_capacity(1024);
        assert_eq!(bio.len(), 0);
    }

    // =========================================================================
    // Write
    // =========================================================================

    #[test]
    fn test_write_to_empty() {
        let mut bio = BytesIO::new();
        let n = bio.write(b"hello").unwrap();
        assert_eq!(n, 5);
        assert_eq!(bio.getvalue(), b"hello");
    }

    #[test]
    fn test_write_multiple() {
        let mut bio = BytesIO::new();
        bio.write(b"hello").unwrap();
        bio.write(b" world").unwrap();
        assert_eq!(bio.getvalue(), b"hello world");
    }

    #[test]
    fn test_write_empty() {
        let mut bio = BytesIO::new();
        let n = bio.write(b"").unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn test_write_overwrite() {
        let mut bio = BytesIO::with_initial(b"hello");
        bio.write(b"HEL").unwrap();
        assert_eq!(bio.getvalue(), b"HELlo");
    }

    #[test]
    fn test_write_beyond_end() {
        let mut bio = BytesIO::with_initial(b"hi");
        bio.seek(5, 0).unwrap();
        bio.write(b"!").unwrap();
        assert_eq!(bio.len(), 6);
        assert_eq!(bio.getvalue()[0..2], *b"hi");
        assert_eq!(bio.getvalue()[5], b'!');
    }

    #[test]
    fn test_write_closed_errors() {
        let mut bio = BytesIO::new();
        bio.close();
        assert!(bio.write(b"test").is_err());
    }

    // =========================================================================
    // Read
    // =========================================================================

    #[test]
    fn test_read_all() {
        let mut bio = BytesIO::with_initial(b"hello world");
        let result = bio.read(None).unwrap().to_vec();
        assert_eq!(result, b"hello world");
    }

    #[test]
    fn test_read_n() {
        let mut bio = BytesIO::with_initial(b"hello world");
        let result = bio.read(Some(5)).unwrap().to_vec();
        assert_eq!(result, b"hello");
    }

    #[test]
    fn test_read_sequential() {
        let mut bio = BytesIO::with_initial(b"hello world");
        let r1 = bio.read(Some(5)).unwrap().to_vec();
        let r2 = bio.read(Some(6)).unwrap().to_vec();
        assert_eq!(r1, b"hello");
        assert_eq!(r2, b" world");
    }

    #[test]
    fn test_read_at_end() {
        let mut bio = BytesIO::with_initial(b"hi");
        let _ = bio.read(None).unwrap();
        let result = bio.read(None).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_read_zero() {
        let mut bio = BytesIO::with_initial(b"hello");
        let result = bio.read(Some(0)).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_read_closed_errors() {
        let mut bio = BytesIO::with_initial(b"hello");
        bio.close();
        assert!(bio.read(None).is_err());
    }

    // =========================================================================
    // Readexactly
    // =========================================================================

    #[test]
    fn test_readexactly_success() {
        let mut bio = BytesIO::with_initial(b"hello world");
        let result = bio.readexactly(5).unwrap().to_vec();
        assert_eq!(result, b"hello");
    }

    #[test]
    fn test_readexactly_not_enough() {
        let mut bio = BytesIO::with_initial(b"hi");
        assert!(bio.readexactly(10).is_err());
    }

    #[test]
    fn test_readexactly_exact_match() {
        let mut bio = BytesIO::with_initial(b"hello");
        let result = bio.readexactly(5).unwrap().to_vec();
        assert_eq!(result, b"hello");
    }

    // =========================================================================
    // Readline
    // =========================================================================

    #[test]
    fn test_readline() {
        let mut bio = BytesIO::with_initial(b"hello\nworld\n");
        let line = bio.readline().unwrap().to_vec();
        assert_eq!(line, b"hello\n");
    }

    #[test]
    fn test_readline_no_newline() {
        let mut bio = BytesIO::with_initial(b"hello");
        let line = bio.readline().unwrap().to_vec();
        assert_eq!(line, b"hello");
    }

    #[test]
    fn test_readline_sequential() {
        let mut bio = BytesIO::with_initial(b"l1\nl2\nl3");
        let l1 = bio.readline().unwrap().to_vec();
        let l2 = bio.readline().unwrap().to_vec();
        let l3 = bio.readline().unwrap().to_vec();
        assert_eq!(l1, b"l1\n");
        assert_eq!(l2, b"l2\n");
        assert_eq!(l3, b"l3");
    }

    #[test]
    fn test_readline_at_end() {
        let mut bio = BytesIO::with_initial(b"hi\n");
        let _ = bio.readline().unwrap();
        let line = bio.readline().unwrap();
        assert!(line.is_empty());
    }

    #[test]
    fn test_getbuffer_returns_current_contents() {
        let mut bio = BytesIO::with_initial(b"hello");
        bio.seek(0, 2).unwrap();
        bio.write(b" world").unwrap();
        assert_eq!(bio.getbuffer(), b"hello world");
    }

    // =========================================================================
    // Seek and Tell
    // =========================================================================

    #[test]
    fn test_tell_initial() {
        let bio = BytesIO::new();
        assert_eq!(bio.tell().unwrap(), 0);
    }

    #[test]
    fn test_tell_after_write() {
        let mut bio = BytesIO::new();
        bio.write(b"hello").unwrap();
        assert_eq!(bio.tell().unwrap(), 5);
    }

    #[test]
    fn test_seek_set() {
        let mut bio = BytesIO::with_initial(b"hello");
        bio.read(None).unwrap();
        bio.seek(0, 0).unwrap();
        assert_eq!(bio.tell().unwrap(), 0);
    }

    #[test]
    fn test_seek_cur() {
        let mut bio = BytesIO::with_initial(b"hello");
        bio.seek(2, 0).unwrap();
        bio.seek(1, 1).unwrap();
        assert_eq!(bio.tell().unwrap(), 3);
    }

    #[test]
    fn test_seek_end() {
        let mut bio = BytesIO::with_initial(b"hello");
        bio.seek(0, 2).unwrap();
        assert_eq!(bio.tell().unwrap(), 5);
    }

    #[test]
    fn test_seek_negative_errors() {
        let mut bio = BytesIO::with_initial(b"hello");
        assert!(bio.seek(-1, 0).is_err());
    }

    #[test]
    fn test_seek_invalid_whence() {
        let mut bio = BytesIO::with_initial(b"hello");
        assert!(bio.seek(0, 3).is_err());
    }

    #[test]
    fn test_seek_then_read() {
        let mut bio = BytesIO::with_initial(b"hello world");
        bio.seek(6, 0).unwrap();
        let result = bio.read(None).unwrap().to_vec();
        assert_eq!(result, b"world");
    }

    #[test]
    fn test_seek_then_write() {
        let mut bio = BytesIO::with_initial(b"aaaaa");
        bio.seek(2, 0).unwrap();
        bio.write(b"BB").unwrap();
        assert_eq!(bio.getvalue(), b"aaBBa");
    }

    // =========================================================================
    // Truncate
    // =========================================================================

    #[test]
    fn test_truncate_at_position() {
        let mut bio = BytesIO::with_initial(b"hello world");
        bio.seek(5, 0).unwrap();
        bio.truncate(None).unwrap();
        assert_eq!(bio.getvalue(), b"hello");
    }

    #[test]
    fn test_truncate_explicit_size() {
        let mut bio = BytesIO::with_initial(b"hello world");
        bio.truncate(Some(5)).unwrap();
        assert_eq!(bio.getvalue(), b"hello");
    }

    #[test]
    fn test_truncate_to_zero() {
        let mut bio = BytesIO::with_initial(b"hello");
        bio.truncate(Some(0)).unwrap();
        assert_eq!(bio.getvalue(), b"");
    }

    // =========================================================================
    // Close
    // =========================================================================

    #[test]
    fn test_close() {
        let mut bio = BytesIO::new();
        assert!(!bio.is_closed());
        bio.close();
        assert!(bio.is_closed());
    }

    #[test]
    fn test_operations_after_close() {
        let mut bio = BytesIO::new();
        bio.close();
        assert!(bio.read(None).is_err());
        assert!(bio.write(b"test").is_err());
        assert!(bio.seek(0, 0).is_err());
        assert!(bio.tell().is_err());
        assert!(bio.truncate(None).is_err());
    }

    // =========================================================================
    // Capabilities
    // =========================================================================

    #[test]
    fn test_readable() {
        let bio = BytesIO::new();
        assert!(bio.readable());
    }

    #[test]
    fn test_writable() {
        let bio = BytesIO::new();
        assert!(bio.writable());
    }

    #[test]
    fn test_seekable() {
        let bio = BytesIO::new();
        assert!(bio.seekable());
    }

    // =========================================================================
    // Display and Debug
    // =========================================================================

    #[test]
    fn test_display() {
        let bio = BytesIO::new();
        assert_eq!(format!("{}", bio), "<_io.BytesIO object>");
    }

    #[test]
    fn test_debug() {
        let bio = BytesIO::new();
        let debug = format!("{:?}", bio);
        assert!(debug.contains("BytesIO"));
    }

    // =========================================================================
    // Binary data
    // =========================================================================

    #[test]
    fn test_binary_data_with_zeros() {
        let mut bio = BytesIO::new();
        bio.write(&[0x00, 0xFF, 0x00, 0xFF]).unwrap();
        assert_eq!(bio.getvalue(), &[0x00, 0xFF, 0x00, 0xFF]);
    }

    #[test]
    fn test_all_byte_values() {
        let mut bio = BytesIO::new();
        let all_bytes: Vec<u8> = (0..=255).collect();
        bio.write(&all_bytes).unwrap();
        assert_eq!(bio.getvalue(), &all_bytes[..]);
    }

    #[test]
    fn test_large_binary_write() {
        let mut bio = BytesIO::new();
        let data = vec![0xAB; 100_000];
        bio.write(&data).unwrap();
        assert_eq!(bio.len(), 100_000);
    }

    #[test]
    fn test_write_read_cycle() {
        let mut bio = BytesIO::new();
        for i in 0u8..100 {
            bio.write(&[i]).unwrap();
        }
        bio.seek(0, 0).unwrap();
        let result = bio.read(None).unwrap();
        assert_eq!(result.len(), 100);
        assert_eq!(result[0], 0);
        assert_eq!(result[99], 99);
    }
}

// =============================================================================
// FileMode Tests
// =============================================================================

mod file_mode_tests {
    use super::*;

    // =========================================================================
    // Parse - Basic modes
    // =========================================================================

    #[test]
    fn test_parse_read() {
        let mode = FileMode::parse("r").unwrap();
        assert!(mode.is_reading());
        assert!(!mode.is_writing());
        assert!(mode.is_text());
        assert!(!mode.is_binary());
    }

    #[test]
    fn test_parse_write() {
        let mode = FileMode::parse("w").unwrap();
        assert!(!mode.is_reading() || mode.is_update());
        assert!(mode.is_writing());
        assert!(mode.is_truncating());
    }

    #[test]
    fn test_parse_append() {
        let mode = FileMode::parse("a").unwrap();
        assert!(mode.is_append());
        assert!(mode.is_writing());
        assert!(!mode.is_truncating());
    }

    #[test]
    fn test_parse_exclusive() {
        let mode = FileMode::parse("x").unwrap();
        assert!(mode.is_exclusive());
        assert!(mode.is_writing());
    }

    // =========================================================================
    // Parse - Combined modes
    // =========================================================================

    #[test]
    fn test_parse_rb() {
        let mode = FileMode::parse("rb").unwrap();
        assert!(mode.is_reading());
        assert!(mode.is_binary());
    }

    #[test]
    fn test_parse_wb() {
        let mode = FileMode::parse("wb").unwrap();
        assert!(mode.is_writing());
        assert!(mode.is_binary());
    }

    #[test]
    fn test_parse_rt() {
        let mode = FileMode::parse("rt").unwrap();
        assert!(mode.is_reading());
        assert!(mode.is_text());
    }

    #[test]
    fn test_parse_rplus() {
        let mode = FileMode::parse("r+").unwrap();
        assert!(mode.is_reading());
        assert!(mode.is_writing());
        assert!(mode.is_update());
    }

    #[test]
    fn test_parse_wplus() {
        let mode = FileMode::parse("w+").unwrap();
        assert!(mode.is_writing());
        assert!(mode.is_reading());
        assert!(mode.is_truncating());
    }

    #[test]
    fn test_parse_rbplus() {
        let mode = FileMode::parse("rb+").unwrap();
        assert!(mode.is_reading());
        assert!(mode.is_writing());
        assert!(mode.is_binary());
    }

    #[test]
    fn test_parse_ab() {
        let mode = FileMode::parse("ab").unwrap();
        assert!(mode.is_append());
        assert!(mode.is_binary());
    }

    #[test]
    fn test_parse_xb() {
        let mode = FileMode::parse("xb").unwrap();
        assert!(mode.is_exclusive());
        assert!(mode.is_binary());
    }

    // =========================================================================
    // Parse - Error cases
    // =========================================================================

    #[test]
    fn test_parse_empty() {
        assert!(FileMode::parse("").is_err());
    }

    #[test]
    fn test_parse_invalid_char() {
        assert!(FileMode::parse("z").is_err());
    }

    #[test]
    fn test_parse_multiple_base_modes() {
        assert!(FileMode::parse("rw").is_err());
        assert!(FileMode::parse("ra").is_err());
        assert!(FileMode::parse("wa").is_err());
    }

    #[test]
    fn test_parse_text_and_binary() {
        assert!(FileMode::parse("rbt").is_err());
        assert!(FileMode::parse("rtb").is_err());
    }

    // =========================================================================
    // Display
    // =========================================================================

    #[test]
    fn test_display_read() {
        let mode = FileMode::parse("r").unwrap();
        assert_eq!(format!("{}", mode), "rt");
    }

    #[test]
    fn test_display_rb() {
        let mode = FileMode::parse("rb").unwrap();
        assert_eq!(format!("{}", mode), "rb");
    }

    #[test]
    fn test_display_wplus() {
        let mode = FileMode::parse("w+").unwrap();
        assert_eq!(format!("{}", mode), "wt+");
    }
}

// =============================================================================
// IoModule Tests
// =============================================================================

mod io_module_tests {
    use super::*;
    use crate::VirtualMachine;
    use crate::builtins::builtin_issubclass;
    use crate::error::RuntimeErrorKind;
    use crate::ops::calls::{invoke_callable_value, invoke_callable_value_with_keywords};
    use crate::ops::objects::get_attribute_value;
    use crate::stdlib::nt::NtModule;
    use prism_core::Value;
    use prism_core::intern::intern;
    use prism_runtime::object::ObjectHeader;
    use prism_runtime::object::type_obj::TypeId;
    use prism_runtime::types::bytes::BytesObject;
    use prism_runtime::types::list::ListObject;
    use prism_runtime::types::string::{StringObject, value_as_string_ref};
    use prism_runtime::types::tuple::TupleObject;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_path(prefix: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "prism_io_{prefix}_{}_{}",
            std::process::id(),
            nonce
        ))
    }

    fn temp_path_value(path: &std::path::Path) -> Value {
        Value::string(intern(
            path.to_str()
                .expect("temp file path should be valid unicode"),
        ))
    }

    fn assert_builtin_function(value: Value, context: &str) {
        let ptr = value.as_object_ptr().expect(context);
        let header = unsafe { &*(ptr as *const ObjectHeader) };
        assert_eq!(header.type_id, TypeId::BUILTIN_FUNCTION);
    }

    #[test]
    fn test_module_name() {
        let module = IoModule::new();
        assert_eq!(module.name(), "io");
    }

    #[test]
    fn test_private_module_alias_name() {
        let module = IoModule::with_name("_io");
        assert_eq!(module.name(), "_io");
    }

    #[test]
    fn test_default_buffer_size() {
        let module = IoModule::new();
        let result = module.get_attr("DEFAULT_BUFFER_SIZE");
        assert!(result.is_ok());
    }

    #[test]
    fn test_seek_constants() {
        let module = IoModule::new();
        assert!(module.get_attr("SEEK_SET").is_ok());
        assert!(module.get_attr("SEEK_CUR").is_ok());
        assert!(module.get_attr("SEEK_END").is_ok());
    }

    #[test]
    fn test_stringio_attribute_is_class() {
        let module = IoModule::new();
        let value = module
            .get_attr("StringIO")
            .expect("io.StringIO should be exposed");
        let ptr = value
            .as_object_ptr()
            .expect("io.StringIO should be a class object");
        let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
        assert_eq!(
            header.type_id,
            prism_runtime::object::type_obj::TypeId::TYPE
        );
    }

    #[test]
    fn test_bytesio_attribute_is_class() {
        let module = IoModule::new();
        let value = module
            .get_attr("BytesIO")
            .expect("io.BytesIO should be exposed");
        let ptr = value
            .as_object_ptr()
            .expect("io.BytesIO should be a class object");
        let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
        assert_eq!(
            header.type_id,
            prism_runtime::object::type_obj::TypeId::TYPE
        );
    }

    #[test]
    fn test_textiowrapper_attribute_is_class() {
        let module = IoModule::new();
        let value = module
            .get_attr("TextIOWrapper")
            .expect("io.TextIOWrapper should be exposed");
        let ptr = value
            .as_object_ptr()
            .expect("io.TextIOWrapper should be a class object");
        let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
        assert_eq!(
            header.type_id,
            prism_runtime::object::type_obj::TypeId::TYPE
        );
    }

    #[test]
    fn test_open_and_file_aliases_are_callable_on_public_and_private_modules() {
        for module in [IoModule::new(), IoModule::with_name("_io")] {
            for name in [
                "open",
                "open_code",
                "FileIO",
                "BufferedReader",
                "BufferedWriter",
                "BufferedRandom",
            ] {
                let value = module
                    .get_attr(name)
                    .unwrap_or_else(|_| panic!("{}.{} should be exposed", module.name(), name));
                assert_builtin_function(value, "I/O open aliases should be builtin callables");
            }
        }
    }

    #[test]
    fn test_open_code_reads_source_paths_as_binary_streams() {
        let path = unique_temp_path("open_code");
        fs::write(&path, b"print('hello')\n").expect("fixture file should be written");

        let module = IoModule::with_name("_io");
        let open_code = module
            .get_attr("open_code")
            .expect("_io.open_code should be exposed");
        let mut vm = VirtualMachine::new();
        let stream = invoke_callable_value(&mut vm, open_code, &[temp_path_value(&path)])
            .expect("_io.open_code(path) should create a binary stream");
        let read = get_attribute_value(&mut vm, stream, &intern("read"))
            .expect("open_code stream should expose read");
        let rendered = invoke_callable_value(&mut vm, read, &[]).expect("read() should succeed");
        let ptr = rendered
            .as_object_ptr()
            .expect("open_code stream read() should return bytes");
        let header = unsafe { &*(ptr as *const ObjectHeader) };
        assert_eq!(header.type_id, TypeId::BYTES);
        assert_eq!(
            unsafe { &*(ptr as *const BytesObject) }.as_bytes(),
            b"print('hello')\n"
        );

        let close = get_attribute_value(&mut vm, stream, &intern("close"))
            .expect("open_code stream should expose close");
        invoke_callable_value(&mut vm, close, &[]).expect("close() should succeed");
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_open_reads_text_path_with_default_encoding() {
        let path = unique_temp_path("open_text");
        fs::write(&path, b"alpha\nbeta\n").expect("fixture file should be written");

        let module = IoModule::new();
        let open = module.get_attr("open").expect("io.open should be exposed");
        let mut vm = VirtualMachine::new();
        let stream = invoke_callable_value(&mut vm, open, &[temp_path_value(&path)])
            .expect("io.open(path) should create a text stream");
        let read = get_attribute_value(&mut vm, stream, &intern("read"))
            .expect("text stream should expose read");
        let rendered = invoke_callable_value(&mut vm, read, &[]).expect("read() should succeed");
        assert_eq!(
            value_as_string_ref(rendered)
                .expect("text mode read() should return str")
                .as_str(),
            "alpha\nbeta\n"
        );

        let close = get_attribute_value(&mut vm, stream, &intern("close"))
            .expect("text stream should expose close");
        invoke_callable_value(&mut vm, close, &[]).expect("close() should succeed");
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_open_reads_binary_path_as_bytes() {
        let path = unique_temp_path("open_binary");
        fs::write(&path, b"alpha\0beta").expect("fixture file should be written");

        let module = IoModule::new();
        let open = module.get_attr("open").expect("io.open should be exposed");
        let mut vm = VirtualMachine::new();
        let stream = invoke_callable_value(
            &mut vm,
            open,
            &[temp_path_value(&path), Value::string(intern("rb"))],
        )
        .expect("io.open(path, 'rb') should create a binary stream");
        let read = get_attribute_value(&mut vm, stream, &intern("read"))
            .expect("binary stream should expose read");
        let rendered = invoke_callable_value(&mut vm, read, &[]).expect("read() should succeed");
        let ptr = rendered
            .as_object_ptr()
            .expect("binary mode read() should return bytes");
        let header = unsafe { &*(ptr as *const ObjectHeader) };
        assert_eq!(header.type_id, TypeId::BYTES);
        assert_eq!(
            unsafe { &*(ptr as *const BytesObject) }.as_bytes(),
            b"alpha\0beta"
        );

        let close = get_attribute_value(&mut vm, stream, &intern("close"))
            .expect("binary stream should expose close");
        invoke_callable_value(&mut vm, close, &[]).expect("close() should succeed");
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_open_reads_nt_file_descriptor_as_bytes_and_closes_it() {
        let path = unique_temp_path("open_fd");
        fs::write(&path, b"descriptor payload").expect("fixture file should be written");

        let io_module = IoModule::new();
        let open = io_module
            .get_attr("open")
            .expect("io.open should be exposed");
        let nt_module = NtModule::new();
        let nt_open = nt_module
            .get_attr("open")
            .expect("nt.open should be exposed");
        let flags = nt_module
            .get_attr("O_RDONLY")
            .expect("nt.O_RDONLY should exist")
            .as_int()
            .expect("nt.O_RDONLY should be an integer")
            | nt_module
                .get_attr("O_BINARY")
                .expect("nt.O_BINARY should exist")
                .as_int()
                .expect("nt.O_BINARY should be an integer");

        let mut vm = VirtualMachine::new();
        let fd = invoke_callable_value(
            &mut vm,
            nt_open,
            &[
                temp_path_value(&path),
                Value::int(flags).expect("flags should fit"),
            ],
        )
        .expect("nt.open should create a file descriptor");
        let stream = invoke_callable_value(&mut vm, open, &[fd, Value::string(intern("rb"))])
            .expect("io.open(fd, 'rb') should create a binary stream");

        let fileno = get_attribute_value(&mut vm, stream, &intern("fileno"))
            .expect("descriptor stream should expose fileno");
        let rendered_fd =
            invoke_callable_value(&mut vm, fileno, &[]).expect("fileno() should succeed");
        assert_eq!(rendered_fd.as_int(), fd.as_int());

        let read = get_attribute_value(&mut vm, stream, &intern("read"))
            .expect("descriptor stream should expose read");
        let rendered = invoke_callable_value(&mut vm, read, &[]).expect("read() should succeed");
        let ptr = rendered
            .as_object_ptr()
            .expect("descriptor read() should return bytes");
        let header = unsafe { &*(ptr as *const ObjectHeader) };
        assert_eq!(header.type_id, TypeId::BYTES);
        assert_eq!(
            unsafe { &*(ptr as *const BytesObject) }.as_bytes(),
            b"descriptor payload"
        );

        let close = get_attribute_value(&mut vm, stream, &intern("close"))
            .expect("descriptor stream should expose close");
        invoke_callable_value(&mut vm, close, &[]).expect("close() should succeed");
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_open_rejects_invalid_path_options_before_host_open() {
        let path = unique_temp_path("open_invalid");
        let module = IoModule::new();
        let open = module.get_attr("open").expect("io.open should be exposed");
        let mut vm = VirtualMachine::new();

        let err = invoke_callable_value_with_keywords(
            &mut vm,
            open,
            &[temp_path_value(&path)],
            &[("closefd", Value::bool(false))],
        )
        .expect_err("closefd=False with a path should be rejected");
        assert!(matches!(err.kind, RuntimeErrorKind::ValueError { .. }));

        let err = invoke_callable_value_with_keywords(
            &mut vm,
            open,
            &[temp_path_value(&path), Value::string(intern("rb"))],
            &[("encoding", Value::string(intern("utf-8")))],
        )
        .expect_err("binary mode with encoding should be rejected");
        assert!(matches!(err.kind, RuntimeErrorKind::ValueError { .. }));

        let err = invoke_callable_value_with_keywords(
            &mut vm,
            open,
            &[temp_path_value(&path), Value::string(intern("r"))],
            &[("mode", Value::string(intern("rb")))],
        )
        .expect_err("duplicate mode should be rejected");
        assert!(matches!(err.kind, RuntimeErrorKind::TypeError { .. }));
    }

    #[test]
    fn test_stringio_class_call_uses_heap_type_instantiation_and_keywords() {
        let module = IoModule::new();
        let string_io = module
            .get_attr("StringIO")
            .expect("io.StringIO should be exposed");
        let mut vm = VirtualMachine::new();
        let buffer = invoke_callable_value_with_keywords(
            &mut vm,
            string_io,
            &[],
            &[
                ("initial_value", Value::string(intern("seed"))),
                ("newline", Value::string(intern(""))),
            ],
        )
        .expect("StringIO should accept keyword initialization");

        let getvalue = get_attribute_value(&mut vm, buffer, &intern("getvalue"))
            .expect("StringIO instance should expose getvalue");
        let rendered =
            invoke_callable_value(&mut vm, getvalue, &[]).expect("getvalue() should succeed");
        assert_eq!(
            value_as_string_ref(rendered)
                .expect("StringIO.getvalue() should return str")
                .as_str(),
            "seed"
        );
    }

    #[test]
    fn test_bytesio_class_call_uses_heap_type_instantiation_and_keywords() {
        let module = IoModule::new();
        let bytes_io = module
            .get_attr("BytesIO")
            .expect("io.BytesIO should be exposed");
        let mut vm = VirtualMachine::new();
        let payload = Value::object_ptr(
            Box::into_raw(Box::new(BytesObject::from_slice(b"seed"))) as *const ()
        );
        let buffer = invoke_callable_value_with_keywords(
            &mut vm,
            bytes_io,
            &[],
            &[("initial_bytes", payload)],
        )
        .expect("BytesIO should accept keyword initialization");

        let getvalue = get_attribute_value(&mut vm, buffer, &intern("getvalue"))
            .expect("BytesIO instance should expose getvalue");
        let rendered =
            invoke_callable_value(&mut vm, getvalue, &[]).expect("getvalue() should succeed");
        let ptr = rendered
            .as_object_ptr()
            .expect("BytesIO.getvalue() should allocate bytes");
        let header = unsafe { &*(ptr as *const ObjectHeader) };
        assert_eq!(header.type_id, TypeId::BYTES);
        assert_eq!(unsafe { &*(ptr as *const BytesObject) }.as_bytes(), b"seed");
    }

    #[test]
    fn test_textiowrapper_class_wraps_binary_file_stream_with_keywords() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        let temp_path = std::env::temp_dir().join(format!(
            "prism_text_wrapper_{}_{}.txt",
            std::process::id(),
            nonce
        ));
        fs::write(&temp_path, b"alpha\nbeta\n").expect("temp file should be written");

        let module = IoModule::new();
        let text_wrapper = module
            .get_attr("TextIOWrapper")
            .expect("io.TextIOWrapper should be exposed");
        let raw_stream = open_file_stream_object(
            temp_path
                .to_str()
                .expect("temp file path should be valid unicode"),
            "rb",
            None,
        )
        .expect("binary file stream should open");
        let mut vm = VirtualMachine::new();
        let wrapper = invoke_callable_value_with_keywords(
            &mut vm,
            text_wrapper,
            &[raw_stream],
            &[
                ("encoding", Value::string(intern("utf-8"))),
                ("line_buffering", Value::bool(true)),
            ],
        )
        .expect("TextIOWrapper should accept keyword initialization");

        let encoding = get_attribute_value(&mut vm, wrapper, &intern("encoding"))
            .expect("TextIOWrapper instance should expose encoding");
        assert_eq!(
            value_as_string_ref(encoding)
                .expect("encoding should be a string")
                .as_str(),
            "utf-8"
        );

        let read = get_attribute_value(&mut vm, wrapper, &intern("read"))
            .expect("TextIOWrapper instance should expose read");
        let rendered = invoke_callable_value(&mut vm, read, &[]).expect("read() should succeed");
        assert_eq!(
            value_as_string_ref(rendered)
                .expect("TextIOWrapper.read() should return text")
                .as_str(),
            "alpha\nbeta\n"
        );

        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_textiowrapper_readlines_returns_text_list_and_honors_hint() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        let temp_path = std::env::temp_dir().join(format!(
            "prism_text_wrapper_lines_{}_{}.txt",
            std::process::id(),
            nonce
        ));
        fs::write(&temp_path, b"alpha\nbeta\ngamma\n").expect("temp file should be written");

        let module = IoModule::new();
        let text_wrapper = module
            .get_attr("TextIOWrapper")
            .expect("io.TextIOWrapper should be exposed");
        let raw_stream = open_file_stream_object(
            temp_path
                .to_str()
                .expect("temp file path should be valid unicode"),
            "rb",
            None,
        )
        .expect("binary file stream should open");
        let mut vm = VirtualMachine::new();
        let wrapper = invoke_callable_value_with_keywords(
            &mut vm,
            text_wrapper,
            &[raw_stream],
            &[("encoding", Value::string(intern("utf-8")))],
        )
        .expect("TextIOWrapper should accept keyword initialization");

        let readlines = get_attribute_value(&mut vm, wrapper, &intern("readlines"))
            .expect("TextIOWrapper instance should expose readlines");
        let rendered = invoke_callable_value(&mut vm, readlines, &[Value::int(6).unwrap()])
            .expect("readlines() should succeed");
        let ptr = rendered
            .as_object_ptr()
            .expect("readlines() should allocate a list");
        let header = unsafe { &*(ptr as *const ObjectHeader) };
        assert_eq!(header.type_id, TypeId::LIST);
        let list = unsafe { &*(ptr as *const ListObject) };
        assert_eq!(list.len(), 1, "size hint should stop after the first line");
        assert_eq!(
            value_as_string_ref(list.get(0).expect("first line should exist"))
                .expect("first readlines() entry should be a string")
                .as_str(),
            "alpha\n"
        );

        let read = get_attribute_value(&mut vm, wrapper, &intern("read"))
            .expect("TextIOWrapper instance should expose read");
        let rendered = invoke_callable_value(&mut vm, read, &[]).expect("read() should succeed");
        assert_eq!(
            value_as_string_ref(rendered)
                .expect("read() after readlines() should return text")
                .as_str(),
            "beta\ngamma\n"
        );

        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_incremental_newline_decoder_attribute_is_callable() {
        let module = IoModule::new();
        let value = module
            .get_attr("IncrementalNewlineDecoder")
            .expect("io.IncrementalNewlineDecoder should be exposed");
        let ptr = value
            .as_object_ptr()
            .expect("io.IncrementalNewlineDecoder should be callable");
        let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
        assert_eq!(
            header.type_id,
            prism_runtime::object::type_obj::TypeId::BUILTIN_FUNCTION
        );
    }

    #[test]
    fn test_incremental_newline_decoder_normalizes_universal_newlines() {
        let module = IoModule::new();
        let constructor = module
            .get_attr("IncrementalNewlineDecoder")
            .expect("IncrementalNewlineDecoder should be exposed");
        let mut vm = VirtualMachine::new();
        let decoder =
            invoke_callable_value(&mut vm, constructor, &[Value::none(), Value::bool(true)])
                .expect("constructor should succeed");
        let decode = get_attribute_value(&mut vm, decoder, &intern("decode"))
            .expect("decode method should exist");
        let input = Value::object_ptr(
            Box::into_raw(Box::new(StringObject::new("a\r\nb\rc\n"))) as *const ()
        );
        let output = invoke_callable_value(&mut vm, decode, &[input, Value::bool(true)])
            .expect("decode should succeed");
        let output = value_as_string_ref(output).expect("decode should return str");
        assert_eq!(output.as_str(), "a\nb\nc\n");

        let newlines = get_attribute_value(&mut vm, decoder, &intern("newlines"))
            .expect("newlines should exist");
        let ptr = newlines
            .as_object_ptr()
            .expect("newlines should be a tuple");
        let header = unsafe { &*(ptr as *const ObjectHeader) };
        assert_eq!(header.type_id, TypeId::TUPLE);

        let tuple = unsafe { &*(ptr as *const TupleObject) };
        let rendered: Vec<String> = tuple
            .as_slice()
            .iter()
            .map(|item| {
                value_as_string_ref(*item)
                    .expect("newline entry should be str")
                    .as_str()
                    .to_string()
            })
            .collect();
        assert_eq!(rendered, vec!["\r", "\n", "\r\n"]);
    }

    #[test]
    fn test_unknown_attr() {
        let module = IoModule::new();
        assert!(module.get_attr("nonexistent").is_err());
    }

    #[test]
    fn test_dir() {
        let module = IoModule::new();
        let attrs = module.dir();
        assert!(attrs.iter().any(|a| a.as_ref() == "StringIO"));
        assert!(attrs.iter().any(|a| a.as_ref() == "BytesIO"));
        assert!(attrs.iter().any(|a| a.as_ref() == "DEFAULT_BUFFER_SIZE"));
    }

    #[test]
    fn test_buffered_io_base_is_exposed_as_a_class() {
        let module = IoModule::new();
        let buffered = module
            .get_attr("BufferedIOBase")
            .expect("BufferedIOBase should be exposed");
        assert!(buffered.as_object_ptr().is_some());
    }

    #[test]
    fn test_io_base_hierarchy_is_registered_for_issubclass() {
        let module = IoModule::new();
        let io_base = module.get_attr("IOBase").expect("IOBase should exist");
        let buffered = module
            .get_attr("BufferedIOBase")
            .expect("BufferedIOBase should exist");
        let raw = module
            .get_attr("RawIOBase")
            .expect("RawIOBase should exist");
        let text = module
            .get_attr("TextIOBase")
            .expect("TextIOBase should exist");
        let string_io = module.get_attr("StringIO").expect("StringIO should exist");
        let bytes_io = module.get_attr("BytesIO").expect("BytesIO should exist");
        let text_wrapper = module
            .get_attr("TextIOWrapper")
            .expect("TextIOWrapper should exist");

        assert_eq!(
            builtin_issubclass(&[buffered, io_base]).unwrap().as_bool(),
            Some(true)
        );
        assert_eq!(
            builtin_issubclass(&[raw, io_base]).unwrap().as_bool(),
            Some(true)
        );
        assert_eq!(
            builtin_issubclass(&[text, io_base]).unwrap().as_bool(),
            Some(true)
        );
        assert_eq!(
            builtin_issubclass(&[string_io, text]).unwrap().as_bool(),
            Some(true)
        );
        assert_eq!(
            builtin_issubclass(&[text_wrapper, text]).unwrap().as_bool(),
            Some(true)
        );
        assert_eq!(
            builtin_issubclass(&[bytes_io, buffered]).unwrap().as_bool(),
            Some(true)
        );
    }
}

// =============================================================================
// Constants Tests
// =============================================================================

mod constants_tests {
    use super::*;

    #[test]
    fn test_seek_set() {
        assert_eq!(SEEK_SET, 0);
    }

    #[test]
    fn test_seek_cur() {
        assert_eq!(SEEK_CUR, 1);
    }

    #[test]
    fn test_seek_end() {
        assert_eq!(SEEK_END, 2);
    }

    #[test]
    fn test_default_buffer_size() {
        assert_eq!(DEFAULT_BUFFER_SIZE, 8192);
    }

    #[test]
    fn test_seek_constants_distinct() {
        assert_ne!(SEEK_SET, SEEK_CUR);
        assert_ne!(SEEK_CUR, SEEK_END);
        assert_ne!(SEEK_SET, SEEK_END);
    }
}

// =============================================================================
// IoError Tests
// =============================================================================

mod io_error_tests {
    use super::*;

    #[test]
    fn test_value_error_display() {
        let e = IoError::ValueError("test".to_string());
        assert!(e.to_string().contains("ValueError"));
        assert!(e.to_string().contains("test"));
    }

    #[test]
    fn test_unsupported_display() {
        let e = IoError::UnsupportedOperation("not supported".to_string());
        assert!(e.to_string().contains("UnsupportedOperation"));
    }

    #[test]
    fn test_os_error_display() {
        let e = IoError::OsError("disk full".to_string());
        assert!(e.to_string().contains("OSError"));
    }
}
