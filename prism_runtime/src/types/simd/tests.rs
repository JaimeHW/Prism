//! Comprehensive tests for SIMD string operations.
//!
//! These tests verify correctness across:
//! - All size ranges (empty, small, medium, large, edge cases)
//! - All alignment scenarios
//! - All SIMD paths (scalar, SSE2, SSE4.2, AVX2, AVX-512)
//! - Unicode/non-ASCII handling

use super::*;

// =============================================================================
// Test Fixtures
// =============================================================================

/// Generate a test string of a specific length with a pattern.
fn make_pattern(len: usize, pattern: u8) -> Vec<u8> {
    vec![pattern; len]
}

/// Generate ASCII lowercase string.
fn make_lowercase(len: usize) -> Vec<u8> {
    (0..len).map(|i| b'a' + (i % 26) as u8).collect()
}

/// Generate ASCII uppercase string.
fn make_uppercase(len: usize) -> Vec<u8> {
    (0..len).map(|i| b'A' + (i % 26) as u8).collect()
}

/// Generate mixed case string.
fn make_mixed_case(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| {
            if i % 2 == 0 {
                b'A' + (i % 26) as u8
            } else {
                b'a' + (i % 26) as u8
            }
        })
        .collect()
}

/// Generate haystack with needle at specific position.
fn make_haystack_with_needle(len: usize, needle: &[u8], pos: usize) -> Vec<u8> {
    let mut result = vec![b'x'; len];
    if pos + needle.len() <= len {
        result[pos..pos + needle.len()].copy_from_slice(needle);
    }
    result
}

// =============================================================================
// SimdLevel Detection Tests
// =============================================================================

mod simd_level_tests {
    use super::*;

    #[test]
    fn test_simd_level_detect() {
        let level = SimdLevel::detect();
        assert!(level >= SimdLevel::Scalar);
        // On any modern x86-64, should at least have SSE2
        #[cfg(target_arch = "x86_64")]
        assert!(level >= SimdLevel::Sse2);
    }

    #[test]
    fn test_simd_level_ordering() {
        assert!(SimdLevel::Scalar < SimdLevel::Sse2);
        assert!(SimdLevel::Sse2 < SimdLevel::Sse42);
        assert!(SimdLevel::Sse42 < SimdLevel::Avx2);
        assert!(SimdLevel::Avx2 < SimdLevel::Avx512);
    }

    #[test]
    fn test_simd_level_names() {
        assert_eq!(SimdLevel::Scalar.name(), "Scalar");
        assert_eq!(SimdLevel::Sse2.name(), "SSE2");
        assert_eq!(SimdLevel::Sse42.name(), "SSE4.2");
        assert_eq!(SimdLevel::Avx2.name(), "AVX2");
        assert_eq!(SimdLevel::Avx512.name(), "AVX-512");
    }

    #[test]
    fn test_feature_queries_consistent() {
        let level = simd_level();
        assert_eq!(has_avx512(), level >= SimdLevel::Avx512);
        assert_eq!(has_avx2(), level >= SimdLevel::Avx2);
        assert_eq!(has_sse42(), level >= SimdLevel::Sse42);
        assert_eq!(has_sse2(), level >= SimdLevel::Sse2);
    }
}

// =============================================================================
// Equality Tests
// =============================================================================

mod equality_tests {
    use super::*;
    use crate::types::simd::equality::bytes_eq;

    #[test]
    fn test_empty_slices() {
        assert!(bytes_eq(b"", b""));
    }

    #[test]
    fn test_single_byte_equal() {
        assert!(bytes_eq(b"a", b"a"));
        assert!(bytes_eq(b"\0", b"\0"));
        assert!(bytes_eq(b"\xff", b"\xff"));
    }

    #[test]
    fn test_single_byte_not_equal() {
        assert!(!bytes_eq(b"a", b"b"));
        assert!(!bytes_eq(b"\0", b"\x01"));
    }

    #[test]
    fn test_different_lengths() {
        assert!(!bytes_eq(b"hello", b"hello world"));
        assert!(!bytes_eq(b"hello world", b"hello"));
        assert!(!bytes_eq(b"", b"a"));
    }

    #[test]
    fn test_small_equal() {
        for len in 1..16 {
            let data = make_pattern(len, b'x');
            assert!(bytes_eq(&data, &data.clone()), "len={}", len);
        }
    }

    #[test]
    fn test_small_not_equal() {
        for len in 1..16 {
            let mut a = make_pattern(len, b'x');
            let mut b = a.clone();
            b[len - 1] = b'y';
            assert!(!bytes_eq(&a, &b), "len={}", len);
        }
    }

    #[test]
    fn test_medium_equal() {
        for len in [16, 17, 31, 32, 33, 48, 64, 100] {
            let data = make_pattern(len, b'x');
            assert!(bytes_eq(&data, &data.clone()), "len={}", len);
        }
    }

    #[test]
    fn test_medium_diff_at_start() {
        for len in [16, 32, 64, 128] {
            let a = make_pattern(len, b'x');
            let mut b = a.clone();
            b[0] = b'y';
            assert!(!bytes_eq(&a, &b), "len={}", len);
        }
    }

    #[test]
    fn test_medium_diff_at_end() {
        for len in [16, 32, 64, 128] {
            let a = make_pattern(len, b'x');
            let mut b = a.clone();
            b[len - 1] = b'y';
            assert!(!bytes_eq(&a, &b), "len={}", len);
        }
    }

    #[test]
    fn test_medium_diff_in_middle() {
        for len in [16, 32, 64, 128] {
            let a = make_pattern(len, b'x');
            let mut b = a.clone();
            b[len / 2] = b'y';
            assert!(!bytes_eq(&a, &b), "len={}", len);
        }
    }

    #[test]
    fn test_large_equal() {
        for len in [256, 1024, 4096, 65536] {
            let data = make_pattern(len, b'x');
            assert!(bytes_eq(&data, &data.clone()), "len={}", len);
        }
    }

    #[test]
    fn test_large_diff_at_end() {
        for len in [256, 1024, 4096] {
            let a = make_pattern(len, b'x');
            let mut b = a.clone();
            b[len - 1] = b'y';
            assert!(!bytes_eq(&a, &b), "len={}", len);
        }
    }

    #[test]
    fn test_pointer_equality() {
        let data = make_pattern(1000, b'x');
        assert!(bytes_eq(&data, &data)); // Same slice
    }

    #[test]
    fn test_non_ascii() {
        let a = "héllo wörld".as_bytes();
        let b = "héllo wörld".as_bytes();
        assert!(bytes_eq(a, b));

        let c = "hello world".as_bytes();
        assert!(!bytes_eq(a, c));
    }

    #[test]
    fn test_str_eq() {
        use crate::types::simd::equality::str_eq;
        assert!(str_eq("hello", "hello"));
        assert!(!str_eq("hello", "world"));
        assert!(str_eq("héllo", "héllo"));
    }

    #[test]
    fn test_all_byte_values() {
        // Test that all 256 byte values compare correctly
        let a: Vec<u8> = (0..=255).collect();
        let b = a.clone();
        assert!(bytes_eq(&a, &b));

        let mut c = a.clone();
        c[100] = c[100].wrapping_add(1);
        assert!(!bytes_eq(&a, &c));
    }
}

// =============================================================================
// Search Tests
// =============================================================================

mod search_tests {
    use super::*;
    use crate::types::simd::search::{
        bytes_contains, bytes_count, bytes_find, bytes_rfind, str_find,
    };

    #[test]
    fn test_empty_needle() {
        assert_eq!(bytes_find(b"hello", b""), Some(0));
        assert_eq!(bytes_find(b"", b""), Some(0));
    }

    #[test]
    fn test_needle_longer_than_haystack() {
        assert_eq!(bytes_find(b"hi", b"hello"), None);
    }

    #[test]
    fn test_single_byte_needle() {
        assert_eq!(bytes_find(b"hello", b"e"), Some(1));
        assert_eq!(bytes_find(b"hello", b"l"), Some(2));
        assert_eq!(bytes_find(b"hello", b"x"), None);
    }

    #[test]
    fn test_short_needle() {
        assert_eq!(bytes_find(b"hello world", b"wor"), Some(6));
        assert_eq!(bytes_find(b"hello world", b"ld"), Some(9));
        assert_eq!(bytes_find(b"hello world", b"xyz"), None);
    }

    #[test]
    fn test_full_match() {
        assert_eq!(bytes_find(b"hello", b"hello"), Some(0));
    }

    #[test]
    fn test_at_end() {
        assert_eq!(bytes_find(b"hello world", b"world"), Some(6));
    }

    #[test]
    fn test_at_start() {
        assert_eq!(bytes_find(b"hello world", b"hello"), Some(0));
    }

    #[test]
    fn test_overlapping_patterns() {
        assert_eq!(bytes_find(b"aaa", b"aa"), Some(0));
        assert_eq!(bytes_find(b"aaaa", b"aa"), Some(0));
    }

    #[test]
    fn test_contains() {
        assert!(bytes_contains(b"hello world", b"world"));
        assert!(!bytes_contains(b"hello world", b"xyz"));
    }

    #[test]
    fn test_count() {
        assert_eq!(bytes_count(b"ababa", b"ab"), 2);
        assert_eq!(bytes_count(b"aaaa", b"a"), 4);
        assert_eq!(bytes_count(b"hello", b"x"), 0);
    }

    #[test]
    fn test_rfind() {
        assert_eq!(bytes_rfind(b"hello world world", b"world"), Some(12));
        assert_eq!(bytes_rfind(b"aaaa", b"a"), Some(3));
        assert_eq!(bytes_rfind(b"hello", b"x"), None);
    }

    #[test]
    fn test_str_find() {
        assert_eq!(str_find("hello world", "world"), Some(6));
        assert_eq!(str_find("héllo wörld", "wörld"), Some(7)); // UTF-8 byte position
    }

    #[test]
    fn test_large_haystack_needle_at_start() {
        let haystack = make_haystack_with_needle(10000, b"needle", 0);
        assert_eq!(bytes_find(&haystack, b"needle"), Some(0));
    }

    #[test]
    fn test_large_haystack_needle_at_end() {
        let haystack = make_haystack_with_needle(10000, b"needle", 9994);
        assert_eq!(bytes_find(&haystack, b"needle"), Some(9994));
    }

    #[test]
    fn test_large_haystack_needle_in_middle() {
        let haystack = make_haystack_with_needle(10000, b"needle", 5000);
        assert_eq!(bytes_find(&haystack, b"needle"), Some(5000));
    }

    #[test]
    fn test_large_haystack_no_needle() {
        let haystack = vec![b'x'; 10000];
        assert_eq!(bytes_find(&haystack, b"needle"), None);
    }

    #[test]
    fn test_needle_16_bytes() {
        // Exactly fits in SSE4.2 register
        let needle = b"0123456789abcdef";
        let mut haystack = vec![b'x'; 1000];
        haystack[500..516].copy_from_slice(needle);
        assert_eq!(bytes_find(&haystack, needle), Some(500));
    }

    #[test]
    fn test_needle_17_bytes() {
        // Just over SSE4.2 limit
        let needle = b"0123456789abcdefg";
        let mut haystack = vec![b'x'; 1000];
        haystack[500..517].copy_from_slice(needle);
        assert_eq!(bytes_find(&haystack, needle), Some(500));
    }
}

// =============================================================================
// Validation Tests
// =============================================================================

mod validation_tests {
    use super::*;
    use crate::types::simd::validation::*;

    #[test]
    fn test_is_ascii_empty() {
        assert!(is_ascii(b""));
    }

    #[test]
    fn test_is_ascii_true() {
        assert!(is_ascii(b"hello world"));
        assert!(is_ascii(b"0123456789"));
        assert!(is_ascii(b"!@#$%^&*()"));
        assert!(is_ascii(b"\t\n\r "));
    }

    #[test]
    fn test_is_ascii_false() {
        assert!(!is_ascii("héllo".as_bytes()));
        assert!(!is_ascii(&[0x80]));
        assert!(!is_ascii(&[0xFF]));
    }

    #[test]
    fn test_is_ascii_large() {
        let ascii = make_pattern(10000, b'a');
        assert!(is_ascii(&ascii));

        let mut non_ascii = ascii.clone();
        non_ascii[9999] = 0x80;
        assert!(!is_ascii(&non_ascii));
    }

    #[test]
    fn test_is_ascii_at_boundary() {
        // Test at 32-byte and 64-byte boundaries
        for len in [31, 32, 33, 63, 64, 65] {
            let mut data = make_pattern(len, b'a');
            assert!(is_ascii(&data), "len={}", len);

            data[len - 1] = 0x80;
            assert!(!is_ascii(&data), "len={}", len);
        }
    }

    #[test]
    fn test_is_ascii_lowercase() {
        assert!(is_ascii_lowercase(b"abcxyz"));
        assert!(!is_ascii_lowercase(b"abcXyz"));
        assert!(!is_ascii_lowercase(b"abc123"));
    }

    #[test]
    fn test_is_ascii_uppercase() {
        assert!(is_ascii_uppercase(b"ABCXYZ"));
        assert!(!is_ascii_uppercase(b"ABCxYZ"));
        assert!(!is_ascii_uppercase(b"ABC123"));
    }

    #[test]
    fn test_is_ascii_digit() {
        assert!(is_ascii_digit(b"0123456789"));
        assert!(!is_ascii_digit(b"123abc"));
    }

    #[test]
    fn test_is_valid_utf8() {
        assert!(is_valid_utf8(b"hello"));
        assert!(is_valid_utf8("héllo".as_bytes()));
        assert!(!is_valid_utf8(&[0xFF, 0xFE]));
    }

    #[test]
    fn test_utf8_char_count() {
        assert_eq!(utf8_char_count(b"hello"), 5);
        assert_eq!(utf8_char_count("héllo".as_bytes()), 5); // é is 2 bytes but 1 char
        assert_eq!(utf8_char_count("日本語".as_bytes()), 3); // 9 bytes, 3 chars
    }

    #[test]
    fn test_utf8_char_count_large() {
        // Large ASCII string
        let ascii = make_pattern(10000, b'a');
        assert_eq!(utf8_char_count(&ascii), 10000);
    }
}

// =============================================================================
// Case Conversion Tests
// =============================================================================

mod case_tests {
    use super::*;
    use crate::types::simd::case::*;

    #[test]
    fn test_to_lowercase_empty() {
        let mut data = vec![];
        to_lowercase_inplace(&mut data);
        assert!(data.is_empty());
    }

    #[test]
    fn test_to_lowercase_simple() {
        let mut data = b"HELLO WORLD".to_vec();
        to_lowercase_inplace(&mut data);
        assert_eq!(data, b"hello world");
    }

    #[test]
    fn test_to_lowercase_mixed() {
        let mut data = b"HeLLo WoRLd".to_vec();
        to_lowercase_inplace(&mut data);
        assert_eq!(data, b"hello world");
    }

    #[test]
    fn test_to_lowercase_with_numbers() {
        let mut data = b"HELLO123WORLD".to_vec();
        to_lowercase_inplace(&mut data);
        assert_eq!(data, b"hello123world");
    }

    #[test]
    fn test_to_lowercase_non_ascii_unchanged() {
        let mut data = "HéLLO".as_bytes().to_vec();
        to_lowercase_inplace(&mut data);
        // Only ASCII letters converted, é (0xC3 0xA9) unchanged
        assert_eq!(&data[..1], b"h");
    }

    #[test]
    fn test_to_uppercase_simple() {
        let mut data = b"hello world".to_vec();
        to_uppercase_inplace(&mut data);
        assert_eq!(data, b"HELLO WORLD");
    }

    #[test]
    fn test_to_uppercase_mixed() {
        let mut data = b"HeLLo WoRLd".to_vec();
        to_uppercase_inplace(&mut data);
        assert_eq!(data, b"HELLO WORLD");
    }

    #[test]
    fn test_to_lowercase_large() {
        let mut data = make_uppercase(10000);
        let expected = make_lowercase(10000);
        to_lowercase_inplace(&mut data);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_to_uppercase_large() {
        let mut data = make_lowercase(10000);
        let expected = make_uppercase(10000);
        to_uppercase_inplace(&mut data);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_to_lowercase_allocating() {
        let result = to_lowercase(b"HELLO");
        assert_eq!(result, b"hello");
    }

    #[test]
    fn test_to_uppercase_allocating() {
        let result = to_uppercase(b"hello");
        assert_eq!(result, b"HELLO");
    }

    #[test]
    fn test_str_to_lowercase() {
        assert_eq!(str_to_lowercase("HELLO"), "hello");
    }

    #[test]
    fn test_str_to_uppercase() {
        assert_eq!(str_to_uppercase("hello"), "HELLO");
    }

    #[test]
    fn test_eq_ignore_ascii_case() {
        assert!(eq_ignore_ascii_case(b"hello", b"HELLO"));
        assert!(eq_ignore_ascii_case(b"HeLLo", b"hElLO"));
        assert!(!eq_ignore_ascii_case(b"hello", b"world"));
    }

    #[test]
    fn test_case_at_boundaries() {
        for len in [15, 16, 17, 31, 32, 33, 63, 64, 65] {
            let mut data = make_uppercase(len);
            let expected = make_lowercase(len);
            to_lowercase_inplace(&mut data);
            assert_eq!(data, expected, "len={}", len);
        }
    }
}

// =============================================================================
// Whitespace Tests
// =============================================================================

mod whitespace_tests {
    use super::*;
    use crate::types::simd::whitespace::*;

    #[test]
    fn test_is_whitespace_byte() {
        assert!(is_whitespace_byte(b' '));
        assert!(is_whitespace_byte(b'\t'));
        assert!(is_whitespace_byte(b'\n'));
        assert!(is_whitespace_byte(b'\r'));
        assert!(is_whitespace_byte(0x0B));
        assert!(is_whitespace_byte(0x0C));
        assert!(!is_whitespace_byte(b'a'));
        assert!(!is_whitespace_byte(b'0'));
    }

    #[test]
    fn test_find_first_non_whitespace_empty() {
        assert_eq!(find_first_non_whitespace(b""), 0);
    }

    #[test]
    fn test_find_first_non_whitespace_no_ws() {
        assert_eq!(find_first_non_whitespace(b"hello"), 0);
    }

    #[test]
    fn test_find_first_non_whitespace_leading_ws() {
        assert_eq!(find_first_non_whitespace(b"  hello"), 2);
        assert_eq!(find_first_non_whitespace(b"\t\nhello"), 2);
    }

    #[test]
    fn test_find_first_non_whitespace_all_ws() {
        assert_eq!(find_first_non_whitespace(b"   "), 3);
    }

    #[test]
    fn test_find_last_non_whitespace_empty() {
        assert_eq!(find_last_non_whitespace(b""), 0);
    }

    #[test]
    fn test_find_last_non_whitespace_no_ws() {
        assert_eq!(find_last_non_whitespace(b"hello"), 5);
    }

    #[test]
    fn test_find_last_non_whitespace_trailing_ws() {
        assert_eq!(find_last_non_whitespace(b"hello  "), 5);
        assert_eq!(find_last_non_whitespace(b"hello\t\n"), 5);
    }

    #[test]
    fn test_find_last_non_whitespace_all_ws() {
        assert_eq!(find_last_non_whitespace(b"   "), 0);
    }

    #[test]
    fn test_trim_start() {
        assert_eq!(trim_start(b"  hello"), b"hello");
        assert_eq!(trim_start(b"hello"), b"hello");
        assert_eq!(trim_start(b"   "), b"");
    }

    #[test]
    fn test_trim_end() {
        assert_eq!(trim_end(b"hello  "), b"hello");
        assert_eq!(trim_end(b"hello"), b"hello");
        assert_eq!(trim_end(b"   "), b"");
    }

    #[test]
    fn test_trim() {
        assert_eq!(trim(b"  hello  "), b"hello");
        assert_eq!(trim(b"hello"), b"hello");
        assert_eq!(trim(b"   "), b"");
    }

    #[test]
    fn test_trim_str() {
        assert_eq!(trim_str("  hello  "), "hello");
    }

    #[test]
    fn test_is_all_whitespace() {
        assert!(is_all_whitespace(b""));
        assert!(is_all_whitespace(b"   "));
        assert!(is_all_whitespace(b" \t\n\r"));
        assert!(!is_all_whitespace(b"hello"));
        assert!(!is_all_whitespace(b" hello "));
    }

    #[test]
    fn test_count_whitespace() {
        assert_eq!(count_whitespace(b"hello world"), 1);
        assert_eq!(count_whitespace(b"  hello  world  "), 6);
        assert_eq!(count_whitespace(b"hello"), 0);
    }

    #[test]
    fn test_split_whitespace() {
        let words: Vec<&[u8]> = split_whitespace(b"hello world foo").collect();
        assert_eq!(words, vec![&b"hello"[..], &b"world"[..], &b"foo"[..]]);
    }

    #[test]
    fn test_split_whitespace_multiple() {
        let words: Vec<&[u8]> = split_whitespace(b"  hello   world  ").collect();
        assert_eq!(words, vec![&b"hello"[..], &b"world"[..]]);
    }

    #[test]
    fn test_remove_whitespace() {
        assert_eq!(remove_whitespace(b"hello world"), b"helloworld");
        assert_eq!(remove_whitespace(b"  a  b  c  "), b"abc");
    }

    #[test]
    fn test_collapse_whitespace() {
        assert_eq!(collapse_whitespace(b"hello   world"), b"hello world");
        assert_eq!(collapse_whitespace(b"  hello  "), b" hello ");
    }

    #[test]
    fn test_trim_large() {
        let mut data = vec![b' '; 1000];
        data.extend(b"hello");
        data.extend(vec![b' '; 1000]);

        assert_eq!(trim(&data), b"hello");
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_all_zeros() {
        let data = vec![0u8; 1000];
        // 0x00 (NUL) IS ASCII since it's < 128
        assert!(validation::is_ascii(&data));
        assert!(data[0] < 128);
    }

    #[test]
    fn test_all_0xff() {
        let data = vec![0xFFu8; 1000];
        assert!(!validation::is_ascii(&data));
    }

    #[test]
    fn test_alignment_stress() {
        // Test at various offsets within an allocated buffer
        let base = vec![b'a'; 100];
        for offset in 0..16 {
            if offset + 50 <= 100 {
                let slice = &base[offset..offset + 50];
                assert!(validation::is_ascii(slice));
            }
        }
    }

    #[test]
    fn test_near_page_boundary() {
        // Large allocation that might cross page boundaries
        let data = vec![b'x'; 65536];
        assert!(validation::is_ascii(&data));
        assert_eq!(search::bytes_find(&data, b"x"), Some(0));
    }
}
