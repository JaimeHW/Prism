//! SIMD-accelerated byte and substring search.
//!
//! This module centralizes Prism's hot byte-search primitives. Single-byte
//! search uses `memchr`/`memrchr`; multi-byte search uses `memmem`, whose
//! prefilters and two-way fallback keep worst-case behavior linear on periodic
//! inputs such as CPython's adaptive bytes tests.

use memchr::{memchr, memmem, memrchr};

/// Find the first occurrence of `needle` in `haystack`.
#[inline]
pub fn bytes_find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    if needle.len() > haystack.len() {
        return None;
    }
    if needle.len() == 1 {
        return memchr(needle[0], haystack);
    }

    memmem::find(haystack, needle)
}

/// Find the first occurrence of `needle` in `haystack`.
#[inline]
pub fn str_find(haystack: &str, needle: &str) -> Option<usize> {
    bytes_find(haystack.as_bytes(), needle.as_bytes())
}

/// Check if `haystack` contains `needle`.
#[inline]
pub fn bytes_contains(haystack: &[u8], needle: &[u8]) -> bool {
    bytes_find(haystack, needle).is_some()
}

/// Check if `haystack` contains `needle`.
#[inline]
pub fn str_contains(haystack: &str, needle: &str) -> bool {
    bytes_contains(haystack.as_bytes(), needle.as_bytes())
}

/// Count non-overlapping occurrences of `needle` in `haystack`.
#[inline]
pub fn bytes_count(haystack: &[u8], needle: &[u8]) -> usize {
    if needle.is_empty() {
        return haystack.len() + 1;
    }
    if needle.len() > haystack.len() {
        return 0;
    }
    if needle.len() == 1 {
        return memchr::memchr_iter(needle[0], haystack).count();
    }

    let finder = memmem::Finder::new(needle);
    let mut count = 0usize;
    let mut pos = 0usize;
    while let Some(index) = finder.find(&haystack[pos..]) {
        count += 1;
        pos += index + needle.len();
    }
    count
}

/// Find the last occurrence of `needle` in `haystack`.
#[inline]
pub fn bytes_rfind(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(haystack.len());
    }
    if needle.len() > haystack.len() {
        return None;
    }
    if needle.len() == 1 {
        return memrchr(needle[0], haystack);
    }

    memmem::rfind(haystack, needle)
}

/// Find the last occurrence of `needle` in `haystack`.
#[inline]
pub fn str_rfind(haystack: &str, needle: &str) -> Option<usize> {
    bytes_rfind(haystack.as_bytes(), needle.as_bytes())
}
