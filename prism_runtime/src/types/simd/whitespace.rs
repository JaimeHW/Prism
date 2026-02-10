//! SIMD-accelerated whitespace detection and trimming.
//!
//! This module provides fast whitespace operations:
//!
//! - **Detection**: Check if all bytes are whitespace
//! - **Trimming**: Find leading/trailing whitespace boundaries
//! - **Counting**: Count whitespace and non-whitespace bytes
//!
//! # Whitespace Definition
//!
//! ASCII whitespace characters: ' ' (0x20), '\t' (0x09), '\n' (0x0A),
//! '\r' (0x0D), '\x0B' (vertical tab), '\x0C' (form feed).
//!
//! # Algorithm
//!
//! Uses a lookup table approach: whitespace characters are 0x09-0x0D and 0x20.
//! We can check this with a range test: (b == 0x20) || (b >= 0x09 && b <= 0x0D).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// =============================================================================
// Public API
// =============================================================================

/// Check if a byte is ASCII whitespace.
#[inline]
pub const fn is_whitespace_byte(b: u8) -> bool {
    b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' || b == 0x0B || b == 0x0C
}

/// Find the index of the first non-whitespace byte.
///
/// Returns the length if all bytes are whitespace (or empty).
#[inline]
pub fn find_first_non_whitespace(data: &[u8]) -> usize {
    if data.is_empty() {
        return 0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if super::has_avx2() && data.len() >= 32 {
            // SAFETY: AVX2 detected, length checked
            return unsafe { find_first_non_whitespace_avx2(data) };
        }
    }

    find_first_non_whitespace_scalar(data)
}

/// Find the index after the last non-whitespace byte.
///
/// Returns 0 if all bytes are whitespace (or empty).
#[inline]
pub fn find_last_non_whitespace(data: &[u8]) -> usize {
    if data.is_empty() {
        return 0;
    }

    // Reverse search - currently scalar only
    // TODO: SIMD reverse whitespace search
    find_last_non_whitespace_scalar(data)
}

/// Get the slice with leading whitespace removed.
#[inline]
pub fn trim_start(data: &[u8]) -> &[u8] {
    let start = find_first_non_whitespace(data);
    &data[start..]
}

/// Get the slice with trailing whitespace removed.
#[inline]
pub fn trim_end(data: &[u8]) -> &[u8] {
    let end = find_last_non_whitespace(data);
    &data[..end]
}

/// Get the slice with leading and trailing whitespace removed.
#[inline]
pub fn trim(data: &[u8]) -> &[u8] {
    trim_end(trim_start(data))
}

/// Trim string, returning a string slice.
#[inline]
pub fn trim_str(s: &str) -> &str {
    // SAFETY: trim only removes ASCII whitespace bytes, preserving UTF-8 validity
    let bytes = trim(s.as_bytes());
    unsafe { std::str::from_utf8_unchecked(bytes) }
}

/// Trim leading whitespace from string.
#[inline]
pub fn trim_start_str(s: &str) -> &str {
    let bytes = trim_start(s.as_bytes());
    unsafe { std::str::from_utf8_unchecked(bytes) }
}

/// Trim trailing whitespace from string.
#[inline]
pub fn trim_end_str(s: &str) -> &str {
    let bytes = trim_end(s.as_bytes());
    unsafe { std::str::from_utf8_unchecked(bytes) }
}

/// Check if all bytes are whitespace.
#[inline]
pub fn is_all_whitespace(data: &[u8]) -> bool {
    if data.is_empty() {
        return true;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if super::has_avx2() && data.len() >= 32 {
            // SAFETY: AVX2 detected, length checked
            return unsafe { is_all_whitespace_avx2(data) };
        }
    }

    is_all_whitespace_scalar(data)
}

/// Count whitespace bytes.
#[inline]
pub fn count_whitespace(data: &[u8]) -> usize {
    data.iter().filter(|&&b| is_whitespace_byte(b)).count()
}

/// Count non-whitespace bytes.
#[inline]
pub fn count_non_whitespace(data: &[u8]) -> usize {
    data.len() - count_whitespace(data)
}

// =============================================================================
// AVX2 Implementation
// =============================================================================

/// AVX2 find first non-whitespace byte.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn find_first_non_whitespace_avx2(data: &[u8]) -> usize {
    unsafe {
        let len = data.len();
        let ptr = data.as_ptr();
        let mut offset = 0;

        // Whitespace check: (b == 0x20) || (b >= 0x09 && b <= 0x0D)
        let space = _mm256_set1_epi8(0x20);
        let tab_low = _mm256_set1_epi8(0x09);
        let tab_high = _mm256_set1_epi8(0x0D);

        while offset + 32 <= len {
            let chunk = _mm256_loadu_si256(ptr.add(offset) as *const __m256i);

            // Check for space (0x20)
            let is_space = _mm256_cmpeq_epi8(chunk, space);

            // Check for 0x09-0x0D range
            // (chunk >= 0x09) && (chunk <= 0x0D)
            let ge_tab_low =
                _mm256_cmpgt_epi8(chunk, _mm256_sub_epi8(tab_low, _mm256_set1_epi8(1)));
            let le_tab_high =
                _mm256_cmpgt_epi8(_mm256_add_epi8(tab_high, _mm256_set1_epi8(1)), chunk);
            let is_tab_range = _mm256_and_si256(ge_tab_low, le_tab_high);

            // Combine: is_space || is_tab_range
            let is_whitespace = _mm256_or_si256(is_space, is_tab_range);

            // Invert to find non-whitespace
            let is_not_whitespace = _mm256_andnot_si256(is_whitespace, _mm256_set1_epi8(-1i8));

            let mask = _mm256_movemask_epi8(is_not_whitespace) as u32;

            if mask != 0 {
                return offset + mask.trailing_zeros() as usize;
            }
            offset += 32;
        }

        // Scalar tail
        for i in offset..len {
            if !is_whitespace_byte(*ptr.add(i)) {
                return i;
            }
        }

        len
    }
}

/// AVX2 check if all bytes are whitespace.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn is_all_whitespace_avx2(data: &[u8]) -> bool {
    // Find first non-whitespace - if it equals len, all are whitespace
    unsafe { find_first_non_whitespace_avx2(data) == data.len() }
}

// =============================================================================
// Scalar Implementation
// =============================================================================

/// Scalar find first non-whitespace.
#[inline]
fn find_first_non_whitespace_scalar(data: &[u8]) -> usize {
    for (i, &b) in data.iter().enumerate() {
        if !is_whitespace_byte(b) {
            return i;
        }
    }
    data.len()
}

/// Scalar find last non-whitespace (returns index after last non-ws byte).
#[inline]
fn find_last_non_whitespace_scalar(data: &[u8]) -> usize {
    for (i, &b) in data.iter().enumerate().rev() {
        if !is_whitespace_byte(b) {
            return i + 1;
        }
    }
    0
}

/// Scalar check if all whitespace.
#[inline]
fn is_all_whitespace_scalar(data: &[u8]) -> bool {
    data.iter().all(|&b| is_whitespace_byte(b))
}

// =============================================================================
// Split on Whitespace
// =============================================================================

/// Iterator over whitespace-separated words.
pub struct WhitespaceSplitter<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> WhitespaceSplitter<'a> {
    /// Create a new whitespace splitter.
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
}

impl<'a> Iterator for WhitespaceSplitter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        let len = self.data.len();

        // Skip leading whitespace
        while self.pos < len && is_whitespace_byte(self.data[self.pos]) {
            self.pos += 1;
        }

        if self.pos >= len {
            return None;
        }

        // Find end of word
        let start = self.pos;
        while self.pos < len && !is_whitespace_byte(self.data[self.pos]) {
            self.pos += 1;
        }

        Some(&self.data[start..self.pos])
    }
}

/// Split bytes on whitespace, returning an iterator.
#[inline]
pub fn split_whitespace(data: &[u8]) -> WhitespaceSplitter<'_> {
    WhitespaceSplitter::new(data)
}

/// Split string on whitespace, returning an iterator of string slices.
#[inline]
pub fn split_whitespace_str(s: &str) -> impl Iterator<Item = &str> {
    WhitespaceSplitter::new(s.as_bytes()).map(|bytes| {
        // SAFETY: We're splitting on ASCII whitespace, so each segment is valid UTF-8
        unsafe { std::str::from_utf8_unchecked(bytes) }
    })
}

// =============================================================================
// Specialized Whitespace Removal
// =============================================================================

/// Remove all whitespace from a byte slice, returning a new vector.
#[inline]
pub fn remove_whitespace(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(data.len());
    for &b in data {
        if !is_whitespace_byte(b) {
            result.push(b);
        }
    }
    result
}

/// Collapse multiple consecutive whitespace into single spaces.
#[inline]
pub fn collapse_whitespace(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(data.len());
    let mut in_whitespace = false;

    for &b in data {
        if is_whitespace_byte(b) {
            if !in_whitespace {
                result.push(b' ');
                in_whitespace = true;
            }
        } else {
            result.push(b);
            in_whitespace = false;
        }
    }

    result
}
