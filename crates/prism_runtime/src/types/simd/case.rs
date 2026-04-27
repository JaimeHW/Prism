//! SIMD-accelerated ASCII case conversion.
//!
//! This module provides fast lowercase/uppercase conversion for ASCII text:
//!
//! | Operation    | Strategy      | Throughput     |
//! |--------------|---------------|----------------|
//! | to_lowercase | AVX2 32B/iter | ~16 GB/s       |
//! | to_uppercase | AVX2 32B/iter | ~16 GB/s       |
//!
//! # Algorithm
//!
//! ASCII case conversion uses a simple bitwise operation:
//! - uppercase = byte & ~0x20 (clear bit 5) for a-z
//! - lowercase = byte | 0x20 (set bit 5) for A-Z
//!
//! The SIMD version uses range checks to only modify letters in the target range.
//!
//! # Non-ASCII Handling
//!
//! These functions only convert ASCII letters. Non-ASCII bytes (≥128) are
//! passed through unchanged. For full Unicode case mapping, use the standard
//! library's `str::to_lowercase()` and `str::to_uppercase()`.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// =============================================================================
// Public API
// =============================================================================

/// Convert ASCII bytes to lowercase in-place.
///
/// Only converts ASCII uppercase letters (A-Z) to lowercase (a-z).
/// Non-ASCII bytes are left unchanged.
///
/// # Examples
///
/// ```ignore
/// let mut data = b"Hello World".to_vec();
/// to_lowercase_inplace(&mut data);
/// assert_eq!(data, b"hello world");
/// ```
#[inline]
pub fn to_lowercase_inplace(data: &mut [u8]) {
    if data.is_empty() {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if super::has_avx2() && data.len() >= 32 {
            // SAFETY: AVX2 detected, length checked, mutable access
            unsafe {
                to_lowercase_avx2(data);
            }
            return;
        }
        if super::has_sse2() && data.len() >= 16 {
            // SAFETY: SSE2 available, length checked
            unsafe {
                to_lowercase_sse2(data);
            }
            return;
        }
    }

    to_lowercase_scalar(data)
}

/// Convert ASCII bytes to uppercase in-place.
///
/// Only converts ASCII lowercase letters (a-z) to uppercase (A-Z).
/// Non-ASCII bytes are left unchanged.
#[inline]
pub fn to_uppercase_inplace(data: &mut [u8]) {
    if data.is_empty() {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if super::has_avx2() && data.len() >= 32 {
            // SAFETY: AVX2 detected, length checked, mutable access
            unsafe {
                to_uppercase_avx2(data);
            }
            return;
        }
        if super::has_sse2() && data.len() >= 16 {
            // SAFETY: SSE2 available, length checked
            unsafe {
                to_uppercase_sse2(data);
            }
            return;
        }
    }

    to_uppercase_scalar(data)
}

/// Convert ASCII bytes to lowercase, returning a new vector.
#[inline]
pub fn to_lowercase(data: &[u8]) -> Vec<u8> {
    let mut result = data.to_vec();
    to_lowercase_inplace(&mut result);
    result
}

/// Convert ASCII bytes to uppercase, returning a new vector.
#[inline]
pub fn to_uppercase(data: &[u8]) -> Vec<u8> {
    let mut result = data.to_vec();
    to_uppercase_inplace(&mut result);
    result
}

/// Convert ASCII string to lowercase.
#[inline]
pub fn str_to_lowercase(s: &str) -> String {
    // SAFETY: We only modify ASCII bytes, preserving UTF-8 validity
    let bytes = to_lowercase(s.as_bytes());
    unsafe { String::from_utf8_unchecked(bytes) }
}

/// Convert ASCII string to uppercase.
#[inline]
pub fn str_to_uppercase(s: &str) -> String {
    // SAFETY: We only modify ASCII bytes, preserving UTF-8 validity
    let bytes = to_uppercase(s.as_bytes());
    unsafe { String::from_utf8_unchecked(bytes) }
}

// =============================================================================
// AVX2 Implementation
// =============================================================================

/// AVX2 lowercase conversion: 32 bytes at a time.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn to_lowercase_avx2(data: &mut [u8]) {
    unsafe {
        let len = data.len();
        let ptr = data.as_mut_ptr();
        let mut offset = 0;

        // Constants for range check A-Z
        let upper_a = _mm256_set1_epi8(b'A' as i8);
        let upper_z = _mm256_set1_epi8(b'Z' as i8);
        let case_bit = _mm256_set1_epi8(0x20);

        while offset + 32 <= len {
            let chunk = _mm256_loadu_si256(ptr.add(offset) as *const __m256i);

            // Check which bytes are in range A-Z
            // byte >= 'A' && byte <= 'Z'
            let ge_a = _mm256_cmpgt_epi8(chunk, _mm256_sub_epi8(upper_a, _mm256_set1_epi8(1)));
            let le_z = _mm256_cmpgt_epi8(_mm256_add_epi8(upper_z, _mm256_set1_epi8(1)), chunk);
            let is_upper = _mm256_and_si256(ge_a, le_z);

            // Set bit 5 only for uppercase letters
            let to_add = _mm256_and_si256(is_upper, case_bit);
            let result = _mm256_or_si256(chunk, to_add);

            _mm256_storeu_si256(ptr.add(offset) as *mut __m256i, result);
            offset += 32;
        }

        // Handle remaining 16-byte chunk
        if offset + 16 <= len {
            let upper_a_sse = _mm_set1_epi8(b'A' as i8);
            let upper_z_sse = _mm_set1_epi8(b'Z' as i8);
            let case_bit_sse = _mm_set1_epi8(0x20);

            let chunk = _mm_loadu_si128(ptr.add(offset) as *const __m128i);

            let ge_a = _mm_cmpgt_epi8(chunk, _mm_sub_epi8(upper_a_sse, _mm_set1_epi8(1)));
            let le_z = _mm_cmpgt_epi8(_mm_add_epi8(upper_z_sse, _mm_set1_epi8(1)), chunk);
            let is_upper = _mm_and_si128(ge_a, le_z);

            let to_add = _mm_and_si128(is_upper, case_bit_sse);
            let result = _mm_or_si128(chunk, to_add);

            _mm_storeu_si128(ptr.add(offset) as *mut __m128i, result);
            offset += 16;
        }

        // Scalar tail
        for i in offset..len {
            let b = *ptr.add(i);
            if b.is_ascii_uppercase() {
                *ptr.add(i) = b | 0x20;
            }
        }
    }
}

/// AVX2 uppercase conversion: 32 bytes at a time.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn to_uppercase_avx2(data: &mut [u8]) {
    unsafe {
        let len = data.len();
        let ptr = data.as_mut_ptr();
        let mut offset = 0;

        // Constants for range check a-z
        let lower_a = _mm256_set1_epi8(b'a' as i8);
        let lower_z = _mm256_set1_epi8(b'z' as i8);
        let case_bit = _mm256_set1_epi8(0x20);

        while offset + 32 <= len {
            let chunk = _mm256_loadu_si256(ptr.add(offset) as *const __m256i);

            // Check which bytes are in range a-z
            let ge_a = _mm256_cmpgt_epi8(chunk, _mm256_sub_epi8(lower_a, _mm256_set1_epi8(1)));
            let le_z = _mm256_cmpgt_epi8(_mm256_add_epi8(lower_z, _mm256_set1_epi8(1)), chunk);
            let is_lower = _mm256_and_si256(ge_a, le_z);

            // XOR with 0x20 where is_lower to clear bit 5
            let to_xor = _mm256_and_si256(is_lower, case_bit);
            let result = _mm256_xor_si256(chunk, to_xor);

            _mm256_storeu_si256(ptr.add(offset) as *mut __m256i, result);
            offset += 32;
        }

        // Handle remaining 16-byte chunk
        if offset + 16 <= len {
            let lower_a_sse = _mm_set1_epi8(b'a' as i8);
            let lower_z_sse = _mm_set1_epi8(b'z' as i8);
            let case_bit_sse = _mm_set1_epi8(0x20);

            let chunk = _mm_loadu_si128(ptr.add(offset) as *const __m128i);

            let ge_a = _mm_cmpgt_epi8(chunk, _mm_sub_epi8(lower_a_sse, _mm_set1_epi8(1)));
            let le_z = _mm_cmpgt_epi8(_mm_add_epi8(lower_z_sse, _mm_set1_epi8(1)), chunk);
            let is_lower = _mm_and_si128(ge_a, le_z);

            let to_xor = _mm_and_si128(is_lower, case_bit_sse);
            let result = _mm_xor_si128(chunk, to_xor);

            _mm_storeu_si128(ptr.add(offset) as *mut __m128i, result);
            offset += 16;
        }

        // Scalar tail
        for i in offset..len {
            let b = *ptr.add(i);
            if b.is_ascii_lowercase() {
                *ptr.add(i) = b & !0x20;
            }
        }
    }
}

// =============================================================================
// SSE2 Implementation
// =============================================================================

/// SSE2 lowercase conversion: 16 bytes at a time.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn to_lowercase_sse2(data: &mut [u8]) {
    unsafe {
        let len = data.len();
        let ptr = data.as_mut_ptr();
        let mut offset = 0;

        let upper_a = _mm_set1_epi8(b'A' as i8);
        let upper_z = _mm_set1_epi8(b'Z' as i8);
        let case_bit = _mm_set1_epi8(0x20);

        while offset + 16 <= len {
            let chunk = _mm_loadu_si128(ptr.add(offset) as *const __m128i);

            // Check range A-Z
            let ge_a = _mm_cmpgt_epi8(chunk, _mm_sub_epi8(upper_a, _mm_set1_epi8(1)));
            let le_z = _mm_cmpgt_epi8(_mm_add_epi8(upper_z, _mm_set1_epi8(1)), chunk);
            let is_upper = _mm_and_si128(ge_a, le_z);

            // Set bit 5 for uppercase letters
            let to_add = _mm_and_si128(is_upper, case_bit);
            let result = _mm_or_si128(chunk, to_add);

            _mm_storeu_si128(ptr.add(offset) as *mut __m128i, result);
            offset += 16;
        }

        // Scalar tail
        for i in offset..len {
            let b = *ptr.add(i);
            if b.is_ascii_uppercase() {
                *ptr.add(i) = b | 0x20;
            }
        }
    }
}

/// SSE2 uppercase conversion: 16 bytes at a time.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn to_uppercase_sse2(data: &mut [u8]) {
    unsafe {
        let len = data.len();
        let ptr = data.as_mut_ptr();
        let mut offset = 0;

        let lower_a = _mm_set1_epi8(b'a' as i8);
        let lower_z = _mm_set1_epi8(b'z' as i8);
        let case_bit = _mm_set1_epi8(0x20);

        while offset + 16 <= len {
            let chunk = _mm_loadu_si128(ptr.add(offset) as *const __m128i);

            // Check range a-z
            let ge_a = _mm_cmpgt_epi8(chunk, _mm_sub_epi8(lower_a, _mm_set1_epi8(1)));
            let le_z = _mm_cmpgt_epi8(_mm_add_epi8(lower_z, _mm_set1_epi8(1)), chunk);
            let is_lower = _mm_and_si128(ge_a, le_z);

            // XOR with 0x20 for lowercase letters (clears bit 5)
            let to_xor = _mm_and_si128(is_lower, case_bit);
            let result = _mm_xor_si128(chunk, to_xor);

            _mm_storeu_si128(ptr.add(offset) as *mut __m128i, result);
            offset += 16;
        }

        // Scalar tail
        for i in offset..len {
            let b = *ptr.add(i);
            if b.is_ascii_lowercase() {
                *ptr.add(i) = b & !0x20;
            }
        }
    }
}

// =============================================================================
// Scalar Implementation
// =============================================================================

/// Scalar lowercase conversion.
#[inline]
fn to_lowercase_scalar(data: &mut [u8]) {
    for b in data.iter_mut() {
        if b.is_ascii_uppercase() {
            *b |= 0x20;
        }
    }
}

/// Scalar uppercase conversion.
#[inline]
fn to_uppercase_scalar(data: &mut [u8]) {
    for b in data.iter_mut() {
        if b.is_ascii_lowercase() {
            *b &= !0x20;
        }
    }
}

// =============================================================================
// Case-Insensitive Comparison
// =============================================================================

/// Compare two byte slices for equality, ignoring ASCII case.
///
/// Only ASCII letters are compared case-insensitively. Non-ASCII bytes
/// must match exactly.
#[inline]
pub fn eq_ignore_ascii_case(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    // For now, scalar implementation
    // TODO: SIMD implementation
    a.iter()
        .zip(b.iter())
        .all(|(&x, &y)| x.eq_ignore_ascii_case(&y))
}

/// Compare two strings for equality, ignoring ASCII case.
#[inline]
pub fn str_eq_ignore_ascii_case(a: &str, b: &str) -> bool {
    eq_ignore_ascii_case(a.as_bytes(), b.as_bytes())
}
