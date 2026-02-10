//! SIMD-accelerated ASCII and UTF-8 validation.
//!
//! This module provides fast validation checks for string content:
//!
//! - **ASCII validation**: Check if all bytes are < 128
//! - **Character class checks**: lowercase, uppercase, alphanumeric, etc.
//!
//! # Performance
//!
//! | Operation         | Strategy      | Throughput     |
//! |-------------------|---------------|----------------|
//! | is_ascii          | AVX2 32B/iter | ~32 GB/s       |
//! | is_ascii_lowercase| AVX2 ranges   | ~24 GB/s       |
//! | is_ascii_uppercase| AVX2 ranges   | ~24 GB/s       |
//!
//! # Algorithm
//!
//! ASCII validation uses a simple high-bit check: if any byte has bit 7 set,
//! the string is not ASCII. This can be done very efficiently with SIMD.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// =============================================================================
// ASCII Validation
// =============================================================================

/// Check if all bytes in the slice are ASCII (< 128).
///
/// # Performance
///
/// - O(1) for empty slices
/// - O(n/32) with AVX2, O(n/16) with SSE2
/// - Uses `movemask` to quickly detect any high bits
///
/// # Examples
///
/// ```ignore
/// use prism_runtime::types::simd::validation::is_ascii;
///
/// assert!(is_ascii(b"hello world"));
/// assert!(!is_ascii("héllo".as_bytes())); // é is not ASCII
/// ```
#[inline]
pub fn is_ascii(data: &[u8]) -> bool {
    if data.is_empty() {
        return true;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if super::has_avx2() && data.len() >= 32 {
            // SAFETY: AVX2 detected, length checked
            return unsafe { is_ascii_avx2(data) };
        }
        if super::has_sse2() && data.len() >= 16 {
            // SAFETY: SSE2 available, length checked
            return unsafe { is_ascii_sse2(data) };
        }
    }

    // Scalar fallback
    is_ascii_scalar(data)
}

/// AVX2 ASCII check: 32 bytes at a time.
///
/// Checks if any byte has the high bit set using `movemask`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn is_ascii_avx2(data: &[u8]) -> bool {
    unsafe {
        let len = data.len();
        let ptr = data.as_ptr();
        let high_bit = _mm256_set1_epi8(0x80u8 as i8);
        let mut offset = 0;

        // Process 32 bytes at a time
        while offset + 32 <= len {
            let chunk = _mm256_loadu_si256(ptr.add(offset) as *const __m256i);
            let has_high = _mm256_and_si256(chunk, high_bit);

            if _mm256_movemask_epi8(has_high) != 0 {
                return false;
            }
            offset += 32;
        }

        // Handle remaining 16-byte chunk
        if offset + 16 <= len {
            let high_bit_sse = _mm_set1_epi8(0x80u8 as i8);
            let chunk = _mm_loadu_si128(ptr.add(offset) as *const __m128i);
            let has_high = _mm_and_si128(chunk, high_bit_sse);

            if _mm_movemask_epi8(has_high) != 0 {
                return false;
            }
            offset += 16;
        }

        // Scalar tail
        for i in offset..len {
            if *ptr.add(i) >= 0x80 {
                return false;
            }
        }

        true
    }
}

/// SSE2 ASCII check: 16 bytes at a time.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn is_ascii_sse2(data: &[u8]) -> bool {
    unsafe {
        let len = data.len();
        let ptr = data.as_ptr();
        let high_bit = _mm_set1_epi8(0x80u8 as i8);
        let mut offset = 0;

        while offset + 16 <= len {
            let chunk = _mm_loadu_si128(ptr.add(offset) as *const __m128i);
            let has_high = _mm_and_si128(chunk, high_bit);

            if _mm_movemask_epi8(has_high) != 0 {
                return false;
            }
            offset += 16;
        }

        // Scalar tail
        for i in offset..len {
            if *ptr.add(i) >= 0x80 {
                return false;
            }
        }

        true
    }
}

/// Scalar ASCII check.
#[inline]
fn is_ascii_scalar(data: &[u8]) -> bool {
    // Use word-sized checks for better throughput
    let mut offset = 0;
    let len = data.len();
    let ptr = data.as_ptr();

    // Check 8 bytes at a time
    const HIGH_BITS: u64 = 0x8080808080808080;

    while offset + 8 <= len {
        // SAFETY: bounds checked above
        let word = unsafe { (ptr.add(offset) as *const u64).read_unaligned() };
        if word & HIGH_BITS != 0 {
            return false;
        }
        offset += 8;
    }

    // Scalar tail
    while offset < len {
        if data[offset] >= 0x80 {
            return false;
        }
        offset += 1;
    }

    true
}

// =============================================================================
// ASCII Character Class Checks
// =============================================================================

/// Check if all bytes are ASCII lowercase letters (a-z).
///
/// Empty strings return true (vacuous truth).
#[inline]
pub fn is_ascii_lowercase(data: &[u8]) -> bool {
    if data.is_empty() {
        return true;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if super::has_avx2() && data.len() >= 32 {
            // SAFETY: AVX2 detected, length checked
            return unsafe { is_ascii_range_avx2(data, b'a', b'z') };
        }
    }

    // Scalar fallback
    data.iter().all(|&b| b.is_ascii_lowercase())
}

/// Check if all bytes are ASCII uppercase letters (A-Z).
#[inline]
pub fn is_ascii_uppercase(data: &[u8]) -> bool {
    if data.is_empty() {
        return true;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if super::has_avx2() && data.len() >= 32 {
            // SAFETY: AVX2 detected, length checked
            return unsafe { is_ascii_range_avx2(data, b'A', b'Z') };
        }
    }

    // Scalar fallback
    data.iter().all(|&b| b.is_ascii_uppercase())
}

/// Check if all bytes are ASCII alphabetic (a-z or A-Z).
#[inline]
pub fn is_ascii_alphabetic(data: &[u8]) -> bool {
    if data.is_empty() {
        return true;
    }

    // Scalar for now - range check is more complex for two ranges
    data.iter().all(|&b| b.is_ascii_alphabetic())
}

/// Check if all bytes are ASCII digits (0-9).
#[inline]
pub fn is_ascii_digit(data: &[u8]) -> bool {
    if data.is_empty() {
        return true;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if super::has_avx2() && data.len() >= 32 {
            // SAFETY: AVX2 detected, length checked
            return unsafe { is_ascii_range_avx2(data, b'0', b'9') };
        }
    }

    // Scalar fallback
    data.iter().all(|&b| b.is_ascii_digit())
}

/// Check if all bytes are ASCII alphanumeric (a-z, A-Z, 0-9).
#[inline]
pub fn is_ascii_alphanumeric(data: &[u8]) -> bool {
    if data.is_empty() {
        return true;
    }

    // Scalar for now - multiple ranges
    data.iter().all(|&b| b.is_ascii_alphanumeric())
}

/// Check if all bytes are ASCII whitespace.
#[inline]
pub fn is_ascii_whitespace(data: &[u8]) -> bool {
    if data.is_empty() {
        return true;
    }

    // Whitespace is ' ', '\t', '\n', '\r', '\x0B', '\x0C'
    data.iter().all(|&b| b.is_ascii_whitespace())
}

// =============================================================================
// AVX2 Range Check Implementation
// =============================================================================

/// AVX2 range check: verify all bytes are in [low, high].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn is_ascii_range_avx2(data: &[u8], low: u8, high: u8) -> bool {
    unsafe {
        let len = data.len();
        let ptr = data.as_ptr();
        let mut offset = 0;

        // Bias trick: subtract low, then check if result < (high - low + 1)
        // This converts a range check into a single unsigned comparison
        let low_vec = _mm256_set1_epi8(low as i8);
        let range = _mm256_set1_epi8((high - low + 1) as i8);

        while offset + 32 <= len {
            let chunk = _mm256_loadu_si256(ptr.add(offset) as *const __m256i);

            // Subtract low (saturating for unsigned comparison)
            let biased = _mm256_sub_epi8(chunk, low_vec);

            // Compare if biased >= range (i.e., out of range)
            // We use saturating subtraction and check for zero
            let out_of_range =
                _mm256_cmpgt_epi8(biased, _mm256_sub_epi8(range, _mm256_set1_epi8(1)));

            // Also check if original was less than low (underflow in sub)
            let below_low = _mm256_cmpgt_epi8(low_vec, chunk);

            let invalid = _mm256_or_si256(out_of_range, below_low);

            if _mm256_movemask_epi8(invalid) != 0 {
                return false;
            }
            offset += 32;
        }

        // Scalar tail
        for i in offset..len {
            let b = *ptr.add(i);
            if b < low || b > high {
                return false;
            }
        }

        true
    }
}

// =============================================================================
// Count Functions
// =============================================================================

/// Count bytes that are ASCII (< 128).
#[inline]
pub fn count_ascii(data: &[u8]) -> usize {
    // For pure ASCII data, all bytes are ASCII
    // For mixed data, we count byte by byte
    // TODO: SIMD popcnt optimization

    data.iter().filter(|&&b| b < 128).count()
}

/// Count bytes that match a predicate.
#[inline]
pub fn count_matching<F>(data: &[u8], predicate: F) -> usize
where
    F: Fn(u8) -> bool,
{
    data.iter().filter(|&&b| predicate(b)).count()
}

// =============================================================================
// UTF-8 Validation (Future)
// =============================================================================

/// Validate that data is valid UTF-8.
///
/// This is a placeholder for future SIMD UTF-8 validation.
/// Currently delegates to std's implementation.
#[inline]
pub fn is_valid_utf8(data: &[u8]) -> bool {
    std::str::from_utf8(data).is_ok()
}

/// Get the number of UTF-8 characters (codepoints) in valid UTF-8 data.
///
/// Counts by checking leading bytes (those not starting with 10xxxxxx).
#[inline]
pub fn utf8_char_count(data: &[u8]) -> usize {
    // A byte is a leading byte if it doesn't match 10xxxxxx pattern
    // i.e., (byte & 0xC0) != 0x80

    #[cfg(target_arch = "x86_64")]
    {
        if super::has_avx2() && data.len() >= 32 {
            // SAFETY: AVX2 detected, length checked
            return unsafe { utf8_char_count_avx2(data) };
        }
    }

    // Scalar fallback
    data.iter().filter(|&&b| (b & 0xC0) != 0x80).count()
}

/// AVX2 UTF-8 character count.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn utf8_char_count_avx2(data: &[u8]) -> usize {
    unsafe {
        let len = data.len();
        let ptr = data.as_ptr();
        let mut count = 0usize;
        let mut offset = 0;

        // Mask to check if byte is a continuation byte (10xxxxxx)
        let cont_mask = _mm256_set1_epi8(0xC0u8 as i8);
        let cont_pattern = _mm256_set1_epi8(0x80u8 as i8);

        while offset + 32 <= len {
            let chunk = _mm256_loadu_si256(ptr.add(offset) as *const __m256i);

            // (chunk & 0xC0) == 0x80 for continuation bytes
            let masked = _mm256_and_si256(chunk, cont_mask);
            let is_continuation = _mm256_cmpeq_epi8(masked, cont_pattern);

            // Count non-continuation bytes (leading bytes)
            let mask = _mm256_movemask_epi8(is_continuation) as u32;
            count += 32 - mask.count_ones() as usize;

            offset += 32;
        }

        // Scalar tail
        for i in offset..len {
            if (*ptr.add(i) & 0xC0) != 0x80 {
                count += 1;
            }
        }

        count
    }
}
