//! SIMD-accelerated byte and string equality comparison.
//!
//! This module provides vectorized equality checking with automatic dispatch
//! based on CPU capabilities:
//!
//! | Level    | Throughput      | Min Size |
//! |----------|-----------------|----------|
//! | AVX-512  | 64 bytes/iter   | 64       |
//! | AVX2     | 32 bytes/iter   | 32       |
//! | SSE2     | 16 bytes/iter   | 16       |
//! | Scalar   | 8 bytes/iter    | 0        |
//!
//! # Algorithm
//!
//! 1. Early exit on length mismatch
//! 2. Pointer equality check (same backing memory)
//! 3. SIMD comparison of aligned chunks
//! 4. Scalar fallback for tail bytes
//!
//! # Performance Notes
//!
//! - All SIMD paths use unaligned loads (negligible penalty on modern CPUs)
//! - Short strings (&lt;16 bytes) use optimized scalar comparison
//! - Uses `#[cold]` hints for unlikely early-exit paths

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// =============================================================================
// Public API
// =============================================================================

/// Compare two byte slices for equality using SIMD acceleration.
///
/// # Performance
///
/// - O(1) for pointer-equal slices
/// - O(n) with SIMD acceleration for large slices
/// - Automatically selects best SIMD path based on CPU features
///
/// # Examples
///
/// ```ignore
/// use prism_runtime::types::simd::equality::bytes_eq;
///
/// assert!(bytes_eq(b"hello", b"hello"));
/// assert!(!bytes_eq(b"hello", b"world"));
/// ```
#[inline]
pub fn bytes_eq(a: &[u8], b: &[u8]) -> bool {
    // Fast path: length mismatch
    if a.len() != b.len() {
        return false;
    }

    // Fast path: empty slices
    if a.is_empty() {
        return true;
    }

    // Fast path: pointer equality (same backing memory)
    if a.as_ptr() == b.as_ptr() {
        return true;
    }

    let len = a.len();

    // Dispatch based on length and CPU features
    #[cfg(target_arch = "x86_64")]
    {
        if len >= 64 && super::has_avx512() {
            // SAFETY: AVX-512 feature detected, pointers valid for len bytes
            return unsafe { bytes_eq_avx512(a, b) };
        }
        if len >= 32 && super::has_avx2() {
            // SAFETY: AVX2 feature detected, pointers valid for len bytes
            return unsafe { bytes_eq_avx2(a, b) };
        }
        if len >= 16 && super::has_sse2() {
            // SAFETY: SSE2 feature detected (always on x86-64), pointers valid
            return unsafe { bytes_eq_sse2(a, b) };
        }
    }

    // Scalar fallback
    bytes_eq_scalar(a, b)
}

/// Compare two strings for equality using SIMD acceleration.
///
/// This is a thin wrapper around [`bytes_eq`] that operates on UTF-8 bytes.
#[inline]
pub fn str_eq(a: &str, b: &str) -> bool {
    bytes_eq(a.as_bytes(), b.as_bytes())
}

// =============================================================================
// AVX-512 Implementation (64 bytes/iteration)
// =============================================================================

/// AVX-512 equality: Process 64 bytes at a time using 512-bit vectors.
///
/// Uses `cmpeq_epi8_mask` for efficient comparison with direct mask output.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
#[inline]
unsafe fn bytes_eq_avx512(a: &[u8], b: &[u8]) -> bool {
    unsafe {
        let len = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let mut offset = 0usize;

        // Process 64-byte chunks
        while offset + 64 <= len {
            let va = _mm512_loadu_si512(a_ptr.add(offset) as *const __m512i);
            let vb = _mm512_loadu_si512(b_ptr.add(offset) as *const __m512i);

            // Compare bytes, get 64-bit mask of equal positions
            let mask = _mm512_cmpeq_epi8_mask(va, vb);

            // All 64 bytes equal means mask = 0xFFFFFFFFFFFFFFFF
            if mask != !0u64 {
                return false;
            }
            offset += 64;
        }

        // Process remaining 32-byte chunk with AVX2
        if offset + 32 <= len {
            let va = _mm256_loadu_si256(a_ptr.add(offset) as *const __m256i);
            let vb = _mm256_loadu_si256(b_ptr.add(offset) as *const __m256i);
            let cmp = _mm256_cmpeq_epi8(va, vb);
            if _mm256_movemask_epi8(cmp) != -1i32 {
                return false;
            }
            offset += 32;
        }

        // Process remaining 16-byte chunk with SSE2
        if offset + 16 <= len {
            let va = _mm_loadu_si128(a_ptr.add(offset) as *const __m128i);
            let vb = _mm_loadu_si128(b_ptr.add(offset) as *const __m128i);
            let cmp = _mm_cmpeq_epi8(va, vb);
            if _mm_movemask_epi8(cmp) != 0xFFFF {
                return false;
            }
            offset += 16;
        }

        // Scalar tail (0-15 bytes)
        a[offset..] == b[offset..]
    }
}

// =============================================================================
// AVX2 Implementation (32 bytes/iteration)
// =============================================================================

/// AVX2 equality: Process 32 bytes at a time using 256-bit vectors.
///
/// Uses `cmpeq_epi8` + `movemask` for efficient comparison.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn bytes_eq_avx2(a: &[u8], b: &[u8]) -> bool {
    unsafe {
        let len = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let mut offset = 0usize;

        // Unroll loop 2x for better instruction-level parallelism
        while offset + 64 <= len {
            // First 32 bytes
            let va1 = _mm256_loadu_si256(a_ptr.add(offset) as *const __m256i);
            let vb1 = _mm256_loadu_si256(b_ptr.add(offset) as *const __m256i);

            // Second 32 bytes
            let va2 = _mm256_loadu_si256(a_ptr.add(offset + 32) as *const __m256i);
            let vb2 = _mm256_loadu_si256(b_ptr.add(offset + 32) as *const __m256i);

            // Compare both pairs
            let cmp1 = _mm256_cmpeq_epi8(va1, vb1);
            let cmp2 = _mm256_cmpeq_epi8(va2, vb2);

            // Check results
            let mask1 = _mm256_movemask_epi8(cmp1);
            let mask2 = _mm256_movemask_epi8(cmp2);

            if mask1 != -1i32 || mask2 != -1i32 {
                return false;
            }
            offset += 64;
        }

        // Process remaining 32-byte chunk
        if offset + 32 <= len {
            let va = _mm256_loadu_si256(a_ptr.add(offset) as *const __m256i);
            let vb = _mm256_loadu_si256(b_ptr.add(offset) as *const __m256i);
            let cmp = _mm256_cmpeq_epi8(va, vb);
            if _mm256_movemask_epi8(cmp) != -1i32 {
                return false;
            }
            offset += 32;
        }

        // Process remaining 16-byte chunk with SSE2
        if offset + 16 <= len {
            let va = _mm_loadu_si128(a_ptr.add(offset) as *const __m128i);
            let vb = _mm_loadu_si128(b_ptr.add(offset) as *const __m128i);
            let cmp = _mm_cmpeq_epi8(va, vb);
            if _mm_movemask_epi8(cmp) != 0xFFFF {
                return false;
            }
            offset += 16;
        }

        // Scalar tail (0-15 bytes)
        a[offset..] == b[offset..]
    }
}

// =============================================================================
// SSE2 Implementation (16 bytes/iteration)
// =============================================================================

/// SSE2 equality: Process 16 bytes at a time using 128-bit vectors.
///
/// SSE2 is baseline for x86-64, so this is always available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn bytes_eq_sse2(a: &[u8], b: &[u8]) -> bool {
    unsafe {
        let len = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let mut offset = 0usize;

        // Unroll loop 4x for better instruction-level parallelism
        while offset + 64 <= len {
            let va1 = _mm_loadu_si128(a_ptr.add(offset) as *const __m128i);
            let vb1 = _mm_loadu_si128(b_ptr.add(offset) as *const __m128i);
            let va2 = _mm_loadu_si128(a_ptr.add(offset + 16) as *const __m128i);
            let vb2 = _mm_loadu_si128(b_ptr.add(offset + 16) as *const __m128i);
            let va3 = _mm_loadu_si128(a_ptr.add(offset + 32) as *const __m128i);
            let vb3 = _mm_loadu_si128(b_ptr.add(offset + 32) as *const __m128i);
            let va4 = _mm_loadu_si128(a_ptr.add(offset + 48) as *const __m128i);
            let vb4 = _mm_loadu_si128(b_ptr.add(offset + 48) as *const __m128i);

            let cmp1 = _mm_cmpeq_epi8(va1, vb1);
            let cmp2 = _mm_cmpeq_epi8(va2, vb2);
            let cmp3 = _mm_cmpeq_epi8(va3, vb3);
            let cmp4 = _mm_cmpeq_epi8(va4, vb4);

            // Combine all comparisons with AND
            let combined = _mm_and_si128(_mm_and_si128(cmp1, cmp2), _mm_and_si128(cmp3, cmp4));

            if _mm_movemask_epi8(combined) != 0xFFFF {
                return false;
            }
            offset += 64;
        }

        // Process remaining 16-byte chunks
        while offset + 16 <= len {
            let va = _mm_loadu_si128(a_ptr.add(offset) as *const __m128i);
            let vb = _mm_loadu_si128(b_ptr.add(offset) as *const __m128i);
            let cmp = _mm_cmpeq_epi8(va, vb);
            if _mm_movemask_epi8(cmp) != 0xFFFF {
                return false;
            }
            offset += 16;
        }

        // Scalar tail (0-15 bytes)
        a[offset..] == b[offset..]
    }
}

// =============================================================================
// Scalar Implementation
// =============================================================================

/// Scalar equality comparison with word-sized optimization.
///
/// Compares 8 bytes at a time using native word loads.
#[inline]
fn bytes_eq_scalar(a: &[u8], b: &[u8]) -> bool {
    // Use word-sized comparison for better throughput
    let len = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mut offset = 0;

    // Compare 8 bytes at a time using u64
    while offset + 8 <= len {
        // SAFETY: offset + 8 <= len, so reads are within bounds
        let word_a = unsafe { (a_ptr.add(offset) as *const u64).read_unaligned() };
        let word_b = unsafe { (b_ptr.add(offset) as *const u64).read_unaligned() };
        if word_a != word_b {
            return false;
        }
        offset += 8;
    }

    // Compare remaining bytes one at a time
    while offset < len {
        if a[offset] != b[offset] {
            return false;
        }
        offset += 1;
    }

    true
}

// =============================================================================
// Short String Optimization
// =============================================================================

/// Optimized equality for very short strings (0-16 bytes).
///
/// Uses branchless word-sized comparison for predictable performance.
#[inline]
#[allow(dead_code)]
pub fn bytes_eq_short(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let len = a.len();

    if len == 0 {
        return true;
    }

    if len <= 8 {
        // Use overlapping reads for strings 1-8 bytes
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        if len >= 4 {
            // Read first and last 4 bytes (may overlap)
            let a_first = unsafe { (a_ptr as *const u32).read_unaligned() };
            let b_first = unsafe { (b_ptr as *const u32).read_unaligned() };
            let a_last = unsafe { (a_ptr.add(len - 4) as *const u32).read_unaligned() };
            let b_last = unsafe { (b_ptr.add(len - 4) as *const u32).read_unaligned() };
            return a_first == b_first && a_last == b_last;
        }

        if len >= 2 {
            // Read first and last 2 bytes (may overlap)
            let a_first = unsafe { (a_ptr as *const u16).read_unaligned() };
            let b_first = unsafe { (b_ptr as *const u16).read_unaligned() };
            let a_last = unsafe { (a_ptr.add(len - 2) as *const u16).read_unaligned() };
            let b_last = unsafe { (b_ptr.add(len - 2) as *const u16).read_unaligned() };
            return a_first == b_first && a_last == b_last;
        }

        // Single byte
        return a[0] == b[0];
    }

    // 9-16 bytes: use overlapping 8-byte reads
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let a_first = unsafe { (a_ptr as *const u64).read_unaligned() };
    let b_first = unsafe { (b_ptr as *const u64).read_unaligned() };
    let a_last = unsafe { (a_ptr.add(len - 8) as *const u64).read_unaligned() };
    let b_last = unsafe { (b_ptr.add(len - 8) as *const u64).read_unaligned() };
    a_first == b_first && a_last == b_last
}
