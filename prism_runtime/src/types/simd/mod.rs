//! SIMD-accelerated string operations with runtime CPU dispatch.
//!
//! This module provides vectorized implementations of common string operations
//! with automatic dispatch based on detected CPU features:
//!
//! - **AVX-512**: 64-byte operations (Skylake-X+, Zen4+)
//! - **AVX2**: 32-byte operations (Haswell+, Zen+)
//! - **SSE4.2**: 16-byte + PCMPESTRI (Nehalem+)
//! - **SSE2**: 16-byte basic operations (all x86-64)
//! - **Scalar**: Fallback for edge cases
//!
//! # Architecture
//!
//! The module is organized by operation type:
//! - [`equality`]: Fast byte/string equality comparison
//! - [`search`]: Substring search using PCMPESTRI
//! - [`validation`]: ASCII/UTF-8 validation
//! - [`case`]: Case conversion (upper/lower)
//! - [`whitespace`]: Whitespace detection and trimming
//!
//! # Performance Notes
//!
//! - Feature detection is cached at startup via `OnceLock` (zero runtime overhead)
//! - All public functions use `#[inline]` for cross-crate inlining
//! - SIMD paths use `#[target_feature]` for safe intrinsic usage
//! - Alignment is handled automatically (unaligned loads used)
//!
//! # Safety
//!
//! All unsafe code is encapsulated within this module. Public functions are safe
//! and automatically dispatch to the best available implementation.

pub mod case;
pub mod equality;
pub mod search;
pub mod validation;
pub mod whitespace;

#[cfg(test)]
mod tests;

use std::sync::OnceLock;

// =============================================================================
// CPU Feature Detection
// =============================================================================

/// Cached CPU feature level for fast dispatch.
///
/// This is initialized once at first use and provides O(1) feature queries.
static CPU_LEVEL: OnceLock<SimdLevel> = OnceLock::new();

/// SIMD capability level for string operations.
///
/// Ordered by capability: higher levels support more operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum SimdLevel {
    /// Scalar only (no SIMD acceleration)
    Scalar = 0,
    /// SSE2: 16-byte operations (baseline x86-64)
    Sse2 = 1,
    /// SSE4.2: 16-byte + PCMPESTRI for string matching
    Sse42 = 2,
    /// AVX2: 32-byte operations
    Avx2 = 3,
    /// AVX-512: 64-byte operations
    Avx512 = 4,
}

impl SimdLevel {
    /// Detect the highest SIMD level supported by the current CPU.
    #[cfg(target_arch = "x86_64")]
    pub fn detect() -> Self {
        // Use std::arch for reliable feature detection
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            SimdLevel::Avx512
        } else if is_x86_feature_detected!("avx2") {
            SimdLevel::Avx2
        } else if is_x86_feature_detected!("sse4.2") {
            SimdLevel::Sse42
        } else if is_x86_feature_detected!("sse2") {
            SimdLevel::Sse2
        } else {
            SimdLevel::Scalar
        }
    }

    /// Fallback detection for non-x86 architectures.
    #[cfg(not(target_arch = "x86_64"))]
    pub fn detect() -> Self {
        SimdLevel::Scalar
    }

    /// Get a human-readable name for this level.
    pub const fn name(self) -> &'static str {
        match self {
            SimdLevel::Scalar => "Scalar",
            SimdLevel::Sse2 => "SSE2",
            SimdLevel::Sse42 => "SSE4.2",
            SimdLevel::Avx2 => "AVX2",
            SimdLevel::Avx512 => "AVX-512",
        }
    }
}

// =============================================================================
// Public Feature Query API
// =============================================================================

/// Get the cached SIMD level for the current CPU.
///
/// This is initialized on first call and provides O(1) subsequent access.
#[inline]
pub fn simd_level() -> SimdLevel {
    *CPU_LEVEL.get_or_init(SimdLevel::detect)
}

/// Check if AVX-512 is available.
#[inline]
pub fn has_avx512() -> bool {
    simd_level() >= SimdLevel::Avx512
}

/// Check if AVX2 is available.
#[inline]
pub fn has_avx2() -> bool {
    simd_level() >= SimdLevel::Avx2
}

/// Check if SSE4.2 is available.
#[inline]
pub fn has_sse42() -> bool {
    simd_level() >= SimdLevel::Sse42
}

/// Check if SSE2 is available (always true on x86-64).
#[inline]
pub fn has_sse2() -> bool {
    simd_level() >= SimdLevel::Sse2
}

// =============================================================================
// Re-exports for Convenience
// =============================================================================

// Re-export primary operations for easy access
pub use equality::{bytes_eq, str_eq};
pub use search::{bytes_contains, bytes_find, str_contains, str_find};
pub use validation::{is_ascii, is_ascii_lowercase, is_ascii_uppercase};
