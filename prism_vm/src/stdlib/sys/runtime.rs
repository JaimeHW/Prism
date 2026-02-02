//! Runtime version and platform information.
//!
//! Provides compile-time constants for Python version compatibility
//! and platform detection with zero runtime overhead.

use prism_core::Value;

// =============================================================================
// Version Constants
// =============================================================================

/// Python version we're compatible with: 3.12.0
pub const VERSION_MAJOR: u8 = 3;
pub const VERSION_MINOR: u8 = 12;
pub const VERSION_MICRO: u8 = 0;
pub const VERSION_RELEASELEVEL: &str = "final";
pub const VERSION_SERIAL: u8 = 0;

/// Version string matching Python's sys.version format.
pub const VERSION_STRING: &str = concat!("3.12.0 (Prism ", env!("CARGO_PKG_VERSION"), ")");

/// Hexadecimal version for easy comparison.
/// Format: 0xAABBCCDD where AA=major, BB=minor, CC=micro, DD=release
/// Release: 0xA0=alpha, 0xB0=beta, 0xC0=candidate, 0xF0=final
pub const HEXVERSION: u32 = 0x030C00F0;

/// API version for C extensions (not used but defined for compatibility).
pub const API_VERSION: u32 = 1013;

/// Copyright notice.
pub const COPYRIGHT: &str = "Copyright (c) Prism Python Runtime. All Rights Reserved.";

// =============================================================================
// Platform Detection
// =============================================================================

/// Supported platforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    Windows,
    Linux,
    MacOS,
    FreeBSD,
    Unknown,
}

impl Platform {
    /// Detect the current platform at compile time.
    #[inline]
    pub const fn detect() -> Self {
        #[cfg(target_os = "windows")]
        {
            Platform::Windows
        }
        #[cfg(target_os = "linux")]
        {
            Platform::Linux
        }
        #[cfg(target_os = "macos")]
        {
            Platform::MacOS
        }
        #[cfg(target_os = "freebsd")]
        {
            Platform::FreeBSD
        }
        #[cfg(not(any(
            target_os = "windows",
            target_os = "linux",
            target_os = "macos",
            target_os = "freebsd"
        )))]
        {
            Platform::Unknown
        }
    }

    /// Get the Python platform string.
    #[inline]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Platform::Windows => "win32",
            Platform::Linux => "linux",
            Platform::MacOS => "darwin",
            Platform::FreeBSD => "freebsd",
            Platform::Unknown => "unknown",
        }
    }

    /// Check if this is a POSIX platform.
    #[inline]
    pub const fn is_posix(&self) -> bool {
        matches!(self, Platform::Linux | Platform::MacOS | Platform::FreeBSD)
    }

    /// Check if this is Windows.
    #[inline]
    pub const fn is_windows(&self) -> bool {
        matches!(self, Platform::Windows)
    }
}

// =============================================================================
// Byte Order
// =============================================================================

/// Get the native byte order string.
#[inline]
pub const fn byte_order() -> &'static str {
    #[cfg(target_endian = "little")]
    {
        "little"
    }
    #[cfg(target_endian = "big")]
    {
        "big"
    }
}

// =============================================================================
// Version Info Tuple
// =============================================================================

/// Create the sys.version_info named tuple.
///
/// Returns (major, minor, micro, releaselevel, serial)
pub fn version_info_tuple() -> Value {
    // TODO: Return actual named tuple when tuple type is implemented
    // For now, return a list representation
    Value::none()
}

/// Create the sys.implementation info.
pub fn implementation_info() -> Value {
    // TODO: Return SimpleNamespace with name, version, etc.
    Value::none()
}

// =============================================================================
// Float Info
// =============================================================================

/// Create the sys.float_info struct sequence.
pub fn float_info_tuple() -> Value {
    // TODO: Return named tuple with float characteristics
    // max, max_exp, max_10_exp, min, min_exp, min_10_exp,
    // dig, mant_dig, epsilon, radix, rounds
    Value::none()
}

// =============================================================================
// Int Info
// =============================================================================

/// Create the sys.int_info struct sequence.
pub fn int_info_tuple() -> Value {
    // TODO: Return named tuple with int characteristics
    // bits_per_digit, sizeof_digit, default_max_str_digits, str_digits_check_threshold
    Value::none()
}

// =============================================================================
// Hash Info
// =============================================================================

/// Create the sys.hash_info struct sequence.
pub fn hash_info_tuple() -> Value {
    // TODO: Return named tuple with hash characteristics
    // width, modulus, inf, nan, imag, algorithm, hash_bits, seed_bits
    Value::none()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Version Constants Tests
    // =========================================================================

    #[test]
    fn test_version_major() {
        assert_eq!(VERSION_MAJOR, 3);
    }

    #[test]
    fn test_version_minor() {
        assert_eq!(VERSION_MINOR, 12);
    }

    #[test]
    fn test_version_micro() {
        assert_eq!(VERSION_MICRO, 0);
    }

    #[test]
    fn test_version_string_format() {
        assert!(VERSION_STRING.starts_with("3.12.0"));
        assert!(VERSION_STRING.contains("Prism"));
    }

    #[test]
    fn test_hexversion_format() {
        // Decode hexversion
        let major = (HEXVERSION >> 24) & 0xFF;
        let minor = (HEXVERSION >> 16) & 0xFF;
        let micro = (HEXVERSION >> 8) & 0xFF;
        let release = HEXVERSION & 0xFF;

        assert_eq!(major, 3);
        assert_eq!(minor, 12);
        assert_eq!(micro, 0);
        assert_eq!(release, 0xF0); // final
    }

    #[test]
    fn test_hexversion_comparison() {
        // 3.12.0 should be greater than 3.11.0
        let python_3_11 = 0x030B00F0u32;
        assert!(HEXVERSION > python_3_11);
    }

    // =========================================================================
    // Platform Detection Tests
    // =========================================================================

    #[test]
    fn test_platform_detect() {
        let platform = Platform::detect();
        // Should be one of the known platforms on any supported system
        assert!(matches!(
            platform,
            Platform::Windows | Platform::Linux | Platform::MacOS | Platform::FreeBSD
        ));
    }

    #[test]
    fn test_platform_as_str() {
        assert_eq!(Platform::Windows.as_str(), "win32");
        assert_eq!(Platform::Linux.as_str(), "linux");
        assert_eq!(Platform::MacOS.as_str(), "darwin");
        assert_eq!(Platform::FreeBSD.as_str(), "freebsd");
        assert_eq!(Platform::Unknown.as_str(), "unknown");
    }

    #[test]
    fn test_platform_is_posix() {
        assert!(!Platform::Windows.is_posix());
        assert!(Platform::Linux.is_posix());
        assert!(Platform::MacOS.is_posix());
        assert!(Platform::FreeBSD.is_posix());
    }

    #[test]
    fn test_platform_is_windows() {
        assert!(Platform::Windows.is_windows());
        assert!(!Platform::Linux.is_windows());
        assert!(!Platform::MacOS.is_windows());
    }

    #[test]
    fn test_current_platform_consistency() {
        let platform = Platform::detect();
        let platform_str = platform.as_str();

        #[cfg(target_os = "windows")]
        assert_eq!(platform_str, "win32");

        #[cfg(target_os = "linux")]
        assert_eq!(platform_str, "linux");

        #[cfg(target_os = "macos")]
        assert_eq!(platform_str, "darwin");
    }

    // =========================================================================
    // Byte Order Tests
    // =========================================================================

    #[test]
    fn test_byte_order() {
        let order = byte_order();
        assert!(order == "little" || order == "big");
    }

    #[test]
    fn test_byte_order_consistency() {
        // Verify byte order matches actual memory layout
        let value: u16 = 0x0102;
        let bytes = value.to_ne_bytes();

        if bytes[0] == 0x02 {
            assert_eq!(byte_order(), "little");
        } else {
            assert_eq!(byte_order(), "big");
        }
    }

    // =========================================================================
    // Copyright Tests
    // =========================================================================

    #[test]
    fn test_copyright_not_empty() {
        assert!(!COPYRIGHT.is_empty());
    }

    #[test]
    fn test_copyright_contains_name() {
        assert!(COPYRIGHT.contains("Prism"));
    }

    // =========================================================================
    // API Version Tests
    // =========================================================================

    #[test]
    fn test_api_version() {
        assert!(API_VERSION > 0);
    }
}
