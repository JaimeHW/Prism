//! Runtime version and platform information.
//!
//! Provides compile-time constants for Python version compatibility
//! and platform detection with zero runtime overhead.

use num_bigint::BigInt;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::types::int::bigint_to_value;
use prism_runtime::types::tuple::TupleObject;
use std::sync::LazyLock;

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

/// Windows version string exposed as `sys.winver` on CPython-compatible builds.
pub const WINVER: &str = "3.12";

/// Platform library directory exposed via `sys.platlibdir`.
pub const PLATLIBDIR: &str = if cfg!(windows) { "DLLs" } else { "lib" };

/// Source-build VPATH metadata used by CPython's `sysconfig` on Windows.
pub const VPATH: &str = if cfg!(windows) { r"..\.." } else { "" };

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
    static VALUE: LazyLock<Value> = LazyLock::new(|| {
        leak_object_value(TupleObject::from_vec(vec![
            Value::int(VERSION_MAJOR as i64).expect("version major fits"),
            Value::int(VERSION_MINOR as i64).expect("version minor fits"),
            Value::int(VERSION_MICRO as i64).expect("version micro fits"),
            Value::string(intern(VERSION_RELEASELEVEL)),
            Value::int(VERSION_SERIAL as i64).expect("version serial fits"),
        ]))
    });
    *VALUE
}

/// Create the sys.implementation info.
pub fn implementation_info() -> Value {
    static VALUE: LazyLock<Value> = LazyLock::new(|| {
        let registry = shape_registry();
        let mut implementation = Box::new(ShapedObject::with_empty_shape(registry.empty_shape()));
        implementation.set_property(intern("name"), Value::string(intern("prism")), registry);
        implementation.set_property(
            intern("cache_tag"),
            Value::string(intern("prism-312")),
            registry,
        );
        implementation.set_property(intern("version"), version_info_tuple(), registry);
        implementation.set_property(
            intern("hexversion"),
            Value::int(HEXVERSION as i64).expect("hexversion fits"),
            registry,
        );
        implementation.set_property(
            intern("supports_isolated_interpreters"),
            Value::bool(false),
            registry,
        );
        Value::object_ptr(Box::into_raw(implementation) as *const ())
    });
    *VALUE
}

/// Create the `sys.getwindowsversion()` result object.
pub fn windows_version_info() -> Value {
    static VALUE: LazyLock<Value> = LazyLock::new(|| {
        let registry = shape_registry();
        let mut version = Box::new(ShapedObject::with_empty_shape(registry.empty_shape()));

        #[cfg(windows)]
        {
            use std::mem::MaybeUninit;
            use windows_sys::Win32::System::SystemInformation::{
                GetVersionExW, OSVERSIONINFOEXW, OSVERSIONINFOW,
            };

            unsafe {
                let mut info = MaybeUninit::<OSVERSIONINFOEXW>::zeroed();
                (*info.as_mut_ptr()).dwOSVersionInfoSize =
                    std::mem::size_of::<OSVERSIONINFOEXW>() as u32;
                let info = if GetVersionExW(info.as_mut_ptr().cast::<OSVERSIONINFOW>()) != 0 {
                    info.assume_init()
                } else {
                    OSVERSIONINFOEXW {
                        dwOSVersionInfoSize: 0,
                        dwMajorVersion: 10,
                        dwMinorVersion: 0,
                        dwBuildNumber: 0,
                        dwPlatformId: 2,
                        szCSDVersion: [0; 128],
                        wServicePackMajor: 0,
                        wServicePackMinor: 0,
                        wSuiteMask: 0,
                        wProductType: 1,
                        wReserved: 0,
                    }
                };

                let service_pack = String::from_utf16_lossy(&info.szCSDVersion)
                    .trim_end_matches('\0')
                    .to_string();
                for (name, value) in [
                    (
                        "major",
                        Value::int(info.dwMajorVersion as i64).expect("major fits"),
                    ),
                    (
                        "minor",
                        Value::int(info.dwMinorVersion as i64).expect("minor fits"),
                    ),
                    (
                        "build",
                        Value::int(info.dwBuildNumber as i64).expect("build fits"),
                    ),
                    (
                        "platform",
                        Value::int(info.dwPlatformId as i64).expect("platform fits"),
                    ),
                    ("service_pack", Value::string(intern(&service_pack))),
                    (
                        "service_pack_major",
                        Value::int(info.wServicePackMajor as i64).expect("service pack major fits"),
                    ),
                    (
                        "service_pack_minor",
                        Value::int(info.wServicePackMinor as i64).expect("service pack minor fits"),
                    ),
                    (
                        "suite_mask",
                        Value::int(info.wSuiteMask as i64).expect("suite mask fits"),
                    ),
                    (
                        "product_type",
                        Value::int(info.wProductType as i64).expect("product type fits"),
                    ),
                ] {
                    version.set_property(intern(name), value, registry);
                }
            }
        }

        #[cfg(not(windows))]
        {
            for (name, value) in [
                ("major", Value::int(0).expect("major fits")),
                ("minor", Value::int(0).expect("minor fits")),
                ("build", Value::int(0).expect("build fits")),
                ("platform", Value::int(0).expect("platform fits")),
                ("service_pack", Value::string(intern(""))),
                (
                    "service_pack_major",
                    Value::int(0).expect("service pack major fits"),
                ),
                (
                    "service_pack_minor",
                    Value::int(0).expect("service pack minor fits"),
                ),
                ("suite_mask", Value::int(0).expect("suite mask fits")),
                ("product_type", Value::int(0).expect("product type fits")),
            ] {
                version.set_property(intern(name), value, registry);
            }
        }

        Value::object_ptr(Box::into_raw(version) as *const ())
    });
    *VALUE
}

fn info_record(fields: &[(&'static str, Value)]) -> Value {
    let registry = shape_registry();
    let mut record = Box::new(ShapedObject::with_empty_shape(registry.empty_shape()));
    for (name, value) in fields {
        record.set_property(intern(name), *value, registry);
    }
    Value::object_ptr(Box::into_raw(record) as *const ())
}

// =============================================================================
// Float Info
// =============================================================================

/// Create the sys.float_info struct sequence.
pub fn float_info_tuple() -> Value {
    static VALUE: LazyLock<Value> = LazyLock::new(|| {
        info_record(&[
            ("max", Value::float(f64::MAX)),
            (
                "max_exp",
                Value::int(f64::MAX_EXP as i64).expect("f64::MAX_EXP should fit"),
            ),
            (
                "max_10_exp",
                Value::int(f64::MAX_10_EXP as i64).expect("f64::MAX_10_EXP should fit"),
            ),
            ("min", Value::float(f64::MIN_POSITIVE)),
            (
                "min_exp",
                Value::int(f64::MIN_EXP as i64).expect("f64::MIN_EXP should fit"),
            ),
            (
                "min_10_exp",
                Value::int(f64::MIN_10_EXP as i64).expect("f64::MIN_10_EXP should fit"),
            ),
            (
                "dig",
                Value::int(f64::DIGITS as i64).expect("f64::DIGITS should fit"),
            ),
            (
                "mant_dig",
                Value::int(f64::MANTISSA_DIGITS as i64).expect("f64::MANTISSA_DIGITS should fit"),
            ),
            ("epsilon", Value::float(f64::EPSILON)),
            (
                "radix",
                Value::int(f64::RADIX as i64).expect("f64::RADIX should fit"),
            ),
            ("rounds", Value::int(1).expect("rounds should fit")),
        ])
    });
    *VALUE
}

// =============================================================================
// Int Info
// =============================================================================

/// Create the sys.int_info struct sequence.
pub fn int_info_tuple() -> Value {
    static VALUE: LazyLock<Value> = LazyLock::new(|| {
        let (bits_per_digit, sizeof_digit) = if cfg!(target_pointer_width = "64") {
            (30, 4)
        } else {
            (15, 2)
        };

        info_record(&[
            (
                "bits_per_digit",
                Value::int(bits_per_digit).expect("bits_per_digit should fit"),
            ),
            (
                "sizeof_digit",
                Value::int(sizeof_digit).expect("sizeof_digit should fit"),
            ),
            (
                "default_max_str_digits",
                Value::int(4300).expect("default_max_str_digits should fit"),
            ),
            (
                "str_digits_check_threshold",
                Value::int(640).expect("str_digits_check_threshold should fit"),
            ),
        ])
    });
    *VALUE
}

// =============================================================================
// Hash Info
// =============================================================================

/// Create the sys.hash_info struct sequence.
pub fn hash_info_tuple() -> Value {
    static VALUE: LazyLock<Value> = LazyLock::new(|| {
        let width = i64::from(usize::BITS);
        let modulus = if width >= 64 {
            2_305_843_009_213_693_951_i64
        } else {
            2_147_483_647_i64
        };

        info_record(&[
            ("width", Value::int(width).expect("hash width should fit")),
            ("modulus", bigint_to_value(BigInt::from(modulus))),
            ("inf", Value::int(314_159).expect("hash inf should fit")),
            ("nan", Value::int(0).expect("hash nan should fit")),
            ("imag", Value::int(1_000_003).expect("hash imag should fit")),
            ("algorithm", Value::string(intern("siphash13"))),
            (
                "hash_bits",
                Value::int(width).expect("hash_bits should fit"),
            ),
            ("seed_bits", Value::int(128).expect("seed_bits should fit")),
            ("cutoff", Value::int(0).expect("cutoff should fit")),
        ])
    });
    *VALUE
}

#[inline]
fn leak_object_value<T>(object: T) -> Value {
    Value::object_ptr(Box::into_raw(Box::new(object)) as *const ())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_runtime::object::shaped_object::ShapedObject;
    use prism_runtime::types::tuple::TupleObject;

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
    fn test_winver_format() {
        assert_eq!(WINVER, "3.12");
    }

    #[test]
    fn test_platlibdir_matches_platform_contract() {
        if cfg!(windows) {
            assert_eq!(PLATLIBDIR, "DLLs");
            assert_eq!(VPATH, r"..\..");
        } else {
            assert_eq!(PLATLIBDIR, "lib");
            assert_eq!(VPATH, "");
        }
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

    #[test]
    fn test_version_info_tuple_is_real_sequence() {
        let value = version_info_tuple();
        let ptr = value
            .as_object_ptr()
            .expect("version_info should be a heap tuple");
        let tuple = unsafe { &*(ptr as *const TupleObject) };
        assert_eq!(tuple.len(), 5);
        assert_eq!(tuple.get(0).and_then(|value| value.as_int()), Some(3));
        assert_eq!(tuple.get(1).and_then(|value| value.as_int()), Some(12));
    }

    #[test]
    fn test_implementation_info_exposes_namespace_fields() {
        let value = implementation_info();
        let ptr = value
            .as_object_ptr()
            .expect("implementation should be a heap object");
        let object = unsafe { &*(ptr as *const ShapedObject) };
        assert_eq!(
            object.get_property("name"),
            Some(Value::string(intern("prism")))
        );
        assert!(object.get_property("version").is_some());
        assert!(object.get_property("cache_tag").is_some());
    }

    #[test]
    fn test_windows_version_info_exposes_platform_field() {
        let value = windows_version_info();
        let ptr = value
            .as_object_ptr()
            .expect("windows version info should be a heap object");
        let object = unsafe { &*(ptr as *const ShapedObject) };
        if cfg!(windows) {
            assert_eq!(
                object
                    .get_property("platform")
                    .and_then(|value| value.as_int()),
                Some(2)
            );
        } else {
            assert_eq!(
                object
                    .get_property("platform")
                    .and_then(|value| value.as_int()),
                Some(0)
            );
        }
    }

    #[test]
    fn test_float_info_exposes_ieee_fields() {
        let value = float_info_tuple();
        let ptr = value
            .as_object_ptr()
            .expect("float_info should be a heap object");
        let object = unsafe { &*(ptr as *const ShapedObject) };

        assert_eq!(
            object
                .get_property("mant_dig")
                .and_then(|value| value.as_int()),
            Some(53)
        );
        assert_eq!(
            object
                .get_property("radix")
                .and_then(|value| value.as_int()),
            Some(2)
        );
        assert!(object.get_property("epsilon").is_some());
    }

    #[test]
    fn test_int_info_exposes_digit_configuration() {
        let value = int_info_tuple();
        let ptr = value
            .as_object_ptr()
            .expect("int_info should be a heap object");
        let object = unsafe { &*(ptr as *const ShapedObject) };
        let expected_bits = if cfg!(target_pointer_width = "64") {
            Some(30)
        } else {
            Some(15)
        };

        assert_eq!(
            object
                .get_property("bits_per_digit")
                .and_then(|value| value.as_int()),
            expected_bits
        );
        assert_eq!(
            object
                .get_property("default_max_str_digits")
                .and_then(|value| value.as_int()),
            Some(4300)
        );
    }

    #[test]
    fn test_hash_info_exposes_width_and_algorithm() {
        let value = hash_info_tuple();
        let ptr = value
            .as_object_ptr()
            .expect("hash_info should be a heap object");
        let object = unsafe { &*(ptr as *const ShapedObject) };
        let algorithm = object
            .get_property("algorithm")
            .and_then(|value| value.as_string_object_ptr())
            .and_then(|ptr| prism_core::intern::interned_by_ptr(ptr as *const u8))
            .map(|text| text.as_str().to_string());

        assert_eq!(
            object
                .get_property("width")
                .and_then(|value| value.as_int()),
            Some(i64::from(usize::BITS))
        );
        assert_eq!(
            object
                .get_property("hash_bits")
                .and_then(|value| value.as_int()),
            Some(i64::from(usize::BITS))
        );
        assert_eq!(algorithm.as_deref(), Some("siphash13"));
    }
}
