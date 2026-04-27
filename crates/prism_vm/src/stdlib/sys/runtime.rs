//! Runtime version and platform information.
//!
//! Provides compile-time constants for Python version compatibility
//! and platform detection with zero runtime overhead.

use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::python_numeric::int_like_value;
use num_bigint::BigInt;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::allocation_context::alloc_static_value;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::int::bigint_to_value;
use prism_runtime::types::slice::SliceObject;
use prism_runtime::types::tuple::TupleObject;
use std::sync::{Arc, LazyLock};

// =============================================================================
// Version Constants
// =============================================================================

/// Python version we're compatible with: 3.12.0
pub const VERSION_MAJOR: u8 = 3;
pub const VERSION_MINOR: u8 = 12;
pub const VERSION_MICRO: u8 = 0;
pub const VERSION_RELEASELEVEL: &str = "final";
pub const VERSION_SERIAL: u8 = 0;

/// Version string matching CPython's parseable `sys.version` format.
pub const VERSION_STRING: &str = concat!(
    "3.12.0 (#0, Apr 24 2026, 00:00:00) [Prism ",
    env!("CARGO_PKG_VERSION"),
    "]"
);

/// Windows version string exposed as `sys.winver` on CPython-compatible builds.
pub const WINVER: &str = "3.12";

/// Platform library directory exposed via `sys.platlibdir`.
pub const PLATLIBDIR: &str = if cfg!(windows) { "DLLs" } else { "lib" };

/// Source-build VPATH metadata used by CPython's `sysconfig` on Windows.
pub const VPATH: &str = if cfg!(windows) { r"..\.." } else { "" };

const WINDOWS_VERSION_SEQUENCE_FIELDS: usize = 5;
const WINDOWS_VERSION_FIELD_NAMES: [&str; 10] = [
    "major",
    "minor",
    "build",
    "platform",
    "service_pack",
    "service_pack_major",
    "service_pack_minor",
    "suite_mask",
    "product_type",
    "platform_version",
];

static WINDOWS_VERSION_LEN_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("sys.getwindowsversion.__len__"),
        windows_version_len,
    )
});
static WINDOWS_VERSION_GETITEM_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("sys.getwindowsversion.__getitem__"),
        windows_version_getitem,
    )
});
static WINDOWS_VERSION_COUNT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("sys.getwindowsversion.count"),
        windows_version_count,
    )
});
static WINDOWS_VERSION_INDEX_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("sys.getwindowsversion.index"),
        windows_version_index,
    )
});

#[cfg(windows)]
#[link(name = "ntdll")]
unsafe extern "system" {
    fn RtlGetVersion(
        lp_version_information: *mut windows_sys::Win32::System::SystemInformation::OSVERSIONINFOW,
    ) -> i32;
}

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
        static_object_value(TupleObject::from_vec(vec![
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
        let fields = windows_version_fields();
        let indexed_fields = fields[..WINDOWS_VERSION_SEQUENCE_FIELDS]
            .iter()
            .map(|(_, value)| *value)
            .collect();
        let mut version = Box::new(ShapedObject::new_tuple_backed(
            TypeId::OBJECT,
            registry.empty_shape(),
            TupleObject::from_vec(indexed_fields),
        ));
        let receiver = Value::object_ptr(version.as_mut() as *mut ShapedObject as *const ());

        for (name, value) in fields {
            version.set_property(intern(name), value, registry);
        }
        for (name, method) in [
            ("__len__", &*WINDOWS_VERSION_LEN_FUNCTION),
            ("__getitem__", &*WINDOWS_VERSION_GETITEM_FUNCTION),
            ("count", &*WINDOWS_VERSION_COUNT_FUNCTION),
            ("index", &*WINDOWS_VERSION_INDEX_FUNCTION),
        ] {
            version.set_property(
                intern(name),
                static_bound_builtin_attr_value(method, receiver),
                registry,
            );
        }

        Value::object_ptr(Box::into_raw(version) as *const ())
    });
    *VALUE
}

struct WindowsVersionData {
    major: u32,
    minor: u32,
    build: u32,
    platform: u32,
    service_pack: String,
    service_pack_major: u16,
    service_pack_minor: u16,
    suite_mask: u16,
    product_type: u8,
    platform_version: [u32; 3],
}

fn windows_version_fields() -> [(&'static str, Value); 10] {
    let data = windows_version_data();
    let platform_version = static_tuple_value(&[
        int_value(data.platform_version[0] as i64),
        int_value(data.platform_version[1] as i64),
        int_value(data.platform_version[2] as i64),
    ]);

    [
        (WINDOWS_VERSION_FIELD_NAMES[0], int_value(data.major as i64)),
        (WINDOWS_VERSION_FIELD_NAMES[1], int_value(data.minor as i64)),
        (WINDOWS_VERSION_FIELD_NAMES[2], int_value(data.build as i64)),
        (
            WINDOWS_VERSION_FIELD_NAMES[3],
            int_value(data.platform as i64),
        ),
        (
            WINDOWS_VERSION_FIELD_NAMES[4],
            Value::string(intern(&data.service_pack)),
        ),
        (
            WINDOWS_VERSION_FIELD_NAMES[5],
            int_value(data.service_pack_major as i64),
        ),
        (
            WINDOWS_VERSION_FIELD_NAMES[6],
            int_value(data.service_pack_minor as i64),
        ),
        (
            WINDOWS_VERSION_FIELD_NAMES[7],
            int_value(data.suite_mask as i64),
        ),
        (
            WINDOWS_VERSION_FIELD_NAMES[8],
            int_value(data.product_type as i64),
        ),
        (WINDOWS_VERSION_FIELD_NAMES[9], platform_version),
    ]
}

#[cfg(windows)]
fn windows_version_data() -> WindowsVersionData {
    use std::mem::MaybeUninit;
    use windows_sys::Win32::System::SystemInformation::{
        GetVersionExW, OSVERSIONINFOEXW, OSVERSIONINFOW,
    };

    let info = unsafe {
        let mut info = MaybeUninit::<OSVERSIONINFOEXW>::zeroed();
        (*info.as_mut_ptr()).dwOSVersionInfoSize = std::mem::size_of::<OSVERSIONINFOEXW>() as u32;
        let info_ptr = info.as_mut_ptr().cast::<OSVERSIONINFOW>();
        if RtlGetVersion(info_ptr) >= 0 || GetVersionExW(info_ptr) != 0 {
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
        }
    };

    let service_pack = String::from_utf16_lossy(&info.szCSDVersion)
        .trim_end_matches('\0')
        .to_string();
    let fallback = [info.dwMajorVersion, info.dwMinorVersion, info.dwBuildNumber];

    WindowsVersionData {
        major: info.dwMajorVersion,
        minor: info.dwMinorVersion,
        build: info.dwBuildNumber,
        platform: info.dwPlatformId,
        service_pack,
        service_pack_major: info.wServicePackMajor,
        service_pack_minor: info.wServicePackMinor,
        suite_mask: info.wSuiteMask,
        product_type: info.wProductType,
        platform_version: kernel32_platform_version(fallback),
    }
}

#[cfg(not(windows))]
fn windows_version_data() -> WindowsVersionData {
    WindowsVersionData {
        major: 0,
        minor: 0,
        build: 0,
        platform: 0,
        service_pack: String::new(),
        service_pack_major: 0,
        service_pack_minor: 0,
        suite_mask: 0,
        product_type: 0,
        platform_version: [0, 0, 0],
    }
}

#[cfg(windows)]
fn kernel32_platform_version(fallback: [u32; 3]) -> [u32; 3] {
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStrExt;
    use std::path::PathBuf;
    use windows_sys::Win32::Storage::FileSystem::{
        GetFileVersionInfoSizeW, GetFileVersionInfoW, VS_FIXEDFILEINFO, VerQueryValueW,
    };

    let mut path = PathBuf::from(
        std::env::var_os("SystemRoot").unwrap_or_else(|| OsString::from(r"C:\Windows")),
    );
    path.push("System32");
    path.push("kernel32.dll");
    let wide_path: Vec<u16> = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();

    unsafe {
        let size = GetFileVersionInfoSizeW(wide_path.as_ptr(), std::ptr::null_mut());
        if size == 0 {
            return fallback;
        }

        let mut block = vec![0_u8; size as usize];
        if GetFileVersionInfoW(wide_path.as_ptr(), 0, size, block.as_mut_ptr().cast()) == 0 {
            return fallback;
        }

        let root = [0_u16];
        let mut fixed_info: *mut core::ffi::c_void = std::ptr::null_mut();
        let mut fixed_info_len = 0_u32;
        if VerQueryValueW(
            block.as_ptr().cast(),
            root.as_ptr(),
            &mut fixed_info,
            &mut fixed_info_len,
        ) == 0
            || fixed_info.is_null()
            || fixed_info_len < std::mem::size_of::<VS_FIXEDFILEINFO>() as u32
        {
            return fallback;
        }

        let fixed_info = &*(fixed_info as *const VS_FIXEDFILEINFO);
        if fixed_info.dwSignature != 0xFEEF04BD {
            return fallback;
        }

        [
            hiword(fixed_info.dwProductVersionMS),
            loword(fixed_info.dwProductVersionMS),
            hiword(fixed_info.dwProductVersionLS),
        ]
    }
}

#[cfg(windows)]
#[inline]
fn hiword(value: u32) -> u32 {
    (value >> 16) & 0xffff
}

#[cfg(windows)]
#[inline]
fn loword(value: u32) -> u32 {
    value & 0xffff
}

#[inline]
fn int_value(value: i64) -> Value {
    Value::int(value).expect("version field should fit in tagged int")
}

#[inline]
fn static_tuple_value(values: &[Value]) -> Value {
    static_object_value(TupleObject::from_slice(values))
}

fn static_bound_builtin_attr_value(
    function: &'static BuiltinFunctionObject,
    receiver: Value,
) -> Value {
    alloc_static_value(function.bind(receiver))
}

#[inline]
fn static_object_value<T: prism_runtime::Trace + 'static>(object: T) -> Value {
    alloc_static_value(object)
}

fn windows_version_tuple(value: Value) -> Result<&'static TupleObject, BuiltinError> {
    prism_runtime::types::tuple::value_as_tuple_ref(value).ok_or_else(|| {
        BuiltinError::TypeError("descriptor requires a sys.getwindowsversion object".to_string())
    })
}

fn windows_version_len(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "sys.getwindowsversion.__len__() takes no arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let tuple = windows_version_tuple(args[0])?;
    Ok(int_value(tuple.len() as i64))
}

fn windows_version_getitem(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "sys.getwindowsversion.__getitem__() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let tuple = windows_version_tuple(args[0])?;
    if let Some(index) = int_like_value(args[1]) {
        return tuple.get(index).ok_or_else(|| {
            BuiltinError::IndexError(format!(
                "index {index} out of range for length {}",
                tuple.len()
            ))
        });
    }

    if let Some(slice) = slice_from_value(args[1]) {
        return Ok(leak_object_value(tuple_slice(tuple, slice)));
    }

    Err(BuiltinError::TypeError(format!(
        "tuple indices must be integers or slices, not {}",
        args[1].type_name()
    )))
}

fn windows_version_count(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "sys.getwindowsversion.count() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let tuple = windows_version_tuple(args[0])?;
    let count = tuple
        .iter()
        .copied()
        .filter(|item| crate::ops::comparison::values_equal(*item, args[1]))
        .count();
    Ok(int_value(count as i64))
}

fn windows_version_index(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 4 {
        return Err(BuiltinError::TypeError(format!(
            "sys.getwindowsversion.index() takes from 1 to 3 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let tuple = windows_version_tuple(args[0])?;
    let start = tuple_index_bound(args.get(2).copied(), 0, tuple.len(), "start")?;
    let stop = tuple_index_bound(
        args.get(3).copied(),
        tuple.len() as i64,
        tuple.len(),
        "stop",
    )?;

    for index in start..stop.max(start) {
        let item = tuple
            .get(index as i64)
            .expect("normalized tuple index should be in bounds");
        if crate::ops::comparison::values_equal(item, args[1]) {
            return Ok(int_value(index as i64));
        }
    }

    Err(BuiltinError::ValueError(
        "tuple.index(x): x not in tuple".to_string(),
    ))
}

fn tuple_index_bound(
    value: Option<Value>,
    default: i64,
    len: usize,
    name: &'static str,
) -> Result<usize, BuiltinError> {
    let raw = match value {
        Some(value) => int_like_value(value).ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "slice indices must be integers or have an __index__ method ({name})"
            ))
        })?,
        None => default,
    };

    let len_i64 = len as i64;
    let normalized = if raw < 0 { len_i64 + raw } else { raw };
    Ok(normalized.clamp(0, len_i64) as usize)
}

fn slice_from_value(value: Value) -> Option<&'static SliceObject> {
    let ptr = value.as_object_ptr()?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    (header.type_id == TypeId::SLICE).then(|| unsafe { &*(ptr as *const SliceObject) })
}

fn tuple_slice(tuple: &TupleObject, slice: &SliceObject) -> TupleObject {
    let indices = slice.indices(tuple.len());
    let mut items = Vec::with_capacity(indices.length);
    for index in indices.iter() {
        if index < tuple.len() {
            items.push(
                tuple
                    .get(index as i64)
                    .expect("slice index should be in bounds"),
            );
        }
    }
    TupleObject::from_vec(items)
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
fn leak_object_value<T: prism_runtime::Trace + 'static>(object: T) -> Value {
    crate::alloc_managed_value(object)
}
