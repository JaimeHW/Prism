use super::*;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::types::tuple::value_as_tuple_ref;

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
    assert!(VERSION_STRING.contains(") ["));
    assert!(VERSION_STRING.contains(", "));
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
    let tuple = value_as_tuple_ref(value).expect("version_info should expose tuple storage");
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
    let tuple = object
        .tuple_backing()
        .expect("windows version info should expose sequence storage");
    assert_eq!(tuple.len(), WINDOWS_VERSION_SEQUENCE_FIELDS);

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
fn test_windows_version_info_exposes_named_fields_beyond_sequence_prefix() {
    let value = windows_version_info();
    let ptr = value
        .as_object_ptr()
        .expect("windows version info should be a heap object");
    let object = unsafe { &*(ptr as *const ShapedObject) };

    for name in WINDOWS_VERSION_FIELD_NAMES {
        assert!(
            object.get_property(name).is_some(),
            "{name} should be exposed as an attribute"
        );
    }

    let platform_version = object
        .get_property("platform_version")
        .expect("platform_version should be exposed");
    let platform_tuple =
        value_as_tuple_ref(platform_version).expect("platform_version should be a tuple");
    assert_eq!(platform_tuple.len(), 3);

    if cfg!(windows) {
        assert_eq!(
            platform_tuple.as_slice()[0].as_int(),
            object
                .get_property("major")
                .and_then(|value| value.as_int())
        );
        assert_eq!(
            platform_tuple.as_slice()[1].as_int(),
            object
                .get_property("minor")
                .and_then(|value| value.as_int())
        );
    }
}

#[test]
fn test_windows_version_info_bound_sequence_methods_use_five_item_prefix() {
    let value = windows_version_info();
    let ptr = value
        .as_object_ptr()
        .expect("windows version info should be a heap object");
    let object = unsafe { &*(ptr as *const ShapedObject) };
    let len_method = object
        .get_property("__len__")
        .expect("__len__ should be installed");
    let len_method = unsafe {
        &*(len_method
            .as_object_ptr()
            .expect("__len__ should be a builtin function")
            as *const BuiltinFunctionObject)
    };
    assert_eq!(
        len_method
            .call(&[])
            .expect("__len__ should succeed")
            .as_int(),
        Some(WINDOWS_VERSION_SEQUENCE_FIELDS as i64)
    );

    let getitem_method = object
        .get_property("__getitem__")
        .expect("__getitem__ should be installed");
    let getitem_method = unsafe {
        &*(getitem_method
            .as_object_ptr()
            .expect("__getitem__ should be a builtin function")
            as *const BuiltinFunctionObject)
    };
    assert_eq!(
        getitem_method
            .call(&[Value::int(0).expect("index should fit")])
            .expect("__getitem__ should succeed"),
        object.get_property("major").expect("major should exist")
    );

    let slice = SliceObject::start_stop(0, 3);
    let slice_value = leak_object_value(slice);
    let sliced = getitem_method
        .call(&[slice_value])
        .expect("__getitem__ should slice");
    let tuple = value_as_tuple_ref(sliced).expect("slice result should be a tuple");
    assert_eq!(tuple.len(), 3);
    assert_eq!(tuple.get(0), object.get_property("major"));
    assert_eq!(tuple.get(1), object.get_property("minor"));
    assert_eq!(tuple.get(2), object.get_property("build"));
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
