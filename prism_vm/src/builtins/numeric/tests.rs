use super::*;

fn tagged_string_value_to_rust_string(value: Value) -> String {
    let ptr = value
        .as_string_object_ptr()
        .expect("expected interned string value");
    let interned =
        interned_by_ptr(ptr as *const u8).expect("interned string pointer should resolve");
    interned.as_str().to_string()
}

// =========================================================================
// FormatBuffer Tests
// =========================================================================

#[test]
fn test_format_buffer_new() {
    let buf = FormatBuffer::new();
    assert_eq!(buf.len(), 0);
    assert_eq!(buf.as_str(), "");
}

#[test]
fn test_format_buffer_push_single() {
    let mut buf = FormatBuffer::new();
    buf.push(b'X');
    assert_eq!(buf.len(), 1);
    assert_eq!(buf.as_str(), "X");
}

#[test]
fn test_format_buffer_push_multiple() {
    let mut buf = FormatBuffer::new();
    buf.push(b'C');
    buf.push(b'B');
    buf.push(b'A');
    assert_eq!(buf.as_str(), "ABC"); // Right-to-left, so reversed
}

#[test]
fn test_format_buffer_push_prefix() {
    let mut buf = FormatBuffer::new();
    buf.push(b'1');
    buf.push_prefix(b"0x");
    assert_eq!(buf.as_str(), "0x1");
}

// =========================================================================
// bin() Argument Validation Tests
// =========================================================================

#[test]
fn test_bin_no_args() {
    let result = builtin_bin(&[]);
    assert!(result.is_err());
    match result {
        Err(BuiltinError::TypeError(msg)) => {
            assert!(msg.contains("exactly one argument"));
        }
        _ => panic!("Expected TypeError"),
    }
}

#[test]
fn test_bin_too_many_args() {
    let result = builtin_bin(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    assert!(result.is_err());
}

#[test]
fn test_bin_float_error() {
    let result = builtin_bin(&[Value::float(3.14)]);
    assert!(result.is_err());
    match result {
        Err(BuiltinError::TypeError(msg)) => {
            assert!(msg.contains("float"));
        }
        _ => panic!("Expected TypeError"),
    }
}

#[test]
fn test_bin_returns_tagged_string_value() {
    let result = builtin_bin(&[Value::int(13).unwrap()]).unwrap();
    assert_eq!(tagged_string_value_to_rust_string(result), "0b1101");
}

#[test]
fn test_bin_none_error() {
    let result = builtin_bin(&[Value::none()]);
    assert!(result.is_err());
}

// =========================================================================
// bin() Formatting Tests
// =========================================================================

#[test]
fn test_bin_zero() {
    assert_eq!(format_binary_string(0), "0b0");
}

#[test]
fn test_bin_one() {
    assert_eq!(format_binary_string(1), "0b1");
}

#[test]
fn test_bin_two() {
    assert_eq!(format_binary_string(2), "0b10");
}

#[test]
fn test_bin_three() {
    assert_eq!(format_binary_string(3), "0b11");
}

#[test]
fn test_bin_five() {
    assert_eq!(format_binary_string(5), "0b101");
}

#[test]
fn test_bin_255() {
    assert_eq!(format_binary_string(255), "0b11111111");
}

#[test]
fn test_bin_256() {
    assert_eq!(format_binary_string(256), "0b100000000");
}

#[test]
fn test_bin_negative_one() {
    assert_eq!(format_binary_string(-1), "-0b1");
}

#[test]
fn test_bin_negative_five() {
    assert_eq!(format_binary_string(-5), "-0b101");
}

#[test]
fn test_bin_negative_127() {
    assert_eq!(format_binary_string(-127), "-0b1111111");
}

#[test]
fn test_bin_negative_128() {
    assert_eq!(format_binary_string(-128), "-0b10000000");
}

#[test]
fn test_bin_i64_max() {
    assert_eq!(
        format_binary_string(i64::MAX),
        "0b111111111111111111111111111111111111111111111111111111111111111"
    );
}

#[test]
fn test_bin_i64_min() {
    // i64::MIN = -9223372036854775808 = 1 followed by 63 zeros
    assert_eq!(
        format_binary_string(i64::MIN),
        "-0b1000000000000000000000000000000000000000000000000000000000000000"
    );
}

#[test]
fn test_bin_powers_of_two() {
    assert_eq!(format_binary_string(8), "0b1000");
    assert_eq!(format_binary_string(16), "0b10000");
    assert_eq!(format_binary_string(32), "0b100000");
    assert_eq!(format_binary_string(64), "0b1000000");
    assert_eq!(format_binary_string(128), "0b10000000");
    assert_eq!(format_binary_string(1024), "0b10000000000");
}

// =========================================================================
// hex() Argument Validation Tests
// =========================================================================

#[test]
fn test_hex_no_args() {
    let result = builtin_hex(&[]);
    assert!(result.is_err());
}

#[test]
fn test_hex_too_many_args() {
    let result = builtin_hex(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    assert!(result.is_err());
}

#[test]
fn test_hex_float_error() {
    let result = builtin_hex(&[Value::float(3.14)]);
    assert!(result.is_err());
}

#[test]
fn test_hex_returns_tagged_string_value() {
    let result = builtin_hex(&[Value::int(255).unwrap()]).unwrap();
    assert_eq!(tagged_string_value_to_rust_string(result), "0xff");
}

// =========================================================================
// hex() Formatting Tests
// =========================================================================

#[test]
fn test_hex_zero() {
    assert_eq!(format_hex_string(0), "0x0");
}

#[test]
fn test_hex_one() {
    assert_eq!(format_hex_string(1), "0x1");
}

#[test]
fn test_hex_fifteen() {
    assert_eq!(format_hex_string(15), "0xf");
}

#[test]
fn test_hex_sixteen() {
    assert_eq!(format_hex_string(16), "0x10");
}

#[test]
fn test_hex_255() {
    assert_eq!(format_hex_string(255), "0xff");
}

#[test]
fn test_hex_256() {
    assert_eq!(format_hex_string(256), "0x100");
}

#[test]
fn test_hex_0xdeadbeef() {
    assert_eq!(format_hex_string(0xDEADBEEF), "0xdeadbeef");
}

#[test]
fn test_hex_negative_one() {
    assert_eq!(format_hex_string(-1), "-0x1");
}

#[test]
fn test_hex_negative_255() {
    assert_eq!(format_hex_string(-255), "-0xff");
}

#[test]
fn test_hex_i64_max() {
    assert_eq!(format_hex_string(i64::MAX), "0x7fffffffffffffff");
}

#[test]
fn test_hex_i64_min() {
    assert_eq!(format_hex_string(i64::MIN), "-0x8000000000000000");
}

#[test]
fn test_hex_common_values() {
    assert_eq!(format_hex_string(10), "0xa");
    assert_eq!(format_hex_string(11), "0xb");
    assert_eq!(format_hex_string(12), "0xc");
    assert_eq!(format_hex_string(13), "0xd");
    assert_eq!(format_hex_string(14), "0xe");
}

// =========================================================================
// oct() Argument Validation Tests
// =========================================================================

#[test]
fn test_oct_no_args() {
    let result = builtin_oct(&[]);
    assert!(result.is_err());
}

#[test]
fn test_oct_too_many_args() {
    let result = builtin_oct(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    assert!(result.is_err());
}

#[test]
fn test_oct_float_error() {
    let result = builtin_oct(&[Value::float(3.14)]);
    assert!(result.is_err());
}

#[test]
fn test_oct_returns_tagged_string_value() {
    let result = builtin_oct(&[Value::int(9).unwrap()]).unwrap();
    assert_eq!(tagged_string_value_to_rust_string(result), "0o11");
}

// =========================================================================
// oct() Formatting Tests
// =========================================================================

#[test]
fn test_oct_zero() {
    assert_eq!(format_oct_string(0), "0o0");
}

#[test]
fn test_oct_one() {
    assert_eq!(format_oct_string(1), "0o1");
}

#[test]
fn test_oct_seven() {
    assert_eq!(format_oct_string(7), "0o7");
}

#[test]
fn test_oct_eight() {
    assert_eq!(format_oct_string(8), "0o10");
}

#[test]
fn test_oct_63() {
    assert_eq!(format_oct_string(63), "0o77");
}

#[test]
fn test_oct_64() {
    assert_eq!(format_oct_string(64), "0o100");
}

#[test]
fn test_oct_255() {
    assert_eq!(format_oct_string(255), "0o377");
}

#[test]
fn test_oct_negative_one() {
    assert_eq!(format_oct_string(-1), "-0o1");
}

#[test]
fn test_oct_negative_eight() {
    assert_eq!(format_oct_string(-8), "-0o10");
}

#[test]
fn test_oct_i64_max() {
    assert_eq!(format_oct_string(i64::MAX), "0o777777777777777777777");
}

#[test]
fn test_oct_i64_min() {
    assert_eq!(format_oct_string(i64::MIN), "-0o1000000000000000000000");
}

#[test]
fn test_oct_powers_of_eight() {
    assert_eq!(format_oct_string(8), "0o10");
    assert_eq!(format_oct_string(64), "0o100");
    assert_eq!(format_oct_string(512), "0o1000");
    assert_eq!(format_oct_string(4096), "0o10000");
}

// =========================================================================
// complex() Argument Validation Tests
// =========================================================================

#[test]
fn test_complex_too_many_args() {
    let result = builtin_complex(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);
    assert!(result.is_err());
}

#[test]
fn test_complex_none_error() {
    let result = builtin_complex(&[Value::none()]);
    assert!(result.is_err());
    match result {
        Err(BuiltinError::TypeError(msg)) => {
            assert!(msg.contains("NoneType"));
        }
        _ => panic!("Expected TypeError"),
    }
}

// =========================================================================
// Boolean Input Tests (__index__ protocol)
// =========================================================================

#[test]
fn test_bin_bool_true() {
    let result = builtin_bin(&[Value::bool(true)]).unwrap();
    assert_eq!(tagged_string_value_to_rust_string(result), "0b1");
}

#[test]
fn test_bin_bool_false() {
    let result = builtin_bin(&[Value::bool(false)]).unwrap();
    assert_eq!(tagged_string_value_to_rust_string(result), "0b0");
}

#[test]
fn test_hex_bool_true() {
    let result = builtin_hex(&[Value::bool(true)]).unwrap();
    assert_eq!(tagged_string_value_to_rust_string(result), "0x1");
}

#[test]
fn test_oct_bool_true() {
    let result = builtin_oct(&[Value::bool(true)]).unwrap();
    assert_eq!(tagged_string_value_to_rust_string(result), "0o1");
}

// =========================================================================
// Helper Function Tests
// =========================================================================

#[test]
fn test_extract_float_from_int() {
    let val = Value::int(42).unwrap();
    let result = extract_complex_parts(val, "test");
    assert_eq!(result.unwrap().real, 42.0);
}

#[test]
fn test_extract_float_from_float() {
    let val = Value::float(3.14);
    let result = extract_complex_parts(val, "test");
    assert!((result.unwrap().real - 3.14).abs() < 1e-10);
}

#[test]
fn test_extract_float_from_bool() {
    let val = Value::bool(true);
    let result = extract_complex_parts(val, "test");
    assert_eq!(result.unwrap().real, 1.0);

    let val = Value::bool(false);
    let result = extract_complex_parts(val, "test");
    assert_eq!(result.unwrap().real, 0.0);
}

#[test]
fn test_complex_bool_input_creates_complex_object() {
    let value = builtin_complex(&[Value::bool(true)]).expect("complex(True) should succeed");
    let ptr = value
        .as_object_ptr()
        .expect("complex should allocate an object");
    assert_eq!(crate::ops::objects::extract_type_id(ptr), TypeId::COMPLEX);

    let complex = unsafe { &*(ptr as *const ComplexObject) };
    assert_eq!(complex.real(), 1.0);
    assert_eq!(complex.imag(), 0.0);
}

#[test]
fn test_type_name_of() {
    assert_eq!(type_name_of(&Value::none()), "NoneType");
    assert_eq!(type_name_of(&Value::bool(true)), "bool");
    assert_eq!(type_name_of(&Value::int(1).unwrap()), "int");
    assert_eq!(type_name_of(&Value::float(1.0)), "float");
}

// =========================================================================
// Cross-Format Consistency Tests
// =========================================================================

#[test]
fn test_all_formats_for_zero() {
    assert_eq!(format_binary_string(0), "0b0");
    assert_eq!(format_hex_string(0), "0x0");
    assert_eq!(format_oct_string(0), "0o0");
}

#[test]
fn test_all_formats_for_255() {
    assert_eq!(format_binary_string(255), "0b11111111");
    assert_eq!(format_hex_string(255), "0xff");
    assert_eq!(format_oct_string(255), "0o377");
}

#[test]
fn test_all_formats_negative_255() {
    assert_eq!(format_binary_string(-255), "-0b11111111");
    assert_eq!(format_hex_string(-255), "-0xff");
    assert_eq!(format_oct_string(-255), "-0o377");
}

// =========================================================================
// Lookup Table Integrity Tests
// =========================================================================

#[test]
fn test_hex_lookup_table_integrity() {
    assert_eq!(HEX_CHARS_LOWER[0], b'0');
    assert_eq!(HEX_CHARS_LOWER[9], b'9');
    assert_eq!(HEX_CHARS_LOWER[10], b'a');
    assert_eq!(HEX_CHARS_LOWER[15], b'f');
}

#[test]
fn test_oct_lookup_table_integrity() {
    assert_eq!(OCT_CHARS[0], b'0');
    assert_eq!(OCT_CHARS[7], b'7');
}

// =========================================================================
// Edge Case and Boundary Tests
// =========================================================================

#[test]
fn test_bin_byte_boundary() {
    // Values at byte boundaries
    assert_eq!(format_binary_string(127), "0b1111111");
    assert_eq!(format_binary_string(128), "0b10000000");
}

#[test]
fn test_hex_nibble_boundaries() {
    for i in 0..=15 {
        let s = format_hex_string(i);
        assert!(s.starts_with("0x"));
        assert_eq!(s.len(), 3); // "0x" + 1 char
    }
}

#[test]
fn test_oct_3bit_boundaries() {
    for i in 0..=7 {
        let s = format_oct_string(i);
        assert!(s.starts_with("0o"));
        assert_eq!(s.len(), 3); // "0o" + 1 char
    }
}
