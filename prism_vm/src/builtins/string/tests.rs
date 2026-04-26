use super::*;
use crate::error::RuntimeErrorKind;
use crate::stdlib::exceptions::ExceptionTypeId;
use prism_core::intern::{intern, interned_by_ptr};
use prism_core::python_unicode::encode_python_code_point;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::string::StringObject;

fn value_to_rust_string(value: Value) -> String {
    if value.is_string() {
        let ptr = value
            .as_string_object_ptr()
            .expect("tagged string should have pointer");
        let interned = interned_by_ptr(ptr as *const u8).expect("interned pointer should resolve");
        return interned.as_str().to_string();
    }

    let ptr = value
        .as_object_ptr()
        .expect("string value should be object-backed");
    assert_eq!(crate::ops::objects::extract_type_id(ptr), TypeId::STR);
    let string_obj = unsafe { &*(ptr as *const StringObject) };
    string_obj.as_str().to_string()
}

fn boxed_value<T>(obj: T) -> (Value, *mut T) {
    let ptr = Box::into_raw(Box::new(obj));
    (Value::object_ptr(ptr as *const ()), ptr)
}

unsafe fn drop_boxed<T>(ptr: *mut T) {
    drop(unsafe { Box::from_raw(ptr) });
}

fn value_to_byte_vec(value: Value) -> Vec<u8> {
    let ptr = value
        .as_object_ptr()
        .expect("byte sequence should be object-backed");
    let type_id = crate::ops::objects::extract_type_id(ptr);
    assert!(
        type_id == TypeId::BYTES || type_id == TypeId::BYTEARRAY,
        "unexpected type id for byte sequence: {:?}",
        type_id
    );
    let bytes_obj = unsafe { &*(ptr as *const BytesObject) };
    bytes_obj.as_bytes().to_vec()
}

fn byte_sequence_type(value: Value) -> TypeId {
    let ptr = value
        .as_object_ptr()
        .expect("byte sequence should be object-backed");
    crate::ops::objects::extract_type_id(ptr)
}

fn assert_unicode_encode_error(err: BuiltinError, expected_message: &str) {
    match err {
        BuiltinError::Raised(runtime_err) => match runtime_err.kind() {
            RuntimeErrorKind::Exception { type_id, message } => {
                assert_eq!(*type_id, ExceptionTypeId::UnicodeEncodeError.as_u8() as u16);
                assert_eq!(message.as_ref(), expected_message);
            }
            kind => panic!("expected UnicodeEncodeError, got {kind:?}"),
        },
        other => panic!("expected UnicodeEncodeError, got {other:?}"),
    }
}

fn assert_unicode_decode_error(err: BuiltinError, expected_message: &str) {
    match err {
        BuiltinError::Raised(runtime_err) => match runtime_err.kind() {
            RuntimeErrorKind::Exception { type_id, message } => {
                assert_eq!(*type_id, ExceptionTypeId::UnicodeDecodeError.as_u8() as u16);
                assert_eq!(message.as_ref(), expected_message);
            }
            kind => panic!("expected UnicodeDecodeError, got {kind:?}"),
        },
        other => panic!("expected UnicodeDecodeError, got {other:?}"),
    }
}

// =========================================================================
// ord() Argument Validation Tests
// =========================================================================

#[test]
fn test_ord_no_args() {
    let result = builtin_ord(&[]);
    assert!(result.is_err());
    match result {
        Err(BuiltinError::TypeError(msg)) => {
            assert!(msg.contains("exactly one argument"));
        }
        _ => panic!("Expected TypeError"),
    }
}

#[test]
fn test_ord_too_many_args() {
    let result = builtin_ord(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    assert!(result.is_err());
}

#[test]
fn test_ord_tagged_string() {
    let result = builtin_ord(&[Value::string(intern("A"))]).unwrap();
    assert_eq!(result.as_int(), Some(65));
}

#[test]
fn test_ord_heap_string() {
    let (string_value, string_ptr) = boxed_value(StringObject::new("€"));
    let result = builtin_ord(&[string_value]).unwrap();
    assert_eq!(result.as_int(), Some(8364));
    unsafe { drop_boxed(string_ptr) };
}

// =========================================================================
// ord_from_str() Tests
// =========================================================================

#[test]
fn test_ord_from_str_ascii() {
    assert_eq!(ord_from_str("a").unwrap(), 97);
    assert_eq!(ord_from_str("A").unwrap(), 65);
    assert_eq!(ord_from_str("0").unwrap(), 48);
    assert_eq!(ord_from_str(" ").unwrap(), 32);
    assert_eq!(ord_from_str("~").unwrap(), 126);
}

#[test]
fn test_ord_from_str_control_chars() {
    assert_eq!(ord_from_str("\0").unwrap(), 0);
    assert_eq!(ord_from_str("\t").unwrap(), 9);
    assert_eq!(ord_from_str("\n").unwrap(), 10);
    assert_eq!(ord_from_str("\r").unwrap(), 13);
}

#[test]
fn test_ord_from_str_unicode_bmp() {
    // Basic Multilingual Plane
    assert_eq!(ord_from_str("€").unwrap(), 8364); // Euro sign
    assert_eq!(ord_from_str("£").unwrap(), 163); // Pound sign
    assert_eq!(ord_from_str("¥").unwrap(), 165); // Yen sign
    assert_eq!(ord_from_str("©").unwrap(), 169); // Copyright
    assert_eq!(ord_from_str("®").unwrap(), 174); // Registered
}

#[test]
fn test_ord_from_str_unicode_supplementary() {
    // Supplementary planes (emoji, etc.)
    assert_eq!(ord_from_str("🎉").unwrap(), 127881); // Party popper
    assert_eq!(ord_from_str("😀").unwrap(), 128512); // Grinning face
    assert_eq!(ord_from_str("🚀").unwrap(), 128640); // Rocket
    assert_eq!(ord_from_str("💻").unwrap(), 128187); // Laptop
}

#[test]
fn test_ord_from_str_unicode_cjk() {
    assert_eq!(ord_from_str("中").unwrap(), 20013); // Chinese
    assert_eq!(ord_from_str("日").unwrap(), 26085); // Japanese
    assert_eq!(ord_from_str("한").unwrap(), 54620); // Korean
}

#[test]
fn test_ord_from_str_maps_internal_surrogate_carriers_back_to_python_code_points() {
    let surrogate =
        encode_python_code_point(0xDC80).expect("surrogate should map into carrier range");
    let text = surrogate.to_string();
    assert_eq!(ord_from_str(&text).unwrap(), 0xDC80);
}

#[test]
fn test_ord_from_str_empty() {
    let result = ord_from_str("");
    assert!(result.is_err());
    match result {
        Err(BuiltinError::TypeError(msg)) => {
            assert!(msg.contains("length 0"));
        }
        _ => panic!("Expected TypeError"),
    }
}

#[test]
fn test_ord_from_str_multiple_chars() {
    let result = ord_from_str("ab");
    assert!(result.is_err());
    match result {
        Err(BuiltinError::TypeError(msg)) => {
            assert!(msg.contains("length 2"));
        }
        _ => panic!("Expected TypeError"),
    }
}

#[test]
fn test_ord_from_str_multiple_emoji() {
    // This is 2 graphemes, 2 code points
    let result = ord_from_str("🎉🚀");
    assert!(result.is_err());
}

// =========================================================================
// chr() Argument Validation Tests
// =========================================================================

#[test]
fn test_chr_no_args() {
    let result = builtin_chr(&[]);
    assert!(result.is_err());
}

#[test]
fn test_chr_too_many_args() {
    let result = builtin_chr(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    assert!(result.is_err());
}

#[test]
fn test_chr_float_error() {
    let result = builtin_chr(&[Value::float(3.14)]);
    assert!(result.is_err());
    match result {
        Err(BuiltinError::TypeError(msg)) => {
            assert!(msg.contains("integer"));
        }
        _ => panic!("Expected TypeError"),
    }
}

#[test]
fn test_chr_ascii_and_unicode_return_strings() {
    let ascii = builtin_chr(&[Value::int(97).unwrap()]).unwrap();
    assert_eq!(value_to_rust_string(ascii), "a");

    let unicode = builtin_chr(&[Value::int(127881).unwrap()]).unwrap();
    assert_eq!(value_to_rust_string(unicode), "🎉");
}

// =========================================================================
// chr_from_code_point() Tests
// =========================================================================

#[test]
fn test_chr_from_code_point_ascii() {
    assert_eq!(chr_from_code_point(97).unwrap(), 'a');
    assert_eq!(chr_from_code_point(65).unwrap(), 'A');
    assert_eq!(chr_from_code_point(48).unwrap(), '0');
    assert_eq!(chr_from_code_point(32).unwrap(), ' ');
    assert_eq!(chr_from_code_point(126).unwrap(), '~');
}

#[test]
fn test_chr_from_code_point_control() {
    assert_eq!(chr_from_code_point(0).unwrap(), '\0');
    assert_eq!(chr_from_code_point(9).unwrap(), '\t');
    assert_eq!(chr_from_code_point(10).unwrap(), '\n');
    assert_eq!(chr_from_code_point(13).unwrap(), '\r');
}

#[test]
fn test_chr_from_code_point_unicode() {
    assert_eq!(chr_from_code_point(8364).unwrap(), '€');
    assert_eq!(chr_from_code_point(127881).unwrap(), '🎉');
    assert_eq!(chr_from_code_point(128512).unwrap(), '😀');
}

#[test]
fn test_chr_from_code_point_boundary() {
    // Minimum valid
    assert!(chr_from_code_point(0).is_ok());
    // Maximum valid (U+10FFFF)
    assert!(chr_from_code_point(MAX_UNICODE_CODE_POINT).is_ok());
}

#[test]
fn test_chr_from_code_point_surrogate_start() {
    let result = chr_from_code_point(SURROGATE_START);
    assert!(result.is_err());
    match result {
        Err(BuiltinError::ValueError(msg)) => {
            assert!(msg.contains("surrogate"));
        }
        _ => panic!("Expected ValueError"),
    }
}

#[test]
fn test_chr_from_code_point_surrogate_middle() {
    let result = chr_from_code_point(0xDA00);
    assert!(result.is_err());
}

#[test]
fn test_chr_from_code_point_surrogate_end() {
    let result = chr_from_code_point(SURROGATE_END);
    assert!(result.is_err());
}

#[test]
fn test_chr_from_code_point_too_large() {
    let result = chr_from_code_point(0x110000);
    assert!(result.is_err());
    match result {
        Err(BuiltinError::ValueError(msg)) => {
            assert!(msg.contains("too large"));
        }
        _ => panic!("Expected ValueError"),
    }
}

// =========================================================================
// ord/chr Roundtrip Tests
// =========================================================================

#[test]
fn test_ord_chr_roundtrip_ascii() {
    for cp in 0..=127u32 {
        let c = chr_from_code_point(cp).unwrap();
        let s = c.to_string();
        let result = ord_from_str(&s).unwrap();
        assert_eq!(result, cp, "Roundtrip failed for code point {}", cp);
    }
}

#[test]
fn test_ord_chr_roundtrip_extended() {
    let test_points = [
        128, 255, 256, 1000, 8364, 20013, 65535, 66000, 100000, 127881, 128512, 0x10FFFF,
    ];
    for cp in test_points {
        let c = chr_from_code_point(cp).unwrap();
        let s = c.to_string();
        let result = ord_from_str(&s).unwrap();
        assert_eq!(result, cp, "Roundtrip failed for code point {}", cp);
    }
}

// =========================================================================
// bytes() Constructor Tests
// =========================================================================

#[test]
fn test_bytes_too_many_args() {
    let err = builtin_bytes(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
        Value::int(4).unwrap(),
    ])
    .unwrap_err();
    assert!(matches!(err, BuiltinError::TypeError(_)));
}

#[test]
fn test_bytes_empty_constructor() {
    let value = builtin_bytes(&[]).unwrap();
    assert_eq!(byte_sequence_type(value), TypeId::BYTES);
    assert_eq!(value_to_byte_vec(value), Vec::<u8>::new());
}

#[test]
fn test_bytes_count_constructor() {
    let value = builtin_bytes(&[Value::int(4).unwrap()]).unwrap();
    assert_eq!(byte_sequence_type(value), TypeId::BYTES);
    assert_eq!(value_to_byte_vec(value), vec![0, 0, 0, 0]);
}

#[test]
fn test_bytes_bool_count_constructor() {
    let false_value = builtin_bytes(&[Value::bool(false)]).unwrap();
    assert_eq!(value_to_byte_vec(false_value), Vec::<u8>::new());

    let true_value = builtin_bytes(&[Value::bool(true)]).unwrap();
    assert_eq!(value_to_byte_vec(true_value), vec![0]);
}

#[test]
fn test_bytes_negative_count() {
    let err = builtin_bytes(&[Value::int(-5).unwrap()]).unwrap_err();
    assert!(matches!(err, BuiltinError::ValueError(_)));
    assert!(err.to_string().contains("negative count"));
}

#[test]
fn test_bytes_overflow_count() {
    let err = builtin_bytes(&[Value::int(MAX_BYTE_SEQUENCE_SIZE + 1).unwrap()]).unwrap_err();
    assert!(matches!(err, BuiltinError::OverflowError(_)));
}

#[test]
fn test_bytes_string_without_encoding_errors() {
    let err = builtin_bytes(&[Value::string(intern("abc"))]).unwrap_err();
    assert!(matches!(err, BuiltinError::TypeError(_)));
    assert!(err.to_string().contains("without an encoding"));
}

#[test]
fn test_bytes_from_tagged_string_with_utf8() {
    let value = builtin_bytes(&[
        Value::string(intern("h\u{00e9}")),
        Value::string(intern("utf-8")),
    ])
    .unwrap();
    assert_eq!(byte_sequence_type(value), TypeId::BYTES);
    assert_eq!(value_to_byte_vec(value), "h\u{00e9}".as_bytes().to_vec());
}

#[test]
fn test_bytes_from_heap_string_with_latin1() {
    let (heap_str, str_ptr) = boxed_value(StringObject::new("\u{00e9}"));
    let value = builtin_bytes(&[heap_str, Value::string(intern("latin-1"))]).unwrap();
    assert_eq!(value_to_byte_vec(value), vec![0xe9]);
    unsafe { drop_boxed(str_ptr) };
}

#[test]
fn test_bytes_ascii_error_policies() {
    let strict_err = builtin_bytes(&[
        Value::string(intern("A\u{00e9}")),
        Value::string(intern("ascii")),
    ])
    .unwrap_err();
    assert_unicode_encode_error(
        strict_err,
        "'ascii' codec can't encode character '\\xe9' in position 1: ordinal not in range(128)",
    );

    let ignore = builtin_bytes(&[
        Value::string(intern("A\u{00e9}")),
        Value::string(intern("ascii")),
        Value::string(intern("ignore")),
    ])
    .unwrap();
    assert_eq!(value_to_byte_vec(ignore), b"A");

    let replace = builtin_bytes(&[
        Value::string(intern("A\u{00e9}")),
        Value::string(intern("ascii")),
        Value::string(intern("replace")),
    ])
    .unwrap();
    assert_eq!(value_to_byte_vec(replace), b"A?");
}

#[test]
fn test_bytes_backslashreplace_error_policy() {
    let ascii = builtin_bytes(&[
        Value::string(intern("A\u{00e9}")),
        Value::string(intern("ascii")),
        Value::string(intern("backslashreplace")),
    ])
    .expect("ascii backslashreplace should escape non-ASCII characters");
    assert_eq!(value_to_byte_vec(ascii), b"A\\xe9");

    let latin1 = builtin_bytes(&[
        Value::string(intern("\u{20ac}")),
        Value::string(intern("latin-1")),
        Value::string(intern("backslashreplace")),
    ])
    .expect("latin-1 backslashreplace should escape non-Latin-1 characters");
    assert_eq!(value_to_byte_vec(latin1), b"\\u20ac");
}

#[test]
fn test_decode_bytes_backslashreplace_error_policy() {
    let ascii = decode_bytes_to_value(&[0x41, 0xFF, 0x42], Some("ascii"), Some("backslashreplace"))
        .expect("ascii backslashreplace should preserve invalid bytes");
    assert_eq!(value_to_rust_string(ascii), r"A\xffB");

    let utf8 = decode_bytes_to_value(&[0x41, 0xFF, 0x42], Some("utf-8"), Some("backslashreplace"))
        .expect("utf-8 backslashreplace should preserve invalid bytes");
    assert_eq!(value_to_rust_string(utf8), r"A\xffB");
}

#[test]
fn test_raw_unicode_escape_codec_roundtrips_python_wire_format() {
    let input = "A\u{00e9}\u{0100}\u{1d11e}";
    let encoded = encode_text_to_data(input, Some("raw-unicode-escape"), Some("strict"))
        .expect("raw-unicode-escape should encode all Python code points");
    assert_eq!(encoded, b"A\xe9\\u0100\\U0001d11e");

    let decoded = decode_bytes_to_text(&encoded, Some("raw-unicode-escape"), Some("strict"))
        .expect("raw-unicode-escape should decode its wire format");
    assert_eq!(decoded, input);
}

#[test]
fn test_raw_unicode_escape_codec_preserves_surrogate_code_points() {
    let surrogate =
        encode_python_code_point(0xDC80).expect("surrogate should map into carrier range");
    let encoded = encode_text_to_data(
        &surrogate.to_string(),
        Some("raw_unicode_escape"),
        Some("strict"),
    )
    .expect("raw-unicode-escape should encode surrogate code points");
    assert_eq!(encoded, b"\\udc80");

    let decoded = decode_bytes_to_text(b"\\udc80", Some("raw-unicode-escape"), Some("strict"))
        .expect("raw-unicode-escape should decode surrogate escapes");
    assert_eq!(decoded, surrogate.to_string());
}

#[test]
fn test_raw_unicode_escape_decode_error_policies() {
    let replaced = decode_bytes_to_text(b"\\u12zz", Some("raw-unicode-escape"), Some("replace"))
        .expect("replace should emit a replacement character");
    assert_eq!(replaced, "\u{FFFD}zz");

    let escaped = decode_bytes_to_text(
        b"\\u12",
        Some("raw-unicode-escape"),
        Some("backslashreplace"),
    )
    .expect("backslashreplace should preserve offending bytes");
    assert_eq!(escaped, r"\x5c\x75\x31\x32");
}

#[test]
fn test_bytes_utf8_strict_rejects_internal_surrogate_carriers() {
    let surrogate =
        encode_python_code_point(0xDC80).expect("surrogate should map into carrier range");
    let text = format!("A{surrogate}");
    let (heap_string, heap_ptr) = boxed_value(StringObject::from_string(text));

    let err = builtin_bytes(&[heap_string, Value::string(intern("utf-8"))]).unwrap_err();
    assert_unicode_encode_error(
        err,
        "'utf-8' codec can't encode character '\\udc80' in position 1: surrogates not allowed",
    );

    unsafe { drop_boxed(heap_ptr) };
}

#[test]
fn test_bytes_utf8_surrogatepass_encodes_internal_surrogate_carriers() {
    let surrogate =
        encode_python_code_point(0xDC80).expect("surrogate should map into carrier range");
    let text = format!("A{surrogate}");
    let (heap_string, heap_ptr) = boxed_value(StringObject::from_string(text));

    let encoded = builtin_bytes(&[
        heap_string,
        Value::string(intern("utf-8")),
        Value::string(intern("surrogatepass")),
    ])
    .expect("surrogatepass should encode surrogate carriers");
    assert_eq!(value_to_byte_vec(encoded), vec![0x41, 0xED, 0xB2, 0x80]);

    unsafe { drop_boxed(heap_ptr) };
}

#[test]
fn test_decode_bytes_utf8_surrogatepass_roundtrips_python_surrogate_code_points() {
    let decoded = decode_bytes_to_value(
        &[0x41, 0xED, 0xB2, 0x80],
        Some("utf-8"),
        Some("surrogatepass"),
    )
    .expect("surrogatepass should decode surrogate UTF-8 sequences");

    let expected_surrogate =
        encode_python_code_point(0xDC80).expect("surrogate should map into carrier range");
    assert_eq!(
        value_to_rust_string(decoded),
        format!("A{expected_surrogate}")
    );
}

#[test]
fn test_decode_bytes_utf8_surrogateescape_maps_invalid_bytes_to_surrogate_carriers() {
    let decoded =
        decode_bytes_to_value(&[0x41, 0xFF, 0x42], Some("utf-8"), Some("surrogateescape"))
            .expect("surrogateescape should preserve invalid bytes");

    let escaped = encode_python_code_point(0xDCFF).expect("surrogate should map");
    assert_eq!(value_to_rust_string(decoded), format!("A{escaped}B"));
}

#[test]
fn test_decode_bytes_utf8_strict_reports_unicode_decode_error() {
    let err = decode_bytes_to_value(&[0xFF], Some("utf-8"), Some("strict"))
        .expect_err("invalid UTF-8 should raise UnicodeDecodeError");
    assert_unicode_decode_error(
        err,
        "'utf-8' codec can't decode byte 0xff in position 0: invalid start byte",
    );
}

#[test]
fn test_bytes_latin1_strict_error() {
    let err = builtin_bytes(&[
        Value::string(intern("\u{20ac}")),
        Value::string(intern("latin-1")),
    ])
    .unwrap_err();
    assert_unicode_encode_error(
        err,
        "'latin-1' codec can't encode character '\\u20ac' in position 0: ordinal not in range(256)",
    );
}

#[test]
fn test_bytes_unknown_encoding_and_error_policy() {
    let unknown_encoding = builtin_bytes(&[
        Value::string(intern("abc")),
        Value::string(intern("does-not-exist")),
    ])
    .unwrap_err();
    assert!(matches!(unknown_encoding, BuiltinError::ValueError(_)));
    assert!(unknown_encoding.to_string().contains("unknown encoding"));

    let unknown_policy = builtin_bytes(&[
        Value::string(intern("abc")),
        Value::string(intern("utf-8")),
        Value::string(intern("not-a-policy")),
    ])
    .unwrap_err();
    assert!(matches!(unknown_policy, BuiltinError::ValueError(_)));
    assert!(unknown_policy.to_string().contains("unknown error handler"));
}

#[test]
fn test_bytes_encoding_requires_string_source() {
    let err = builtin_bytes(&[Value::int(1).unwrap(), Value::string(intern("utf-8"))]).unwrap_err();
    assert!(matches!(err, BuiltinError::TypeError(_)));
    assert!(
        err.to_string()
            .contains("encoding without a string argument")
    );
}

#[test]
fn test_bytes_encoding_and_errors_argument_types() {
    let err2 = builtin_bytes(&[Value::string(intern("abc")), Value::int(1).unwrap()]).unwrap_err();
    assert!(matches!(err2, BuiltinError::TypeError(_)));
    assert!(err2.to_string().contains("argument 2 must be str"));

    let err3 = builtin_bytes(&[
        Value::string(intern("abc")),
        Value::string(intern("utf-8")),
        Value::int(1).unwrap(),
    ])
    .unwrap_err();
    assert!(matches!(err3, BuiltinError::TypeError(_)));
    assert!(err3.to_string().contains("argument 3 must be str"));
}

#[test]
fn test_bytes_from_iterable_values() {
    let list = prism_runtime::types::list::ListObject::from_slice(&[
        Value::int(65).unwrap(),
        Value::bool(true),
        Value::int(0).unwrap(),
        Value::int(255).unwrap(),
    ]);
    let (list_value, list_ptr) = boxed_value(list);

    let value = builtin_bytes(&[list_value]).unwrap();
    assert_eq!(value_to_byte_vec(value), vec![65, 1, 0, 255]);

    unsafe { drop_boxed(list_ptr) };
}

#[test]
fn test_bytes_from_iterable_out_of_range_error() {
    let list = prism_runtime::types::list::ListObject::from_slice(&[Value::int(256).unwrap()]);
    let (list_value, list_ptr) = boxed_value(list);
    let err = builtin_bytes(&[list_value]).unwrap_err();
    assert!(matches!(err, BuiltinError::ValueError(_)));
    assert!(err.to_string().contains("range(0, 256)"));
    unsafe { drop_boxed(list_ptr) };
}

#[test]
fn test_bytes_from_iterable_non_int_error() {
    let list = prism_runtime::types::list::ListObject::from_slice(&[Value::string(intern("x"))]);
    let (list_value, list_ptr) = boxed_value(list);
    let err = builtin_bytes(&[list_value]).unwrap_err();
    assert!(matches!(err, BuiltinError::TypeError(_)));
    assert!(
        err.to_string()
            .contains("cannot be interpreted as an integer")
    );
    unsafe { drop_boxed(list_ptr) };
}

#[test]
fn test_bytes_from_bytes_returns_same_object() {
    let (src, src_ptr) = boxed_value(BytesObject::from_slice(b"abc"));
    let out = builtin_bytes(&[src]).unwrap();
    assert_eq!(out.as_object_ptr(), src.as_object_ptr());
    unsafe { drop_boxed(src_ptr) };
}

#[test]
fn test_bytes_from_bytearray_copies() {
    let (src, src_ptr) = boxed_value(BytesObject::bytearray_from_slice(&[1, 2, 3]));
    let out = builtin_bytes(&[src]).unwrap();
    assert_eq!(byte_sequence_type(out), TypeId::BYTES);
    assert_eq!(value_to_byte_vec(out), vec![1, 2, 3]);
    assert_ne!(out.as_object_ptr(), src.as_object_ptr());
    unsafe { drop_boxed(src_ptr) };
}

// =========================================================================
// bytearray() Constructor Tests
// =========================================================================

#[test]
fn test_bytearray_too_many_args() {
    let err = builtin_bytearray(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
        Value::int(4).unwrap(),
    ])
    .unwrap_err();
    assert!(matches!(err, BuiltinError::TypeError(_)));
}

#[test]
fn test_bytearray_empty_and_count_constructor() {
    let empty = builtin_bytearray(&[]).unwrap();
    assert_eq!(byte_sequence_type(empty), TypeId::BYTEARRAY);
    assert_eq!(value_to_byte_vec(empty), Vec::<u8>::new());

    let counted = builtin_bytearray(&[Value::int(3).unwrap()]).unwrap();
    assert_eq!(byte_sequence_type(counted), TypeId::BYTEARRAY);
    assert_eq!(value_to_byte_vec(counted), vec![0, 0, 0]);
}

#[test]
fn test_bytearray_negative_count() {
    let err = builtin_bytearray(&[Value::int(-5).unwrap()]).unwrap_err();
    assert!(matches!(err, BuiltinError::ValueError(_)));
}

#[test]
fn test_bytearray_from_string_with_encoding() {
    let out = builtin_bytearray(&[
        Value::string(intern("h\u{00e9}")),
        Value::string(intern("utf-8")),
    ])
    .unwrap();
    assert_eq!(byte_sequence_type(out), TypeId::BYTEARRAY);
    assert_eq!(value_to_byte_vec(out), "h\u{00e9}".as_bytes());
}

#[test]
fn test_bytearray_encoding_without_string_error() {
    let err =
        builtin_bytearray(&[Value::int(1).unwrap(), Value::string(intern("utf-8"))]).unwrap_err();
    assert!(matches!(err, BuiltinError::TypeError(_)));
    assert!(
        err.to_string()
            .contains("encoding without a string argument")
    );
}

#[test]
fn test_bytearray_from_iterable_and_range_error() {
    let list = prism_runtime::types::list::ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);
    let (list_value, list_ptr) = boxed_value(list);
    let out = builtin_bytearray(&[list_value]).unwrap();
    assert_eq!(value_to_byte_vec(out), vec![1, 2, 3]);
    unsafe { drop_boxed(list_ptr) };

    let bad = prism_runtime::types::list::ListObject::from_slice(&[Value::int(-1).unwrap()]);
    let (bad_value, bad_ptr) = boxed_value(bad);
    let err = builtin_bytearray(&[bad_value]).unwrap_err();
    assert!(matches!(err, BuiltinError::ValueError(_)));
    assert!(err.to_string().contains("range(0, 256)"));
    unsafe { drop_boxed(bad_ptr) };
}

#[test]
fn test_bytearray_from_bytes_and_bytearray_copy() {
    let (bytes_src, bytes_src_ptr) = boxed_value(BytesObject::from_slice(&[5, 6]));
    let from_bytes = builtin_bytearray(&[bytes_src]).unwrap();
    assert_eq!(byte_sequence_type(from_bytes), TypeId::BYTEARRAY);
    assert_eq!(value_to_byte_vec(from_bytes), vec![5, 6]);
    assert_ne!(from_bytes.as_object_ptr(), bytes_src.as_object_ptr());
    unsafe { drop_boxed(bytes_src_ptr) };

    let (bytearray_src, bytearray_src_ptr) =
        boxed_value(BytesObject::bytearray_from_slice(&[7, 8]));
    let from_bytearray = builtin_bytearray(&[bytearray_src]).unwrap();
    assert_eq!(byte_sequence_type(from_bytearray), TypeId::BYTEARRAY);
    assert_eq!(value_to_byte_vec(from_bytearray), vec![7, 8]);
    assert_ne!(
        from_bytearray.as_object_ptr(),
        bytearray_src.as_object_ptr()
    );
    unsafe { drop_boxed(bytearray_src_ptr) };
}

// =========================================================================
// format() Argument Validation Tests
// =========================================================================

#[test]
fn test_format_no_args() {
    let result = builtin_format(&[]);
    assert!(result.is_err());
}

#[test]
fn test_format_too_many_args() {
    let result = builtin_format(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);
    assert!(result.is_err());
}

#[test]
fn test_format_default_and_numeric_specs() {
    let default = builtin_format(&[Value::int(42).unwrap()]).unwrap();
    assert_eq!(value_to_rust_string(default), "42");

    let hex = builtin_format(&[Value::int(255).unwrap(), Value::string(intern("#x"))]).unwrap();
    assert_eq!(value_to_rust_string(hex), "0xff");

    let padded_hex =
        builtin_format(&[Value::int(255).unwrap(), Value::string(intern("08X"))]).unwrap();
    assert_eq!(value_to_rust_string(padded_hex), "000000FF");

    let grouped =
        builtin_format(&[Value::int(1_234_567).unwrap(), Value::string(intern(","))]).unwrap();
    assert_eq!(value_to_rust_string(grouped), "1,234,567");
}

#[test]
fn test_format_second_arg_must_be_str() {
    let err = builtin_format(&[Value::int(1).unwrap(), Value::int(2).unwrap()]).unwrap_err();
    assert!(matches!(err, BuiltinError::TypeError(_)));
    assert!(err.to_string().contains("argument 2 must be str"));
}

#[test]
fn test_format_non_empty_spec_on_unsupported_type_errors() {
    let err = builtin_format(&[Value::none(), Value::string(intern("x"))]).unwrap_err();
    assert!(matches!(err, BuiltinError::TypeError(_)));
    assert!(err.to_string().contains("unsupported format string"));
}

// =========================================================================
// Utility Function Tests
// =========================================================================

#[test]
fn test_is_ascii() {
    assert!(is_ascii(0));
    assert!(is_ascii(65));
    assert!(is_ascii(127));
    assert!(!is_ascii(128));
    assert!(!is_ascii(256));
    assert!(!is_ascii(8364));
}

#[test]
fn test_is_valid_code_point() {
    // Valid points
    assert!(is_valid_code_point(0));
    assert!(is_valid_code_point(127));
    assert!(is_valid_code_point(128));
    assert!(is_valid_code_point(8364));
    assert!(is_valid_code_point(MAX_UNICODE_CODE_POINT));

    // Surrogates are invalid
    assert!(!is_valid_code_point(0xD800));
    assert!(!is_valid_code_point(0xDFFF));
    assert!(!is_valid_code_point(0xDA00));

    // Above max is invalid
    assert!(!is_valid_code_point(0x110000));
    assert!(!is_valid_code_point(0x1FFFFF));
}

#[test]
fn test_is_surrogate() {
    // Before surrogate range
    assert!(!is_surrogate(0xD7FF));

    // Surrogate range
    assert!(is_surrogate(0xD800));
    assert!(is_surrogate(0xDA00));
    assert!(is_surrogate(0xDC00));
    assert!(is_surrogate(0xDFFF));

    // After surrogate range
    assert!(!is_surrogate(0xE000));
}

// =========================================================================
// Format Helper Tests
// =========================================================================

#[test]
fn test_format_with_thousands_separator() {
    assert_eq!(format_with_thousands_separator(0), "0");
    assert_eq!(format_with_thousands_separator(1), "1");
    assert_eq!(format_with_thousands_separator(12), "12");
    assert_eq!(format_with_thousands_separator(123), "123");
    assert_eq!(format_with_thousands_separator(1234), "1,234");
    assert_eq!(format_with_thousands_separator(12345), "12,345");
    assert_eq!(format_with_thousands_separator(123456), "123,456");
    assert_eq!(format_with_thousands_separator(1234567), "1,234,567");
    assert_eq!(format_with_thousands_separator(-1234567), "-1,234,567");
}

#[test]
fn test_format_with_underscore_separator() {
    assert_eq!(format_with_underscore_separator(0), "0");
    assert_eq!(format_with_underscore_separator(1234), "1_234");
    assert_eq!(format_with_underscore_separator(1234567), "1_234_567");
    assert_eq!(format_with_underscore_separator(-1234567), "-1_234_567");
}

#[test]
fn test_parse_precision() {
    assert_eq!(parse_precision(".2f"), Some(2));
    assert_eq!(parse_precision(".10g"), Some(10));
    assert_eq!(parse_precision(".0"), Some(0));
    assert_eq!(parse_precision(""), None);
    assert_eq!(parse_precision("f"), None);
}

// =========================================================================
// extract_code_point() Tests
// =========================================================================

#[test]
fn test_extract_code_point_int() {
    assert_eq!(extract_code_point(&Value::int(97).unwrap()).unwrap(), 97);
    assert_eq!(extract_code_point(&Value::int(0).unwrap()).unwrap(), 0);
    assert_eq!(
        extract_code_point(&Value::int(0x10FFFF).unwrap()).unwrap(),
        0x10FFFF
    );
}

#[test]
fn test_extract_code_point_bool() {
    assert_eq!(extract_code_point(&Value::bool(true)).unwrap(), 1);
    assert_eq!(extract_code_point(&Value::bool(false)).unwrap(), 0);
}

#[test]
fn test_extract_code_point_negative() {
    let result = extract_code_point(&Value::int(-1).unwrap());
    assert!(result.is_err());
    match result {
        Err(BuiltinError::ValueError(msg)) => {
            assert!(msg.contains("negative"));
        }
        _ => panic!("Expected ValueError"),
    }
}

#[test]
fn test_extract_code_point_too_large() {
    let result = extract_code_point(&Value::int(0x110000).unwrap());
    assert!(result.is_err());
    match result {
        Err(BuiltinError::ValueError(msg)) => {
            assert!(msg.contains("too large"));
        }
        _ => panic!("Expected ValueError"),
    }
}

#[test]
fn test_extract_code_point_float_error() {
    let result = extract_code_point(&Value::float(97.0));
    assert!(result.is_err());
    match result {
        Err(BuiltinError::TypeError(msg)) => {
            assert!(msg.contains("integer"));
        }
        _ => panic!("Expected TypeError"),
    }
}

#[test]
fn test_extract_code_point_none_error() {
    let result = extract_code_point(&Value::none());
    assert!(result.is_err());
}

// =========================================================================
// Edge Case Tests
// =========================================================================

#[test]
fn test_chr_boundary_before_surrogate() {
    // U+D7FF is last valid before surrogate range
    assert!(chr_from_code_point(0xD7FF).is_ok());
}

#[test]
fn test_chr_boundary_after_surrogate() {
    // U+E000 is first valid after surrogate range
    assert!(chr_from_code_point(0xE000).is_ok());
}

#[test]
fn test_chr_max_bmp() {
    // U+FFFF is last code point in BMP
    assert!(chr_from_code_point(0xFFFF).is_ok());
}

#[test]
fn test_chr_first_supplementary() {
    // U+10000 is first supplementary plane code point
    assert!(chr_from_code_point(0x10000).is_ok());
}

// =========================================================================
// Constant Verification Tests
// =========================================================================

#[test]
fn test_unicode_constants() {
    assert_eq!(MAX_UNICODE_CODE_POINT, 0x10FFFF);
    assert_eq!(SURROGATE_START, 0xD800);
    assert_eq!(SURROGATE_END, 0xDFFF);
    assert_eq!(ASCII_MAX, 0x7F);
}
