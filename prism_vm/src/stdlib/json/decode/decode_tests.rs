
use super::*;

#[test]
fn test_decode_null() {
    let val = loads("null").unwrap();
    assert!(val.is_none());
}

#[test]
fn test_decode_true() {
    let val = loads("true").unwrap();
    assert_eq!(val.as_bool(), Some(true));
}

#[test]
fn test_decode_false() {
    let val = loads("false").unwrap();
    assert_eq!(val.as_bool(), Some(false));
}

#[test]
fn test_decode_integer() {
    let val = loads("42").unwrap();
    assert_eq!(val.as_int(), Some(42));
}

#[test]
fn test_decode_negative_integer() {
    let val = loads("-123").unwrap();
    assert_eq!(val.as_int(), Some(-123));
}

#[test]
fn test_decode_float() {
    let val = loads("3.14").unwrap();
    let f = val.as_float().unwrap();
    assert!((f - 3.14).abs() < 1e-10);
}

#[test]
fn test_decode_float_exponent() {
    let val = loads("1.5e10").unwrap();
    let f = val.as_float().unwrap();
    assert!((f - 1.5e10).abs() < 1e5);
}

#[test]
fn test_decode_string() {
    let val = loads(r#""hello""#).unwrap();
    assert!(val.is_string());
}

#[test]
fn test_decode_string_escapes() {
    let val = loads(r#""hello\nworld""#).unwrap();
    assert!(val.is_string());
}

#[test]
fn test_decode_empty_object() {
    let val = loads("{}").unwrap();
    // Currently returns None until dict support
    assert!(val.is_none());
}

#[test]
fn test_decode_empty_array() {
    let val = loads("[]").unwrap();
    // Currently returns None until list support
    assert!(val.is_none());
}

#[test]
fn test_decode_whitespace() {
    let val = loads("  42  ").unwrap();
    assert_eq!(val.as_int(), Some(42));
}

#[test]
fn test_decode_error_extra_data() {
    let err = loads("42 extra").unwrap_err();
    assert!(err.msg.contains("Extra data"));
}

#[test]
fn test_decode_error_unterminated_string() {
    let err = loads(r#""unterminated"#).unwrap_err();
    assert!(err.msg.contains("Unterminated"));
}

#[test]
fn test_decode_error_invalid_escape() {
    let err = loads(r#""\x""#).unwrap_err();
    assert!(err.msg.contains("Invalid escape"));
}

#[test]
fn test_decode_error_unexpected_char() {
    let err = loads("undefined").unwrap_err();
    assert!(err.msg.contains("Unexpected"));
}
