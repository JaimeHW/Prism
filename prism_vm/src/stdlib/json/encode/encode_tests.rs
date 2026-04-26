
use super::*;

#[test]
fn test_encode_null() {
    let s = dumps(Value::none()).unwrap();
    assert_eq!(s, "null");
}

#[test]
fn test_encode_true() {
    let s = dumps(Value::bool(true)).unwrap();
    assert_eq!(s, "true");
}

#[test]
fn test_encode_false() {
    let s = dumps(Value::bool(false)).unwrap();
    assert_eq!(s, "false");
}

#[test]
fn test_encode_integer() {
    let s = dumps(Value::int_unchecked(42)).unwrap();
    assert_eq!(s, "42");
}

#[test]
fn test_encode_negative() {
    let s = dumps(Value::int_unchecked(-123)).unwrap();
    assert_eq!(s, "-123");
}

#[test]
fn test_encode_float() {
    let s = dumps(Value::float(3.14)).unwrap();
    assert!(s.starts_with("3.14"));
}

#[test]
fn test_encode_nan_error() {
    let err = dumps(Value::float(f64::NAN)).unwrap_err();
    assert!(err.msg.contains("NaN"));
}

#[test]
fn test_encode_infinity_error() {
    let err = dumps(Value::float(f64::INFINITY)).unwrap_err();
    assert!(err.msg.contains("Infinity"));
}

#[test]
fn test_escape_string() {
    let escaped = escape_string("hello\nworld");
    assert_eq!(escaped, r#""hello\nworld""#);
}

#[test]
fn test_escape_quotes() {
    let escaped = escape_string(r#"say "hello""#);
    assert_eq!(escaped, r#""say \"hello\"""#);
}

#[test]
fn test_escape_backslash() {
    let escaped = escape_string(r"path\to\file");
    assert_eq!(escaped, r#""path\\to\\file""#);
}

#[test]
fn test_escape_control_char() {
    let escaped = escape_string("\x00");
    assert_eq!(escaped, r#""\u0000""#);
}
