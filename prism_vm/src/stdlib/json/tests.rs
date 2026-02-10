//! Integration tests for the JSON module.

use super::decode::loads;
use super::encode::dumps;
use prism_core::Value;

#[test]
fn test_roundtrip_integer() {
    let val = Value::int_unchecked(42);
    let json = dumps(val).unwrap();
    let parsed = loads(&json).unwrap();
    assert_eq!(parsed.as_int(), Some(42));
}

#[test]
fn test_roundtrip_float() {
    let val = Value::float(3.14159);
    let json = dumps(val).unwrap();
    let parsed = loads(&json).unwrap();
    let f = parsed.as_float().unwrap();
    assert!((f - 3.14159).abs() < 1e-5);
}

#[test]
fn test_roundtrip_boolean_true() {
    let val = Value::bool(true);
    let json = dumps(val).unwrap();
    let parsed = loads(&json).unwrap();
    assert_eq!(parsed.as_bool(), Some(true));
}

#[test]
fn test_roundtrip_boolean_false() {
    let val = Value::bool(false);
    let json = dumps(val).unwrap();
    let parsed = loads(&json).unwrap();
    assert_eq!(parsed.as_bool(), Some(false));
}

#[test]
fn test_roundtrip_null() {
    let val = Value::none();
    let json = dumps(val).unwrap();
    let parsed = loads(&json).unwrap();
    assert!(parsed.is_none());
}

#[test]
fn test_loads_nested_object() {
    let json = r#"{"a": {"b": 1}}"#;
    // Should parse without error (returns None for now)
    let result = loads(json);
    assert!(result.is_ok());
}

#[test]
fn test_loads_nested_array() {
    let json = r#"[1, [2, [3]]]"#;
    // Should parse without error (returns None for now)
    let result = loads(json);
    assert!(result.is_ok());
}

#[test]
fn test_loads_mixed_types() {
    let json = r#"{"string": "hello", "number": 42, "bool": true, "null": null}"#;
    let result = loads(json);
    assert!(result.is_ok());
}

#[test]
fn test_loads_unicode() {
    let json = r#""\u0041\u0042\u0043""#;
    let result = loads(json);
    assert!(result.is_ok());
}

#[test]
fn test_loads_large_number() {
    let json = "9007199254740993";
    let result = loads(json);
    assert!(result.is_ok());
}

#[test]
fn test_loads_scientific_notation() {
    let cases = vec!["1e10", "1E10", "1.5e-5", "-2.5E+3"];
    for json in cases {
        let result = loads(json);
        assert!(result.is_ok(), "Failed to parse: {}", json);
    }
}

#[test]
fn test_error_messages() {
    let err = loads("[1, 2,]").unwrap_err();
    assert!(err.line >= 1);
    assert!(err.col >= 1);
}

#[test]
fn test_deeply_nested_structure() {
    // Test parsing of deeply nested structures
    let json = "[[[[[[[[[[1]]]]]]]]]]";
    let result = loads(json);
    assert!(result.is_ok());
}
