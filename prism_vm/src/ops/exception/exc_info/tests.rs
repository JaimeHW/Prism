use super::*;

// ════════════════════════════════════════════════════════════════════════
// ExcInfo Construction Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_exc_info_empty() {
    let info = ExcInfo::empty();
    assert!(info.is_empty());
    assert!(!info.is_active());
    assert!(!info.has_traceback());
}

#[test]
fn test_exc_info_empty_values() {
    let info = ExcInfo::empty();
    assert!(info.exc_type.is_none());
    assert!(info.exc_value.is_none());
    assert!(info.exc_traceback.is_none());
}

#[test]
fn test_exc_info_from_type_value() {
    let exc_type = Value::int(24).unwrap(); // TypeError
    let exc_value = Value::int(100).unwrap(); // placeholder
    let info = ExcInfo::from_type_value(exc_type, exc_value);

    assert!(!info.is_empty());
    assert!(info.is_active());
    assert!(!info.has_traceback());
}

#[test]
fn test_exc_info_new() {
    let exc_type = Value::int(24).unwrap(); // TypeError
    let exc_value = Value::int(100).unwrap();
    let exc_traceback = Value::int(200).unwrap();
    let info = ExcInfo::new(exc_type, exc_value, exc_traceback);

    assert!(!info.is_empty());
    assert!(info.is_active());
    // traceback is not none
    assert!(info.has_traceback());
}

#[test]
fn test_exc_info_with_none_traceback() {
    let info = ExcInfo::new(
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::none(),
    );
    assert!(!info.has_traceback());
}

// ════════════════════════════════════════════════════════════════════════
// ExcInfo Method Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_exc_info_is_empty_with_type_only() {
    // Type present but value is None - still considered empty
    let info = ExcInfo::new(Value::int(1).unwrap(), Value::none(), Value::none());
    assert!(info.is_empty());
}

#[test]
fn test_exc_info_to_tuple() {
    let exc_type = Value::int(1).unwrap();
    let exc_value = Value::int(2).unwrap();
    let exc_traceback = Value::int(3).unwrap();
    let info = ExcInfo::new(exc_type.clone(), exc_value.clone(), exc_traceback.clone());

    let (t, v, tb) = info.to_tuple();
    // Compare the values (they should be equal)
    assert!(!t.is_none());
    assert!(!v.is_none());
    assert!(!tb.is_none());
}

// ════════════════════════════════════════════════════════════════════════
// Default Implementation Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_exc_info_debug_format() {
    let info = ExcInfo::empty();
    let debug = format!("{:?}", info);
    assert!(debug.contains("ExcInfo"));
    assert!(debug.contains("exc_type"));
    assert!(debug.contains("exc_value"));
    assert!(debug.contains("has_traceback"));
}

// ════════════════════════════════════════════════════════════════════════
// Type Value Construction Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_build_type_value_valid() {
    let type_id = ExceptionTypeId::TypeError as u16;
    let value = build_type_value(type_id);
    // Currently returns int representation
    assert!(!value.is_none());
}

#[test]
fn test_build_type_value_invalid() {
    let type_id = 255u16; // Invalid type ID
    let value = build_type_value(type_id);
    // Invalid type ID returns None
    assert!(value.is_none());
}

#[test]
fn test_build_type_value_exception() {
    let type_id = ExceptionTypeId::Exception as u16;
    let value = build_type_value(type_id);
    assert!(!value.is_none());
}

// ════════════════════════════════════════════════════════════════════════
// Traceback Value Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_build_traceback_value_none() {
    let exc_value = Value::none();
    let tb = build_traceback_value(&exc_value);
    // Currently returns None as placeholder
    assert!(tb.is_none());
}

#[test]
fn test_build_traceback_value_int() {
    let exc_value = Value::int(42).unwrap();
    let tb = build_traceback_value(&exc_value);
    // Not an exception, returns None
    assert!(tb.is_none());
}

// ════════════════════════════════════════════════════════════════════════
// Explicit Build Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_build_exc_info_explicit() {
    let type_id = ExceptionTypeId::ValueError as u16;
    let exc_value = Value::int(42).unwrap();
    let exc_traceback = Value::none();

    let info = build_exc_info_explicit(type_id, exc_value, exc_traceback);

    assert!(!info.exc_type.is_none());
    assert!(!info.exc_value.is_none());
    assert!(info.exc_traceback.is_none());
}

#[test]
fn test_build_exc_info_explicit_with_traceback() {
    let type_id = ExceptionTypeId::TypeError as u16;
    let exc_value = Value::int(100).unwrap();
    let exc_traceback = Value::int(200).unwrap(); // placeholder

    let info = build_exc_info_explicit(type_id, exc_value, exc_traceback);

    assert!(info.has_traceback());
}

// ════════════════════════════════════════════════════════════════════════
// Edge Case Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_exc_info_all_none() {
    let info = ExcInfo::new(Value::none(), Value::none(), Value::none());
    assert!(info.is_empty());
}

#[test]
fn test_build_type_value_base_exception() {
    let type_id = ExceptionTypeId::BaseException as u16;
    let value = build_type_value(type_id);
    assert!(!value.is_none());
}

#[test]
fn test_build_type_value_stop_iteration() {
    let type_id = ExceptionTypeId::StopIteration as u16;
    let value = build_type_value(type_id);
    assert!(!value.is_none());
}
