use super::*;
use prism_runtime::types::TupleObject;

fn exception_type_value(name: &str) -> Value {
    let exc_type = crate::builtins::get_exception_type(name)
        .unwrap_or_else(|| panic!("missing exception type: {name}"));
    Value::object_ptr(exc_type as *const _ as *const ())
}

fn tuple_value(values: &[Value]) -> Value {
    let tuple = TupleObject::from_slice(values);
    let ptr = Box::leak(Box::new(tuple)) as *mut TupleObject as *const ();
    Value::object_ptr(ptr)
}

unsafe fn drop_boxed<T>(ptr: *mut T) {
    unsafe {
        drop(Box::from_raw(ptr));
    }
}

// ════════════════════════════════════════════════════════════════════════
// Constant Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_no_type_id_sentinel() {
    assert_eq!(NO_TYPE_ID, 0xFFFF);
    assert_eq!(NO_TYPE_ID, u16::MAX);
}

#[test]
fn test_default_exception_type() {
    assert_eq!(DEFAULT_EXCEPTION_TYPE, 0);
}

#[test]
fn test_inline_tuple_check_size() {
    assert_eq!(INLINE_TUPLE_CHECK_SIZE, 4);
}

// ════════════════════════════════════════════════════════════════════════
// Type Extraction Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_extract_type_id_from_none() {
    let value = Value::none();
    let type_id = extract_type_id_from_value(&value);
    // None is not an exception, should return Exception type
    assert_eq!(type_id, ExceptionTypeId::Exception as u16);
}

#[test]
fn test_extract_type_id_from_int() {
    let value = Value::int(42).unwrap();
    let type_id = extract_type_id_from_value(&value);
    // Int is not an exception, should return Exception type
    assert_eq!(type_id, ExceptionTypeId::Exception as u16);
}

#[test]
fn test_extract_type_id_from_bool() {
    let value = Value::bool(true);
    let type_id = extract_type_id_from_value(&value);
    assert_eq!(type_id, ExceptionTypeId::Exception as u16);
}

#[test]
fn test_extract_type_id_fast_path() {
    let value = Value::none();
    // With explicit type ID, should return that directly
    let type_id = extract_type_id(24, &value); // TypeError
    assert_eq!(type_id, 24);
}

#[test]
fn test_extract_type_id_slow_path() {
    let value = Value::none();
    // With NO_TYPE_ID, should call dynamic extraction
    let type_id = extract_type_id(NO_TYPE_ID, &value);
    assert_eq!(type_id, ExceptionTypeId::Exception as u16);
}

// ════════════════════════════════════════════════════════════════════════
// is_subclass Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_is_subclass_self() {
    // Every type is a subclass of itself
    let exc = ExceptionTypeId::Exception as u16;
    assert!(is_subclass(exc, exc));
}

#[test]
fn test_is_subclass_type_error_of_exception() {
    let type_error = ExceptionTypeId::TypeError as u16;
    let exception = ExceptionTypeId::Exception as u16;
    assert!(is_subclass(type_error, exception));
}

#[test]
fn test_is_subclass_value_error_of_exception() {
    let value_error = ExceptionTypeId::ValueError as u16;
    let exception = ExceptionTypeId::Exception as u16;
    assert!(is_subclass(value_error, exception));
}

#[test]
fn test_is_subclass_not_related() {
    let type_error = ExceptionTypeId::TypeError as u16;
    let value_error = ExceptionTypeId::ValueError as u16;
    // Neither is a subclass of the other
    assert!(!is_subclass(type_error, value_error));
    assert!(!is_subclass(value_error, type_error));
}

#[test]
fn test_is_subclass_invalid_types() {
    // Invalid type IDs should return false
    assert!(!is_subclass(255, 255));
    assert!(!is_subclass(200, 4));
}

// ════════════════════════════════════════════════════════════════════════
// Dynamic Match Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_check_dynamic_match_none() {
    let exc_type_id = ExceptionTypeId::TypeError as u16;
    let type_value = Value::none();
    // None is not a valid type, should return false
    assert!(!check_dynamic_match(None, exc_type_id, &type_value));
}

#[test]
fn test_check_dynamic_match_int() {
    let exc_type_id = ExceptionTypeId::TypeError as u16;
    let type_value = Value::int(42).unwrap();
    // Int is not a valid type, should return false
    assert!(!check_dynamic_match(None, exc_type_id, &type_value));
}

#[test]
fn test_check_dynamic_match_type_object_direct_match() {
    let exc_type_id = ExceptionTypeId::TypeError as u16;
    let type_value = exception_type_value("TypeError");
    assert!(check_dynamic_match(None, exc_type_id, &type_value));
}

#[test]
fn test_check_dynamic_match_type_object_subclass_match() {
    let exc_type_id = ExceptionTypeId::TypeError as u16;
    let type_value = exception_type_value("Exception");
    assert!(check_dynamic_match(None, exc_type_id, &type_value));
}

// ════════════════════════════════════════════════════════════════════════
// Tuple Match Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_check_tuple_match_none() {
    let exc_type_id = ExceptionTypeId::TypeError as u16;
    let types_tuple = Value::none();
    // None is not a tuple, should return false
    assert!(!check_tuple_match(None, exc_type_id, &types_tuple));
}

#[test]
fn test_check_tuple_match_int() {
    let exc_type_id = ExceptionTypeId::TypeError as u16;
    let types_tuple = Value::int(42).unwrap();
    // Int is not a tuple, should return false
    assert!(!check_tuple_match(None, exc_type_id, &types_tuple));
}

#[test]
fn test_check_tuple_match_bool() {
    let exc_type_id = ExceptionTypeId::ValueError as u16;
    let types_tuple = Value::bool(true);
    assert!(!check_tuple_match(None, exc_type_id, &types_tuple));
}

#[test]
fn test_check_tuple_match_type_object_elements() {
    let exc_type_id = ExceptionTypeId::TypeError as u16;
    let value_error = exception_type_value("ValueError");
    let type_error = exception_type_value("TypeError");
    let types_tuple = tuple_value(&[value_error, type_error]);
    assert!(check_tuple_match(None, exc_type_id, &types_tuple));

    let tuple_ptr =
        types_tuple.as_object_ptr().expect("tuple should be object") as *mut TupleObject;
    unsafe { drop_boxed(tuple_ptr) };
}

// ════════════════════════════════════════════════════════════════════════
// Inline Tuple Match Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_match_tuple_inline_empty() {
    let exc_type_id = ExceptionTypeId::TypeError as u16;
    assert!(!match_tuple_inline(None, exc_type_id, &[]));
}

#[test]
fn test_match_tuple_inline_one_no_match() {
    let exc_type_id = ExceptionTypeId::TypeError as u16;
    let elements = [Value::none()];
    assert!(!match_tuple_inline(None, exc_type_id, &elements));
}

#[test]
fn test_match_tuple_inline_two_no_match() {
    let exc_type_id = ExceptionTypeId::TypeError as u16;
    let elements = [Value::none(), Value::int(1).unwrap()];
    assert!(!match_tuple_inline(None, exc_type_id, &elements));
}

#[test]
fn test_match_tuple_inline_three_no_match() {
    let exc_type_id = ExceptionTypeId::TypeError as u16;
    let elements = [Value::none(), Value::int(1).unwrap(), Value::bool(false)];
    assert!(!match_tuple_inline(None, exc_type_id, &elements));
}

#[test]
fn test_match_tuple_inline_four_no_match() {
    let exc_type_id = ExceptionTypeId::TypeError as u16;
    let elements = [
        Value::none(),
        Value::int(1).unwrap(),
        Value::bool(false),
        Value::float(3.14),
    ];
    assert!(!match_tuple_inline(None, exc_type_id, &elements));
}

// ════════════════════════════════════════════════════════════════════════
// Loop Match Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_match_tuple_loop_empty() {
    let exc_type_id = ExceptionTypeId::TypeError as u16;
    assert!(!match_tuple_loop(None, exc_type_id, &[]));
}

#[test]
fn test_match_tuple_loop_no_match() {
    let exc_type_id = ExceptionTypeId::TypeError as u16;
    let elements: Vec<Value> = (0..10).map(|i| Value::int(i).unwrap()).collect();
    assert!(!match_tuple_loop(None, exc_type_id, &elements));
}

// ════════════════════════════════════════════════════════════════════════
// Edge Case Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_extract_type_from_type_value_none() {
    let type_value = Value::none();
    assert!(extract_type_from_type_value(&type_value).is_none());
}

#[test]
fn test_extract_type_from_type_value_int() {
    let type_value = Value::int(42).unwrap();
    assert!(extract_type_from_type_value(&type_value).is_none());
}

#[test]
fn test_extract_type_from_type_value_exception_type_object() {
    let type_value = exception_type_value("TypeError");
    assert_eq!(
        extract_type_from_type_value(&type_value),
        Some(ExceptionTypeId::TypeError as u16)
    );
}

#[test]
fn test_try_extract_exception_type_id_non_ptr() {
    let value = Value::int(42).unwrap();
    assert!(try_extract_exception_type_id(&value).is_none());
}

#[test]
fn test_try_extract_type_object_id_non_ptr() {
    let value = Value::bool(true);
    assert!(try_extract_type_object_id(&value).is_none());
}

#[test]
fn test_try_extract_tuple_elements_non_ptr() {
    let value = Value::float(1.23);
    assert!(try_extract_tuple_elements(&value).is_none());
}
