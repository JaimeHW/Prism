use super::*;
use prism_code::CodeObject;
use std::sync::Arc;

fn generator_value(flags: GeneratorFlags) -> Value {
    let code = Arc::new(CodeObject::new("test_get_awaitable", "<test>"));
    let generator = GeneratorObject::with_flags(code, flags);
    let ptr = Box::into_raw(Box::new(generator)) as *const ();
    Value::object_ptr(ptr)
}

// =========================================================================
// Type Check Tests
// =========================================================================

#[test]
fn test_none_not_awaitable() {
    assert!(!is_native_awaitable(&Value::none()));
}

#[test]
fn test_int_not_awaitable() {
    let val = Value::int(42).unwrap();
    assert!(!is_native_awaitable(&val));
}

#[test]
fn test_bool_not_awaitable() {
    assert!(!is_native_awaitable(&Value::bool(true)));
    assert!(!is_native_awaitable(&Value::bool(false)));
}

#[test]
fn test_float_not_awaitable() {
    let val = Value::float(3.14);
    assert!(!is_native_awaitable(&val));
}

#[test]
fn test_coroutine_generator_is_native_awaitable() {
    let value = generator_value(GeneratorFlags::IS_COROUTINE | GeneratorFlags::INLINE_STORAGE);
    assert!(is_native_awaitable(&value));
}

#[test]
fn test_async_generator_is_native_awaitable() {
    let value = generator_value(GeneratorFlags::IS_ASYNC | GeneratorFlags::INLINE_STORAGE);
    assert!(is_native_awaitable(&value));
}

#[test]
fn test_regular_generator_not_native_awaitable() {
    let value = generator_value(GeneratorFlags::INLINE_STORAGE);
    assert!(!is_native_awaitable(&value));
}

// =========================================================================
// Type Name Tests
// =========================================================================

#[test]
fn test_type_name_none() {
    assert_eq!(type_name(&Value::none()), "NoneType");
}

#[test]
fn test_type_name_bool() {
    assert_eq!(type_name(&Value::bool(true)), "bool");
}

#[test]
fn test_type_name_int() {
    let val = Value::int(42).unwrap();
    assert_eq!(type_name(&val), "int");
}

#[test]
fn test_type_name_float() {
    let val = Value::float(3.14);
    assert_eq!(type_name(&val), "float");
}

// =========================================================================
// Iterator Check Tests
// =========================================================================

#[test]
fn test_is_iterator_rejects_none() {
    assert!(!is_iterator(&Value::none()));
}

#[test]
fn test_generator_is_iterator() {
    let value = generator_value(GeneratorFlags::INLINE_STORAGE);
    assert!(is_iterator(&value));
}
