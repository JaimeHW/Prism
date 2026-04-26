use super::*;
use prism_code::CodeObject;
use std::sync::Arc;

fn generator_value(flags: GeneratorFlags) -> Value {
    let code = Arc::new(CodeObject::new("test_get_aiter", "<test>"));
    let generator = GeneratorObject::with_flags(code, flags);
    let ptr = Box::into_raw(Box::new(generator)) as *const ();
    Value::object_ptr(ptr)
}

#[test]
fn test_none_not_async_iterator() {
    assert!(!is_async_iterator(&Value::none()));
}

#[test]
fn test_int_not_async_iterator() {
    let val = Value::int(42).unwrap();
    assert!(!is_async_iterator(&val));
}

#[test]
fn test_async_generator_is_async_iterator() {
    let val = generator_value(GeneratorFlags::IS_ASYNC | GeneratorFlags::INLINE_STORAGE);
    assert!(is_async_iterator(&val));
    assert!(has_anext(&val));
}

#[test]
fn test_regular_generator_not_async_iterator() {
    let val = generator_value(GeneratorFlags::INLINE_STORAGE);
    assert!(!is_async_iterator(&val));
    assert!(!has_anext(&val));
}

#[test]
fn test_type_name_various() {
    assert_eq!(type_name(&Value::none()), "NoneType");
    assert_eq!(type_name(&Value::bool(true)), "bool");
    assert_eq!(type_name(&Value::int(1).unwrap()), "int");
    assert_eq!(type_name(&Value::float(1.0)), "float");
}
