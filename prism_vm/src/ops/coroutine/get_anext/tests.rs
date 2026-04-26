use super::*;
use prism_code::CodeObject;
use std::sync::Arc;

fn generator_value(flags: GeneratorFlags) -> Value {
    let code = Arc::new(CodeObject::new("test_get_anext", "<test>"));
    let generator = GeneratorObject::with_flags(code, flags);
    let ptr = Box::into_raw(Box::new(generator)) as *const ();
    Value::object_ptr(ptr)
}

#[test]
fn test_none_has_no_native_anext() {
    assert!(try_native_anext(&Value::none()).is_none());
}

#[test]
fn test_int_has_no_native_anext() {
    let val = Value::int(42).unwrap();
    assert!(try_native_anext(&val).is_none());
}

#[test]
fn test_async_generator_has_native_anext() {
    let val = generator_value(GeneratorFlags::IS_ASYNC | GeneratorFlags::INLINE_STORAGE);
    assert_eq!(try_native_anext(&val), Some(val));
}

#[test]
fn test_regular_generator_has_no_native_anext() {
    let val = generator_value(GeneratorFlags::INLINE_STORAGE);
    assert!(try_native_anext(&val).is_none());
}

#[test]
fn test_type_name_coverage() {
    assert_eq!(type_name(&Value::none()), "NoneType");
    assert_eq!(type_name(&Value::bool(false)), "bool");
    assert_eq!(type_name(&Value::int(0).unwrap()), "int");
    assert_eq!(type_name(&Value::float(0.0)), "float");
}
