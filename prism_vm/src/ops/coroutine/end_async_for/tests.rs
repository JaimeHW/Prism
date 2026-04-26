use super::*;
use crate::builtins::create_exception;

#[test]
fn test_none_not_stop_async_iteration() {
    // None is never StopAsyncIteration
    assert!(!is_stop_async_iteration(&Value::none()));
}

#[test]
fn test_int_not_stop_async_iteration() {
    let val = Value::int(42).unwrap();
    assert!(!is_stop_async_iteration(&val));
}

#[test]
fn test_bool_not_stop_async_iteration() {
    assert!(!is_stop_async_iteration(&Value::bool(true)));
    assert!(!is_stop_async_iteration(&Value::bool(false)));
}

#[test]
fn test_float_not_stop_async_iteration() {
    let val = Value::float(3.14);
    assert!(!is_stop_async_iteration(&val));
}

#[test]
fn test_stop_async_iteration_detected() {
    let exc = create_exception(ExceptionTypeId::StopAsyncIteration, None);
    assert!(is_stop_async_iteration(&exc));
    assert_eq!(
        exception_type_id(&exc),
        Some(ExceptionTypeId::StopAsyncIteration as u16)
    );
}

#[test]
fn test_stop_iteration_not_stop_async_iteration() {
    let exc = create_exception(ExceptionTypeId::StopIteration, None);
    assert!(!is_stop_async_iteration(&exc));
    assert_eq!(
        exception_type_id(&exc),
        Some(ExceptionTypeId::StopIteration as u16)
    );
}
