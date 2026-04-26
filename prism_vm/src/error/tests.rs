use super::*;

#[test]
fn test_error_display() {
    let err = RuntimeError::type_error("expected int, got str");
    assert!(err.to_string().contains("TypeError"));
    assert!(err.to_string().contains("expected int"));
}

#[test]
fn test_zero_division() {
    let err = RuntimeError::zero_division();
    assert!(err.to_string().contains("ZeroDivisionError"));
}

#[test]
fn test_python_attribute_error_exception_is_classified_as_attribute_error() {
    let err = RuntimeError::exception(
        crate::stdlib::exceptions::ExceptionTypeId::AttributeError.as_u8() as u16,
        "_mock_methods",
    );
    assert!(err.is_attribute_error());
}

#[test]
fn test_name_error() {
    let err = RuntimeError::name_error("undefined_var");
    assert!(err.to_string().contains("NameError"));
    assert!(err.to_string().contains("undefined_var"));
}

#[test]
fn test_traceback() {
    let mut err = RuntimeError::type_error("test");
    err.add_traceback(TracebackEntry {
        func_name: "foo".into(),
        filename: "test.py".into(),
        line: 10,
    });
    assert_eq!(err.traceback.len(), 1);
    assert_eq!(&*err.traceback[0].func_name, "foo");
}

#[test]
fn test_builtin_stop_iteration_maps_to_runtime_stop_iteration() {
    let err = RuntimeError::from(crate::builtins::BuiltinError::StopIteration);
    assert!(matches!(err.kind(), RuntimeErrorKind::StopIteration));
    assert_eq!(err.to_string(), "StopIteration");
}

#[test]
fn test_execution_limit_error_display() {
    let err = RuntimeError::execution_limit_exceeded(128);
    assert_eq!(
        err.to_string(),
        "RuntimeError: execution step limit exceeded (128)"
    );
}

#[test]
fn test_python_exception_message_omits_type_prefix_for_type_error() {
    let err = RuntimeError::type_error("instance must not be None");
    assert_eq!(
        err.python_exception_message().as_deref(),
        Some("instance must not be None")
    );
}

#[test]
fn test_python_exception_message_formats_not_callable_without_prefix() {
    let err = RuntimeError::not_callable("widget");
    assert_eq!(
        err.python_exception_message().as_deref(),
        Some("'widget' object is not callable")
    );
}
