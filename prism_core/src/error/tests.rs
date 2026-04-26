use super::*;

#[test]
fn test_lex_error_creation() {
    let span = Span::new(0, 10);
    let err = PrismError::lex("unexpected character", span);

    match &err {
        PrismError::LexError { message, span: s } => {
            assert_eq!(message, "unexpected character");
            assert_eq!(s.start, 0);
            assert_eq!(s.end, 10);
        }
        _ => panic!("Expected LexError"),
    }

    assert_eq!(err.exception_type(), "SyntaxError");
    assert_eq!(err.to_string(), "SyntaxError: unexpected character");
}

#[test]
fn test_syntax_error_creation() {
    let span = Span::new(5, 15);
    let err = PrismError::syntax("invalid syntax", span);

    assert_eq!(err.exception_type(), "SyntaxError");
    assert_eq!(err.to_string(), "SyntaxError: invalid syntax");
}

#[test]
fn test_name_error_creation() {
    let err = PrismError::name("undefined_var");

    match &err {
        PrismError::NameError { name } => {
            assert_eq!(name, "undefined_var");
        }
        _ => panic!("Expected NameError"),
    }

    assert_eq!(err.exception_type(), "NameError");
    assert_eq!(
        err.to_string(),
        "NameError: name 'undefined_var' is not defined"
    );
}

#[test]
fn test_type_error_creation() {
    let err = PrismError::type_error("unsupported operand type(s) for +: 'int' and 'str'");

    assert_eq!(err.exception_type(), "TypeError");
    assert!(err.to_string().contains("unsupported operand type"));
}

#[test]
fn test_value_error_creation() {
    let err = PrismError::value_error("invalid literal for int()");

    assert_eq!(err.exception_type(), "ValueError");
    assert!(err.to_string().contains("invalid literal"));
}

#[test]
fn test_attribute_error_creation() {
    let err = PrismError::attribute("'int' object has no attribute 'foo'");

    assert_eq!(err.exception_type(), "AttributeError");
}

#[test]
fn test_index_error_creation() {
    let err = PrismError::index("list index out of range");

    assert_eq!(err.exception_type(), "IndexError");
    assert_eq!(err.to_string(), "IndexError: list index out of range");
}

#[test]
fn test_key_error_creation() {
    let err = PrismError::key("missing_key");

    assert_eq!(err.exception_type(), "KeyError");
    assert_eq!(err.to_string(), "KeyError: missing_key");
}

#[test]
fn test_zero_division_error_creation() {
    let err = PrismError::zero_division("division by zero");

    assert_eq!(err.exception_type(), "ZeroDivisionError");
    assert_eq!(err.to_string(), "ZeroDivisionError: division by zero");
}

#[test]
fn test_import_error_creation() {
    let err = PrismError::import("No module named 'nonexistent'");

    assert_eq!(err.exception_type(), "ImportError");
}

#[test]
fn test_assertion_error_creation() {
    let err = PrismError::assertion("expected true");

    assert_eq!(err.exception_type(), "AssertionError");
    assert_eq!(err.to_string(), "AssertionError: expected true");
}

#[test]
fn test_stop_iteration() {
    let err = PrismError::StopIteration;

    assert_eq!(err.exception_type(), "StopIteration");
    assert_eq!(err.to_string(), "StopIteration");
}

#[test]
fn test_recursion_error() {
    let err = PrismError::RecursionError;

    assert_eq!(err.exception_type(), "RecursionError");
    assert!(err.to_string().contains("recursion depth"));
}

#[test]
fn test_overflow_error() {
    let err = PrismError::OverflowError {
        message: "int too large".into(),
    };

    assert_eq!(err.exception_type(), "OverflowError");
}

#[test]
fn test_memory_error() {
    let err = PrismError::MemoryError {
        message: "allocation failed".into(),
    };

    assert_eq!(err.exception_type(), "MemoryError");
}

#[test]
fn test_internal_error_creation() {
    let err = PrismError::internal("VM stack corruption");

    assert_eq!(err.exception_type(), "SystemError");
    assert_eq!(err.to_string(), "InternalError: VM stack corruption");
}

#[test]
fn test_compile_error_with_span() {
    let span = Span::new(100, 120);
    let err = PrismError::compile("cannot assign to literal", Some(span));

    match &err {
        PrismError::CompileError { message, span: s } => {
            assert_eq!(message, "cannot assign to literal");
            assert!(s.is_some());
            let s = s.unwrap();
            assert_eq!(s.start, 100);
            assert_eq!(s.end, 120);
        }
        _ => panic!("Expected CompileError"),
    }
}

#[test]
fn test_compile_error_without_span() {
    let err = PrismError::compile("unknown error", None);

    match &err {
        PrismError::CompileError { span, .. } => {
            assert!(span.is_none());
        }
        _ => panic!("Expected CompileError"),
    }
}

#[test]
fn test_runtime_error_kind_display() {
    assert_eq!(RuntimeErrorKind::Runtime.to_string(), "RuntimeError");
    assert_eq!(RuntimeErrorKind::Exception.to_string(), "Exception");
    assert_eq!(RuntimeErrorKind::SystemExit.to_string(), "SystemExit");
    assert_eq!(
        RuntimeErrorKind::KeyboardInterrupt.to_string(),
        "KeyboardInterrupt"
    );
    assert_eq!(RuntimeErrorKind::GeneratorExit.to_string(), "GeneratorExit");
}

#[test]
fn test_runtime_error_with_kind() {
    let err = PrismError::runtime(RuntimeErrorKind::SystemExit, "exit code 1");

    assert_eq!(err.exception_type(), "SystemExit");
    assert_eq!(err.to_string(), "SystemExit: exit code 1");
}
