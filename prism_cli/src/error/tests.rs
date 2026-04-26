use super::*;
use prism_core::error::RuntimeErrorKind;
use prism_core::span::Span;
use prism_vm::TracebackEntry;

// =========================================================================
// Exit Code Tests
// =========================================================================

#[test]
fn test_exit_code_name_error() {
    let err = PrismError::name("x");
    let code = exit_code_for_error(&err);
    assert_eq!(code, ExitCode::from(EXIT_ERROR));
}

#[test]
fn test_exit_code_system_exit_numeric() {
    let err = PrismError::runtime(RuntimeErrorKind::SystemExit, "42");
    let code = exit_code_for_error(&err);
    assert_eq!(code, ExitCode::from(42));
}

#[test]
fn test_exit_code_system_exit_zero() {
    let err = PrismError::runtime(RuntimeErrorKind::SystemExit, "0");
    let code = exit_code_for_error(&err);
    assert_eq!(code, ExitCode::from(0));
}

#[test]
fn test_exit_code_system_exit_non_numeric() {
    let err = PrismError::runtime(RuntimeErrorKind::SystemExit, "goodbye");
    let code = exit_code_for_error(&err);
    assert_eq!(code, ExitCode::from(EXIT_ERROR));
}

#[test]
fn test_exit_code_internal_error() {
    let err = PrismError::internal("corruption");
    let code = exit_code_for_error(&err);
    assert_eq!(code, ExitCode::from(EXIT_INTERNAL_ERROR));
}

// =========================================================================
// Error Formatting Tests
// =========================================================================

#[test]
fn test_format_syntax_error_with_source() {
    let err = PrismError::syntax("invalid syntax", Span::new(4, 5));
    let output = format_error_string(&err, Some("x = ?"), "test.py");
    assert!(output.contains("Traceback"));
    assert!(output.contains("File \"test.py\", line 1"));
    assert!(output.contains("x = ?"));
    assert!(output.contains("SyntaxError: invalid syntax"));
}

#[test]
fn test_format_syntax_error_without_source() {
    let err = PrismError::syntax("unexpected EOF", Span::new(0, 1));
    let output = format_error_string(&err, None, "test.py");
    assert!(output.contains("File \"test.py\""));
    assert!(output.contains("SyntaxError: unexpected EOF"));
}

#[test]
fn test_format_lex_error() {
    let err = PrismError::lex("unexpected character", Span::new(2, 3));
    let output = format_error_string(&err, Some("x$y"), "test.py");
    assert!(output.contains("SyntaxError: unexpected character"));
}

#[test]
fn test_format_compile_error_with_span() {
    let err = PrismError::compile("cannot assign to literal", Some(Span::new(0, 1)));
    let output = format_error_string(&err, Some("1 = x"), "test.py");
    assert!(output.contains("SyntaxError: cannot assign to literal"));
}

#[test]
fn test_format_compile_error_without_span() {
    let err = PrismError::compile("internal error", None);
    let output = format_error_string(&err, Some("x = 1"), "test.py");
    assert!(output.contains("SyntaxError: internal error"));
}

#[test]
fn test_format_runtime_error() {
    let err = PrismError::name("undefined_var");
    let output = format_error_string(&err, Some("print(undefined_var)"), "test.py");
    assert!(output.contains("Traceback"));
    assert!(output.contains("NameError"));
    assert!(output.contains("undefined_var"));
    assert!(!output.contains("line 1, in <module>"));
}

#[test]
fn test_format_type_error() {
    let err = PrismError::type_error("unsupported operand");
    let output = format_error_string(&err, None, "test.py");
    assert!(output.contains("TypeError: unsupported operand"));
}

#[test]
fn test_format_zero_division() {
    let err = PrismError::zero_division("division by zero");
    let output = format_error_string(&err, None, "test.py");
    assert!(output.contains("ZeroDivisionError: division by zero"));
}

#[test]
fn test_format_runtime_error_string_uses_recorded_traceback_lines() {
    let mut err = prism_vm::RuntimeError::name_error("missing");
    err.add_traceback(TracebackEntry {
        func_name: "wrapper".into(),
        filename: "bootstrap.py".into(),
        line: 27,
    });
    err.add_traceback(TracebackEntry {
        func_name: "<module>".into(),
        filename: "helpers.py".into(),
        line: 91,
    });

    let output = format_runtime_error_string(&err, None, "bootstrap.py");
    assert!(output.contains("File \"bootstrap.py\", line 27, in wrapper"));
    assert!(output.contains("File \"helpers.py\", line 91, in <module>"));
    assert!(output.contains("NameError: name 'missing' is not defined"));
}

#[test]
fn test_format_runtime_error_string_without_traceback_omits_fake_line_number() {
    let err = prism_vm::RuntimeError::type_error("bad operand");
    let output = format_runtime_error_string(&err, None, "test.py");
    assert!(output.contains("File \"test.py\""));
    assert!(!output.contains("line 1"));
    assert!(output.contains("TypeError: bad operand"));
}

#[test]
fn test_exit_code_for_runtime_system_exit_numeric() {
    let err = prism_vm::RuntimeError::exception(ExceptionTypeId::SystemExit.as_u8() as u16, "42");
    let code = exit_code_for_runtime_error(&err);
    assert_eq!(code, ExitCode::from(42));
}

// =========================================================================
// CompileError Formatting Tests
// =========================================================================

#[test]
fn test_format_compiler_error_with_source() {
    let err = prism_compiler::compiler::CompileError {
        message: "undeclared variable".to_string(),
        line: 1,
        column: 0,
    };
    let output = format_compile_error_string(&err, Some("x = 1"), "test.py");
    assert!(output.contains("SyntaxError: undeclared variable"));
}

#[test]
fn test_format_compiler_error_without_source() {
    let err = prism_compiler::compiler::CompileError {
        message: "unknown error".to_string(),
        line: 5,
        column: 0,
    };
    let output = format_compile_error_string(&err, None, "test.py");
    assert!(output.contains("File \"test.py\", line 5"));
    assert!(output.contains("SyntaxError: unknown error"));
}

#[test]
fn test_format_source_compile_error_string_for_parse_error() {
    let err = prism_compiler::SourceCompileError::Parse(PrismError::syntax(
        "unexpected EOF",
        Span::new(0, 1),
    ));
    let output = format_source_compile_error_string(&err, Some("def"), "test.py");
    assert!(output.contains("SyntaxError: unexpected EOF"));
}

#[test]
fn test_format_source_compile_error_string_for_compile_error() {
    let err = prism_compiler::SourceCompileError::Compile(prism_compiler::CompileError {
        message: "continue outside loop".to_string(),
        line: 1,
        column: 0,
    });
    let output = format_source_compile_error_string(&err, Some("continue"), "test.py");
    assert!(output.contains("SyntaxError: continue outside loop"));
}

// =========================================================================
// Constant Tests
// =========================================================================

#[test]
fn test_exit_code_constants() {
    assert_eq!(EXIT_SUCCESS, 0);
    assert_eq!(EXIT_ERROR, 1);
    assert_eq!(EXIT_USAGE_ERROR, 2);
    assert_eq!(EXIT_INTERNAL_ERROR, 120);
}
