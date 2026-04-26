//! CPython-compatible error formatting and exit code handling.
//!
//! Formats `PrismError` and `CompileError` as CPython-style tracebacks
//! on stderr, then returns the appropriate process exit code.

use crate::diagnostics::{self, SourceMap};
use prism_core::PrismError;
use std::process::ExitCode;

// =============================================================================
// Exit Codes (matching CPython)
// =============================================================================

/// Successful execution.
pub const EXIT_SUCCESS: u8 = 0;
/// Runtime error (unhandled exception).
pub const EXIT_ERROR: u8 = 1;
/// Command-line usage error (bad flags, missing args).
pub const EXIT_USAGE_ERROR: u8 = 2;
/// Internal error (should never happen).
pub const EXIT_INTERNAL_ERROR: u8 = 120;

// =============================================================================
// Error Formatting
// =============================================================================

/// Format a `PrismError` to stderr in CPython's traceback format.
///
/// Returns the appropriate exit code.
pub fn format_prism_error(error: &PrismError, source: Option<&str>, filename: &str) -> ExitCode {
    let output = format_error_string(error, source, filename);
    eprint!("{}", output);
    exit_code_for_error(error)
}

/// Format a VM `RuntimeError` to stderr in CPython-style traceback format.
pub fn format_runtime_error(
    error: &prism_vm::RuntimeError,
    source: Option<&str>,
    filename: &str,
) -> ExitCode {
    if runtime_error_type_id(error)
        == Some(prism_vm::stdlib::exceptions::ExceptionTypeId::SystemExit)
    {
        let payload = system_exit_stderr_payload(error);
        if !payload.is_empty() {
            eprintln!("{payload}");
        }
        return exit_code_for_runtime_error(error);
    }

    let output = format_runtime_error_string(error, source, filename);
    eprint!("{}", output);
    exit_code_for_runtime_error(error)
}

/// Format a `PrismError` into a string (for testing).
pub fn format_error_string(error: &PrismError, source: Option<&str>, filename: &str) -> String {
    match error {
        PrismError::LexError { message, span } | PrismError::SyntaxError { message, span } => {
            if let Some(src) = source {
                let sm = SourceMap::new(src, filename);
                format!(
                    "{}\n{}\n",
                    diagnostics::render_traceback_header(),
                    diagnostics::render_source_error(&sm, span, "SyntaxError", message),
                )
            } else {
                format!(
                    "{}\n  File \"{}\"\nSyntaxError: {}\n",
                    diagnostics::render_traceback_header(),
                    filename,
                    message,
                )
            }
        }

        PrismError::CompileError { message, span } => {
            if let (Some(src), Some(span)) = (source, span) {
                let sm = SourceMap::new(src, filename);
                format!(
                    "{}\n{}\n",
                    diagnostics::render_traceback_header(),
                    diagnostics::render_source_error(&sm, span, "SyntaxError", message),
                )
            } else {
                format!(
                    "{}\n  File \"{}\"\nSyntaxError: {}\n",
                    diagnostics::render_traceback_header(),
                    filename,
                    message,
                )
            }
        }

        // Runtime errors arriving as `PrismError` no longer have structured
        // traceback data, so avoid printing a fabricated line number.
        _ => {
            format!(
                "{}\n  File \"{}\"\n{}\n",
                diagnostics::render_traceback_header(),
                filename,
                error,
            )
        }
    }
}

/// Format a VM `RuntimeError` into a string (for testing).
pub fn format_runtime_error_string(
    error: &prism_vm::RuntimeError,
    _source: Option<&str>,
    filename: &str,
) -> String {
    if runtime_error_type_id(error)
        == Some(prism_vm::stdlib::exceptions::ExceptionTypeId::SystemExit)
    {
        let payload = system_exit_stderr_payload(error);
        return if payload.is_empty() {
            String::new()
        } else {
            format!("{payload}\n")
        };
    }

    let mut output = String::new();
    output.push_str(diagnostics::render_traceback_header());
    output.push('\n');

    if error.traceback.is_empty() {
        output.push_str(&format!("  File \"{}\"\n", filename));
    } else {
        for entry in &error.traceback {
            output.push_str(&format!(
                "  File \"{}\", line {}, in {}\n",
                entry.filename, entry.line, entry.func_name
            ));
        }
    }

    output.push_str(&runtime_error_exception_line(error));
    output.push('\n');
    output
}

/// Format a `CompileError` from the compiler crate to stderr.
pub fn format_compile_error(
    error: &prism_compiler::compiler::CompileError,
    source: Option<&str>,
    filename: &str,
) -> ExitCode {
    let output = format_compile_error_string(error, source, filename);
    eprint!("{}", output);
    ExitCode::from(EXIT_ERROR)
}

/// Format a `CompileError` into a string (for testing).
pub fn format_compile_error_string(
    error: &prism_compiler::compiler::CompileError,
    source: Option<&str>,
    filename: &str,
) -> String {
    if let Some(src) = source {
        let sm = SourceMap::new(src, filename);
        // CompileError has line (1-indexed) and column (0-indexed).
        // Convert to byte offset by finding the line start.
        let lines: Vec<&str> = src.lines().collect();
        let line_idx = (error.line as usize).saturating_sub(1);
        let line_start: usize = lines[..line_idx]
            .iter()
            .map(|l| l.len() + 1) // +1 for newline
            .sum();
        let offset = line_start + error.column as usize;
        let span = prism_core::span::Span::new(offset as u32, (offset + 1) as u32);
        format!(
            "{}\n{}\n",
            diagnostics::render_traceback_header(),
            diagnostics::render_source_error(&sm, &span, "SyntaxError", &error.message),
        )
    } else {
        format!(
            "  File \"{}\", line {}\nSyntaxError: {}\n",
            filename, error.line, error.message,
        )
    }
}

/// Format a high-level source compilation error to stderr.
pub fn format_source_compile_error(
    error: &prism_compiler::SourceCompileError,
    source: Option<&str>,
    filename: &str,
) -> ExitCode {
    let output = format_source_compile_error_string(error, source, filename);
    eprint!("{}", output);
    ExitCode::from(EXIT_ERROR)
}

/// Format a high-level source compilation error into a string.
pub fn format_source_compile_error_string(
    error: &prism_compiler::SourceCompileError,
    source: Option<&str>,
    filename: &str,
) -> String {
    if let Some(parse_error) = error.as_parse_error() {
        format_error_string(parse_error, source, filename)
    } else if let Some(compile_error) = error.as_compile_error() {
        format_compile_error_string(compile_error, source, filename)
    } else {
        format!(
            "{}\n  File \"{}\"\nSyntaxError: {}\n",
            diagnostics::render_traceback_header(),
            filename,
            error,
        )
    }
}

/// Map a `PrismError` to its exit code.
#[inline]
fn exit_code_for_error(error: &PrismError) -> ExitCode {
    match error {
        PrismError::RuntimeError {
            kind: prism_core::error::RuntimeErrorKind::SystemExit,
            message,
        } => {
            // SystemExit: message might contain the exit code.
            if let Ok(code) = message.parse::<u8>() {
                ExitCode::from(code)
            } else {
                ExitCode::from(EXIT_ERROR)
            }
        }
        PrismError::InternalError { .. } => ExitCode::from(EXIT_INTERNAL_ERROR),
        _ => ExitCode::from(EXIT_ERROR),
    }
}

#[inline]
fn exit_code_for_runtime_error(error: &prism_vm::RuntimeError) -> ExitCode {
    use prism_vm::RuntimeErrorKind;
    use prism_vm::stdlib::exceptions::ExceptionTypeId;

    if runtime_error_type_id(error) == Some(ExceptionTypeId::SystemExit) {
        return ExitCode::from(system_exit_status_code(error).unwrap_or(EXIT_ERROR));
    }

    match &error.kind {
        RuntimeErrorKind::InternalError { .. }
        | RuntimeErrorKind::InvalidOpcode { .. }
        | RuntimeErrorKind::ControlTransferred => ExitCode::from(EXIT_INTERNAL_ERROR),
        _ => ExitCode::from(EXIT_ERROR),
    }
}

fn system_exit_status_code(error: &prism_vm::RuntimeError) -> Option<u8> {
    if let Some(value) = error.raised_value
        && let Some(exception) = unsafe { prism_vm::builtins::ExceptionValue::from_value(value) }
    {
        return match exception.args().unwrap_or(&[]) {
            [] => Some(EXIT_SUCCESS),
            [status] => system_exit_status_value(*status),
            _ => None,
        };
    }

    let payload = runtime_error_payload(error);
    if payload.is_empty() {
        return Some(EXIT_SUCCESS);
    }
    payload.parse::<u8>().ok()
}

fn system_exit_status_value(value: prism_core::Value) -> Option<u8> {
    if value.is_none() {
        return Some(EXIT_SUCCESS);
    }
    if let Some(boolean) = value.as_bool() {
        return Some(if boolean { EXIT_ERROR } else { EXIT_SUCCESS });
    }
    value
        .as_int()
        .and_then(|integer| u8::try_from(integer).ok())
}

fn system_exit_stderr_payload(error: &prism_vm::RuntimeError) -> String {
    if let Some(value) = error.raised_value
        && let Some(exception) = unsafe { prism_vm::builtins::ExceptionValue::from_value(value) }
    {
        return match exception.args().unwrap_or(&[]) {
            [] => String::new(),
            [status] if system_exit_status_value(*status).is_some() => String::new(),
            _ => exception.display_text(),
        };
    }

    let payload = runtime_error_payload(error);
    if payload.is_empty() || payload.parse::<u8>().is_ok() {
        String::new()
    } else {
        payload
    }
}

fn runtime_error_type_id(
    error: &prism_vm::RuntimeError,
) -> Option<prism_vm::stdlib::exceptions::ExceptionTypeId> {
    use prism_vm::RuntimeErrorKind;
    use prism_vm::stdlib::exceptions::ExceptionTypeId;

    if let Some(value) = error.raised_value
        && let Some(exception) = unsafe { prism_vm::builtins::ExceptionValue::from_value(value) }
    {
        return Some(exception.type_id());
    }

    match &error.kind {
        RuntimeErrorKind::Exception { type_id, .. } => ExceptionTypeId::from_u8(*type_id as u8),
        _ => None,
    }
}

fn runtime_error_payload(error: &prism_vm::RuntimeError) -> String {
    use prism_vm::RuntimeErrorKind;

    if let Some(value) = error.raised_value
        && let Some(exception) = unsafe { prism_vm::builtins::ExceptionValue::from_value(value) }
    {
        return exception.display_text();
    }

    match &error.kind {
        RuntimeErrorKind::Exception { message, .. } => message.to_string(),
        _ => String::new(),
    }
}

fn runtime_error_exception_line(error: &prism_vm::RuntimeError) -> String {
    use prism_vm::RuntimeErrorKind;

    if let Some(value) = error.raised_value
        && let Some(exception) = unsafe { prism_vm::builtins::ExceptionValue::from_value(value) }
    {
        let payload = exception.display_text();
        return if payload.is_empty() {
            exception.type_name().to_string()
        } else {
            format!("{}: {}", exception.type_name(), payload)
        };
    }

    match &error.kind {
        RuntimeErrorKind::Exception { type_id, message } => {
            let type_name = prism_vm::stdlib::exceptions::ExceptionTypeId::from_u8(*type_id as u8)
                .map_or(
                    "Exception",
                    prism_vm::stdlib::exceptions::ExceptionTypeId::name,
                );
            if message.is_empty() {
                type_name.to_string()
            } else {
                format!("{type_name}: {message}")
            }
        }
        _ => error.to_string(),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
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
        let err = prism_vm::RuntimeError::exception(
            prism_vm::stdlib::exceptions::ExceptionTypeId::SystemExit.as_u8() as u16,
            "42",
        );
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
}
