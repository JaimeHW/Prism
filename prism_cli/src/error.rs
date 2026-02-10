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

        // Runtime errors: simple display (traceback would need frame info).
        _ => {
            format!(
                "{}\n  File \"{}\", line 1, in <module>\n{}\n",
                diagnostics::render_traceback_header(),
                filename,
                error,
            )
        }
    }
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::error::RuntimeErrorKind;
    use prism_core::span::Span;

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
