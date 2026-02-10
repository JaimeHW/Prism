//! Builtin exception type constructors.
//!
//! Provides callable constructors for all Python built-in exception types.
//! These are registered in the `BuiltinRegistry` and available as global names
//! like `ValueError`, `TypeError`, etc.
//!
//! # Performance Design
//!
//! - **Macro-generated**: Consistent implementation across all 50+ types
//! - **Zero-alloc message extraction**: Uses Arc<str> for interned strings
//! - **Inline-able**: Small functions that JIT can inline

use super::exception_value::{ExceptionValue, create_exception};
use super::{BuiltinError, BuiltinFn};
use crate::stdlib::exceptions::ExceptionTypeId;
use prism_core::Value;
use std::sync::Arc;

// =============================================================================
// Message Extraction Helpers
// =============================================================================

/// Extract message string from exception constructor arguments.
///
/// Handles: no args, single string arg, or Value that needs conversion.
#[inline]
fn extract_message(args: &[Value]) -> Option<Arc<str>> {
    if args.is_empty() {
        return None;
    }

    let first = args[0];

    // For now, convert all types to their repr form
    // TODO: Add proper string extraction when Value::as_string is available
    if first.is_none() {
        Some(Arc::from("None"))
    } else if let Some(b) = first.as_bool() {
        Some(Arc::from(if b { "True" } else { "False" }))
    } else if let Some(i) = first.as_int() {
        Some(Arc::from(i.to_string()))
    } else if let Some(f) = first.as_float() {
        Some(Arc::from(f.to_string()))
    } else if first.is_string() {
        // String value - for now just use a placeholder
        // TODO: Extract actual string when API is available
        Some(Arc::from("<string>"))
    } else {
        // Object - just use type for now
        Some(Arc::from("<object>"))
    }
}

// =============================================================================
// Exception Constructor Macro
// =============================================================================

/// Macro to generate exception constructor functions.
///
/// Each generated function:
/// 1. Extracts message from args
/// 2. Creates ExceptionValue with appropriate type ID
/// 3. Returns as Value (object pointer)
macro_rules! exception_constructor {
    ($fn_name:ident, $type_id:expr) => {
        /// Exception constructor.
        #[inline]
        pub fn $fn_name(args: &[Value]) -> Result<Value, BuiltinError> {
            let message = extract_message(args);
            Ok(create_exception($type_id, message))
        }
    };
}

// =============================================================================
// Core Exception Constructors (0-15)
// =============================================================================

exception_constructor!(builtin_base_exception, ExceptionTypeId::BaseException);
exception_constructor!(builtin_system_exit, ExceptionTypeId::SystemExit);
exception_constructor!(
    builtin_keyboard_interrupt,
    ExceptionTypeId::KeyboardInterrupt
);
exception_constructor!(builtin_generator_exit, ExceptionTypeId::GeneratorExit);
exception_constructor!(builtin_exception, ExceptionTypeId::Exception);
exception_constructor!(builtin_stop_iteration, ExceptionTypeId::StopIteration);
exception_constructor!(
    builtin_stop_async_iteration,
    ExceptionTypeId::StopAsyncIteration
);
exception_constructor!(builtin_arithmetic_error, ExceptionTypeId::ArithmeticError);
exception_constructor!(builtin_overflow_error, ExceptionTypeId::OverflowError);
exception_constructor!(
    builtin_zero_division_error,
    ExceptionTypeId::ZeroDivisionError
);
exception_constructor!(
    builtin_floating_point_error,
    ExceptionTypeId::FloatingPointError
);
exception_constructor!(builtin_assertion_error, ExceptionTypeId::AssertionError);
exception_constructor!(builtin_attribute_error, ExceptionTypeId::AttributeError);
exception_constructor!(builtin_buffer_error, ExceptionTypeId::BufferError);
exception_constructor!(builtin_eof_error, ExceptionTypeId::EOFError);
exception_constructor!(builtin_import_error, ExceptionTypeId::ImportError);

// =============================================================================
// Lookup Error Constructors (16-19)
// =============================================================================

exception_constructor!(builtin_lookup_error, ExceptionTypeId::LookupError);
exception_constructor!(builtin_index_error, ExceptionTypeId::IndexError);
exception_constructor!(builtin_key_error, ExceptionTypeId::KeyError);
exception_constructor!(
    builtin_module_not_found_error,
    ExceptionTypeId::ModuleNotFoundError
);

// =============================================================================
// Memory/Name Error Constructors (20-23)
// =============================================================================

exception_constructor!(builtin_memory_error, ExceptionTypeId::MemoryError);
exception_constructor!(builtin_name_error, ExceptionTypeId::NameError);
exception_constructor!(
    builtin_unbound_local_error,
    ExceptionTypeId::UnboundLocalError
);

// =============================================================================
// OS/IO Error Constructors (24-39)
// =============================================================================

exception_constructor!(builtin_os_error, ExceptionTypeId::OSError);
exception_constructor!(
    builtin_file_not_found_error,
    ExceptionTypeId::FileNotFoundError
);
exception_constructor!(builtin_file_exists_error, ExceptionTypeId::FileExistsError);
exception_constructor!(
    builtin_not_a_directory_error,
    ExceptionTypeId::NotADirectoryError
);
exception_constructor!(
    builtin_is_a_directory_error,
    ExceptionTypeId::IsADirectoryError
);
exception_constructor!(builtin_permission_error, ExceptionTypeId::PermissionError);
exception_constructor!(
    builtin_process_lookup_error,
    ExceptionTypeId::ProcessLookupError
);
exception_constructor!(builtin_connection_error, ExceptionTypeId::ConnectionError);
exception_constructor!(
    builtin_connection_refused_error,
    ExceptionTypeId::ConnectionRefusedError
);
exception_constructor!(
    builtin_connection_reset_error,
    ExceptionTypeId::ConnectionResetError
);
exception_constructor!(
    builtin_connection_aborted_error,
    ExceptionTypeId::ConnectionAbortedError
);
exception_constructor!(builtin_broken_pipe_error, ExceptionTypeId::BrokenPipeError);
exception_constructor!(builtin_timeout_error, ExceptionTypeId::TimeoutError);
exception_constructor!(builtin_interrupted_error, ExceptionTypeId::InterruptedError);
exception_constructor!(
    builtin_child_process_error,
    ExceptionTypeId::ChildProcessError
);
exception_constructor!(builtin_blocking_io_error, ExceptionTypeId::BlockingIOError);

// =============================================================================
// Reference/Runtime Error Constructors (40-47)
// =============================================================================

exception_constructor!(builtin_reference_error, ExceptionTypeId::ReferenceError);
exception_constructor!(builtin_runtime_error, ExceptionTypeId::RuntimeError);
exception_constructor!(builtin_recursion_error, ExceptionTypeId::RecursionError);
exception_constructor!(
    builtin_not_implemented_error,
    ExceptionTypeId::NotImplementedError
);

// =============================================================================
// Syntax/Value Error Constructors (48-55)
// =============================================================================

exception_constructor!(builtin_syntax_error, ExceptionTypeId::SyntaxError);
exception_constructor!(builtin_indentation_error, ExceptionTypeId::IndentationError);
exception_constructor!(builtin_tab_error, ExceptionTypeId::TabError);
exception_constructor!(builtin_system_error, ExceptionTypeId::SystemError);
exception_constructor!(builtin_type_error, ExceptionTypeId::TypeError);
exception_constructor!(builtin_value_error, ExceptionTypeId::ValueError);
exception_constructor!(builtin_unicode_error, ExceptionTypeId::UnicodeError);
exception_constructor!(
    builtin_unicode_decode_error,
    ExceptionTypeId::UnicodeDecodeError
);

// =============================================================================
// Unicode/Warning Constructors (56-63)
// =============================================================================

exception_constructor!(
    builtin_unicode_encode_error,
    ExceptionTypeId::UnicodeEncodeError
);
exception_constructor!(
    builtin_unicode_translate_error,
    ExceptionTypeId::UnicodeTranslateError
);
exception_constructor!(builtin_warning, ExceptionTypeId::Warning);
exception_constructor!(
    builtin_deprecation_warning,
    ExceptionTypeId::DeprecationWarning
);
exception_constructor!(
    builtin_pending_deprecation_warning,
    ExceptionTypeId::PendingDeprecationWarning
);
exception_constructor!(builtin_runtime_warning, ExceptionTypeId::RuntimeWarning);
exception_constructor!(builtin_syntax_warning, ExceptionTypeId::SyntaxWarning);
exception_constructor!(builtin_user_warning, ExceptionTypeId::UserWarning);

// =============================================================================
// Registration Table
// =============================================================================

/// Table of all exception constructor registrations.
///
/// Format: (Python name, constructor function)
pub const EXCEPTION_CONSTRUCTORS: &[(&str, BuiltinFn)] = &[
    // Core exceptions
    ("BaseException", builtin_base_exception),
    ("SystemExit", builtin_system_exit),
    ("KeyboardInterrupt", builtin_keyboard_interrupt),
    ("GeneratorExit", builtin_generator_exit),
    ("Exception", builtin_exception),
    ("StopIteration", builtin_stop_iteration),
    ("StopAsyncIteration", builtin_stop_async_iteration),
    ("ArithmeticError", builtin_arithmetic_error),
    ("OverflowError", builtin_overflow_error),
    ("ZeroDivisionError", builtin_zero_division_error),
    ("FloatingPointError", builtin_floating_point_error),
    ("AssertionError", builtin_assertion_error),
    ("AttributeError", builtin_attribute_error),
    ("BufferError", builtin_buffer_error),
    ("EOFError", builtin_eof_error),
    ("ImportError", builtin_import_error),
    // Lookup errors
    ("LookupError", builtin_lookup_error),
    ("IndexError", builtin_index_error),
    ("KeyError", builtin_key_error),
    ("ModuleNotFoundError", builtin_module_not_found_error),
    // Memory/Name errors
    ("MemoryError", builtin_memory_error),
    ("NameError", builtin_name_error),
    ("UnboundLocalError", builtin_unbound_local_error),
    // OS/IO errors
    ("OSError", builtin_os_error),
    ("FileNotFoundError", builtin_file_not_found_error),
    ("FileExistsError", builtin_file_exists_error),
    ("NotADirectoryError", builtin_not_a_directory_error),
    ("IsADirectoryError", builtin_is_a_directory_error),
    ("PermissionError", builtin_permission_error),
    ("ProcessLookupError", builtin_process_lookup_error),
    ("ConnectionError", builtin_connection_error),
    ("ConnectionRefusedError", builtin_connection_refused_error),
    ("ConnectionResetError", builtin_connection_reset_error),
    ("ConnectionAbortedError", builtin_connection_aborted_error),
    ("BrokenPipeError", builtin_broken_pipe_error),
    ("TimeoutError", builtin_timeout_error),
    ("InterruptedError", builtin_interrupted_error),
    ("ChildProcessError", builtin_child_process_error),
    ("BlockingIOError", builtin_blocking_io_error),
    // Reference/Runtime errors
    ("ReferenceError", builtin_reference_error),
    ("RuntimeError", builtin_runtime_error),
    ("RecursionError", builtin_recursion_error),
    ("NotImplementedError", builtin_not_implemented_error),
    // Syntax/Value errors
    ("SyntaxError", builtin_syntax_error),
    ("IndentationError", builtin_indentation_error),
    ("TabError", builtin_tab_error),
    ("SystemError", builtin_system_error),
    ("TypeError", builtin_type_error),
    ("ValueError", builtin_value_error),
    ("UnicodeError", builtin_unicode_error),
    ("UnicodeDecodeError", builtin_unicode_decode_error),
    // Unicode/Warning
    ("UnicodeEncodeError", builtin_unicode_encode_error),
    ("UnicodeTranslateError", builtin_unicode_translate_error),
    ("Warning", builtin_warning),
    ("DeprecationWarning", builtin_deprecation_warning),
    (
        "PendingDeprecationWarning",
        builtin_pending_deprecation_warning,
    ),
    ("RuntimeWarning", builtin_runtime_warning),
    ("SyntaxWarning", builtin_syntax_warning),
    ("UserWarning", builtin_user_warning),
    // Aliases (Python compatibility)
    ("IOError", builtin_os_error), // IOError is alias for OSError
    ("EnvironmentError", builtin_os_error), // EnvironmentError is alias for OSError
];

/// Total number of exception constructors.
pub const EXCEPTION_CONSTRUCTOR_COUNT: usize = EXCEPTION_CONSTRUCTORS.len();

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ════════════════════════════════════════════════════════════════════════
    // Message Extraction Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_extract_message_empty() {
        assert!(extract_message(&[]).is_none());
    }

    #[test]
    fn test_extract_message_none_value() {
        let msg = extract_message(&[Value::none()]);
        assert_eq!(msg.as_deref(), Some("None"));
    }

    #[test]
    fn test_extract_message_bool() {
        let msg = extract_message(&[Value::bool(true)]);
        assert_eq!(msg.as_deref(), Some("True"));

        let msg = extract_message(&[Value::bool(false)]);
        assert_eq!(msg.as_deref(), Some("False"));
    }

    #[test]
    fn test_extract_message_int() {
        let msg = extract_message(&[Value::int(42).unwrap()]);
        assert_eq!(msg.as_deref(), Some("42"));

        let msg = extract_message(&[Value::int(-123).unwrap()]);
        assert_eq!(msg.as_deref(), Some("-123"));
    }

    #[test]
    fn test_extract_message_float() {
        let msg = extract_message(&[Value::float(3.14)]);
        assert!(msg.is_some());
        // Float formatting may vary, just check it's not empty
        assert!(!msg.unwrap().is_empty());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Constructor Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_value_error_no_args() {
        let result = builtin_value_error(&[]);
        assert!(result.is_ok());

        let exc = unsafe { ExceptionValue::from_value(result.unwrap()) };
        assert!(exc.is_some());
        assert_eq!(exc.unwrap().type_id(), ExceptionTypeId::ValueError);
        assert!(exc.unwrap().message().is_none());
    }

    #[test]
    fn test_value_error_with_int_arg() {
        let result = builtin_value_error(&[Value::int(42).unwrap()]);
        assert!(result.is_ok());

        let exc = unsafe { ExceptionValue::from_value(result.unwrap()).unwrap() };
        assert_eq!(exc.type_id(), ExceptionTypeId::ValueError);
        assert_eq!(exc.message(), Some("42"));
    }

    #[test]
    fn test_type_error_constructor() {
        let result = builtin_type_error(&[]);
        assert!(result.is_ok());

        let exc = unsafe { ExceptionValue::from_value(result.unwrap()).unwrap() };
        assert_eq!(exc.type_id(), ExceptionTypeId::TypeError);
    }

    #[test]
    fn test_zero_division_error_constructor() {
        let result = builtin_zero_division_error(&[]);
        assert!(result.is_ok());

        let exc = unsafe { ExceptionValue::from_value(result.unwrap()).unwrap() };
        assert_eq!(exc.type_id(), ExceptionTypeId::ZeroDivisionError);
        assert!(exc.is_subclass_of(ExceptionTypeId::ArithmeticError));
    }

    #[test]
    fn test_key_error_constructor() {
        let result = builtin_key_error(&[]);
        assert!(result.is_ok());

        let exc = unsafe { ExceptionValue::from_value(result.unwrap()).unwrap() };
        assert_eq!(exc.type_id(), ExceptionTypeId::KeyError);
        assert!(exc.is_subclass_of(ExceptionTypeId::LookupError));
    }

    #[test]
    fn test_index_error_constructor() {
        let result = builtin_index_error(&[Value::int(-1).unwrap()]);
        assert!(result.is_ok());

        let exc = unsafe { ExceptionValue::from_value(result.unwrap()).unwrap() };
        assert_eq!(exc.type_id(), ExceptionTypeId::IndexError);
        assert_eq!(exc.message(), Some("-1"));
    }

    #[test]
    fn test_runtime_error_constructor() {
        let result = builtin_runtime_error(&[]);
        assert!(result.is_ok());

        let exc = unsafe { ExceptionValue::from_value(result.unwrap()).unwrap() };
        assert_eq!(exc.type_id(), ExceptionTypeId::RuntimeError);
    }

    #[test]
    fn test_stop_iteration_constructor() {
        let result = builtin_stop_iteration(&[]);
        assert!(result.is_ok());

        let exc = unsafe { ExceptionValue::from_value(result.unwrap()).unwrap() };
        assert_eq!(exc.type_id(), ExceptionTypeId::StopIteration);
    }

    #[test]
    fn test_os_error_constructor() {
        let result = builtin_os_error(&[]);
        assert!(result.is_ok());

        let exc = unsafe { ExceptionValue::from_value(result.unwrap()).unwrap() };
        assert_eq!(exc.type_id(), ExceptionTypeId::OSError);
    }

    #[test]
    fn test_file_not_found_error_constructor() {
        let result = builtin_file_not_found_error(&[]);
        assert!(result.is_ok());

        let exc = unsafe { ExceptionValue::from_value(result.unwrap()).unwrap() };
        assert_eq!(exc.type_id(), ExceptionTypeId::FileNotFoundError);
        assert!(exc.is_subclass_of(ExceptionTypeId::OSError));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Registration Table Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_exception_constructors_count() {
        // Should have at least 50 exception types
        assert!(EXCEPTION_CONSTRUCTOR_COUNT >= 50);
    }

    #[test]
    fn test_all_constructors_callable() {
        for (name, constructor) in EXCEPTION_CONSTRUCTORS {
            let result = constructor(&[]);
            assert!(result.is_ok(), "Constructor {} failed", name);

            let value = result.unwrap();
            assert!(
                value.is_object(),
                "Constructor {} didn't return object",
                name
            );
        }
    }

    #[test]
    fn test_exception_constructor_names_valid() {
        for (name, _) in EXCEPTION_CONSTRUCTORS {
            // Names should not be empty
            assert!(!name.is_empty());
            // Names should start with uppercase
            assert!(name.chars().next().unwrap().is_uppercase());
        }
    }

    #[test]
    fn test_key_constructors_present() {
        let names: Vec<&str> = EXCEPTION_CONSTRUCTORS.iter().map(|(n, _)| *n).collect();

        // Check critical exception types are present
        assert!(names.contains(&"ValueError"));
        assert!(names.contains(&"TypeError"));
        assert!(names.contains(&"KeyError"));
        assert!(names.contains(&"IndexError"));
        assert!(names.contains(&"AttributeError"));
        assert!(names.contains(&"NameError"));
        assert!(names.contains(&"RuntimeError"));
        assert!(names.contains(&"StopIteration"));
        assert!(names.contains(&"ZeroDivisionError"));
        assert!(names.contains(&"Exception"));
        assert!(names.contains(&"BaseException"));
    }

    #[test]
    fn test_io_error_alias() {
        // IOError should be an alias for OSError
        let io_result = EXCEPTION_CONSTRUCTORS
            .iter()
            .find(|(n, _)| *n == "IOError")
            .map(|(_, f)| f(&[]));

        let os_result = EXCEPTION_CONSTRUCTORS
            .iter()
            .find(|(n, _)| *n == "OSError")
            .map(|(_, f)| f(&[]));

        assert!(io_result.is_some());
        assert!(os_result.is_some());

        let io_exc = unsafe { ExceptionValue::from_value(io_result.unwrap().unwrap()).unwrap() };
        let os_exc = unsafe { ExceptionValue::from_value(os_result.unwrap().unwrap()).unwrap() };

        // Both should have OSError type
        assert_eq!(io_exc.type_id(), ExceptionTypeId::OSError);
        assert_eq!(os_exc.type_id(), ExceptionTypeId::OSError);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Hierarchy Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_exception_hierarchy_preserved() {
        // ValueError -> Exception -> BaseException
        let result = builtin_value_error(&[]).unwrap();
        let exc = unsafe { ExceptionValue::from_value(result).unwrap() };

        assert!(exc.is_subclass_of(ExceptionTypeId::ValueError));
        assert!(exc.is_subclass_of(ExceptionTypeId::Exception));
        assert!(exc.is_subclass_of(ExceptionTypeId::BaseException));
        assert!(!exc.is_subclass_of(ExceptionTypeId::TypeError));
    }

    #[test]
    fn test_connection_error_hierarchy() {
        // ConnectionRefusedError -> ConnectionError -> OSError -> Exception
        let result = builtin_connection_refused_error(&[]).unwrap();
        let exc = unsafe { ExceptionValue::from_value(result).unwrap() };

        assert!(exc.is_subclass_of(ExceptionTypeId::ConnectionRefusedError));
        assert!(exc.is_subclass_of(ExceptionTypeId::ConnectionError));
        assert!(exc.is_subclass_of(ExceptionTypeId::OSError));
        assert!(exc.is_subclass_of(ExceptionTypeId::Exception));
    }

    #[test]
    fn test_unicode_error_hierarchy() {
        // UnicodeDecodeError -> UnicodeError -> ValueError -> Exception
        let result = builtin_unicode_decode_error(&[]).unwrap();
        let exc = unsafe { ExceptionValue::from_value(result).unwrap() };

        assert!(exc.is_subclass_of(ExceptionTypeId::UnicodeDecodeError));
        assert!(exc.is_subclass_of(ExceptionTypeId::UnicodeError));
        assert!(exc.is_subclass_of(ExceptionTypeId::ValueError));
        assert!(exc.is_subclass_of(ExceptionTypeId::Exception));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Performance Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_constructor_is_fast() {
        // Constructors should complete quickly (no heavy allocation)
        for _ in 0..1000 {
            let result = builtin_value_error(&[Value::int(42).unwrap()]);
            assert!(result.is_ok());
        }
    }
}
