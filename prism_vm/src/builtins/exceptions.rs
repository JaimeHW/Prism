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
exception_constructor!(
    builtin_base_exception_group,
    ExceptionTypeId::BaseExceptionGroup
);
exception_constructor!(builtin_exception_group, ExceptionTypeId::ExceptionGroup);

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
    ("BaseExceptionGroup", builtin_base_exception_group),
    ("ExceptionGroup", builtin_exception_group),
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
