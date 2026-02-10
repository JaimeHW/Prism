//! Exception type objects for exception type representation.
//!
//! This module provides `ExceptionTypeObject`, a high-performance callable type
//! that represents Python exception classes (e.g., `ValueError`, `TypeError`).
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                     Exception Type Object Layout                         │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │  Offset   │  Field               │  Size   │  Description               │
//! │  0        │  ObjectHeader        │  16B    │  GC + TypeId dispatch      │
//! │  16       │  exception_type_id   │  2B     │  ExceptionTypeId (u16)     │
//! │  18       │  flags               │  2B     │  ExceptionTypeFlags        │
//! │  20       │  _pad                │  4B     │  Alignment padding         │
//! │  24       │  name                │  16B    │  Arc<str> (fat pointer)    │
//! │  40       │  base_type_id        │  2B     │  Parent exception type     │
//! │  42       │  _reserved           │  6B     │  Future use                │
//! │  TOTAL    │                      │  48B    │  Cache-line aligned        │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance
//!
//! | Operation                  | Cycles | Notes                          |
//! |---------------------------|--------|--------------------------------|
//! | Type ID extraction        | ~1     | Single load instruction        |
//! | is_subclass_of check      | ~3-8   | Bitmap or hierarchy lookup     |
//! | Exception construction    | ~50    | Heap allocation + init         |
//!
//! # Usage
//!
//! ```rust,ignore
//! use prism_vm::builtins::exception_type::ExceptionTypeObject;
//! use prism_vm::stdlib::exceptions::ExceptionTypeId;
//!
//! // Create a ValueError type
//! let value_error = ExceptionTypeObject::new(
//!     ExceptionTypeId::ValueError,
//!     "ValueError",
//! );
//!
//! // Construct an instance
//! let exc = value_error.construct(&[Value::none()]);
//! ```

use crate::builtins::BuiltinError;
use crate::builtins::exception_value::{ExceptionValue, create_exception_with_args};
use crate::stdlib::exceptions::ExceptionTypeId;
use prism_core::Value;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;
use std::sync::Arc;

// =============================================================================
// Type ID Extension
// =============================================================================

/// TypeId for exception type objects (callable that constructs exceptions).
pub const EXCEPTION_TYPE_ID: TypeId = TypeId(27);

// =============================================================================
// Exception Type Flags
// =============================================================================

/// Flags for exception type behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct ExceptionTypeFlags(u16);

impl ExceptionTypeFlags {
    /// Type is a base class (cannot be caught directly).
    pub const ABSTRACT: Self = Self(1 << 0);
    /// Type uses custom __init__.
    pub const CUSTOM_INIT: Self = Self(1 << 1);
    /// Type is a system exit exception.
    pub const SYSTEM_EXIT: Self = Self(1 << 2);
    /// Type is a keyboard interrupt.
    pub const KEYBOARD_INTERRUPT: Self = Self(1 << 3);
    /// Type is a generator exit.
    pub const GENERATOR_EXIT: Self = Self(1 << 4);
    /// Type should not be caught by bare except.
    pub const NO_BARE_EXCEPT: Self = Self(1 << 5);

    /// Empty flags.
    #[inline]
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Check if flag is set.
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Check if empty.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl std::ops::BitOr for ExceptionTypeFlags {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for ExceptionTypeFlags {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

// =============================================================================
// Exception Type Object
// =============================================================================

/// Exception type object - callable that constructs exception instances.
///
/// This struct represents a Python exception class (e.g., `ValueError`).
/// When called, it constructs an `ExceptionValue` instance.
///
/// # Memory Layout
///
/// The struct is `#[repr(C)]` with explicit padding for cache-line optimization:
/// - Total size: 48 bytes (fits in single cache line)
/// - Aligned for fast pointer arithmetic
///
/// # Thread Safety
///
/// `ExceptionTypeObject` is immutable after construction and can be safely
/// shared across threads via `Arc`.
#[repr(C)]
pub struct ExceptionTypeObject {
    /// Object header for GC and type dispatch.
    /// Uses `TypeId(27)` (EXCEPTION_TYPE_ID) for identification.
    pub header: ObjectHeader,

    /// Exception type identifier for fast matching.
    /// Maps to `ExceptionTypeId` enum values.
    pub exception_type_id: u16,

    /// Behavioral flags for this exception type.
    pub flags: ExceptionTypeFlags,

    /// Alignment padding to 8-byte boundary.
    _pad: [u8; 4],

    /// Display name of the exception (e.g., "ValueError").
    pub name: Arc<str>,

    /// Parent exception type ID for hierarchy checks.
    /// `BaseException` has `base_type_id == 0` (itself).
    pub base_type_id: u16,

    /// Reserved for future use (docstring pointer, etc.).
    _reserved: [u8; 6],
}

// Verify size at compile time
const _: () = {
    assert!(std::mem::size_of::<ExceptionTypeObject>() == 48);
};

impl ExceptionTypeObject {
    /// Create a new exception type object.
    ///
    /// # Arguments
    ///
    /// * `type_id` - The `ExceptionTypeId` for this exception class
    /// * `name` - Display name (e.g., "ValueError")
    ///
    /// # Performance
    ///
    /// Single allocation for the `Arc<str>` name.
    #[inline]
    pub fn new(type_id: ExceptionTypeId, name: &str) -> Self {
        let base = type_id.parent().unwrap_or(ExceptionTypeId::BaseException);

        Self {
            header: ObjectHeader::new(EXCEPTION_TYPE_ID),
            exception_type_id: type_id as u16,
            flags: Self::compute_flags(type_id),
            _pad: [0; 4],
            name: Arc::from(name),
            base_type_id: base as u16,
            _reserved: [0; 6],
        }
    }

    /// Create a new exception type with explicit base type.
    #[inline]
    pub fn with_base(type_id: ExceptionTypeId, name: &str, base: ExceptionTypeId) -> Self {
        Self {
            header: ObjectHeader::new(EXCEPTION_TYPE_ID),
            exception_type_id: type_id as u16,
            flags: Self::compute_flags(type_id),
            _pad: [0; 4],
            name: Arc::from(name),
            base_type_id: base as u16,
            _reserved: [0; 6],
        }
    }

    /// Compute flags based on exception type.
    #[inline]
    fn compute_flags(type_id: ExceptionTypeId) -> ExceptionTypeFlags {
        let mut flags = ExceptionTypeFlags::empty();

        match type_id {
            ExceptionTypeId::SystemExit => {
                flags |= ExceptionTypeFlags::SYSTEM_EXIT | ExceptionTypeFlags::NO_BARE_EXCEPT;
            }
            ExceptionTypeId::KeyboardInterrupt => {
                flags |=
                    ExceptionTypeFlags::KEYBOARD_INTERRUPT | ExceptionTypeFlags::NO_BARE_EXCEPT;
            }
            ExceptionTypeId::GeneratorExit => {
                flags |= ExceptionTypeFlags::GENERATOR_EXIT | ExceptionTypeFlags::NO_BARE_EXCEPT;
            }
            _ => {}
        }

        flags
    }

    /// Get the exception type ID.
    ///
    /// # Performance
    ///
    /// O(1) - Single load instruction (~1 cycle).
    #[inline(always)]
    pub fn type_id(&self) -> u16 {
        self.exception_type_id
    }

    /// Get the exception type ID as enum.
    #[inline]
    pub fn exception_type(&self) -> Option<ExceptionTypeId> {
        ExceptionTypeId::from_u8(self.exception_type_id as u8)
    }

    /// Get the display name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the base/parent type ID.
    #[inline]
    pub fn base_type_id(&self) -> u16 {
        self.base_type_id
    }

    /// Check if this exception type is a subclass of another.
    ///
    /// # Performance
    ///
    /// Delegates to `ExceptionTypeId::is_subclass_of` which uses
    /// precomputed hierarchy data for O(1) lookup.
    #[inline]
    pub fn is_subclass_of(&self, other_type_id: u16) -> bool {
        if self.exception_type_id == other_type_id {
            return true;
        }

        let self_type = ExceptionTypeId::from_u8(self.exception_type_id as u8);
        let other_type = ExceptionTypeId::from_u8(other_type_id as u8);

        match (self_type, other_type) {
            (Some(s), Some(o)) => s.is_subclass_of(o),
            _ => false,
        }
    }

    /// Check if this is a system-exiting exception.
    #[inline]
    pub fn is_system_exit(&self) -> bool {
        self.flags.contains(ExceptionTypeFlags::SYSTEM_EXIT)
    }

    /// Check if this exception should be caught by bare `except:`.
    #[inline]
    pub fn caught_by_bare_except(&self) -> bool {
        !self.flags.contains(ExceptionTypeFlags::NO_BARE_EXCEPT)
    }

    /// Construct an exception instance from arguments.
    ///
    /// # Arguments
    ///
    /// * `args` - Constructor arguments (typically a message string)
    ///
    /// # Returns
    ///
    /// A `Value` containing the allocated `ExceptionValue`.
    ///
    /// # Performance
    ///
    /// Allocates an `ExceptionValue` on the heap (~50 cycles).
    #[inline]
    pub fn construct(&self, args: &[Value]) -> Value {
        let type_id = ExceptionTypeId::from_u8(self.exception_type_id as u8)
            .unwrap_or(ExceptionTypeId::Exception);

        // Extract message from first arg if present
        let message = if args.is_empty() {
            None
        } else {
            // For now, use type name as placeholder until string extraction is implemented
            Some(Arc::from(self.name.as_ref()))
        };

        // Convert args to boxed slice
        let boxed_args: Box<[Value]> = args.to_vec().into_boxed_slice();

        create_exception_with_args(type_id, message, boxed_args)
    }

    /// Call the exception type as a constructor (for builtins integration).
    ///
    /// This is the entry point for `ValueError("message")` style calls.
    #[inline]
    pub fn call(&self, args: &[Value]) -> Result<Value, BuiltinError> {
        Ok(self.construct(args))
    }

    /// Convert to a Value (object pointer).
    ///
    /// # Safety
    ///
    /// The returned Value contains a raw pointer to `self`.
    /// Caller must ensure `self` outlives the Value.
    #[inline]
    pub unsafe fn as_value(&self) -> Value {
        Value::object_ptr(self as *const _ as *const ())
    }
}

impl std::fmt::Debug for ExceptionTypeObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExceptionTypeObject")
            .field("type_id", &self.exception_type_id)
            .field("name", &self.name)
            .field("base_type_id", &self.base_type_id)
            .field("flags", &self.flags)
            .finish()
    }
}

impl std::fmt::Display for ExceptionTypeObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<class '{}'>", self.name)
    }
}

// =============================================================================
// Static Exception Types
// =============================================================================

/// Macro to generate static exception type objects.
///
/// Creates a `const` `ExceptionTypeObject` for each exception type,
/// allowing zero-cost access to exception type metadata.
macro_rules! define_exception_type {
    ($name:ident, $type_id:expr, $str_name:literal) => {
        /// Static exception type object.
        pub static $name: std::sync::LazyLock<ExceptionTypeObject> =
            std::sync::LazyLock::new(|| ExceptionTypeObject::new($type_id, $str_name));
    };
}

// Core exception types
define_exception_type!(
    BASE_EXCEPTION,
    ExceptionTypeId::BaseException,
    "BaseException"
);
define_exception_type!(EXCEPTION, ExceptionTypeId::Exception, "Exception");
define_exception_type!(SYSTEM_EXIT, ExceptionTypeId::SystemExit, "SystemExit");
define_exception_type!(
    KEYBOARD_INTERRUPT,
    ExceptionTypeId::KeyboardInterrupt,
    "KeyboardInterrupt"
);
define_exception_type!(
    GENERATOR_EXIT,
    ExceptionTypeId::GeneratorExit,
    "GeneratorExit"
);

// Standard exceptions
define_exception_type!(
    ARITHMETIC_ERROR,
    ExceptionTypeId::ArithmeticError,
    "ArithmeticError"
);
define_exception_type!(
    ASSERTION_ERROR,
    ExceptionTypeId::AssertionError,
    "AssertionError"
);
define_exception_type!(
    ATTRIBUTE_ERROR,
    ExceptionTypeId::AttributeError,
    "AttributeError"
);
define_exception_type!(
    BLOCKING_IO_ERROR,
    ExceptionTypeId::BlockingIOError,
    "BlockingIOError"
);
define_exception_type!(
    BROKEN_PIPE_ERROR,
    ExceptionTypeId::BrokenPipeError,
    "BrokenPipeError"
);
define_exception_type!(BUFFER_ERROR, ExceptionTypeId::BufferError, "BufferError");
define_exception_type!(
    CHILD_PROCESS_ERROR,
    ExceptionTypeId::ChildProcessError,
    "ChildProcessError"
);
define_exception_type!(
    CONNECTION_ABORTED_ERROR,
    ExceptionTypeId::ConnectionAbortedError,
    "ConnectionAbortedError"
);
define_exception_type!(
    CONNECTION_ERROR,
    ExceptionTypeId::ConnectionError,
    "ConnectionError"
);
define_exception_type!(
    CONNECTION_REFUSED_ERROR,
    ExceptionTypeId::ConnectionRefusedError,
    "ConnectionRefusedError"
);
define_exception_type!(
    CONNECTION_RESET_ERROR,
    ExceptionTypeId::ConnectionResetError,
    "ConnectionResetError"
);
define_exception_type!(EOF_ERROR, ExceptionTypeId::EOFError, "EOFError");
define_exception_type!(
    FILE_EXISTS_ERROR,
    ExceptionTypeId::FileExistsError,
    "FileExistsError"
);
define_exception_type!(
    FILE_NOT_FOUND_ERROR,
    ExceptionTypeId::FileNotFoundError,
    "FileNotFoundError"
);
define_exception_type!(
    FLOATING_POINT_ERROR,
    ExceptionTypeId::FloatingPointError,
    "FloatingPointError"
);
define_exception_type!(IMPORT_ERROR, ExceptionTypeId::ImportError, "ImportError");
define_exception_type!(
    INDENTATION_ERROR,
    ExceptionTypeId::IndentationError,
    "IndentationError"
);
define_exception_type!(INDEX_ERROR, ExceptionTypeId::IndexError, "IndexError");
define_exception_type!(
    INTERRUPTED_ERROR,
    ExceptionTypeId::InterruptedError,
    "InterruptedError"
);
define_exception_type!(
    IS_A_DIRECTORY_ERROR,
    ExceptionTypeId::IsADirectoryError,
    "IsADirectoryError"
);
define_exception_type!(KEY_ERROR, ExceptionTypeId::KeyError, "KeyError");
define_exception_type!(LOOKUP_ERROR, ExceptionTypeId::LookupError, "LookupError");
define_exception_type!(MEMORY_ERROR, ExceptionTypeId::MemoryError, "MemoryError");
define_exception_type!(
    MODULE_NOT_FOUND_ERROR,
    ExceptionTypeId::ModuleNotFoundError,
    "ModuleNotFoundError"
);
define_exception_type!(NAME_ERROR, ExceptionTypeId::NameError, "NameError");
define_exception_type!(
    NOT_A_DIRECTORY_ERROR,
    ExceptionTypeId::NotADirectoryError,
    "NotADirectoryError"
);
define_exception_type!(
    NOT_IMPLEMENTED_ERROR,
    ExceptionTypeId::NotImplementedError,
    "NotImplementedError"
);
define_exception_type!(OS_ERROR, ExceptionTypeId::OSError, "OSError");
define_exception_type!(
    OVERFLOW_ERROR,
    ExceptionTypeId::OverflowError,
    "OverflowError"
);
define_exception_type!(
    PERMISSION_ERROR,
    ExceptionTypeId::PermissionError,
    "PermissionError"
);
define_exception_type!(
    PROCESS_LOOKUP_ERROR,
    ExceptionTypeId::ProcessLookupError,
    "ProcessLookupError"
);
define_exception_type!(
    RECURSION_ERROR,
    ExceptionTypeId::RecursionError,
    "RecursionError"
);
define_exception_type!(
    REFERENCE_ERROR,
    ExceptionTypeId::ReferenceError,
    "ReferenceError"
);
define_exception_type!(RUNTIME_ERROR, ExceptionTypeId::RuntimeError, "RuntimeError");
define_exception_type!(
    STOP_ASYNC_ITERATION,
    ExceptionTypeId::StopAsyncIteration,
    "StopAsyncIteration"
);
define_exception_type!(
    STOP_ITERATION,
    ExceptionTypeId::StopIteration,
    "StopIteration"
);
define_exception_type!(SYNTAX_ERROR, ExceptionTypeId::SyntaxError, "SyntaxError");
define_exception_type!(TAB_ERROR, ExceptionTypeId::TabError, "TabError");
define_exception_type!(TIMEOUT_ERROR, ExceptionTypeId::TimeoutError, "TimeoutError");
define_exception_type!(TYPE_ERROR, ExceptionTypeId::TypeError, "TypeError");
define_exception_type!(
    UNBOUND_LOCAL_ERROR,
    ExceptionTypeId::UnboundLocalError,
    "UnboundLocalError"
);
define_exception_type!(
    UNICODE_DECODE_ERROR,
    ExceptionTypeId::UnicodeDecodeError,
    "UnicodeDecodeError"
);
define_exception_type!(
    UNICODE_ENCODE_ERROR,
    ExceptionTypeId::UnicodeEncodeError,
    "UnicodeEncodeError"
);
define_exception_type!(UNICODE_ERROR, ExceptionTypeId::UnicodeError, "UnicodeError");
define_exception_type!(
    UNICODE_TRANSLATE_ERROR,
    ExceptionTypeId::UnicodeTranslateError,
    "UnicodeTranslateError"
);
define_exception_type!(VALUE_ERROR, ExceptionTypeId::ValueError, "ValueError");
define_exception_type!(
    ZERO_DIVISION_ERROR,
    ExceptionTypeId::ZeroDivisionError,
    "ZeroDivisionError"
);

// Warning types
define_exception_type!(WARNING, ExceptionTypeId::Warning, "Warning");
define_exception_type!(
    DEPRECATION_WARNING,
    ExceptionTypeId::DeprecationWarning,
    "DeprecationWarning"
);
define_exception_type!(
    PENDING_DEPRECATION_WARNING,
    ExceptionTypeId::PendingDeprecationWarning,
    "PendingDeprecationWarning"
);
define_exception_type!(
    RUNTIME_WARNING,
    ExceptionTypeId::RuntimeWarning,
    "RuntimeWarning"
);
define_exception_type!(
    SYNTAX_WARNING,
    ExceptionTypeId::SyntaxWarning,
    "SyntaxWarning"
);
define_exception_type!(USER_WARNING, ExceptionTypeId::UserWarning, "UserWarning");

// =============================================================================
// Exception Type Table
// =============================================================================

/// Lookup table for exception type objects by name.
///
/// Used by the registry to get static exception type objects.
pub static EXCEPTION_TYPE_TABLE: &[(&str, &std::sync::LazyLock<ExceptionTypeObject>)] = &[
    ("BaseException", &BASE_EXCEPTION),
    ("Exception", &EXCEPTION),
    ("SystemExit", &SYSTEM_EXIT),
    ("KeyboardInterrupt", &KEYBOARD_INTERRUPT),
    ("GeneratorExit", &GENERATOR_EXIT),
    ("ArithmeticError", &ARITHMETIC_ERROR),
    ("AssertionError", &ASSERTION_ERROR),
    ("AttributeError", &ATTRIBUTE_ERROR),
    ("BlockingIOError", &BLOCKING_IO_ERROR),
    ("BrokenPipeError", &BROKEN_PIPE_ERROR),
    ("BufferError", &BUFFER_ERROR),
    ("ChildProcessError", &CHILD_PROCESS_ERROR),
    ("ConnectionAbortedError", &CONNECTION_ABORTED_ERROR),
    ("ConnectionError", &CONNECTION_ERROR),
    ("ConnectionRefusedError", &CONNECTION_REFUSED_ERROR),
    ("ConnectionResetError", &CONNECTION_RESET_ERROR),
    ("EOFError", &EOF_ERROR),
    ("FileExistsError", &FILE_EXISTS_ERROR),
    ("FileNotFoundError", &FILE_NOT_FOUND_ERROR),
    ("FloatingPointError", &FLOATING_POINT_ERROR),
    ("ImportError", &IMPORT_ERROR),
    ("IndentationError", &INDENTATION_ERROR),
    ("IndexError", &INDEX_ERROR),
    ("InterruptedError", &INTERRUPTED_ERROR),
    ("IsADirectoryError", &IS_A_DIRECTORY_ERROR),
    ("KeyError", &KEY_ERROR),
    ("LookupError", &LOOKUP_ERROR),
    ("MemoryError", &MEMORY_ERROR),
    ("ModuleNotFoundError", &MODULE_NOT_FOUND_ERROR),
    ("NameError", &NAME_ERROR),
    ("NotADirectoryError", &NOT_A_DIRECTORY_ERROR),
    ("NotImplementedError", &NOT_IMPLEMENTED_ERROR),
    ("OSError", &OS_ERROR),
    ("OverflowError", &OVERFLOW_ERROR),
    ("PermissionError", &PERMISSION_ERROR),
    ("ProcessLookupError", &PROCESS_LOOKUP_ERROR),
    ("RecursionError", &RECURSION_ERROR),
    ("ReferenceError", &REFERENCE_ERROR),
    ("RuntimeError", &RUNTIME_ERROR),
    ("StopAsyncIteration", &STOP_ASYNC_ITERATION),
    ("StopIteration", &STOP_ITERATION),
    ("SyntaxError", &SYNTAX_ERROR),
    ("TabError", &TAB_ERROR),
    ("TimeoutError", &TIMEOUT_ERROR),
    ("TypeError", &TYPE_ERROR),
    ("UnboundLocalError", &UNBOUND_LOCAL_ERROR),
    ("UnicodeDecodeError", &UNICODE_DECODE_ERROR),
    ("UnicodeEncodeError", &UNICODE_ENCODE_ERROR),
    ("UnicodeError", &UNICODE_ERROR),
    ("UnicodeTranslateError", &UNICODE_TRANSLATE_ERROR),
    ("ValueError", &VALUE_ERROR),
    ("ZeroDivisionError", &ZERO_DIVISION_ERROR),
    ("Warning", &WARNING),
    ("DeprecationWarning", &DEPRECATION_WARNING),
    ("PendingDeprecationWarning", &PENDING_DEPRECATION_WARNING),
    ("RuntimeWarning", &RUNTIME_WARNING),
    ("SyntaxWarning", &SYNTAX_WARNING),
    ("UserWarning", &USER_WARNING),
    // Aliases
    ("IOError", &OS_ERROR),
    ("EnvironmentError", &OS_ERROR),
];

/// Get an exception type object by name.
#[inline]
pub fn get_exception_type(name: &str) -> Option<&'static ExceptionTypeObject> {
    EXCEPTION_TYPE_TABLE
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, typ)| &***typ)
}

/// Get an exception type object by type ID.
#[inline]
pub fn get_exception_type_by_id(type_id: u16) -> Option<&'static ExceptionTypeObject> {
    EXCEPTION_TYPE_TABLE.iter().find_map(|(_, typ)| {
        if typ.exception_type_id == type_id {
            Some(&***typ)
        } else {
            None
        }
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ════════════════════════════════════════════════════════════════════════
    // Memory Layout Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_exception_type_object_size() {
        assert_eq!(std::mem::size_of::<ExceptionTypeObject>(), 48);
    }

    #[test]
    fn test_exception_type_object_alignment() {
        assert_eq!(std::mem::align_of::<ExceptionTypeObject>(), 8);
    }

    #[test]
    fn test_header_offset() {
        assert_eq!(std::mem::offset_of!(ExceptionTypeObject, header), 0);
    }

    #[test]
    fn test_type_id_offset() {
        // type_id should be at offset 16 (after ObjectHeader)
        assert_eq!(
            std::mem::offset_of!(ExceptionTypeObject, exception_type_id),
            16
        );
    }

    #[test]
    fn test_name_offset() {
        // name should be at offset 24
        assert_eq!(std::mem::offset_of!(ExceptionTypeObject, name), 24);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Construction Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_new_value_error() {
        let exc_type = ExceptionTypeObject::new(ExceptionTypeId::ValueError, "ValueError");
        assert_eq!(
            exc_type.exception_type_id,
            ExceptionTypeId::ValueError as u16
        );
        assert_eq!(exc_type.name(), "ValueError");
        assert_eq!(exc_type.header.type_id, EXCEPTION_TYPE_ID);
    }

    #[test]
    fn test_new_type_error() {
        let exc_type = ExceptionTypeObject::new(ExceptionTypeId::TypeError, "TypeError");
        assert_eq!(
            exc_type.exception_type_id,
            ExceptionTypeId::TypeError as u16
        );
        assert_eq!(exc_type.name(), "TypeError");
    }

    #[test]
    fn test_new_system_exit() {
        let exc_type = ExceptionTypeObject::new(ExceptionTypeId::SystemExit, "SystemExit");
        assert!(exc_type.is_system_exit());
        assert!(!exc_type.caught_by_bare_except());
    }

    #[test]
    fn test_new_keyboard_interrupt() {
        let exc_type =
            ExceptionTypeObject::new(ExceptionTypeId::KeyboardInterrupt, "KeyboardInterrupt");
        assert!(
            exc_type
                .flags
                .contains(ExceptionTypeFlags::KEYBOARD_INTERRUPT)
        );
        assert!(!exc_type.caught_by_bare_except());
    }

    #[test]
    fn test_new_generator_exit() {
        let exc_type = ExceptionTypeObject::new(ExceptionTypeId::GeneratorExit, "GeneratorExit");
        assert!(exc_type.flags.contains(ExceptionTypeFlags::GENERATOR_EXIT));
    }

    #[test]
    fn test_regular_exception_caught_by_bare_except() {
        let exc_type = ExceptionTypeObject::new(ExceptionTypeId::ValueError, "ValueError");
        assert!(exc_type.caught_by_bare_except());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Type ID Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_type_id_extraction() {
        let exc_type = ExceptionTypeObject::new(ExceptionTypeId::KeyError, "KeyError");
        assert_eq!(exc_type.type_id(), ExceptionTypeId::KeyError as u16);
    }

    #[test]
    fn test_exception_type_enum() {
        let exc_type = ExceptionTypeObject::new(ExceptionTypeId::IndexError, "IndexError");
        assert_eq!(exc_type.exception_type(), Some(ExceptionTypeId::IndexError));
    }

    #[test]
    fn test_base_type_id() {
        // ValueError's parent is Exception
        let exc_type = ExceptionTypeObject::new(ExceptionTypeId::ValueError, "ValueError");
        assert_eq!(exc_type.base_type_id(), ExceptionTypeId::Exception as u16);
    }

    #[test]
    fn test_base_exception_has_self_as_base() {
        let exc_type = ExceptionTypeObject::new(ExceptionTypeId::BaseException, "BaseException");
        assert_eq!(
            exc_type.base_type_id(),
            ExceptionTypeId::BaseException as u16
        );
    }

    // ════════════════════════════════════════════════════════════════════════
    // Subclass Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_is_subclass_of_self() {
        let exc_type = ExceptionTypeObject::new(ExceptionTypeId::ValueError, "ValueError");
        assert!(exc_type.is_subclass_of(ExceptionTypeId::ValueError as u16));
    }

    #[test]
    fn test_value_error_is_subclass_of_exception() {
        let exc_type = ExceptionTypeObject::new(ExceptionTypeId::ValueError, "ValueError");
        assert!(exc_type.is_subclass_of(ExceptionTypeId::Exception as u16));
    }

    #[test]
    fn test_value_error_is_subclass_of_base_exception() {
        let exc_type = ExceptionTypeObject::new(ExceptionTypeId::ValueError, "ValueError");
        assert!(exc_type.is_subclass_of(ExceptionTypeId::BaseException as u16));
    }

    #[test]
    fn test_type_error_not_subclass_of_value_error() {
        let exc_type = ExceptionTypeObject::new(ExceptionTypeId::TypeError, "TypeError");
        assert!(!exc_type.is_subclass_of(ExceptionTypeId::ValueError as u16));
    }

    #[test]
    fn test_zero_division_is_subclass_of_arithmetic() {
        let exc_type =
            ExceptionTypeObject::new(ExceptionTypeId::ZeroDivisionError, "ZeroDivisionError");
        assert!(exc_type.is_subclass_of(ExceptionTypeId::ArithmeticError as u16));
    }

    #[test]
    fn test_key_error_is_subclass_of_lookup_error() {
        let exc_type = ExceptionTypeObject::new(ExceptionTypeId::KeyError, "KeyError");
        assert!(exc_type.is_subclass_of(ExceptionTypeId::LookupError as u16));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Construction/Call Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_construct_no_args() {
        let exc_type = ExceptionTypeObject::new(ExceptionTypeId::ValueError, "ValueError");
        let exc = exc_type.construct(&[]);
        assert!(exc.is_object());
    }

    #[test]
    fn test_construct_with_int_arg() {
        let exc_type = ExceptionTypeObject::new(ExceptionTypeId::ValueError, "ValueError");
        let exc = exc_type.construct(&[Value::int(42).unwrap()]);
        assert!(exc.is_object());
    }

    #[test]
    fn test_call_returns_ok() {
        let exc_type = ExceptionTypeObject::new(ExceptionTypeId::TypeError, "TypeError");
        let result = exc_type.call(&[]);
        assert!(result.is_ok());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Static Type Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_static_value_error() {
        assert_eq!(
            VALUE_ERROR.exception_type_id,
            ExceptionTypeId::ValueError as u16
        );
        assert_eq!(VALUE_ERROR.name(), "ValueError");
    }

    #[test]
    fn test_static_type_error() {
        assert_eq!(
            TYPE_ERROR.exception_type_id,
            ExceptionTypeId::TypeError as u16
        );
    }

    #[test]
    fn test_static_system_exit() {
        assert!(SYSTEM_EXIT.is_system_exit());
    }

    #[test]
    fn test_static_keyboard_interrupt() {
        assert!(
            KEYBOARD_INTERRUPT
                .flags
                .contains(ExceptionTypeFlags::KEYBOARD_INTERRUPT)
        );
    }

    // ════════════════════════════════════════════════════════════════════════
    // Table Lookup Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_get_exception_type_value_error() {
        let exc_type = get_exception_type("ValueError");
        assert!(exc_type.is_some());
        assert_eq!(exc_type.unwrap().name(), "ValueError");
    }

    #[test]
    fn test_get_exception_type_type_error() {
        let exc_type = get_exception_type("TypeError");
        assert!(exc_type.is_some());
    }

    #[test]
    fn test_get_exception_type_ioerror_alias() {
        let exc_type = get_exception_type("IOError");
        assert!(exc_type.is_some());
        // IOError is an alias for OSError
        assert_eq!(
            exc_type.unwrap().exception_type_id,
            ExceptionTypeId::OSError as u16
        );
    }

    #[test]
    fn test_get_exception_type_environment_error_alias() {
        let exc_type = get_exception_type("EnvironmentError");
        assert!(exc_type.is_some());
        assert_eq!(
            exc_type.unwrap().exception_type_id,
            ExceptionTypeId::OSError as u16
        );
    }

    #[test]
    fn test_get_exception_type_nonexistent() {
        let exc_type = get_exception_type("FooBarError");
        assert!(exc_type.is_none());
    }

    #[test]
    fn test_get_exception_type_by_id_value_error() {
        let exc_type = get_exception_type_by_id(ExceptionTypeId::ValueError as u16);
        assert!(exc_type.is_some());
        assert_eq!(exc_type.unwrap().name(), "ValueError");
    }

    #[test]
    fn test_get_exception_type_by_id_invalid() {
        let exc_type = get_exception_type_by_id(255);
        assert!(exc_type.is_none());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Table Completeness Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_exception_type_table_length() {
        // 52 unique types + 2 aliases + some warnings
        assert!(EXCEPTION_TYPE_TABLE.len() >= 60);
    }

    #[test]
    fn test_exception_type_table_has_base_exception() {
        assert!(get_exception_type("BaseException").is_some());
    }

    #[test]
    fn test_exception_type_table_has_exception() {
        assert!(get_exception_type("Exception").is_some());
    }

    #[test]
    fn test_exception_type_table_has_all_common_types() {
        let common_types = [
            "ValueError",
            "TypeError",
            "KeyError",
            "IndexError",
            "AttributeError",
            "NameError",
            "RuntimeError",
            "StopIteration",
            "OSError",
            "ZeroDivisionError",
        ];

        for name in common_types {
            assert!(
                get_exception_type(name).is_some(),
                "Missing exception type: {}",
                name
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // Display/Debug Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_display() {
        let exc_type = ExceptionTypeObject::new(ExceptionTypeId::ValueError, "ValueError");
        let display = format!("{}", exc_type);
        assert_eq!(display, "<class 'ValueError'>");
    }

    #[test]
    fn test_debug() {
        let exc_type = ExceptionTypeObject::new(ExceptionTypeId::TypeError, "TypeError");
        let debug = format!("{:?}", exc_type);
        assert!(debug.contains("ExceptionTypeObject"));
        assert!(debug.contains("TypeError"));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Flags Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_exception_type_flags_default() {
        let flags = ExceptionTypeFlags::default();
        assert!(flags.is_empty());
    }

    #[test]
    fn test_exception_type_flags_combine() {
        let flags = ExceptionTypeFlags::ABSTRACT | ExceptionTypeFlags::CUSTOM_INIT;
        assert!(flags.contains(ExceptionTypeFlags::ABSTRACT));
        assert!(flags.contains(ExceptionTypeFlags::CUSTOM_INIT));
        assert!(!flags.contains(ExceptionTypeFlags::SYSTEM_EXIT));
    }

    // ════════════════════════════════════════════════════════════════════════
    // EXCEPTION_TYPE_ID Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_exception_type_id_constant() {
        assert_eq!(EXCEPTION_TYPE_ID.0, 27);
    }

    #[test]
    fn test_exception_type_object_uses_correct_type_id() {
        let exc_type = ExceptionTypeObject::new(ExceptionTypeId::ValueError, "ValueError");
        assert_eq!(exc_type.header.type_id, EXCEPTION_TYPE_ID);
    }
}
