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

use crate::VirtualMachine;
use crate::builtins::BuiltinError;
use crate::builtins::exception_value::{
    create_exception, create_exception_in_vm, create_exception_with_args,
    create_exception_with_args_in_vm,
};
use crate::error::RuntimeError;
use crate::stdlib::exceptions::ExceptionTypeId;
use prism_core::Value;
use prism_core::intern::{InternedString, intern};
use prism_runtime::allocation_context::alloc_static_value;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::class::{ClassDict, ClassFlags, PyClassObject};
use prism_runtime::object::descriptor::StaticMethodDescriptor;
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::type_builtins::{
    global_class_registry, register_global_class, type_new,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::tuple::TupleObject;
use rustc_hash::FxHashMap;
use std::sync::{Arc, LazyLock, Mutex};

// =============================================================================
// Type ID Extension
// =============================================================================

/// TypeId for exception type objects (callable that constructs exceptions).
pub const EXCEPTION_TYPE_ID: TypeId = TypeId(27);

static EXCEPTION_PROXY_CLASSES: LazyLock<Mutex<FxHashMap<u16, Arc<PyClassObject>>>> =
    LazyLock::new(|| Mutex::new(FxHashMap::default()));
static EXCEPTION_TYPE_IDS_BY_PROXY_CLASS: LazyLock<Mutex<FxHashMap<u32, u16>>> =
    LazyLock::new(|| Mutex::new(FxHashMap::default()));

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

        if args.is_empty() {
            return create_exception(type_id, None);
        }

        create_exception_with_args(type_id, None, args.to_vec().into_boxed_slice())
    }

    /// Construct an exception instance using the VM-managed heap.
    #[inline]
    pub fn construct_in_vm(
        &self,
        vm: &VirtualMachine,
        args: &[Value],
    ) -> Result<Value, RuntimeError> {
        let type_id = ExceptionTypeId::from_u8(self.exception_type_id as u8)
            .unwrap_or(ExceptionTypeId::Exception);

        if args.is_empty() {
            return create_exception_in_vm(vm, type_id, None);
        }

        create_exception_with_args_in_vm(vm, type_id, None, args.to_vec().into_boxed_slice())
    }

    /// Call the exception type as a constructor (for builtins integration).
    ///
    /// This is the entry point for `ValueError("message")` style calls.
    #[inline]
    pub fn call(&self, args: &[Value]) -> Result<Value, BuiltinError> {
        Ok(self.construct(args))
    }

    /// Call the exception type using the VM-managed heap.
    #[inline]
    pub fn call_in_vm(&self, vm: &VirtualMachine, args: &[Value]) -> Result<Value, RuntimeError> {
        self.construct_in_vm(vm, args)
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
    BASE_EXCEPTION_GROUP,
    ExceptionTypeId::BaseExceptionGroup,
    "BaseExceptionGroup"
);
define_exception_type!(
    EXCEPTION_GROUP,
    ExceptionTypeId::ExceptionGroup,
    "ExceptionGroup"
);
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
define_exception_type!(SYSTEM_ERROR, ExceptionTypeId::SystemError, "SystemError");
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
    ("BaseExceptionGroup", &BASE_EXCEPTION_GROUP),
    ("ExceptionGroup", &EXCEPTION_GROUP),
    ("StopAsyncIteration", &STOP_ASYNC_ITERATION),
    ("StopIteration", &STOP_ITERATION),
    ("SyntaxError", &SYNTAX_ERROR),
    ("TabError", &TAB_ERROR),
    ("SystemError", &SYSTEM_ERROR),
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

#[inline]
fn exception_type_value(exception_type: &'static ExceptionTypeObject) -> Value {
    Value::object_ptr(exception_type as *const ExceptionTypeObject as *const ())
}

#[inline]
pub(crate) fn exception_type_value_for_id(type_id: u16) -> Option<Value> {
    Some(exception_type_value(get_exception_type_by_id(type_id)?))
}

#[inline]
fn boxed_tuple_value(items: Vec<Value>) -> Value {
    crate::alloc_managed_value(TupleObject::from_vec(items))
}

#[inline]
fn exception_type_base_value(exception_type: &ExceptionTypeObject) -> Value {
    match exception_type.exception_type() {
        Some(ExceptionTypeId::BaseException) => {
            crate::builtins::builtin_type_object_for_type_id(TypeId::OBJECT)
        }
        _ => get_exception_type_by_id(exception_type.base_type_id())
            .map(exception_type_value)
            .unwrap_or_else(Value::none),
    }
}

#[inline]
fn exception_type_direct_base_values(exception_type: &ExceptionTypeObject) -> Vec<Value> {
    let Some(kind) = exception_type.exception_type() else {
        return Vec::new();
    };

    if kind == ExceptionTypeId::BaseException {
        return vec![crate::builtins::builtin_type_object_for_type_id(
            TypeId::OBJECT,
        )];
    }

    let mut bases = Vec::with_capacity(2);
    for base in [kind.parent(), kind.secondary_parent()]
        .into_iter()
        .flatten()
    {
        let Some(base_type) = get_exception_type_by_id(base as u16) else {
            continue;
        };
        let value = exception_type_value(base_type);
        if !bases.contains(&value) {
            bases.push(value);
        }
    }

    bases
}

#[inline]
fn exception_type_bases_tuple_value(exception_type: &ExceptionTypeObject) -> Value {
    boxed_tuple_value(exception_type_direct_base_values(exception_type))
}

#[inline]
fn exception_type_mro_tuple_value(exception_type: &'static ExceptionTypeObject) -> Value {
    let mut mro = vec![exception_type_value(exception_type)];
    let Some(kind) = exception_type.exception_type() else {
        mro.push(crate::builtins::builtin_type_object_for_type_id(
            TypeId::OBJECT,
        ));
        return boxed_tuple_value(mro);
    };

    fn append_exception_mro(kind: ExceptionTypeId, mro: &mut Vec<Value>) {
        for base in [kind.parent(), kind.secondary_parent()]
            .into_iter()
            .flatten()
        {
            let base_type = get_exception_type_by_id(base as u16)
                .expect("parent exception type should always be registered");
            let value = exception_type_value(base_type);
            if !mro.contains(&value) {
                mro.push(value);
            }
        }

        for base in [kind.parent(), kind.secondary_parent()]
            .into_iter()
            .flatten()
        {
            append_exception_mro(base, mro);
        }
    }

    append_exception_mro(kind, &mut mro);

    let object_type = crate::builtins::builtin_type_object_for_type_id(TypeId::OBJECT);
    if !mro.contains(&object_type) {
        mro.push(object_type);
    }
    boxed_tuple_value(mro)
}

pub(crate) fn exception_type_attribute_value(
    exception_type: &'static ExceptionTypeObject,
    name: &InternedString,
) -> Option<Value> {
    match name.as_str() {
        "__new__" => super::exception_method_value("__new__"),
        "__init__" => super::exception_method_value("__init__"),
        "__str__" => super::exception_method_value("__str__"),
        "__repr__" => super::exception_method_value("__repr__"),
        "with_traceback" => super::exception_method_value("with_traceback"),
        "__name__" | "__qualname__" => Some(Value::string(intern(exception_type.name()))),
        "__module__" => Some(Value::string(intern("builtins"))),
        "__bases__" => Some(exception_type_bases_tuple_value(exception_type)),
        "__base__" => Some(exception_type_base_value(exception_type)),
        "__mro__" => Some(exception_type_mro_tuple_value(exception_type)),
        _ => None,
    }
}

#[inline]
fn exception_proxy_base_ids(exception_type_id: ExceptionTypeId) -> Vec<ClassId> {
    let mut bases = Vec::with_capacity(2);
    for base in [
        exception_type_id.parent(),
        exception_type_id.secondary_parent(),
    ]
    .into_iter()
    .flatten()
    {
        let class_id = exception_proxy_class(base).class_id();
        if !bases.contains(&class_id) {
            bases.push(class_id);
        }
    }
    bases
}

#[inline]
fn exception_proxy_namespace_value(method_name: &str, method: Value) -> Value {
    if method_name == "__new__" {
        alloc_static_value(StaticMethodDescriptor::new(method))
    } else {
        method
    }
}

fn build_exception_proxy_class(exception_type_id: ExceptionTypeId) -> Arc<PyClassObject> {
    let bases = exception_proxy_base_ids(exception_type_id);
    let namespace = ClassDict::new();
    for method_name in [
        "__new__",
        "__init__",
        "__str__",
        "__repr__",
        "with_traceback",
    ] {
        if let Some(method) = super::exception_method_value(method_name) {
            namespace.set(
                intern(method_name),
                exception_proxy_namespace_value(method_name, method),
            );
        }
    }
    let name = intern(
        get_exception_type_by_id(exception_type_id as u16)
            .expect("every ExceptionTypeId should resolve to a registered exception type")
            .name(),
    );
    let result = type_new(name, &bases, &namespace, global_class_registry())
        .expect("built-in exception proxy classes should always construct successfully");
    let mut class =
        Arc::try_unwrap(result.class).expect("freshly created exception proxy should be unique");
    class.add_flags(ClassFlags::NATIVE_HEAPTYPE);
    let class = Arc::new(class);
    register_global_class(class.clone(), result.bitmap);
    class
}

pub(crate) fn exception_proxy_class(exception_type_id: ExceptionTypeId) -> Arc<PyClassObject> {
    {
        let cache = EXCEPTION_PROXY_CLASSES
            .lock()
            .expect("exception proxy cache lock poisoned");
        if let Some(class) = cache.get(&(exception_type_id as u16)) {
            return class.clone();
        }
    }

    if let Some(parent) = exception_type_id.parent() {
        let _ = exception_proxy_class(parent);
    }
    if let Some(parent) = exception_type_id.secondary_parent() {
        let _ = exception_proxy_class(parent);
    }

    let class = build_exception_proxy_class(exception_type_id);
    let class_id = class.class_id().0;

    let mut cache = EXCEPTION_PROXY_CLASSES
        .lock()
        .expect("exception proxy cache lock poisoned");
    if let Some(existing) = cache.get(&(exception_type_id as u16)) {
        return existing.clone();
    }
    cache.insert(exception_type_id as u16, class.clone());

    EXCEPTION_TYPE_IDS_BY_PROXY_CLASS
        .lock()
        .expect("exception proxy reverse cache lock poisoned")
        .insert(class_id, exception_type_id as u16);

    class
}

#[inline]
pub(crate) fn exception_proxy_class_id(exception_type_id: u16) -> Option<ClassId> {
    let exception_type = ExceptionTypeId::from_u8(exception_type_id as u8)?;
    Some(exception_proxy_class(exception_type).class_id())
}

#[inline]
pub(crate) fn exception_type_id_for_proxy_class_id(class_id: ClassId) -> Option<u16> {
    EXCEPTION_TYPE_IDS_BY_PROXY_CLASS
        .lock()
        .expect("exception proxy reverse cache lock poisoned")
        .get(&class_id.0)
        .copied()
}

#[inline]
pub(crate) fn exception_proxy_class_id_from_ptr(ptr: *const ()) -> Option<ClassId> {
    let exc_type = unsafe { &*(ptr as *const ExceptionTypeObject) };
    exception_proxy_class_id(exc_type.exception_type_id)
}

#[inline]
pub(crate) fn exception_type_value_for_proxy_class_id(class_id: ClassId) -> Option<Value> {
    let exception_type_id = EXCEPTION_TYPE_IDS_BY_PROXY_CLASS
        .lock()
        .expect("exception proxy reverse cache lock poisoned")
        .get(&class_id.0)
        .copied()?;
    Some(exception_type_value(get_exception_type_by_id(
        exception_type_id,
    )?))
}

fn build_builtin_warning_category_class(name: &'static str) -> Arc<PyClassObject> {
    let warning_base = exception_proxy_class(ExceptionTypeId::Warning).class_id();
    let namespace = ClassDict::new();
    let result = type_new(
        intern(name),
        &[warning_base],
        &namespace,
        global_class_registry(),
    )
    .expect("built-in warning category classes should always construct successfully");
    register_global_class(result.class.clone(), result.bitmap);
    result.class
}

static BYTES_WARNING_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_builtin_warning_category_class("BytesWarning"));
static FUTURE_WARNING_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_builtin_warning_category_class("FutureWarning"));
static IMPORT_WARNING_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_builtin_warning_category_class("ImportWarning"));
static RESOURCE_WARNING_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_builtin_warning_category_class("ResourceWarning"));
static UNICODE_WARNING_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_builtin_warning_category_class("UnicodeWarning"));
static ENCODING_WARNING_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_builtin_warning_category_class("EncodingWarning"));

pub static SUPPLEMENTAL_EXCEPTION_CLASS_TABLE: &[(&str, &LazyLock<Arc<PyClassObject>>)] = &[
    ("BytesWarning", &BYTES_WARNING_CLASS),
    ("FutureWarning", &FUTURE_WARNING_CLASS),
    ("ImportWarning", &IMPORT_WARNING_CLASS),
    ("ResourceWarning", &RESOURCE_WARNING_CLASS),
    ("UnicodeWarning", &UNICODE_WARNING_CLASS),
    ("EncodingWarning", &ENCODING_WARNING_CLASS),
];

#[inline]
#[cfg(test)]
pub(crate) fn supplemental_exception_class(name: &str) -> Option<&'static Arc<PyClassObject>> {
    SUPPLEMENTAL_EXCEPTION_CLASS_TABLE
        .iter()
        .find(|(registered_name, _)| *registered_name == name)
        .map(|(_, class)| &***class)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::ExceptionValue;
    use prism_gc::heap::GcHeap;
    use prism_runtime::allocation_context::RuntimeHeapBinding;

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

    #[test]
    fn test_exception_group_is_subclass_of_exception_and_base_exception_group() {
        let exc_type = ExceptionTypeObject::new(ExceptionTypeId::ExceptionGroup, "ExceptionGroup");
        assert!(exc_type.is_subclass_of(ExceptionTypeId::Exception as u16));
        assert!(exc_type.is_subclass_of(ExceptionTypeId::BaseExceptionGroup as u16));
        assert!(exc_type.is_subclass_of(ExceptionTypeId::BaseException as u16));
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
    fn test_construct_with_string_arg_preserves_original_args() {
        let exc_type = ExceptionTypeObject::new(ExceptionTypeId::ValueError, "ValueError");
        let exc = exc_type.construct(&[Value::string(intern("boom"))]);
        let exc = unsafe { ExceptionValue::from_value(exc).expect("exception instance") };

        assert!(exc.message().is_none());
        let args = exc
            .args
            .as_deref()
            .expect("constructor should preserve args");
        assert_eq!(args.len(), 1);
        assert!(args[0].is_string());
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

    #[test]
    fn test_exception_proxy_class_bridge_preserves_exception_hierarchy() {
        let exception_id =
            exception_proxy_class_id(ExceptionTypeId::Exception as u16).expect("Exception proxy");
        let runtime_error_id = exception_proxy_class_id(ExceptionTypeId::RuntimeError as u16)
            .expect("RuntimeError proxy");
        let warning_id =
            exception_proxy_class_id(ExceptionTypeId::Warning as u16).expect("Warning proxy");

        let runtime_error_bitmap =
            prism_runtime::object::type_builtins::global_class_bitmap(runtime_error_id)
                .expect("RuntimeError proxy should be registered");
        let warning_bitmap = prism_runtime::object::type_builtins::global_class_bitmap(warning_id)
            .expect("Warning proxy should be registered");

        assert!(runtime_error_bitmap.is_subclass_of(TypeId::from_raw(exception_id.0)));
        assert!(!runtime_error_bitmap.is_subclass_of(TypeId::from_raw(warning_id.0)));
        assert!(warning_bitmap.is_subclass_of(TypeId::from_raw(exception_id.0)));
    }

    #[test]
    fn test_exception_group_proxy_class_preserves_dual_exception_bases() {
        let exception_id =
            exception_proxy_class_id(ExceptionTypeId::Exception as u16).expect("Exception proxy");
        let base_group_id = exception_proxy_class_id(ExceptionTypeId::BaseExceptionGroup as u16)
            .expect("BaseExceptionGroup proxy");
        let group_id = exception_proxy_class_id(ExceptionTypeId::ExceptionGroup as u16)
            .expect("ExceptionGroup proxy");

        let group_bitmap = prism_runtime::object::type_builtins::global_class_bitmap(group_id)
            .expect("ExceptionGroup proxy should be registered");

        assert!(group_bitmap.is_subclass_of(TypeId::from_raw(exception_id.0)));
        assert!(group_bitmap.is_subclass_of(TypeId::from_raw(base_group_id.0)));
    }

    #[test]
    fn test_exception_proxy_class_id_round_trips_to_builtin_exception_value() {
        let runtime_error_id = exception_proxy_class_id(ExceptionTypeId::RuntimeError as u16)
            .expect("RuntimeError proxy");
        let value = exception_type_value_for_proxy_class_id(runtime_error_id)
            .expect("RuntimeError proxy should map back to builtin exception type value");
        let ptr = value
            .as_object_ptr()
            .expect("builtin exception type should be a heap object");
        let exc_type = unsafe { &*(ptr as *const ExceptionTypeObject) };

        assert_eq!(
            exc_type.exception_type_id,
            ExceptionTypeId::RuntimeError as u16
        );
    }

    #[test]
    fn test_exception_type_attribute_value_exposes_class_metadata() {
        let warning = get_exception_type("Warning").expect("Warning type should exist");

        let name = exception_type_attribute_value(warning, &intern("__name__"))
            .expect("__name__ should exist");
        let name_ptr = name
            .as_string_object_ptr()
            .expect("__name__ should be an interned string");
        assert_eq!(
            prism_core::intern::interned_by_ptr(name_ptr as *const u8)
                .unwrap()
                .as_str(),
            "Warning"
        );

        let bases = exception_type_attribute_value(warning, &intern("__bases__"))
            .expect("__bases__ should exist");
        let bases_ptr = bases
            .as_object_ptr()
            .expect("__bases__ should be a tuple object");
        let bases = unsafe { &*(bases_ptr as *const TupleObject) };
        assert_eq!(bases.len(), 1);

        let base_ptr = bases.as_slice()[0]
            .as_object_ptr()
            .expect("base should be an exception type object");
        let base = unsafe { &*(base_ptr as *const ExceptionTypeObject) };
        assert_eq!(base.name(), "Exception");
    }

    #[test]
    fn test_exception_proxy_classes_expose_base_exception_slots() {
        let proxy = exception_proxy_class(ExceptionTypeId::Exception);

        for name in [
            "__new__",
            "__init__",
            "__str__",
            "__repr__",
            "with_traceback",
        ] {
            assert!(
                proxy.get_attr(&intern(name)).is_some(),
                "exception proxy should expose {name}"
            );
        }
    }

    #[test]
    fn test_exception_proxy_new_staticmethod_ignores_bound_vm_heap() {
        let heap = GcHeap::with_defaults();
        let _binding = RuntimeHeapBinding::register(&heap);
        let method = crate::builtins::exception_method_value("__new__")
            .expect("BaseException.__new__ should be registered");

        let value = exception_proxy_namespace_value("__new__", method);
        let ptr = value
            .as_object_ptr()
            .expect("staticmethod descriptor should be object-backed");
        let descriptor = unsafe { &*(ptr as *const StaticMethodDescriptor) };

        assert_eq!(descriptor.header.type_id, TypeId::STATICMETHOD);
        assert_eq!(descriptor.function(), method);
        assert!(!heap.contains(ptr));
    }

    #[test]
    fn test_exception_proxy_classes_are_native_heaptypes() {
        let proxy = exception_proxy_class(ExceptionTypeId::Exception);
        assert!(proxy.is_native_heaptype());
    }

    #[test]
    fn test_exception_type_attribute_value_exposes_base_exception_methods() {
        let base = get_exception_type("BaseException").expect("BaseException type should exist");

        for (name, expected) in [
            ("__new__", "BaseException.__new__"),
            ("__init__", "BaseException.__init__"),
            ("__str__", "BaseException.__str__"),
            ("__repr__", "BaseException.__repr__"),
            ("with_traceback", "BaseException.with_traceback"),
        ] {
            let value = exception_type_attribute_value(base, &intern(name))
                .unwrap_or_else(|| panic!("{name} should resolve from exception type metadata"));
            let ptr = value
                .as_object_ptr()
                .expect("base exception methods should be heap allocated builtins");
            let builtin = unsafe { &*(ptr as *const crate::builtins::BuiltinFunctionObject) };
            assert_eq!(builtin.name(), expected);
            assert_eq!(builtin.bound_self(), None);
        }
    }

    #[test]
    fn test_exception_group_type_metadata_exposes_both_bases_and_full_mro() {
        let exc_group =
            get_exception_type("ExceptionGroup").expect("ExceptionGroup type should exist");

        let bases = exception_type_attribute_value(exc_group, &intern("__bases__"))
            .expect("__bases__ should exist");
        let bases_ptr = bases
            .as_object_ptr()
            .expect("__bases__ should be a tuple object");
        let bases = unsafe { &*(bases_ptr as *const TupleObject) };
        assert_eq!(bases.len(), 2);

        let first_base = unsafe {
            &*(bases.as_slice()[0]
                .as_object_ptr()
                .expect("first base should be an exception type object")
                as *const ExceptionTypeObject)
        };
        let second_base = unsafe {
            &*(bases.as_slice()[1]
                .as_object_ptr()
                .expect("second base should be an exception type object")
                as *const ExceptionTypeObject)
        };
        assert_eq!(first_base.name(), "BaseExceptionGroup");
        assert_eq!(second_base.name(), "Exception");

        let mro = exception_type_attribute_value(exc_group, &intern("__mro__"))
            .expect("__mro__ should exist");
        let mro_ptr = mro
            .as_object_ptr()
            .expect("__mro__ should be a tuple object");
        let mro = unsafe { &*(mro_ptr as *const TupleObject) };
        let mro_names = mro
            .as_slice()
            .iter()
            .map(|value| {
                let ptr = value
                    .as_object_ptr()
                    .expect("mro entries should be type objects");
                if crate::ops::objects::extract_type_id(ptr) == TypeId::TYPE
                    && let Some(type_id) = crate::builtins::builtin_type_object_type_id(ptr)
                {
                    return type_id.name().to_string();
                }

                unsafe { &*(ptr as *const ExceptionTypeObject) }
                    .name()
                    .to_string()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            mro_names,
            vec![
                "ExceptionGroup",
                "BaseExceptionGroup",
                "Exception",
                "BaseException",
                "object",
            ]
        );
    }

    // ════════════════════════════════════════════════════════════════════════
    // Table Completeness Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_supplemental_warning_category_table_contains_python_builtin_names() {
        let names: Vec<_> = SUPPLEMENTAL_EXCEPTION_CLASS_TABLE
            .iter()
            .map(|(name, _)| *name)
            .collect();

        assert!(names.contains(&"BytesWarning"));
        assert!(names.contains(&"ImportWarning"));
        assert!(names.contains(&"ResourceWarning"));
        assert!(names.contains(&"EncodingWarning"));
    }

    #[test]
    fn test_supplemental_warning_categories_inherit_warning_proxy_class() {
        let warning_id =
            exception_proxy_class_id(ExceptionTypeId::Warning as u16).expect("Warning proxy");

        for name in [
            "BytesWarning",
            "FutureWarning",
            "ImportWarning",
            "ResourceWarning",
            "UnicodeWarning",
            "EncodingWarning",
        ] {
            let class = supplemental_exception_class(name)
                .unwrap_or_else(|| panic!("missing supplemental warning category {name}"));
            assert!(
                prism_runtime::object::type_builtins::issubclass(
                    class.class_id(),
                    warning_id,
                    prism_runtime::object::type_builtins::global_class_bitmap,
                ),
                "{name} should inherit from Warning",
            );
        }
    }

    #[test]
    fn test_supplemental_warning_categories_are_type_objects() {
        for name in [
            "BytesWarning",
            "FutureWarning",
            "ImportWarning",
            "ResourceWarning",
            "UnicodeWarning",
            "EncodingWarning",
        ] {
            let class = supplemental_exception_class(name)
                .unwrap_or_else(|| panic!("missing supplemental warning category {name}"));
            assert_eq!(
                class.header.type_id,
                TypeId::TYPE,
                "{name} should be a type"
            );
        }
    }

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
            "SystemError",
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
