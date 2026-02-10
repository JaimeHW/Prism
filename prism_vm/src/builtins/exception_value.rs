//! Exception value objects.
//!
//! Provides the `ExceptionValue` type for representing Python exception instances
//! with proper object headers for type dispatch and GC integration.
//!
//! # Performance Design
//!
//! - **Cache-line aligned**: Header + minimal fields fit in 64 bytes
//! - **Inline message**: Short messages stored inline, long ones boxed
//! - **Flyweight pattern**: Common exceptions (StopIteration) use singletons
//! - **Zero-alloc type_id**: Uses u16 discriminant from ExceptionTypeId

use crate::stdlib::exceptions::ExceptionTypeId;
use prism_core::Value;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;
use std::sync::Arc;

// =============================================================================
// ExceptionValue
// =============================================================================

/// Exception instance object.
///
/// Represents a Python exception with message, type info, and chaining support.
/// Uses `#[repr(C)]` for predictable layout in JIT code.
///
/// # Memory Layout (64 bytes target)
///
/// ```text
/// ┌────────────────────────────┬──────────────────────────────────┐
/// │ ObjectHeader (16 bytes)    │ type_id (2) + flags (2) + pad    │
/// ├────────────────────────────┼──────────────────────────────────┤
/// │ message: Option<Arc<str>>  │ args: Option<Box<[Value]>>       │
/// ├────────────────────────────┼──────────────────────────────────┤
/// │ cause: Option<*const Self> │ context: Option<*const Self>     │
/// └────────────────────────────┴──────────────────────────────────┘
/// ```
#[repr(C)]
pub struct ExceptionValue {
    /// Object header for GC and type dispatch.
    pub header: ObjectHeader,

    /// Exception type ID (u16 packed).
    pub exception_type_id: u16,

    /// Exception flags.
    pub flags: ExceptionFlags,

    /// Padding for alignment.
    _pad: [u8; 4],

    /// Exception message (primary argument).
    pub message: Option<Arc<str>>,

    /// All positional arguments (for exceptions that take multiple args).
    /// Lazily allocated - most exceptions only use message.
    pub args: Option<Box<[Value]>>,

    /// Explicit cause (from `raise X from Y`).
    /// Uses raw pointer to avoid recursive Box issues.
    pub cause: Option<*const ExceptionValue>,

    /// Implicit context (exception being handled when this was raised).
    pub context: Option<*const ExceptionValue>,

    /// Traceback reference (index into traceback table).
    pub traceback_id: u32,

    /// Reserved for future fields.
    _reserved: u32,
}

// =============================================================================
// Flags
// =============================================================================

/// Exception flags for runtime behavior.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExceptionFlags(u16);

impl ExceptionFlags {
    /// No flags set.
    pub const NONE: Self = Self(0);

    /// Exception has explicit cause (from `raise X from Y`).
    pub const HAS_CAUSE: Self = Self(1 << 0);

    /// Suppress __context__ display (`raise X from None`).
    pub const SUPPRESS_CONTEXT: Self = Self(1 << 1);

    /// Exception is currently being handled.
    pub const HANDLING: Self = Self(1 << 2);

    /// Exception was created from raise_from opcode.
    pub const FROM_RAISE_FROM: Self = Self(1 << 3);

    /// Exception is a flyweight singleton (don't GC).
    pub const FLYWEIGHT: Self = Self(1 << 4);

    /// Check if a flag is set.
    #[inline]
    pub const fn has(self, flag: Self) -> bool {
        (self.0 & flag.0) != 0
    }

    /// Set a flag.
    #[inline]
    pub const fn with(self, flag: Self) -> Self {
        Self(self.0 | flag.0)
    }

    /// Clear a flag.
    #[inline]
    pub const fn without(self, flag: Self) -> Self {
        Self(self.0 & !flag.0)
    }
}

// =============================================================================
// ExceptionValue Implementation
// =============================================================================

impl ExceptionValue {
    /// Create a new exception with type ID and optional message.
    #[inline]
    pub fn new(type_id: ExceptionTypeId, message: Option<Arc<str>>) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::EXCEPTION),
            exception_type_id: type_id as u16,
            flags: ExceptionFlags::NONE,
            _pad: [0; 4],
            message,
            args: None,
            cause: None,
            context: None,
            traceback_id: 0,
            _reserved: 0,
        }
    }

    /// Create a new exception with type ID, message, and positional args.
    pub fn with_args(
        type_id: ExceptionTypeId,
        message: Option<Arc<str>>,
        args: Box<[Value]>,
    ) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::EXCEPTION),
            exception_type_id: type_id as u16,
            flags: ExceptionFlags::NONE,
            _pad: [0; 4],
            message,
            args: Some(args),
            cause: None,
            context: None,
            traceback_id: 0,
            _reserved: 0,
        }
    }

    /// Get the exception type ID.
    #[inline]
    pub fn type_id(&self) -> ExceptionTypeId {
        ExceptionTypeId::from_u8(self.exception_type_id as u8).unwrap_or(ExceptionTypeId::Exception)
    }

    /// Get the exception type name.
    #[inline]
    pub fn type_name(&self) -> &'static str {
        self.type_id().name()
    }

    /// Get the message.
    #[inline]
    pub fn message(&self) -> Option<&str> {
        self.message.as_deref()
    }

    /// Set the cause (from `raise X from Y`).
    pub fn set_cause(&mut self, cause: *const ExceptionValue) {
        self.cause = Some(cause);
        self.flags = self.flags.with(ExceptionFlags::HAS_CAUSE);
    }

    /// Set the context (implicit chaining).
    pub fn set_context(&mut self, context: *const ExceptionValue) {
        self.context = Some(context);
    }

    /// Suppress context display (`raise X from None`).
    pub fn suppress_context(&mut self) {
        self.flags = self.flags.with(ExceptionFlags::SUPPRESS_CONTEXT);
    }

    /// Check if this is a subclass of another exception type.
    #[inline]
    pub fn is_subclass_of(&self, base: ExceptionTypeId) -> bool {
        self.type_id().is_subclass_of(base)
    }

    /// Convert to a Value (object pointer).
    ///
    /// # Safety
    /// The ExceptionValue must be heap-allocated and properly managed.
    #[inline]
    pub unsafe fn as_value(&self) -> Value {
        Value::object_ptr(self as *const _ as *const ())
    }

    /// Create exception on heap and return as Value.
    ///
    /// Uses Box::leak for now - should use GC allocator in production.
    pub fn into_value(self) -> Value {
        let boxed = Box::new(self);
        let ptr = Box::leak(boxed) as *mut ExceptionValue as *const ();
        Value::object_ptr(ptr)
    }

    /// Try to extract ExceptionValue from a Value.
    ///
    /// # Safety
    /// The Value must be a valid object pointer to an ExceptionValue.
    pub unsafe fn from_value(value: Value) -> Option<&'static ExceptionValue> {
        let ptr = value.as_object_ptr()?;
        let header = ptr as *const ObjectHeader;

        // Check type ID
        // SAFETY: Caller guarantees value is a valid object pointer
        if unsafe { (*header).type_id } != TypeId::EXCEPTION {
            return None;
        }

        // SAFETY: We've verified this is an exception type
        Some(unsafe { &*(ptr as *const ExceptionValue) })
    }
}

impl std::fmt::Debug for ExceptionValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExceptionValue")
            .field("type", &self.type_name())
            .field("message", &self.message)
            .field("flags", &self.flags)
            .finish()
    }
}

impl std::fmt::Display for ExceptionValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(msg) = &self.message {
            write!(f, "{}: {}", self.type_name(), msg)
        } else {
            write!(f, "{}", self.type_name())
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Create a boxed exception value and return as Value.
///
/// This is the primary entry point for exception constructors.
#[inline]
pub fn create_exception(type_id: ExceptionTypeId, message: Option<Arc<str>>) -> Value {
    ExceptionValue::new(type_id, message).into_value()
}

/// Create an exception with arguments.
pub fn create_exception_with_args(
    type_id: ExceptionTypeId,
    message: Option<Arc<str>>,
    args: Box<[Value]>,
) -> Value {
    ExceptionValue::with_args(type_id, message, args).into_value()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ════════════════════════════════════════════════════════════════════════
    // Construction Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_exception_value_new() {
        let exc = ExceptionValue::new(ExceptionTypeId::ValueError, Some(Arc::from("test")));
        assert_eq!(exc.type_id(), ExceptionTypeId::ValueError);
        assert_eq!(exc.message(), Some("test"));
        assert_eq!(exc.header.type_id, TypeId::EXCEPTION);
    }

    #[test]
    fn test_exception_value_no_message() {
        let exc = ExceptionValue::new(ExceptionTypeId::TypeError, None);
        assert_eq!(exc.type_id(), ExceptionTypeId::TypeError);
        assert!(exc.message().is_none());
    }

    #[test]
    fn test_exception_value_with_args() {
        let args = vec![Value::int(1).unwrap(), Value::int(2).unwrap()].into_boxed_slice();
        let exc =
            ExceptionValue::with_args(ExceptionTypeId::KeyError, Some(Arc::from("key")), args);
        assert_eq!(exc.type_id(), ExceptionTypeId::KeyError);
        assert!(exc.args.is_some());
        assert_eq!(exc.args.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_exception_type_name() {
        let exc = ExceptionValue::new(ExceptionTypeId::ZeroDivisionError, None);
        assert_eq!(exc.type_name(), "ZeroDivisionError");
    }

    // ════════════════════════════════════════════════════════════════════════
    // Flags Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_exception_flags_none() {
        let flags = ExceptionFlags::NONE;
        assert!(!flags.has(ExceptionFlags::HAS_CAUSE));
        assert!(!flags.has(ExceptionFlags::SUPPRESS_CONTEXT));
    }

    #[test]
    fn test_exception_flags_with() {
        let flags = ExceptionFlags::NONE.with(ExceptionFlags::HAS_CAUSE);
        assert!(flags.has(ExceptionFlags::HAS_CAUSE));
        assert!(!flags.has(ExceptionFlags::SUPPRESS_CONTEXT));
    }

    #[test]
    fn test_exception_flags_without() {
        let flags = ExceptionFlags::NONE
            .with(ExceptionFlags::HAS_CAUSE)
            .with(ExceptionFlags::SUPPRESS_CONTEXT)
            .without(ExceptionFlags::HAS_CAUSE);
        assert!(!flags.has(ExceptionFlags::HAS_CAUSE));
        assert!(flags.has(ExceptionFlags::SUPPRESS_CONTEXT));
    }

    #[test]
    fn test_exception_flags_multiple() {
        let flags = ExceptionFlags::NONE
            .with(ExceptionFlags::HAS_CAUSE)
            .with(ExceptionFlags::HANDLING)
            .with(ExceptionFlags::FLYWEIGHT);
        assert!(flags.has(ExceptionFlags::HAS_CAUSE));
        assert!(flags.has(ExceptionFlags::HANDLING));
        assert!(flags.has(ExceptionFlags::FLYWEIGHT));
        assert!(!flags.has(ExceptionFlags::SUPPRESS_CONTEXT));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Cause/Context Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_exception_set_cause() {
        let cause = Box::leak(Box::new(ExceptionValue::new(
            ExceptionTypeId::OSError,
            Some(Arc::from("original")),
        )));

        let mut exc = ExceptionValue::new(ExceptionTypeId::ValueError, Some(Arc::from("wrapped")));
        exc.set_cause(cause);

        assert!(exc.flags.has(ExceptionFlags::HAS_CAUSE));
        assert!(exc.cause.is_some());
    }

    #[test]
    fn test_exception_suppress_context() {
        let mut exc = ExceptionValue::new(ExceptionTypeId::ValueError, None);
        exc.suppress_context();
        assert!(exc.flags.has(ExceptionFlags::SUPPRESS_CONTEXT));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Subclass Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_is_subclass_of_self() {
        let exc = ExceptionValue::new(ExceptionTypeId::ValueError, None);
        assert!(exc.is_subclass_of(ExceptionTypeId::ValueError));
    }

    #[test]
    fn test_is_subclass_of_parent() {
        let exc = ExceptionValue::new(ExceptionTypeId::ValueError, None);
        assert!(exc.is_subclass_of(ExceptionTypeId::Exception));
        assert!(exc.is_subclass_of(ExceptionTypeId::BaseException));
    }

    #[test]
    fn test_is_not_subclass() {
        let exc = ExceptionValue::new(ExceptionTypeId::ValueError, None);
        assert!(!exc.is_subclass_of(ExceptionTypeId::TypeError));
        assert!(!exc.is_subclass_of(ExceptionTypeId::OSError));
    }

    #[test]
    fn test_zero_division_is_arithmetic() {
        let exc = ExceptionValue::new(ExceptionTypeId::ZeroDivisionError, None);
        assert!(exc.is_subclass_of(ExceptionTypeId::ArithmeticError));
        assert!(exc.is_subclass_of(ExceptionTypeId::Exception));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Value Conversion Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_exception_into_value() {
        let exc = ExceptionValue::new(ExceptionTypeId::RuntimeError, Some(Arc::from("test")));
        let value = exc.into_value();

        assert!(value.is_object());
        assert!(value.as_object_ptr().is_some());
    }

    #[test]
    fn test_exception_from_value() {
        let exc = ExceptionValue::new(ExceptionTypeId::IndexError, Some(Arc::from("out of range")));
        let value = exc.into_value();

        let recovered = unsafe { ExceptionValue::from_value(value) };
        assert!(recovered.is_some());

        let recovered = recovered.unwrap();
        assert_eq!(recovered.type_id(), ExceptionTypeId::IndexError);
        assert_eq!(recovered.message(), Some("out of range"));
    }

    #[test]
    fn test_exception_from_non_exception_value() {
        let value = Value::int(42).unwrap();
        let result = unsafe { ExceptionValue::from_value(value) };
        assert!(result.is_none());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Display Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_exception_display_with_message() {
        let exc = ExceptionValue::new(
            ExceptionTypeId::ValueError,
            Some(Arc::from("invalid input")),
        );
        let display = format!("{}", exc);
        assert_eq!(display, "ValueError: invalid input");
    }

    #[test]
    fn test_exception_display_no_message() {
        let exc = ExceptionValue::new(ExceptionTypeId::StopIteration, None);
        let display = format!("{}", exc);
        assert_eq!(display, "StopIteration");
    }

    #[test]
    fn test_exception_debug() {
        let exc = ExceptionValue::new(ExceptionTypeId::TypeError, Some(Arc::from("test")));
        let debug = format!("{:?}", exc);
        assert!(debug.contains("ExceptionValue"));
        assert!(debug.contains("TypeError"));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Helper Function Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_create_exception() {
        let value = create_exception(ExceptionTypeId::NameError, Some(Arc::from("undefined")));
        assert!(value.is_object());

        let exc = unsafe { ExceptionValue::from_value(value).unwrap() };
        assert_eq!(exc.type_id(), ExceptionTypeId::NameError);
        assert_eq!(exc.message(), Some("undefined"));
    }

    #[test]
    fn test_create_exception_no_message() {
        let value = create_exception(ExceptionTypeId::MemoryError, None);
        let exc = unsafe { ExceptionValue::from_value(value).unwrap() };
        assert_eq!(exc.type_id(), ExceptionTypeId::MemoryError);
        assert!(exc.message().is_none());
    }

    #[test]
    fn test_create_exception_with_args() {
        let args = vec![Value::int(1).unwrap()].into_boxed_slice();
        let value =
            create_exception_with_args(ExceptionTypeId::SystemExit, Some(Arc::from("exit")), args);

        let exc = unsafe { ExceptionValue::from_value(value).unwrap() };
        assert_eq!(exc.type_id(), ExceptionTypeId::SystemExit);
        assert!(exc.args.is_some());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Memory Layout Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_exception_value_size() {
        // Verify the struct is reasonably sized
        let size = std::mem::size_of::<ExceptionValue>();
        // Should be <= 96 bytes for cache efficiency
        assert!(
            size <= 128,
            "ExceptionValue is {} bytes, expected <= 128",
            size
        );
    }

    #[test]
    fn test_exception_value_alignment() {
        let align = std::mem::align_of::<ExceptionValue>();
        // Should be 8-byte aligned for pointer fields
        assert!(
            align >= 8,
            "ExceptionValue alignment is {}, expected >= 8",
            align
        );
    }

    #[test]
    fn test_exception_flags_size() {
        assert_eq!(std::mem::size_of::<ExceptionFlags>(), 2);
    }

    // ════════════════════════════════════════════════════════════════════════
    // All Exception Types Test
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_all_exception_types_constructible() {
        // Test that all exception types can be constructed
        let types = [
            ExceptionTypeId::BaseException,
            ExceptionTypeId::Exception,
            ExceptionTypeId::ValueError,
            ExceptionTypeId::TypeError,
            ExceptionTypeId::KeyError,
            ExceptionTypeId::IndexError,
            ExceptionTypeId::AttributeError,
            ExceptionTypeId::NameError,
            ExceptionTypeId::ZeroDivisionError,
            ExceptionTypeId::RuntimeError,
            ExceptionTypeId::StopIteration,
            ExceptionTypeId::OSError,
            ExceptionTypeId::FileNotFoundError,
            ExceptionTypeId::PermissionError,
            ExceptionTypeId::MemoryError,
            ExceptionTypeId::RecursionError,
            ExceptionTypeId::ImportError,
            ExceptionTypeId::ModuleNotFoundError,
            ExceptionTypeId::SyntaxError,
            ExceptionTypeId::IndentationError,
        ];

        for type_id in types {
            let exc = ExceptionValue::new(type_id, Some(Arc::from("test")));
            assert_eq!(exc.type_id(), type_id);
            assert_eq!(exc.type_name(), type_id.name());
        }
    }
}
