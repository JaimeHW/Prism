//! Exception object implementation.
//!
//! This module provides the core `ExceptionObject` type that represents
//! a Python exception instance. The design follows CPython 3.11+ patterns
//! with lazy allocation of args and tracebacks.
//!
//! # Performance Design
//!
//! - **Lazy args**: Exception args are only allocated when accessed
//! - **Lazy traceback**: Traceback is built incrementally as exception propagates
//! - **Compact header**: Type ID + flags = 2 bytes
//! - **OnceCell for lazy init**: Thread-safe lazy initialization

use super::flags::ExceptionFlags;
use super::traceback::TracebackObject;
use super::types::ExceptionTypeId;
use prism_core::Value;
use std::fmt;
use std::sync::Arc;
use std::sync::OnceLock;

// ============================================================================
// Exception Arguments
// ============================================================================

/// Lazy exception arguments.
///
/// Exception args are stored lazily to avoid allocation for exceptions
/// that don't need their args (e.g., control-flow exceptions).
#[derive(Clone)]
pub struct ExceptionArgs {
    /// The args tuple contents.
    values: Box<[Value]>,
}

impl ExceptionArgs {
    /// Creates empty args (no values).
    pub fn empty() -> Self {
        Self {
            values: Box::new([]),
        }
    }

    /// Creates args from a single value.
    pub fn single(value: Value) -> Self {
        Self {
            values: Box::new([value]),
        }
    }

    /// Creates args from a slice of values.
    pub fn from_slice(values: &[Value]) -> Self {
        Self {
            values: values.into(),
        }
    }

    /// Creates args from an iterator.
    pub fn from_iter(iter: impl IntoIterator<Item = Value>) -> Self {
        Self {
            values: iter.into_iter().collect(),
        }
    }

    /// Returns true if there are no args.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns the number of args.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns the args as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[Value] {
        &self.values
    }

    /// Returns the first arg, if any.
    #[inline]
    pub fn first(&self) -> Option<&Value> {
        self.values.first()
    }
}

impl Default for ExceptionArgs {
    fn default() -> Self {
        Self::empty()
    }
}

impl fmt::Debug for ExceptionArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ExceptionArgs")
            .field(&self.values.len())
            .finish()
    }
}

// ============================================================================
// Exception Reference
// ============================================================================

/// Reference to an exception object.
///
/// This is the handle used for exception chaining (__cause__, __context__).
pub type ExceptionRef = Arc<ExceptionObject>;

// ============================================================================
// Exception Object
// ============================================================================

/// Python exception object.
///
/// Represents an active exception instance with lazy args and traceback.
/// The design minimizes allocations for common cases like StopIteration.
pub struct ExceptionObject {
    // ═══════════════════════════════════════════════════════════════════════
    // Compact Header (2 bytes total)
    // ═══════════════════════════════════════════════════════════════════════
    /// Exception type identifier.
    type_id: ExceptionTypeId,

    /// Exception state flags.
    flags: ExceptionFlags,

    // ═══════════════════════════════════════════════════════════════════════
    // Lazy Fields (allocated on demand)
    // ═══════════════════════════════════════════════════════════════════════
    /// Lazy exception arguments (the args tuple).
    args: OnceLock<ExceptionArgs>,

    /// Lazy traceback.
    traceback: OnceLock<TracebackObject>,

    /// Lazy message (first arg as string, or formatted).
    message: OnceLock<Arc<str>>,

    // ═══════════════════════════════════════════════════════════════════════
    // Exception Chaining (PEP 3134)
    // ═══════════════════════════════════════════════════════════════════════
    /// Explicit cause (`raise X from Y`).
    cause: Option<ExceptionRef>,

    /// Implicit context (exception that was being handled).
    context: Option<ExceptionRef>,
}

impl ExceptionObject {
    // ════════════════════════════════════════════════════════════════════════
    // Constructors
    // ════════════════════════════════════════════════════════════════════════

    /// Creates a new exception with just a type (no args, no traceback).
    ///
    /// This is the minimal allocation path for control-flow exceptions.
    #[inline]
    pub fn new(type_id: ExceptionTypeId) -> Self {
        Self {
            type_id,
            flags: ExceptionFlags::new_exception(),
            args: OnceLock::new(),
            traceback: OnceLock::new(),
            message: OnceLock::new(),
            cause: None,
            context: None,
        }
    }

    /// Creates an exception with a message.
    pub fn with_message(type_id: ExceptionTypeId, message: impl Into<Arc<str>>) -> Self {
        let msg = message.into();
        let mut exc = Self::new(type_id);
        exc.flags = exc.flags.set_normalized().set_has_args();
        exc.message.set(Arc::clone(&msg)).ok();
        exc.args.set(ExceptionArgs::single(Value::none())).ok(); // TODO: proper string value
        exc
    }

    /// Creates an exception with args.
    pub fn with_args(type_id: ExceptionTypeId, args: ExceptionArgs) -> Self {
        let mut exc = Self::new(type_id);
        exc.flags = exc.flags.set_normalized().set_has_args();
        exc.args.set(args).ok();
        exc
    }

    /// Creates a flyweight exception (singleton, no args).
    ///
    /// Used for StopIteration, GeneratorExit, etc.
    pub(super) fn flyweight(type_id: ExceptionTypeId) -> Self {
        Self {
            type_id,
            flags: ExceptionFlags::flyweight(),
            args: OnceLock::new(),
            traceback: OnceLock::new(),
            message: OnceLock::new(),
            cause: None,
            context: None,
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // Type Information
    // ════════════════════════════════════════════════════════════════════════

    /// Returns the exception type ID.
    #[inline(always)]
    pub const fn type_id(&self) -> ExceptionTypeId {
        self.type_id
    }

    /// Returns the exception type name.
    #[inline]
    pub const fn type_name(&self) -> &'static str {
        self.type_id.name()
    }

    /// Returns true if this exception is an instance of the given type.
    #[inline]
    pub fn is_instance(&self, type_id: ExceptionTypeId) -> bool {
        self.type_id == type_id
    }

    /// Returns true if this exception is a subclass of the given type.
    #[inline]
    pub fn is_subclass(&self, base: ExceptionTypeId) -> bool {
        self.type_id.is_subclass_of(base)
    }

    // ════════════════════════════════════════════════════════════════════════
    // Flags
    // ════════════════════════════════════════════════════════════════════════

    /// Returns the exception flags.
    #[inline(always)]
    pub const fn flags(&self) -> ExceptionFlags {
        self.flags
    }

    /// Returns true if this is a flyweight exception.
    #[inline]
    pub const fn is_flyweight(&self) -> bool {
        self.flags.is_flyweight()
    }

    /// Returns true if the exception has been normalized.
    #[inline]
    pub const fn is_normalized(&self) -> bool {
        self.flags.is_normalized()
    }

    // ════════════════════════════════════════════════════════════════════════
    // Args Access (Lazy)
    // ════════════════════════════════════════════════════════════════════════

    /// Returns the exception args, if set.
    #[inline]
    pub fn args(&self) -> Option<&ExceptionArgs> {
        self.args.get()
    }

    /// Returns the exception args, initializing to empty if needed.
    pub fn args_or_empty(&self) -> &ExceptionArgs {
        self.args.get_or_init(ExceptionArgs::empty)
    }

    /// Sets the exception args.
    ///
    /// Returns Err if args were already set.
    pub fn set_args(&self, args: ExceptionArgs) -> Result<(), ExceptionArgs> {
        self.args.set(args)
    }

    // ════════════════════════════════════════════════════════════════════════
    // Traceback Access (Lazy)
    // ════════════════════════════════════════════════════════════════════════

    /// Returns the traceback, if set.
    #[inline]
    pub fn traceback(&self) -> Option<&TracebackObject> {
        self.traceback.get()
    }

    /// Returns a mutable reference to the traceback, if set.
    #[inline]
    pub fn traceback_mut(&mut self) -> Option<&mut TracebackObject> {
        self.traceback.get_mut()
    }

    /// Returns the traceback, initializing to empty if needed.
    pub fn traceback_or_empty(&self) -> &TracebackObject {
        self.traceback.get_or_init(TracebackObject::empty)
    }

    /// Sets the traceback.
    pub fn set_traceback(&self, tb: TracebackObject) -> Result<(), TracebackObject> {
        self.traceback.set(tb)
    }

    /// Returns true if the exception has a traceback.
    #[inline]
    pub fn has_traceback(&self) -> bool {
        self.traceback.get().is_some()
    }

    // ════════════════════════════════════════════════════════════════════════
    // Message Access (Lazy)
    // ════════════════════════════════════════════════════════════════════════

    /// Returns the exception message.
    pub fn message(&self) -> Arc<str> {
        self.message
            .get_or_init(|| {
                // Try to get message from first arg
                if let Some(args) = self.args.get() {
                    if let Some(_first) = args.first() {
                        // TODO: Convert first arg to string
                        return Arc::from("");
                    }
                }
                Arc::from("")
            })
            .clone()
    }

    // ════════════════════════════════════════════════════════════════════════
    // Exception Chaining (PEP 3134)
    // ════════════════════════════════════════════════════════════════════════

    /// Returns the explicit cause (__cause__).
    #[inline]
    pub fn cause(&self) -> Option<&ExceptionRef> {
        self.cause.as_ref()
    }

    /// Sets the explicit cause (`raise X from Y`).
    pub fn set_cause(&mut self, cause: ExceptionRef) {
        self.cause = Some(cause);
        self.flags = self.flags.set_has_cause().set_suppress_context();
    }

    /// Clears the explicit cause.
    pub fn clear_cause(&mut self) {
        self.cause = None;
        self.flags = self.flags.clear_has_cause().clear_suppress_context();
    }

    /// Returns the implicit context (__context__).
    #[inline]
    pub fn context(&self) -> Option<&ExceptionRef> {
        self.context.as_ref()
    }

    /// Sets the implicit context.
    pub fn set_context(&mut self, context: ExceptionRef) {
        self.context = Some(context);
        self.flags = self.flags.set_has_context();
    }

    /// Returns true if __suppress_context__ is True.
    #[inline]
    pub const fn suppress_context(&self) -> bool {
        self.flags.suppress_context()
    }

    // ════════════════════════════════════════════════════════════════════════
    // Formatting
    // ════════════════════════════════════════════════════════════════════════

    /// Formats the exception as a string.
    pub fn format(&self) -> String {
        let msg = self.message();
        if msg.is_empty() {
            self.type_name().to_string()
        } else {
            format!("{}: {}", self.type_name(), msg)
        }
    }

    /// Formats the full exception with traceback.
    pub fn format_with_traceback(&mut self) -> String {
        let mut output = String::new();

        if let Some(tb) = self.traceback_mut() {
            output.push_str(&tb.format());
        }

        output.push_str(&self.format());
        output.push('\n');
        output
    }
}

impl fmt::Debug for ExceptionObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExceptionObject")
            .field("type_id", &self.type_id)
            .field("flags", &self.flags)
            .field("has_args", &self.args.get().is_some())
            .field("has_traceback", &self.traceback.get().is_some())
            .field("has_cause", &self.cause.is_some())
            .field("has_context", &self.context.is_some())
            .finish()
    }
}

impl fmt::Display for ExceptionObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests;
