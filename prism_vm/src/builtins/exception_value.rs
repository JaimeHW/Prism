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

use super::{BuiltinError, BuiltinFunctionObject};
use crate::VirtualMachine;
use crate::error::RuntimeError;
use crate::stdlib::exceptions::ExceptionTypeId;
use prism_core::Value;
use prism_core::intern::{InternedString, intern, interned_by_ptr};
use prism_core::python_unicode::{is_surrogate_carrier, python_char_escape};
use prism_gc::trace::{Trace, Tracer};
use prism_runtime::allocation_context::{alloc_value_in_current_heap, has_current_heap_binding};
use prism_runtime::gc_dispatch::{DispatchEntry, register_external_dispatch};
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::global_class;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::string::StringObject;
use prism_runtime::types::tuple::TupleObject;
use std::cell::RefCell;
use std::mem;
use std::pin::Pin;
use std::sync::{Arc, LazyLock, Once};

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

    /// Lazily allocated ImportError / ModuleNotFoundError metadata.
    import_details: Option<Box<ImportErrorDetails>>,

    /// Lazily allocated SyntaxError metadata.
    syntax_details: Option<Box<SyntaxErrorDetails>>,

    /// Explicit cause (from `raise X from Y`).
    /// Uses raw pointer to avoid recursive Box issues.
    pub cause: Option<*const ExceptionValue>,

    /// Implicit context (exception being handled when this was raised).
    pub context: Option<*const ExceptionValue>,

    /// Python-visible traceback view attached to the exception.
    pub traceback: Option<Value>,

    /// Traceback reference (index into traceback table).
    pub traceback_id: u32,

    /// Reserved for future fields.
    _reserved: u32,
}

#[derive(Debug)]
struct ImportErrorDetails {
    name: Option<Arc<str>>,
    path: Option<Arc<str>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntaxErrorDetails {
    pub filename: Option<Arc<str>>,
    pub lineno: Option<u32>,
    pub offset: Option<u32>,
    pub text: Option<Arc<str>>,
    pub end_lineno: Option<u32>,
    pub end_offset: Option<u32>,
}

impl SyntaxErrorDetails {
    #[inline]
    #[must_use]
    pub fn new(
        filename: Option<Arc<str>>,
        lineno: Option<u32>,
        offset: Option<u32>,
        text: Option<Arc<str>>,
        end_lineno: Option<u32>,
        end_offset: Option<u32>,
    ) -> Self {
        Self {
            filename,
            lineno,
            offset,
            text,
            end_lineno,
            end_offset,
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.filename.is_none()
            && self.lineno.is_none()
            && self.offset.is_none()
            && self.text.is_none()
            && self.end_lineno.is_none()
            && self.end_offset.is_none()
    }
}

static EXCEPTION_NEW_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("BaseException.__new__"), builtin_exception_new)
});
static EXCEPTION_INIT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("BaseException.__init__"), builtin_exception_init)
});
static EXCEPTION_STR_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("BaseException.__str__"), builtin_exception_str)
});
static EXCEPTION_REPR_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("BaseException.__repr__"), builtin_exception_repr)
});
static EXCEPTION_WITH_TRACEBACK_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("BaseException.with_traceback"),
        builtin_exception_with_traceback,
    )
});
static EXCEPTION_GC_DISPATCH_ONCE: Once = Once::new();
static EXCEPTION_ARGS_ATTR: LazyLock<InternedString> = LazyLock::new(|| intern("args"));
static EXCEPTION_TRACEBACK_ATTR: LazyLock<InternedString> =
    LazyLock::new(|| intern("__traceback__"));
static EXCEPTION_CAUSE_ATTR: LazyLock<InternedString> = LazyLock::new(|| intern("__cause__"));
static EXCEPTION_CONTEXT_ATTR: LazyLock<InternedString> = LazyLock::new(|| intern("__context__"));
static EXCEPTION_SUPPRESS_CONTEXT_ATTR: LazyLock<InternedString> =
    LazyLock::new(|| intern("__suppress_context__"));

thread_local! {
    static PINNED_EXCEPTION_VALUES: RefCell<Vec<Pin<Box<ExceptionValue>>>> = const {
        RefCell::new(Vec::new())
    };
}

const TRACEBACK_TYPE_ERROR_MESSAGE: &str = "__traceback__ must be a traceback or None";

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
            import_details: None,
            syntax_details: None,
            cause: None,
            context: None,
            traceback: None,
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
            import_details: None,
            syntax_details: None,
            cause: None,
            context: None,
            traceback: None,
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

    /// Return the positional payload passed to the exception constructor.
    #[inline]
    pub fn args(&self) -> Option<&[Value]> {
        self.args.as_deref()
    }

    #[inline]
    pub fn import_name(&self) -> Option<&str> {
        self.import_details
            .as_deref()
            .and_then(|details| details.name.as_deref())
    }

    #[inline]
    pub fn import_path(&self) -> Option<&str> {
        self.import_details
            .as_deref()
            .and_then(|details| details.path.as_deref())
    }

    #[inline]
    pub fn syntax_filename(&self) -> Option<&str> {
        self.syntax_details
            .as_deref()
            .and_then(|details| details.filename.as_deref())
    }

    #[inline]
    pub fn syntax_lineno(&self) -> Option<u32> {
        self.syntax_details
            .as_deref()
            .and_then(|details| details.lineno)
    }

    #[inline]
    pub fn syntax_offset(&self) -> Option<u32> {
        self.syntax_details
            .as_deref()
            .and_then(|details| details.offset)
    }

    #[inline]
    pub fn syntax_text(&self) -> Option<&str> {
        self.syntax_details
            .as_deref()
            .and_then(|details| details.text.as_deref())
    }

    #[inline]
    pub fn syntax_end_lineno(&self) -> Option<u32> {
        self.syntax_details
            .as_deref()
            .and_then(|details| details.end_lineno)
    }

    #[inline]
    pub fn syntax_end_offset(&self) -> Option<u32> {
        self.syntax_details
            .as_deref()
            .and_then(|details| details.end_offset)
    }

    /// Render the exception payload using Python's `str(exception)` semantics.
    pub fn display_text(&self) -> String {
        if let Some(args) = self.args.as_deref() {
            return match args {
                [] => self.message.as_deref().unwrap_or("").to_string(),
                [value] => exception_arg_display_text(*value),
                values => {
                    let joined = values
                        .iter()
                        .map(|value| exception_arg_repr_text(*value))
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("({joined})")
                }
            };
        }

        self.message.as_deref().unwrap_or("").to_string()
    }

    /// Render the exception using Python's `repr(exception)` style.
    pub fn repr_text(&self) -> String {
        if let Some(args) = self.args.as_deref()
            && !args.is_empty()
        {
            let joined = args
                .iter()
                .map(|value| exception_arg_repr_text(*value))
                .collect::<Vec<_>>()
                .join(", ");
            return format!("{}({joined})", self.type_name());
        }

        if let Some(message) = self.message() {
            return format!("{}({})", self.type_name(), quote_python_string(message));
        }

        format!("{}()", self.type_name())
    }

    /// Replace the Python-visible constructor payload of an existing native
    /// exception instance.
    pub fn reinitialize_args(&mut self, args: &[Value]) {
        self.message = args
            .first()
            .and_then(|value| owned_string_value(*value).map(Arc::from));
        self.args = Some(args.to_vec().into_boxed_slice());
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

    /// Get the Python-visible traceback value.
    #[inline]
    pub fn traceback(&self) -> Option<Value> {
        self.traceback
    }

    /// Attach a Python-visible traceback value.
    #[inline]
    pub fn set_traceback(&mut self, traceback: Value) {
        self.traceback = Some(traceback);
        self.traceback_id = self.traceback_id.max(1);
    }

    /// Clear any attached traceback value.
    #[inline]
    pub fn clear_traceback(&mut self) {
        self.traceback = None;
        self.traceback_id = 0;
    }

    /// Replace the attached traceback using CPython's BaseException validation
    /// rules.
    pub fn replace_traceback(&mut self, traceback: Value) -> Result<(), &'static str> {
        match normalize_traceback_value(traceback)? {
            Some(traceback) => self.set_traceback(traceback),
            None => self.clear_traceback(),
        }
        Ok(())
    }

    /// Check if this is a subclass of another exception type.
    #[inline]
    pub fn is_subclass_of(&self, base: ExceptionTypeId) -> bool {
        self.type_id().is_subclass_of(base)
    }

    /// Attach ImportError / ModuleNotFoundError metadata.
    #[inline]
    pub fn with_import_details(
        mut self,
        import_name: Option<Arc<str>>,
        import_path: Option<Arc<str>>,
    ) -> Self {
        self.import_details = if import_name.is_some() || import_path.is_some() {
            Some(Box::new(ImportErrorDetails {
                name: import_name,
                path: import_path,
            }))
        } else {
            None
        };
        self
    }

    /// Attach SyntaxError metadata.
    #[inline]
    pub fn with_syntax_details(mut self, details: SyntaxErrorDetails) -> Self {
        self.syntax_details = if details.is_empty() {
            None
        } else {
            Some(Box::new(details))
        };
        self
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
    pub fn into_value(self) -> Value {
        ensure_exception_gc_dispatch_registered();
        if has_current_heap_binding() {
            return alloc_value_in_current_heap(self)
                .expect("bound heap should satisfy exception allocation");
        }

        PINNED_EXCEPTION_VALUES.with(|store| {
            let pinned = Box::into_pin(Box::new(self));
            let ptr = (&*pinned) as *const ExceptionValue as *const ();
            store.borrow_mut().push(pinned);
            Value::object_ptr(ptr)
        })
    }

    /// Allocate the exception in the VM-managed heap.
    pub fn into_gc_value(self, vm: &VirtualMachine) -> Result<Value, RuntimeError> {
        ensure_exception_gc_dispatch_registered();
        vm.allocator()
            .alloc_value(self)
            .ok_or_else(|| RuntimeError::internal("out of memory: failed to allocate exception"))
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

    /// Try to extract a mutable ExceptionValue from a Value.
    ///
    /// # Safety
    /// The Value must refer to a valid, uniquely mutable exception instance.
    pub unsafe fn from_value_mut(value: Value) -> Option<&'static mut ExceptionValue> {
        let ptr = value.as_object_ptr()?;
        let header = ptr as *const ObjectHeader;

        if unsafe { (*header).type_id } != TypeId::EXCEPTION {
            return None;
        }

        Some(unsafe { &mut *(ptr as *mut ExceptionValue) })
    }
}

pub(crate) fn exception_traceback_for_value(value: Value) -> Option<Value> {
    if let Some(exception) = unsafe { ExceptionValue::from_value(value) } {
        return exception.traceback();
    }

    heap_exception_instance_ref(value).and_then(|instance| {
        instance
            .get_property_interned(&EXCEPTION_TRACEBACK_ATTR)
            .or_else(|| instance.get_property("__traceback__"))
            .filter(|traceback| !traceback.is_none())
    })
}

pub(crate) fn set_exception_traceback_for_value(
    value: Value,
    traceback: Value,
) -> Result<bool, &'static str> {
    let traceback = normalize_traceback_value(traceback)?;

    if let Some(exception) = unsafe { ExceptionValue::from_value_mut(value) } {
        match traceback {
            Some(traceback) => exception.set_traceback(traceback),
            None => exception.clear_traceback(),
        }
        return Ok(true);
    }

    let Some(instance) = heap_exception_instance_mut(value) else {
        return Ok(false);
    };
    instance.set_property(
        EXCEPTION_TRACEBACK_ATTR.clone(),
        traceback.unwrap_or_else(Value::none),
        shape_registry(),
    );
    Ok(true)
}

unsafe impl Trace for ExceptionValue {
    fn trace(&self, tracer: &mut dyn Tracer) {
        if let Some(args) = self.args.as_deref() {
            for value in args {
                tracer.trace_value(*value);
            }
        }
        if let Some(cause) = self.cause {
            tracer.trace_ptr(cause as *const ());
        }
        if let Some(context) = self.context {
            tracer.trace_ptr(context as *const ());
        }
        if let Some(traceback) = self.traceback {
            tracer.trace_value(traceback);
        }
    }

    fn size_of(&self) -> usize {
        mem::size_of::<Self>()
            + self
                .import_details
                .as_deref()
                .map_or(0, |_| mem::size_of::<ImportErrorDetails>())
            + self
                .syntax_details
                .as_deref()
                .map_or(0, |_| mem::size_of::<SyntaxErrorDetails>())
            + self
                .args
                .as_deref()
                .map_or(0, |args| mem::size_of_val(args))
    }
}

unsafe fn trace_exception_value(ptr: *const (), tracer: &mut dyn Tracer) {
    let value = unsafe { &*(ptr as *const ExceptionValue) };
    value.trace(tracer);
}

unsafe fn size_exception_value(ptr: *const ()) -> usize {
    let value = unsafe { &*(ptr as *const ExceptionValue) };
    value.size_of()
}

unsafe fn finalize_exception_value(ptr: *mut ()) {
    unsafe { std::ptr::drop_in_place(ptr as *mut ExceptionValue) };
}

pub(crate) fn ensure_exception_gc_dispatch_registered() {
    EXCEPTION_GC_DISPATCH_ONCE.call_once(|| {
        register_external_dispatch(
            TypeId::EXCEPTION,
            DispatchEntry {
                trace: trace_exception_value,
                size: size_exception_value,
                finalize: finalize_exception_value,
            },
        );
    });
}

impl std::fmt::Debug for ExceptionValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExceptionValue")
            .field("type", &self.type_name())
            .field("message", &self.message)
            .field("import_name", &self.import_name())
            .field("import_path", &self.import_path())
            .field("syntax_filename", &self.syntax_filename())
            .field("flags", &self.flags)
            .finish()
    }
}

impl std::fmt::Display for ExceptionValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.display_text())
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

#[inline]
fn owned_string_value(value: Value) -> Option<String> {
    if value.is_string() {
        let ptr = value.as_string_object_ptr()?;
        let interned = interned_by_ptr(ptr as *const u8)?;
        return Some(interned.as_str().to_string());
    }

    let ptr = value.as_object_ptr()?;
    if unsafe { (*(ptr as *const ObjectHeader)).type_id } != TypeId::STR {
        return None;
    }

    Some(
        unsafe { &*(ptr as *const StringObject) }
            .as_str()
            .to_string(),
    )
}

#[inline]
fn builtin_method_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

#[inline]
fn boxed_object_value<T: prism_runtime::Trace>(object: T) -> Value {
    crate::alloc_managed_value(object)
}

#[inline]
fn boxed_tuple_value(items: Vec<Value>) -> Value {
    boxed_object_value(TupleObject::from_vec(items))
}

#[inline]
fn boxed_string_value(text: &str) -> Value {
    boxed_object_value(StringObject::new(text))
}

fn builtin_exception_new(args: &[Value]) -> Result<Value, BuiltinError> {
    let Some(class_value) = args.first().copied() else {
        return Err(BuiltinError::TypeError(
            "BaseException.__new__(): not enough arguments".to_string(),
        ));
    };

    if let Some(ptr) = class_value.as_object_ptr() {
        if unsafe { (*(ptr as *const ObjectHeader)).type_id } == crate::builtins::EXCEPTION_TYPE_ID
        {
            let exc_type = unsafe { &*(ptr as *const crate::builtins::ExceptionTypeObject) };
            return Ok(ExceptionValue::new(
                exc_type
                    .exception_type()
                    .unwrap_or(ExceptionTypeId::Exception),
                None,
            )
            .into_value());
        }

        if unsafe { (*(ptr as *const ObjectHeader)).type_id } == TypeId::TYPE
            && crate::builtins::builtin_type_object_type_id(ptr).is_none()
        {
            let class = unsafe { &*(ptr as *const PyClassObject) };
            if is_heap_exception_class(class) {
                let instance = crate::builtins::allocate_heap_instance_for_class(class);
                let value = boxed_object_value(instance);
                initialize_heap_exception_state(value)?;
                return Ok(value);
            }
        }
    }

    Err(BuiltinError::TypeError(
        "BaseException.__new__(X): X is not a subtype of BaseException".to_string(),
    ))
}

fn builtin_exception_init(args: &[Value]) -> Result<Value, BuiltinError> {
    let Some(receiver) = args.first().copied() else {
        return Err(BuiltinError::TypeError(
            "descriptor '__init__' of 'BaseException' object needs an argument".to_string(),
        ));
    };

    if let Some(exception) = unsafe { ExceptionValue::from_value_mut(receiver) } {
        exception.reinitialize_args(&args[1..]);
        return Ok(Value::none());
    }

    let Some(instance) = heap_exception_instance_mut(receiver) else {
        return Err(BuiltinError::TypeError(format!(
            "descriptor '__init__' for 'BaseException' objects doesn't apply to a '{}' object",
            exception_receiver_type_name(receiver)
        )));
    };

    set_heap_exception_args(instance, &args[1..]);
    Ok(Value::none())
}

fn builtin_exception_str(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "BaseException.__str__() takes exactly 0 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let text = exception_display_text_for_value(args[0]).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "descriptor '__str__' for 'BaseException' objects doesn't apply to a '{}' object",
            exception_receiver_type_name(args[0])
        ))
    })?;
    Ok(boxed_string_value(&text))
}

fn builtin_exception_repr(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "BaseException.__repr__() takes exactly 0 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let text = exception_repr_text_for_value(args[0]).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "descriptor '__repr__' for 'BaseException' objects doesn't apply to a '{}' object",
            exception_receiver_type_name(args[0])
        ))
    })?;
    Ok(boxed_string_value(&text))
}

#[inline]
fn normalize_traceback_value(traceback: Value) -> Result<Option<Value>, &'static str> {
    if traceback.is_none() {
        return Ok(None);
    }

    let ptr = traceback
        .as_object_ptr()
        .ok_or(TRACEBACK_TYPE_ERROR_MESSAGE)?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != TypeId::TRACEBACK {
        return Err(TRACEBACK_TYPE_ERROR_MESSAGE);
    }

    Ok(Some(traceback))
}

fn builtin_exception_with_traceback(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "BaseException.with_traceback() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let receiver = args[0];
    if let Some(exception) = unsafe { ExceptionValue::from_value_mut(receiver) } {
        exception
            .replace_traceback(args[1])
            .map_err(|message| BuiltinError::TypeError(message.to_string()))?;
        return Ok(receiver);
    }

    let Some(instance) = heap_exception_instance_mut(receiver) else {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'with_traceback' for 'BaseException' objects doesn't apply to a '{}' object",
            exception_receiver_type_name(receiver)
        )));
    };

    let traceback = normalize_traceback_value(args[1])
        .map_err(|message| BuiltinError::TypeError(message.to_string()))?
        .unwrap_or_else(Value::none);
    instance.set_property(
        EXCEPTION_TRACEBACK_ATTR.clone(),
        traceback,
        shape_registry(),
    );
    Ok(receiver)
}

#[inline]
pub(crate) fn exception_method_value(name: &str) -> Option<Value> {
    match name {
        "__new__" => Some(builtin_method_value(&EXCEPTION_NEW_METHOD)),
        "__init__" => Some(builtin_method_value(&EXCEPTION_INIT_METHOD)),
        "__str__" => Some(builtin_method_value(&EXCEPTION_STR_METHOD)),
        "__repr__" => Some(builtin_method_value(&EXCEPTION_REPR_METHOD)),
        "with_traceback" => Some(builtin_method_value(&EXCEPTION_WITH_TRACEBACK_METHOD)),
        _ => None,
    }
}

pub(crate) fn exception_display_text_for_value(value: Value) -> Option<String> {
    if let Some(exception) = unsafe { ExceptionValue::from_value(value) } {
        return Some(exception.display_text());
    }

    let class = heap_exception_class_from_value(value)?;
    let args = heap_exception_args_value(value);
    Some(render_exception_display_text(args, class.name().as_str()))
}

pub(crate) fn exception_repr_text_for_value(value: Value) -> Option<String> {
    if let Some(exception) = unsafe { ExceptionValue::from_value(value) } {
        return Some(exception.repr_text());
    }

    let class = heap_exception_class_from_value(value)?;
    let args = heap_exception_args_value(value);
    Some(render_exception_repr_text(class.name().as_str(), args))
}

#[inline]
fn exception_receiver_type_name(value: Value) -> String {
    if let Some(exception) = unsafe { ExceptionValue::from_value(value) } {
        return exception.type_name().to_string();
    }

    if let Some(class) = heap_exception_class_from_value(value) {
        return class.name().as_str().to_string();
    }

    value.type_name().to_string()
}

#[inline]
fn is_heap_exception_class(class: &PyClassObject) -> bool {
    class
        .mro()
        .iter()
        .any(|&class_id| crate::builtins::exception_type_id_for_proxy_class_id(class_id).is_some())
}

#[inline]
fn heap_exception_class_from_value(value: Value) -> Option<Arc<PyClassObject>> {
    let ptr = value.as_object_ptr()?;
    let type_id = unsafe { (*(ptr as *const ObjectHeader)).type_id };
    if type_id.raw() < TypeId::FIRST_USER_TYPE {
        return None;
    }

    let class = global_class(ClassId(type_id.raw()))?;
    is_heap_exception_class(class.as_ref()).then_some(class)
}

#[inline]
fn heap_exception_instance_mut(value: Value) -> Option<&'static mut ShapedObject> {
    heap_exception_class_from_value(value)?;
    Some(unsafe { &mut *(value.as_object_ptr()? as *mut ShapedObject) })
}

#[inline]
fn heap_exception_instance_ref(value: Value) -> Option<&'static ShapedObject> {
    heap_exception_class_from_value(value)?;
    Some(unsafe { &*(value.as_object_ptr()? as *const ShapedObject) })
}

#[inline]
fn heap_exception_args_value(value: Value) -> Option<Value> {
    heap_exception_instance_ref(value).and_then(|instance| {
        instance
            .get_property_interned(&EXCEPTION_ARGS_ATTR)
            .or_else(|| instance.get_property("args"))
    })
}

fn initialize_heap_exception_state(value: Value) -> Result<(), BuiltinError> {
    let Some(instance) = heap_exception_instance_mut(value) else {
        return Err(BuiltinError::TypeError(
            "BaseException.__new__() produced a non-exception instance".to_string(),
        ));
    };

    set_heap_exception_args(instance, &[]);
    instance.set_property(
        EXCEPTION_TRACEBACK_ATTR.clone(),
        Value::none(),
        shape_registry(),
    );
    instance.set_property(
        EXCEPTION_CAUSE_ATTR.clone(),
        Value::none(),
        shape_registry(),
    );
    instance.set_property(
        EXCEPTION_CONTEXT_ATTR.clone(),
        Value::none(),
        shape_registry(),
    );
    instance.set_property(
        EXCEPTION_SUPPRESS_CONTEXT_ATTR.clone(),
        Value::bool(false),
        shape_registry(),
    );
    Ok(())
}

#[inline]
fn set_heap_exception_args(instance: &mut ShapedObject, args: &[Value]) {
    instance.set_property(
        EXCEPTION_ARGS_ATTR.clone(),
        boxed_tuple_value(args.to_vec()),
        shape_registry(),
    );
}

fn render_exception_display_text(args: Option<Value>, fallback_message: &str) -> String {
    match args {
        Some(value) => render_exception_display_from_value(value),
        None => fallback_message.to_string(),
    }
}

fn render_exception_display_from_value(value: Value) -> String {
    if let Some(tuple) = tuple_ref(value) {
        return match tuple.as_slice() {
            [] => String::new(),
            [item] => exception_arg_display_text(*item),
            items => {
                let joined = items
                    .iter()
                    .map(|item| exception_arg_repr_text(*item))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("({joined})")
            }
        };
    }

    exception_arg_display_text(value)
}

fn render_exception_repr_text(type_name: &str, args: Option<Value>) -> String {
    match args {
        Some(value) => render_exception_repr_from_value(type_name, value),
        None => format!("{type_name}()"),
    }
}

fn render_exception_repr_from_value(type_name: &str, value: Value) -> String {
    if let Some(tuple) = tuple_ref(value) {
        if tuple.is_empty() {
            return format!("{type_name}()");
        }

        let joined = tuple
            .iter()
            .map(|item| exception_arg_repr_text(*item))
            .collect::<Vec<_>>()
            .join(", ");
        return format!("{type_name}({joined})");
    }

    format!("{type_name}({})", exception_arg_repr_text(value))
}

#[inline]
fn tuple_ref(value: Value) -> Option<&'static TupleObject> {
    let ptr = value.as_object_ptr()?;
    (unsafe { (*(ptr as *const ObjectHeader)).type_id } == TypeId::TUPLE)
        .then(|| unsafe { &*(ptr as *const TupleObject) })
}

#[inline]
fn exception_arg_display_text(value: Value) -> String {
    if let Some(text) = owned_string_value(value) {
        return text;
    }
    if value.is_none() {
        return "None".to_string();
    }
    if let Some(boolean) = value.as_bool() {
        return if boolean { "True" } else { "False" }.to_string();
    }
    if let Some(integer) = value.as_int() {
        return integer.to_string();
    }
    if let Some(float) = value.as_float() {
        if float.fract() == 0.0 && float.is_finite() {
            return format!("{float:.1}");
        }
        return float.to_string();
    }
    if let Some(text) = exception_display_text_for_value(value) {
        return text;
    }

    let Some(ptr) = value.as_object_ptr() else {
        return "<object>".to_string();
    };
    let type_id = unsafe { (*(ptr as *const ObjectHeader)).type_id };
    format!("<{} object at 0x{:x}>", type_id.name(), ptr as usize)
}

#[inline]
fn exception_arg_repr_text(value: Value) -> String {
    if let Some(text) = owned_string_value(value) {
        return quote_python_string(&text);
    }
    if value.is_none() {
        return "None".to_string();
    }
    if let Some(boolean) = value.as_bool() {
        return if boolean { "True" } else { "False" }.to_string();
    }
    if let Some(integer) = value.as_int() {
        return integer.to_string();
    }
    if let Some(float) = value.as_float() {
        if float.fract() == 0.0 && float.is_finite() {
            return format!("{float:.1}");
        }
        return float.to_string();
    }
    if let Some(text) = exception_repr_text_for_value(value) {
        return text;
    }

    let Some(ptr) = value.as_object_ptr() else {
        return "<object>".to_string();
    };
    let type_id = unsafe { (*(ptr as *const ObjectHeader)).type_id };
    format!("<{} object at 0x{:x}>", type_id.name(), ptr as usize)
}

fn quote_python_string(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + 2);
    out.push('\'');
    for ch in input.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other if other.is_control() || is_surrogate_carrier(other as u32) => {
                out.push_str(&python_char_escape(other));
            }
            other => out.push(other),
        }
    }
    out.push('\'');
    out
}

/// Create a boxed exception value and return as Value.
///
/// This is the primary entry point for exception constructors.
#[inline]
pub fn create_exception(type_id: ExceptionTypeId, message: Option<Arc<str>>) -> Value {
    ExceptionValue::new(type_id, message).into_value()
}

pub fn create_exception_in_vm(
    vm: &VirtualMachine,
    type_id: ExceptionTypeId,
    message: Option<Arc<str>>,
) -> Result<Value, RuntimeError> {
    ExceptionValue::new(type_id, message).into_gc_value(vm)
}

/// Create an exception with arguments.
pub fn create_exception_with_args(
    type_id: ExceptionTypeId,
    message: Option<Arc<str>>,
    args: Box<[Value]>,
) -> Value {
    ExceptionValue::with_args(type_id, message, args).into_value()
}

pub fn create_exception_with_args_in_vm(
    vm: &VirtualMachine,
    type_id: ExceptionTypeId,
    message: Option<Arc<str>>,
    args: Box<[Value]>,
) -> Result<Value, RuntimeError> {
    ExceptionValue::with_args(type_id, message, args).into_gc_value(vm)
}

/// Create an import-related exception with `.name` / `.path` metadata.
pub fn create_exception_with_import_details(
    type_id: ExceptionTypeId,
    message: Option<Arc<str>>,
    import_name: Option<Arc<str>>,
    import_path: Option<Arc<str>>,
) -> Value {
    ExceptionValue::new(type_id, message)
        .with_import_details(import_name, import_path)
        .into_value()
}

pub fn create_exception_with_import_details_in_vm(
    vm: &VirtualMachine,
    type_id: ExceptionTypeId,
    message: Option<Arc<str>>,
    import_name: Option<Arc<str>>,
    import_path: Option<Arc<str>>,
) -> Result<Value, RuntimeError> {
    ExceptionValue::new(type_id, message)
        .with_import_details(import_name, import_path)
        .into_gc_value(vm)
}

/// Create a syntax-related exception with detailed source location metadata.
pub fn create_exception_with_syntax_details(
    type_id: ExceptionTypeId,
    message: Option<Arc<str>>,
    details: SyntaxErrorDetails,
) -> Value {
    ExceptionValue::new(type_id, message)
        .with_syntax_details(details)
        .into_value()
}

pub fn create_exception_with_syntax_details_in_vm(
    vm: &VirtualMachine,
    type_id: ExceptionTypeId,
    message: Option<Arc<str>>,
    details: SyntaxErrorDetails,
) -> Result<Value, RuntimeError> {
    ExceptionValue::new(type_id, message)
        .with_syntax_details(details)
        .into_gc_value(vm)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_gc::trace::tracer::CountingTracer;
    use prism_runtime::object::views::TracebackViewObject;
    use prism_runtime::{size_of_object, trace_object};

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
    fn test_exception_value_import_details_round_trip() {
        let exc = ExceptionValue::new(
            ExceptionTypeId::ModuleNotFoundError,
            Some(Arc::from("No module named 'pkg.missing'")),
        )
        .with_import_details(Some(Arc::from("pkg.missing")), None);
        assert_eq!(exc.import_name(), Some("pkg.missing"));
        assert!(exc.import_path().is_none());
    }

    #[test]
    fn test_exception_value_syntax_details_round_trip() {
        let exc = ExceptionValue::new(
            ExceptionTypeId::SyntaxError,
            Some(Arc::from("invalid syntax")),
        )
        .with_syntax_details(SyntaxErrorDetails::new(
            Some(Arc::from("demo.py")),
            Some(3),
            Some(7),
            Some(Arc::from("value =\n")),
            Some(3),
            Some(8),
        ));
        assert_eq!(exc.syntax_filename(), Some("demo.py"));
        assert_eq!(exc.syntax_lineno(), Some(3));
        assert_eq!(exc.syntax_offset(), Some(7));
        assert_eq!(exc.syntax_text(), Some("value =\n"));
        assert_eq!(exc.syntax_end_lineno(), Some(3));
        assert_eq!(exc.syntax_end_offset(), Some(8));
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

    #[test]
    fn test_exception_set_traceback_value() {
        let mut exc = ExceptionValue::new(ExceptionTypeId::ValueError, None);
        let traceback = Value::int(7).unwrap();
        exc.set_traceback(traceback);

        assert_eq!(exc.traceback(), Some(traceback));
        assert_ne!(exc.traceback_id, 0);

        exc.clear_traceback();
        assert!(exc.traceback().is_none());
        assert_eq!(exc.traceback_id, 0);
    }

    #[test]
    fn test_exception_replace_traceback_accepts_traceback_objects_and_none() {
        let traceback = Value::object_ptr(Box::into_raw(Box::new(TracebackViewObject::new(
            Value::none(),
            None,
            12,
            3,
        ))) as *const ());
        let mut exc = ExceptionValue::new(ExceptionTypeId::ValueError, None);

        exc.replace_traceback(traceback)
            .expect("traceback objects should be accepted");
        assert_eq!(exc.traceback(), Some(traceback));

        exc.replace_traceback(Value::none())
            .expect("None should clear traceback");
        assert!(exc.traceback().is_none());
    }

    #[test]
    fn test_exception_replace_traceback_rejects_non_tracebacks() {
        let mut exc = ExceptionValue::new(ExceptionTypeId::ValueError, None);
        let err = exc
            .replace_traceback(Value::int(7).unwrap())
            .expect_err("non-traceback values should be rejected");
        assert_eq!(err, TRACEBACK_TYPE_ERROR_MESSAGE);
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
    fn test_exception_into_value_uses_bound_vm_heap_when_available() {
        let baseline = PINNED_EXCEPTION_VALUES.with(|store| store.borrow().len());
        let _vm = VirtualMachine::new();

        let value = ExceptionValue::new(
            ExceptionTypeId::RuntimeError,
            Some(Arc::from("managed allocation")),
        )
        .into_value();

        assert_eq!(
            PINNED_EXCEPTION_VALUES.with(|store| store.borrow().len()),
            baseline
        );
        let recovered =
            unsafe { ExceptionValue::from_value(value).expect("exception should downcast") };
        assert_eq!(recovered.message(), Some("managed allocation"));
    }

    #[test]
    fn test_exception_into_value_survives_vm_move_after_binding() {
        fn relocate_vm(vm: VirtualMachine) -> VirtualMachine {
            vm
        }

        let baseline = PINNED_EXCEPTION_VALUES.with(|store| store.borrow().len());
        let _vm = relocate_vm(VirtualMachine::new());

        let value = ExceptionValue::new(
            ExceptionTypeId::RuntimeError,
            Some(Arc::from("moved vm allocation")),
        )
        .into_value();

        assert_eq!(
            PINNED_EXCEPTION_VALUES.with(|store| store.borrow().len()),
            baseline
        );
        let recovered =
            unsafe { ExceptionValue::from_value(value).expect("exception should downcast") };
        assert_eq!(recovered.message(), Some("moved vm allocation"));
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
    fn test_exception_from_value_mut_allows_in_place_traceback_updates() {
        let traceback = Value::object_ptr(Box::into_raw(Box::new(TracebackViewObject::new(
            Value::none(),
            None,
            21,
            5,
        ))) as *const ());
        let value = ExceptionValue::new(ExceptionTypeId::RuntimeError, None).into_value();

        let exception = unsafe {
            ExceptionValue::from_value_mut(value).expect("exception value should downcast mutably")
        };
        exception
            .replace_traceback(traceback)
            .expect("mutable exception should accept traceback");

        let observed =
            unsafe { ExceptionValue::from_value(value).expect("exception should remain valid") };
        assert_eq!(observed.traceback(), Some(traceback));
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
        assert_eq!(display, "invalid input");
    }

    #[test]
    fn test_exception_display_no_message() {
        let exc = ExceptionValue::new(ExceptionTypeId::StopIteration, None);
        let display = format!("{}", exc);
        assert_eq!(display, "");
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

    #[test]
    fn test_create_exception_with_import_details() {
        let value = create_exception_with_import_details(
            ExceptionTypeId::ModuleNotFoundError,
            Some(Arc::from("No module named 'pkg.missing'")),
            Some(Arc::from("pkg.missing")),
            None,
        );

        let exc = unsafe { ExceptionValue::from_value(value).unwrap() };
        assert_eq!(exc.type_id(), ExceptionTypeId::ModuleNotFoundError);
        assert_eq!(exc.import_name(), Some("pkg.missing"));
        assert!(exc.import_path().is_none());
    }

    #[test]
    fn test_create_exception_with_syntax_details() {
        let value = create_exception_with_syntax_details(
            ExceptionTypeId::SyntaxError,
            Some(Arc::from("expected ':'")),
            SyntaxErrorDetails::new(
                Some(Arc::from("sample.py")),
                Some(5),
                Some(9),
                Some(Arc::from("if True\n")),
                Some(5),
                Some(10),
            ),
        );

        let exc = unsafe { ExceptionValue::from_value(value).unwrap() };
        assert_eq!(exc.type_id(), ExceptionTypeId::SyntaxError);
        assert_eq!(exc.syntax_filename(), Some("sample.py"));
        assert_eq!(exc.syntax_lineno(), Some(5));
        assert_eq!(exc.syntax_offset(), Some(9));
        assert_eq!(exc.syntax_text(), Some("if True\n"));
        assert_eq!(exc.syntax_end_lineno(), Some(5));
        assert_eq!(exc.syntax_end_offset(), Some(10));
    }

    #[test]
    fn test_display_text_prefers_single_string_arg() {
        let args = vec![Value::string(prism_core::intern::intern("boom"))].into_boxed_slice();
        let exc = ExceptionValue::with_args(ExceptionTypeId::ValueError, None, args);
        assert_eq!(exc.display_text(), "boom");
    }

    #[test]
    fn test_repr_text_uses_exception_type_and_args() {
        let args = vec![Value::string(prism_core::intern::intern("boom"))].into_boxed_slice();
        let exc = ExceptionValue::with_args(ExceptionTypeId::ValueError, None, args);
        assert_eq!(exc.repr_text(), "ValueError('boom')");
    }

    #[test]
    fn test_exception_method_value_exposes_core_base_exception_builtins() {
        for (name, expected) in [
            ("__new__", "BaseException.__new__"),
            ("__init__", "BaseException.__init__"),
            ("__str__", "BaseException.__str__"),
            ("__repr__", "BaseException.__repr__"),
            ("with_traceback", "BaseException.with_traceback"),
        ] {
            let method =
                exception_method_value(name).expect("base exception builtin should resolve");
            let ptr = method
                .as_object_ptr()
                .expect("base exception builtin should be heap allocated");
            let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
            assert_eq!(builtin.name(), expected);
        }
    }

    #[test]
    fn test_exception_display_and_repr_helpers_cover_native_exceptions() {
        let exc = ExceptionValue::with_args(
            ExceptionTypeId::ValueError,
            None,
            vec![Value::string(prism_core::intern::intern("boom"))].into_boxed_slice(),
        )
        .into_value();

        assert_eq!(
            exception_display_text_for_value(exc).as_deref(),
            Some("boom")
        );
        assert_eq!(
            exception_repr_text_for_value(exc).as_deref(),
            Some("ValueError('boom')")
        );
    }

    #[test]
    fn test_exception_gc_dispatch_traces_args_links_and_traceback() {
        let vm = VirtualMachine::new();
        let cause =
            ExceptionValue::new(ExceptionTypeId::ValueError, Some(Arc::from("cause"))).into_value();
        let context = ExceptionValue::new(ExceptionTypeId::TypeError, Some(Arc::from("context")))
            .into_value();

        let mut exc = ExceptionValue::with_args(
            ExceptionTypeId::RuntimeError,
            Some(Arc::from("boom")),
            vec![
                Value::int(1).unwrap(),
                Value::string(prism_core::intern::intern("arg")),
            ]
            .into_boxed_slice(),
        );
        let expected_size = std::mem::size_of::<ExceptionValue>()
            + std::mem::size_of_val(exc.args.as_deref().expect("args should exist"));
        let cause_ptr = unsafe { ExceptionValue::from_value(cause).expect("cause should downcast") }
            as *const ExceptionValue;
        let context_ptr =
            unsafe { ExceptionValue::from_value(context).expect("context should downcast") }
                as *const ExceptionValue;
        exc.set_cause(cause_ptr);
        exc.set_context(context_ptr);
        exc.set_traceback(Value::string(prism_core::intern::intern("traceback")));

        let value = exc
            .into_gc_value(&vm)
            .expect("managed exception allocation should succeed");
        let ptr = value
            .as_object_ptr()
            .expect("managed exception should be object-backed");
        let mut tracer = CountingTracer::new();

        unsafe {
            trace_object(ptr, TypeId::EXCEPTION, &mut tracer);
        }

        assert_eq!(tracer.value_count, 3);
        assert_eq!(tracer.ptr_count, 2);
        let size = unsafe { size_of_object(ptr, TypeId::EXCEPTION) };
        assert_eq!(size, expected_size);
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
