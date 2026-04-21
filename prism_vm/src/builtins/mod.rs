//! Core builtin registry and function infrastructure.
//!
//! Modular builtins organized by category for maintainability.
//! All builtins use static dispatch for maximum performance.

mod builtin_function;
mod exception_type;
mod exception_value;
mod exceptions;
mod execution;
mod functions;
mod introspect;
mod io;
mod iter_dispatch;
mod itertools;
mod numeric;
mod percent_format;
mod string;
mod type_reflection;
mod types;

pub use builtin_function::*;
pub use exception_type::*;
pub(crate) use exception_type::{
    exception_proxy_class_id_from_ptr, exception_type_attribute_value,
    exception_type_id_for_proxy_class_id,
};
pub use exception_value::*;
pub use exceptions::*;
pub use execution::*;
pub use functions::*;
pub use introspect::*;
pub use io::*;
pub use iter_dispatch::*;
pub use itertools::*;
pub use numeric::*;
pub(crate) use percent_format::percent_format_string;
pub use string::*;
pub(crate) use type_reflection::*;
pub(crate) use types::builtin_type_object_type_id;
pub use types::*;

use crate::error::{RuntimeError, RuntimeErrorKind};
use prism_core::Value;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::object::views::SingletonObject;
use rustc_hash::FxHashMap;
use std::sync::{Arc, LazyLock};

// =============================================================================
// BuiltinFn Type
// =============================================================================

/// Builtin function type.
///
/// Takes a slice of arguments and returns a Result.
/// Using Result allows proper error propagation.
pub type BuiltinFn = fn(&[Value]) -> Result<Value, BuiltinError>;
pub type VmBuiltinFn = fn(&mut crate::VirtualMachine, &[Value]) -> Result<Value, BuiltinError>;
pub type VmBuiltinKwFn =
    fn(&mut crate::VirtualMachine, &[Value], &[(&str, Value)]) -> Result<Value, BuiltinError>;

/// Error type for builtin functions.
#[derive(Debug, Clone)]
pub enum BuiltinError {
    /// Wrong number of arguments.
    TypeError(String),
    /// Value error (e.g., negative index).
    ValueError(String),
    /// Syntax error raised while parsing dynamically executed source.
    SyntaxError(String),
    /// Operating-system level failure.
    OSError(String),
    /// Import failure while loading a module or attribute.
    ImportError(String),
    /// Requested module does not exist.
    ModuleNotFoundError(String),
    /// Iterator exhaustion.
    StopIteration,
    /// Attribute error.
    AttributeError(String),
    /// Key error.
    KeyError(String),
    /// Index out of range.
    IndexError(String),
    /// Overflow error.
    OverflowError(String),
    /// Preserve a fully classified runtime exception raised from nested execution.
    Raised(RuntimeError),
    /// Not implemented.
    NotImplemented(String),
}

impl std::fmt::Display for BuiltinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuiltinError::TypeError(msg) => write!(f, "TypeError: {}", msg),
            BuiltinError::ValueError(msg) => write!(f, "ValueError: {}", msg),
            BuiltinError::SyntaxError(msg) => write!(f, "SyntaxError: {}", msg),
            BuiltinError::OSError(msg) => write!(f, "OSError: {}", msg),
            BuiltinError::ImportError(msg) => write!(f, "ImportError: {}", msg),
            BuiltinError::ModuleNotFoundError(msg) => {
                write!(f, "ModuleNotFoundError: {}", msg)
            }
            BuiltinError::StopIteration => write!(f, "StopIteration"),
            BuiltinError::AttributeError(msg) => write!(f, "AttributeError: {}", msg),
            BuiltinError::KeyError(msg) => write!(f, "KeyError: {}", msg),
            BuiltinError::IndexError(msg) => write!(f, "IndexError: {}", msg),
            BuiltinError::OverflowError(msg) => write!(f, "OverflowError: {}", msg),
            BuiltinError::Raised(err) => write!(f, "{err}"),
            BuiltinError::NotImplemented(msg) => write!(f, "NotImplementedError: {}", msg),
        }
    }
}

impl std::error::Error for BuiltinError {}

#[inline]
fn runtime_error_to_builtin_error(err: RuntimeError) -> BuiltinError {
    match err.kind {
        RuntimeErrorKind::TypeError { message } => BuiltinError::TypeError(message.to_string()),
        RuntimeErrorKind::UnsupportedOperandTypes { op, left, right } => BuiltinError::TypeError(
            format!("unsupported operand type(s) for {op}: '{left}' and '{right}'"),
        ),
        RuntimeErrorKind::NotCallable { type_name } => {
            BuiltinError::TypeError(format!("'{}' object is not callable", type_name))
        }
        RuntimeErrorKind::NotIterable { type_name } => {
            BuiltinError::TypeError(format!("'{}' object is not iterable", type_name))
        }
        RuntimeErrorKind::NotSubscriptable { type_name } => {
            BuiltinError::TypeError(format!("'{}' object is not subscriptable", type_name))
        }
        RuntimeErrorKind::AttributeError { type_name, attr } => BuiltinError::AttributeError(
            format!("'{}' object has no attribute '{}'", type_name, attr),
        ),
        RuntimeErrorKind::KeyError { key } => BuiltinError::KeyError(key.to_string()),
        RuntimeErrorKind::IndexError { index, length } => {
            BuiltinError::IndexError(format!("index {index} out of range for length {length}"))
        }
        RuntimeErrorKind::ValueError { message } => BuiltinError::ValueError(message.to_string()),
        RuntimeErrorKind::OverflowError { message } => {
            BuiltinError::OverflowError(message.to_string())
        }
        RuntimeErrorKind::StopIteration => BuiltinError::StopIteration,
        _ => BuiltinError::Raised(err),
    }
}

// =============================================================================
// BuiltinRegistry
// =============================================================================

/// Registry of builtin functions and values.
///
/// Uses FxHashMap for O(1) lookup by name.
/// Static function pointers ensure zero-cost dispatch.
#[derive(Default, Clone)]
pub struct BuiltinRegistry {
    /// Name to value mappings.
    entries: FxHashMap<Arc<str>, Value>,
    /// Function table for direct dispatch.
    functions: FxHashMap<Arc<str>, BuiltinFn>,
    /// VM-aware function table for builtins that need runtime execution context.
    vm_functions: FxHashMap<Arc<str>, VmBuiltinFn>,
    /// VM-aware function table for builtins that accept keyword arguments.
    vm_keyword_functions: FxHashMap<Arc<str>, VmBuiltinKwFn>,
}

static ELLIPSIS_SINGLETON: LazyLock<Value> =
    LazyLock::new(|| singleton_builtin_value(TypeId::ELLIPSIS));
static NOT_IMPLEMENTED_SINGLETON: LazyLock<Value> =
    LazyLock::new(|| singleton_builtin_value(TypeId::NOT_IMPLEMENTED));

#[inline]
fn singleton_builtin_value(type_id: TypeId) -> Value {
    let ptr = Box::leak(Box::new(SingletonObject::new(type_id))) as *mut SingletonObject;
    Value::object_ptr(ptr.cast_const().cast())
}

#[inline]
pub fn builtin_ellipsis_value() -> Value {
    *ELLIPSIS_SINGLETON
}

#[inline]
pub fn builtin_not_implemented_value() -> Value {
    *NOT_IMPLEMENTED_SINGLETON
}

impl BuiltinRegistry {
    /// Create a new empty registry.
    #[inline]
    pub fn new() -> Self {
        Self {
            entries: FxHashMap::default(),
            functions: FxHashMap::default(),
            vm_functions: FxHashMap::default(),
            vm_keyword_functions: FxHashMap::default(),
        }
    }

    /// Create registry with standard Python builtins.
    pub fn with_standard_builtins() -> Self {
        let mut registry = Self::new();

        // Register builtin constants
        registry.register_value("None", Value::none());
        registry.register_value("True", Value::bool(true));
        registry.register_value("False", Value::bool(false));
        registry.register_value("Ellipsis", builtin_ellipsis_value());
        registry.register_value("NotImplemented", builtin_not_implemented_value());

        // Register core functions
        registry.register_function_vm("len", functions::builtin_len_vm);
        registry.register_function("abs", functions::builtin_abs);
        registry.register_function_vm_kw("min", functions::builtin_min_vm_kw);
        registry.register_function_vm_kw("max", functions::builtin_max_vm_kw);
        registry.register_function_vm("sum", functions::builtin_sum_vm);
        registry.register_function("pow", functions::builtin_pow);
        registry.register_function("round", functions::builtin_round);
        registry.register_function("divmod", functions::builtin_divmod);
        registry.register_function("hash", functions::builtin_hash);
        registry.register_function("id", functions::builtin_id);
        registry.register_function("callable", functions::builtin_callable);
        registry.register_function("repr", functions::builtin_repr);
        registry.register_function("ascii", functions::builtin_ascii);

        // Register numeric formatting functions
        registry.register_function("bin", numeric::builtin_bin);
        registry.register_function("hex", numeric::builtin_hex);
        registry.register_function("oct", numeric::builtin_oct);
        registry.register_callable_type(
            "complex",
            prism_runtime::object::type_obj::TypeId::COMPLEX,
            numeric::builtin_complex,
        );

        // Register string utilities
        registry.register_function("ord", string::builtin_ord);
        registry.register_function("chr", string::builtin_chr);
        registry.register_function("format", string::builtin_format);

        // Register introspection functions
        registry.register_function_vm("dir", introspect::builtin_dir_vm);
        registry.register_function("vars", introspect::builtin_vars);
        registry.register_function("globals", introspect::builtin_globals);
        registry.register_function("locals", introspect::builtin_locals);
        registry.register_function("help", introspect::builtin_help);
        registry.register_function_vm("__import__", introspect::builtin_import_vm);

        // Register execution functions
        registry.register_function_vm("exec", execution::builtin_exec_vm);
        registry.register_function_vm("eval", execution::builtin_eval_vm);
        registry.register_function("compile", execution::builtin_compile);
        registry.register_function("breakpoint", execution::builtin_breakpoint);

        // Register iteration functions
        registry.register_function_vm("iter", itertools::builtin_iter_vm);
        registry.register_function_vm("next", itertools::builtin_next_vm);
        registry.register_function_vm("enumerate", itertools::builtin_enumerate_vm);
        registry.register_function("zip", itertools::builtin_zip);
        registry.register_function("map", itertools::builtin_map);
        registry.register_function("filter", itertools::builtin_filter);
        registry.register_function("reversed", itertools::builtin_reversed);
        registry.register_function_vm("sorted", itertools::builtin_sorted_vm);
        registry.register_function_vm("all", itertools::builtin_all_vm);
        registry.register_function_vm("any", itertools::builtin_any_vm);

        // Register I/O functions
        registry.register_function("print", io::builtin_print);
        registry.register_function("input", io::builtin_input);
        registry.register_function_vm_kw("open", io::builtin_open_vm_kw);

        // Register type constructors/converters as callable type objects.
        registry.register_callable_type(
            "int",
            prism_runtime::object::type_obj::TypeId::INT,
            types::builtin_int,
        );
        registry.register_callable_type(
            "float",
            prism_runtime::object::type_obj::TypeId::FLOAT,
            types::builtin_float,
        );
        registry.register_callable_type(
            "str",
            prism_runtime::object::type_obj::TypeId::STR,
            types::builtin_str,
        );
        registry.register_callable_type(
            "bool",
            prism_runtime::object::type_obj::TypeId::BOOL,
            types::builtin_bool,
        );
        registry.register_callable_type(
            "bytes",
            prism_runtime::object::type_obj::TypeId::BYTES,
            string::builtin_bytes,
        );
        registry.register_callable_type(
            "bytearray",
            prism_runtime::object::type_obj::TypeId::BYTEARRAY,
            string::builtin_bytearray,
        );
        registry.register_callable_type(
            "memoryview",
            prism_runtime::object::type_obj::TypeId::MEMORYVIEW,
            types::builtin_memoryview,
        );
        registry.register_callable_type(
            "list",
            prism_runtime::object::type_obj::TypeId::LIST,
            types::builtin_list,
        );
        registry.register_callable_type(
            "tuple",
            prism_runtime::object::type_obj::TypeId::TUPLE,
            types::builtin_tuple,
        );
        registry.register_callable_type(
            "dict",
            prism_runtime::object::type_obj::TypeId::DICT,
            types::builtin_dict,
        );
        registry.register_callable_type(
            "set",
            prism_runtime::object::type_obj::TypeId::SET,
            types::builtin_set,
        );
        registry.register_callable_type(
            "frozenset",
            prism_runtime::object::type_obj::TypeId::FROZENSET,
            types::builtin_frozenset,
        );
        registry.register_callable_type(
            "range",
            prism_runtime::object::type_obj::TypeId::RANGE,
            itertools::builtin_range,
        );
        registry.register_callable_type(
            "slice",
            prism_runtime::object::type_obj::TypeId::SLICE,
            types::builtin_slice,
        );
        registry.register_callable_type(
            "type",
            prism_runtime::object::type_obj::TypeId::TYPE,
            types::builtin_type,
        );
        registry.register_function("isinstance", types::builtin_isinstance);
        registry.register_function("issubclass", types::builtin_issubclass);
        registry.register_callable_type(
            "object",
            prism_runtime::object::type_obj::TypeId::OBJECT,
            types::builtin_object,
        );
        registry.register_value(
            "super",
            types::builtin_type_object_for_type_id(prism_runtime::object::type_obj::TypeId::SUPER),
        );
        registry.register_callable_type(
            "classmethod",
            prism_runtime::object::type_obj::TypeId::CLASSMETHOD,
            types::builtin_classmethod,
        );
        registry.register_callable_type(
            "staticmethod",
            prism_runtime::object::type_obj::TypeId::STATICMETHOD,
            types::builtin_staticmethod,
        );
        registry.register_callable_type(
            "property",
            prism_runtime::object::type_obj::TypeId::PROPERTY,
            types::builtin_property,
        );
        registry.register_function_vm("getattr", types::builtin_getattr_vm);
        registry.register_function_vm("setattr", types::builtin_setattr_vm);
        registry.register_function_vm("hasattr", types::builtin_hasattr_vm);
        registry.register_function_vm("delattr", types::builtin_delattr_vm);

        // Register exception type objects (as callable types, not functions)
        // This allows except ValueError: to match correctly via type_id extraction
        for (name, exc_type) in exception_type::EXCEPTION_TYPE_TABLE {
            let exc_type_obj: &exception_type::ExceptionTypeObject = exc_type;
            let ptr = exc_type_obj as *const exception_type::ExceptionTypeObject as *const ();
            registry
                .entries
                .insert(Arc::from(*name), Value::object_ptr(ptr));
        }

        for (name, class) in exception_type::SUPPLEMENTAL_EXCEPTION_CLASS_TABLE {
            let ptr = std::sync::Arc::as_ptr(&***class) as *const ();
            registry
                .entries
                .insert(Arc::from(*name), Value::object_ptr(ptr));
        }

        registry
    }

    /// Register a builtin value.
    #[inline]
    pub fn register_value(&mut self, name: impl Into<Arc<str>>, value: Value) {
        self.entries.insert(name.into(), value);
    }

    /// Register a builtin function.
    ///
    /// Creates a BuiltinFunctionObject wrapped as a Value for proper
    /// call dispatch via the Call opcode.
    #[inline]
    pub fn register_function(&mut self, name: impl Into<Arc<str>>, func: BuiltinFn) {
        let name = name.into();
        self.functions.insert(name.clone(), func);

        // Create a BuiltinFunctionObject on the heap and store as Value
        // TODO: Use GC allocator instead of Box::leak
        let builtin_obj = Box::new(BuiltinFunctionObject::new(name.clone(), func));
        let ptr = Box::leak(builtin_obj) as *mut BuiltinFunctionObject as *const ();
        self.entries.insert(name, Value::object_ptr(ptr));
    }

    /// Register a builtin function that needs VM context at call time.
    #[inline]
    pub fn register_function_vm(&mut self, name: impl Into<Arc<str>>, func: VmBuiltinFn) {
        let name = name.into();
        self.vm_functions.insert(name.clone(), func);

        let builtin_obj = Box::new(BuiltinFunctionObject::new_vm(name.clone(), func));
        let ptr = Box::leak(builtin_obj) as *mut BuiltinFunctionObject as *const ();
        self.entries.insert(name, Value::object_ptr(ptr));
    }

    /// Register a builtin function that needs VM context and accepts keyword arguments.
    #[inline]
    pub fn register_function_vm_kw(&mut self, name: impl Into<Arc<str>>, func: VmBuiltinKwFn) {
        let name = name.into();
        self.vm_keyword_functions.insert(name.clone(), func);

        let builtin_obj = Box::new(BuiltinFunctionObject::new_vm_kw(name.clone(), func));
        let ptr = Box::leak(builtin_obj) as *mut BuiltinFunctionObject as *const ();
        self.entries.insert(name, Value::object_ptr(ptr));
    }

    /// Register a builtin type object while preserving the direct-call
    /// constructor in the function table.
    #[inline]
    pub fn register_callable_type(
        &mut self,
        name: impl Into<Arc<str>>,
        type_id: prism_runtime::object::type_obj::TypeId,
        constructor: BuiltinFn,
    ) {
        let name = name.into();
        self.functions.insert(name.clone(), constructor);
        self.entries
            .insert(name, types::builtin_type_object_for_type_id(type_id));
    }

    /// Get a builtin value by name.
    #[inline]
    pub fn get(&self, name: &str) -> Option<Value> {
        self.entries.get(name).copied()
    }

    /// Get a builtin function by name.
    #[inline]
    pub fn get_function(&self, name: &str) -> Option<BuiltinFn> {
        self.functions.get(name).copied()
    }

    /// Get a VM-aware builtin function by name.
    #[inline]
    pub fn get_vm_function(&self, name: &str) -> Option<VmBuiltinFn> {
        self.vm_functions.get(name).copied()
    }

    /// Call a builtin function by name.
    #[inline]
    pub fn call(&self, name: &str, args: &[Value]) -> Result<Value, BuiltinError> {
        if let Some(func) = self.functions.get(name) {
            func(args)
        } else {
            Err(BuiltinError::AttributeError(format!(
                "builtin function '{}' not found",
                name
            )))
        }
    }

    /// Call a builtin function by name, supplying VM context when required.
    #[inline]
    pub fn call_with_vm(
        &self,
        vm: &mut crate::VirtualMachine,
        name: &str,
        args: &[Value],
    ) -> Result<Value, BuiltinError> {
        if let Some(func) = self.functions.get(name) {
            return func(args);
        }
        if let Some(func) = self.vm_functions.get(name) {
            return func(vm, args);
        }
        if let Some(func) = self.vm_keyword_functions.get(name) {
            return func(vm, args, &[]);
        }

        Err(BuiltinError::AttributeError(format!(
            "builtin function '{}' not found",
            name
        )))
    }

    /// Check if a name is a builtin.
    #[inline]
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Check if a name is a builtin function.
    #[inline]
    pub fn is_function(&self, name: &str) -> bool {
        self.functions.contains_key(name)
            || self.vm_functions.contains_key(name)
            || self.vm_keyword_functions.contains_key(name)
    }

    /// Iterate over all builtins.
    pub fn iter(&self) -> impl Iterator<Item = (&Arc<str>, &Value)> {
        self.entries.iter()
    }

    /// Get the number of registered builtins.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if registry is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl std::fmt::Debug for BuiltinRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BuiltinRegistry")
            .field("entries", &self.entries.len())
            .field("functions", &self.functions.len())
            .field("vm_functions", &self.vm_functions.len())
            .finish()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_registry() {
        let registry = BuiltinRegistry::with_standard_builtins();

        assert!(registry.get("None").unwrap().is_none());
        assert!(registry.get("True").unwrap().is_truthy());
        assert!(!registry.get("False").unwrap().is_truthy());
        assert!(registry.get("Ellipsis").unwrap().as_object_ptr().is_some());
        assert!(
            registry
                .get("NotImplemented")
                .unwrap()
                .as_object_ptr()
                .is_some()
        );
    }

    #[test]
    fn test_builtin_registry_singletons_have_runtime_type_ids() {
        let registry = BuiltinRegistry::with_standard_builtins();

        let ellipsis = registry
            .get("Ellipsis")
            .expect("Ellipsis should be registered");
        let not_implemented = registry
            .get("NotImplemented")
            .expect("NotImplemented should be registered");

        let ellipsis_ptr = ellipsis
            .as_object_ptr()
            .expect("Ellipsis should be a heap singleton");
        let not_implemented_ptr = not_implemented
            .as_object_ptr()
            .expect("NotImplemented should be a heap singleton");

        let ellipsis_type =
            unsafe { (*(ellipsis_ptr as *const prism_runtime::object::ObjectHeader)).type_id };
        let not_implemented_type = unsafe {
            (*(not_implemented_ptr as *const prism_runtime::object::ObjectHeader)).type_id
        };

        assert_eq!(ellipsis_type, TypeId::ELLIPSIS);
        assert_eq!(not_implemented_type, TypeId::NOT_IMPLEMENTED);
    }

    #[test]
    fn test_registry_contains_functions() {
        let registry = BuiltinRegistry::with_standard_builtins();

        assert!(registry.is_function("len"));
        assert!(registry.is_function("print"));
        assert!(registry.is_function("range"));
        assert!(registry.is_function("type"));
        assert!(registry.is_function("getattr"));
        assert!(!registry.is_function("None")); // Not a function
    }

    #[test]
    fn test_registry_exposes_vm_aware_builtin_function_entries() {
        let registry = BuiltinRegistry::with_standard_builtins();

        assert!(registry.get_vm_function("getattr").is_some());
        assert!(
            registry
                .get("getattr")
                .and_then(|value| value.as_object_ptr())
                .is_some()
        );
        assert!(registry.get_vm_function("__import__").is_some());
        assert!(
            registry
                .get("__import__")
                .and_then(|value| value.as_object_ptr())
                .is_some()
        );
    }

    #[test]
    fn test_registry_contains_memoryview_type_object() {
        let registry = BuiltinRegistry::with_standard_builtins();
        let memoryview = registry
            .get("memoryview")
            .expect("memoryview should be registered");
        let ptr = memoryview
            .as_object_ptr()
            .expect("memoryview should be exposed as a type object");

        let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
        assert_eq!(header.type_id, TypeId::TYPE);
        assert_eq!(
            crate::builtins::builtin_type_object_type_id(ptr),
            Some(TypeId::MEMORYVIEW)
        );
    }

    #[test]
    fn test_registry_contains_slice_type_object() {
        let registry = BuiltinRegistry::with_standard_builtins();
        let slice_type = registry.get("slice").expect("slice should be registered");
        let ptr = slice_type
            .as_object_ptr()
            .expect("slice should be exposed as a type object");

        let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
        assert_eq!(header.type_id, TypeId::TYPE);
        assert_eq!(
            crate::builtins::builtin_type_object_type_id(ptr),
            Some(TypeId::SLICE)
        );
    }

    #[test]
    fn test_registry_contains_super_type_object() {
        let registry = BuiltinRegistry::with_standard_builtins();
        let super_type = registry.get("super").expect("super should be registered");
        let ptr = super_type
            .as_object_ptr()
            .expect("super should be exposed as a type object");

        let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
        assert_eq!(header.type_id, TypeId::TYPE);
        assert_eq!(
            crate::builtins::builtin_type_object_type_id(ptr),
            Some(TypeId::SUPER)
        );
    }

    #[test]
    fn test_registry_contains_supplemental_warning_category_types() {
        let registry = BuiltinRegistry::with_standard_builtins();

        for name in [
            "BytesWarning",
            "FutureWarning",
            "ImportWarning",
            "ResourceWarning",
            "UnicodeWarning",
            "EncodingWarning",
        ] {
            let value = registry
                .get(name)
                .unwrap_or_else(|| panic!("builtin {name} should be registered"));
            let ptr = value
                .as_object_ptr()
                .unwrap_or_else(|| panic!("builtin {name} should be a type object"));
            let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
            assert_eq!(header.type_id, TypeId::TYPE, "{name} should be a heap type");
        }
    }

    #[test]
    fn test_registry_warning_categories_are_warning_subclasses() {
        let registry = BuiltinRegistry::with_standard_builtins();
        let warning = registry.get("Warning").expect("Warning should exist");

        for name in [
            "BytesWarning",
            "FutureWarning",
            "ImportWarning",
            "ResourceWarning",
            "UnicodeWarning",
            "EncodingWarning",
        ] {
            let category = registry
                .get(name)
                .unwrap_or_else(|| panic!("builtin {name} should be registered"));
            let result = builtin_issubclass(&[category, warning])
                .unwrap_or_else(|_| panic!("{name} should be comparable with Warning"));
            assert_eq!(
                result.as_bool(),
                Some(true),
                "{name} should subclass Warning",
            );
        }
    }

    #[test]
    fn test_call_abs() {
        let registry = BuiltinRegistry::with_standard_builtins();
        let result = registry.call("abs", &[Value::int(-42).unwrap()]);
        assert_eq!(result.unwrap().as_int(), Some(42));
    }

    #[test]
    fn test_builtin_function_as_object_ptr() {
        let registry = BuiltinRegistry::with_standard_builtins();

        // Get the range function value
        let range_val = registry.get("range").expect("range should be registered");

        // Verify it's an object_ptr
        assert!(
            range_val.as_object_ptr().is_some(),
            "range should be stored as object_ptr, got value: {:?}",
            range_val
        );

        // Verify the type is BUILTIN_FUNCTION
        let ptr = range_val.as_object_ptr().unwrap();
        let header_ptr = ptr as *const prism_runtime::object::ObjectHeader;
        let type_id = unsafe { (*header_ptr).type_id };

        assert_eq!(
            type_id,
            prism_runtime::object::type_obj::TypeId::BUILTIN_FUNCTION,
            "range should have TypeId::BUILTIN_FUNCTION, got {:?}",
            type_id
        );
    }
}
