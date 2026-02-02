//! Core builtin registry and function infrastructure.
//!
//! Modular builtins organized by category for maintainability.
//! All builtins use static dispatch for maximum performance.

mod builtin_function;
mod execution;
mod functions;
mod introspect;
mod io;
mod itertools;
mod numeric;
mod string;
mod types;

pub use builtin_function::*;
pub use execution::*;
pub use functions::*;
pub use introspect::*;
pub use io::*;
pub use itertools::*;
pub use numeric::*;
pub use string::*;
pub use types::*;

use prism_core::Value;
use rustc_hash::FxHashMap;
use std::sync::Arc;

// =============================================================================
// BuiltinFn Type
// =============================================================================

/// Builtin function type.
///
/// Takes a slice of arguments and returns a Result.
/// Using Result allows proper error propagation.
pub type BuiltinFn = fn(&[Value]) -> Result<Value, BuiltinError>;

/// Error type for builtin functions.
#[derive(Debug, Clone)]
pub enum BuiltinError {
    /// Wrong number of arguments.
    TypeError(String),
    /// Value error (e.g., negative index).
    ValueError(String),
    /// Attribute error.
    AttributeError(String),
    /// Key error.
    KeyError(String),
    /// Index out of range.
    IndexError(String),
    /// Overflow error.
    OverflowError(String),
    /// Not implemented.
    NotImplemented(String),
}

impl std::fmt::Display for BuiltinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuiltinError::TypeError(msg) => write!(f, "TypeError: {}", msg),
            BuiltinError::ValueError(msg) => write!(f, "ValueError: {}", msg),
            BuiltinError::AttributeError(msg) => write!(f, "AttributeError: {}", msg),
            BuiltinError::KeyError(msg) => write!(f, "KeyError: {}", msg),
            BuiltinError::IndexError(msg) => write!(f, "IndexError: {}", msg),
            BuiltinError::OverflowError(msg) => write!(f, "OverflowError: {}", msg),
            BuiltinError::NotImplemented(msg) => write!(f, "NotImplementedError: {}", msg),
        }
    }
}

impl std::error::Error for BuiltinError {}

// =============================================================================
// BuiltinRegistry
// =============================================================================

/// Registry of builtin functions and values.
///
/// Uses FxHashMap for O(1) lookup by name.
/// Static function pointers ensure zero-cost dispatch.
#[derive(Default)]
pub struct BuiltinRegistry {
    /// Name to value mappings.
    entries: FxHashMap<Arc<str>, Value>,
    /// Function table for direct dispatch.
    functions: FxHashMap<Arc<str>, BuiltinFn>,
}

impl BuiltinRegistry {
    /// Create a new empty registry.
    #[inline]
    pub fn new() -> Self {
        Self {
            entries: FxHashMap::default(),
            functions: FxHashMap::default(),
        }
    }

    /// Create registry with standard Python builtins.
    pub fn with_standard_builtins() -> Self {
        let mut registry = Self::new();

        // Register builtin constants
        registry.register_value("None", Value::none());
        registry.register_value("True", Value::bool(true));
        registry.register_value("False", Value::bool(false));

        // Register core functions
        registry.register_function("len", functions::builtin_len);
        registry.register_function("abs", functions::builtin_abs);
        registry.register_function("min", functions::builtin_min);
        registry.register_function("max", functions::builtin_max);
        registry.register_function("sum", functions::builtin_sum);
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
        registry.register_function("complex", numeric::builtin_complex);

        // Register string/bytes functions
        registry.register_function("ord", string::builtin_ord);
        registry.register_function("chr", string::builtin_chr);
        registry.register_function("bytes", string::builtin_bytes);
        registry.register_function("bytearray", string::builtin_bytearray);
        registry.register_function("format", string::builtin_format);

        // Register introspection functions
        registry.register_function("dir", introspect::builtin_dir);
        registry.register_function("vars", introspect::builtin_vars);
        registry.register_function("globals", introspect::builtin_globals);
        registry.register_function("locals", introspect::builtin_locals);
        registry.register_function("help", introspect::builtin_help);
        registry.register_function("__import__", introspect::builtin_import);

        // Register execution functions
        registry.register_function("exec", execution::builtin_exec);
        registry.register_function("eval", execution::builtin_eval);
        registry.register_function("compile", execution::builtin_compile);
        registry.register_function("breakpoint", execution::builtin_breakpoint);

        // Register iteration functions
        registry.register_function("range", itertools::builtin_range);
        registry.register_function("iter", itertools::builtin_iter);
        registry.register_function("next", itertools::builtin_next);
        registry.register_function("enumerate", itertools::builtin_enumerate);
        registry.register_function("zip", itertools::builtin_zip);
        registry.register_function("map", itertools::builtin_map);
        registry.register_function("filter", itertools::builtin_filter);
        registry.register_function("reversed", itertools::builtin_reversed);
        registry.register_function("sorted", itertools::builtin_sorted);
        registry.register_function("all", itertools::builtin_all);
        registry.register_function("any", itertools::builtin_any);

        // Register I/O functions
        registry.register_function("print", io::builtin_print);
        registry.register_function("input", io::builtin_input);
        registry.register_function("open", io::builtin_open);

        // Register type constructors/converters
        registry.register_function("int", types::builtin_int);
        registry.register_function("float", types::builtin_float);
        registry.register_function("str", types::builtin_str);
        registry.register_function("bool", types::builtin_bool);
        registry.register_function("list", types::builtin_list);
        registry.register_function("tuple", types::builtin_tuple);
        registry.register_function("dict", types::builtin_dict);
        registry.register_function("set", types::builtin_set);
        registry.register_function("frozenset", types::builtin_frozenset);
        registry.register_function("type", types::builtin_type);
        registry.register_function("isinstance", types::builtin_isinstance);
        registry.register_function("issubclass", types::builtin_issubclass);
        registry.register_function("object", types::builtin_object);
        registry.register_function("getattr", types::builtin_getattr);
        registry.register_function("setattr", types::builtin_setattr);
        registry.register_function("hasattr", types::builtin_hasattr);
        registry.register_function("delattr", types::builtin_delattr);

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

    /// Check if a name is a builtin.
    #[inline]
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Check if a name is a builtin function.
    #[inline]
    pub fn is_function(&self, name: &str) -> bool {
        self.functions.contains_key(name)
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
    }

    #[test]
    fn test_registry_contains_functions() {
        let registry = BuiltinRegistry::with_standard_builtins();

        assert!(registry.is_function("len"));
        assert!(registry.is_function("print"));
        assert!(registry.is_function("range"));
        assert!(registry.is_function("type"));
        assert!(!registry.is_function("None")); // Not a function
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
