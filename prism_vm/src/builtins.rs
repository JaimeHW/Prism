//! Builtin function registry.
//!
//! Provides built-in Python functions like print, len, range, etc.

use prism_core::Value;
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// Registry of builtin functions and values.
#[derive(Debug, Default)]
pub struct BuiltinRegistry {
    /// Name to value mappings.
    entries: FxHashMap<Arc<str>, Value>,
}

impl BuiltinRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            entries: FxHashMap::default(),
        }
    }

    /// Create registry with standard Python builtins.
    pub fn with_standard_builtins() -> Self {
        let mut registry = Self::new();

        // Register builtin constants
        registry.register("None", Value::none());
        registry.register("True", Value::bool(true));
        registry.register("False", Value::bool(false));

        // TODO: Register builtin functions
        // These require function objects which depend on the object system
        // registry.register("print", make_builtin_function(builtin_print));
        // registry.register("len", make_builtin_function(builtin_len));
        // registry.register("range", make_builtin_function(builtin_range));
        // etc.

        registry
    }

    /// Register a builtin name.
    #[inline]
    pub fn register(&mut self, name: impl Into<Arc<str>>, value: Value) {
        self.entries.insert(name.into(), value);
    }

    /// Get a builtin by name.
    #[inline]
    pub fn get(&self, name: &str) -> Option<Value> {
        self.entries.get(name).copied()
    }

    /// Get by Arc<str> (faster, avoids rehashing).
    #[inline]
    pub fn get_arc(&self, name: &Arc<str>) -> Option<Value> {
        self.entries.get(name).copied()
    }

    /// Check if a name is a builtin.
    #[inline]
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Iterate over all builtins.
    pub fn iter(&self) -> impl Iterator<Item = (&Arc<str>, &Value)> {
        self.entries.iter()
    }

    /// Get the number of registered builtins.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// =============================================================================
// Builtin Function Implementations (Stubs)
// =============================================================================

/// Builtin print function.
pub fn builtin_print(_args: &[Value]) -> Value {
    // TODO: Implement print
    Value::none()
}

/// Builtin len function.
pub fn builtin_len(_args: &[Value]) -> Value {
    // TODO: Implement len
    Value::none()
}

/// Builtin range function.
pub fn builtin_range(_args: &[Value]) -> Value {
    // TODO: Implement range
    Value::none()
}

/// Builtin type function.
pub fn builtin_type(_args: &[Value]) -> Value {
    // TODO: Implement type
    Value::none()
}

/// Builtin int function.
pub fn builtin_int(_args: &[Value]) -> Value {
    // TODO: Implement int conversion
    Value::none()
}

/// Builtin float function.
pub fn builtin_float(_args: &[Value]) -> Value {
    // TODO: Implement float conversion
    Value::none()
}

/// Builtin str function.
pub fn builtin_str(_args: &[Value]) -> Value {
    // TODO: Implement str conversion
    Value::none()
}

/// Builtin bool function.
pub fn builtin_bool(_args: &[Value]) -> Value {
    // TODO: Implement bool conversion
    Value::none()
}

/// Builtin list function.
pub fn builtin_list(_args: &[Value]) -> Value {
    // TODO: Implement list conversion
    Value::none()
}

/// Builtin dict function.
pub fn builtin_dict(_args: &[Value]) -> Value {
    // TODO: Implement dict conversion
    Value::none()
}

/// Builtin abs function.
pub fn builtin_abs(args: &[Value]) -> Value {
    if let Some(arg) = args.first() {
        if let Some(i) = arg.as_int() {
            return Value::int(i.abs()).unwrap_or_else(|| Value::none());
        }
        if let Some(f) = arg.as_float() {
            return Value::float(f.abs());
        }
    }
    Value::none()
}

/// Builtin min function.
pub fn builtin_min(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::none();
    }

    let mut min = args[0];
    for arg in &args[1..] {
        // TODO: Proper comparison
        if let (Some(a), Some(b)) = (arg.as_int(), min.as_int()) {
            if a < b {
                min = *arg;
            }
        }
    }
    min
}

/// Builtin max function.
pub fn builtin_max(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::none();
    }

    let mut max = args[0];
    for arg in &args[1..] {
        if let (Some(a), Some(b)) = (arg.as_int(), max.as_int()) {
            if a > b {
                max = *arg;
            }
        }
    }
    max
}

/// Builtin sum function.
pub fn builtin_sum(args: &[Value]) -> Value {
    // TODO: Implement proper iteration
    Value::none()
}

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
    fn test_builtin_abs() {
        assert_eq!(builtin_abs(&[Value::int(-42).unwrap()]).as_int(), Some(42));
        assert_eq!(builtin_abs(&[Value::int(42).unwrap()]).as_int(), Some(42));
    }
}
