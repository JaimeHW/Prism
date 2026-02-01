//! Global scope management with fast hash map lookup.
//!
//! The global scope contains module-level names and provides fast
//! lookups using FxHashMap for minimal hashing overhead.

use prism_core::Value;
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// Global scope containing module-level bindings.
///
/// This is a thin wrapper around FxHashMap for fast name→value lookups.
/// FxHashMap uses a fast non-cryptographic hash function optimized
/// for small keys (like identifier strings).
#[derive(Debug, Default)]
pub struct GlobalScope {
    /// Name to value bindings.
    bindings: FxHashMap<Arc<str>, Value>,
}

impl GlobalScope {
    /// Create a new empty global scope.
    #[inline]
    pub fn new() -> Self {
        Self {
            bindings: FxHashMap::default(),
        }
    }

    /// Create with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bindings: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
        }
    }

    /// Get a value by name.
    #[inline]
    pub fn get(&self, name: &str) -> Option<Value> {
        self.bindings.get(name).copied()
    }

    /// Get a value by Arc<str> name (faster, avoids rehashing).
    #[inline]
    pub fn get_arc(&self, name: &Arc<str>) -> Option<Value> {
        self.bindings.get(name).copied()
    }

    /// Set a value.
    #[inline]
    pub fn set(&mut self, name: Arc<str>, value: Value) {
        self.bindings.insert(name, value);
    }

    /// Delete a name, returning the old value if present.
    #[inline]
    pub fn delete(&mut self, name: &str) -> Option<Value> {
        self.bindings.remove(name)
    }

    /// Check if a name exists.
    #[inline]
    pub fn contains(&self, name: &str) -> bool {
        self.bindings.contains_key(name)
    }

    /// Get the number of bindings.
    #[inline]
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Iterate over all bindings.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&Arc<str>, &Value)> {
        self.bindings.iter()
    }

    /// Get mutable access to bindings (for bulk operations).
    #[inline]
    pub fn bindings_mut(&mut self) -> &mut FxHashMap<Arc<str>, Value> {
        &mut self.bindings
    }
}

impl Clone for GlobalScope {
    fn clone(&self) -> Self {
        Self {
            bindings: self.bindings.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_scope_basic() {
        let mut globals = GlobalScope::new();

        // Set and get
        globals.set("x".into(), Value::int(42).unwrap());
        assert_eq!(globals.get("x").unwrap().as_int(), Some(42));

        // Not found
        assert!(globals.get("y").is_none());
    }

    #[test]
    fn test_global_scope_delete() {
        let mut globals = GlobalScope::new();

        globals.set("x".into(), Value::int(10).unwrap());
        assert!(globals.contains("x"));

        let old = globals.delete("x");
        assert_eq!(old.unwrap().as_int(), Some(10));
        assert!(!globals.contains("x"));
    }

    #[test]
    fn test_global_scope_overwrite() {
        let mut globals = GlobalScope::new();

        globals.set("x".into(), Value::int(1).unwrap());
        globals.set("x".into(), Value::int(2).unwrap());

        assert_eq!(globals.get("x").unwrap().as_int(), Some(2));
        assert_eq!(globals.len(), 1);
    }
}
