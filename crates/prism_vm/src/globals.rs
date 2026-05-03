//! Global scope management with fast hash map lookup and version tags.
//!
//! The global scope contains module-level names and provides fast lookups using
//! `FxHashMap` for minimal hashing overhead. Mutations advance a monotonic
//! version so interpreter inline caches and native JIT guards can validate
//! cached global reads with a pair of integer compares.

use prism_core::Value;
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// Global scope containing module-level bindings.
///
/// This is a thin wrapper around `FxHashMap` for fast name-to-value lookups.
/// `FxHashMap` uses a fast non-cryptographic hash function optimized for small
/// keys such as identifier strings.
#[derive(Debug, Default)]
pub struct GlobalScope {
    /// Name to value bindings.
    bindings: FxHashMap<Arc<str>, Value>,
    /// Monotonic mutation counter used by global lookup inline caches.
    version: u64,
}

impl GlobalScope {
    /// Create a new empty global scope.
    #[inline]
    pub fn new() -> Self {
        Self {
            bindings: FxHashMap::default(),
            version: 0,
        }
    }

    /// Create with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bindings: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            version: 0,
        }
    }

    /// Current namespace version.
    #[inline]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Get a value by name.
    #[inline]
    pub fn get(&self, name: &str) -> Option<Value> {
        self.bindings.get(name).copied()
    }

    /// Get a value by `Arc<str>` name.
    #[inline]
    pub fn get_arc(&self, name: &Arc<str>) -> Option<Value> {
        self.bindings.get(name).copied()
    }

    /// Set a value.
    #[inline]
    pub fn set(&mut self, name: Arc<str>, value: Value) {
        self.bindings.insert(name, value);
        self.bump_version();
    }

    /// Delete a name, returning the old value if present.
    #[inline]
    pub fn delete(&mut self, name: &str) -> Option<Value> {
        let removed = self.bindings.remove(name);
        if removed.is_some() {
            self.bump_version();
        }
        removed
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

    /// Get mutable access to bindings for bulk operations.
    #[inline]
    pub fn bindings_mut(&mut self) -> &mut FxHashMap<Arc<str>, Value> {
        self.bump_version();
        &mut self.bindings
    }

    #[inline]
    fn bump_version(&mut self) {
        self.version = self.version.wrapping_add(1).max(1);
    }
}

impl Clone for GlobalScope {
    fn clone(&self) -> Self {
        Self {
            bindings: self.bindings.clone(),
            version: self.version,
        }
    }
}
