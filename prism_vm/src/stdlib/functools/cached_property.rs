//! Cached property descriptor implementation.
//!
//! `cached_property(func)` is a descriptor that converts a method into a
//! property whose value is computed once and then cached as an instance
//! attribute.
//!
//! # Performance
//!
//! | Operation | Time |
//! |-----------|------|
//! | First access | O(cost of func) |
//! | Subsequent access | O(1) |
//! | Delete/invalidate | O(1) |
//!
//! # Architecture
//!
//! The `CachedProperty` stores:
//! - The getter function (as a Value)
//! - The attribute name (for storing the cached result in `__dict__`)
//! - A local cache for cases where instance dict isn't available
//!
//! In full Python integration, the cached value would be stored in the
//! instance's `__dict__`, making subsequent accesses bypass the descriptor
//! entirely. Here we provide the core caching logic.

use prism_core::Value;

// =============================================================================
// CachedProperty
// =============================================================================

/// A cached property descriptor.
///
/// On first access, calls the getter function and caches the result.
/// Subsequent accesses return the cached value without calling the getter.
///
/// # Thread Safety
///
/// This implementation is NOT thread-safe (matching Python's semantics).
/// The VM should ensure single-threaded access to descriptors.
#[derive(Debug, Clone)]
pub struct CachedProperty {
    /// The getter function.
    func: Value,
    /// The attribute name (for __dict__ storage).
    attr_name: Option<String>,
    /// Local cache (used when instance dict isn't available).
    cached_value: Option<Value>,
    /// The docstring, if any.
    doc: Option<Value>,
}

impl CachedProperty {
    /// Create a new cached property with a getter function.
    #[inline]
    pub fn new(func: Value) -> Self {
        Self {
            func,
            attr_name: None,
            cached_value: None,
            doc: None,
        }
    }

    /// Create a cached property with a name.
    pub fn with_name(func: Value, name: String) -> Self {
        Self {
            func,
            attr_name: Some(name),
            cached_value: None,
            doc: None,
        }
    }

    /// Create a cached property with a name and docstring.
    pub fn with_doc(func: Value, name: String, doc: Value) -> Self {
        Self {
            func,
            attr_name: Some(name),
            cached_value: None,
            doc: Some(doc),
        }
    }

    // =========================================================================
    // Descriptor Protocol
    // =========================================================================

    /// Get the cached value, or compute and cache it.
    ///
    /// The `compute` closure is called with `&self.func` and should
    /// invoke the getter function on the instance, returning the result.
    ///
    /// Returns the cached value (either from cache or newly computed).
    pub fn get_or_compute<F>(&mut self, compute: F) -> Value
    where
        F: FnOnce(&Value) -> Value,
    {
        if let Some(ref cached) = self.cached_value {
            return cached.clone();
        }

        let value = compute(&self.func);
        self.cached_value = Some(value.clone());
        value
    }

    /// Get the cached value without computing.
    ///
    /// Returns `None` if no value has been cached yet.
    #[inline]
    pub fn get_cached(&self) -> Option<&Value> {
        self.cached_value.as_ref()
    }

    /// Check if a value has been cached.
    #[inline]
    pub fn is_cached(&self) -> bool {
        self.cached_value.is_some()
    }

    /// Delete (invalidate) the cached value.
    ///
    /// The next access will recompute the value by calling the getter.
    #[inline]
    pub fn invalidate(&mut self) {
        self.cached_value = None;
    }

    /// Set the cached value directly (for `__set__` support).
    #[inline]
    pub fn set_cached(&mut self, value: Value) {
        self.cached_value = Some(value);
    }

    // =========================================================================
    // Attribute Access
    // =========================================================================

    /// Get the getter function.
    #[inline]
    pub fn func(&self) -> &Value {
        &self.func
    }

    /// Get the attribute name.
    #[inline]
    pub fn attr_name(&self) -> Option<&str> {
        self.attr_name.as_deref()
    }

    /// Get the docstring.
    #[inline]
    pub fn doc(&self) -> Option<&Value> {
        self.doc.as_ref()
    }

    /// Set the attribute name (called by `__set_name__`).
    pub fn set_name(&mut self, name: String) {
        self.attr_name = Some(name);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod cached_property_tests;
