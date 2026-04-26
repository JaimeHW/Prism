//! High-performance `partial` implementation.
//!
//! `partial(func, *args, **kwargs)` freezes some portion of a function's
//! arguments and/or keywords, producing a new callable with a simplified
//! signature.
//!
//! # Performance Characteristics
//!
//! | Operation | Time | Space |
//! |-----------|------|-------|
//! | Construction | O(a + k) | Inline for ≤8 args |
//! | Call | O(a + k + n) | Merged result vec |
//! | Attribute access | O(1) | - |
//!
//! Where a = frozen args, k = frozen kwargs, n = call-time args.
//!
//! # Memory Optimization
//!
//! Uses `SmallVec<[Value; 8]>` for frozen positional arguments, avoiding
//! heap allocation for the common case of ≤8 frozen arguments.

use prism_core::Value;
use smallvec::SmallVec;
use std::collections::HashMap;

use crate::stdlib::collections::counter::HashableValue;

// =============================================================================
// Partial
// =============================================================================

/// A partially-applied function object.
///
/// Stores a reference to the wrapped function along with pre-bound
/// positional arguments and keyword arguments. When called, the frozen
/// arguments are prepended to (or merged with) the call-time arguments.
///
/// # Nested Partial Flattening
///
/// When a `Partial` wraps another `Partial`, the frozen arguments are
/// automatically flattened to avoid unnecessary indirection.
#[derive(Debug, Clone)]
pub struct Partial {
    /// The wrapped callable's identifier or representation.
    func: Value,

    /// Pre-bound positional arguments.
    /// Uses SmallVec to avoid heap allocation for ≤8 args.
    args: SmallVec<[Value; 8]>,

    /// Pre-bound keyword arguments.
    keywords: HashMap<HashableValue, Value>,
}

impl Partial {
    /// Create a new Partial with only positional arguments.
    #[inline]
    pub fn new(func: Value, args: Vec<Value>) -> Self {
        Self {
            func,
            args: SmallVec::from_vec(args),
            keywords: HashMap::new(),
        }
    }

    /// Create a new Partial with positional and keyword arguments.
    pub fn with_kwargs(
        func: Value,
        args: Vec<Value>,
        kwargs: HashMap<HashableValue, Value>,
    ) -> Self {
        Self {
            func,
            args: SmallVec::from_vec(args),
            keywords: kwargs,
        }
    }

    /// Create from SmallVec directly (avoids allocation).
    #[inline]
    pub fn from_smallvec(
        func: Value,
        args: SmallVec<[Value; 8]>,
        keywords: HashMap<HashableValue, Value>,
    ) -> Self {
        Self {
            func,
            args,
            keywords,
        }
    }

    // =========================================================================
    // Attribute Access
    // =========================================================================

    /// Get the wrapped function.
    #[inline]
    pub fn func(&self) -> &Value {
        &self.func
    }

    /// Get the frozen positional arguments.
    #[inline]
    pub fn args(&self) -> &[Value] {
        &self.args
    }

    /// Get the frozen keyword arguments.
    #[inline]
    pub fn keywords(&self) -> &HashMap<HashableValue, Value> {
        &self.keywords
    }

    /// Get the number of frozen positional arguments.
    #[inline]
    pub fn num_args(&self) -> usize {
        self.args.len()
    }

    /// Get the number of frozen keyword arguments.
    #[inline]
    pub fn num_keywords(&self) -> usize {
        self.keywords.len()
    }

    /// Check if any arguments are frozen.
    #[inline]
    pub fn has_frozen_args(&self) -> bool {
        !self.args.is_empty() || !self.keywords.is_empty()
    }

    // =========================================================================
    // Call Support
    // =========================================================================

    /// Build the merged positional argument list for a call.
    ///
    /// Returns `frozen_args + call_args` as a single vector.
    ///
    /// # Performance
    ///
    /// Pre-allocates the exact required capacity.
    pub fn merge_args(&self, call_args: &[Value]) -> Vec<Value> {
        let mut merged = Vec::with_capacity(self.args.len() + call_args.len());
        merged.extend_from_slice(&self.args);
        merged.extend_from_slice(call_args);
        merged
    }

    /// Build the merged keyword argument map for a call.
    ///
    /// Call-time kwargs override frozen kwargs (matching Python semantics).
    pub fn merge_kwargs(
        &self,
        call_kwargs: &HashMap<HashableValue, Value>,
    ) -> HashMap<HashableValue, Value> {
        let mut merged = self.keywords.clone();
        // Call-time kwargs override frozen kwargs
        for (k, v) in call_kwargs {
            merged.insert(k.clone(), v.clone());
        }
        merged
    }

    /// Build merged kwargs from an iterator (avoids allocating input HashMap).
    pub fn merge_kwargs_iter<I>(&self, call_kwargs: I) -> HashMap<HashableValue, Value>
    where
        I: IntoIterator<Item = (HashableValue, Value)>,
    {
        let mut merged = self.keywords.clone();
        for (k, v) in call_kwargs {
            merged.insert(k, v);
        }
        merged
    }

    // =========================================================================
    // Nested Partial Flattening
    // =========================================================================

    /// Flatten nested partials into a single partial.
    ///
    /// If `self.func` is itself a Partial, extract the inner function
    /// and concatenate the argument chains.
    ///
    /// This returns the resolved function and the complete argument chain.
    pub fn flatten_args(&self, inner: &Partial) -> (Value, Vec<Value>) {
        // Inner partial's args come first, then our args
        let mut merged = Vec::with_capacity(inner.args.len() + self.args.len());
        merged.extend_from_slice(&inner.args);
        merged.extend_from_slice(&self.args);
        (inner.func.clone(), merged)
    }

    /// Flatten nested partial kwargs.
    pub fn flatten_kwargs(&self, inner: &Partial) -> HashMap<HashableValue, Value> {
        let mut merged = inner.keywords.clone();
        // Outer kwargs override inner kwargs
        for (k, v) in &self.keywords {
            merged.insert(k.clone(), v.clone());
        }
        merged
    }

    // =========================================================================
    // Mutation (for update_wrapper pattern)
    // =========================================================================

    /// Add a positional argument.
    #[inline]
    pub fn push_arg(&mut self, arg: Value) {
        self.args.push(arg);
    }

    /// Add a keyword argument.
    #[inline]
    pub fn set_keyword(&mut self, key: HashableValue, value: Value) {
        self.keywords.insert(key, value);
    }

    /// Remove a keyword argument.
    #[inline]
    pub fn remove_keyword(&mut self, key: &HashableValue) -> Option<Value> {
        self.keywords.remove(key)
    }

    /// Clear all frozen arguments.
    pub fn clear_args(&mut self) {
        self.args.clear();
        self.keywords.clear();
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod partial_tests;
