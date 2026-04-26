//! Wrapper utilities for functools.
//!
//! Implements `update_wrapper` and `wraps` for copying function metadata
//! from wrapped functions to their wrappers. This is essential for maintaining
//! introspection capabilities when using decorators.
//!
//! # Attributes Copied
//!
//! | Attribute | Description |
//! |-----------|-------------|
//! | `__module__` | Module where the function was defined |
//! | `__name__` | Function name |
//! | `__qualname__` | Qualified name (e.g., `Class.method`) |
//! | `__doc__` | Documentation string |
//! | `__wrapped__` | Reference to the original function |
//!
//! # Performance
//!
//! All attribute operations are O(1). The wrapper metadata is stored
//! in a compact `WrapperMetadata` struct using interned strings for
//! zero-copy attribute names.

use prism_core::Value;

// =============================================================================
// Constants
// =============================================================================

/// Attributes that are assigned (replaced) on the wrapper.
///
/// These are the default `WRAPPER_ASSIGNMENTS` in Python's functools.
pub const WRAPPER_ASSIGNMENTS: &[&str] = &[
    "__module__",
    "__name__",
    "__qualname__",
    "__annotations__",
    "__doc__",
];

/// Attributes that are updated (merged) on the wrapper.
///
/// These are the default `WRAPPER_UPDATES` in Python's functools.
pub const WRAPPER_UPDATES: &[&str] = &["__dict__"];

// =============================================================================
// Wrapper Metadata
// =============================================================================

/// Metadata extracted from a wrapped function.
///
/// Stores all the standard attributes that `update_wrapper` copies.
/// Uses `Option<Value>` for each field since not all functions have
/// all attributes.
#[derive(Debug, Clone)]
pub struct WrapperMetadata {
    /// `__module__` — the module name.
    pub module: Option<Value>,
    /// `__name__` — the function name.
    pub name: Option<Value>,
    /// `__qualname__` — the qualified name.
    pub qualname: Option<Value>,
    /// `__doc__` — the docstring.
    pub doc: Option<Value>,
    /// `__wrapped__` — reference to the original function.
    pub wrapped: Option<Value>,
    /// `__annotations__` — type annotations dict.
    pub annotations: Option<Value>,
}

impl WrapperMetadata {
    /// Create empty metadata.
    #[inline]
    pub fn new() -> Self {
        Self {
            module: None,
            name: None,
            qualname: None,
            doc: None,
            wrapped: None,
            annotations: None,
        }
    }

    /// Create metadata from a wrapped function's attributes.
    pub fn from_wrapped(
        module: Option<Value>,
        name: Option<Value>,
        qualname: Option<Value>,
        doc: Option<Value>,
        wrapped: Value,
    ) -> Self {
        Self {
            module,
            name,
            qualname,
            doc,
            wrapped: Some(wrapped),
            annotations: None,
        }
    }

    /// Create metadata with all fields populated.
    pub fn full(
        module: Value,
        name: Value,
        qualname: Value,
        doc: Value,
        wrapped: Value,
        annotations: Option<Value>,
    ) -> Self {
        Self {
            module: Some(module),
            name: Some(name),
            qualname: Some(qualname),
            doc: Some(doc),
            wrapped: Some(wrapped),
            annotations,
        }
    }

    /// Get an attribute by name.
    pub fn get_attr(&self, name: &str) -> Option<&Value> {
        match name {
            "__module__" => self.module.as_ref(),
            "__name__" => self.name.as_ref(),
            "__qualname__" => self.qualname.as_ref(),
            "__doc__" => self.doc.as_ref(),
            "__wrapped__" => self.wrapped.as_ref(),
            "__annotations__" => self.annotations.as_ref(),
            _ => None,
        }
    }

    /// Set an attribute by name.
    pub fn set_attr(&mut self, name: &str, value: Value) -> bool {
        match name {
            "__module__" => {
                self.module = Some(value);
                true
            }
            "__name__" => {
                self.name = Some(value);
                true
            }
            "__qualname__" => {
                self.qualname = Some(value);
                true
            }
            "__doc__" => {
                self.doc = Some(value);
                true
            }
            "__wrapped__" => {
                self.wrapped = Some(value);
                true
            }
            "__annotations__" => {
                self.annotations = Some(value);
                true
            }
            _ => false,
        }
    }

    /// Check if any metadata is set.
    pub fn has_any(&self) -> bool {
        self.module.is_some()
            || self.name.is_some()
            || self.qualname.is_some()
            || self.doc.is_some()
            || self.wrapped.is_some()
            || self.annotations.is_some()
    }

    /// Count how many attributes are set.
    pub fn count(&self) -> usize {
        let mut n = 0;
        if self.module.is_some() {
            n += 1;
        }
        if self.name.is_some() {
            n += 1;
        }
        if self.qualname.is_some() {
            n += 1;
        }
        if self.doc.is_some() {
            n += 1;
        }
        if self.wrapped.is_some() {
            n += 1;
        }
        if self.annotations.is_some() {
            n += 1;
        }
        n
    }

    /// List all set attribute names.
    pub fn dir(&self) -> Vec<&'static str> {
        let mut attrs = Vec::with_capacity(6);
        if self.module.is_some() {
            attrs.push("__module__");
        }
        if self.name.is_some() {
            attrs.push("__name__");
        }
        if self.qualname.is_some() {
            attrs.push("__qualname__");
        }
        if self.doc.is_some() {
            attrs.push("__doc__");
        }
        if self.wrapped.is_some() {
            attrs.push("__wrapped__");
        }
        if self.annotations.is_some() {
            attrs.push("__annotations__");
        }
        attrs
    }
}

impl Default for WrapperMetadata {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// update_wrapper
// =============================================================================

/// Copy attributes from a source metadata to a destination metadata.
///
/// This is the core of both `update_wrapper` and `wraps`. It copies
/// the standard `WRAPPER_ASSIGNMENTS` attributes from source to destination.
///
/// # Arguments
///
/// * `dest` — The wrapper's metadata (will be modified).
/// * `source` — The wrapped function's metadata.
/// * `assignments` — Which attributes to copy (default: `WRAPPER_ASSIGNMENTS`).
pub fn update_wrapper(
    dest: &mut WrapperMetadata,
    source: &WrapperMetadata,
    assignments: Option<&[&str]>,
) {
    let attrs = assignments.unwrap_or(WRAPPER_ASSIGNMENTS);
    for attr in attrs {
        if let Some(value) = source.get_attr(attr) {
            dest.set_attr(attr, value.clone());
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod wraps_tests;
