//! High-performance defaultdict implementation.
//!
//! A defaultdict is a dictionary subclass that provides a default value for
//! missing keys. Unlike regular dicts, accessing a missing key automatically
//! creates a new entry with a default value.
//!
//! # Performance Characteristics
//!
//! | Operation | Time | Notes |
//! |-----------|------|-------|
//! | `get()` | O(1) | May insert default |
//! | `set()` | O(1) | Standard dict set |
//! | `contains()` | O(1) | Hash lookup |
//! | `remove()` | O(1) | Hash removal |
//!
//! # Default Factories
//!
//! Common factory types:
//! - `DefaultFactory::List` - Empty list `[]`
//! - `DefaultFactory::Dict` - Empty dict `{}`
//! - `DefaultFactory::Int` - Zero `0`
//! - `DefaultFactory::Float` - Zero `0.0`
//! - `DefaultFactory::String` - Empty string `""`
//! - `DefaultFactory::Set` - Empty set `set()`

use super::counter::HashableValue;
use prism_core::Value;
use std::collections::HashMap;

// =============================================================================
// Default Factory
// =============================================================================

/// Factory for creating default values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefaultFactory {
    /// No default factory (raises KeyError).
    None,
    /// Default to zero integer.
    Int,
    /// Default to zero float.
    Float,
    /// Default to empty list (represented as None for now).
    List,
    /// Default to empty dict (represented as None for now).
    Dict,
    /// Default to None value.
    NoneValue,
    /// Default to false boolean.
    Bool,
}

impl DefaultFactory {
    /// Create the default value for this factory.
    pub fn create(&self) -> Option<Value> {
        match self {
            DefaultFactory::None => Option::None,
            DefaultFactory::Int => Some(Value::int_unchecked(0)),
            DefaultFactory::Float => Some(Value::float(0.0)),
            DefaultFactory::List => Some(Value::none()), // TODO: Return actual list
            DefaultFactory::Dict => Some(Value::none()), // TODO: Return actual dict
            DefaultFactory::NoneValue => Some(Value::none()),
            DefaultFactory::Bool => Some(Value::bool(false)),
        }
    }
}

// =============================================================================
// DefaultDict
// =============================================================================

/// A dictionary that provides default values for missing keys.
///
/// # Examples
///
/// ```ignore
/// let mut d = DefaultDict::with_factory(DefaultFactory::Int);
/// d.get_or_insert(&key)?; // Returns 0 if key doesn't exist
/// ```
#[derive(Debug, Clone)]
pub struct DefaultDict {
    /// The underlying storage.
    data: HashMap<HashableValue, Value>,
    /// Factory for creating default values.
    default_factory: DefaultFactory,
}

impl DefaultDict {
    /// Create a new defaultdict with no default factory.
    #[inline]
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            default_factory: DefaultFactory::None,
        }
    }

    /// Create a new defaultdict with a specific factory.
    #[inline]
    pub fn with_factory(factory: DefaultFactory) -> Self {
        Self {
            data: HashMap::new(),
            default_factory: factory,
        }
    }

    /// Create with capacity.
    #[inline]
    pub fn with_capacity(factory: DefaultFactory, capacity: usize) -> Self {
        Self {
            data: HashMap::with_capacity(capacity),
            default_factory: factory,
        }
    }

    // =========================================================================
    // Core Operations
    // =========================================================================

    /// Get a value, inserting the default if missing.
    /// Returns None if no default factory is set.
    pub fn get_or_insert(&mut self, key: &Value) -> Option<Value> {
        let hkey = HashableValue(key.clone());

        if let Some(value) = self.data.get(&hkey) {
            return Some(value.clone());
        }

        // Create and insert default
        let default = self.default_factory.create()?;
        self.data.insert(hkey, default.clone());
        Some(default)
    }

    /// Get a value without inserting a default.
    #[inline]
    pub fn get(&self, key: &Value) -> Option<&Value> {
        self.data.get(&HashableValue(key.clone()))
    }

    /// Set a value.
    #[inline]
    pub fn set(&mut self, key: Value, value: Value) {
        self.data.insert(HashableValue(key), value);
    }

    /// Check if key exists.
    #[inline]
    pub fn contains(&self, key: &Value) -> bool {
        self.data.contains_key(&HashableValue(key.clone()))
    }

    /// Remove a key.
    #[inline]
    pub fn remove(&mut self, key: &Value) -> Option<Value> {
        self.data.remove(&HashableValue(key.clone()))
    }

    /// Clear all entries.
    #[inline]
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Get the number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the default factory.
    #[inline]
    pub fn default_factory(&self) -> &DefaultFactory {
        &self.default_factory
    }

    /// Set the default factory.
    #[inline]
    pub fn set_default_factory(&mut self, factory: DefaultFactory) {
        self.default_factory = factory;
    }

    // =========================================================================
    // Iterator Support
    // =========================================================================

    /// Iterator over keys.
    pub fn keys(&self) -> impl Iterator<Item = &Value> {
        self.data.keys().map(|k| &k.0)
    }

    /// Iterator over values.
    pub fn values(&self) -> impl Iterator<Item = &Value> {
        self.data.values()
    }

    /// Iterator over (key, value) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&Value, &Value)> {
        self.data.iter().map(|(k, v)| (&k.0, v))
    }
}

impl Default for DefaultDict {
    fn default() -> Self {
        Self::new()
    }
}
