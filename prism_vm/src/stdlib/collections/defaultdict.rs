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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod defaultdict_tests {
    use super::*;
    use prism_core::intern::intern;

    fn int(i: i64) -> Value {
        Value::int_unchecked(i)
    }

    fn str_val(s: &str) -> Value {
        Value::string(intern(s))
    }

    // =========================================================================
    // Construction Tests
    // =========================================================================

    #[test]
    fn test_new_creates_empty() {
        let d = DefaultDict::new();
        assert!(d.is_empty());
        assert_eq!(d.len(), 0);
    }

    #[test]
    fn test_with_factory() {
        let d = DefaultDict::with_factory(DefaultFactory::Int);
        assert_eq!(*d.default_factory(), DefaultFactory::Int);
    }

    // =========================================================================
    // Get/Set Tests
    // =========================================================================

    #[test]
    fn test_set_and_get() {
        let mut d = DefaultDict::new();
        d.set(int(1), int(100));
        assert_eq!(d.get(&int(1)).and_then(|v| v.as_int()), Some(100));
    }

    #[test]
    fn test_get_missing_no_factory() {
        let d = DefaultDict::new();
        assert_eq!(d.get(&int(99)), None);
    }

    #[test]
    fn test_get_or_insert_no_factory_returns_none() {
        let mut d = DefaultDict::new();
        assert_eq!(d.get_or_insert(&int(1)), None);
    }

    #[test]
    fn test_get_or_insert_with_int_factory() {
        let mut d = DefaultDict::with_factory(DefaultFactory::Int);
        let val = d.get_or_insert(&int(1)).unwrap();
        assert_eq!(val.as_int(), Some(0));
        // Should be inserted now
        assert!(d.contains(&int(1)));
    }

    #[test]
    fn test_get_or_insert_with_float_factory() {
        let mut d = DefaultDict::with_factory(DefaultFactory::Float);
        let val = d.get_or_insert(&int(1)).unwrap();
        assert_eq!(val.as_float(), Some(0.0));
    }

    #[test]
    fn test_get_or_insert_with_bool_factory() {
        let mut d = DefaultDict::with_factory(DefaultFactory::Bool);
        let val = d.get_or_insert(&int(1)).unwrap();
        assert_eq!(val.as_bool(), Some(false));
    }

    #[test]
    fn test_get_returns_existing() {
        let mut d = DefaultDict::with_factory(DefaultFactory::Int);
        d.set(int(1), int(42));

        let val = d.get_or_insert(&int(1)).unwrap();
        assert_eq!(val.as_int(), Some(42));
    }

    // =========================================================================
    // Contains/Remove Tests
    // =========================================================================

    #[test]
    fn test_contains() {
        let mut d = DefaultDict::new();
        d.set(int(1), int(100));

        assert!(d.contains(&int(1)));
        assert!(!d.contains(&int(99)));
    }

    #[test]
    fn test_remove() {
        let mut d = DefaultDict::new();
        d.set(int(1), int(100));

        let removed = d.remove(&int(1));
        assert_eq!(removed.and_then(|v| v.as_int()), Some(100));
        assert!(!d.contains(&int(1)));
    }

    #[test]
    fn test_remove_missing() {
        let mut d = DefaultDict::new();
        assert_eq!(d.remove(&int(99)), None);
    }

    #[test]
    fn test_clear() {
        let mut d = DefaultDict::new();
        d.set(int(1), int(100));
        d.set(int(2), int(200));
        d.clear();

        assert!(d.is_empty());
    }

    // =========================================================================
    // Factory Change Tests
    // =========================================================================

    #[test]
    fn test_set_default_factory() {
        let mut d = DefaultDict::new();
        assert_eq!(*d.default_factory(), DefaultFactory::None);

        d.set_default_factory(DefaultFactory::Float);
        assert_eq!(*d.default_factory(), DefaultFactory::Float);
    }

    // =========================================================================
    // Iterator Tests
    // =========================================================================

    #[test]
    fn test_keys_iterator() {
        let mut d = DefaultDict::new();
        d.set(int(1), int(100));
        d.set(int(2), int(200));

        let keys: Vec<_> = d.keys().collect();
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn test_values_iterator() {
        let mut d = DefaultDict::new();
        d.set(int(1), int(100));
        d.set(int(2), int(200));

        let sum: i64 = d.values().filter_map(|v| v.as_int()).sum();
        assert_eq!(sum, 300);
    }

    #[test]
    fn test_iter() {
        let mut d = DefaultDict::new();
        d.set(int(1), int(100));

        let pairs: Vec<_> = d.iter().collect();
        assert_eq!(pairs.len(), 1);
    }

    // =========================================================================
    // String Key Tests
    // =========================================================================

    #[test]
    fn test_string_keys() {
        let mut d = DefaultDict::with_factory(DefaultFactory::Int);
        d.set(str_val("key1"), int(1));

        assert_eq!(d.get(&str_val("key1")).and_then(|v| v.as_int()), Some(1));

        let val = d.get_or_insert(&str_val("key2")).unwrap();
        assert_eq!(val.as_int(), Some(0));
    }

    // =========================================================================
    // Counting Pattern Tests
    // =========================================================================

    #[test]
    fn test_counting_pattern() {
        // Simulating: defaultdict(int)
        let mut d = DefaultDict::with_factory(DefaultFactory::Int);

        // Count occurrences
        let items = vec![int(1), int(2), int(1), int(3), int(1), int(2)];
        for item in items {
            // Get current count
            let count = d.get_or_insert(&item).unwrap().as_int().unwrap();
            // Increment
            d.set(item, Value::int_unchecked(count + 1));
        }

        assert_eq!(d.get(&int(1)).and_then(|v| v.as_int()), Some(3));
        assert_eq!(d.get(&int(2)).and_then(|v| v.as_int()), Some(2));
        assert_eq!(d.get(&int(3)).and_then(|v| v.as_int()), Some(1));
    }

    // =========================================================================
    // Stress Tests
    // =========================================================================

    #[test]
    fn test_stress_many_keys() {
        let mut d = DefaultDict::with_factory(DefaultFactory::Int);

        for i in 0..1000 {
            d.get_or_insert(&int(i));
        }

        assert_eq!(d.len(), 1000);

        for i in 0..1000 {
            assert!(d.contains(&int(i)));
        }
    }
}
