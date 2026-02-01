//! Dictionary object implementation.
//!
//! High-performance hash map for Python's dict type.

use crate::object::type_obj::TypeId;
use crate::object::{HASH_NOT_COMPUTED, ObjectHeader, PyObject};
use prism_core::Value;
use rustc_hash::FxHashMap;
use std::hash::{Hash, Hasher};

// =============================================================================
// Value Wrapper for Hashing
// =============================================================================

/// Wrapper to make Value hashable for dict keys.
///
/// Uses the NaN-boxed representation for fast hashing of primitives.
#[derive(Clone, Copy)]
struct HashableValue(Value);

impl Hash for HashableValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // For integers and simple values, hash the payload directly
        if let Some(i) = self.0.as_int() {
            i.hash(state);
        } else if let Some(f) = self.0.as_float() {
            // Hash float bits (handles NaN etc)
            f.to_bits().hash(state);
        } else if self.0.is_none() {
            0u64.hash(state);
        } else if let Some(b) = self.0.as_bool() {
            b.hash(state);
        } else {
            // For objects, use pointer as hash
            if let Some(ptr) = self.0.as_object_ptr() {
                (ptr as usize).hash(state);
            }
        }
    }
}

impl PartialEq for HashableValue {
    fn eq(&self, other: &Self) -> bool {
        // Fast path: same bits
        if let (Some(a), Some(b)) = (self.0.as_int(), other.0.as_int()) {
            return a == b;
        }
        if let (Some(a), Some(b)) = (self.0.as_float(), other.0.as_float()) {
            return a == b;
        }
        if self.0.is_none() && other.0.is_none() {
            return true;
        }
        if let (Some(a), Some(b)) = (self.0.as_bool(), other.0.as_bool()) {
            return a == b;
        }
        if let (Some(a), Some(b)) = (self.0.as_object_ptr(), other.0.as_object_ptr()) {
            return a == b;
        }
        false
    }
}

impl Eq for HashableValue {}

// =============================================================================
// Dictionary Object
// =============================================================================

/// Python dict object.
///
/// Uses FxHashMap for fast insertion and lookup.
/// Order is not preserved (unlike Python 3.7+, but we can change this).
#[repr(C)]
pub struct DictObject {
    /// Object header.
    pub header: ObjectHeader,
    /// Key-value pairs.
    items: FxHashMap<HashableValue, Value>,
}

impl DictObject {
    /// Create a new empty dict.
    #[inline]
    pub fn new() -> Self {
        Self {
            header: ObjectHeader::new(TypeId::DICT),
            items: FxHashMap::default(),
        }
    }

    /// Create a dict with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::DICT),
            items: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
        }
    }

    /// Get the number of items.
    #[inline]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if the dict is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Get a value by key.
    #[inline]
    pub fn get(&self, key: Value) -> Option<Value> {
        self.items.get(&HashableValue(key)).copied()
    }

    /// Set a key-value pair.
    #[inline]
    pub fn set(&mut self, key: Value, value: Value) {
        self.items.insert(HashableValue(key), value);
    }

    /// Remove a key and return its value.
    #[inline]
    pub fn remove(&mut self, key: Value) -> Option<Value> {
        self.items.remove(&HashableValue(key))
    }

    /// Check if the dict contains a key.
    #[inline]
    pub fn contains_key(&self, key: Value) -> bool {
        self.items.contains_key(&HashableValue(key))
    }

    /// Clear all items.
    #[inline]
    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Get an iterator over keys.
    pub fn keys(&self) -> impl Iterator<Item = Value> + '_ {
        self.items.keys().map(|k| k.0)
    }

    /// Get an iterator over values.
    pub fn values(&self) -> impl Iterator<Item = Value> + '_ {
        self.items.values().copied()
    }

    /// Get an iterator over key-value pairs.
    pub fn iter(&self) -> impl Iterator<Item = (Value, Value)> + '_ {
        self.items.iter().map(|(k, v)| (k.0, *v))
    }

    /// Update this dict with items from another.
    pub fn update(&mut self, other: &DictObject) {
        for (k, v) in other.items.iter() {
            self.items.insert(*k, *v);
        }
    }

    /// Get value or insert default.
    pub fn get_or_insert(&mut self, key: Value, default: Value) -> Value {
        *self.items.entry(HashableValue(key)).or_insert(default)
    }

    /// Pop a key and return (key, value) or None.
    pub fn popitem(&mut self) -> Option<(Value, Value)> {
        // FxHashMap doesn't have pop, so we iterate
        let key = self.items.keys().next().copied()?;
        let value = self.items.remove(&key)?;
        Some((key.0, value))
    }
}

impl Default for DictObject {
    fn default() -> Self {
        Self::new()
    }
}

impl PyObject for DictObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dict_basic() {
        let mut dict = DictObject::new();
        assert!(dict.is_empty());

        let key1 = Value::int(1).unwrap();
        let key2 = Value::int(2).unwrap();
        let val1 = Value::int(100).unwrap();
        let val2 = Value::int(200).unwrap();

        dict.set(key1, val1);
        dict.set(key2, val2);

        assert_eq!(dict.len(), 2);
        assert_eq!(dict.get(key1).unwrap().as_int(), Some(100));
        assert_eq!(dict.get(key2).unwrap().as_int(), Some(200));
    }

    #[test]
    fn test_dict_overwrite() {
        let mut dict = DictObject::new();
        let key = Value::int(1).unwrap();

        dict.set(key, Value::int(100).unwrap());
        dict.set(key, Value::int(200).unwrap());

        assert_eq!(dict.len(), 1);
        assert_eq!(dict.get(key).unwrap().as_int(), Some(200));
    }

    #[test]
    fn test_dict_remove() {
        let mut dict = DictObject::new();
        let key = Value::int(1).unwrap();

        dict.set(key, Value::int(100).unwrap());
        assert!(dict.contains_key(key));

        let removed = dict.remove(key);
        assert_eq!(removed.unwrap().as_int(), Some(100));
        assert!(!dict.contains_key(key));
    }

    #[test]
    fn test_dict_none_key() {
        let mut dict = DictObject::new();
        let key = Value::none();

        dict.set(key, Value::int(42).unwrap());
        assert_eq!(dict.get(key).unwrap().as_int(), Some(42));
    }
}
