//! Dictionary object implementation.
//!
//! High-performance hash map for Python's dict type.

use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use crate::types::hashable::HashableValue;
use crate::types::int::value_to_bigint;
use crate::types::tuple::TupleObject;
use prism_core::Value;
use rustc_hash::{FxHashMap, FxHashSet};

// =============================================================================
// Dictionary Object
// =============================================================================

/// Python dict object.
///
/// Uses FxHashMap for fast insertion and lookup.
/// Insertion order is preserved to match Python 3.7+ semantics.
#[repr(C)]
#[derive(Debug)]
pub struct DictObject {
    /// Object header.
    pub header: ObjectHeader,
    entries: DictEntries,
    version: u64,
}

#[derive(Debug, Clone, Default)]
struct DictEntries {
    items: FxHashMap<HashableValue, Value>,
    hashes: FxHashMap<HashableValue, i64>,
    order: Vec<Option<HashableValue>>,
    positions: FxHashMap<HashableValue, usize>,
    protocol_keys: Option<FxHashSet<HashableValue>>,
    tombstones: usize,
}

impl DictObject {
    /// Create a new empty dict.
    #[inline]
    pub fn new() -> Self {
        Self {
            header: ObjectHeader::new(TypeId::DICT),
            entries: DictEntries::default(),
            version: 0,
        }
    }

    /// Create a dict with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::DICT),
            entries: DictEntries {
                items: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
                hashes: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
                positions: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
                protocol_keys: None,
                order: Vec::with_capacity(capacity),
                tombstones: 0,
            },
            version: 0,
        }
    }

    /// Get the number of items.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.items.len()
    }

    /// Monotonic structural mutation counter for iterator invalidation.
    #[inline]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Check if the dict is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.items.is_empty()
    }

    /// Return true when at least one key can require Python-level equality for
    /// collision resolution.
    #[inline]
    pub fn has_protocol_keys(&self) -> bool {
        self.entries
            .protocol_keys
            .as_ref()
            .is_some_and(|keys| !keys.is_empty())
    }

    /// Get a value by key.
    #[inline]
    pub fn get(&self, key: Value) -> Option<Value> {
        self.entries.items.get(&HashableValue(key)).copied()
    }

    /// Set a key-value pair.
    #[inline]
    pub fn set(&mut self, key: Value, value: Value) {
        self.insert(key, value, None, requires_protocol_lookup(key));
    }

    /// Set a key-value pair after the VM has computed the Python hash.
    #[inline]
    pub fn set_with_hash(&mut self, key: Value, value: Value, hash: i64) {
        self.insert(key, value, Some(hash), requires_protocol_lookup(key));
    }

    /// Set a key-value pair when the caller already classified whether lookup
    /// may need Python-level equality.
    #[inline]
    pub fn set_with_hash_and_protocol_lookup(
        &mut self,
        key: Value,
        value: Value,
        hash: i64,
        requires_protocol_lookup: bool,
    ) {
        self.insert(key, value, Some(hash), requires_protocol_lookup);
    }

    #[inline]
    fn insert(
        &mut self,
        key: Value,
        value: Value,
        hash: Option<i64>,
        requires_protocol_lookup: bool,
    ) {
        let key = HashableValue(key);
        if self.entries.items.insert(key, value).is_none() {
            self.bump_version();
            self.entries.positions.insert(key, self.entries.order.len());
            self.entries.order.push(Some(key));
        }
        if requires_protocol_lookup {
            self.mark_protocol_key(key);
        }
        if let Some(hash) = hash {
            self.entries.hashes.insert(key, hash);
        }
    }

    /// Remove a key and return its value.
    #[inline]
    pub fn remove(&mut self, key: Value) -> Option<Value> {
        let key = HashableValue(key);
        let Some((stored_key, value)) = self.entries.items.remove_entry(&key) else {
            return None;
        };

        self.bump_version();
        self.unmark_protocol_key(stored_key);
        self.entries.hashes.remove(&stored_key);
        if let Some(index) = self.entries.positions.remove(&stored_key)
            && let Some(slot) = self.entries.order.get_mut(index)
            && slot.take().is_some()
        {
            self.entries.tombstones += 1;
        }
        self.maybe_compact_order();
        Some(value)
    }

    /// Check if the dict contains a key.
    #[inline]
    pub fn contains_key(&self, key: Value) -> bool {
        self.entries.items.contains_key(&HashableValue(key))
    }

    /// Clear all items.
    #[inline]
    pub fn clear(&mut self) {
        if !self.entries.items.is_empty() {
            self.bump_version();
        }
        self.entries.items.clear();
        self.entries.hashes.clear();
        self.entries.order.clear();
        self.entries.positions.clear();
        self.entries.protocol_keys = None;
        self.entries.tombstones = 0;
    }

    /// Get an iterator over keys.
    pub fn keys(&self) -> impl Iterator<Item = Value> + '_ {
        self.entries
            .order
            .iter()
            .filter_map(|key| key.as_ref().map(|key| key.0))
    }

    /// Get an iterator over values.
    pub fn values(&self) -> impl Iterator<Item = Value> + '_ {
        self.entries.order.iter().filter_map(move |key| {
            key.as_ref()
                .and_then(|key| self.entries.items.get(key).copied())
        })
    }

    /// Get an iterator over key-value pairs.
    pub fn iter(&self) -> impl Iterator<Item = (Value, Value)> + '_ {
        self.entries.order.iter().filter_map(move |key| {
            key.as_ref()
                .and_then(|key| self.entries.items.get(key).map(|value| (key.0, *value)))
        })
    }

    /// Return the cached Python hash for a stored key, when available.
    #[inline]
    pub fn stored_hash(&self, key: Value) -> Option<i64> {
        self.entries.hashes.get(&HashableValue(key)).copied()
    }

    /// Update this dict with items from another.
    pub fn update(&mut self, other: &DictObject) {
        for (key, value) in other.iter() {
            if let Some(hash) = other.stored_hash(key) {
                self.set_with_hash(key, value, hash);
            } else {
                self.set(key, value);
            }
        }
    }

    /// Get value or insert default.
    pub fn get_or_insert(&mut self, key: Value, default: Value) -> Value {
        if let Some(existing) = self.get(key) {
            return existing;
        }

        self.set(key, default);
        default
    }

    /// Return the current value for a key, inserting a default when absent.
    #[inline]
    pub fn setdefault(&mut self, key: Value, default: Value) -> Value {
        self.get_or_insert(key, default)
    }

    /// Move an existing key to either end of the insertion order.
    pub fn move_to_end(&mut self, key: Value, last: bool) -> bool {
        let key = HashableValue(key);
        let Some(index) = self.entries.positions.get(&key).copied() else {
            return false;
        };

        if last {
            if self.entries.order.last().and_then(Option::as_ref) == Some(&key) {
                return true;
            }

            self.bump_version();
            if let Some(slot) = self.entries.order.get_mut(index)
                && slot.take().is_some()
            {
                self.entries.tombstones += 1;
            }
            self.entries.positions.insert(key, self.entries.order.len());
            self.entries.order.push(Some(key));
            self.maybe_compact_order();
            return true;
        }

        if index != 0 {
            self.bump_version();
        }
        self.compact_order_with_prefix(key);
        true
    }

    /// Pop a key and return (key, value) or None.
    pub fn popitem(&mut self) -> Option<(Value, Value)> {
        let key = loop {
            match self.entries.order.pop()? {
                Some(key) => break key,
                None => self.entries.tombstones = self.entries.tombstones.saturating_sub(1),
            }
        };
        self.bump_version();
        self.entries.positions.remove(&key);
        self.entries.hashes.remove(&key);
        self.unmark_protocol_key(key);
        let value = self.entries.items.remove(&key)?;
        Some((key.0, value))
    }

    #[inline]
    fn mark_protocol_key(&mut self, key: HashableValue) {
        self.entries
            .protocol_keys
            .get_or_insert_with(FxHashSet::default)
            .insert(key);
    }

    #[inline]
    fn unmark_protocol_key(&mut self, key: HashableValue) {
        let Some(keys) = self.entries.protocol_keys.as_mut() else {
            return;
        };
        keys.remove(&key);
        if keys.is_empty() {
            self.entries.protocol_keys = None;
        }
    }

    #[inline]
    fn bump_version(&mut self) {
        self.version = self.version.wrapping_add(1);
    }

    #[inline]
    fn maybe_compact_order(&mut self) {
        if self.entries.tombstones > 16 && self.entries.tombstones > self.entries.items.len() {
            self.compact_order();
        }
    }

    fn compact_order(&mut self) {
        if self.entries.tombstones == 0 {
            return;
        }

        let mut compacted = Vec::with_capacity(self.entries.items.len());
        self.entries.positions.clear();
        self.entries
            .hashes
            .retain(|key, _| self.entries.items.contains_key(key));
        if let Some(protocol_keys) = self.entries.protocol_keys.as_mut() {
            protocol_keys.retain(|key| self.entries.items.contains_key(key));
            if protocol_keys.is_empty() {
                self.entries.protocol_keys = None;
            }
        }
        for key in self.entries.order.iter().filter_map(|key| *key) {
            self.entries.positions.insert(key, compacted.len());
            compacted.push(Some(key));
        }
        self.entries.order = compacted;
        self.entries.tombstones = 0;
    }

    fn compact_order_with_prefix(&mut self, prefix: HashableValue) {
        let mut compacted = Vec::with_capacity(self.entries.items.len());
        self.entries.positions.clear();
        self.entries.positions.insert(prefix, 0);
        compacted.push(Some(prefix));

        for key in self.entries.order.iter().filter_map(|key| *key) {
            if key == prefix {
                continue;
            }
            self.entries.positions.insert(key, compacted.len());
            compacted.push(Some(key));
        }

        self.entries.order = compacted;
        self.entries.tombstones = 0;
    }
}

#[inline]
fn requires_protocol_lookup(value: Value) -> bool {
    !is_fast_lookup_key(value)
}

fn is_fast_lookup_key(value: Value) -> bool {
    if value.as_bool().is_some()
        || value.as_int().is_some()
        || value.as_float().is_some()
        || value_to_bigint(value).is_some()
        || value.is_none()
        || value.is_string()
    {
        return true;
    }

    let Some(ptr) = value.as_object_ptr() else {
        return false;
    };
    match unsafe { (*(ptr as *const ObjectHeader)).type_id } {
        TypeId::STR => true,
        TypeId::TUPLE => {
            let tuple = unsafe { &*(ptr as *const TupleObject) };
            tuple.iter().copied().all(is_fast_lookup_key)
        }
        _ => false,
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
