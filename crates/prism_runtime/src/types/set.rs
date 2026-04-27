//! Python set object implementation.
//!
//! High-performance mutable set using FxHashSet with Value wrapper.

use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use crate::types::hashable::HashableValue;
use crate::types::int::value_to_bigint;
use crate::types::string::value_as_string_ref;
use crate::types::tuple::TupleObject;
use prism_core::Value;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::atomic::{AtomicI64, Ordering};

const HASH_CACHE_EMPTY: i64 = i64::MIN;

// =============================================================================
// SetObject
// =============================================================================

/// Python set object.
///
/// Mutable unordered collection of unique hashable elements.
/// Uses FxHashSet for fast insertion, removal, and membership tests.
///
/// # Performance
///
/// - Contains: O(1) average
/// - Insert: O(1) average
/// - Remove: O(1) average
/// - Union/Intersection: O(n)
#[repr(C)]
#[derive(Debug)]
pub struct SetObject {
    /// Object header.
    pub header: ObjectHeader,
    /// Set items.
    items: FxHashSet<HashableValue>,
    /// Python hash values computed by the VM for keys that entered through a
    /// protocol-aware path.
    hashes: FxHashMap<HashableValue, i64>,
    /// Keys whose equality may execute Python code during collision checks.
    protocol_keys: Option<FxHashSet<HashableValue>>,
    /// Cached hash for immutable frozenset views of this storage.
    hash_cache: AtomicI64,
}

impl SetObject {
    /// Create an empty set.
    #[inline]
    pub fn new() -> Self {
        Self {
            header: ObjectHeader::new(TypeId::SET),
            items: FxHashSet::default(),
            hashes: FxHashMap::default(),
            protocol_keys: None,
            hash_cache: AtomicI64::new(HASH_CACHE_EMPTY),
        }
    }

    /// Create a set with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::SET),
            items: FxHashSet::with_capacity_and_hasher(capacity, Default::default()),
            hashes: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            protocol_keys: None,
            hash_cache: AtomicI64::new(HASH_CACHE_EMPTY),
        }
    }

    /// Create a set from a slice of values.
    pub fn from_slice(values: &[Value]) -> Self {
        let mut set = Self::with_capacity(values.len());
        for v in values {
            set.add(*v);
        }
        set
    }

    /// Create a set from an iterator.
    pub fn from_iter<I: IntoIterator<Item = Value>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (size_hint, _) = iter.size_hint();
        let mut set = Self::with_capacity(size_hint);
        for v in iter {
            set.add(v);
        }
        set
    }

    /// Get the number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if the set is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Return true when at least one key can require Python-level equality for
    /// collision resolution.
    #[inline]
    pub fn has_protocol_keys(&self) -> bool {
        self.protocol_keys
            .as_ref()
            .is_some_and(|keys| !keys.is_empty())
    }

    /// Add an element to the set.
    ///
    /// Returns true if the element was not already present.
    #[inline]
    pub fn add(&mut self, value: Value) -> bool {
        self.insert(value, None, requires_protocol_lookup(value))
    }

    /// Add an element after the VM has computed the Python hash.
    #[inline]
    pub fn add_with_hash(&mut self, value: Value, hash: i64) -> bool {
        self.insert(value, Some(hash), requires_protocol_lookup(value))
    }

    /// Add an element when the caller already classified whether lookup may
    /// need Python-level equality.
    #[inline]
    pub fn add_with_hash_and_protocol_lookup(
        &mut self,
        value: Value,
        hash: i64,
        requires_protocol_lookup: bool,
    ) -> bool {
        self.insert(value, Some(hash), requires_protocol_lookup)
    }

    #[inline]
    fn insert(&mut self, value: Value, hash: Option<i64>, requires_protocol_lookup: bool) -> bool {
        let key = HashableValue(value);
        let inserted = self.items.insert(key);
        if inserted {
            if requires_protocol_lookup {
                self.mark_protocol_key(key);
            }
            if let Some(hash) = hash {
                self.hashes.insert(key, hash);
            }
            self.clear_hash_cache();
        } else if let Some(hash) = hash {
            self.hashes.entry(key).or_insert(hash);
        }
        inserted
    }

    /// Remove an element from the set.
    ///
    /// Returns true if the element was present.
    #[inline]
    pub fn remove(&mut self, value: Value) -> bool {
        let key = HashableValue(value);
        if let Some(stored_key) = self.items.take(&key) {
            self.hashes.remove(&stored_key);
            self.unmark_protocol_key(stored_key);
            self.clear_hash_cache();
            true
        } else {
            false
        }
    }

    /// Discard an element (same as remove, but doesn't error if missing).
    #[inline]
    pub fn discard(&mut self, value: Value) {
        self.remove(value);
    }

    /// Check if the set contains an element.
    #[inline]
    pub fn contains(&self, value: Value) -> bool {
        self.items.contains(&HashableValue(value))
    }

    /// Return the cached Python hash for a stored key, when available.
    #[inline]
    pub fn stored_hash(&self, key: Value) -> Option<i64> {
        self.hashes.get(&HashableValue(key)).copied()
    }

    /// Remove and return an arbitrary element.
    ///
    /// Returns None if the set is empty.
    pub fn pop(&mut self) -> Option<Value> {
        if let Some(hv) = self.items.iter().next().copied() {
            self.items.remove(&hv);
            self.hashes.remove(&hv);
            self.unmark_protocol_key(hv);
            self.clear_hash_cache();
            Some(hv.0)
        } else {
            None
        }
    }

    /// Clear all elements from the set.
    #[inline]
    pub fn clear(&mut self) {
        if !self.items.is_empty() {
            self.items.clear();
            self.hashes.clear();
            self.protocol_keys = None;
            self.clear_hash_cache();
        }
    }

    /// Get an iterator over the elements.
    pub fn iter(&self) -> impl Iterator<Item = Value> + '_ {
        self.items.iter().map(|hv| hv.0)
    }

    /// Return the union of two sets as a new set.
    pub fn union(&self, other: &SetObject) -> SetObject {
        let mut result = self.clone_set();
        let mut changed = false;
        for hv in other.items.iter() {
            changed |= result.insert_from(*hv, other);
        }
        if changed {
            result.clear_hash_cache();
        }
        result
    }

    /// Return the intersection of two sets as a new set.
    pub fn intersection(&self, other: &SetObject) -> SetObject {
        let mut result = SetObject::new();
        for hv in self.items.iter() {
            if other.items.contains(hv) {
                result.insert_from(*hv, self);
            }
        }
        result
    }

    /// Return the difference of two sets as a new set (self - other).
    pub fn difference(&self, other: &SetObject) -> SetObject {
        let mut result = SetObject::new();
        for hv in self.items.iter() {
            if !other.items.contains(hv) {
                result.insert_from(*hv, self);
            }
        }
        result
    }

    /// Return the symmetric difference of two sets as a new set.
    pub fn symmetric_difference(&self, other: &SetObject) -> SetObject {
        let mut result = SetObject::new();
        // Elements in self but not other
        for hv in self.items.iter() {
            if !other.items.contains(hv) {
                result.insert_from(*hv, self);
            }
        }
        // Elements in other but not self
        for hv in other.items.iter() {
            if !self.items.contains(hv) {
                result.insert_from(*hv, other);
            }
        }
        result
    }

    /// Check if self is a subset of other.
    pub fn is_subset(&self, other: &SetObject) -> bool {
        if self.len() > other.len() {
            return false;
        }
        self.items.iter().all(|hv| other.items.contains(hv))
    }

    /// Check if self is a superset of other.
    #[inline]
    pub fn is_superset(&self, other: &SetObject) -> bool {
        other.is_subset(self)
    }

    /// Check if sets are disjoint (no common elements).
    pub fn is_disjoint(&self, other: &SetObject) -> bool {
        // Iterate over smaller set for efficiency
        if self.len() <= other.len() {
            !self.items.iter().any(|hv| other.items.contains(hv))
        } else {
            !other.items.iter().any(|hv| self.items.contains(hv))
        }
    }

    /// Update self with the union of self and other.
    pub fn update(&mut self, other: &SetObject) {
        let mut changed = false;
        for hv in other.items.iter() {
            changed |= self.insert_from(*hv, other);
        }
        if changed {
            self.clear_hash_cache();
        }
    }

    /// Update self with the intersection of self and other.
    pub fn intersection_update(&mut self, other: &SetObject) {
        let old_len = self.items.len();
        self.items.retain(|hv| other.items.contains(hv));
        if self.items.len() != old_len {
            self.retain_metadata_for_items();
            self.clear_hash_cache();
        }
    }

    /// Update self with the difference of self and other.
    pub fn difference_update(&mut self, other: &SetObject) {
        let old_len = self.items.len();
        for hv in other.items.iter() {
            if let Some(stored_key) = self.items.take(hv) {
                self.hashes.remove(&stored_key);
                self.unmark_protocol_key(stored_key);
            }
        }
        if self.items.len() != old_len {
            self.clear_hash_cache();
        }
    }

    /// Update self with the symmetric difference of self and other.
    pub fn symmetric_difference_update(&mut self, other: &SetObject) {
        let old_len = self.items.len();
        for hv in other.items.iter() {
            if let Some(stored_key) = self.items.take(hv) {
                self.hashes.remove(&stored_key);
                self.unmark_protocol_key(stored_key);
            } else {
                self.insert_from(*hv, other);
            }
        }
        if self.items.len() != old_len || !other.items.is_empty() {
            self.clear_hash_cache();
        }
    }

    /// Return the cached frozenset hash, if one has been computed.
    #[inline]
    pub fn cached_hash(&self) -> Option<i64> {
        match self.hash_cache.load(Ordering::Relaxed) {
            HASH_CACHE_EMPTY => None,
            hash => Some(hash),
        }
    }

    /// Cache the frozenset hash for immutable users of this storage.
    #[inline]
    pub fn set_cached_hash(&self, hash: i64) {
        self.hash_cache.store(hash, Ordering::Relaxed);
    }

    #[inline]
    fn clear_hash_cache(&self) {
        self.hash_cache.store(HASH_CACHE_EMPTY, Ordering::Relaxed);
    }

    #[inline]
    fn insert_from(&mut self, key: HashableValue, source: &SetObject) -> bool {
        let inserted = self.items.insert(key);
        if inserted {
            if let Some(hash) = source.hashes.get(&key).copied() {
                self.hashes.insert(key, hash);
            }
            if source
                .protocol_keys
                .as_ref()
                .is_some_and(|keys| keys.contains(&key))
            {
                self.mark_protocol_key(key);
            }
        }
        inserted
    }

    #[inline]
    fn mark_protocol_key(&mut self, key: HashableValue) {
        self.protocol_keys
            .get_or_insert_with(FxHashSet::default)
            .insert(key);
    }

    #[inline]
    fn unmark_protocol_key(&mut self, key: HashableValue) {
        let Some(keys) = self.protocol_keys.as_mut() else {
            return;
        };
        keys.remove(&key);
        if keys.is_empty() {
            self.protocol_keys = None;
        }
    }

    fn retain_metadata_for_items(&mut self) {
        self.hashes.retain(|key, _| self.items.contains(key));
        if let Some(keys) = self.protocol_keys.as_mut() {
            keys.retain(|key| self.items.contains(key));
            if keys.is_empty() {
                self.protocol_keys = None;
            }
        }
    }

    /// Clone the set.
    fn clone_set(&self) -> SetObject {
        SetObject {
            header: ObjectHeader::new(TypeId::SET),
            items: self.items.clone(),
            hashes: self.hashes.clone(),
            protocol_keys: self.protocol_keys.clone(),
            hash_cache: AtomicI64::new(self.hash_cache.load(Ordering::Relaxed)),
        }
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
        || value_as_string_ref(value).is_some()
    {
        return true;
    }

    let Some(ptr) = value.as_object_ptr() else {
        return false;
    };
    match unsafe { (*(ptr as *const ObjectHeader)).type_id } {
        TypeId::TUPLE => {
            let tuple = unsafe { &*(ptr as *const TupleObject) };
            tuple.iter().copied().all(is_fast_lookup_key)
        }
        _ => false,
    }
}

impl Default for SetObject {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SetObject {
    fn clone(&self) -> Self {
        self.clone_set()
    }
}

impl PyObject for SetObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}
