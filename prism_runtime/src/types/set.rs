//! Python set object implementation.
//!
//! High-performance mutable set using FxHashSet with Value wrapper.

use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use crate::types::hashable::HashableValue;
use prism_core::Value;
use rustc_hash::FxHashSet;
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
            hash_cache: AtomicI64::new(HASH_CACHE_EMPTY),
        }
    }

    /// Create a set with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::SET),
            items: FxHashSet::with_capacity_and_hasher(capacity, Default::default()),
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

    /// Add an element to the set.
    ///
    /// Returns true if the element was not already present.
    #[inline]
    pub fn add(&mut self, value: Value) -> bool {
        let inserted = self.items.insert(HashableValue(value));
        if inserted {
            self.clear_hash_cache();
        }
        inserted
    }

    /// Remove an element from the set.
    ///
    /// Returns true if the element was present.
    #[inline]
    pub fn remove(&mut self, value: Value) -> bool {
        let removed = self.items.remove(&HashableValue(value));
        if removed {
            self.clear_hash_cache();
        }
        removed
    }

    /// Discard an element (same as remove, but doesn't error if missing).
    #[inline]
    pub fn discard(&mut self, value: Value) {
        if self.items.remove(&HashableValue(value)) {
            self.clear_hash_cache();
        }
    }

    /// Check if the set contains an element.
    #[inline]
    pub fn contains(&self, value: Value) -> bool {
        self.items.contains(&HashableValue(value))
    }

    /// Remove and return an arbitrary element.
    ///
    /// Returns None if the set is empty.
    pub fn pop(&mut self) -> Option<Value> {
        if let Some(hv) = self.items.iter().next().copied() {
            self.items.remove(&hv);
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
        for hv in other.items.iter() {
            result.items.insert(*hv);
        }
        result
    }

    /// Return the intersection of two sets as a new set.
    pub fn intersection(&self, other: &SetObject) -> SetObject {
        let mut result = SetObject::new();
        for hv in self.items.iter() {
            if other.items.contains(hv) {
                result.items.insert(*hv);
            }
        }
        result
    }

    /// Return the difference of two sets as a new set (self - other).
    pub fn difference(&self, other: &SetObject) -> SetObject {
        let mut result = SetObject::new();
        for hv in self.items.iter() {
            if !other.items.contains(hv) {
                result.items.insert(*hv);
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
                result.items.insert(*hv);
            }
        }
        // Elements in other but not self
        for hv in other.items.iter() {
            if !self.items.contains(hv) {
                result.items.insert(*hv);
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
        for hv in other.items.iter() {
            self.items.insert(*hv);
        }
    }

    /// Update self with the intersection of self and other.
    pub fn intersection_update(&mut self, other: &SetObject) {
        let old_len = self.items.len();
        self.items.retain(|hv| other.items.contains(hv));
        if self.items.len() != old_len {
            self.clear_hash_cache();
        }
    }

    /// Update self with the difference of self and other.
    pub fn difference_update(&mut self, other: &SetObject) {
        let old_len = self.items.len();
        for hv in other.items.iter() {
            self.items.remove(hv);
        }
        if self.items.len() != old_len {
            self.clear_hash_cache();
        }
    }

    /// Update self with the symmetric difference of self and other.
    pub fn symmetric_difference_update(&mut self, other: &SetObject) {
        let old_len = self.items.len();
        for hv in other.items.iter() {
            if self.items.contains(hv) {
                self.items.remove(hv);
            } else {
                self.items.insert(*hv);
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

    /// Clone the set.
    fn clone_set(&self) -> SetObject {
        SetObject {
            header: ObjectHeader::new(TypeId::SET),
            items: self.items.clone(),
            hash_cache: AtomicI64::new(self.hash_cache.load(Ordering::Relaxed)),
        }
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
