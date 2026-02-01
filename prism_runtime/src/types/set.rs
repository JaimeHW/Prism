//! Python set object implementation.
//!
//! High-performance mutable set using FxHashSet with Value wrapper.

use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use prism_core::Value;
use rustc_hash::FxHashSet;
use std::hash::{Hash, Hasher};

// =============================================================================
// HashableValue (Same as DictObject)
// =============================================================================

/// Wrapper for Value that implements Hash + Eq for use in HashSet.
///
/// Only hashable types (int, float, str, bool, None, tuple of hashables)
/// can be used as set elements.
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
pub struct SetObject {
    /// Object header.
    pub header: ObjectHeader,
    /// Set items.
    items: FxHashSet<HashableValue>,
}

impl SetObject {
    /// Create an empty set.
    #[inline]
    pub fn new() -> Self {
        Self {
            header: ObjectHeader::new(TypeId::SET),
            items: FxHashSet::default(),
        }
    }

    /// Create a set with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::SET),
            items: FxHashSet::with_capacity_and_hasher(capacity, Default::default()),
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
        self.items.insert(HashableValue(value))
    }

    /// Remove an element from the set.
    ///
    /// Returns true if the element was present.
    #[inline]
    pub fn remove(&mut self, value: Value) -> bool {
        self.items.remove(&HashableValue(value))
    }

    /// Discard an element (same as remove, but doesn't error if missing).
    #[inline]
    pub fn discard(&mut self, value: Value) {
        self.items.remove(&HashableValue(value));
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
        self.items.clear();
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
        self.items.retain(|hv| other.items.contains(hv));
    }

    /// Update self with the difference of self and other.
    pub fn difference_update(&mut self, other: &SetObject) {
        for hv in other.items.iter() {
            self.items.remove(hv);
        }
    }

    /// Update self with the symmetric difference of self and other.
    pub fn symmetric_difference_update(&mut self, other: &SetObject) {
        for hv in other.items.iter() {
            if self.items.contains(hv) {
                self.items.remove(hv);
            } else {
                self.items.insert(*hv);
            }
        }
    }

    /// Clone the set.
    fn clone_set(&self) -> SetObject {
        SetObject {
            header: ObjectHeader::new(TypeId::SET),
            items: self.items.clone(),
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_basic() {
        let mut set = SetObject::new();
        assert!(set.is_empty());

        set.add(Value::int(1).unwrap());
        set.add(Value::int(2).unwrap());
        set.add(Value::int(3).unwrap());

        assert_eq!(set.len(), 3);
        assert!(set.contains(Value::int(1).unwrap()));
        assert!(set.contains(Value::int(2).unwrap()));
        assert!(set.contains(Value::int(3).unwrap()));
        assert!(!set.contains(Value::int(4).unwrap()));
    }

    #[test]
    fn test_set_duplicates() {
        let mut set = SetObject::new();
        assert!(set.add(Value::int(1).unwrap())); // First insert
        assert!(!set.add(Value::int(1).unwrap())); // Duplicate
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_set_remove() {
        let mut set = SetObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]);

        assert!(set.remove(Value::int(2).unwrap()));
        assert!(!set.contains(Value::int(2).unwrap()));
        assert_eq!(set.len(), 2);

        assert!(!set.remove(Value::int(2).unwrap())); // Already removed
    }

    #[test]
    fn test_set_pop() {
        let mut set = SetObject::from_slice(&[Value::int(42).unwrap()]);
        let popped = set.pop();
        assert!(popped.is_some());
        assert_eq!(popped.unwrap().as_int(), Some(42));
        assert!(set.is_empty());
        assert!(set.pop().is_none());
    }

    #[test]
    fn test_set_none_element() {
        let mut set = SetObject::new();
        set.add(Value::none());
        assert!(set.contains(Value::none()));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_set_union() {
        let set1 = SetObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        let set2 = SetObject::from_slice(&[Value::int(2).unwrap(), Value::int(3).unwrap()]);

        let union = set1.union(&set2);
        assert_eq!(union.len(), 3);
        assert!(union.contains(Value::int(1).unwrap()));
        assert!(union.contains(Value::int(2).unwrap()));
        assert!(union.contains(Value::int(3).unwrap()));
    }

    #[test]
    fn test_set_intersection() {
        let set1 = SetObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]);
        let set2 = SetObject::from_slice(&[
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
            Value::int(4).unwrap(),
        ]);

        let inter = set1.intersection(&set2);
        assert_eq!(inter.len(), 2);
        assert!(!inter.contains(Value::int(1).unwrap()));
        assert!(inter.contains(Value::int(2).unwrap()));
        assert!(inter.contains(Value::int(3).unwrap()));
        assert!(!inter.contains(Value::int(4).unwrap()));
    }

    #[test]
    fn test_set_difference() {
        let set1 = SetObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]);
        let set2 = SetObject::from_slice(&[Value::int(2).unwrap(), Value::int(4).unwrap()]);

        let diff = set1.difference(&set2);
        assert_eq!(diff.len(), 2);
        assert!(diff.contains(Value::int(1).unwrap()));
        assert!(!diff.contains(Value::int(2).unwrap()));
        assert!(diff.contains(Value::int(3).unwrap()));
    }

    #[test]
    fn test_set_symmetric_difference() {
        let set1 = SetObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        let set2 = SetObject::from_slice(&[Value::int(2).unwrap(), Value::int(3).unwrap()]);

        let sym_diff = set1.symmetric_difference(&set2);
        assert_eq!(sym_diff.len(), 2);
        assert!(sym_diff.contains(Value::int(1).unwrap()));
        assert!(!sym_diff.contains(Value::int(2).unwrap()));
        assert!(sym_diff.contains(Value::int(3).unwrap()));
    }

    #[test]
    fn test_set_subset_superset() {
        let set1 = SetObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        let set2 = SetObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]);

        assert!(set1.is_subset(&set2));
        assert!(!set2.is_subset(&set1));
        assert!(set2.is_superset(&set1));
        assert!(!set1.is_superset(&set2));
    }

    #[test]
    fn test_set_disjoint() {
        let set1 = SetObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        let set2 = SetObject::from_slice(&[Value::int(3).unwrap(), Value::int(4).unwrap()]);
        let set3 = SetObject::from_slice(&[Value::int(2).unwrap(), Value::int(3).unwrap()]);

        assert!(set1.is_disjoint(&set2));
        assert!(!set1.is_disjoint(&set3));
    }

    #[test]
    fn test_set_update() {
        let mut set1 = SetObject::from_slice(&[Value::int(1).unwrap()]);
        let set2 = SetObject::from_slice(&[Value::int(2).unwrap(), Value::int(3).unwrap()]);

        set1.update(&set2);
        assert_eq!(set1.len(), 3);
        assert!(set1.contains(Value::int(1).unwrap()));
        assert!(set1.contains(Value::int(2).unwrap()));
        assert!(set1.contains(Value::int(3).unwrap()));
    }

    #[test]
    fn test_set_clone() {
        let set1 = SetObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        let set2 = set1.clone();

        assert_eq!(set1.len(), set2.len());
        assert!(set2.contains(Value::int(1).unwrap()));
        assert!(set2.contains(Value::int(2).unwrap()));
    }

    #[test]
    fn test_set_with_floats() {
        let mut set = SetObject::new();
        set.add(Value::float(1.5));
        set.add(Value::float(2.5));
        set.add(Value::float(1.5)); // Duplicate

        assert_eq!(set.len(), 2);
        assert!(set.contains(Value::float(1.5)));
        assert!(set.contains(Value::float(2.5)));
    }

    #[test]
    fn test_set_iter() {
        let set = SetObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]);

        let collected: Vec<_> = set.iter().collect();
        assert_eq!(collected.len(), 3);
    }
}
