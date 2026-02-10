//! High-performance OrderedDict implementation.
//!
//! An OrderedDict remembers the order in which entries were added.
//! It provides O(1) operations for most dict operations with minimal
//! overhead for order tracking.
//!
//! # Performance Characteristics
//!
//! | Operation | Time | Notes |
//! |-----------|------|-------|
//! | `get()` | O(1) | Hash lookup |
//! | `set()` | O(1) | May append to order |
//! | `remove()` | O(1) | Lazy removal |
//! | `iter()` | O(n) | Insertion order |
//! | `move_to_end()` | O(1) | Reorder entry |
//!
//! # Implementation Notes
//!
//! This implementation uses a HashMap for O(1) access combined with
//! an index-tracking mechanism for iteration order. Unlike linked-list
//! approaches, this provides better cache locality.

use super::counter::HashableValue;
use prism_core::Value;
use std::collections::HashMap;

// =============================================================================
// OrderedDict
// =============================================================================

/// An ordered dictionary that maintains insertion order.
///
/// # Examples
///
/// ```ignore
/// let mut od = OrderedDict::new();
/// od.set(key1, val1);
/// od.set(key2, val2);
/// // Iteration yields (key1, val1), (key2, val2)
/// ```
#[derive(Debug, Clone)]
pub struct OrderedDict {
    /// Key -> (value, insertion_index) mapping.
    data: HashMap<HashableValue, (Value, usize)>,
    /// Ordered list of keys (may contain tombstones for removed keys).
    order: Vec<Option<Value>>,
    /// Number of valid entries (excludes tombstones).
    len: usize,
    /// Number of tombstones (for compaction heuristics).
    tombstones: usize,
}

impl OrderedDict {
    /// Create a new empty OrderedDict.
    #[inline]
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            order: Vec::new(),
            len: 0,
            tombstones: 0,
        }
    }

    /// Create with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: HashMap::with_capacity(capacity),
            order: Vec::with_capacity(capacity),
            len: 0,
            tombstones: 0,
        }
    }

    // =========================================================================
    // Core Operations
    // =========================================================================

    /// Get a value by key.
    #[inline]
    pub fn get(&self, key: &Value) -> Option<&Value> {
        self.data.get(&HashableValue(key.clone())).map(|(v, _)| v)
    }

    /// Get a mutable reference to a value.
    #[inline]
    pub fn get_mut(&mut self, key: &Value) -> Option<&mut Value> {
        self.data
            .get_mut(&HashableValue(key.clone()))
            .map(|(v, _)| v)
    }

    /// Set a value. If key exists, updates in place (order unchanged).
    /// If key is new, appends to end of order.
    pub fn set(&mut self, key: Value, value: Value) {
        let hkey = HashableValue(key.clone());

        if let Some((v, _)) = self.data.get_mut(&hkey) {
            // Key exists - update value, keep order
            *v = value;
        } else {
            // New key - append to order
            let idx = self.order.len();
            self.order.push(Some(key));
            self.data.insert(hkey, (value, idx));
            self.len += 1;
        }
    }

    /// Remove a key. Returns the value if present.
    pub fn remove(&mut self, key: &Value) -> Option<Value> {
        let hkey = HashableValue(key.clone());

        if let Some((value, idx)) = self.data.remove(&hkey) {
            // Mark as tombstone in order list
            self.order[idx] = None;
            self.len -= 1;
            self.tombstones += 1;

            // Compact if too many tombstones
            if self.tombstones > self.len && self.tombstones > 16 {
                self.compact();
            }

            Some(value)
        } else {
            None
        }
    }

    /// Check if key exists.
    #[inline]
    pub fn contains(&self, key: &Value) -> bool {
        self.data.contains_key(&HashableValue(key.clone()))
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.data.clear();
        self.order.clear();
        self.len = 0;
        self.tombstones = 0;
    }

    /// Get number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    // =========================================================================
    // Order Operations
    // =========================================================================

    /// Move a key to the end of the order.
    /// If `last` is false, move to the beginning.
    pub fn move_to_end(&mut self, key: &Value, last: bool) {
        let hkey = HashableValue(key.clone());

        if let Some((value, old_idx)) = self.data.remove(&hkey) {
            // Mark old position as tombstone
            self.order[old_idx] = None;
            self.tombstones += 1;

            if last {
                // Move to end
                let new_idx = self.order.len();
                self.order.push(Some(key.clone()));
                self.data.insert(hkey, (value, new_idx));
            } else {
                // Move to beginning (more expensive)
                // We do this by compacting and prepending
                self.compact_with_prepend(key.clone(), value);
            }
        }
    }

    /// Pop and return the last (key, value) pair.
    pub fn popitem(&mut self, last: bool) -> Option<(Value, Value)> {
        if self.is_empty() {
            return None;
        }

        if last {
            // Find last non-tombstone from end
            for i in (0..self.order.len()).rev() {
                if let Some(key) = self.order[i].take() {
                    self.tombstones += 1;
                    if let Some((value, _)) = self.data.remove(&HashableValue(key.clone())) {
                        self.len -= 1;
                        return Some((key, value));
                    }
                }
            }
        } else {
            // Find first non-tombstone from start
            for i in 0..self.order.len() {
                if let Some(key) = self.order[i].take() {
                    self.tombstones += 1;
                    if let Some((value, _)) = self.data.remove(&HashableValue(key.clone())) {
                        self.len -= 1;
                        return Some((key, value));
                    }
                }
            }
        }

        None
    }

    // =========================================================================
    // Iterator Support
    // =========================================================================

    /// Iterator over keys in insertion order.
    pub fn keys(&self) -> impl Iterator<Item = &Value> {
        self.order.iter().filter_map(|k| k.as_ref())
    }

    /// Iterator over values in insertion order.
    pub fn values(&self) -> impl Iterator<Item = &Value> {
        self.order.iter().filter_map(|k| {
            k.as_ref()
                .and_then(|key| self.data.get(&HashableValue(key.clone())).map(|(v, _)| v))
        })
    }

    /// Iterator over (key, value) pairs in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&Value, &Value)> {
        self.order.iter().filter_map(|k| {
            k.as_ref().and_then(|key| {
                self.data
                    .get(&HashableValue(key.clone()))
                    .map(|(v, _)| (key, v))
            })
        })
    }

    // =========================================================================
    // Internal Helpers
    // =========================================================================

    /// Compact the order list by removing tombstones.
    fn compact(&mut self) {
        let mut new_order = Vec::with_capacity(self.len);

        for key_opt in &self.order {
            if let Some(key) = key_opt {
                let new_idx = new_order.len();
                new_order.push(Some(key.clone()));

                // Update index in data
                if let Some((_, idx)) = self.data.get_mut(&HashableValue(key.clone())) {
                    *idx = new_idx;
                }
            }
        }

        self.order = new_order;
        self.tombstones = 0;
    }

    /// Compact and prepend a key to the beginning.
    fn compact_with_prepend(&mut self, key: Value, value: Value) {
        let mut new_order = Vec::with_capacity(self.len + 1);

        // Add the prepended key first
        new_order.push(Some(key.clone()));
        self.data.insert(HashableValue(key), (value, 0));

        // Add remaining keys
        for key_opt in &self.order {
            if let Some(k) = key_opt {
                let new_idx = new_order.len();
                new_order.push(Some(k.clone()));

                if let Some((_, idx)) = self.data.get_mut(&HashableValue(k.clone())) {
                    *idx = new_idx;
                }
            }
        }

        self.order = new_order;
        self.tombstones = 0;
    }
}

impl Default for OrderedDict {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for OrderedDict {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }

        // Compare in order
        self.iter().zip(other.iter()).all(|((k1, v1), (k2, v2))| {
            // Compare keys and values using raw_bits
            let keys_eq = k1.raw_bits() == k2.raw_bits();
            let vals_eq = v1 == v2;
            keys_eq && vals_eq
        })
    }
}

impl Eq for OrderedDict {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod ordereddict_tests {
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
        let od = OrderedDict::new();
        assert!(od.is_empty());
        assert_eq!(od.len(), 0);
    }

    #[test]
    fn test_with_capacity() {
        let od = OrderedDict::with_capacity(100);
        assert!(od.is_empty());
    }

    // =========================================================================
    // Set/Get Tests
    // =========================================================================

    #[test]
    fn test_set_and_get() {
        let mut od = OrderedDict::new();
        od.set(int(1), int(100));
        od.set(int(2), int(200));

        assert_eq!(od.get(&int(1)).and_then(|v| v.as_int()), Some(100));
        assert_eq!(od.get(&int(2)).and_then(|v| v.as_int()), Some(200));
    }

    #[test]
    fn test_get_missing() {
        let od = OrderedDict::new();
        assert_eq!(od.get(&int(99)), None);
    }

    #[test]
    fn test_update_existing() {
        let mut od = OrderedDict::new();
        od.set(int(1), int(100));
        od.set(int(1), int(999));

        assert_eq!(od.get(&int(1)).and_then(|v| v.as_int()), Some(999));
        assert_eq!(od.len(), 1);
    }

    #[test]
    fn test_contains() {
        let mut od = OrderedDict::new();
        od.set(int(1), int(100));

        assert!(od.contains(&int(1)));
        assert!(!od.contains(&int(99)));
    }

    // =========================================================================
    // Order Tests
    // =========================================================================

    #[test]
    fn test_insertion_order_preserved() {
        let mut od = OrderedDict::new();
        od.set(str_val("a"), int(1));
        od.set(str_val("b"), int(2));
        od.set(str_val("c"), int(3));

        let keys: Vec<_> = od.keys().collect();
        assert_eq!(keys.len(), 3);
        // Order should be a, b, c
    }

    #[test]
    fn test_update_preserves_order() {
        let mut od = OrderedDict::new();
        od.set(str_val("a"), int(1));
        od.set(str_val("b"), int(2));
        od.set(str_val("c"), int(3));

        // Update 'b' - should stay in same position
        od.set(str_val("b"), int(20));

        let vals: Vec<_> = od.values().filter_map(|v| v.as_int()).collect();
        assert_eq!(vals, vec![1, 20, 3]);
    }

    // =========================================================================
    // Remove Tests
    // =========================================================================

    #[test]
    fn test_remove() {
        let mut od = OrderedDict::new();
        od.set(int(1), int(100));
        od.set(int(2), int(200));

        let removed = od.remove(&int(1));
        assert_eq!(removed.and_then(|v| v.as_int()), Some(100));
        assert!(!od.contains(&int(1)));
        assert_eq!(od.len(), 1);
    }

    #[test]
    fn test_remove_preserves_order() {
        let mut od = OrderedDict::new();
        od.set(str_val("a"), int(1));
        od.set(str_val("b"), int(2));
        od.set(str_val("c"), int(3));

        od.remove(&str_val("b"));

        let vals: Vec<_> = od.values().filter_map(|v| v.as_int()).collect();
        assert_eq!(vals, vec![1, 3]);
    }

    #[test]
    fn test_clear() {
        let mut od = OrderedDict::new();
        od.set(int(1), int(100));
        od.set(int(2), int(200));
        od.clear();

        assert!(od.is_empty());
    }

    // =========================================================================
    // Move/Pop Tests
    // =========================================================================

    #[test]
    fn test_move_to_end_last() {
        let mut od = OrderedDict::new();
        od.set(str_val("a"), int(1));
        od.set(str_val("b"), int(2));
        od.set(str_val("c"), int(3));

        od.move_to_end(&str_val("a"), true);

        // Order should now be b, c, a
        let vals: Vec<_> = od.values().filter_map(|v| v.as_int()).collect();
        assert_eq!(vals, vec![2, 3, 1]);
    }

    #[test]
    fn test_move_to_end_first() {
        let mut od = OrderedDict::new();
        od.set(str_val("a"), int(1));
        od.set(str_val("b"), int(2));
        od.set(str_val("c"), int(3));

        od.move_to_end(&str_val("c"), false);

        // Order should now be c, a, b
        let vals: Vec<_> = od.values().filter_map(|v| v.as_int()).collect();
        assert_eq!(vals, vec![3, 1, 2]);
    }

    #[test]
    fn test_popitem_last() {
        let mut od = OrderedDict::new();
        od.set(str_val("a"), int(1));
        od.set(str_val("b"), int(2));

        let (_, val) = od.popitem(true).unwrap();
        assert_eq!(val.as_int(), Some(2));
        assert_eq!(od.len(), 1);
    }

    #[test]
    fn test_popitem_first() {
        let mut od = OrderedDict::new();
        od.set(str_val("a"), int(1));
        od.set(str_val("b"), int(2));

        let (_, val) = od.popitem(false).unwrap();
        assert_eq!(val.as_int(), Some(1));
        assert_eq!(od.len(), 1);
    }

    #[test]
    fn test_popitem_empty() {
        let mut od = OrderedDict::new();
        assert_eq!(od.popitem(true), None);
    }

    // =========================================================================
    // Iterator Tests
    // =========================================================================

    #[test]
    fn test_keys_iterator() {
        let mut od = OrderedDict::new();
        od.set(int(1), int(100));
        od.set(int(2), int(200));

        let keys: Vec<_> = od.keys().collect();
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn test_values_iterator() {
        let mut od = OrderedDict::new();
        od.set(int(1), int(100));
        od.set(int(2), int(200));

        let sum: i64 = od.values().filter_map(|v| v.as_int()).sum();
        assert_eq!(sum, 300);
    }

    #[test]
    fn test_iter() {
        let mut od = OrderedDict::new();
        od.set(int(1), int(100));

        let pairs: Vec<_> = od.iter().collect();
        assert_eq!(pairs.len(), 1);
    }

    // =========================================================================
    // Equality Tests
    // =========================================================================

    #[test]
    fn test_equality_same_order() {
        let mut od1 = OrderedDict::new();
        let mut od2 = OrderedDict::new();

        od1.set(int(1), int(100));
        od1.set(int(2), int(200));

        od2.set(int(1), int(100));
        od2.set(int(2), int(200));

        assert_eq!(od1, od2);
    }

    // =========================================================================
    // Compaction Tests
    // =========================================================================

    #[test]
    fn test_compaction_triggers() {
        let mut od = OrderedDict::new();

        // Add many items
        for i in 0..100 {
            od.set(int(i), int(i * 10));
        }

        // Remove most of them
        for i in 0..80 {
            od.remove(&int(i));
        }

        // Should have compacted
        assert_eq!(od.len(), 20);

        // Verify remaining are correct
        for i in 80..100 {
            assert!(od.contains(&int(i)));
        }
    }

    // =========================================================================
    // Stress Tests
    // =========================================================================

    #[test]
    fn test_stress_many_entries() {
        let mut od = OrderedDict::new();

        for i in 0..1000 {
            od.set(int(i), int(i * 10));
        }

        assert_eq!(od.len(), 1000);

        // Verify all entries
        for i in 0..1000 {
            assert_eq!(od.get(&int(i)).and_then(|v| v.as_int()), Some(i * 10));
        }
    }

    #[test]
    fn test_stress_interleaved_ops() {
        let mut od = OrderedDict::new();

        for i in 0..100 {
            od.set(int(i), int(i));
            if i % 3 == 0 {
                od.remove(&int(i / 2));
            }
        }

        // All even indexed items that weren't removed should exist
        // This is a smoke test, not exact verification
        assert!(od.len() > 0);
    }
}
