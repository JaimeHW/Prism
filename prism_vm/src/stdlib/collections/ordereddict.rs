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
mod ordereddict_tests;
