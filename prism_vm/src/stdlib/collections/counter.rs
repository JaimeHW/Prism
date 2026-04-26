//! High-performance Counter implementation for counting hashable elements.
//!
//! Counter is a dict subclass for counting hashable elements. Elements are
//! stored as dictionary keys and their counts as dictionary values.
//!
//! # Performance Characteristics
//!
//! | Operation | Time | Notes |
//! |-----------|------|-------|
//! | `count(elem)` | O(1) | Direct hash lookup |
//! | `increment()` | O(1) | In-place update |
//! | `update()` | O(k) | k = elements to add |
//! | `most_common()` | O(n log n) | Requires sorting |
//! | `total()` | O(n) | Summing all counts |
//! | `subtract()` | O(k) | k = elements to subtract |
//!
//! # Memory Efficiency
//!
//! Uses a single HashMap with inline keys for small integers.
//! Element counts are stored as i64 to allow negative values.

use prism_core::Value;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

// =============================================================================
// Value Wrapper for Hashing
// =============================================================================

/// A hashable wrapper around Value for use as HashMap keys.
///
/// Provides consistent hashing semantics matching Python's behavior.
/// With interned strings properly sharing Arcs, bit-level comparison
/// via `Value::PartialEq` and `Value::Hash` is correct for all types.
#[derive(Clone, Debug)]
pub struct HashableValue(pub Value);

impl PartialEq for HashableValue {
    fn eq(&self, other: &Self) -> bool {
        value_eq(&self.0, &other.0)
    }
}

impl Eq for HashableValue {}

impl Hash for HashableValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        value_hash(&self.0, state);
    }
}

/// Compare two Values for equality in a HashMap-safe manner.
///
/// Uses a fast-path raw bits comparison for identical values (with NaN
/// special-casing), then falls through to Value's PartialEq for cross-type
/// numeric coercion (int == float when values match).
fn value_eq(a: &Value, b: &Value) -> bool {
    // Fast path: identical raw bits means same value
    if a.raw_bits() == b.raw_bits() {
        // Float NaN check: NaN != NaN even if bits are identical
        if a.is_float() {
            if let Some(f) = a.as_float() {
                return !f.is_nan();
            }
        }
        return true;
    }

    // Slow path: delegate to Value's PartialEq which handles
    // int/float coercion (e.g., 1 == 1.0)
    *a == *b
}

/// Hash a Value in a manner consistent with value_eq.
///
/// Delegates to Value's built-in Hash implementation.
fn value_hash<H: Hasher>(v: &Value, state: &mut H) {
    v.hash(state);
}

// =============================================================================
// Counter
// =============================================================================

/// A Counter for counting hashable elements.
///
/// # Examples
///
/// ```ignore
/// let mut c = Counter::new();
/// c.update_single(Value::string(intern("a")));
/// c.update_single(Value::string(intern("b")));
/// c.update_single(Value::string(intern("a")));
/// assert_eq!(c.get(&Value::string(intern("a"))), 2);
/// ```
#[derive(Debug, Clone)]
pub struct Counter {
    /// Element -> count mapping.
    counts: HashMap<HashableValue, i64>,
}

impl Counter {
    /// Create a new empty Counter.
    #[inline]
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
        }
    }

    /// Create a Counter with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            counts: HashMap::with_capacity(capacity),
        }
    }

    /// Create a Counter from an iterable of elements.
    pub fn from_iter<I: IntoIterator<Item = Value>>(iter: I) -> Self {
        let mut counter = Self::new();
        counter.update(iter);
        counter
    }

    /// Create a Counter from an iterable of (element, count) pairs.
    pub fn from_pairs<I: IntoIterator<Item = (Value, i64)>>(iter: I) -> Self {
        let mut counter = Self::new();
        for (elem, count) in iter {
            counter.add_count(elem, count);
        }
        counter
    }

    // =========================================================================
    // Core Operations
    // =========================================================================

    /// Get the count for an element (returns 0 if not present).
    #[inline]
    pub fn get(&self, elem: &Value) -> i64 {
        self.counts
            .get(&HashableValue(elem.clone()))
            .copied()
            .unwrap_or(0)
    }

    /// Set the count for an element.
    #[inline]
    pub fn set(&mut self, elem: Value, count: i64) {
        if count == 0 {
            self.counts.remove(&HashableValue(elem));
        } else {
            self.counts.insert(HashableValue(elem), count);
        }
    }

    /// Increment the count for an element by 1.
    #[inline]
    pub fn increment(&mut self, elem: Value) {
        self.add_count(elem, 1);
    }

    /// Add a value to the count for an element.
    #[inline]
    pub fn add_count(&mut self, elem: Value, delta: i64) {
        *self.counts.entry(HashableValue(elem)).or_insert(0) += delta;
    }

    /// Update the counter with elements from an iterable.
    pub fn update<I: IntoIterator<Item = Value>>(&mut self, iter: I) {
        for elem in iter {
            self.increment(elem);
        }
    }

    /// Subtract counts from another counter.
    pub fn subtract<I: IntoIterator<Item = Value>>(&mut self, iter: I) {
        for elem in iter {
            self.add_count(elem, -1);
        }
    }

    /// Subtract counts from (element, count) pairs.
    pub fn subtract_pairs<I: IntoIterator<Item = (Value, i64)>>(&mut self, iter: I) {
        for (elem, count) in iter {
            self.add_count(elem, -count);
        }
    }

    // =========================================================================
    // Query Operations
    // =========================================================================

    /// Return the total of all counts.
    #[inline]
    pub fn total(&self) -> i64 {
        self.counts.values().sum()
    }

    /// Return the number of unique elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.counts.len()
    }

    /// Check if the counter is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    /// Return elements and counts as a Vec, sorted by count (descending).
    pub fn most_common(&self) -> Vec<(Value, i64)> {
        let mut items: Vec<_> = self.counts.iter().map(|(k, v)| (k.0.clone(), *v)).collect();

        // Sort by count descending
        items.sort_by(|a, b| b.1.cmp(&a.1));
        items
    }

    /// Return the n most common elements.
    pub fn most_common_n(&self, n: usize) -> Vec<(Value, i64)> {
        let mut items = self.most_common();
        items.truncate(n);
        items
    }

    /// Return elements (may include duplicates based on count).
    pub fn elements(&self) -> CounterElements<'_> {
        CounterElements {
            iter: self.counts.iter(),
            current: None,
            remaining: 0,
        }
    }

    /// Check if the counter contains an element.
    #[inline]
    pub fn contains(&self, elem: &Value) -> bool {
        self.counts.contains_key(&HashableValue(elem.clone()))
    }

    /// Remove an element and return its count.
    pub fn remove(&mut self, elem: &Value) -> Option<i64> {
        self.counts.remove(&HashableValue(elem.clone()))
    }

    /// Clear all elements.
    pub fn clear(&mut self) {
        self.counts.clear();
    }

    // =========================================================================
    // Set-like Operations
    // =========================================================================

    /// Return a new Counter with counts from both counters.
    pub fn add(&self, other: &Counter) -> Counter {
        let mut result = self.clone();
        for (elem, count) in &other.counts {
            result.add_count(elem.0.clone(), *count);
        }
        result
    }

    /// Return a new Counter with counts subtracted.
    pub fn sub(&self, other: &Counter) -> Counter {
        let mut result = self.clone();
        for (elem, count) in &other.counts {
            result.add_count(elem.0.clone(), -*count);
        }
        result
    }

    /// Return intersection (minimum of counts).
    pub fn intersection(&self, other: &Counter) -> Counter {
        let mut result = Counter::new();
        for (elem, count) in &self.counts {
            let other_count = other.get(&elem.0);
            if other_count > 0 && *count > 0 {
                result.set(elem.0.clone(), (*count).min(other_count));
            }
        }
        result
    }

    /// Return union (maximum of counts).
    pub fn union(&self, other: &Counter) -> Counter {
        let mut result = self.clone();
        for (elem, count) in &other.counts {
            let current = result.get(&elem.0);
            if *count > current {
                result.set(elem.0.clone(), *count);
            }
        }
        result
    }

    /// Keep only positive counts.
    pub fn positive(&mut self) {
        self.counts.retain(|_, count| *count > 0);
    }

    // =========================================================================
    // Iterator Support
    // =========================================================================

    /// Iterator over (element, count) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&Value, &i64)> {
        self.counts.iter().map(|(k, v)| (&k.0, v))
    }

    /// Iterator over elements (key references).
    pub fn keys(&self) -> impl Iterator<Item = &Value> {
        self.counts.keys().map(|k| &k.0)
    }

    /// Iterator over counts.
    pub fn values(&self) -> impl Iterator<Item = &i64> {
        self.counts.values()
    }
}

impl Default for Counter {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for Counter {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.counts
            .iter()
            .all(|(k, v)| other.counts.get(k) == Some(v))
    }
}

impl Eq for Counter {}

// =============================================================================
// Iterators
// =============================================================================

/// Iterator that yields each element count times.
pub struct CounterElements<'a> {
    iter: std::collections::hash_map::Iter<'a, HashableValue, i64>,
    current: Option<&'a Value>,
    remaining: i64,
}

impl<'a> Iterator for CounterElements<'a> {
    type Item = &'a Value;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.remaining > 0 {
                self.remaining -= 1;
                return self.current;
            }

            // Get next element
            match self.iter.next() {
                Some((elem, &count)) if count > 0 => {
                    self.current = Some(&elem.0);
                    self.remaining = count - 1;
                    return Some(&elem.0);
                }
                Some(_) => continue, // Skip non-positive counts
                None => return None,
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod counter_tests;
