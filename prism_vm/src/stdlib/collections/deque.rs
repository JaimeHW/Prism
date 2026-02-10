//! High-performance double-ended queue (deque) implementation.
//!
//! This deque uses a ring buffer (circular array) for optimal cache locality
//! and O(1) operations at both ends. Unlike linked-list implementations,
//! this provides better cache performance for sequential operations.
//!
//! # Performance Characteristics
//!
//! | Operation | Time | Notes |
//! |-----------|------|-------|
//! | `append()` | O(1) amortized | May trigger growth |
//! | `appendleft()` | O(1) amortized | May trigger growth |
//! | `pop()` | O(1) | Returns None if empty |
//! | `popleft()` | O(1) | Returns None if empty |
//! | `extend()` | O(k) | k = elements added |
//! | `rotate()` | O(k) | k = rotation amount |
//! | `clear()` | O(1) | Resets indices |
//! | `len()` | O(1) | Cached |
//! | `index(i)` | O(1) | Direct array access |
//!
//! # Memory Layout
//!
//! The ring buffer uses a contiguous memory allocation with head/tail
//! indices that wrap around. This provides:
//! - Better cache utilization than linked lists
//! - Fewer allocations (single array vs many nodes)
//! - Predictable memory layout for prefetching
//!
//! # Growth Strategy
//!
//! When full, the buffer doubles in size (minimum 16 elements).
//! This gives O(1) amortized insertion cost.

use prism_core::Value;
use std::ops::{Index, IndexMut};

// =============================================================================
// Constants
// =============================================================================

/// Minimum capacity for deque allocation.
/// Chosen to be cache-line aligned (64 bytes / 8 bytes per Value = 8).
/// We use 16 for slightly more room before first resize.
const MIN_CAPACITY: usize = 16;

/// Growth factor when resizing (2x).
const GROWTH_FACTOR: usize = 2;

// =============================================================================
// Deque
// =============================================================================

/// A double-ended queue with O(1) operations at both ends.
///
/// # Examples
///
/// ```ignore
/// let mut d = Deque::new();
/// d.append(Value::int_unchecked(1));
/// d.appendleft(Value::int_unchecked(0));
/// assert_eq!(d.popleft(), Some(Value::int_unchecked(0)));
/// ```
#[derive(Debug, Clone)]
pub struct Deque {
    /// Ring buffer storage.
    buffer: Vec<Option<Value>>,
    /// Index of the first element.
    head: usize,
    /// Index one past the last element.
    tail: usize,
    /// Number of elements in the deque.
    len: usize,
    /// Optional maximum length (Python's maxlen parameter).
    maxlen: Option<usize>,
}

impl Deque {
    /// Create a new empty deque.
    #[inline]
    pub fn new() -> Self {
        Self::with_capacity(MIN_CAPACITY)
    }

    /// Create a deque with a specific initial capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(MIN_CAPACITY).next_power_of_two();
        Self {
            buffer: vec![None; capacity],
            head: 0,
            tail: 0,
            len: 0,
            maxlen: None,
        }
    }

    /// Create a deque with a maximum length.
    /// When full, adding new elements drops elements from the opposite end.
    #[inline]
    pub fn with_maxlen(maxlen: usize) -> Self {
        let mut d = Self::with_capacity(maxlen.max(MIN_CAPACITY));
        d.maxlen = Some(maxlen);
        d
    }

    /// Create a deque from an iterator.
    pub fn from_iter<I: IntoIterator<Item = Value>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut d = Self::with_capacity(lower.max(MIN_CAPACITY));
        for item in iter {
            d.append(item);
        }
        d
    }

    // =========================================================================
    // Core Properties
    // =========================================================================

    /// Returns the number of elements in the deque.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the deque is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the maximum length, if set.
    #[inline]
    pub fn maxlen(&self) -> Option<usize> {
        self.maxlen
    }

    /// Returns the current capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    // =========================================================================
    // Append Operations (O(1) amortized)
    // =========================================================================

    /// Add an element to the right end.
    #[inline]
    pub fn append(&mut self, value: Value) {
        // Handle maxlen constraint
        if let Some(maxlen) = self.maxlen {
            if maxlen == 0 {
                return; // Deque is bounded to 0, ignore append
            }
            if self.len == maxlen {
                // Drop from left to make room
                self.popleft();
            }
        }

        // Grow if needed (only when not bounded)
        if self.len == self.buffer.len() {
            self.grow();
        }

        // Insert at tail
        self.buffer[self.tail] = Some(value);
        self.tail = self.wrap_add(self.tail, 1);
        self.len += 1;
    }

    /// Add an element to the left end.
    #[inline]
    pub fn appendleft(&mut self, value: Value) {
        // Handle maxlen constraint
        if let Some(maxlen) = self.maxlen {
            if maxlen == 0 {
                return;
            }
            if self.len == maxlen {
                // Drop from right to make room
                self.pop();
            }
        }

        // Grow if needed
        if self.len == self.buffer.len() {
            self.grow();
        }

        // Move head back and insert
        self.head = self.wrap_sub(self.head, 1);
        self.buffer[self.head] = Some(value);
        self.len += 1;
    }

    /// Extend the deque by appending elements from an iterator.
    #[inline]
    pub fn extend<I: IntoIterator<Item = Value>>(&mut self, iter: I) {
        for item in iter {
            self.append(item);
        }
    }

    /// Extend the deque by appending elements from an iterator to the left.
    #[inline]
    pub fn extendleft<I: IntoIterator<Item = Value>>(&mut self, iter: I) {
        for item in iter {
            self.appendleft(item);
        }
    }

    // =========================================================================
    // Pop Operations (O(1))
    // =========================================================================

    /// Remove and return an element from the right end.
    #[inline]
    pub fn pop(&mut self) -> Option<Value> {
        if self.is_empty() {
            return None;
        }

        self.tail = self.wrap_sub(self.tail, 1);
        let value = self.buffer[self.tail].take();
        self.len -= 1;
        value
    }

    /// Remove and return an element from the left end.
    #[inline]
    pub fn popleft(&mut self) -> Option<Value> {
        if self.is_empty() {
            return None;
        }

        let value = self.buffer[self.head].take();
        self.head = self.wrap_add(self.head, 1);
        self.len -= 1;
        value
    }

    // =========================================================================
    // Access Operations
    // =========================================================================

    /// Get a reference to the element at index.
    /// Supports negative indices (Python semantics).
    #[inline]
    pub fn get(&self, index: isize) -> Option<&Value> {
        let idx = self.normalize_index(index)?;
        let physical = self.wrap_add(self.head, idx);
        self.buffer[physical].as_ref()
    }

    /// Get a mutable reference to the element at index.
    #[inline]
    pub fn get_mut(&mut self, index: isize) -> Option<&mut Value> {
        let idx = self.normalize_index(index)?;
        let physical = self.wrap_add(self.head, idx);
        self.buffer[physical].as_mut()
    }

    /// Get the first element.
    #[inline]
    pub fn front(&self) -> Option<&Value> {
        if self.is_empty() {
            None
        } else {
            self.buffer[self.head].as_ref()
        }
    }

    /// Get the last element.
    #[inline]
    pub fn back(&self) -> Option<&Value> {
        if self.is_empty() {
            None
        } else {
            let idx = self.wrap_sub(self.tail, 1);
            self.buffer[idx].as_ref()
        }
    }

    // =========================================================================
    // Mutation Operations
    // =========================================================================

    /// Rotate the deque n steps to the right.
    /// If n is negative, rotate to the left.
    pub fn rotate(&mut self, n: isize) {
        if self.len <= 1 || n == 0 {
            return;
        }

        // Normalize rotation amount
        let len = self.len as isize;
        let n = ((n % len) + len) % len;

        if n == 0 {
            return;
        }

        // Rotate right by n: move n elements from right to left
        for _ in 0..n {
            if let Some(v) = self.pop() {
                // Skip maxlen check for rotation
                self.head = self.wrap_sub(self.head, 1);
                self.buffer[self.head] = Some(v);
                self.len += 1;
            }
        }
    }

    /// Reverse the deque in place.
    pub fn reverse(&mut self) {
        if self.len <= 1 {
            return;
        }

        let mut left = 0;
        let mut right = self.len - 1;

        while left < right {
            let left_phys = self.wrap_add(self.head, left);
            let right_phys = self.wrap_add(self.head, right);
            self.buffer.swap(left_phys, right_phys);
            left += 1;
            right -= 1;
        }
    }

    /// Clear all elements.
    #[inline]
    pub fn clear(&mut self) {
        // Fast path: just reset indices without clearing memory
        // The Option<Value> will be overwritten on next insert
        for i in 0..self.buffer.len() {
            self.buffer[i] = None;
        }
        self.head = 0;
        self.tail = 0;
        self.len = 0;
    }

    /// Remove the first occurrence of a value.
    /// Returns true if found and removed.
    pub fn remove(&mut self, value: &Value) -> bool {
        // Find the index
        let pos = self.iter().position(|v| v == value);

        if let Some(index) = pos {
            // Shift elements to fill the gap
            // Choose direction based on which is shorter
            if index < self.len / 2 {
                // Shift left half to the right
                for i in (1..=index).rev() {
                    let from = self.wrap_add(self.head, i - 1);
                    let to = self.wrap_add(self.head, i);
                    self.buffer[to] = self.buffer[from].take();
                }
                self.buffer[self.head] = None;
                self.head = self.wrap_add(self.head, 1);
            } else {
                // Shift right half to the left
                for i in index..self.len - 1 {
                    let from = self.wrap_add(self.head, i + 1);
                    let to = self.wrap_add(self.head, i);
                    self.buffer[to] = self.buffer[from].take();
                }
                self.tail = self.wrap_sub(self.tail, 1);
                self.buffer[self.tail] = None;
            }

            self.len -= 1;
            true
        } else {
            false
        }
    }

    /// Insert an element at the given index.
    pub fn insert(&mut self, index: isize, value: Value) {
        let idx = match self.normalize_index_insert(index) {
            Some(i) => i,
            None => {
                // Append at end if index is out of range
                self.append(value);
                return;
            }
        };

        if self.len == self.buffer.len() {
            self.grow();
        }

        // Choose direction based on which is shorter
        if idx < self.len / 2 {
            // Shift left half to the left
            self.head = self.wrap_sub(self.head, 1);
            for i in 0..idx {
                let from = self.wrap_add(self.head, i + 1);
                let to = self.wrap_add(self.head, i);
                self.buffer[to] = self.buffer[from].take();
            }
        } else {
            // Shift right half to the right
            for i in (idx..self.len).rev() {
                let from = self.wrap_add(self.head, i);
                let to = self.wrap_add(self.head, i + 1);
                self.buffer[to] = self.buffer[from].take();
            }
            self.tail = self.wrap_add(self.tail, 1);
        }

        let phys = self.wrap_add(self.head, idx);
        self.buffer[phys] = Some(value);
        self.len += 1;
    }

    /// Count occurrences of a value.
    pub fn count(&self, value: &Value) -> usize {
        self.iter().filter(|v| *v == value).count()
    }

    /// Find the index of the first occurrence of a value.
    pub fn index_of(&self, value: &Value) -> Option<usize> {
        self.iter().position(|v| v == value)
    }

    // =========================================================================
    // Iterator Support
    // =========================================================================

    /// Returns an iterator over the elements.
    #[inline]
    pub fn iter(&self) -> DequeIter<'_> {
        DequeIter {
            deque: self,
            front: 0,
            back: self.len,
        }
    }

    /// Returns a mutable iterator over the elements.
    #[inline]
    pub fn iter_mut(&mut self) -> DequeIterMut<'_> {
        let len = self.len;
        DequeIterMut {
            deque: self,
            front: 0,
            back: len,
        }
    }

    // =========================================================================
    // Internal Helpers
    // =========================================================================

    /// Wrap-around addition for ring buffer.
    #[inline]
    fn wrap_add(&self, index: usize, offset: usize) -> usize {
        // Use bitwise AND for power-of-two sizes (faster than modulo)
        (index + offset) & (self.buffer.len() - 1)
    }

    /// Wrap-around subtraction for ring buffer.
    #[inline]
    fn wrap_sub(&self, index: usize, offset: usize) -> usize {
        (index + self.buffer.len() - offset) & (self.buffer.len() - 1)
    }

    /// Normalize a potentially negative index to a valid array index.
    #[inline]
    fn normalize_index(&self, index: isize) -> Option<usize> {
        let len = self.len as isize;
        let normalized = if index < 0 { index + len } else { index };

        if normalized >= 0 && normalized < len {
            Some(normalized as usize)
        } else {
            None
        }
    }

    /// Normalize index for insertion (allows index == len).
    #[inline]
    fn normalize_index_insert(&self, index: isize) -> Option<usize> {
        let len = self.len as isize;
        let normalized = if index < 0 { index + len } else { index };

        if normalized >= 0 && normalized <= len {
            Some(normalized as usize)
        } else {
            None
        }
    }

    /// Grow the buffer when full.
    fn grow(&mut self) {
        let old_cap = self.buffer.len();
        let new_cap = old_cap * GROWTH_FACTOR;

        // Create new buffer
        let mut new_buffer = vec![None; new_cap];

        // Copy elements in order
        for (i, item) in self.iter().enumerate() {
            new_buffer[i] = Some(item.clone());
        }

        // Clear old buffer
        for slot in &mut self.buffer {
            *slot = None;
        }

        self.buffer = new_buffer;
        self.head = 0;
        self.tail = self.len;
    }
}

impl Default for Deque {
    fn default() -> Self {
        Self::new()
    }
}

impl Index<usize> for Deque {
    type Output = Value;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index as isize).expect("deque index out of bounds")
    }
}

impl IndexMut<usize> for Deque {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index as isize)
            .expect("deque index out of bounds")
    }
}

impl PartialEq for Deque {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl Eq for Deque {}

// =============================================================================
// Iterators
// =============================================================================

/// Iterator over deque elements.
pub struct DequeIter<'a> {
    deque: &'a Deque,
    front: usize,
    back: usize,
}

impl<'a> Iterator for DequeIter<'a> {
    type Item = &'a Value;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.front == self.back {
            return None;
        }

        let idx = self.deque.wrap_add(self.deque.head, self.front);
        self.front += 1;
        self.deque.buffer[idx].as_ref()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back - self.front;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for DequeIter<'a> {}

impl<'a> DoubleEndedIterator for DequeIter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front == self.back {
            return None;
        }

        self.back -= 1;
        let idx = self.deque.wrap_add(self.deque.head, self.back);
        self.deque.buffer[idx].as_ref()
    }
}

/// Mutable iterator over deque elements.
pub struct DequeIterMut<'a> {
    deque: &'a mut Deque,
    front: usize,
    back: usize,
}

impl<'a> Iterator for DequeIterMut<'a> {
    type Item = &'a mut Value;

    fn next(&mut self) -> Option<Self::Item> {
        if self.front == self.back {
            return None;
        }

        let idx = self.deque.wrap_add(self.deque.head, self.front);
        self.front += 1;

        // Safety: We maintain unique access through the iterator
        unsafe {
            let ptr = self.deque.buffer.as_mut_ptr().add(idx);
            (*ptr).as_mut()
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back - self.front;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for DequeIterMut<'a> {}

// =============================================================================
// Conversion Traits
// =============================================================================

impl FromIterator<Value> for Deque {
    fn from_iter<I: IntoIterator<Item = Value>>(iter: I) -> Self {
        Deque::from_iter(iter)
    }
}

impl IntoIterator for Deque {
    type Item = Value;
    type IntoIter = DequeIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        DequeIntoIter { deque: self }
    }
}

/// Owning iterator over deque elements.
pub struct DequeIntoIter {
    deque: Deque,
}

impl Iterator for DequeIntoIter {
    type Item = Value;

    fn next(&mut self) -> Option<Self::Item> {
        self.deque.popleft()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.deque.len, Some(self.deque.len))
    }
}

impl ExactSizeIterator for DequeIntoIter {}

impl DoubleEndedIterator for DequeIntoIter {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.deque.pop()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod deque_tests {
    use super::*;

    // =========================================================================
    // Construction Tests
    // =========================================================================

    #[test]
    fn test_new_creates_empty_deque() {
        let d = Deque::new();
        assert!(d.is_empty());
        assert_eq!(d.len(), 0);
    }

    #[test]
    fn test_with_capacity_respects_minimum() {
        let d = Deque::with_capacity(4);
        assert!(d.capacity() >= MIN_CAPACITY);
    }

    #[test]
    fn test_with_capacity_rounds_to_power_of_two() {
        let d = Deque::with_capacity(17);
        assert_eq!(d.capacity(), 32);
    }

    #[test]
    fn test_with_maxlen() {
        let d = Deque::with_maxlen(5);
        assert_eq!(d.maxlen(), Some(5));
    }

    #[test]
    fn test_from_iter_creates_deque() {
        let values = vec![
            Value::int_unchecked(1),
            Value::int_unchecked(2),
            Value::int_unchecked(3),
        ];
        let d = Deque::from_iter(values);
        assert_eq!(d.len(), 3);
        assert_eq!(d.get(0).and_then(|v| v.as_int()), Some(1));
        assert_eq!(d.get(2).and_then(|v| v.as_int()), Some(3));
    }

    // =========================================================================
    // Append/Pop Tests
    // =========================================================================

    #[test]
    fn test_append_increases_length() {
        let mut d = Deque::new();
        d.append(Value::int_unchecked(1));
        assert_eq!(d.len(), 1);
        d.append(Value::int_unchecked(2));
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn test_appendleft_increases_length() {
        let mut d = Deque::new();
        d.appendleft(Value::int_unchecked(1));
        assert_eq!(d.len(), 1);
        d.appendleft(Value::int_unchecked(2));
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn test_append_and_pop_order() {
        let mut d = Deque::new();
        d.append(Value::int_unchecked(1));
        d.append(Value::int_unchecked(2));
        d.append(Value::int_unchecked(3));

        assert_eq!(d.pop().and_then(|v| v.as_int()), Some(3));
        assert_eq!(d.pop().and_then(|v| v.as_int()), Some(2));
        assert_eq!(d.pop().and_then(|v| v.as_int()), Some(1));
        assert_eq!(d.pop(), None);
    }

    #[test]
    fn test_appendleft_and_popleft_order() {
        let mut d = Deque::new();
        d.appendleft(Value::int_unchecked(1));
        d.appendleft(Value::int_unchecked(2));
        d.appendleft(Value::int_unchecked(3));

        assert_eq!(d.popleft().and_then(|v| v.as_int()), Some(3));
        assert_eq!(d.popleft().and_then(|v| v.as_int()), Some(2));
        assert_eq!(d.popleft().and_then(|v| v.as_int()), Some(1));
        assert_eq!(d.popleft(), None);
    }

    #[test]
    fn test_mixed_operations() {
        let mut d = Deque::new();
        d.append(Value::int_unchecked(1));
        d.appendleft(Value::int_unchecked(0));
        d.append(Value::int_unchecked(2));

        // [0, 1, 2]
        assert_eq!(d.popleft().and_then(|v| v.as_int()), Some(0));
        assert_eq!(d.pop().and_then(|v| v.as_int()), Some(2));
        assert_eq!(d.popleft().and_then(|v| v.as_int()), Some(1));
    }

    #[test]
    fn test_pop_empty_returns_none() {
        let mut d = Deque::new();
        assert_eq!(d.pop(), None);
        assert_eq!(d.popleft(), None);
    }

    // =========================================================================
    // Maxlen Tests
    // =========================================================================

    #[test]
    fn test_maxlen_drops_from_left_on_append() {
        let mut d = Deque::with_maxlen(3);
        d.append(Value::int_unchecked(1));
        d.append(Value::int_unchecked(2));
        d.append(Value::int_unchecked(3));
        assert_eq!(d.len(), 3);

        d.append(Value::int_unchecked(4));
        assert_eq!(d.len(), 3);
        assert_eq!(d.get(0).and_then(|v| v.as_int()), Some(2));
        assert_eq!(d.get(2).and_then(|v| v.as_int()), Some(4));
    }

    #[test]
    fn test_maxlen_drops_from_right_on_appendleft() {
        let mut d = Deque::with_maxlen(3);
        d.appendleft(Value::int_unchecked(1));
        d.appendleft(Value::int_unchecked(2));
        d.appendleft(Value::int_unchecked(3));
        assert_eq!(d.len(), 3);

        d.appendleft(Value::int_unchecked(4));
        assert_eq!(d.len(), 3);
        assert_eq!(d.get(0).and_then(|v| v.as_int()), Some(4));
        assert_eq!(d.get(2).and_then(|v| v.as_int()), Some(2));
    }

    #[test]
    fn test_maxlen_zero_ignores_appends() {
        let mut d = Deque::with_maxlen(0);
        d.append(Value::int_unchecked(1));
        d.appendleft(Value::int_unchecked(2));
        assert!(d.is_empty());
    }

    // =========================================================================
    // Index Access Tests
    // =========================================================================

    #[test]
    fn test_positive_index() {
        let mut d = Deque::new();
        d.append(Value::int_unchecked(10));
        d.append(Value::int_unchecked(20));
        d.append(Value::int_unchecked(30));

        assert_eq!(d.get(0).and_then(|v| v.as_int()), Some(10));
        assert_eq!(d.get(1).and_then(|v| v.as_int()), Some(20));
        assert_eq!(d.get(2).and_then(|v| v.as_int()), Some(30));
    }

    #[test]
    fn test_negative_index() {
        let mut d = Deque::new();
        d.append(Value::int_unchecked(10));
        d.append(Value::int_unchecked(20));
        d.append(Value::int_unchecked(30));

        assert_eq!(d.get(-1).and_then(|v| v.as_int()), Some(30));
        assert_eq!(d.get(-2).and_then(|v| v.as_int()), Some(20));
        assert_eq!(d.get(-3).and_then(|v| v.as_int()), Some(10));
    }

    #[test]
    fn test_out_of_bounds_index() {
        let mut d = Deque::new();
        d.append(Value::int_unchecked(1));

        assert_eq!(d.get(5), None);
        assert_eq!(d.get(-5), None);
    }

    #[test]
    fn test_front_and_back() {
        let mut d = Deque::new();
        assert_eq!(d.front(), None);
        assert_eq!(d.back(), None);

        d.append(Value::int_unchecked(1));
        d.append(Value::int_unchecked(2));

        assert_eq!(d.front().and_then(|v| v.as_int()), Some(1));
        assert_eq!(d.back().and_then(|v| v.as_int()), Some(2));
    }

    // =========================================================================
    // Rotation Tests
    // =========================================================================

    #[test]
    fn test_rotate_right() {
        let mut d = Deque::new();
        for i in 0..5 {
            d.append(Value::int_unchecked(i));
        }
        // [0, 1, 2, 3, 4]
        d.rotate(2);
        // [3, 4, 0, 1, 2]

        assert_eq!(d.get(0).and_then(|v| v.as_int()), Some(3));
        assert_eq!(d.get(1).and_then(|v| v.as_int()), Some(4));
        assert_eq!(d.get(2).and_then(|v| v.as_int()), Some(0));
    }

    #[test]
    fn test_rotate_left() {
        let mut d = Deque::new();
        for i in 0..5 {
            d.append(Value::int_unchecked(i));
        }
        // [0, 1, 2, 3, 4]
        d.rotate(-2);
        // [2, 3, 4, 0, 1]

        assert_eq!(d.get(0).and_then(|v| v.as_int()), Some(2));
        assert_eq!(d.get(4).and_then(|v| v.as_int()), Some(1));
    }

    #[test]
    fn test_rotate_empty() {
        let mut d = Deque::new();
        d.rotate(5);
        assert!(d.is_empty());
    }

    #[test]
    fn test_rotate_single_element() {
        let mut d = Deque::new();
        d.append(Value::int_unchecked(1));
        d.rotate(100);
        assert_eq!(d.get(0).and_then(|v| v.as_int()), Some(1));
    }

    // =========================================================================
    // Reverse Tests
    // =========================================================================

    #[test]
    fn test_reverse() {
        let mut d = Deque::new();
        for i in 0..5 {
            d.append(Value::int_unchecked(i));
        }
        d.reverse();

        assert_eq!(d.get(0).and_then(|v| v.as_int()), Some(4));
        assert_eq!(d.get(4).and_then(|v| v.as_int()), Some(0));
    }

    #[test]
    fn test_reverse_empty() {
        let mut d = Deque::new();
        d.reverse();
        assert!(d.is_empty());
    }

    #[test]
    fn test_reverse_single() {
        let mut d = Deque::new();
        d.append(Value::int_unchecked(1));
        d.reverse();
        assert_eq!(d.get(0).and_then(|v| v.as_int()), Some(1));
    }

    // =========================================================================
    // Clear Tests
    // =========================================================================

    #[test]
    fn test_clear() {
        let mut d = Deque::new();
        for i in 0..10 {
            d.append(Value::int_unchecked(i));
        }
        d.clear();

        assert!(d.is_empty());
        assert_eq!(d.len(), 0);
    }

    // =========================================================================
    // Remove/Insert Tests
    // =========================================================================

    #[test]
    fn test_remove_existing() {
        let mut d = Deque::new();
        d.append(Value::int_unchecked(1));
        d.append(Value::int_unchecked(2));
        d.append(Value::int_unchecked(3));

        assert!(d.remove(&Value::int_unchecked(2)));
        assert_eq!(d.len(), 2);
        assert_eq!(d.get(0).and_then(|v| v.as_int()), Some(1));
        assert_eq!(d.get(1).and_then(|v| v.as_int()), Some(3));
    }

    #[test]
    fn test_remove_not_found() {
        let mut d = Deque::new();
        d.append(Value::int_unchecked(1));

        assert!(!d.remove(&Value::int_unchecked(99)));
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn test_insert_beginning() {
        let mut d = Deque::new();
        d.append(Value::int_unchecked(2));
        d.append(Value::int_unchecked(3));

        d.insert(0, Value::int_unchecked(1));

        assert_eq!(d.len(), 3);
        assert_eq!(d.get(0).and_then(|v| v.as_int()), Some(1));
    }

    #[test]
    fn test_insert_middle() {
        let mut d = Deque::new();
        d.append(Value::int_unchecked(1));
        d.append(Value::int_unchecked(3));

        d.insert(1, Value::int_unchecked(2));

        assert_eq!(d.len(), 3);
        assert_eq!(d.get(1).and_then(|v| v.as_int()), Some(2));
    }

    #[test]
    fn test_insert_end() {
        let mut d = Deque::new();
        d.append(Value::int_unchecked(1));
        d.append(Value::int_unchecked(2));

        d.insert(2, Value::int_unchecked(3));

        assert_eq!(d.len(), 3);
        assert_eq!(d.get(2).and_then(|v| v.as_int()), Some(3));
    }

    // =========================================================================
    // Count/Index Tests
    // =========================================================================

    #[test]
    fn test_count() {
        let mut d = Deque::new();
        d.append(Value::int_unchecked(1));
        d.append(Value::int_unchecked(2));
        d.append(Value::int_unchecked(1));
        d.append(Value::int_unchecked(1));

        assert_eq!(d.count(&Value::int_unchecked(1)), 3);
        assert_eq!(d.count(&Value::int_unchecked(2)), 1);
        assert_eq!(d.count(&Value::int_unchecked(99)), 0);
    }

    #[test]
    fn test_index_of() {
        let mut d = Deque::new();
        d.append(Value::int_unchecked(10));
        d.append(Value::int_unchecked(20));
        d.append(Value::int_unchecked(30));

        assert_eq!(d.index_of(&Value::int_unchecked(20)), Some(1));
        assert_eq!(d.index_of(&Value::int_unchecked(99)), None);
    }

    // =========================================================================
    // Iterator Tests
    // =========================================================================

    #[test]
    fn test_iter() {
        let mut d = Deque::new();
        d.append(Value::int_unchecked(1));
        d.append(Value::int_unchecked(2));
        d.append(Value::int_unchecked(3));

        let vals: Vec<i64> = d.iter().filter_map(|v| v.as_int()).collect();
        assert_eq!(vals, vec![1, 2, 3]);
    }

    #[test]
    fn test_iter_reverse() {
        let mut d = Deque::new();
        d.append(Value::int_unchecked(1));
        d.append(Value::int_unchecked(2));
        d.append(Value::int_unchecked(3));

        let vals: Vec<i64> = d.iter().rev().filter_map(|v| v.as_int()).collect();
        assert_eq!(vals, vec![3, 2, 1]);
    }

    #[test]
    fn test_into_iter() {
        let mut d = Deque::new();
        d.append(Value::int_unchecked(1));
        d.append(Value::int_unchecked(2));

        let vals: Vec<i64> = d.into_iter().filter_map(|v| v.as_int()).collect();
        assert_eq!(vals, vec![1, 2]);
    }

    // =========================================================================
    // Extend Tests
    // =========================================================================

    #[test]
    fn test_extend() {
        let mut d = Deque::new();
        d.append(Value::int_unchecked(1));

        d.extend(vec![Value::int_unchecked(2), Value::int_unchecked(3)]);

        assert_eq!(d.len(), 3);
        assert_eq!(d.get(2).and_then(|v| v.as_int()), Some(3));
    }

    #[test]
    fn test_extendleft() {
        let mut d = Deque::new();
        d.append(Value::int_unchecked(3));

        d.extendleft(vec![Value::int_unchecked(2), Value::int_unchecked(1)]);

        // Note: extendleft adds in reverse order (like Python)
        assert_eq!(d.len(), 3);
        assert_eq!(d.get(0).and_then(|v| v.as_int()), Some(1));
    }

    // =========================================================================
    // Growth Tests
    // =========================================================================

    #[test]
    fn test_growth_on_overflow() {
        let mut d = Deque::with_capacity(16);
        let initial_cap = d.capacity();

        for i in 0..20 {
            d.append(Value::int_unchecked(i));
        }

        assert!(d.capacity() > initial_cap);
        assert_eq!(d.len(), 20);

        // Verify all elements are correct
        for i in 0..20 {
            assert_eq!(d.get(i as isize).and_then(|v| v.as_int()), Some(i));
        }
    }

    #[test]
    fn test_growth_preserves_order() {
        let mut d = Deque::with_capacity(16);

        // Add elements from both ends
        for i in 0..10 {
            d.append(Value::int_unchecked(i));
            d.appendleft(Value::int_unchecked(-(i + 1)));
        }

        // Force growth
        for i in 10..30 {
            d.append(Value::int_unchecked(i));
        }

        // Verify order is preserved
        assert_eq!(d.get(0).and_then(|v| v.as_int()), Some(-10));
        assert_eq!(d.get(10).and_then(|v| v.as_int()), Some(0));
    }

    // =========================================================================
    // Equality Tests
    // =========================================================================

    #[test]
    fn test_equality() {
        let mut d1 = Deque::new();
        let mut d2 = Deque::new();

        for i in 0..5 {
            d1.append(Value::int_unchecked(i));
            d2.append(Value::int_unchecked(i));
        }

        assert_eq!(d1, d2);
    }

    #[test]
    fn test_inequality_different_length() {
        let mut d1 = Deque::new();
        let mut d2 = Deque::new();

        d1.append(Value::int_unchecked(1));
        d2.append(Value::int_unchecked(1));
        d2.append(Value::int_unchecked(2));

        assert_ne!(d1, d2);
    }

    #[test]
    fn test_inequality_different_values() {
        let mut d1 = Deque::new();
        let mut d2 = Deque::new();

        d1.append(Value::int_unchecked(1));
        d2.append(Value::int_unchecked(2));

        assert_ne!(d1, d2);
    }

    // =========================================================================
    // Index Trait Tests
    // =========================================================================

    #[test]
    fn test_index_trait() {
        let mut d = Deque::new();
        d.append(Value::int_unchecked(10));
        d.append(Value::int_unchecked(20));

        assert_eq!(d[0].as_int(), Some(10));
        assert_eq!(d[1].as_int(), Some(20));
    }

    #[test]
    #[should_panic(expected = "deque index out of bounds")]
    fn test_index_out_of_bounds_panics() {
        let d = Deque::new();
        let _ = &d[0];
    }

    // =========================================================================
    // Stress Tests
    // =========================================================================

    #[test]
    fn test_stress_alternating_operations() {
        let mut d = Deque::new();

        for i in 0..1000 {
            if i % 2 == 0 {
                d.append(Value::int_unchecked(i));
            } else {
                d.appendleft(Value::int_unchecked(i));
            }
        }

        for _ in 0..500 {
            d.pop();
            d.popleft();
        }

        assert!(d.is_empty());
    }

    #[test]
    fn test_stress_rotate_many() {
        let mut d = Deque::new();
        for i in 0..100 {
            d.append(Value::int_unchecked(i));
        }

        // Rotate by full length should result in same order
        d.rotate(100);

        for i in 0..100 {
            assert_eq!(d.get(i).and_then(|v| v.as_int()), Some(i as i64));
        }
    }
}
