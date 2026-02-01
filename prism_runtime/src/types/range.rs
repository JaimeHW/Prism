//! Python range object implementation.
//!
//! Provides a memory-efficient representation of integer sequences
//! that generates values on-demand during iteration.

use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use std::fmt;

// =============================================================================
// RangeObject
// =============================================================================

/// Python range object.
///
/// Represents an immutable sequence of integers with start, stop, and step.
/// Unlike lists, ranges are memory-efficient because they compute values
/// on-demand rather than storing them.
///
/// # Examples
///
/// ```text
/// range(5)        -> 0, 1, 2, 3, 4
/// range(2, 8)     -> 2, 3, 4, 5, 6, 7
/// range(0, 10, 2) -> 0, 2, 4, 6, 8
/// range(10, 0, -1) -> 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
/// ```
///
/// # Performance
///
/// - O(1) creation regardless of range size
/// - O(1) length calculation
/// - O(1) indexed access
/// - O(1) membership test
#[repr(C)]
pub struct RangeObject {
    /// Object header.
    pub header: ObjectHeader,
    /// Start value (inclusive).
    pub start: i64,
    /// Stop value (exclusive).
    pub stop: i64,
    /// Step value (non-zero).
    pub step: i64,
}

impl RangeObject {
    /// Create a new range with start, stop, and step.
    ///
    /// # Panics
    ///
    /// Panics if step is zero.
    #[inline]
    pub fn new(start: i64, stop: i64, step: i64) -> Self {
        assert!(step != 0, "range() arg 3 must not be zero");
        Self {
            header: ObjectHeader::new(TypeId::RANGE),
            start,
            stop,
            step,
        }
    }

    /// Create a range from 0 to stop with step 1.
    #[inline]
    pub fn from_stop(stop: i64) -> Self {
        Self::new(0, stop, 1)
    }

    /// Create a range from start to stop with step 1.
    #[inline]
    pub fn from_start_stop(start: i64, stop: i64) -> Self {
        Self::new(start, stop, 1)
    }

    /// Get the number of elements in the range.
    ///
    /// This is O(1) - computed from start, stop, step.
    #[inline]
    pub fn len(&self) -> usize {
        if self.step > 0 {
            if self.stop <= self.start {
                0
            } else {
                ((self.stop - self.start - 1) / self.step + 1) as usize
            }
        } else {
            // step < 0
            if self.stop >= self.start {
                0
            } else {
                ((self.start - self.stop - 1) / (-self.step) + 1) as usize
            }
        }
    }

    /// Check if the range is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the element at index (supports negative indexing).
    ///
    /// Returns None if index is out of bounds.
    #[inline]
    pub fn get(&self, index: i64) -> Option<i64> {
        let len = self.len() as i64;
        let idx = if index < 0 { len + index } else { index };

        if idx < 0 || idx >= len {
            return None;
        }

        Some(self.start + idx * self.step)
    }

    /// Check if a value is in the range.
    ///
    /// This is O(1) - computed mathematically.
    #[inline]
    pub fn contains(&self, value: i64) -> bool {
        // Check if value is within bounds
        if self.step > 0 {
            if value < self.start || value >= self.stop {
                return false;
            }
        } else {
            if value > self.start || value <= self.stop {
                return false;
            }
        }

        // Check if value aligns with step
        (value - self.start) % self.step == 0
    }

    /// Create an iterator over this range.
    #[inline]
    pub fn iter(&self) -> RangeIterator {
        RangeIterator {
            current: self.start,
            stop: self.stop,
            step: self.step,
        }
    }

    /// Get the first element, or None if empty.
    #[inline]
    pub fn first(&self) -> Option<i64> {
        if self.is_empty() {
            None
        } else {
            Some(self.start)
        }
    }

    /// Get the last element, or None if empty.
    #[inline]
    pub fn last(&self) -> Option<i64> {
        let len = self.len();
        if len == 0 {
            None
        } else {
            Some(self.start + ((len - 1) as i64) * self.step)
        }
    }

    /// Reverse the range (returns a new range).
    pub fn reverse(&self) -> RangeObject {
        let len = self.len();
        if len == 0 {
            return RangeObject::new(0, 0, 1);
        }

        let last = self.last().unwrap();
        let new_stop = if self.step > 0 {
            self.start - self.step
        } else {
            self.start - self.step
        };

        RangeObject::new(last, new_stop, -self.step)
    }

    /// Convert to a vector of values.
    ///
    /// Note: This allocates. Use iter() for lazy iteration.
    pub fn to_vec(&self) -> Vec<i64> {
        self.iter().collect()
    }
}

impl Clone for RangeObject {
    fn clone(&self) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::RANGE),
            start: self.start,
            stop: self.stop,
            step: self.step,
        }
    }
}

impl PartialEq for RangeObject {
    fn eq(&self, other: &Self) -> bool {
        // Two ranges are equal if they produce the same sequence
        if self.len() != other.len() {
            return false;
        }
        if self.is_empty() {
            return true; // Both empty
        }
        self.start == other.start && self.step == other.step && self.len() == other.len()
    }
}

impl Eq for RangeObject {}

impl fmt::Debug for RangeObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.step == 1 {
            write!(f, "range({}, {})", self.start, self.stop)
        } else {
            write!(f, "range({}, {}, {})", self.start, self.stop, self.step)
        }
    }
}

impl fmt::Display for RangeObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.step == 1 {
            write!(f, "range({}, {})", self.start, self.stop)
        } else {
            write!(f, "range({}, {}, {})", self.start, self.stop, self.step)
        }
    }
}

impl PyObject for RangeObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

// =============================================================================
// RangeIterator
// =============================================================================

/// Iterator over a range.
///
/// Produces values on-demand with O(1) memory usage regardless of range size.
#[derive(Clone, Debug)]
pub struct RangeIterator {
    current: i64,
    stop: i64,
    step: i64,
}

impl RangeIterator {
    /// Create a new range iterator.
    #[inline]
    pub fn new(start: i64, stop: i64, step: i64) -> Self {
        Self {
            current: start,
            stop,
            step,
        }
    }

    /// Check if there are more elements.
    #[inline]
    pub fn has_next(&self) -> bool {
        if self.step > 0 {
            self.current < self.stop
        } else {
            self.current > self.stop
        }
    }
}

impl Iterator for RangeIterator {
    type Item = i64;

    #[inline]
    fn next(&mut self) -> Option<i64> {
        if self.step > 0 {
            if self.current < self.stop {
                let value = self.current;
                self.current += self.step;
                Some(value)
            } else {
                None
            }
        } else {
            if self.current > self.stop {
                let value = self.current;
                self.current += self.step; // step is negative
                Some(value)
            } else {
                None
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = if self.step > 0 {
            if self.stop <= self.current {
                0
            } else {
                ((self.stop - self.current - 1) / self.step + 1) as usize
            }
        } else {
            if self.stop >= self.current {
                0
            } else {
                ((self.current - self.stop - 1) / (-self.step) + 1) as usize
            }
        };
        (len, Some(len))
    }
}

impl ExactSizeIterator for RangeIterator {}

// =============================================================================
// IntoIterator Implementation
// =============================================================================

impl IntoIterator for &RangeObject {
    type Item = i64;
    type IntoIter = RangeIterator;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_from_stop() {
        let r = RangeObject::from_stop(5);
        assert_eq!(r.start, 0);
        assert_eq!(r.stop, 5);
        assert_eq!(r.step, 1);
        assert_eq!(r.len(), 5);
    }

    #[test]
    fn test_range_from_start_stop() {
        let r = RangeObject::from_start_stop(2, 8);
        assert_eq!(r.start, 2);
        assert_eq!(r.stop, 8);
        assert_eq!(r.len(), 6);
    }

    #[test]
    fn test_range_with_step() {
        let r = RangeObject::new(0, 10, 2);
        assert_eq!(r.len(), 5);
        let values: Vec<_> = r.iter().collect();
        assert_eq!(values, vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_range_negative_step() {
        let r = RangeObject::new(10, 0, -1);
        assert_eq!(r.len(), 10);
        let values: Vec<_> = r.iter().collect();
        assert_eq!(values, vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_range_negative_step_by_two() {
        let r = RangeObject::new(10, 0, -2);
        assert_eq!(r.len(), 5);
        let values: Vec<_> = r.iter().collect();
        assert_eq!(values, vec![10, 8, 6, 4, 2]);
    }

    #[test]
    fn test_empty_range() {
        let r = RangeObject::from_start_stop(5, 5);
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        assert!(r.iter().next().is_none());
    }

    #[test]
    fn test_range_get() {
        let r = RangeObject::new(10, 20, 2);
        assert_eq!(r.get(0), Some(10));
        assert_eq!(r.get(1), Some(12));
        assert_eq!(r.get(4), Some(18));
        assert_eq!(r.get(-1), Some(18)); // Last element
        assert_eq!(r.get(5), None); // Out of bounds
    }

    #[test]
    fn test_range_contains() {
        let r = RangeObject::new(0, 10, 2);
        assert!(r.contains(0));
        assert!(r.contains(2));
        assert!(r.contains(8));
        assert!(!r.contains(1)); // Not aligned with step
        assert!(!r.contains(10)); // Exclusive stop
        assert!(!r.contains(-1)); // Below start
    }

    #[test]
    fn test_range_first_last() {
        let r = RangeObject::new(5, 15, 3);
        assert_eq!(r.first(), Some(5));
        assert_eq!(r.last(), Some(14)); // 5, 8, 11, 14
    }

    #[test]
    fn test_range_equality() {
        let r1 = RangeObject::new(0, 10, 2);
        let r2 = RangeObject::new(0, 10, 2);
        let r3 = RangeObject::new(0, 11, 2);
        assert_eq!(r1, r2);
        assert_ne!(r1, r3);
    }

    #[test]
    fn test_range_display() {
        let r1 = RangeObject::from_start_stop(0, 10);
        assert_eq!(format!("{}", r1), "range(0, 10)");

        let r2 = RangeObject::new(0, 10, 2);
        assert_eq!(format!("{}", r2), "range(0, 10, 2)");
    }

    #[test]
    fn test_range_clone() {
        let r1 = RangeObject::new(1, 100, 5);
        let r2 = r1.clone();
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_iterator_size_hint() {
        let r = RangeObject::from_stop(100);
        let iter = r.iter();
        assert_eq!(iter.size_hint(), (100, Some(100)));
    }

    #[test]
    fn test_exact_size_iterator() {
        let r = RangeObject::new(0, 10, 3);
        let iter = r.iter();
        assert_eq!(iter.len(), 4); // 0, 3, 6, 9
    }

    #[test]
    fn test_range_to_vec() {
        let r = RangeObject::new(1, 6, 1);
        assert_eq!(r.to_vec(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_into_iter() {
        let r = RangeObject::from_stop(3);
        let values: Vec<_> = (&r).into_iter().collect();
        assert_eq!(values, vec![0, 1, 2]);
    }
}
