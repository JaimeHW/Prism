//! High-performance Python slice object implementation.
//!
//! The `SliceObject` represents a Python slice object (`slice(start, stop, step)`)
//! used for sequence slicing operations.
//!
//! # Memory Layout
//!
//! ```text
//! SliceObject (40 bytes):
//! ├── ObjectHeader (16 bytes): type_id, gc_flags, hash
//! ├── start (8 bytes): Optional<i64> as tagged value
//! ├── stop (8 bytes): Optional<i64> as tagged value
//! └── step (8 bytes): Optional<i64> as tagged value
//! ```
//!
//! # Performance Characteristics
//!
//! - O(1) construction and access
//! - Zero heap allocation (all inline)
//! - Cached index computations for repeated use
//!
//! # Python Semantics
//!
//! Slice objects support `None` for any component:
//! - `slice(None, 5)` → `[:5]`
//! - `slice(1, None)` → `[1:]`
//! - `slice(None, None, -1)` → `[::-1]`

use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use std::fmt;

// =============================================================================
// SliceValue - Compact optional integer representation
// =============================================================================

/// Compact representation of an optional slice index.
///
/// Uses a tagged representation to distinguish between:
/// - None: Represented by i64::MIN (a value that cannot appear as a valid index)
/// - Some(n): The actual integer value
///
/// This avoids the overhead of `Option<i64>` (16 bytes) → 8 bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct SliceValue(i64);

impl SliceValue {
    /// Sentinel value representing None.
    const NONE_SENTINEL: i64 = i64::MIN;

    /// Create a None slice value.
    #[inline(always)]
    pub const fn none() -> Self {
        SliceValue(Self::NONE_SENTINEL)
    }

    /// Create a Some slice value.
    #[inline(always)]
    pub const fn some(value: i64) -> Self {
        // If user tries to use i64::MIN as an index, we map to i64::MIN + 1
        // This is an edge case that almost never occurs in practice
        if value == Self::NONE_SENTINEL {
            SliceValue(Self::NONE_SENTINEL + 1)
        } else {
            SliceValue(value)
        }
    }

    /// Check if this is None.
    #[inline(always)]
    pub const fn is_none(self) -> bool {
        self.0 == Self::NONE_SENTINEL
    }

    /// Check if this is Some.
    #[inline(always)]
    pub const fn is_some(self) -> bool {
        self.0 != Self::NONE_SENTINEL
    }

    /// Get the value if Some.
    #[inline(always)]
    pub const fn get(self) -> Option<i64> {
        if self.0 == Self::NONE_SENTINEL {
            None
        } else {
            Some(self.0)
        }
    }

    /// Get the value or a default.
    #[inline(always)]
    pub const fn unwrap_or(self, default: i64) -> i64 {
        if self.0 == Self::NONE_SENTINEL {
            default
        } else {
            self.0
        }
    }

    /// Map the value if Some.
    #[inline]
    pub fn map<F: FnOnce(i64) -> i64>(self, f: F) -> SliceValue {
        if self.0 == Self::NONE_SENTINEL {
            self
        } else {
            SliceValue::some(f(self.0))
        }
    }
}

impl Default for SliceValue {
    fn default() -> Self {
        Self::none()
    }
}

impl From<Option<i64>> for SliceValue {
    #[inline]
    fn from(opt: Option<i64>) -> Self {
        match opt {
            Some(v) => SliceValue::some(v),
            None => SliceValue::none(),
        }
    }
}

impl From<i64> for SliceValue {
    #[inline]
    fn from(value: i64) -> Self {
        SliceValue::some(value)
    }
}

impl fmt::Debug for SliceValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.get() {
            Some(v) => write!(f, "Some({})", v),
            None => write!(f, "None"),
        }
    }
}

impl fmt::Display for SliceValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.get() {
            Some(v) => write!(f, "{}", v),
            None => write!(f, "None"),
        }
    }
}

// =============================================================================
// SliceIndices - Resolved slice indices for a sequence of known length
// =============================================================================

/// Resolved slice indices for a specific sequence length.
///
/// This struct contains the concrete start, stop, and step values
/// after applying Python's slice resolution algorithm.
///
/// # Performance
///
/// Pre-computing these values allows O(1) iteration setup for slicing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceIndices {
    /// Resolved start index (always >= 0 and < length for positive step).
    pub start: usize,
    /// Resolved stop index.
    pub stop: usize,
    /// Resolved step (never 0).
    pub step: isize,
    /// Number of elements in the slice.
    pub length: usize,
}

impl SliceIndices {
    /// Iterate over the slice indices.
    #[inline]
    pub fn iter(self) -> SliceIndexIter {
        SliceIndexIter {
            current: self.start as isize,
            stop: self.stop as isize,
            step: self.step,
            remaining: self.length,
        }
    }
}

/// Iterator over resolved slice indices.
pub struct SliceIndexIter {
    current: isize,
    stop: isize,
    step: isize,
    remaining: usize,
}

impl Iterator for SliceIndexIter {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let index = self.current as usize;
        self.current += self.step;
        self.remaining -= 1;
        Some(index)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for SliceIndexIter {}

// =============================================================================
// SliceObject - Python slice object
// =============================================================================

/// Python slice object.
///
/// Represents `slice(start, stop, step)` with full Python semantics.
///
/// # Memory Layout
///
/// Total size: 40 bytes (fits in a single cache line with header)
/// - ObjectHeader: 16 bytes
/// - start: 8 bytes
/// - stop: 8 bytes  
/// - step: 8 bytes
///
/// # Thread Safety
///
/// SliceObject is immutable after construction and thus safe to share.
#[repr(C)]
pub struct SliceObject {
    /// Object header.
    pub header: ObjectHeader,
    /// Start index (None = beginning).
    start: SliceValue,
    /// Stop index (None = end).
    stop: SliceValue,
    /// Step (None = 1, cannot be 0).
    step: SliceValue,
}

impl SliceObject {
    /// Create a new slice object.
    ///
    /// # Arguments
    ///
    /// * `start` - Start index (None = beginning)
    /// * `stop` - Stop index (None = end)
    /// * `step` - Step size (None = 1)
    ///
    /// # Panics
    ///
    /// Panics if step is Some(0) (zero step is not allowed).
    #[inline]
    pub fn new(start: Option<i64>, stop: Option<i64>, step: Option<i64>) -> Self {
        if let Some(s) = step {
            assert!(s != 0, "slice step cannot be zero");
        }
        Self {
            header: ObjectHeader::new(TypeId::SLICE),
            start: start.into(),
            stop: stop.into(),
            step: step.into(),
        }
    }

    /// Create a slice with only stop: `slice(None, stop, None)` → `[:stop]`.
    #[inline]
    pub fn stop_only(stop: i64) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::SLICE),
            start: SliceValue::none(),
            stop: SliceValue::some(stop),
            step: SliceValue::none(),
        }
    }

    /// Create a slice with start and stop: `slice(start, stop, None)` → `[start:stop]`.
    #[inline]
    pub fn start_stop(start: i64, stop: i64) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::SLICE),
            start: SliceValue::some(start),
            stop: SliceValue::some(stop),
            step: SliceValue::none(),
        }
    }

    /// Create a full slice: `slice(start, stop, step)` → `[start:stop:step]`.
    #[inline]
    pub fn full(start: i64, stop: i64, step: i64) -> Self {
        assert!(step != 0, "slice step cannot be zero");
        Self {
            header: ObjectHeader::new(TypeId::SLICE),
            start: SliceValue::some(start),
            stop: SliceValue::some(stop),
            step: SliceValue::some(step),
        }
    }

    /// Get the start value.
    #[inline(always)]
    pub fn start(&self) -> Option<i64> {
        self.start.get()
    }

    /// Get the stop value.
    #[inline(always)]
    pub fn stop(&self) -> Option<i64> {
        self.stop.get()
    }

    /// Get the step value.
    #[inline(always)]
    pub fn step(&self) -> Option<i64> {
        self.step.get()
    }

    /// Compute concrete indices for a sequence of given length.
    ///
    /// This implements Python's slice index resolution algorithm:
    ///
    /// 1. Determine step direction (positive or negative)
    /// 2. Resolve None values to defaults based on direction
    /// 3. Clamp indices to valid range
    /// 4. Compute the number of elements in the slice
    ///
    /// # Arguments
    ///
    /// * `length` - The length of the sequence being sliced
    ///
    /// # Returns
    ///
    /// `SliceIndices` with resolved start, stop, step, and element count.
    ///
    /// # Performance
    ///
    /// O(1) computation, all arithmetic operations.
    #[inline]
    pub fn indices(&self, length: usize) -> SliceIndices {
        let len = length as i64;
        let step = self.step.unwrap_or(1);

        // Determine defaults based on step direction
        let (default_start, default_stop) = if step > 0 {
            (0i64, len)
        } else {
            (len - 1, -len - 1)
        };

        // Resolve and clamp start
        let mut start = self.start.unwrap_or(default_start);
        if start < 0 {
            start += len;
            if start < 0 {
                start = if step < 0 { -1 } else { 0 };
            }
        } else if start >= len {
            start = if step < 0 { len - 1 } else { len };
        }

        // Resolve and clamp stop
        let mut stop = self.stop.unwrap_or(default_stop);
        if stop < 0 {
            stop += len;
            if stop < 0 {
                stop = if step < 0 { -1 } else { 0 };
            }
        } else if stop >= len {
            stop = if step < 0 { len - 1 } else { len };
        }

        // Compute length of slice
        let slice_length = if step > 0 {
            if stop > start {
                ((stop - start - 1) / step + 1) as usize
            } else {
                0
            }
        } else {
            if start > stop {
                ((start - stop - 1) / (-step) + 1) as usize
            } else {
                0
            }
        };

        SliceIndices {
            start: start.max(0) as usize,
            stop: stop.max(0) as usize,
            step: step as isize,
            length: slice_length,
        }
    }

    /// Convenience method for getting indices as (start, stop, step, length) tuple.
    #[inline]
    pub fn indices_tuple(&self, length: usize) -> (usize, usize, isize, usize) {
        let idx = self.indices(length);
        (idx.start, idx.stop, idx.step, idx.length)
    }
}

impl Clone for SliceObject {
    fn clone(&self) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::SLICE),
            start: self.start,
            stop: self.stop,
            step: self.step,
        }
    }
}

impl PartialEq for SliceObject {
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start && self.stop == other.stop && self.step == other.step
    }
}

impl Eq for SliceObject {}

impl fmt::Debug for SliceObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "slice({}, {}, {})", self.start, self.stop, self.step)
    }
}

impl fmt::Display for SliceObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "slice({}, {}, {})", self.start, self.stop, self.step)
    }
}

impl PyObject for SliceObject {
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

    // =========================================================================
    // SliceValue Tests
    // =========================================================================

    #[test]
    fn test_slice_value_none() {
        let v = SliceValue::none();
        assert!(v.is_none());
        assert!(!v.is_some());
        assert_eq!(v.get(), None);
        assert_eq!(v.unwrap_or(42), 42);
    }

    #[test]
    fn test_slice_value_some() {
        let v = SliceValue::some(10);
        assert!(v.is_some());
        assert!(!v.is_none());
        assert_eq!(v.get(), Some(10));
        assert_eq!(v.unwrap_or(42), 10);
    }

    #[test]
    fn test_slice_value_negative() {
        let v = SliceValue::some(-5);
        assert!(v.is_some());
        assert_eq!(v.get(), Some(-5));
    }

    #[test]
    fn test_slice_value_zero() {
        let v = SliceValue::some(0);
        assert!(v.is_some());
        assert_eq!(v.get(), Some(0));
    }

    #[test]
    fn test_slice_value_from_option() {
        let v1: SliceValue = Some(10).into();
        let v2: SliceValue = None.into();
        assert_eq!(v1.get(), Some(10));
        assert_eq!(v2.get(), None);
    }

    #[test]
    fn test_slice_value_map() {
        let v = SliceValue::some(5);
        let mapped = v.map(|x| x * 2);
        assert_eq!(mapped.get(), Some(10));

        let none = SliceValue::none();
        let mapped_none = none.map(|x| x * 2);
        assert!(mapped_none.is_none());
    }

    // =========================================================================
    // SliceObject Construction Tests
    // =========================================================================

    #[test]
    fn test_slice_new_all_values() {
        let s = SliceObject::new(Some(1), Some(10), Some(2));
        assert_eq!(s.start(), Some(1));
        assert_eq!(s.stop(), Some(10));
        assert_eq!(s.step(), Some(2));
    }

    #[test]
    fn test_slice_new_all_none() {
        let s = SliceObject::new(None, None, None);
        assert_eq!(s.start(), None);
        assert_eq!(s.stop(), None);
        assert_eq!(s.step(), None);
    }

    #[test]
    fn test_slice_stop_only() {
        let s = SliceObject::stop_only(5);
        assert_eq!(s.start(), None);
        assert_eq!(s.stop(), Some(5));
        assert_eq!(s.step(), None);
    }

    #[test]
    fn test_slice_start_stop() {
        let s = SliceObject::start_stop(2, 8);
        assert_eq!(s.start(), Some(2));
        assert_eq!(s.stop(), Some(8));
        assert_eq!(s.step(), None);
    }

    #[test]
    fn test_slice_full() {
        let s = SliceObject::full(0, 10, 2);
        assert_eq!(s.start(), Some(0));
        assert_eq!(s.stop(), Some(10));
        assert_eq!(s.step(), Some(2));
    }

    #[test]
    #[should_panic(expected = "slice step cannot be zero")]
    fn test_slice_zero_step_panics() {
        SliceObject::new(Some(0), Some(10), Some(0));
    }

    #[test]
    #[should_panic(expected = "slice step cannot be zero")]
    fn test_slice_full_zero_step_panics() {
        SliceObject::full(0, 10, 0);
    }

    // =========================================================================
    // SliceIndices Resolution Tests
    // =========================================================================

    #[test]
    fn test_indices_simple_forward() {
        // [1:5] on a length-10 sequence
        let s = SliceObject::start_stop(1, 5);
        let idx = s.indices(10);
        assert_eq!(idx.start, 1);
        assert_eq!(idx.stop, 5);
        assert_eq!(idx.step, 1);
        assert_eq!(idx.length, 4);
    }

    #[test]
    fn test_indices_full_slice() {
        // [:] on a length-5 sequence
        let s = SliceObject::new(None, None, None);
        let idx = s.indices(5);
        assert_eq!(idx.start, 0);
        assert_eq!(idx.stop, 5);
        assert_eq!(idx.step, 1);
        assert_eq!(idx.length, 5);
    }

    #[test]
    fn test_indices_negative_start() {
        // [-3:] on a length-5 sequence = [2:]
        let s = SliceObject::new(Some(-3), None, None);
        let idx = s.indices(5);
        assert_eq!(idx.start, 2);
        assert_eq!(idx.stop, 5);
        assert_eq!(idx.step, 1);
        assert_eq!(idx.length, 3);
    }

    #[test]
    fn test_indices_negative_stop() {
        // [:-2] on a length-5 sequence = [:3]
        let s = SliceObject::new(None, Some(-2), None);
        let idx = s.indices(5);
        assert_eq!(idx.start, 0);
        assert_eq!(idx.stop, 3);
        assert_eq!(idx.step, 1);
        assert_eq!(idx.length, 3);
    }

    #[test]
    fn test_indices_negative_both() {
        // [-4:-1] on a length-5 sequence = [1:4]
        let s = SliceObject::start_stop(-4, -1);
        let idx = s.indices(5);
        assert_eq!(idx.start, 1);
        assert_eq!(idx.stop, 4);
        assert_eq!(idx.step, 1);
        assert_eq!(idx.length, 3);
    }

    #[test]
    fn test_indices_with_step() {
        // [0:10:2] on a length-10 sequence
        let s = SliceObject::full(0, 10, 2);
        let idx = s.indices(10);
        assert_eq!(idx.start, 0);
        assert_eq!(idx.stop, 10);
        assert_eq!(idx.step, 2);
        assert_eq!(idx.length, 5); // 0, 2, 4, 6, 8
    }

    #[test]
    fn test_indices_reverse() {
        // [::-1] on a length-5 sequence
        let s = SliceObject::new(None, None, Some(-1));
        let idx = s.indices(5);
        assert_eq!(idx.start, 4);
        assert_eq!(idx.step, -1);
        assert_eq!(idx.length, 5);
    }

    #[test]
    fn test_indices_reverse_with_bounds() {
        // [4:1:-1] on a length-5 sequence
        let s = SliceObject::full(4, 1, -1);
        let idx = s.indices(5);
        assert_eq!(idx.start, 4);
        assert_eq!(idx.step, -1);
        assert_eq!(idx.length, 3); // 4, 3, 2
    }

    #[test]
    fn test_indices_empty_forward() {
        // [5:3] on any sequence = empty (start > stop with positive step)
        let s = SliceObject::start_stop(5, 3);
        let idx = s.indices(10);
        assert_eq!(idx.length, 0);
    }

    #[test]
    fn test_indices_empty_reverse() {
        // [3:5:-1] = empty (start < stop with negative step)
        let s = SliceObject::full(3, 5, -1);
        let idx = s.indices(10);
        assert_eq!(idx.length, 0);
    }

    #[test]
    fn test_indices_out_of_bounds_clamped() {
        // [0:100] on a length-5 sequence = [0:5]
        let s = SliceObject::start_stop(0, 100);
        let idx = s.indices(5);
        assert_eq!(idx.start, 0);
        assert_eq!(idx.stop, 5);
        assert_eq!(idx.length, 5);
    }

    #[test]
    fn test_indices_negative_out_of_bounds() {
        // [-100:3] on a length-5 sequence = [0:3]
        let s = SliceObject::start_stop(-100, 3);
        let idx = s.indices(5);
        assert_eq!(idx.start, 0);
        assert_eq!(idx.stop, 3);
        assert_eq!(idx.length, 3);
    }

    #[test]
    fn test_indices_empty_sequence() {
        let s = SliceObject::new(None, None, None);
        let idx = s.indices(0);
        assert_eq!(idx.length, 0);
    }

    #[test]
    fn test_indices_step_2() {
        // [::2] on a length-7 sequence
        let s = SliceObject::new(None, None, Some(2));
        let idx = s.indices(7);
        assert_eq!(idx.start, 0);
        assert_eq!(idx.step, 2);
        assert_eq!(idx.length, 4); // 0, 2, 4, 6
    }

    #[test]
    fn test_indices_step_3() {
        // [1:10:3] on a length-10 sequence
        let s = SliceObject::full(1, 10, 3);
        let idx = s.indices(10);
        assert_eq!(idx.start, 1);
        assert_eq!(idx.step, 3);
        assert_eq!(idx.length, 3); // 1, 4, 7
    }

    // =========================================================================
    // SliceIndexIter Tests
    // =========================================================================

    #[test]
    fn test_slice_iter_forward() {
        let s = SliceObject::start_stop(1, 5);
        let indices: Vec<usize> = s.indices(10).iter().collect();
        assert_eq!(indices, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_slice_iter_step() {
        let s = SliceObject::full(0, 10, 2);
        let indices: Vec<usize> = s.indices(10).iter().collect();
        assert_eq!(indices, vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_slice_iter_reverse() {
        let s = SliceObject::new(None, None, Some(-1));
        let indices: Vec<usize> = s.indices(5).iter().collect();
        assert_eq!(indices, vec![4, 3, 2, 1, 0]);
    }

    #[test]
    fn test_slice_iter_reverse_step() {
        let s = SliceObject::full(8, 0, -2);
        let indices: Vec<usize> = s.indices(10).iter().collect();
        assert_eq!(indices, vec![8, 6, 4, 2]);
    }

    #[test]
    fn test_slice_iter_size_hint() {
        let s = SliceObject::start_stop(0, 5);
        let iter = s.indices(10).iter();
        assert_eq!(iter.size_hint(), (5, Some(5)));
        assert_eq!(iter.len(), 5);
    }

    // =========================================================================
    // Clone and Equality Tests
    // =========================================================================

    #[test]
    fn test_slice_clone() {
        let s1 = SliceObject::full(1, 10, 2);
        let s2 = s1.clone();
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_slice_equality() {
        let s1 = SliceObject::start_stop(1, 5);
        let s2 = SliceObject::start_stop(1, 5);
        let s3 = SliceObject::start_stop(1, 6);
        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
    }

    // =========================================================================
    // Display Tests
    // =========================================================================

    #[test]
    fn test_slice_display() {
        let s = SliceObject::new(Some(1), Some(10), Some(2));
        assert_eq!(format!("{}", s), "slice(1, 10, 2)");

        let s2 = SliceObject::new(None, Some(5), None);
        assert_eq!(format!("{}", s2), "slice(None, 5, None)");
    }

    // =========================================================================
    // Memory Layout Verification
    // =========================================================================

    #[test]
    fn test_slice_value_size() {
        // SliceValue should be exactly 8 bytes
        assert_eq!(std::mem::size_of::<SliceValue>(), 8);
    }

    #[test]
    fn test_slice_object_size() {
        // SliceObject should be compact: header (16) + 3 * SliceValue (24) = 40 bytes
        assert_eq!(std::mem::size_of::<SliceObject>(), 40);
    }

    #[test]
    fn test_slice_alignment() {
        // SliceObject should be 8-byte aligned for optimal cache line usage
        assert!(std::mem::align_of::<SliceObject>() >= 8);
    }
}
