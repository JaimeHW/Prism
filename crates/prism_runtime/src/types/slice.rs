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
use crate::types::int::{bigint_to_saturated_i64, bigint_to_value, value_to_bigint};
use num_bigint::BigInt;
use prism_core::Value;
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
        self.remaining -= 1;
        if self.remaining != 0 {
            self.current = self.current.saturating_add(self.step);
        }
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
    /// Original Python start component.
    start: Value,
    /// Original Python stop component.
    stop: Value,
    /// Original Python step component.
    step: Value,
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
    /// A zero step is a valid slice object state; applying that slice to a
    /// sequence raises `ValueError`, matching CPython.
    #[inline]
    pub fn new(start: Value, stop: Value, step: Value) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::SLICE),
            start,
            stop,
            step,
        }
    }

    /// Create a slice from already-validated integer components.
    #[inline]
    pub fn from_indices(start: Option<i64>, stop: Option<i64>, step: Option<i64>) -> Self {
        Self::new(
            optional_index_to_value(start),
            optional_index_to_value(stop),
            optional_index_to_value(step),
        )
    }

    /// Create a slice with only stop: `slice(None, stop, None)` → `[:stop]`.
    #[inline]
    pub fn stop_only(stop: i64) -> Self {
        Self::from_indices(None, Some(stop), None)
    }

    /// Create a slice with start and stop: `slice(start, stop, None)` → `[start:stop]`.
    #[inline]
    pub fn start_stop(start: i64, stop: i64) -> Self {
        Self::from_indices(Some(start), Some(stop), None)
    }

    /// Create a full slice: `slice(start, stop, step)` → `[start:stop:step]`.
    #[inline]
    pub fn full(start: i64, stop: i64, step: i64) -> Self {
        Self::from_indices(Some(start), Some(stop), Some(step))
    }

    /// Return the original Python start component.
    #[inline(always)]
    pub fn start_value(&self) -> Value {
        self.start
    }

    /// Return the original Python stop component.
    #[inline(always)]
    pub fn stop_value(&self) -> Value {
        self.stop
    }

    /// Return the original Python step component.
    #[inline(always)]
    pub fn step_value(&self) -> Value {
        self.step
    }

    /// Get the start value when it is `None` or a native integer.
    #[inline(always)]
    pub fn start(&self) -> Option<i64> {
        slice_component_to_i64(self.start)
    }

    /// Get the stop value when it is `None` or a native integer.
    #[inline(always)]
    pub fn stop(&self) -> Option<i64> {
        slice_component_to_i64(self.stop)
    }

    /// Get the step value when it is `None` or a native integer.
    #[inline(always)]
    pub fn step(&self) -> Option<i64> {
        slice_component_to_i64(self.step)
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
        let step = self.step().unwrap_or(1);

        // Determine defaults based on step direction
        let (default_start, default_stop) = if step > 0 {
            (0i64, len)
        } else {
            (len - 1, -len - 1)
        };

        // Resolve and clamp start
        let mut start = self.start().unwrap_or(default_start);
        if start < 0 {
            start += len;
            if start < 0 {
                start = if step < 0 { -1 } else { 0 };
            }
        } else if start >= len {
            start = if step < 0 { len - 1 } else { len };
        }

        // Resolve and clamp stop
        let mut stop = self.stop().unwrap_or(default_stop);
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
        self.start.raw_bits() == other.start.raw_bits()
            && self.stop.raw_bits() == other.stop.raw_bits()
            && self.step.raw_bits() == other.step.raw_bits()
    }
}

impl Eq for SliceObject {}

impl fmt::Debug for SliceObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "slice({:?}, {:?}, {:?})",
            self.start.raw_bits(),
            self.stop.raw_bits(),
            self.step.raw_bits()
        )
    }
}

impl fmt::Display for SliceObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "slice(...)")
    }
}

#[inline]
fn optional_index_to_value(value: Option<i64>) -> Value {
    match value {
        Some(value) => index_to_value(value),
        None => Value::none(),
    }
}

#[inline]
fn index_to_value(value: i64) -> Value {
    Value::int(value).unwrap_or_else(|| bigint_to_value(BigInt::from(value)))
}

#[inline]
fn slice_component_to_i64(value: Value) -> Option<i64> {
    if value.is_none() {
        return None;
    }
    if let Some(boolean) = value.as_bool() {
        return Some(i64::from(boolean));
    }
    if let Some(integer) = value.as_int() {
        return Some(integer);
    }
    value_to_bigint(value).map(|integer| bigint_to_saturated_i64(&integer))
}

impl PyObject for SliceObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}
