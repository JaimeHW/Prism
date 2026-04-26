//! List object implementation.
//!
//! High-performance mutable sequence type.

use crate::object::shaped_object::ShapedObject;
use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use crate::types::slice::SliceObject;
use prism_core::Value;

/// Errors raised by slice assignment when the replacement sequence does not
/// satisfy Python's extended-slice invariants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListSliceAssignError {
    ExtendedSliceSizeMismatch { expected: usize, actual: usize },
}

impl std::fmt::Display for ListSliceAssignError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExtendedSliceSizeMismatch { expected, actual } => write!(
                f,
                "attempt to assign sequence of size {} to extended slice of size {}",
                actual, expected
            ),
        }
    }
}

impl std::error::Error for ListSliceAssignError {}

/// Python list object.
///
/// Uses a Vec for dynamic growth. The capacity is managed internally
/// for amortized O(1) appends.
#[repr(C)]
#[derive(Debug)]
pub struct ListObject {
    /// Object header.
    pub header: ObjectHeader,
    /// List items.
    items: Vec<Value>,
}

impl ListObject {
    /// Create a new empty list.
    #[inline]
    pub fn new() -> Self {
        Self {
            header: ObjectHeader::new(TypeId::LIST),
            items: Vec::new(),
        }
    }

    /// Create a list with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::LIST),
            items: Vec::with_capacity(capacity),
        }
    }

    /// Create a list from an iterator.
    #[inline]
    pub fn from_iter<I: IntoIterator<Item = Value>>(iter: I) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::LIST),
            items: iter.into_iter().collect(),
        }
    }

    /// Create a list from a slice.
    #[inline]
    pub fn from_slice(slice: &[Value]) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::LIST),
            items: slice.to_vec(),
        }
    }

    /// Get the length of the list.
    #[inline]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if the list is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Get an item by index (supports negative indexing).
    #[inline]
    pub fn get(&self, index: i64) -> Option<Value> {
        let idx = self.normalize_index(index)?;
        self.items.get(idx).copied()
    }

    /// Get an item by index without bounds checking.
    ///
    /// # Safety
    /// Index must be in bounds.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> Value {
        debug_assert!(index < self.items.len());
        unsafe { *self.items.get_unchecked(index) }
    }

    /// Set an item by index (supports negative indexing).
    #[inline]
    pub fn set(&mut self, index: i64, value: Value) -> bool {
        if let Some(idx) = self.normalize_index(index) {
            self.items[idx] = value;
            true
        } else {
            false
        }
    }

    /// Append an item to the end.
    #[inline]
    pub fn push(&mut self, value: Value) {
        self.items.push(value);
    }

    /// Remove and return the last item.
    #[inline]
    pub fn pop(&mut self) -> Option<Value> {
        self.items.pop()
    }

    /// Insert an item at index.
    pub fn insert(&mut self, index: i64, value: Value) {
        let idx = self.normalize_index_for_insert(index);
        self.items.insert(idx, value);
    }

    /// Remove and return item at index.
    pub fn remove(&mut self, index: i64) -> Option<Value> {
        let idx = self.normalize_index(index)?;
        Some(self.items.remove(idx))
    }

    /// Clear all items.
    #[inline]
    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Extend list with items from iterator.
    #[inline]
    pub fn extend<I: IntoIterator<Item = Value>>(&mut self, iter: I) {
        self.items.extend(iter);
    }

    /// Replace the elements selected by `slice` with the provided replacement
    /// sequence, matching Python list slice-assignment semantics.
    pub fn assign_slice<I>(
        &mut self,
        slice: &SliceObject,
        replacement: I,
    ) -> Result<(), ListSliceAssignError>
    where
        I: IntoIterator<Item = Value>,
    {
        let replacement: Vec<Value> = replacement.into_iter().collect();
        let indices = slice.indices(self.len());

        if indices.step == 1 {
            let end = indices.stop.max(indices.start);
            self.items.splice(indices.start..end, replacement);
            return Ok(());
        }

        if replacement.len() != indices.length {
            return Err(ListSliceAssignError::ExtendedSliceSizeMismatch {
                expected: indices.length,
                actual: replacement.len(),
            });
        }

        for (index, value) in indices.iter().zip(replacement.into_iter()) {
            self.items[index] = value;
        }

        Ok(())
    }

    /// Delete the elements selected by `slice`, matching Python's list slice
    /// deletion semantics for both contiguous and extended slices.
    pub fn delete_slice(&mut self, slice: &SliceObject) {
        let indices = slice.indices(self.len());

        if indices.length == 0 {
            return;
        }

        if indices.step == 1 {
            let end = indices.stop.max(indices.start);
            self.items.drain(indices.start..end);
            return;
        }

        let mut removal_indices: Vec<usize> = indices.iter().collect();
        removal_indices.sort_unstable_by(|left, right| right.cmp(left));

        for index in removal_indices {
            self.items.remove(index);
        }
    }

    /// Reverse the list in place.
    #[inline]
    pub fn reverse(&mut self) {
        self.items.reverse();
    }

    /// Concatenate two lists into a new list.
    ///
    /// This is the Python `list + list` operation. Returns a new list containing
    /// all elements from both lists in order.
    ///
    /// # Performance
    ///
    /// Pre-allocates exact capacity for the result to avoid reallocations.
    /// O(n + m) time complexity where n and m are the lengths of the two lists.
    #[inline]
    pub fn concat(&self, other: &ListObject) -> Self {
        let total_len = self.items.len() + other.items.len();
        let mut result = Vec::with_capacity(total_len);
        result.extend(self.items.iter().copied());
        result.extend(other.items.iter().copied());
        Self {
            header: ObjectHeader::new(TypeId::LIST),
            items: result,
        }
    }

    /// Repeat this list `n` times into a freshly allocated list.
    ///
    /// Mirrors Python's `list * int` semantics by returning a new list and
    /// preserving element order for each repetition.
    #[inline]
    pub fn repeat(&self, n: usize) -> Option<Self> {
        if n == 0 || self.items.is_empty() {
            return Some(Self::new());
        }

        let total_len = self.items.len().checked_mul(n)?;
        let mut result = Vec::new();
        result.try_reserve_exact(total_len).ok()?;
        result.extend_from_slice(&self.items);

        while result.len() < total_len {
            let remaining = total_len - result.len();
            let copy_len = result.len().min(remaining);
            result.extend_from_within(..copy_len);
        }

        Some(Self {
            header: ObjectHeader::new(TypeId::LIST),
            items: result,
        })
    }

    /// Repeat this list in place while preserving object identity.
    ///
    /// Uses the same doubling copy strategy as `repeat` so `list *= n` stays
    /// O(len * n) with logarithmic source copies and exact preallocation.
    #[inline]
    pub fn repeat_in_place(&mut self, n: usize) -> Option<()> {
        if n == 0 || self.items.is_empty() {
            self.items.clear();
            return Some(());
        }
        if n == 1 {
            return Some(());
        }

        let original_len = self.items.len();
        let total_len = original_len.checked_mul(n)?;
        self.items
            .try_reserve_exact(total_len - original_len)
            .ok()?;

        while self.items.len() < total_len {
            let remaining = total_len - self.items.len();
            let copy_len = self.items.len().min(remaining);
            self.items.extend_from_within(..copy_len);
        }

        Some(())
    }

    /// Get a slice of the list.
    pub fn slice(&self, start: Option<i64>, end: Option<i64>, step: Option<i64>) -> Self {
        let len = self.len() as i64;
        let step = step.unwrap_or(1);

        if step == 0 {
            return Self::new(); // Invalid step
        }

        let (start, end) = if step > 0 {
            let start = start.map(|s| s.clamp(0, len)).unwrap_or(0);
            let end = end.map(|e| e.clamp(0, len)).unwrap_or(len);
            (start, end)
        } else {
            let start = start.map(|s| s.clamp(-1, len - 1)).unwrap_or(len - 1);
            let end = end.map(|e| e.clamp(-1, len - 1)).unwrap_or(-1);
            (start, end)
        };

        let mut result = Vec::new();
        let mut i = start;

        if step > 0 {
            while i < end {
                if let Some(val) = self.items.get(i as usize) {
                    result.push(*val);
                }
                i += step;
            }
        } else {
            while i > end {
                if let Some(val) = self.items.get(i as usize) {
                    result.push(*val);
                }
                i += step;
            }
        }

        Self::from_iter(result)
    }

    /// Check if list contains a value.
    pub fn contains(&self, value: Value) -> bool {
        // TODO: Use proper equality comparison
        self.items.iter().any(|v| {
            // Fast path: same bit pattern
            if let (Some(a), Some(b)) = (v.as_int(), value.as_int()) {
                return a == b;
            }
            if let (Some(a), Some(b)) = (v.as_float(), value.as_float()) {
                return a == b;
            }
            false
        })
    }

    /// Get iterator over items.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Value> {
        self.items.iter()
    }

    /// Get mutable iterator over items.
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Value> {
        self.items.iter_mut()
    }

    /// Get the underlying slice.
    #[inline]
    pub fn as_slice(&self) -> &[Value] {
        &self.items
    }

    // Internal: normalize index for get/set operations
    fn normalize_index(&self, index: i64) -> Option<usize> {
        let len = self.len() as i64;
        let normalized = if index < 0 { len + index } else { index };
        if normalized >= 0 && normalized < len {
            Some(normalized as usize)
        } else {
            None
        }
    }

    // Internal: normalize index for insert (allows index == len)
    fn normalize_index_for_insert(&self, index: i64) -> usize {
        let len = self.len() as i64;
        let normalized = if index < 0 { len + index } else { index };
        normalized.clamp(0, len) as usize
    }
}

impl Default for ListObject {
    fn default() -> Self {
        Self::new()
    }
}

#[inline(always)]
pub fn value_as_list_ref(value: Value) -> Option<&'static ListObject> {
    let ptr = value.as_object_ptr()?;
    object_ptr_as_list_ref(ptr)
}

#[inline(always)]
pub fn object_ptr_as_list_ref(ptr: *const ()) -> Option<&'static ListObject> {
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    match header.type_id {
        TypeId::LIST => Some(unsafe { &*(ptr as *const ListObject) }),
        type_id if type_id.raw() >= TypeId::FIRST_USER_TYPE => unsafe {
            (&*(ptr as *const ShapedObject)).list_backing()
        },
        _ => None,
    }
}

#[inline(always)]
pub fn object_ptr_as_list_mut(ptr: *mut ()) -> Option<&'static mut ListObject> {
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    match header.type_id {
        TypeId::LIST => Some(unsafe { &mut *(ptr as *mut ListObject) }),
        type_id if type_id.raw() >= TypeId::FIRST_USER_TYPE => unsafe {
            (&mut *(ptr as *mut ShapedObject)).list_backing_mut()
        },
        _ => None,
    }
}

impl PyObject for ListObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}
