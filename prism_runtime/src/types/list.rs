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
    pub fn repeat(&self, n: usize) -> Self {
        if n == 0 || self.items.is_empty() {
            return Self::new();
        }

        let total_len = self.items.len() * n;
        let mut result = Vec::with_capacity(total_len);
        for _ in 0..n {
            result.extend(self.items.iter().copied());
        }

        Self {
            header: ObjectHeader::new(TypeId::LIST),
            items: result,
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::shape::Shape;

    #[test]
    fn test_list_basic() {
        let mut list = ListObject::new();
        assert!(list.is_empty());

        list.push(Value::int(1).unwrap());
        list.push(Value::int(2).unwrap());
        list.push(Value::int(3).unwrap());

        assert_eq!(list.len(), 3);
        assert_eq!(list.get(0).unwrap().as_int(), Some(1));
        assert_eq!(list.get(1).unwrap().as_int(), Some(2));
        assert_eq!(list.get(2).unwrap().as_int(), Some(3));
    }

    #[test]
    fn test_list_negative_index() {
        let list = ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]);

        assert_eq!(list.get(-1).unwrap().as_int(), Some(3));
        assert_eq!(list.get(-2).unwrap().as_int(), Some(2));
        assert_eq!(list.get(-3).unwrap().as_int(), Some(1));
        assert!(list.get(-4).is_none());
    }

    #[test]
    fn test_list_pop() {
        let mut list = ListObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);

        assert_eq!(list.pop().unwrap().as_int(), Some(2));
        assert_eq!(list.pop().unwrap().as_int(), Some(1));
        assert!(list.pop().is_none());
    }

    #[test]
    fn test_list_insert() {
        let mut list = ListObject::from_slice(&[Value::int(1).unwrap(), Value::int(3).unwrap()]);

        list.insert(1, Value::int(2).unwrap());
        assert_eq!(list.get(0).unwrap().as_int(), Some(1));
        assert_eq!(list.get(1).unwrap().as_int(), Some(2));
        assert_eq!(list.get(2).unwrap().as_int(), Some(3));
    }

    #[test]
    fn test_list_repeat() {
        let list = ListObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        let repeated = list.repeat(3);

        assert_eq!(repeated.len(), 6);
        assert_eq!(
            repeated.as_slice(),
            &[
                Value::int_unchecked(1),
                Value::int_unchecked(2),
                Value::int_unchecked(1),
                Value::int_unchecked(2),
                Value::int_unchecked(1),
                Value::int_unchecked(2),
            ]
        );
    }

    #[test]
    fn test_list_repeat_zero_returns_empty() {
        let list = ListObject::from_slice(&[Value::int(1).unwrap()]);
        let repeated = list.repeat(0);

        assert!(repeated.is_empty());
    }

    #[test]
    fn test_assign_slice_replaces_contiguous_region() {
        let mut list = ListObject::from_slice(&[
            Value::int(0).unwrap(),
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]);

        list.assign_slice(
            &SliceObject::start_stop(1, 3),
            [Value::int(10).unwrap(), Value::int(11).unwrap()],
        )
        .expect("contiguous slice assignment should succeed");

        assert_eq!(
            list.as_slice(),
            &[
                Value::int_unchecked(0),
                Value::int_unchecked(10),
                Value::int_unchecked(11),
                Value::int_unchecked(3)
            ]
        );
    }

    #[test]
    fn test_assign_slice_can_grow_list() {
        let mut list = ListObject::from_slice(&[
            Value::int(0).unwrap(),
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
        ]);

        list.assign_slice(
            &SliceObject::start_stop(1, 2),
            [
                Value::int(10).unwrap(),
                Value::int(11).unwrap(),
                Value::int(12).unwrap(),
            ],
        )
        .expect("slice assignment should allow growth");

        assert_eq!(
            list.as_slice(),
            &[
                Value::int_unchecked(0),
                Value::int_unchecked(10),
                Value::int_unchecked(11),
                Value::int_unchecked(12),
                Value::int_unchecked(2),
            ]
        );
    }

    #[test]
    fn test_assign_slice_can_insert_into_empty_region() {
        let mut list = ListObject::from_slice(&[
            Value::int(0).unwrap(),
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
        ]);

        list.assign_slice(&SliceObject::start_stop(2, 1), [Value::int(99).unwrap()])
            .expect("empty-slice assignment should behave as insertion");

        assert_eq!(
            list.as_slice(),
            &[
                Value::int_unchecked(0),
                Value::int_unchecked(1),
                Value::int_unchecked(99),
                Value::int_unchecked(2),
            ]
        );
    }

    #[test]
    fn test_assign_slice_replaces_extended_slice_in_place() {
        let mut list = ListObject::from_slice(&[
            Value::int(0).unwrap(),
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
            Value::int(4).unwrap(),
            Value::int(5).unwrap(),
        ]);

        list.assign_slice(
            &SliceObject::full(0, 6, 2),
            [
                Value::int(10).unwrap(),
                Value::int(11).unwrap(),
                Value::int(12).unwrap(),
            ],
        )
        .expect("extended slice assignment should succeed when lengths match");

        assert_eq!(
            list.as_slice(),
            &[
                Value::int_unchecked(10),
                Value::int_unchecked(1),
                Value::int_unchecked(11),
                Value::int_unchecked(3),
                Value::int_unchecked(12),
                Value::int_unchecked(5),
            ]
        );
    }

    #[test]
    fn test_assign_slice_rejects_mismatched_extended_slice_lengths() {
        let mut list = ListObject::from_slice(&[
            Value::int(0).unwrap(),
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]);

        let error = list
            .assign_slice(&SliceObject::full(0, 4, 2), [Value::int(10).unwrap()])
            .expect_err("extended slice assignment should validate replacement length");

        assert_eq!(
            error,
            ListSliceAssignError::ExtendedSliceSizeMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn test_delete_slice_removes_contiguous_region() {
        let mut list = ListObject::from_slice(&[
            Value::int(0).unwrap(),
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]);

        list.delete_slice(&SliceObject::start_stop(1, 3));

        assert_eq!(
            list.as_slice(),
            &[Value::int_unchecked(0), Value::int_unchecked(3)]
        );
    }

    #[test]
    fn test_delete_slice_removes_extended_indices() {
        let mut list = ListObject::from_slice(&[
            Value::int(0).unwrap(),
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
            Value::int(4).unwrap(),
            Value::int(5).unwrap(),
        ]);

        list.delete_slice(&SliceObject::full(0, 6, 2));

        assert_eq!(
            list.as_slice(),
            &[
                Value::int_unchecked(1),
                Value::int_unchecked(3),
                Value::int_unchecked(5),
            ]
        );
    }

    #[test]
    fn test_object_ptr_as_list_ref_supports_heap_list_subclasses() {
        let object = Box::into_raw(Box::new(ShapedObject::new_list_backed(
            TypeId::from_raw(512),
            Shape::empty(),
        )));
        unsafe { &mut *object }
            .list_backing_mut()
            .expect("list backing should exist")
            .push(Value::int(5).unwrap());

        let list = object_ptr_as_list_ref(object as *const ())
            .expect("heap list subclass should expose native list storage");
        assert_eq!(list.as_slice(), &[Value::int(5).unwrap()]);

        unsafe {
            drop(Box::from_raw(object));
        }
    }
}
