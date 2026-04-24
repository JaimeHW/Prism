//! Python bytes and bytearray object implementation.
//!
//! Provides a compact heap object for immutable `bytes` and mutable
//! `bytearray` values. Both are represented by the same storage layout and
//! distinguished by `header.type_id`.

use crate::object::shaped_object::ShapedObject;
use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use crate::types::slice::SliceObject;
use prism_core::Value;
use std::fmt;

/// Error raised when bytearray slice assignment cannot be applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ByteSliceAssignError {
    /// The receiver is immutable `bytes`, not mutable `bytearray`.
    Immutable,
    /// Extended slice assignment must preserve the selected slice length.
    ExtendedSliceSizeMismatch { expected: usize, actual: usize },
}

impl fmt::Display for ByteSliceAssignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Immutable => write!(f, "bytes object does not support item assignment"),
            Self::ExtendedSliceSizeMismatch { expected, actual } => write!(
                f,
                "attempt to assign bytes of size {} to extended slice of size {}",
                actual, expected
            ),
        }
    }
}

/// Shared object backing for `bytes` and `bytearray`.
///
/// - `TypeId::BYTES`: immutable semantics
/// - `TypeId::BYTEARRAY`: mutable semantics
#[repr(C)]
pub struct BytesObject {
    /// Object header.
    pub header: ObjectHeader,
    /// Byte storage.
    data: Vec<u8>,
}

impl BytesObject {
    /// Create an empty immutable bytes object.
    #[inline]
    pub fn new() -> Self {
        Self::from_vec(Vec::new())
    }

    /// Create an empty mutable bytearray object.
    #[inline]
    pub fn new_bytearray() -> Self {
        Self::from_vec_with_type(Vec::new(), TypeId::BYTEARRAY)
    }

    /// Create immutable bytes from a slice.
    #[inline]
    pub fn from_slice(data: &[u8]) -> Self {
        Self::from_vec(data.to_vec())
    }

    /// Create mutable bytearray from a slice.
    #[inline]
    pub fn bytearray_from_slice(data: &[u8]) -> Self {
        Self::from_vec_with_type(data.to_vec(), TypeId::BYTEARRAY)
    }

    /// Create immutable bytes from an owned vector.
    #[inline]
    pub fn from_vec(data: Vec<u8>) -> Self {
        Self::from_vec_with_type(data, TypeId::BYTES)
    }

    /// Create a byte sequence with an explicit type ID.
    #[inline]
    pub fn from_vec_with_type(data: Vec<u8>, type_id: TypeId) -> Self {
        debug_assert!(
            type_id == TypeId::BYTES || type_id == TypeId::BYTEARRAY,
            "BytesObject only supports TypeId::BYTES/TypeId::BYTEARRAY"
        );
        Self {
            header: ObjectHeader::new(type_id),
            data,
        }
    }

    /// Create bytes filled with a repeated value.
    #[inline]
    pub fn repeat(byte: u8, count: usize) -> Self {
        Self::from_vec(vec![byte; count])
    }

    /// Create a byte sequence with explicit type and repeated value.
    #[inline]
    pub fn repeat_with_type(byte: u8, count: usize, type_id: TypeId) -> Self {
        Self::from_vec_with_type(vec![byte; count], type_id)
    }

    /// Borrow the underlying byte slice.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Get the number of bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Check whether this object is mutable (`bytearray`).
    #[inline]
    pub fn is_bytearray(&self) -> bool {
        self.header.type_id == TypeId::BYTEARRAY
    }

    /// Get a byte by index (supports negative indexing).
    #[inline]
    pub fn get(&self, index: i64) -> Option<u8> {
        let idx = self.normalize_index(index)?;
        self.data.get(idx).copied()
    }

    /// Iterate bytes.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = u8> + '_ {
        self.data.iter().copied()
    }

    /// Append one byte for mutable instances.
    ///
    /// Returns false for immutable `bytes`.
    #[inline]
    pub fn push(&mut self, byte: u8) -> bool {
        if !self.is_bytearray() {
            return false;
        }
        self.data.push(byte);
        true
    }

    /// Append a byte slice for mutable instances.
    ///
    /// Returns false for immutable `bytes`.
    #[inline]
    pub fn extend_from_slice(&mut self, bytes: &[u8]) -> bool {
        if !self.is_bytearray() {
            return false;
        }
        self.data.extend_from_slice(bytes);
        true
    }

    /// Set one byte by index for mutable instances.
    ///
    /// Returns false for immutable `bytes` or out-of-bounds index.
    #[inline]
    pub fn set(&mut self, index: i64, byte: u8) -> bool {
        if !self.is_bytearray() {
            return false;
        }
        let Some(idx) = self.normalize_index(index) else {
            return false;
        };
        self.data[idx] = byte;
        true
    }

    /// Replace the selected slice with replacement bytes.
    ///
    /// Contiguous slices may grow or shrink the bytearray. Extended slices
    /// mirror CPython sequence semantics and therefore require an exact-length
    /// replacement.
    pub fn assign_slice(
        &mut self,
        slice: &SliceObject,
        replacement: &[u8],
    ) -> Result<(), ByteSliceAssignError> {
        if !self.is_bytearray() {
            return Err(ByteSliceAssignError::Immutable);
        }

        let indices = slice.indices(self.data.len());
        if indices.step == 1 {
            let end = indices.stop.max(indices.start);
            self.data
                .splice(indices.start..end, replacement.iter().copied());
            return Ok(());
        }

        if replacement.len() != indices.length {
            return Err(ByteSliceAssignError::ExtendedSliceSizeMismatch {
                expected: indices.length,
                actual: replacement.len(),
            });
        }

        for (index, byte) in indices.iter().zip(replacement.iter().copied()) {
            self.data[index] = byte;
        }

        Ok(())
    }

    /// Clone bytes into a new vector.
    #[inline]
    pub fn to_vec(&self) -> Vec<u8> {
        self.data.clone()
    }

    /// Materialize a Python slice of the byte sequence, preserving the concrete
    /// `bytes`/`bytearray` type.
    #[inline]
    pub fn slice(&self, slice: &SliceObject) -> Self {
        let indices = slice.indices(self.data.len());
        let mut data = Vec::with_capacity(indices.length);
        for idx in indices.iter() {
            if idx < self.data.len() {
                data.push(self.data[idx]);
            }
        }
        Self::from_vec_with_type(data, self.header.type_id)
    }

    /// Concatenate another byte slice, preserving this object's concrete type.
    #[inline]
    pub fn concat(&self, other: &[u8]) -> Self {
        let mut data = Vec::with_capacity(self.data.len() + other.len());
        data.extend_from_slice(&self.data);
        data.extend_from_slice(other);
        Self::from_vec_with_type(data, self.header.type_id)
    }

    /// Repeat the byte sequence `n` times, preserving the concrete type.
    ///
    /// Returns `None` when the repeated byte length would overflow `usize`.
    #[inline]
    pub fn repeat_sequence(&self, n: usize) -> Option<Self> {
        let total_len = self.data.len().checked_mul(n)?;
        if total_len == 0 {
            return Some(Self::from_vec_with_type(Vec::new(), self.header.type_id));
        }

        let mut data = Vec::with_capacity(total_len);
        for _ in 0..n {
            data.extend_from_slice(&self.data);
        }
        Some(Self::from_vec_with_type(data, self.header.type_id))
    }

    #[inline]
    fn normalize_index(&self, index: i64) -> Option<usize> {
        let len = self.data.len() as i64;
        let idx = if index < 0 { len + index } else { index };
        if idx < 0 || idx >= len {
            None
        } else {
            Some(idx as usize)
        }
    }
}

impl Clone for BytesObject {
    fn clone(&self) -> Self {
        Self::from_vec_with_type(self.data.clone(), self.header.type_id)
    }
}

impl fmt::Debug for BytesObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = if self.is_bytearray() {
            "bytearray"
        } else {
            "bytes"
        };
        f.debug_struct("BytesObject")
            .field("kind", &kind)
            .field("len", &self.data.len())
            .finish()
    }
}

impl PyObject for BytesObject {
    #[inline]
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    #[inline]
    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

/// Borrow native byte-sequence storage from exact `bytes`/`bytearray` values
/// and heap subclasses backed by native byte storage.
#[inline]
pub fn value_as_bytes_ref(value: Value) -> Option<&'static BytesObject> {
    let ptr = value.as_object_ptr()?;
    object_ptr_as_bytes_ref(ptr)
}

/// Borrow native byte-sequence storage from an object pointer.
#[inline]
pub fn object_ptr_as_bytes_ref(ptr: *const ()) -> Option<&'static BytesObject> {
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    match header.type_id {
        TypeId::BYTES | TypeId::BYTEARRAY => Some(unsafe { &*(ptr as *const BytesObject) }),
        type_id if type_id.raw() >= TypeId::FIRST_USER_TYPE => unsafe {
            (&*(ptr as *const ShapedObject)).bytes_backing()
        },
        _ => None,
    }
}

/// Clone the native byte-sequence payload from exact objects or subclasses.
#[inline]
pub fn clone_bytes_value(value: Value) -> Option<BytesObject> {
    value_as_bytes_ref(value).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::shape::Shape;

    #[test]
    fn test_new_bytes_empty() {
        let bytes = BytesObject::new();
        assert_eq!(bytes.header.type_id, TypeId::BYTES);
        assert!(bytes.is_empty());
        assert!(!bytes.is_bytearray());
    }

    #[test]
    fn test_new_bytearray_empty() {
        let bytearray = BytesObject::new_bytearray();
        assert_eq!(bytearray.header.type_id, TypeId::BYTEARRAY);
        assert!(bytearray.is_empty());
        assert!(bytearray.is_bytearray());
    }

    #[test]
    fn test_from_slice_and_len() {
        let bytes = BytesObject::from_slice(&[65, 66, 67]);
        assert_eq!(bytes.len(), 3);
        assert_eq!(bytes.as_bytes(), b"ABC");
    }

    #[test]
    fn test_bytearray_from_slice_type() {
        let bytearray = BytesObject::bytearray_from_slice(&[1, 2, 3]);
        assert_eq!(bytearray.header.type_id, TypeId::BYTEARRAY);
        assert_eq!(bytearray.as_bytes(), &[1, 2, 3]);
    }

    #[test]
    fn test_repeat_and_repeat_with_type() {
        let bytes = BytesObject::repeat(0, 5);
        assert_eq!(bytes.header.type_id, TypeId::BYTES);
        assert_eq!(bytes.as_bytes(), &[0, 0, 0, 0, 0]);

        let bytearray = BytesObject::repeat_with_type(255, 3, TypeId::BYTEARRAY);
        assert_eq!(bytearray.header.type_id, TypeId::BYTEARRAY);
        assert_eq!(bytearray.as_bytes(), &[255, 255, 255]);
    }

    #[test]
    fn test_repeat_sequence_preserves_type_and_contents() {
        let bytes = BytesObject::from_slice(b"ab")
            .repeat_sequence(3)
            .expect("bytes repeat should not overflow");
        assert_eq!(bytes.header.type_id, TypeId::BYTES);
        assert_eq!(bytes.as_bytes(), b"ababab");

        let bytearray = BytesObject::bytearray_from_slice(b"xy")
            .repeat_sequence(2)
            .expect("bytearray repeat should not overflow");
        assert_eq!(bytearray.header.type_id, TypeId::BYTEARRAY);
        assert_eq!(bytearray.as_bytes(), b"xyxy");
    }

    #[test]
    fn test_repeat_sequence_zero_count_returns_empty_same_type() {
        let bytes = BytesObject::from_slice(b"ab")
            .repeat_sequence(0)
            .expect("zero repeat should be representable");
        assert_eq!(bytes.header.type_id, TypeId::BYTES);
        assert!(bytes.is_empty());

        let bytearray = BytesObject::bytearray_from_slice(b"xy")
            .repeat_sequence(0)
            .expect("zero repeat should be representable");
        assert_eq!(bytearray.header.type_id, TypeId::BYTEARRAY);
        assert!(bytearray.is_empty());
    }

    #[test]
    fn test_get_positive_and_negative_index() {
        let bytes = BytesObject::from_slice(&[10, 20, 30]);
        assert_eq!(bytes.get(0), Some(10));
        assert_eq!(bytes.get(2), Some(30));
        assert_eq!(bytes.get(-1), Some(30));
        assert_eq!(bytes.get(-3), Some(10));
    }

    #[test]
    fn test_get_out_of_bounds() {
        let bytes = BytesObject::from_slice(&[10, 20, 30]);
        assert_eq!(bytes.get(3), None);
        assert_eq!(bytes.get(-4), None);
    }

    #[test]
    fn test_iter() {
        let bytes = BytesObject::from_slice(&[1, 2, 3, 4]);
        let collected: Vec<u8> = bytes.iter().collect();
        assert_eq!(collected, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_slice_preserves_bytes_type_and_supports_steps() {
        let bytes = BytesObject::from_slice(b"abcdef");
        let slice = SliceObject::full(0, 6, 2);
        let sliced = bytes.slice(&slice);

        assert_eq!(sliced.header.type_id, TypeId::BYTES);
        assert_eq!(sliced.as_bytes(), b"ace");
    }

    #[test]
    fn test_slice_preserves_bytearray_type_and_supports_reverse() {
        let bytes = BytesObject::bytearray_from_slice(b"abcd");
        let slice = SliceObject::new(None, None, Some(-1));
        let sliced = bytes.slice(&slice);

        assert_eq!(sliced.header.type_id, TypeId::BYTEARRAY);
        assert_eq!(sliced.as_bytes(), b"dcba");
    }

    #[test]
    fn test_concat_preserves_bytes_type() {
        let bytes = BytesObject::from_slice(b"abc");
        let result = bytes.concat(b"def");
        assert_eq!(result.header.type_id, TypeId::BYTES);
        assert_eq!(result.as_bytes(), b"abcdef");
    }

    #[test]
    fn test_concat_preserves_bytearray_type() {
        let bytearray = BytesObject::bytearray_from_slice(b"abc");
        let result = bytearray.concat(b"def");
        assert_eq!(result.header.type_id, TypeId::BYTEARRAY);
        assert_eq!(result.as_bytes(), b"abcdef");
    }

    #[test]
    fn test_push_immutable_bytes_rejected() {
        let mut bytes = BytesObject::from_slice(&[1, 2, 3]);
        assert!(!bytes.push(4));
        assert_eq!(bytes.as_bytes(), &[1, 2, 3]);
    }

    #[test]
    fn test_push_mutable_bytearray() {
        let mut bytearray = BytesObject::new_bytearray();
        assert!(bytearray.push(1));
        assert!(bytearray.push(2));
        assert_eq!(bytearray.as_bytes(), &[1, 2]);
    }

    #[test]
    fn test_extend_immutable_bytes_rejected() {
        let mut bytes = BytesObject::from_slice(&[1, 2, 3]);
        assert!(!bytes.extend_from_slice(&[4, 5]));
        assert_eq!(bytes.as_bytes(), &[1, 2, 3]);
    }

    #[test]
    fn test_extend_mutable_bytearray() {
        let mut bytearray = BytesObject::bytearray_from_slice(&[1, 2]);
        assert!(bytearray.extend_from_slice(&[3, 4]));
        assert_eq!(bytearray.as_bytes(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_set_immutable_bytes_rejected() {
        let mut bytes = BytesObject::from_slice(&[1, 2, 3]);
        assert!(!bytes.set(0, 9));
        assert_eq!(bytes.as_bytes(), &[1, 2, 3]);
    }

    #[test]
    fn test_set_mutable_bytearray_positive_and_negative() {
        let mut bytearray = BytesObject::bytearray_from_slice(&[1, 2, 3]);
        assert!(bytearray.set(0, 9));
        assert!(bytearray.set(-1, 7));
        assert_eq!(bytearray.as_bytes(), &[9, 2, 7]);
    }

    #[test]
    fn test_set_mutable_out_of_bounds() {
        let mut bytearray = BytesObject::bytearray_from_slice(&[1, 2, 3]);
        assert!(!bytearray.set(3, 9));
        assert!(!bytearray.set(-4, 9));
        assert_eq!(bytearray.as_bytes(), &[1, 2, 3]);
    }

    #[test]
    fn test_assign_bytearray_contiguous_slice_can_resize() {
        let mut bytearray = BytesObject::bytearray_from_slice(b"abcdef");
        bytearray
            .assign_slice(&SliceObject::start_stop(2, 4), b"XYZ")
            .expect("contiguous bytearray slice assignment should resize");
        assert_eq!(bytearray.as_bytes(), b"abXYZef");

        bytearray
            .assign_slice(&SliceObject::new(Some(-5), None, None), b"12345")
            .expect("negative slice bounds should be normalized");
        assert_eq!(bytearray.as_bytes(), b"ab12345");
    }

    #[test]
    fn test_assign_bytearray_extended_slice_requires_matching_length() {
        let mut bytearray = BytesObject::bytearray_from_slice(b"abcdef");
        bytearray
            .assign_slice(&SliceObject::full(0, 6, 2), b"XYZ")
            .expect("matching extended slice assignment should succeed");
        assert_eq!(bytearray.as_bytes(), b"XbYdZf");

        let err = bytearray
            .assign_slice(&SliceObject::full(0, 6, 2), b"12")
            .expect_err("extended bytearray slice size mismatch should fail");
        assert_eq!(
            err,
            ByteSliceAssignError::ExtendedSliceSizeMismatch {
                expected: 3,
                actual: 2
            }
        );
    }

    #[test]
    fn test_clone_preserves_type_and_contents() {
        let original = BytesObject::bytearray_from_slice(&[1, 2, 3]);
        let cloned = original.clone();
        assert_eq!(cloned.header.type_id, TypeId::BYTEARRAY);
        assert_eq!(cloned.as_bytes(), &[1, 2, 3]);
    }

    #[test]
    fn test_debug_includes_kind() {
        let bytes = BytesObject::from_slice(&[1, 2, 3]);
        let dbg = format!("{:?}", bytes);
        assert!(dbg.contains("bytes"));
    }

    #[test]
    fn test_value_as_bytes_ref_reads_heap_subclass_native_storage() {
        let object = ShapedObject::new_bytes_backed(
            TypeId::from_raw(TypeId::FIRST_USER_TYPE + 50),
            Shape::empty(),
            BytesObject::from_slice(b"payload"),
        );
        let value = Value::object_ptr(Box::into_raw(Box::new(object)) as *const ());

        assert_eq!(
            value_as_bytes_ref(value)
                .expect("bytes backing should be visible")
                .as_bytes(),
            b"payload"
        );

        unsafe {
            drop(Box::from_raw(
                value.as_object_ptr().unwrap() as *mut ShapedObject
            ));
        }
    }
}
