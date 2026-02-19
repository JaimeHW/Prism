//! Python bytes and bytearray object implementation.
//!
//! Provides a compact heap object for immutable `bytes` and mutable
//! `bytearray` values. Both are represented by the same storage layout and
//! distinguished by `header.type_id`.

use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use std::fmt;

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

    /// Clone bytes into a new vector.
    #[inline]
    pub fn to_vec(&self) -> Vec<u8> {
        self.data.clone()
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
