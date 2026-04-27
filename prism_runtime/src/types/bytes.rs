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

        let mut data = Vec::new();
        data.try_reserve_exact(total_len).ok()?;
        data.extend_from_slice(&self.data);

        while data.len() < total_len {
            let remaining = total_len - data.len();
            let copy_len = data.len().min(remaining);
            data.extend_from_within(..copy_len);
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

#[inline(always)]
fn is_shaped_heap_type(type_id: TypeId) -> bool {
    type_id.raw() >= TypeId::FIRST_USER_TYPE
        && !crate::types::iter::is_native_iterator_type_id(type_id)
}

/// Borrow native byte-sequence storage from an object pointer.
#[inline]
pub fn object_ptr_as_bytes_ref(ptr: *const ()) -> Option<&'static BytesObject> {
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    match header.type_id {
        TypeId::BYTES | TypeId::BYTEARRAY => Some(unsafe { &*(ptr as *const BytesObject) }),
        type_id if is_shaped_heap_type(type_id) => unsafe {
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
