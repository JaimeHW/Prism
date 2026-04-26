//! Python `memoryview` object support.
//!
//! This is a compact one-dimensional byte-buffer view used by Prism's current
//! buffer-protocol surface. It supports the formats exercised by the CPython
//! stdlib fast paths (`B`, `b`, `c`, `H`, `h`, `I`, `i`, `L`, `l`, `Q`, `q`)
//! and keeps the original exported object for `memoryview.obj`.

use crate::allocation_context::alloc_value_in_current_heap_or_box;
use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use crate::types::bytes::BytesObject;
use crate::types::int::bigint_to_value;
use crate::types::slice::SliceObject;
use num_bigint::BigInt;
use prism_core::Value;
use std::fmt;

/// Native element format for one-dimensional memory views.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryViewFormat {
    /// Unsigned byte (`B`).
    UnsignedByte,
    /// Signed byte (`b`).
    SignedByte,
    /// Single byte bytes object (`c`).
    Char,
    /// Native-endian unsigned short (`H`).
    UnsignedShort,
    /// Native-endian signed short (`h`).
    SignedShort,
    /// Native-endian unsigned int (`I`).
    UnsignedInt,
    /// Native-endian signed int (`i`).
    SignedInt,
    /// Native-endian unsigned long (`L`).
    UnsignedLong,
    /// Native-endian signed long (`l`).
    SignedLong,
    /// Native-endian unsigned long long (`Q`).
    UnsignedLongLong,
    /// Native-endian signed long long (`q`).
    SignedLongLong,
}

impl MemoryViewFormat {
    /// Parse a CPython memoryview cast format string.
    pub fn parse(format: &str) -> Option<Self> {
        let format = format.strip_prefix('@').unwrap_or(format);
        let mut chars = format.chars();
        let ch = chars.next()?;
        if chars.next().is_some() {
            return None;
        }
        match ch {
            'B' => Some(Self::UnsignedByte),
            'b' => Some(Self::SignedByte),
            'c' => Some(Self::Char),
            'H' => Some(Self::UnsignedShort),
            'h' => Some(Self::SignedShort),
            'I' => Some(Self::UnsignedInt),
            'i' => Some(Self::SignedInt),
            'L' => Some(Self::UnsignedLong),
            'l' => Some(Self::SignedLong),
            'Q' => Some(Self::UnsignedLongLong),
            'q' => Some(Self::SignedLongLong),
            _ => None,
        }
    }

    /// CPython-visible format string.
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UnsignedByte => "B",
            Self::SignedByte => "b",
            Self::Char => "c",
            Self::UnsignedShort => "H",
            Self::SignedShort => "h",
            Self::UnsignedInt => "I",
            Self::SignedInt => "i",
            Self::UnsignedLong => "L",
            Self::SignedLong => "l",
            Self::UnsignedLongLong => "Q",
            Self::SignedLongLong => "q",
        }
    }

    /// Number of bytes per element.
    #[inline]
    pub fn item_size(self) -> usize {
        match self {
            Self::UnsignedByte | Self::SignedByte | Self::Char => 1,
            Self::UnsignedShort | Self::SignedShort => 2,
            Self::UnsignedInt | Self::SignedInt | Self::UnsignedLong | Self::SignedLong => 4,
            Self::UnsignedLongLong | Self::SignedLongLong => 8,
        }
    }
}

/// A one-dimensional memoryview over a byte buffer.
#[repr(C)]
pub struct MemoryViewObject {
    /// Object header.
    pub header: ObjectHeader,
    source: Value,
    data: Vec<u8>,
    format: MemoryViewFormat,
    shape: Vec<usize>,
    readonly: bool,
    released: bool,
}

impl MemoryViewObject {
    /// Create a default unsigned-byte memoryview over copied bytes.
    #[inline]
    pub fn from_bytes(source: Value, bytes: &[u8], readonly: bool) -> Self {
        Self::from_vec(
            source,
            bytes.to_vec(),
            MemoryViewFormat::UnsignedByte,
            readonly,
        )
    }

    /// Create a memoryview from owned byte storage and explicit format.
    #[inline]
    pub fn from_vec(
        source: Value,
        data: Vec<u8>,
        format: MemoryViewFormat,
        readonly: bool,
    ) -> Self {
        let elements = data.len() / format.item_size();
        Self::from_vec_with_shape(source, data, format, readonly, vec![elements])
    }

    /// Create a memoryview from owned byte storage, explicit format, and shape.
    #[inline]
    pub fn from_vec_with_shape(
        source: Value,
        data: Vec<u8>,
        format: MemoryViewFormat,
        readonly: bool,
        shape: Vec<usize>,
    ) -> Self {
        debug_assert_eq!(
            data.len() % format.item_size(),
            0,
            "memoryview data must be an exact multiple of item size"
        );
        debug_assert_eq!(
            shape_element_count(&shape).saturating_mul(format.item_size()),
            data.len(),
            "memoryview shape must match byte length and item size"
        );
        Self {
            header: ObjectHeader::new(TypeId::MEMORYVIEW),
            source,
            data,
            format,
            shape,
            readonly,
            released: false,
        }
    }

    /// Return the object originally exported to this view.
    #[inline]
    pub fn source(&self) -> Value {
        self.source
    }

    /// Borrow the raw bytes represented by this view.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Clone the raw bytes represented by this view.
    #[inline]
    pub fn to_vec(&self) -> Vec<u8> {
        self.data.clone()
    }

    /// Format string.
    #[inline]
    pub fn format(&self) -> MemoryViewFormat {
        self.format
    }

    /// Format string as Python text.
    #[inline]
    pub fn format_str(&self) -> &'static str {
        self.format.as_str()
    }

    /// Element size in bytes.
    #[inline]
    pub fn item_size(&self) -> usize {
        self.format.item_size()
    }

    /// Number of exported dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// C-contiguous shape.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// C-contiguous strides in bytes.
    pub fn strides(&self) -> Vec<usize> {
        let mut strides = vec![self.item_size(); self.shape.len()];
        if self.shape.len() <= 1 {
            return strides;
        }

        for index in (0..self.shape.len() - 1).rev() {
            strides[index] = strides[index + 1].saturating_mul(self.shape[index + 1]);
        }
        strides
    }

    /// Number of logical elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len() / self.item_size()
    }

    /// Number of bytes in the view.
    #[inline]
    pub fn nbytes(&self) -> usize {
        self.data.len()
    }

    /// Whether the view is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Whether the exported storage is read-only.
    #[inline]
    pub fn readonly(&self) -> bool {
        self.readonly
    }

    /// Whether this view has been released.
    #[inline]
    pub fn released(&self) -> bool {
        self.released
    }

    /// Mark the view as released.
    #[inline]
    pub fn release(&mut self) {
        self.released = true;
    }

    /// Create a view with the same bytes and a different format.
    pub fn cast(&self, format: MemoryViewFormat) -> Option<Self> {
        self.cast_with_shape(format, None)
    }

    /// Create a view with the same bytes, a different format, and optional shape.
    pub fn cast_with_shape(
        &self,
        format: MemoryViewFormat,
        shape: Option<Vec<usize>>,
    ) -> Option<Self> {
        if self.data.len() % format.item_size() != 0 {
            return None;
        }
        let shape = shape.unwrap_or_else(|| vec![self.data.len() / format.item_size()]);
        if shape_element_count(&shape).checked_mul(format.item_size())? != self.data.len() {
            return None;
        }
        Some(Self::from_vec_with_shape(
            self.source,
            self.data.clone(),
            format,
            self.readonly,
            shape,
        ))
    }

    /// Slice by logical element indices.
    pub fn slice(&self, slice: &SliceObject) -> Self {
        let item_size = self.item_size();
        let indices = slice.indices(self.len());
        let mut data = Vec::with_capacity(indices.length * item_size);
        for idx in indices.iter() {
            let start = idx * item_size;
            let end = start + item_size;
            if end <= self.data.len() {
                data.extend_from_slice(&self.data[start..end]);
            }
        }
        Self::from_vec(self.source, data, self.format, self.readonly)
    }

    /// Decode an element by Python index semantics.
    pub fn get(&self, index: i64) -> Option<Value> {
        let len = self.len() as i64;
        let index = if index < 0 { len + index } else { index };
        if index < 0 || index >= len {
            return None;
        }
        let start = index as usize * self.item_size();
        let end = start + self.item_size();
        self.decode_item(&self.data[start..end])
    }

    /// Decode all elements as Python values.
    pub fn to_values(&self) -> Option<Vec<Value>> {
        self.data
            .chunks_exact(self.item_size())
            .map(|chunk| self.decode_item(chunk))
            .collect()
    }

    fn decode_item(&self, chunk: &[u8]) -> Option<Value> {
        match self.format {
            MemoryViewFormat::UnsignedByte => Some(Value::int_unchecked(i64::from(chunk[0]))),
            MemoryViewFormat::SignedByte => {
                Some(Value::int_unchecked(i64::from(i8::from_ne_bytes([
                    chunk[0]
                ]))))
            }
            MemoryViewFormat::Char => {
                let bytes = BytesObject::from_slice(&[chunk[0]]);
                Some(alloc_value_in_current_heap_or_box(bytes))
            }
            MemoryViewFormat::UnsignedShort => {
                Some(Value::int_unchecked(i64::from(u16::from_ne_bytes([
                    chunk[0], chunk[1],
                ]))))
            }
            MemoryViewFormat::SignedShort => {
                Some(Value::int_unchecked(i64::from(i16::from_ne_bytes([
                    chunk[0], chunk[1],
                ]))))
            }
            MemoryViewFormat::UnsignedInt | MemoryViewFormat::UnsignedLong => {
                Some(Value::int_unchecked(i64::from(u32::from_ne_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3],
                ]))))
            }
            MemoryViewFormat::SignedInt | MemoryViewFormat::SignedLong => {
                Some(Value::int_unchecked(i64::from(i32::from_ne_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3],
                ]))))
            }
            MemoryViewFormat::UnsignedLongLong => {
                let value = u64::from_ne_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                Some(bigint_to_value(BigInt::from(value)))
            }
            MemoryViewFormat::SignedLongLong => {
                let value = i64::from_ne_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                Some(Value::int_unchecked(value))
            }
        }
    }
}

impl Clone for MemoryViewObject {
    fn clone(&self) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::MEMORYVIEW),
            source: self.source,
            data: self.data.clone(),
            format: self.format,
            shape: self.shape.clone(),
            readonly: self.readonly,
            released: self.released,
        }
    }
}

impl fmt::Debug for MemoryViewObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryViewObject")
            .field("format", &self.format.as_str())
            .field("shape", &self.shape)
            .field("len", &self.len())
            .field("nbytes", &self.nbytes())
            .field("readonly", &self.readonly)
            .field("released", &self.released)
            .finish()
    }
}

impl PyObject for MemoryViewObject {
    #[inline]
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    #[inline]
    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

/// Borrow a memoryview object from a value.
#[inline]
pub fn value_as_memoryview_ref(value: Value) -> Option<&'static MemoryViewObject> {
    let ptr = value.as_object_ptr()?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    (header.type_id == TypeId::MEMORYVIEW).then(|| unsafe { &*(ptr as *const MemoryViewObject) })
}

/// Mutably borrow a memoryview object from a value.
#[inline]
pub fn value_as_memoryview_mut(value: Value) -> Option<&'static mut MemoryViewObject> {
    let ptr = value.as_object_ptr()?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    (header.type_id == TypeId::MEMORYVIEW).then(|| unsafe { &mut *(ptr as *mut MemoryViewObject) })
}

#[inline]
fn shape_element_count(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocation_context::RuntimeHeapBinding;
    use prism_core::Value;
    use prism_gc::heap::GcHeap;

    #[test]
    fn test_unsigned_byte_view_indexes_and_slices() {
        let view = MemoryViewObject::from_bytes(Value::none(), b"abcdef", true);

        assert_eq!(view.len(), 6);
        assert_eq!(view.ndim(), 1);
        assert_eq!(view.shape(), &[6]);
        assert_eq!(view.strides(), vec![1]);
        assert_eq!(view.nbytes(), 6);
        assert_eq!(view.get(1).and_then(|value| value.as_int()), Some(98));

        let sliced = view.slice(&SliceObject::new(Some(1), Some(6), Some(2)));
        assert_eq!(sliced.as_bytes(), b"bdf");
    }

    #[test]
    fn test_cast_native_unsigned_int_decodes_elements() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&7_u32.to_ne_bytes());
        bytes.extend_from_slice(&11_u32.to_ne_bytes());
        let view = MemoryViewObject::from_bytes(Value::none(), &bytes, true)
            .cast(MemoryViewFormat::UnsignedInt)
            .expect("byte length is divisible by u32 itemsize");

        assert_eq!(view.format_str(), "I");
        assert_eq!(view.item_size(), 4);
        assert_eq!(view.len(), 2);
        assert_eq!(view.get(0).and_then(|value| value.as_int()), Some(7));
        assert_eq!(view.get(-1).and_then(|value| value.as_int()), Some(11));
    }

    #[test]
    fn test_char_view_decodes_bytes_into_bound_heap() {
        let heap = GcHeap::with_defaults();
        let _binding = RuntimeHeapBinding::register(&heap);
        let view = MemoryViewObject::from_bytes(Value::none(), b"abc", true)
            .cast(MemoryViewFormat::Char)
            .expect("byte length is divisible by char itemsize");

        let value = view.get(0).expect("char view should decode one-byte bytes");
        let ptr = value
            .as_object_ptr()
            .expect("char view should decode to a bytes object");
        let bytes = unsafe { &*(ptr as *const BytesObject) };

        assert!(heap.contains(ptr));
        assert_eq!(bytes.as_bytes(), b"a");
    }

    #[test]
    fn test_cast_with_shape_records_multidimensional_metadata() {
        let view = MemoryViewObject::from_bytes(Value::none(), b"abcd", true)
            .cast_with_shape(MemoryViewFormat::UnsignedByte, Some(vec![2, 2]))
            .expect("shape product matches byte length");

        assert_eq!(view.ndim(), 2);
        assert_eq!(view.shape(), &[2, 2]);
        assert_eq!(view.strides(), vec![2, 1]);
        assert_eq!(view.as_bytes(), b"abcd");
    }

    #[test]
    fn test_released_flag_is_mutable() {
        let mut view = MemoryViewObject::from_bytes(Value::none(), b"x", true);
        assert!(!view.released());
        view.release();
        assert!(view.released());
    }
}
