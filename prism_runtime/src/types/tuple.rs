//! Tuple object implementation.
//!
//! Immutable sequence type with fixed size.

use crate::object::shaped_object::ShapedObject;
use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use prism_core::Value;

/// Python tuple object.
///
/// Immutable sequence with inline storage for small tuples.
/// Uses Box<[Value]> for zero-copy sharing.
#[repr(C)]
#[derive(Debug)]
pub struct TupleObject {
    /// Object header.
    pub header: ObjectHeader,
    /// Tuple items (immutable after creation).
    items: Box<[Value]>,
}

impl TupleObject {
    /// Create an empty tuple.
    #[inline]
    pub fn empty() -> Self {
        Self {
            header: ObjectHeader::new(TypeId::TUPLE),
            items: Box::new([]),
        }
    }

    /// Create a tuple from a slice.
    #[inline]
    pub fn from_slice(items: &[Value]) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::TUPLE),
            items: items.into(),
        }
    }

    /// Create a tuple from a Vec.
    #[inline]
    pub fn from_vec(items: Vec<Value>) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::TUPLE),
            items: items.into_boxed_slice(),
        }
    }

    /// Create a tuple from an iterator.
    #[inline]
    pub fn from_iter<I: IntoIterator<Item = Value>>(iter: I) -> Self {
        Self::from_vec(iter.into_iter().collect())
    }

    /// Get the length.
    #[inline]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if empty.
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

    /// Get an item without bounds checking.
    ///
    /// # Safety
    /// Index must be in bounds.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> Value {
        debug_assert!(index < self.items.len());
        unsafe { *self.items.get_unchecked(index) }
    }

    /// Get the underlying slice.
    #[inline]
    pub fn as_slice(&self) -> &[Value] {
        &self.items
    }

    /// Iterate over items.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Value> {
        self.items.iter()
    }

    /// Check if tuple contains a value.
    pub fn contains(&self, value: Value) -> bool {
        self.items.iter().any(|v| {
            if let (Some(a), Some(b)) = (v.as_int(), value.as_int()) {
                return a == b;
            }
            if let (Some(a), Some(b)) = (v.as_float(), value.as_float()) {
                return a == b;
            }
            false
        })
    }

    /// Create a new tuple by concatenating with another.
    pub fn concat(&self, other: &TupleObject) -> TupleObject {
        let mut items = Vec::with_capacity(self.len() + other.len());
        items.extend_from_slice(&self.items);
        items.extend_from_slice(&other.items);
        TupleObject::from_vec(items)
    }

    /// Create a new tuple by repeating n times.
    pub fn repeat(&self, n: usize) -> Option<TupleObject> {
        if n == 0 || self.is_empty() {
            return Some(TupleObject::empty());
        }

        let total_len = self.len().checked_mul(n)?;
        let mut items = Vec::new();
        items.try_reserve_exact(total_len).ok()?;
        items.extend_from_slice(&self.items);

        while items.len() < total_len {
            let remaining = total_len - items.len();
            let copy_len = items.len().min(remaining);
            items.extend_from_within(..copy_len);
        }

        Some(TupleObject::from_vec(items))
    }

    /// Get a slice as a new tuple.
    pub fn slice(&self, start: Option<i64>, end: Option<i64>) -> TupleObject {
        let len = self.len() as i64;
        let start = start.map(|s| self.clamp_index(s)).unwrap_or(0);
        let end = end.map(|e| self.clamp_index(e)).unwrap_or(len as usize);

        if start >= end {
            return TupleObject::empty();
        }

        TupleObject::from_slice(&self.items[start..end])
    }

    // Internal: normalize index for negative indexing
    fn normalize_index(&self, index: i64) -> Option<usize> {
        let len = self.len() as i64;
        let normalized = if index < 0 {
            len.checked_add(index)?
        } else {
            index
        };
        if normalized >= 0 && normalized < len {
            Some(normalized as usize)
        } else {
            None
        }
    }

    // Internal: clamp index to valid range
    fn clamp_index(&self, index: i64) -> usize {
        let len = self.len() as i64;
        let normalized = if index < 0 {
            len.saturating_add(index)
        } else {
            index
        };
        normalized.clamp(0, len) as usize
    }
}

/// Borrow native tuple storage from exact tuples and heap/object values that
/// carry tuple-compatible native backing.
#[inline(always)]
pub fn value_as_tuple_ref(value: Value) -> Option<&'static TupleObject> {
    let ptr = value.as_object_ptr()?;
    object_ptr_as_tuple_ref(ptr)
}

#[inline(always)]
fn is_shaped_heap_type(type_id: TypeId) -> bool {
    type_id.raw() >= TypeId::FIRST_USER_TYPE
        && !crate::types::iter::is_native_iterator_type_id(type_id)
}

/// Borrow native tuple storage from an object pointer.
#[inline(always)]
pub fn object_ptr_as_tuple_ref(ptr: *const ()) -> Option<&'static TupleObject> {
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    match header.type_id {
        TypeId::TUPLE => Some(unsafe { &*(ptr as *const TupleObject) }),
        TypeId::OBJECT => unsafe { (&*(ptr as *const ShapedObject)).tuple_backing() },
        type_id if is_shaped_heap_type(type_id) => unsafe {
            (&*(ptr as *const ShapedObject)).tuple_backing()
        },
        _ => None,
    }
}

impl PyObject for TupleObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

// =============================================================================
// Small Tuple Optimization
// =============================================================================

/// Empty tuple singleton for zero-allocation.
static EMPTY_TUPLE_DATA: [Value; 0] = [];

/// Get a reference to the empty tuple.
///
/// This avoids allocating for common `()` usage.
pub fn empty_tuple() -> TupleObject {
    TupleObject::empty()
}
