//! Arbitrary-precision integer support.
//!
//! Prism keeps common small integers inline inside [`Value`] for the hot path,
//! and promotes larger results to heap-backed `int` objects carrying a
//! [`num_bigint::BigInt`]. This mirrors CPython's single visible `int` type
//! while preserving the fast small-int representation.

use crate::allocation_context::alloc_value_in_current_heap_or_box;
use crate::object::shaped_object::ShapedObject;
use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use num_bigint::{BigInt, Sign};
use num_traits::ToPrimitive;
use prism_core::Value;

/// Heap-backed arbitrary-precision Python integer.
#[repr(C)]
pub struct IntObject {
    /// Object header.
    pub header: ObjectHeader,
    value: BigInt,
}

impl IntObject {
    /// Create a new heap-backed integer object.
    #[inline]
    pub fn new(value: BigInt) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::INT),
            value,
        }
    }

    /// Borrow the underlying bigint value.
    #[inline]
    pub fn value(&self) -> &BigInt {
        &self.value
    }
}

impl PyObject for IntObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

/// Return the heap-backed integer object if `value` is a promoted `int`.
#[inline]
pub fn value_as_heap_int(value: Value) -> Option<&'static IntObject> {
    let ptr = value.as_object_ptr()?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != TypeId::INT {
        return None;
    }
    Some(unsafe { &*(ptr as *const IntObject) })
}

/// Borrow the native integer payload for exact ints and heap `int` subclasses.
#[inline]
pub fn value_as_int_ref(value: Value) -> Option<&'static BigInt> {
    let ptr = value.as_object_ptr()?;
    object_ptr_as_int_ref(ptr)
}

/// Borrow the native integer payload for an object pointer.
#[inline]
pub fn object_ptr_as_int_ref(ptr: *const ()) -> Option<&'static BigInt> {
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    match header.type_id {
        TypeId::INT => Some(unsafe { &*(ptr as *const IntObject) }.value()),
        type_id
            if type_id.raw() >= TypeId::FIRST_USER_TYPE
                && !crate::types::iter::is_native_iterator_type_id(type_id) =>
        unsafe { (&*(ptr as *const ShapedObject)).int_backing() },
        _ => None,
    }
}

/// Returns true when `value` is any Python integer, inline or heap-backed.
#[inline]
pub fn is_int_value(value: Value) -> bool {
    value.as_int().is_some() || value_as_int_ref(value).is_some()
}

/// Convert a Python integer value into a bigint.
#[inline]
pub fn value_to_bigint(value: Value) -> Option<BigInt> {
    if let Some(i) = value.as_int() {
        return Some(BigInt::from(i));
    }
    value_as_int_ref(value).cloned()
}

/// Convert a Python integer value into `i64` when it fits.
#[inline]
pub fn value_to_i64(value: Value) -> Option<i64> {
    if let Some(i) = value.as_int() {
        return Some(i);
    }
    value_as_int_ref(value).and_then(ToPrimitive::to_i64)
}

/// Convert a bigint to `i64`, clamping out-of-range values.
#[inline]
pub fn bigint_to_saturated_i64(value: &BigInt) -> i64 {
    value.to_i64().unwrap_or_else(|| match value.sign() {
        Sign::Minus => i64::MIN,
        _ => i64::MAX,
    })
}

/// Convert a Python integer value into `i64`, clamping heap-backed bigints.
#[inline]
pub fn value_to_saturated_i64(value: Value) -> Option<i64> {
    if let Some(i) = value.as_int() {
        return Some(i);
    }
    value_as_int_ref(value).map(bigint_to_saturated_i64)
}

/// Format a Python integer value using Python `int.__repr__` semantics.
#[inline]
pub fn int_value_to_string(value: Value) -> Option<String> {
    if let Some(i) = value.as_int() {
        return Some(i.to_string());
    }
    value_as_int_ref(value).map(ToString::to_string)
}

/// Convert a bigint into the most efficient Prism value representation.
///
/// Values that still fit Prism's inline small-int encoding stay unboxed.
/// Larger values are promoted to a heap-backed `IntObject`.
#[inline]
pub fn bigint_to_value(value: BigInt) -> Value {
    if let Some(i) = value.to_i64() {
        if let Some(inline) = Value::int(i) {
            return inline;
        }
    }

    alloc_value_in_current_heap_or_box(IntObject::new(value))
}
