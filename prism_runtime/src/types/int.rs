//! Arbitrary-precision integer support.
//!
//! Prism keeps common small integers inline inside [`Value`] for the hot path,
//! and promotes larger results to heap-backed `int` objects carrying a
//! [`num_bigint::BigInt`]. This mirrors CPython's single visible `int` type
//! while preserving the fast small-int representation.

use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use num_bigint::BigInt;
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

/// Returns true when `value` is any Python integer, inline or heap-backed.
#[inline]
pub fn is_int_value(value: Value) -> bool {
    value.as_int().is_some() || value_as_heap_int(value).is_some()
}

/// Convert a Python integer value into a bigint.
#[inline]
pub fn value_to_bigint(value: Value) -> Option<BigInt> {
    if let Some(i) = value.as_int() {
        return Some(BigInt::from(i));
    }
    value_as_heap_int(value).map(|obj| obj.value().clone())
}

/// Convert a Python integer value into `i64` when it fits.
#[inline]
pub fn value_to_i64(value: Value) -> Option<i64> {
    if let Some(i) = value.as_int() {
        return Some(i);
    }
    value_as_heap_int(value).and_then(|obj| obj.value().to_i64())
}

/// Format a Python integer value using Python `int.__repr__` semantics.
#[inline]
pub fn int_value_to_string(value: Value) -> Option<String> {
    if let Some(i) = value.as_int() {
        return Some(i.to_string());
    }
    value_as_heap_int(value).map(|obj| obj.value().to_string())
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

    let ptr = Box::leak(Box::new(IntObject::new(value))) as *mut IntObject as *const ();
    Value::object_ptr(ptr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bigint_to_value_keeps_small_ints_inline() {
        let value = bigint_to_value(BigInt::from(42_i64));
        assert_eq!(value.as_int(), Some(42));
        assert!(value_as_heap_int(value).is_none());
    }

    #[test]
    fn test_bigint_to_value_promotes_large_values() {
        let big = BigInt::from(1_u8) << 100_u32;
        let value = bigint_to_value(big.clone());

        let obj = value_as_heap_int(value).expect("large integer should allocate");
        assert_eq!(obj.value(), &big);
        assert_eq!(value_to_bigint(value), Some(big));
    }

    #[test]
    fn test_int_value_to_string_formats_heap_backed_values() {
        let big = BigInt::from(1_u8) << 90_u32;
        let value = bigint_to_value(big.clone());
        assert_eq!(int_value_to_string(value), Some(big.to_string()));
    }
}
