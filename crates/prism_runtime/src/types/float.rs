//! Python float support.
//!
//! Exact floats stay in Prism's tagged `Value` representation. Heap subclasses
//! carry their native payload in `ShapedObject` so numeric fast paths can still
//! extract the IEEE-754 value directly.

use crate::object::ObjectHeader;
use crate::object::shaped_object::ShapedObject;
use crate::object::type_obj::TypeId;
use prism_core::Value;

/// Convert a Python float value, including heap subclasses, to `f64`.
#[inline]
pub fn value_to_f64(value: Value) -> Option<f64> {
    if let Some(float) = value.as_float() {
        return Some(float);
    }

    value.as_object_ptr().and_then(object_ptr_as_float)
}

/// Read the native float payload for a heap object pointer.
#[inline]
pub fn object_ptr_as_float(ptr: *const ()) -> Option<f64> {
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    match header.type_id {
        type_id
            if type_id.raw() >= TypeId::FIRST_USER_TYPE
                && !crate::types::iter::is_native_iterator_type_id(type_id) =>
        unsafe { (&*(ptr as *const ShapedObject)).float_backing() },
        _ => None,
    }
}
