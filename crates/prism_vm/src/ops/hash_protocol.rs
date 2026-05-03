//! Hash-container protocol classification.
//!
//! Dict and set fast paths can use raw storage only when hashing and equality
//! cannot execute Python code. This module keeps that classification shared
//! between VM handlers and Tier 1 helpers so optimized paths stay conservative
//! without needlessly deopting primitive and recursively-fast tuple keys.

use crate::ops::objects::extract_type_id;
use prism_core::Value;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::int::value_to_bigint;
use prism_runtime::types::tuple::TupleObject;

#[inline]
pub(crate) fn value_requires_hash_protocol(value: Value) -> bool {
    !is_fast_hash_lookup_key(value)
}

#[inline]
pub(crate) fn is_fast_hash_lookup_key(value: Value) -> bool {
    if value.as_bool().is_some()
        || value.as_int().is_some()
        || value.as_float().is_some()
        || value_to_bigint(value).is_some()
        || value.is_none()
        || value.is_string()
    {
        return true;
    }

    let Some(ptr) = value.as_object_ptr() else {
        return false;
    };

    match extract_type_id(ptr) {
        TypeId::STR => true,
        TypeId::TUPLE => {
            let tuple = unsafe { &*(ptr as *const TupleObject) };
            tuple.iter().copied().all(is_fast_hash_lookup_key)
        }
        _ => false,
    }
}
