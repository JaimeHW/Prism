//! Shared hashing/equality semantics for hash-based Python containers.
//!
//! Dict and set keys must compare by Python value semantics rather than raw
//! pointer identity. In particular, equivalent interned strings and heap string
//! objects must match, and tuples must hash structurally.

use crate::object::ObjectHeader;
use crate::object::descriptor::BoundMethod;
use crate::object::type_obj::TypeId;
use crate::types::int::value_to_bigint;
use crate::types::set::SetObject;
use crate::types::string::value_as_string_ref;
use crate::types::tuple::TupleObject;
use num_traits::ToPrimitive;
use prism_core::Value;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Hashable wrapper around `Value` for dict/set keys.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HashableValue(pub Value);

impl PartialEq for HashableValue {
    fn eq(&self, other: &Self) -> bool {
        hashable_value_eq(self.0, other.0)
    }
}

impl Eq for HashableValue {}

impl Hash for HashableValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_hashable_value(self.0, state);
    }
}

#[inline]
fn hashable_value_eq(left: Value, right: Value) -> bool {
    if left.raw_bits() == right.raw_bits() {
        return true;
    }

    if numeric_eq(left, right) {
        return true;
    }

    if let Some(equal) = string_eq(left, right) {
        return equal;
    }

    if let (Some(a), Some(b)) = (tuple_value(left), tuple_value(right)) {
        return tuple_eq(a, b);
    }

    if let (Some(a), Some(b)) = (set_value(left), set_value(right)) {
        return set_eq(a, b);
    }

    if let (Some(a), Some(b)) = (bound_method_value(left), bound_method_value(right)) {
        return hashable_value_eq(a.function(), b.function())
            && hashable_value_eq(a.instance(), b.instance());
    }

    matches!(
        (left.as_object_ptr(), right.as_object_ptr()),
        (Some(a), Some(b)) if a == b
    )
}

#[inline]
fn numeric_eq(left: Value, right: Value) -> bool {
    if let (Some(a), Some(b)) = (value_to_bigint(left), value_to_bigint(right)) {
        return a == b;
    }

    if let (Some(a), Some(b)) = (numeric_as_f64(left), numeric_as_f64(right)) {
        return a == b;
    }
    false
}

#[inline]
fn numeric_as_f64(value: Value) -> Option<f64> {
    if let Some(b) = value.as_bool() {
        return Some(if b { 1.0 } else { 0.0 });
    }
    if let Some(i) = value.as_int() {
        return Some(i as f64);
    }
    if let Some(i) = value_to_bigint(value) {
        return i.to_f64();
    }
    value.as_float()
}

#[inline]
fn hash_hashable_value<H: Hasher>(value: Value, state: &mut H) {
    if let Some(b) = value.as_bool() {
        (if b { 1i64 } else { 0i64 }).hash(state);
        return;
    }

    if let Some(i) = value.as_int() {
        i.hash(state);
        return;
    }

    if let Some(i) = value_to_bigint(value) {
        if let Some(small) = i.to_i64() {
            small.hash(state);
        } else {
            0x69u8.hash(state);
            i.hash(state);
        }
        return;
    }

    if let Some(f) = value.as_float() {
        if f.fract() == 0.0 && f.is_finite() && f >= (i64::MIN as f64) && f <= (i64::MAX as f64) {
            (f as i64).hash(state);
        } else {
            f.to_bits().hash(state);
        }
        return;
    }

    if value.is_none() {
        0u8.hash(state);
        return;
    }

    if hash_string_value(value, state) {
        return;
    }

    if let Some(tuple) = tuple_value(value) {
        0x54u8.hash(state);
        tuple.len().hash(state);
        for item in tuple.iter().copied() {
            hash_hashable_value(item, state);
        }
        return;
    }

    if let Some(set) = set_value(value) {
        hash_set_value(set, state);
        return;
    }

    if let Some(bound) = bound_method_value(value) {
        0x6du8.hash(state);
        hash_hashable_value(bound.function(), state);
        hash_hashable_value(bound.instance(), state);
        return;
    }

    if let Some(ptr) = value.as_object_ptr() {
        (ptr as usize).hash(state);
    } else {
        value.raw_bits().hash(state);
    }
}

#[inline]
fn string_eq(left: Value, right: Value) -> Option<bool> {
    let left = value_as_string_ref(left)?;
    let right = value_as_string_ref(right)?;
    Some(left.as_str() == right.as_str())
}

#[inline]
fn hash_string_value<H: Hasher>(value: Value, state: &mut H) -> bool {
    let Some(string) = value_as_string_ref(value) else {
        return false;
    };
    string.as_str().hash(state);
    true
}

#[inline]
fn tuple_value(value: Value) -> Option<&'static TupleObject> {
    let ptr = value.as_object_ptr()?;
    if type_id_of(ptr) != TypeId::TUPLE {
        return None;
    }

    Some(unsafe { &*(ptr as *const TupleObject) })
}

#[inline]
fn set_value(value: Value) -> Option<&'static SetObject> {
    let ptr = value.as_object_ptr()?;
    match type_id_of(ptr) {
        TypeId::SET | TypeId::FROZENSET => Some(unsafe { &*(ptr as *const SetObject) }),
        _ => None,
    }
}

#[inline]
fn bound_method_value(value: Value) -> Option<&'static BoundMethod> {
    let ptr = value.as_object_ptr()?;
    if type_id_of(ptr) != TypeId::METHOD {
        return None;
    }

    Some(unsafe { &*(ptr as *const BoundMethod) })
}

#[inline]
fn tuple_eq(left: &TupleObject, right: &TupleObject) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .copied()
            .zip(right.iter().copied())
            .all(|(a, b)| hashable_value_eq(a, b))
}

#[inline]
fn set_eq(left: &SetObject, right: &SetObject) -> bool {
    left.len() == right.len() && left.iter().all(|item| right.contains(item))
}

#[inline]
fn hash_set_value<H: Hasher>(set: &SetObject, state: &mut H) {
    0x53u8.hash(state);
    set.len().hash(state);

    let mut xor = 0u64;
    let mut sum = 0u64;
    for item in set.iter() {
        let hash = hash_value_to_u64(item);
        xor ^= hash;
        sum = sum.wrapping_add(hash);
    }
    xor.hash(state);
    sum.hash(state);
}

#[inline]
fn hash_value_to_u64(value: Value) -> u64 {
    let mut hasher = DefaultHasher::new();
    hash_hashable_value(value, &mut hasher);
    hasher.finish()
}

#[inline(always)]
fn type_id_of(ptr: *const ()) -> TypeId {
    let header = ptr as *const ObjectHeader;
    unsafe { (*header).type_id }
}
