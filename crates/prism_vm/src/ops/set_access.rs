//! VM-aware set access helpers.
//!
//! `SetObject` keeps the raw storage compact and fast. This module adds the
//! Python protocol layer needed by public set operations: user-defined
//! `__hash__`, collision-time `__eq__`, exception propagation, and CPython's
//! temporary-frozenset probe behavior for unhashable set instances.

use crate::VirtualMachine;
use crate::builtins::{BuiltinError, hash_set_contents_vm, hash_value_vm};
use crate::error::RuntimeError;
use crate::ops::comparison::eq_or_identical;
use crate::ops::objects::{extract_type_id, set_storage_ref_from_ptr};
use prism_core::Value;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::int::value_to_bigint;
use prism_runtime::types::set::SetObject;
use prism_runtime::types::string::value_as_string_ref;

#[inline]
pub(crate) fn set_contains_item(
    vm: &mut VirtualMachine,
    set: &SetObject,
    key: Value,
) -> Result<bool, RuntimeError> {
    match hash_value_vm(vm, key) {
        Ok(key_hash) => set_contains_hashed(vm, set, key, key_hash),
        Err(err) if should_probe_as_frozenset(&err, key) => {
            set_contains_unhashable_set(vm, set, key)
        }
        Err(err) => Err(RuntimeError::from(err)),
    }
}

#[inline]
pub(crate) fn set_add_item(
    vm: &mut VirtualMachine,
    set: &mut SetObject,
    key: Value,
) -> Result<bool, RuntimeError> {
    let key_hash = hash_value_vm(vm, key).map_err(RuntimeError::from)?;
    let key_requires_protocol_lookup = requires_protocol_scan(key);
    if !key_requires_protocol_lookup && !set.has_protocol_keys() {
        return Ok(set.add_with_hash_and_protocol_lookup(key, key_hash, false));
    }

    let candidates = set.iter().collect::<Vec<_>>();
    for candidate in candidates {
        if set_candidate_matches(vm, set, candidate, key, key_hash)? {
            return Ok(false);
        }
    }

    Ok(set.add_with_hash_and_protocol_lookup(key, key_hash, key_requires_protocol_lookup))
}

#[inline]
pub(crate) fn set_remove_item(
    vm: &mut VirtualMachine,
    set: &mut SetObject,
    key: Value,
) -> Result<bool, RuntimeError> {
    match hash_value_vm(vm, key) {
        Ok(key_hash) => set_remove_hashed(vm, set, key, key_hash),
        Err(err) if should_probe_as_frozenset(&err, key) => set_remove_unhashable_set(vm, set, key),
        Err(err) => Err(RuntimeError::from(err)),
    }
}

#[inline]
pub(crate) fn set_discard_item(
    vm: &mut VirtualMachine,
    set: &mut SetObject,
    key: Value,
) -> Result<(), RuntimeError> {
    set_remove_item(vm, set, key).map(|_| ())
}

#[inline]
pub(crate) fn set_from_values(
    vm: &mut VirtualMachine,
    values: impl IntoIterator<Item = Value>,
) -> Result<SetObject, RuntimeError> {
    let values = values.into_iter();
    let (capacity, _) = values.size_hint();
    let mut set = SetObject::with_capacity(capacity);
    for value in values {
        set_add_item(vm, &mut set, value)?;
    }
    Ok(set)
}

#[inline]
fn set_contains_hashed(
    vm: &mut VirtualMachine,
    set: &SetObject,
    key: Value,
    key_hash: i64,
) -> Result<bool, RuntimeError> {
    if !requires_protocol_scan(key) && !set.has_protocol_keys() {
        return Ok(set.contains(key));
    }

    for candidate in set.iter().collect::<Vec<_>>() {
        if set_candidate_matches(vm, set, candidate, key, key_hash)? {
            return Ok(true);
        }
    }
    Ok(false)
}

#[inline]
fn set_remove_hashed(
    vm: &mut VirtualMachine,
    set: &mut SetObject,
    key: Value,
    key_hash: i64,
) -> Result<bool, RuntimeError> {
    if !requires_protocol_scan(key) && !set.has_protocol_keys() {
        return Ok(set.remove(key));
    }

    let candidates = set.iter().collect::<Vec<_>>();
    for candidate in candidates {
        if set_candidate_matches(vm, set, candidate, key, key_hash)? {
            return Ok(set.remove(candidate));
        }
    }
    Ok(false)
}

#[inline]
fn set_candidate_matches(
    vm: &mut VirtualMachine,
    set: &SetObject,
    candidate: Value,
    key: Value,
    key_hash: i64,
) -> Result<bool, RuntimeError> {
    let candidate_hash = match set.stored_hash(candidate) {
        Some(hash) => hash,
        None => hash_value_vm(vm, candidate).map_err(RuntimeError::from)?,
    };
    if candidate_hash != key_hash {
        return Ok(false);
    }
    eq_or_identical(vm, candidate, key)
}

#[inline]
fn set_contains_unhashable_set(
    vm: &mut VirtualMachine,
    set: &SetObject,
    key: Value,
) -> Result<bool, RuntimeError> {
    let probe_hash = set_probe_hash(vm, key)?;
    for candidate in set.iter().collect::<Vec<_>>() {
        if set_content_candidate_matches(vm, set, candidate, key, probe_hash)? {
            return Ok(true);
        }
    }
    Ok(false)
}

#[inline]
fn set_remove_unhashable_set(
    vm: &mut VirtualMachine,
    set: &mut SetObject,
    key: Value,
) -> Result<bool, RuntimeError> {
    let probe_hash = set_probe_hash(vm, key)?;
    let candidates = set.iter().collect::<Vec<_>>();
    for candidate in candidates {
        if set_content_candidate_matches(vm, set, candidate, key, probe_hash)? {
            return Ok(set.remove(candidate));
        }
    }
    Ok(false)
}

#[inline]
fn set_content_candidate_matches(
    vm: &mut VirtualMachine,
    set: &SetObject,
    candidate: Value,
    key: Value,
    key_hash: i64,
) -> Result<bool, RuntimeError> {
    let candidate_hash = match set.stored_hash(candidate) {
        Some(hash) => hash,
        None => hash_value_vm(vm, candidate).map_err(RuntimeError::from)?,
    };
    if candidate_hash != key_hash {
        return Ok(false);
    }
    eq_or_identical(vm, candidate, key)
}

#[inline]
fn set_probe_hash(vm: &mut VirtualMachine, key: Value) -> Result<i64, RuntimeError> {
    let Some(ptr) = key.as_object_ptr() else {
        return Err(RuntimeError::type_error("unhashable type"));
    };
    let Some(set) = set_storage_ref_from_ptr(ptr) else {
        return Err(RuntimeError::type_error("unhashable type"));
    };
    hash_set_contents_vm(vm, set).map_err(RuntimeError::from)
}

#[inline]
fn should_probe_as_frozenset(err: &BuiltinError, key: Value) -> bool {
    matches!(err, BuiltinError::TypeError(_)) && is_set_like_probe(key)
}

#[inline]
fn is_set_like_probe(value: Value) -> bool {
    value
        .as_object_ptr()
        .and_then(set_storage_ref_from_ptr)
        .is_some()
}

#[inline]
fn requires_protocol_scan(value: Value) -> bool {
    if value.as_bool().is_some()
        || value.as_int().is_some()
        || value.as_float().is_some()
        || value_to_bigint(value).is_some()
        || value.is_none()
        || value_as_string_ref(value).is_some()
    {
        return false;
    }

    let Some(ptr) = value.as_object_ptr() else {
        return true;
    };
    let type_id = extract_type_id(ptr);
    match type_id {
        TypeId::TUPLE => true,
        _ => true,
    }
}
