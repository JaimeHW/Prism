//! VM-aware dictionary access helpers.
//!
//! `DictObject` keeps the raw storage compact and fast. This module adds the
//! Python protocol layer needed by public dict operations: user-defined
//! `__hash__`, collision-time `__eq__`, and exception propagation.

use crate::VirtualMachine;
use crate::builtins::hash_value_vm;
use crate::error::RuntimeError;
use crate::ops::comparison::eq_result;
use prism_core::Value;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::int::value_to_bigint;
use prism_runtime::types::string::value_as_string_ref;
use prism_runtime::types::tuple::TupleObject;

#[inline]
pub(crate) fn dict_get_item(
    vm: &mut VirtualMachine,
    dict: &DictObject,
    key: Value,
) -> Result<Option<Value>, RuntimeError> {
    let key_hash = hash_value_vm(vm, key).map_err(RuntimeError::from)?;
    if let Some(value) = dict.get(key) {
        return Ok(Some(value));
    }
    if !requires_protocol_scan(key) {
        return Ok(None);
    }

    for (candidate, value) in dict.iter() {
        if dict_candidate_matches(vm, dict, candidate, key, key_hash)? {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

#[inline]
pub(crate) fn dict_contains_key(
    vm: &mut VirtualMachine,
    dict: &DictObject,
    key: Value,
) -> Result<bool, RuntimeError> {
    dict_get_item(vm, dict, key).map(|value| value.is_some())
}

#[inline]
pub(crate) fn dict_set_item(
    vm: &mut VirtualMachine,
    dict: &mut DictObject,
    key: Value,
    value: Value,
) -> Result<(), RuntimeError> {
    let key_hash = hash_value_vm(vm, key).map_err(RuntimeError::from)?;
    if dict.contains_key(key) {
        dict.set_with_hash(key, value, key_hash);
        return Ok(());
    }
    if !requires_protocol_scan(key) {
        dict.set_with_hash(key, value, key_hash);
        return Ok(());
    }

    let keys = dict.keys().collect::<Vec<_>>();
    for candidate in keys {
        if dict_candidate_matches(vm, dict, candidate, key, key_hash)? {
            dict.set_with_hash(candidate, value, key_hash);
            return Ok(());
        }
    }

    dict.set_with_hash(key, value, key_hash);
    Ok(())
}

#[inline]
pub(crate) fn dict_remove_item(
    vm: &mut VirtualMachine,
    dict: &mut DictObject,
    key: Value,
) -> Result<Option<Value>, RuntimeError> {
    let key_hash = hash_value_vm(vm, key).map_err(RuntimeError::from)?;
    if let Some(value) = dict.remove(key) {
        return Ok(Some(value));
    }
    if !requires_protocol_scan(key) {
        return Ok(None);
    }

    let keys = dict.keys().collect::<Vec<_>>();
    for candidate in keys {
        if dict_candidate_matches(vm, dict, candidate, key, key_hash)? {
            return Ok(dict.remove(candidate));
        }
    }

    Ok(None)
}

#[inline]
pub(crate) fn dict_setdefault(
    vm: &mut VirtualMachine,
    dict: &mut DictObject,
    key: Value,
    default: Value,
) -> Result<Value, RuntimeError> {
    let key_hash = hash_value_vm(vm, key).map_err(RuntimeError::from)?;
    if let Some(value) = dict.get(key) {
        return Ok(value);
    }
    if !requires_protocol_scan(key) {
        dict.set_with_hash(key, default, key_hash);
        return Ok(default);
    }

    let keys = dict.keys().collect::<Vec<_>>();
    for candidate in keys {
        if dict_candidate_matches(vm, dict, candidate, key, key_hash)? {
            return Ok(dict
                .get(candidate)
                .expect("candidate key came from this dictionary"));
        }
    }

    dict.set_with_hash(key, default, key_hash);
    Ok(default)
}

#[inline]
fn dict_candidate_matches(
    vm: &mut VirtualMachine,
    dict: &DictObject,
    candidate: Value,
    key: Value,
    key_hash: i64,
) -> Result<bool, RuntimeError> {
    let candidate_hash = match dict.stored_hash(candidate) {
        Some(hash) => hash,
        None => hash_value_vm(vm, candidate).map_err(RuntimeError::from)?,
    };
    if candidate_hash != key_hash {
        return Ok(false);
    }
    eq_result(vm, candidate, key)
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
    let type_id = crate::ops::objects::extract_type_id(ptr);
    match type_id {
        TypeId::TUPLE => {
            let tuple = unsafe { &*(ptr as *const TupleObject) };
            tuple.iter().copied().any(requires_protocol_scan)
        }
        _ => true,
    }
}
