//! VM-aware dictionary access helpers.
//!
//! `DictObject` keeps the raw storage compact and fast. This module adds the
//! Python protocol layer needed by public dict operations: user-defined
//! `__hash__`, collision-time `__eq__`, and exception propagation.

use crate::VirtualMachine;
use crate::builtins::create_exception_with_args_in_vm;
use crate::builtins::hash_value_vm;
use crate::error::RuntimeError;
use crate::ops::calls::invoke_callable_value;
use crate::ops::comparison::eq_result;
use crate::ops::objects::{bind_user_class_attribute_value_in_vm, extract_type_id};
use crate::stdlib::exceptions::ExceptionTypeId;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::type_builtins::global_class;
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
    if !requires_protocol_scan(key) && !dict.has_protocol_keys() {
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
    let key_requires_protocol_lookup = requires_protocol_scan(key);
    if dict.contains_key(key) {
        dict.set_with_hash_and_protocol_lookup(key, value, key_hash, key_requires_protocol_lookup);
        return Ok(());
    }
    if !key_requires_protocol_lookup && !dict.has_protocol_keys() {
        dict.set_with_hash_and_protocol_lookup(key, value, key_hash, false);
        return Ok(());
    }

    let keys = dict.keys().collect::<Vec<_>>();
    for candidate in keys {
        if dict_candidate_matches(vm, dict, candidate, key, key_hash)? {
            dict.set_with_hash_and_protocol_lookup(
                candidate,
                value,
                key_hash,
                requires_protocol_scan(candidate),
            );
            return Ok(());
        }
    }

    dict.set_with_hash_and_protocol_lookup(key, value, key_hash, key_requires_protocol_lookup);
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
    if !requires_protocol_scan(key) && !dict.has_protocol_keys() {
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
    let key_requires_protocol_lookup = requires_protocol_scan(key);
    if let Some(value) = dict.get(key) {
        return Ok(value);
    }
    if !key_requires_protocol_lookup && !dict.has_protocol_keys() {
        dict.set_with_hash_and_protocol_lookup(key, default, key_hash, false);
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

    dict.set_with_hash_and_protocol_lookup(key, default, key_hash, key_requires_protocol_lookup);
    Ok(default)
}

#[inline]
pub(crate) fn missing_key_error(vm: &VirtualMachine, key: Value) -> RuntimeError {
    match create_exception_with_args_in_vm(
        vm,
        ExceptionTypeId::KeyError,
        None,
        vec![key].into_boxed_slice(),
    ) {
        Ok(exception) => RuntimeError::raised_exception(
            ExceptionTypeId::KeyError.as_u8() as u16,
            exception,
            "key not found",
        ),
        Err(err) => err,
    }
}

#[inline]
pub(crate) fn dict_missing_value(
    vm: &mut VirtualMachine,
    receiver: Value,
    key: Value,
) -> Result<Option<Value>, RuntimeError> {
    let Some(receiver_ptr) = receiver.as_object_ptr() else {
        return Ok(None);
    };

    let receiver_type = extract_type_id(receiver_ptr);
    if receiver_type.raw() < TypeId::FIRST_USER_TYPE {
        return Ok(None);
    }

    let missing_name = intern("__missing__");
    let Some(class) = global_class(ClassId(receiver_type.raw())) else {
        return Ok(None);
    };
    let Some(slot) = class.lookup_method_published(&missing_name) else {
        return Ok(None);
    };

    let callable =
        bind_user_class_attribute_value_in_vm(vm, slot.value, slot.defining_class, receiver)?;
    invoke_callable_value(vm, callable, &[key]).map(Some)
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
    let type_id = extract_type_id(ptr);
    match type_id {
        TypeId::TUPLE => {
            let tuple = unsafe { &*(ptr as *const TupleObject) };
            tuple.iter().copied().any(requires_protocol_scan)
        }
        _ => true,
    }
}
