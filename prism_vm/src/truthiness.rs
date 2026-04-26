//! Python truthiness helpers for VM/runtime values.
//!
//! `prism_core::Value::is_truthy()` intentionally stays lightweight and
//! allocation-free, but the VM needs container-aware semantics for stdlib
//! compatibility. This module layers those richer checks on top of raw values
//! without changing the core tagged-value crate dependency graph.

use crate::VirtualMachine;
use crate::error::{RuntimeError, RuntimeErrorKind};
use crate::ops::calls::invoke_callable_value;
use crate::ops::method_dispatch::load_method::{BoundMethodTarget, resolve_special_method};
use crate::stdlib::collections::deque::DequeObject;
use prism_core::Value;
use prism_core::intern::interned_by_ptr;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::complex::ComplexObject;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::memoryview::MemoryViewObject;
use prism_runtime::types::range::RangeObject;
use prism_runtime::types::set::SetObject;
use prism_runtime::types::string::StringObject;
use prism_runtime::types::tuple::TupleObject;

#[inline(always)]
fn extract_type_id(ptr: *const ()) -> TypeId {
    let header = ptr as *const ObjectHeader;
    unsafe { (*header).type_id }
}

/// Evaluate Python truthiness with container-aware semantics.
#[inline]
pub fn is_truthy(value: Value) -> bool {
    if let Some(result) = exact_truthiness(value) {
        return result;
    }

    true
}

/// Evaluate truthiness using Python's `__bool__` / `__len__` protocol when
/// the fast exact-type path does not apply.
pub fn try_is_truthy(vm: &mut VirtualMachine, value: Value) -> Result<bool, RuntimeError> {
    if let Some(result) = exact_truthiness(value) {
        return Ok(result);
    }

    if let Some(result) = try_bool_protocol(vm, value)? {
        return Ok(result);
    }

    match crate::builtins::try_len_value(vm, value) {
        Ok(len) => Ok(len != 0),
        Err(err) if is_missing_len_type_error(&err) => Ok(true),
        Err(err) => Err(err),
    }
}

#[inline]
fn exact_truthiness(value: Value) -> Option<bool> {
    if value.is_none() {
        return Some(false);
    }
    if let Some(flag) = value.as_bool() {
        return Some(flag);
    }
    if let Some(int_value) = value.as_int() {
        return Some(int_value != 0);
    }
    if let Some(float_value) = value.as_float() {
        return Some(float_value != 0.0);
    }
    if value.is_string() {
        let Some(ptr) = value.as_string_object_ptr() else {
            return Some(true);
        };
        return Some(interned_by_ptr(ptr as *const u8).is_some_and(|s| !s.as_str().is_empty()));
    }

    let Some(ptr) = value.as_object_ptr() else {
        return Some(true);
    };

    if let Some(list) = crate::ops::objects::list_storage_ref_from_ptr(ptr) {
        return Some(!list.is_empty());
    }

    Some(match extract_type_id(ptr) {
        TypeId::STR => !unsafe { &*(ptr as *const StringObject) }.is_empty(),
        TypeId::TUPLE => !unsafe { &*(ptr as *const TupleObject) }.is_empty(),
        TypeId::DICT => !unsafe { &*(ptr as *const DictObject) }.is_empty(),
        TypeId::SET | TypeId::FROZENSET => !unsafe { &*(ptr as *const SetObject) }.is_empty(),
        TypeId::BYTES | TypeId::BYTEARRAY => !unsafe { &*(ptr as *const BytesObject) }.is_empty(),
        TypeId::MEMORYVIEW => !unsafe { &*(ptr as *const MemoryViewObject) }.is_empty(),
        TypeId::DEQUE => !unsafe { &*(ptr as *const DequeObject) }.is_empty(),
        TypeId::RANGE => !unsafe { &*(ptr as *const RangeObject) }.is_empty(),
        TypeId::COMPLEX => !unsafe { &*(ptr as *const ComplexObject) }.is_zero(),
        _ => return None,
    })
}

#[inline]
fn try_bool_protocol(vm: &mut VirtualMachine, value: Value) -> Result<Option<bool>, RuntimeError> {
    let target = match resolve_special_method(value, "__bool__") {
        Ok(target) => target,
        Err(err) if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) => return Ok(None),
        Err(err) => return Err(err),
    };

    let result = invoke_zero_arg_bound_method(vm, target)?;
    let Some(flag) = result.as_bool() else {
        return Err(RuntimeError::type_error(format!(
            "__bool__ should return bool, returned {}",
            result.type_name()
        )));
    };
    Ok(Some(flag))
}

#[inline]
fn invoke_zero_arg_bound_method(
    vm: &mut VirtualMachine,
    target: BoundMethodTarget,
) -> Result<Value, RuntimeError> {
    match target.implicit_self {
        Some(implicit_self) => invoke_callable_value(vm, target.callable, &[implicit_self]),
        None => invoke_callable_value(vm, target.callable, &[]),
    }
}

#[inline]
fn is_missing_len_type_error(err: &RuntimeError) -> bool {
    matches!(
        &err.kind,
        RuntimeErrorKind::TypeError { message }
            if message.starts_with("object of type '") && message.ends_with("' has no len()")
    )
}

#[cfg(test)]
mod tests;
