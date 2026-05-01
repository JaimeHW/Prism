//! Shared coroutine protocol helpers.
//!
//! Centralizes:
//! - object type/name classification for diagnostics
//! - magic method lookup for `__await__`, `__aiter__`, `__anext__`
//! - zero-allocation unary method invocation fast paths

use crate::VirtualMachine;
use crate::error::RuntimeError;
use crate::ops::calls::invoke_callable_value;
use crate::ops::method_dispatch::load_method::{BoundMethodTarget, resolve_special_method};
use prism_core::Value;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;

/// Extract TypeId from an object pointer.
#[inline(always)]
pub(crate) fn extract_type_id(ptr: *const ()) -> TypeId {
    let header = ptr as *const ObjectHeader;
    unsafe { (*header).type_id }
}

/// Returns a stable type name for diagnostics.
#[inline]
pub(crate) fn type_name(value: &Value) -> &'static str {
    if value.is_none() {
        "NoneType"
    } else if value.is_bool() {
        "bool"
    } else if value.is_int() {
        "int"
    } else if value.is_float() {
        "float"
    } else if let Some(ptr) = value.as_object_ptr() {
        extract_type_id(ptr).name()
    } else {
        "unknown"
    }
}

/// True when the value can be treated as an iterator object by runtime type.
#[inline(always)]
pub(crate) fn is_iterator(value: &Value) -> bool {
    let Some(ptr) = value.as_object_ptr() else {
        return false;
    };
    matches!(extract_type_id(ptr), TypeId::ITERATOR | TypeId::GENERATOR)
}

/// Look up a special method using Prism's normal type/MRO binding path.
#[inline]
pub(crate) fn lookup_magic_method(
    _vm: &VirtualMachine,
    obj: Value,
    method_name: &str,
) -> Result<Option<BoundMethodTarget>, RuntimeError> {
    match resolve_special_method(obj, method_name) {
        Ok(target) => Ok(Some(target)),
        Err(err) if err.is_attribute_error() => Ok(None),
        Err(err) => Err(err),
    }
}

/// Invoke an already-resolved unary special method.
#[inline]
pub(crate) fn call_unary_magic_method(
    vm: &mut VirtualMachine,
    target: BoundMethodTarget,
) -> Result<Value, RuntimeError> {
    if let Some(self_arg) = target.implicit_self {
        invoke_callable_value(vm, target.callable, &[self_arg])
    } else {
        invoke_callable_value(vm, target.callable, &[])
    }
}
