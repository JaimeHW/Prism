//! Shared coroutine protocol helpers.
//!
//! Centralizes:
//! - object type/name classification for diagnostics
//! - magic method lookup for `__await__`, `__aiter__`, `__anext__`
//! - zero-allocation unary method invocation fast paths

use crate::VirtualMachine;
use crate::builtins::BuiltinFunctionObject;
use crate::error::RuntimeError;
use crate::ops::attribute::is_user_defined_type;
use prism_core::Value;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::descriptor::BoundMethod;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_obj::TypeId;

#[cfg(test)]
use std::sync::Arc;

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

/// Look up a magic method on object instance storage.
///
/// Current runtime support resolves instance-level attributes on shaped objects.
/// Type/MRO lookup can be layered on top once class registry dispatch is fully wired.
#[inline]
pub(crate) fn lookup_magic_method(
    _vm: &VirtualMachine,
    obj: Value,
    method_name: &str,
) -> Result<Option<Value>, RuntimeError> {
    let Some(ptr) = obj.as_object_ptr() else {
        return Ok(None);
    };

    let type_id = extract_type_id(ptr);
    if type_id == TypeId::OBJECT || is_user_defined_type(type_id) {
        let shaped = unsafe { &*(ptr as *const ShapedObject) };
        return Ok(shaped.get_property(method_name));
    }

    Ok(None)
}

/// Invoke an already-resolved unary magic method (`method(obj)`).
///
/// This path currently supports native builtin-function objects directly.
#[inline]
pub(crate) fn call_unary_magic_method(
    _vm: &mut VirtualMachine,
    method: Value,
    obj: Value,
    method_name: &'static str,
) -> Result<Value, RuntimeError> {
    let Some(ptr) = method.as_object_ptr() else {
        return Err(RuntimeError::type_error(format!(
            "'{}' attribute is not callable",
            method_name
        )));
    };

    match extract_type_id(ptr) {
        TypeId::BUILTIN_FUNCTION => {
            let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
            builtin
                .call(&[obj])
                .map_err(|e| RuntimeError::type_error(e.to_string()))
        }
        TypeId::METHOD => {
            let bound = unsafe { &*(ptr as *const BoundMethod) };
            let function = bound.function();
            let instance = bound.instance();

            let Some(function_ptr) = function.as_object_ptr() else {
                return Err(RuntimeError::type_error(format!(
                    "bound '{}' method has non-callable function",
                    method_name
                )));
            };

            match extract_type_id(function_ptr) {
                TypeId::BUILTIN_FUNCTION => {
                    let builtin = unsafe { &*(function_ptr as *const BuiltinFunctionObject) };
                    builtin
                        .call(&[instance])
                        .map_err(|e| RuntimeError::type_error(e.to_string()))
                }
                other => Err(RuntimeError::internal(format!(
                    "bound '{}' method target '{}' is not integrated yet",
                    method_name,
                    other.name()
                ))),
            }
        }
        other => Err(RuntimeError::type_error(format!(
            "'{}' attribute '{}' is not callable",
            other.name(),
            method_name
        ))),
    }
}

#[cfg(test)]
mod tests;
