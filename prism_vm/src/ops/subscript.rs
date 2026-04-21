//! Subscript opcode handlers for __getitem__/__setitem__/__delitem__.
//!
//! High-performance type-dispatched subscript operations with:
//! - O(1) TypeId dispatch for built-in types (List, Tuple, Dict, String)
//! - SliceObject support with full Python semantics
//! - Negative index normalization
//! - Proper error handling with IndexError/KeyError/TypeError

use crate::VirtualMachine;
use crate::builtins::{builtin_mapping_proxy_get_item, value_to_iterator};
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use crate::ops::calls::{
    InvokeCallableOutcome, invoke_callable_value, invoke_callable_value_with_control_transfer,
};
use crate::ops::method_dispatch::load_method::{BoundMethodTarget, resolve_special_method};
use crate::ops::objects::{
    dict_storage_mut_from_ptr, dict_storage_ref_from_ptr, extract_type_id,
    list_storage_mut_from_ptr, list_storage_ref_from_ptr,
};
use prism_code::Instruction;
use prism_core::Value;
use prism_core::intern::{InternedString, intern, interned_by_ptr};
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::type_builtins::global_class;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::object::views::{GenericAliasObject, MappingProxyObject};
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::slice::SliceObject;
use prism_runtime::types::string::StringObject;
use prism_runtime::types::tuple::TupleObject;

// =============================================================================
// Subscript Result Type
// =============================================================================

/// Result of a subscript operation that may require GC allocation.
///
/// This type separates the computation of the subscript result from heap
/// allocation. Direct value results (list/tuple/dict elements) can be returned
/// immediately, while objects that need allocation (string slices, new
/// containers) can be allocated at the call site with access to the VM's heap.
enum SubscriptResult {
    /// A Value that can be returned directly (no allocation needed).
    Value(Value),
    /// A bytes or bytearray object that needs GC allocation.
    AllocBytes(BytesObject),
    /// A StringObject that needs GC allocation.
    AllocString(StringObject),
    /// A ListObject that needs GC allocation (from slice operation).
    AllocList(ListObject),
    /// A TupleObject that needs GC allocation (from slice operation).
    AllocTuple(TupleObject),
}

#[inline]
fn can_use_dict_storage_fast_path(ptr: *const (), special_method: &str) -> bool {
    let type_id = extract_type_id(ptr);
    if type_id == TypeId::DICT {
        return true;
    }

    if type_id.raw() < TypeId::FIRST_USER_TYPE {
        return false;
    }

    let Some(class) = global_class(ClassId(type_id.raw())) else {
        return false;
    };

    // Keep the direct dict-storage fast path for exact dicts and for heap dict
    // subclasses that inherit the builtin behavior unchanged. If a heap class
    // defines the special method anywhere in its heap MRO, we must fall back to
    // Python dispatch so __prepare__ mappings and other dict subclasses observe
    // user-defined overrides correctly.
    class
        .lookup_method(&intern(special_method), global_class)
        .is_none()
}

// =============================================================================
// BinarySubscr: dst = container[key]
// =============================================================================

/// BinarySubscr: dst = container[key]
///
/// Type-dispatched subscript with speculative fast paths:
/// 1. Integer key → direct index access (most common)
/// 2. SliceObject key → slice operation
/// 3. String/any key for dict → hash lookup
/// 4. Fallback → __getitem__ protocol
pub fn binary_subscr(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    // Read container and key from frame (release borrow immediately)
    let (container, key, dst) = {
        let frame = vm.current_frame();
        (
            frame.get_reg(inst.src1().0),
            frame.get_reg(inst.src2().0),
            inst.dst().0,
        )
    };

    // Fast path: try integer subscript first (most common case)
    if let Some(index) = key.as_int() {
        match subscr_integer(container, index) {
            Ok(Some(subscr_result)) => {
                return finish_subscr(vm, dst, subscr_result);
            }
            Ok(None) => {}
            Err(cf) => return cf,
        }
    }

    // Check if key is a slice object
    if let Some(ptr) = key.as_object_ptr() {
        let header = unsafe { &*(ptr as *const ObjectHeader) };
        if header.type_id == TypeId::SLICE {
            let slice = unsafe { &*(ptr as *const SliceObject) };
            match subscr_slice(container, slice) {
                Ok(Some(subscr_result)) => {
                    return finish_subscr(vm, dst, subscr_result);
                }
                Ok(None) => {}
                Err(cf) => return cf,
            }
        }
    }

    // Dict with any key type (not just integer)
    if let Some(ptr) = container.as_object_ptr() {
        let header = unsafe { &*(ptr as *const ObjectHeader) };
        if header.type_id == TypeId::TYPE {
            let args = if let Some(key_ptr) = key.as_object_ptr() {
                let key_header = unsafe { &*(key_ptr as *const ObjectHeader) };
                if key_header.type_id == TypeId::TUPLE {
                    let tuple = unsafe { &*(key_ptr as *const TupleObject) };
                    tuple.iter().copied().collect::<Vec<_>>()
                } else {
                    vec![key]
                }
            } else {
                vec![key]
            };

            let alias = GenericAliasObject::new(container, args);
            let value = match vm.allocator().alloc(alias) {
                Some(alias_ptr) => Value::object_ptr(alias_ptr as *const ()),
                None => {
                    return ControlFlow::Error(RuntimeError::internal(
                        "out of memory: failed to allocate generic alias",
                    ));
                }
            };
            vm.current_frame_mut().set_reg(dst, value);
            return ControlFlow::Continue;
        }

        if header.type_id == TypeId::MAPPING_PROXY {
            let proxy = unsafe { &*(ptr as *const MappingProxyObject) };
            return match builtin_mapping_proxy_get_item(vm, proxy, key) {
                Ok(Some(value)) => {
                    vm.current_frame_mut().set_reg(dst, value);
                    ControlFlow::Continue
                }
                Ok(None) => ControlFlow::Error(RuntimeError::key_error("key not found")),
                Err(err) => ControlFlow::Error(err),
            };
        }

        if can_use_dict_storage_fast_path(ptr, "__getitem__")
            && let Some(dict) = dict_storage_ref_from_ptr(ptr)
        {
            if let Some(value) = dict.get(key) {
                vm.current_frame_mut().set_reg(dst, value);
                return ControlFlow::Continue;
            }
            return ControlFlow::Error(RuntimeError::key_error("key not found"));
        }
    }

    fallback_binary_subscr(vm, container, key, dst)
}

/// Allocate and store the result of a subscript operation.
///
/// This helper handles the allocation of objects that need to be placed
/// on the GC heap, as well as storing direct values.
#[inline]
fn finish_subscr(vm: &mut VirtualMachine, dst: u8, result: SubscriptResult) -> ControlFlow {
    let value = match result {
        SubscriptResult::Value(v) => v,
        SubscriptResult::AllocBytes(bytes) => match vm.allocator().alloc(bytes) {
            Some(ptr) => Value::object_ptr(ptr as *const ()),
            None => {
                return ControlFlow::Error(RuntimeError::internal(
                    "out of memory: failed to allocate bytes",
                ));
            }
        },
        SubscriptResult::AllocString(s) => match vm.allocator().alloc(s) {
            Some(ptr) => Value::object_ptr(ptr as *const ()),
            None => {
                return ControlFlow::Error(RuntimeError::internal(
                    "out of memory: failed to allocate string",
                ));
            }
        },
        SubscriptResult::AllocList(l) => match vm.allocator().alloc(l) {
            Some(ptr) => Value::object_ptr(ptr as *const ()),
            None => {
                return ControlFlow::Error(RuntimeError::internal(
                    "out of memory: failed to allocate list",
                ));
            }
        },
        SubscriptResult::AllocTuple(t) => match vm.allocator().alloc(t) {
            Some(ptr) => Value::object_ptr(ptr as *const ()),
            None => {
                return ControlFlow::Error(RuntimeError::internal(
                    "out of memory: failed to allocate tuple",
                ));
            }
        },
    };
    vm.current_frame_mut().set_reg(dst, value);
    ControlFlow::Continue
}

/// Integer subscript - O(1) for all sequence types.
///
/// Returns `Ok(None)` when the container does not provide an integer fast path,
/// allowing callers to fall back to the general `__getitem__` protocol.
#[inline]
fn subscr_integer(container: Value, index: i64) -> Result<Option<SubscriptResult>, ControlFlow> {
    if let Some(interned) = tagged_interned_string(container)? {
        return subscr_str_integer(interned.as_str(), index).map(Some);
    }

    if let Some(ptr) = container.as_object_ptr() {
        let header = unsafe { &*(ptr as *const ObjectHeader) };

        if let Some(list) = list_storage_ref_from_ptr(ptr) {
            if let Some(value) = list.get(index) {
                return Ok(Some(SubscriptResult::Value(value)));
            }
            let len = list.len();
            return Err(ControlFlow::Error(RuntimeError::index_error(index, len)));
        }

        match header.type_id {
            TypeId::BYTES | TypeId::BYTEARRAY => {
                let bytes = unsafe { &*(ptr as *const BytesObject) };
                if let Some(value) = bytes.get(index) {
                    return Ok(Some(SubscriptResult::Value(Value::int_unchecked(
                        i64::from(value),
                    ))));
                }
                let len = bytes.len();
                return Err(ControlFlow::Error(RuntimeError::index_error(index, len)));
            }
            TypeId::TUPLE => {
                let tuple = unsafe { &*(ptr as *const TupleObject) };
                if let Some(value) = tuple.get(index) {
                    return Ok(Some(SubscriptResult::Value(value)));
                }
                let len = tuple.len();
                return Err(ControlFlow::Error(RuntimeError::index_error(index, len)));
            }
            TypeId::STR => {
                let string = unsafe { &*(ptr as *const StringObject) };
                return subscr_str_integer(string.as_str(), index).map(Some);
            }
            _ => {}
        }

        if let Some(dict) = dict_storage_ref_from_ptr(ptr) {
            let key = Value::int_unchecked(index);
            if let Some(value) = dict.get(key) {
                return Ok(Some(SubscriptResult::Value(value)));
            }
            return Err(ControlFlow::Error(RuntimeError::key_error(format!(
                "{}",
                index
            ))));
        }
    }

    Ok(None)
}

/// Slice subscript - O(k) where k is slice length.
///
/// Returns `Ok(None)` when the container does not provide a slice fast path,
/// allowing callers to fall back to the general `__getitem__` protocol.
#[inline]
fn subscr_slice(
    container: Value,
    slice: &SliceObject,
) -> Result<Option<SubscriptResult>, ControlFlow> {
    if let Some(interned) = tagged_interned_string(container)? {
        return Ok(Some(SubscriptResult::AllocString(string_slice_str(
            interned.as_str(),
            slice,
        ))));
    }

    if let Some(ptr) = container.as_object_ptr() {
        let header = unsafe { &*(ptr as *const ObjectHeader) };

        if let Some(list) = list_storage_ref_from_ptr(ptr) {
            let result = list_slice(list, slice);
            return Ok(Some(SubscriptResult::AllocList(result)));
        }

        match header.type_id {
            TypeId::BYTES | TypeId::BYTEARRAY => {
                let bytes = unsafe { &*(ptr as *const BytesObject) };
                let result = bytes.slice(slice);
                return Ok(Some(SubscriptResult::AllocBytes(result)));
            }
            TypeId::TUPLE => {
                let tuple = unsafe { &*(ptr as *const TupleObject) };
                let result = tuple_slice(tuple, slice);
                return Ok(Some(SubscriptResult::AllocTuple(result)));
            }
            TypeId::STR => {
                let string = unsafe { &*(ptr as *const StringObject) };
                let result = string_slice(string, slice);
                return Ok(Some(SubscriptResult::AllocString(result)));
            }
            _ => {}
        }
    }

    Ok(None)
}

#[inline]
fn slice_from_value(value: Value) -> Option<&'static SliceObject> {
    let ptr = value.as_object_ptr()?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    (header.type_id == TypeId::SLICE).then(|| unsafe { &*(ptr as *const SliceObject) })
}

/// List slice using SliceObject for proper Python semantics.
#[inline]
fn list_slice(list: &ListObject, slice: &SliceObject) -> ListObject {
    let len = list.len();
    let indices = slice.indices(len);

    // Pre-allocate for exact capacity
    let mut result = ListObject::with_capacity(indices.length);

    // Use iterator for proper step handling
    for idx in indices.iter() {
        if idx < len {
            // Safe: idx bounds checked by SliceIndices
            let value = unsafe { list.get_unchecked(idx) };
            result.push(value);
        }
    }

    result
}

/// Tuple slice using SliceObject for proper Python semantics.
#[inline]
fn tuple_slice(tuple: &TupleObject, slice: &SliceObject) -> TupleObject {
    let len = tuple.len();
    let indices = slice.indices(len);

    // Pre-allocate for exact capacity
    let mut items = Vec::with_capacity(indices.length);

    // Use iterator for proper step handling
    for idx in indices.iter() {
        if idx < len {
            if let Some(value) = tuple.get(idx as i64) {
                items.push(value);
            }
        }
    }

    TupleObject::from_vec(items)
}

/// Resolve the content backing a tagged interned string value.
#[inline]
fn tagged_interned_string(container: Value) -> Result<Option<InternedString>, ControlFlow> {
    let Some(ptr) = container.as_string_object_ptr() else {
        return Ok(None);
    };

    interned_by_ptr(ptr as *const u8).map(Some).ok_or_else(|| {
        ControlFlow::Error(RuntimeError::internal(
            "encountered unresolved interned string payload during subscript",
        ))
    })
}

/// Index a Python string with full Unicode code point semantics.
#[inline]
fn subscr_str_integer(string: &str, index: i64) -> Result<SubscriptResult, ControlFlow> {
    let chars: Vec<char> = string.chars().collect();
    let len = chars.len();
    let Some(normalized) = normalize_string_index(index, len) else {
        return Err(ControlFlow::Error(RuntimeError::index_error(index, len)));
    };

    let ch = chars[normalized];
    Ok(SubscriptResult::AllocString(StringObject::from_string(
        ch.to_string(),
    )))
}

#[inline]
fn normalize_string_index(index: i64, len: usize) -> Option<usize> {
    let normalized = if index < 0 { len as i64 + index } else { index };
    (0..len as i64)
        .contains(&normalized)
        .then_some(normalized as usize)
}

/// String slice using SliceObject for proper Python semantics.
#[inline]
fn string_slice(string: &StringObject, slice: &SliceObject) -> StringObject {
    string_slice_str(string.as_str(), slice)
}

/// Slice a Python string with full Unicode code point semantics.
#[inline]
fn string_slice_str(string: &str, slice: &SliceObject) -> StringObject {
    let len = string.chars().count(); // Character count, not byte count
    let indices = slice.indices(len);

    if indices.length == 0 {
        return StringObject::empty();
    }

    // Collect characters at specified indices
    let chars: Vec<char> = string.chars().collect();
    let mut result = String::with_capacity(indices.length * 4); // Max UTF-8 char size

    for idx in indices.iter() {
        if idx < len {
            result.push(chars[idx]);
        }
    }

    StringObject::from_string(result)
}

// =============================================================================
// StoreSubscr: container[key] = value
// =============================================================================

/// StoreSubscr: src1[dst] = src2
///
/// Type-dispatched subscript store with proper mutability handling.
/// Only mutable types (List, Dict) support store.
///
/// # Encoding
///
/// - `dst`: key register
/// - `src1`: container register
/// - `src2`: value register
pub fn store_subscr(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let container = frame.get_reg(inst.src1().0);
    let key = frame.get_reg(inst.dst().0);
    let value = frame.get_reg(inst.src2().0);

    if let Some(ptr) = container.as_object_ptr() {
        let header = unsafe { &*(ptr as *const ObjectHeader) };

        if let Some(list) = list_storage_mut_from_ptr(ptr) {
            if let Some(index) = key.as_int() {
                if list.set(index, value) {
                    return ControlFlow::Continue;
                }
                let len = list.len();
                return ControlFlow::Error(RuntimeError::index_error(index, len));
            }

            if let Some(slice) = slice_from_value(key) {
                let mut iterator = match value_to_iterator(&value) {
                    Ok(iter) => iter,
                    Err(err) => {
                        return ControlFlow::Error(RuntimeError::type_error(err.to_string()));
                    }
                };
                let replacement = iterator.collect_remaining();
                return match list.assign_slice(slice, replacement) {
                    Ok(()) => ControlFlow::Continue,
                    Err(err) => ControlFlow::Error(RuntimeError::value_error(err.to_string())),
                };
            }

            return ControlFlow::Error(RuntimeError::type_error(
                "list indices must be integers or slices",
            ));
        }

        match header.type_id {
            TypeId::DICT => {
                // Dict[key] = value
                let dict = unsafe { &mut *(ptr as *mut DictObject) };
                dict.set(key, value);
                return ControlFlow::Continue;
            }
            TypeId::TUPLE | TypeId::STR => {
                return ControlFlow::Error(RuntimeError::type_error(
                    "object does not support item assignment",
                ));
            }
            _ => {}
        }

        if can_use_dict_storage_fast_path(ptr, "__setitem__")
            && let Some(dict) = dict_storage_mut_from_ptr(ptr)
        {
            dict.set(key, value);
            return ControlFlow::Continue;
        }
    }

    fallback_store_subscr(vm, container, key, value)
}

// =============================================================================
// DeleteSubscr: del container[key]
// =============================================================================

/// DeleteSubscr: del container[key]
///
/// Type-dispatched subscript deletion.
/// Only mutable types with deletion support (List, Dict) are handled.
pub fn delete_subscr(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let container = frame.get_reg(inst.src1().0);
    let key = frame.get_reg(inst.src2().0);

    if let Some(ptr) = container.as_object_ptr() {
        let header = unsafe { &*(ptr as *const ObjectHeader) };

        if let Some(list) = list_storage_mut_from_ptr(ptr) {
            if let Some(index) = key.as_int() {
                if list.remove(index).is_some() {
                    return ControlFlow::Continue;
                }
                let len = list.len();
                return ControlFlow::Error(RuntimeError::index_error(index, len));
            }

            if let Some(slice) = slice_from_value(key) {
                list.delete_slice(slice);
                return ControlFlow::Continue;
            }

            return ControlFlow::Error(RuntimeError::type_error(
                "list indices must be integers or slices",
            ));
        }

        match header.type_id {
            TypeId::DICT => {
                let dict = unsafe { &mut *(ptr as *mut DictObject) };
                if dict.remove(key).is_some() {
                    return ControlFlow::Continue;
                }
                return ControlFlow::Error(RuntimeError::key_error("key not found"));
            }
            TypeId::TUPLE | TypeId::STR => {
                return ControlFlow::Error(RuntimeError::type_error(
                    "object does not support item deletion",
                ));
            }
            _ => {}
        }

        if can_use_dict_storage_fast_path(ptr, "__delitem__")
            && let Some(dict) = dict_storage_mut_from_ptr(ptr)
        {
            if dict.remove(key).is_some() {
                return ControlFlow::Continue;
            }
            return ControlFlow::Error(RuntimeError::key_error("key not found"));
        }
    }

    fallback_delete_subscr(vm, container, key)
}

#[inline]
fn fallback_binary_subscr(
    vm: &mut VirtualMachine,
    container: Value,
    key: Value,
    dst: u8,
) -> ControlFlow {
    let target = match resolve_special_method(container, "__getitem__") {
        Ok(target) => target,
        Err(_) => {
            return ControlFlow::Error(RuntimeError::type_error(format!(
                "'{}' object is not subscriptable",
                container.type_name()
            )));
        }
    };

    match invoke_bound_method_with_args_allow_control_transfer(vm, target, &[key]) {
        Ok(InvokeCallableOutcome::Returned(value)) => {
            vm.current_frame_mut().set_reg(dst, value);
            ControlFlow::Continue
        }
        Ok(InvokeCallableOutcome::ControlTransferred) => ControlFlow::Continue,
        Err(err) => ControlFlow::Error(err),
    }
}

#[inline]
fn fallback_store_subscr(
    vm: &mut VirtualMachine,
    container: Value,
    key: Value,
    value: Value,
) -> ControlFlow {
    let target = match resolve_special_method(container, "__setitem__") {
        Ok(target) => target,
        Err(_) => {
            return ControlFlow::Error(RuntimeError::type_error(format!(
                "'{}' object does not support item assignment",
                container.type_name()
            )));
        }
    };

    match invoke_bound_method_with_args(vm, target, &[key, value]) {
        Ok(_) => ControlFlow::Continue,
        Err(err) => ControlFlow::Error(err),
    }
}

#[inline]
fn fallback_delete_subscr(vm: &mut VirtualMachine, container: Value, key: Value) -> ControlFlow {
    let target = match resolve_special_method(container, "__delitem__") {
        Ok(target) => target,
        Err(_) => {
            return ControlFlow::Error(RuntimeError::type_error(format!(
                "'{}' object does not support item deletion",
                container.type_name()
            )));
        }
    };

    match invoke_bound_method_with_args(vm, target, &[key]) {
        Ok(_) => ControlFlow::Continue,
        Err(err) => ControlFlow::Error(err),
    }
}

#[inline]
fn invoke_bound_method_with_args(
    vm: &mut VirtualMachine,
    target: BoundMethodTarget,
    args: &[Value],
) -> Result<Value, RuntimeError> {
    match (target.implicit_self, args) {
        (Some(implicit_self), []) => invoke_callable_value(vm, target.callable, &[implicit_self]),
        (Some(implicit_self), [arg0]) => {
            invoke_callable_value(vm, target.callable, &[implicit_self, *arg0])
        }
        (Some(implicit_self), [arg0, arg1]) => {
            invoke_callable_value(vm, target.callable, &[implicit_self, *arg0, *arg1])
        }
        (Some(implicit_self), _) => {
            let mut full_args = Vec::with_capacity(args.len() + 1);
            full_args.push(implicit_self);
            full_args.extend_from_slice(args);
            invoke_callable_value(vm, target.callable, &full_args)
        }
        (None, _) => invoke_callable_value(vm, target.callable, args),
    }
}

#[inline]
fn invoke_bound_method_with_args_allow_control_transfer(
    vm: &mut VirtualMachine,
    target: BoundMethodTarget,
    args: &[Value],
) -> Result<InvokeCallableOutcome, RuntimeError> {
    match (target.implicit_self, args) {
        (Some(implicit_self), []) => {
            invoke_callable_value_with_control_transfer(vm, target.callable, &[implicit_self])
        }
        (Some(implicit_self), [arg0]) => invoke_callable_value_with_control_transfer(
            vm,
            target.callable,
            &[implicit_self, *arg0],
        ),
        (Some(implicit_self), [arg0, arg1]) => invoke_callable_value_with_control_transfer(
            vm,
            target.callable,
            &[implicit_self, *arg0, *arg1],
        ),
        (Some(implicit_self), _) => {
            let mut full_args = Vec::with_capacity(args.len() + 1);
            full_args.push(implicit_self);
            full_args.extend_from_slice(args);
            invoke_callable_value_with_control_transfer(vm, target.callable, &full_args)
        }
        (None, _) => invoke_callable_value_with_control_transfer(vm, target.callable, args),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_code::{Instruction, Opcode, Register};
    use prism_core::intern::intern;
    use prism_runtime::object::class::PyClassObject;
    use prism_runtime::object::mro::ClassId;
    use prism_runtime::object::shaped_object::ShapedObject;
    use prism_runtime::object::type_builtins::{
        SubclassBitmap, builtin_class_mro, class_id_to_type_id, register_global_class,
    };
    use std::sync::Arc;

    fn register_test_class(class: PyClassObject) -> Arc<PyClassObject> {
        let mut bitmap = SubclassBitmap::new();
        for &class_id in class.mro() {
            bitmap.set_bit(TypeId::from_raw(class_id.0));
        }

        let class = Arc::new(class);
        register_global_class(class.clone(), bitmap);
        class
    }

    fn register_dict_subclass(name: &str) -> Arc<PyClassObject> {
        let class = PyClassObject::new(intern(name), &[ClassId(TypeId::DICT.raw())], |id| {
            (id.0 < TypeId::FIRST_USER_TYPE).then(|| {
                builtin_class_mro(class_id_to_type_id(id))
                    .into_iter()
                    .collect()
            })
        })
        .expect("dict subclass should build");
        register_test_class(class)
    }

    fn dict_backed_instance_value(class: &Arc<PyClassObject>) -> (*mut ShapedObject, Value) {
        let ptr = Box::into_raw(Box::new(ShapedObject::new_dict_backed(
            class.class_type_id(),
            class.instance_shape().clone(),
        )));
        (ptr, Value::object_ptr(ptr as *const ()))
    }

    // ==========================================================================
    // List Slice Tests
    // ==========================================================================

    #[test]
    fn test_list_slice_forward() {
        let list = ListObject::from_iter(vec![
            Value::int_unchecked(0),
            Value::int_unchecked(1),
            Value::int_unchecked(2),
            Value::int_unchecked(3),
            Value::int_unchecked(4),
        ]);
        let slice = SliceObject::start_stop(1, 4);
        let result = list_slice(&list, &slice);

        assert_eq!(result.len(), 3);
        assert_eq!(result.get(0).unwrap().as_int(), Some(1));
        assert_eq!(result.get(1).unwrap().as_int(), Some(2));
        assert_eq!(result.get(2).unwrap().as_int(), Some(3));
    }

    #[test]
    fn test_list_slice_step() {
        let list = ListObject::from_iter(vec![
            Value::int_unchecked(0),
            Value::int_unchecked(1),
            Value::int_unchecked(2),
            Value::int_unchecked(3),
            Value::int_unchecked(4),
            Value::int_unchecked(5),
        ]);
        let slice = SliceObject::full(0, 6, 2);
        let result = list_slice(&list, &slice);

        assert_eq!(result.len(), 3);
        assert_eq!(result.get(0).unwrap().as_int(), Some(0));
        assert_eq!(result.get(1).unwrap().as_int(), Some(2));
        assert_eq!(result.get(2).unwrap().as_int(), Some(4));
    }

    #[test]
    fn test_list_slice_reverse() {
        let list = ListObject::from_iter(vec![
            Value::int_unchecked(0),
            Value::int_unchecked(1),
            Value::int_unchecked(2),
            Value::int_unchecked(3),
            Value::int_unchecked(4),
        ]);
        let slice = SliceObject::new(None, None, Some(-1));
        let result = list_slice(&list, &slice);

        assert_eq!(result.len(), 5);
        assert_eq!(result.get(0).unwrap().as_int(), Some(4));
        assert_eq!(result.get(1).unwrap().as_int(), Some(3));
        assert_eq!(result.get(2).unwrap().as_int(), Some(2));
        assert_eq!(result.get(3).unwrap().as_int(), Some(1));
        assert_eq!(result.get(4).unwrap().as_int(), Some(0));
    }

    #[test]
    fn test_list_slice_empty() {
        let list = ListObject::from_iter(vec![
            Value::int_unchecked(0),
            Value::int_unchecked(1),
            Value::int_unchecked(2),
        ]);
        let slice = SliceObject::start_stop(5, 10); // Out of bounds
        let result = list_slice(&list, &slice);

        assert_eq!(result.len(), 0);
    }

    // ==========================================================================
    // Tuple Slice Tests
    // ==========================================================================

    #[test]
    fn test_tuple_slice_forward() {
        let tuple = TupleObject::from_vec(vec![
            Value::int_unchecked(10),
            Value::int_unchecked(20),
            Value::int_unchecked(30),
            Value::int_unchecked(40),
        ]);
        let slice = SliceObject::start_stop(0, 2);
        let result = tuple_slice(&tuple, &slice);

        assert_eq!(result.len(), 2);
        assert_eq!(result.get(0).unwrap().as_int(), Some(10));
        assert_eq!(result.get(1).unwrap().as_int(), Some(20));
    }

    #[test]
    fn test_tuple_slice_step() {
        let tuple = TupleObject::from_vec(vec![
            Value::int_unchecked(0),
            Value::int_unchecked(1),
            Value::int_unchecked(2),
            Value::int_unchecked(3),
            Value::int_unchecked(4),
            Value::int_unchecked(5),
        ]);
        let slice = SliceObject::full(1, 6, 2);
        let result = tuple_slice(&tuple, &slice);

        assert_eq!(result.len(), 3);
        assert_eq!(result.get(0).unwrap().as_int(), Some(1));
        assert_eq!(result.get(1).unwrap().as_int(), Some(3));
        assert_eq!(result.get(2).unwrap().as_int(), Some(5));
    }

    #[test]
    fn test_tuple_slice_reverse() {
        let tuple = TupleObject::from_vec(vec![
            Value::int_unchecked(1),
            Value::int_unchecked(2),
            Value::int_unchecked(3),
        ]);
        let slice = SliceObject::new(None, None, Some(-1));
        let result = tuple_slice(&tuple, &slice);

        assert_eq!(result.len(), 3);
        assert_eq!(result.get(0).unwrap().as_int(), Some(3));
        assert_eq!(result.get(1).unwrap().as_int(), Some(2));
        assert_eq!(result.get(2).unwrap().as_int(), Some(1));
    }

    // ==========================================================================
    // String Slice Tests
    // ==========================================================================

    #[test]
    fn test_string_slice_forward() {
        let string = StringObject::new("hello");
        let slice = SliceObject::start_stop(1, 4);
        let result = string_slice(&string, &slice);

        assert_eq!(result.as_str(), "ell");
    }

    #[test]
    fn test_string_slice_step() {
        let string = StringObject::new("abcdef");
        let slice = SliceObject::full(0, 6, 2);
        let result = string_slice(&string, &slice);

        assert_eq!(result.as_str(), "ace");
    }

    #[test]
    fn test_string_slice_reverse() {
        let string = StringObject::new("hello");
        let slice = SliceObject::new(None, None, Some(-1));
        let result = string_slice(&string, &slice);

        assert_eq!(result.as_str(), "olleh");
    }

    #[test]
    fn test_string_slice_unicode() {
        let string = StringObject::new("héllo");
        let slice = SliceObject::start_stop(0, 3);
        let result = string_slice(&string, &slice);

        assert_eq!(result.as_str(), "hél");
    }

    #[test]
    fn test_string_slice_empty() {
        let string = StringObject::new("test");
        let slice = SliceObject::start_stop(10, 20);
        let result = string_slice(&string, &slice);

        assert_eq!(result.as_str(), "");
    }

    #[test]
    fn test_tagged_string_integer_subscript_forward() {
        let result = subscr_integer(Value::string(intern("hello")), 1)
            .expect("tagged string indexing should succeed");

        match result {
            Some(SubscriptResult::AllocString(string)) => assert_eq!(string.as_str(), "e"),
            Some(SubscriptResult::Value(_)) => panic!("expected allocated string result"),
            Some(SubscriptResult::AllocBytes(_)) => panic!("expected allocated string result"),
            Some(SubscriptResult::AllocList(_)) => panic!("expected allocated string result"),
            Some(SubscriptResult::AllocTuple(_)) => panic!("expected allocated string result"),
            None => panic!("expected integer fast path"),
        }
    }

    #[test]
    fn test_tagged_string_integer_subscript_negative_index() {
        let result = subscr_integer(Value::string(intern("hello")), -1)
            .expect("tagged string negative indexing should succeed");

        match result {
            Some(SubscriptResult::AllocString(string)) => assert_eq!(string.as_str(), "o"),
            Some(SubscriptResult::Value(_)) => panic!("expected allocated string result"),
            Some(SubscriptResult::AllocBytes(_)) => panic!("expected allocated string result"),
            Some(SubscriptResult::AllocList(_)) => panic!("expected allocated string result"),
            Some(SubscriptResult::AllocTuple(_)) => panic!("expected allocated string result"),
            None => panic!("expected integer fast path"),
        }
    }

    #[test]
    fn test_tagged_string_slice_forward() {
        let slice = SliceObject::start_stop(1, 4);
        let result = subscr_slice(Value::string(intern("hello")), &slice)
            .expect("tagged string slicing should succeed");

        match result {
            Some(SubscriptResult::AllocString(string)) => assert_eq!(string.as_str(), "ell"),
            Some(SubscriptResult::Value(_)) => panic!("expected allocated string result"),
            Some(SubscriptResult::AllocBytes(_)) => panic!("expected allocated string result"),
            Some(SubscriptResult::AllocList(_)) => panic!("expected allocated string result"),
            Some(SubscriptResult::AllocTuple(_)) => panic!("expected allocated string result"),
            None => panic!("expected slice fast path"),
        }
    }

    #[test]
    fn test_bytes_integer_subscript_returns_int_value() {
        let bytes = BytesObject::from_slice(b"abc");
        let bytes_ptr = Box::leak(Box::new(bytes)) as *mut BytesObject as *const ();
        let result =
            subscr_integer(Value::object_ptr(bytes_ptr), 1).expect("bytes indexing should succeed");

        match result {
            Some(SubscriptResult::Value(value)) => {
                assert_eq!(value.as_int(), Some(i64::from(b'b')))
            }
            Some(SubscriptResult::AllocBytes(_)) => panic!("expected integer value result"),
            Some(SubscriptResult::AllocString(_)) => panic!("expected integer value result"),
            Some(SubscriptResult::AllocList(_)) => panic!("expected integer value result"),
            Some(SubscriptResult::AllocTuple(_)) => panic!("expected integer value result"),
            None => panic!("expected integer fast path"),
        }
    }

    #[test]
    fn test_bytearray_slice_preserves_concrete_type() {
        let bytes = BytesObject::bytearray_from_slice(b"abcd");
        let bytes_ptr = Box::leak(Box::new(bytes)) as *mut BytesObject as *const ();
        let slice = SliceObject::new(None, None, Some(-1));
        let result = subscr_slice(Value::object_ptr(bytes_ptr), &slice)
            .expect("bytearray slicing should succeed");

        match result {
            Some(SubscriptResult::AllocBytes(bytes)) => {
                assert!(bytes.is_bytearray());
                assert_eq!(bytes.as_bytes(), b"dcba");
            }
            Some(SubscriptResult::Value(_)) => panic!("expected allocated byte sequence result"),
            Some(SubscriptResult::AllocString(_)) => {
                panic!("expected allocated byte sequence result")
            }
            Some(SubscriptResult::AllocList(_)) => {
                panic!("expected allocated byte sequence result")
            }
            Some(SubscriptResult::AllocTuple(_)) => {
                panic!("expected allocated byte sequence result")
            }
            None => panic!("expected slice fast path"),
        }
    }

    #[test]
    fn test_binary_subscr_reads_heap_dict_subclass_native_storage() {
        let mut vm = VirtualMachine::new();
        let code = std::sync::Arc::new(prism_code::CodeObject::new("sub", "<test>"));
        vm.push_frame(code, 0).expect("frame push failed");

        let class = register_dict_subclass("BinaryDictSubclass");
        let (instance_ptr, instance_value) = dict_backed_instance_value(&class);
        unsafe {
            (*instance_ptr)
                .dict_backing_mut()
                .expect("dict subclass should expose native dict storage")
                .set(Value::string(intern("answer")), Value::int_unchecked(42));
        }

        vm.current_frame_mut().set_reg(1, instance_value);
        vm.current_frame_mut()
            .set_reg(2, Value::string(intern("answer")));

        let inst = Instruction::op_dss(
            Opcode::GetItem,
            Register::new(3),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(
            binary_subscr(&mut vm, inst),
            ControlFlow::Continue
        ));
        assert_eq!(vm.current_frame().get_reg(3).as_int(), Some(42));
    }

    #[test]
    fn test_binary_subscr_on_type_object_produces_generic_alias() {
        let mut vm = VirtualMachine::new();
        let code = std::sync::Arc::new(prism_code::CodeObject::new("sub", "<test>"));
        vm.push_frame(code, 0).expect("frame push failed");
        vm.current_frame_mut().set_reg(
            1,
            crate::builtins::builtin_type_object_for_type_id(TypeId::LIST),
        );
        vm.current_frame_mut().set_reg(
            2,
            crate::builtins::builtin_type_object_for_type_id(TypeId::INT),
        );

        let inst = Instruction::op_dss(
            Opcode::GetItem,
            Register::new(3),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(
            binary_subscr(&mut vm, inst),
            ControlFlow::Continue
        ));
        let ptr = vm.current_frame().get_reg(3).as_object_ptr().unwrap();
        let header = unsafe { &*(ptr as *const ObjectHeader) };
        assert_eq!(header.type_id, TypeId::GENERIC_ALIAS);
    }

    #[test]
    fn test_binary_subscr_on_mapping_proxy_returns_descriptor_view() {
        let mut vm = VirtualMachine::new();
        let code = std::sync::Arc::new(prism_code::CodeObject::new("sub", "<test>"));
        vm.push_frame(code, 0).expect("frame push failed");

        let mapping = crate::builtins::builtin_type_attribute_value(
            &mut vm,
            TypeId::DICT,
            &prism_core::intern::intern("__dict__"),
        )
        .expect("mapping proxy allocation should succeed")
        .expect("dict type should expose __dict__");

        vm.current_frame_mut().set_reg(1, mapping);
        vm.current_frame_mut()
            .set_reg(2, Value::string(prism_core::intern::intern("fromkeys")));

        let inst = Instruction::op_dss(
            Opcode::GetItem,
            Register::new(3),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(
            binary_subscr(&mut vm, inst),
            ControlFlow::Continue
        ));
        let ptr = vm.current_frame().get_reg(3).as_object_ptr().unwrap();
        let header = unsafe { &*(ptr as *const ObjectHeader) };
        assert_eq!(header.type_id, TypeId::CLASSMETHOD_DESCRIPTOR);
    }

    #[test]
    fn test_store_subscr_assigns_list_slice_from_iterable() {
        let mut vm = VirtualMachine::new();
        let code =
            std::sync::Arc::new(prism_code::CodeObject::new("store", "<test>"));
        vm.push_frame(code, 0).expect("frame push failed");

        let list = ListObject::from_iter(vec![
            Value::int_unchecked(0),
            Value::int_unchecked(1),
            Value::int_unchecked(2),
            Value::int_unchecked(3),
        ]);
        let list_ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let slice = SliceObject::start_stop(1, 3);
        let slice_ptr = Box::leak(Box::new(slice)) as *mut SliceObject as *const ();
        let replacement =
            ListObject::from_iter(vec![Value::int_unchecked(10), Value::int_unchecked(11)]);
        let replacement_ptr = Box::leak(Box::new(replacement)) as *mut ListObject as *const ();

        vm.current_frame_mut()
            .set_reg(1, Value::object_ptr(list_ptr));
        vm.current_frame_mut()
            .set_reg(2, Value::object_ptr(replacement_ptr));
        vm.current_frame_mut()
            .set_reg(3, Value::object_ptr(slice_ptr));

        let inst = Instruction::op_dss(
            Opcode::SetItem,
            Register::new(3),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(store_subscr(&mut vm, inst), ControlFlow::Continue));

        let stored = unsafe { &*(list_ptr as *const ListObject) };
        assert_eq!(
            stored.as_slice(),
            &[
                Value::int_unchecked(0),
                Value::int_unchecked(10),
                Value::int_unchecked(11),
                Value::int_unchecked(3),
            ]
        );
    }

    #[test]
    fn test_store_subscr_updates_heap_dict_subclass_native_storage() {
        let mut vm = VirtualMachine::new();
        let code =
            std::sync::Arc::new(prism_code::CodeObject::new("store", "<test>"));
        vm.push_frame(code, 0).expect("frame push failed");

        let class = register_dict_subclass("StoreDictSubclass");
        let (instance_ptr, instance_value) = dict_backed_instance_value(&class);

        vm.current_frame_mut().set_reg(1, instance_value);
        vm.current_frame_mut().set_reg(2, Value::int_unchecked(99));
        vm.current_frame_mut()
            .set_reg(3, Value::string(intern("token")));

        let inst = Instruction::op_dss(
            Opcode::SetItem,
            Register::new(3),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(store_subscr(&mut vm, inst), ControlFlow::Continue));

        let stored = unsafe {
            (*instance_ptr)
                .dict_backing()
                .expect("dict subclass should expose native dict storage")
                .get(Value::string(intern("token")))
        };
        assert_eq!(stored.as_ref().and_then(Value::as_int), Some(99));
    }

    #[test]
    fn test_store_subscr_rejects_mismatched_extended_slice_assignment() {
        let mut vm = VirtualMachine::new();
        let code =
            std::sync::Arc::new(prism_code::CodeObject::new("store", "<test>"));
        vm.push_frame(code, 0).expect("frame push failed");

        let list = ListObject::from_iter(vec![
            Value::int_unchecked(0),
            Value::int_unchecked(1),
            Value::int_unchecked(2),
            Value::int_unchecked(3),
        ]);
        let list_ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let slice = SliceObject::full(0, 4, 2);
        let slice_ptr = Box::leak(Box::new(slice)) as *mut SliceObject as *const ();
        let replacement = ListObject::from_iter(vec![Value::int_unchecked(10)]);
        let replacement_ptr = Box::leak(Box::new(replacement)) as *mut ListObject as *const ();

        vm.current_frame_mut()
            .set_reg(1, Value::object_ptr(list_ptr));
        vm.current_frame_mut()
            .set_reg(2, Value::object_ptr(replacement_ptr));
        vm.current_frame_mut()
            .set_reg(3, Value::object_ptr(slice_ptr));

        let inst = Instruction::op_dss(
            Opcode::SetItem,
            Register::new(3),
            Register::new(1),
            Register::new(2),
        );
        match store_subscr(&mut vm, inst) {
            ControlFlow::Error(error) => match error.kind {
                crate::error::RuntimeErrorKind::ValueError { ref message } => {
                    assert!(message.contains("extended slice"));
                }
                other => panic!("expected ValueError, got {:?}", other),
            },
            other => panic!("expected error, got {:?}", other),
        }
    }

    #[test]
    fn test_delete_subscr_removes_list_slice() {
        let mut vm = VirtualMachine::new();
        let code = std::sync::Arc::new(prism_code::CodeObject::new(
            "delete", "<test>",
        ));
        vm.push_frame(code, 0).expect("frame push failed");

        let list = ListObject::from_iter(vec![
            Value::int_unchecked(0),
            Value::int_unchecked(1),
            Value::int_unchecked(2),
            Value::int_unchecked(3),
            Value::int_unchecked(4),
        ]);
        let list_ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let slice = SliceObject::full(0, 5, 2);
        let slice_ptr = Box::leak(Box::new(slice)) as *mut SliceObject as *const ();

        vm.current_frame_mut()
            .set_reg(1, Value::object_ptr(list_ptr));
        vm.current_frame_mut()
            .set_reg(2, Value::object_ptr(slice_ptr));

        let inst = Instruction::op_dss(
            Opcode::DelItem,
            Register::new(0),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(
            delete_subscr(&mut vm, inst),
            ControlFlow::Continue
        ));

        let stored = unsafe { &*(list_ptr as *const ListObject) };
        assert_eq!(
            stored.as_slice(),
            &[Value::int_unchecked(1), Value::int_unchecked(3)]
        );
    }

    #[test]
    fn test_delete_subscr_removes_heap_dict_subclass_native_storage_entry() {
        let mut vm = VirtualMachine::new();
        let code = std::sync::Arc::new(prism_code::CodeObject::new(
            "delete", "<test>",
        ));
        vm.push_frame(code, 0).expect("frame push failed");

        let class = register_dict_subclass("DeleteDictSubclass");
        let (instance_ptr, instance_value) = dict_backed_instance_value(&class);
        unsafe {
            (*instance_ptr)
                .dict_backing_mut()
                .expect("dict subclass should expose native dict storage")
                .set(Value::string(intern("victim")), Value::int_unchecked(7));
        }

        vm.current_frame_mut().set_reg(1, instance_value);
        vm.current_frame_mut()
            .set_reg(2, Value::string(intern("victim")));

        let inst = Instruction::op_dss(
            Opcode::DelItem,
            Register::new(0),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(
            delete_subscr(&mut vm, inst),
            ControlFlow::Continue
        ));

        let stored = unsafe {
            (*instance_ptr)
                .dict_backing()
                .expect("dict subclass should expose native dict storage")
                .get(Value::string(intern("victim")))
        };
        assert!(stored.is_none());
    }
}
