//! Subscript opcode handlers for __getitem__/__setitem__/__delitem__.
//!
//! High-performance type-dispatched subscript operations with:
//! - O(1) TypeId dispatch for built-in types (List, Tuple, Dict, String)
//! - SliceObject support with full Python semantics
//! - Negative index normalization
//! - Proper error handling with IndexError/KeyError/TypeError

use crate::VirtualMachine;
use crate::builtins::builtin_mapping_proxy_get_item;
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use crate::ops::calls::{
    InvokeCallableOutcome, invoke_callable_value, invoke_callable_value_with_control_transfer,
};
use crate::ops::comparison::eq_result;
use crate::ops::iteration::collect_iterable_values;
use crate::ops::method_dispatch::load_method::{BoundMethodTarget, resolve_special_method};
use crate::ops::objects::{
    dict_storage_mut_from_ptr, dict_storage_ref_from_ptr, extract_type_id,
    list_storage_mut_from_ptr, list_storage_ref_from_ptr, tuple_storage_ref_from_ptr,
};
use num_traits::ToPrimitive;
use prism_code::Instruction;
use prism_core::Value;
use prism_core::intern::{InternedString, intern, interned_by_ptr};
use prism_runtime::allocation_context::alloc_value_in_current_heap_or_box;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::type_builtins::global_class;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::object::views::{GenericAliasObject, MappingProxyObject};
use prism_runtime::types::bytes::{BytesObject, value_as_bytes_ref};
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::int::value_to_bigint;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::memoryview::MemoryViewObject;
use prism_runtime::types::range::RangeObject;
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
    /// A RangeObject that needs GC allocation (from slice operation).
    AllocRange(RangeObject),
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
        .lookup_method_published(&intern(special_method))
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
            let value = alloc_value_in_current_heap_or_box(alias);
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
            match dict_get_item_vm(vm, dict, key) {
                Ok(Some(value)) => {
                    vm.current_frame_mut().set_reg(dst, value);
                    return ControlFlow::Continue;
                }
                Ok(None) => return ControlFlow::Error(RuntimeError::key_error("key not found")),
                Err(err) => return ControlFlow::Error(err),
            }
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
        SubscriptResult::AllocBytes(bytes) => alloc_value_in_current_heap_or_box(bytes),
        SubscriptResult::AllocString(string) => alloc_value_in_current_heap_or_box(string),
        SubscriptResult::AllocList(list) => alloc_value_in_current_heap_or_box(list),
        SubscriptResult::AllocTuple(tuple) => alloc_value_in_current_heap_or_box(tuple),
        SubscriptResult::AllocRange(range) => alloc_value_in_current_heap_or_box(range),
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

        let has_builtin_sequence_layout = header.type_id.raw() < TypeId::FIRST_USER_TYPE;

        if has_builtin_sequence_layout && let Some(list) = list_storage_ref_from_ptr(ptr) {
            if let Some(value) = list.get(index) {
                return Ok(Some(SubscriptResult::Value(value)));
            }
            let len = list.len();
            return Err(ControlFlow::Error(RuntimeError::index_error(index, len)));
        }

        if has_builtin_sequence_layout && let Some(tuple) = tuple_storage_ref_from_ptr(ptr) {
            if let Some(value) = tuple.get(index) {
                return Ok(Some(SubscriptResult::Value(value)));
            }
            let len = tuple.len();
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
            TypeId::MEMORYVIEW => {
                let view = unsafe { &*(ptr as *const MemoryViewObject) };
                if view.released() {
                    return Err(ControlFlow::Error(RuntimeError::value_error(
                        "operation forbidden on released memoryview object",
                    )));
                }
                if let Some(value) = view.get(index) {
                    return Ok(Some(SubscriptResult::Value(value)));
                }
                let len = view.len();
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

        let has_builtin_sequence_layout = header.type_id.raw() < TypeId::FIRST_USER_TYPE;

        if has_builtin_sequence_layout && let Some(list) = list_storage_ref_from_ptr(ptr) {
            let result = list_slice(list, slice);
            return Ok(Some(SubscriptResult::AllocList(result)));
        }

        if has_builtin_sequence_layout && let Some(tuple) = tuple_storage_ref_from_ptr(ptr) {
            let result = tuple_slice(tuple, slice);
            return Ok(Some(SubscriptResult::AllocTuple(result)));
        }

        match header.type_id {
            TypeId::BYTES | TypeId::BYTEARRAY => {
                let bytes = unsafe { &*(ptr as *const BytesObject) };
                let result = bytes.slice(slice);
                return Ok(Some(SubscriptResult::AllocBytes(result)));
            }
            TypeId::MEMORYVIEW => {
                let view = unsafe { &*(ptr as *const MemoryViewObject) };
                if view.released() {
                    return Err(ControlFlow::Error(RuntimeError::value_error(
                        "operation forbidden on released memoryview object",
                    )));
                }
                let result = view.slice(slice);
                let value = crate::alloc_managed_value(result);
                return Ok(Some(SubscriptResult::Value(value)));
            }
            TypeId::STR => {
                let string = unsafe { &*(ptr as *const StringObject) };
                let result = string_slice(string, slice);
                return Ok(Some(SubscriptResult::AllocString(result)));
            }
            TypeId::RANGE => {
                let range = unsafe { &*(ptr as *const RangeObject) };
                let result = range_slice(range, slice)?;
                return Ok(Some(SubscriptResult::AllocRange(result)));
            }
            _ => {}
        }
    }

    Ok(None)
}

#[inline]
fn range_slice(range: &RangeObject, slice: &SliceObject) -> Result<RangeObject, ControlFlow> {
    let len = range.try_len().ok_or_else(|| range_slice_overflow())?;
    let indices = slice.indices(len);
    if indices.length == 0 {
        return Ok(RangeObject::new(0, 0, 1));
    }

    let first = range
        .get(indices.start as i64)
        .ok_or_else(|| ControlFlow::Error(RuntimeError::index_error(indices.start as i64, len)))?;
    let base_step = range.step_i64().ok_or_else(range_slice_overflow)?;
    let slice_step = i64::try_from(indices.step).map_err(|_| range_slice_overflow())?;
    let new_step = base_step
        .checked_mul(slice_step)
        .ok_or_else(range_slice_overflow)?;
    let length = i64::try_from(indices.length).map_err(|_| range_slice_overflow())?;
    let stop = first
        .checked_add(
            new_step
                .checked_mul(length)
                .ok_or_else(range_slice_overflow)?,
        )
        .ok_or_else(range_slice_overflow)?;

    Ok(RangeObject::new(first, stop, new_step))
}

#[inline]
fn range_slice_overflow() -> ControlFlow {
    ControlFlow::Error(RuntimeError::exception(
        crate::stdlib::exceptions::ExceptionTypeId::OverflowError.as_u8() as u16,
        "range slice result is too large",
    ))
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

#[inline]
fn bytearray_assignment_byte(value: Value) -> Result<u8, RuntimeError> {
    let integer = if let Some(flag) = value.as_bool() {
        if flag { 1 } else { 0 }
    } else {
        value_to_bigint(value)
            .and_then(|integer| integer.to_i64())
            .ok_or_else(|| RuntimeError::type_error("an integer is required"))?
    };

    u8::try_from(integer).map_err(|_| RuntimeError::value_error("byte must be in range(0, 256)"))
}

fn bytearray_replacement_bytes(
    vm: &mut VirtualMachine,
    value: Value,
) -> Result<Vec<u8>, RuntimeError> {
    if let Some(bytes) = value_as_bytes_ref(value) {
        return Ok(bytes.to_vec());
    }

    if let Some(ptr) = value.as_object_ptr() {
        let header = unsafe { &*(ptr as *const ObjectHeader) };
        if header.type_id == TypeId::MEMORYVIEW {
            let view = unsafe { &*(ptr as *const MemoryViewObject) };
            if view.released() {
                return Err(RuntimeError::value_error(
                    "operation forbidden on released memoryview object",
                ));
            }
            return Ok(view.to_vec());
        }
    }

    collect_iterable_values(vm, value)?
        .into_iter()
        .map(bytearray_assignment_byte)
        .collect()
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
                let replacement = match collect_iterable_values(vm, value) {
                    Ok(values) => values,
                    Err(err) => return ControlFlow::Error(err),
                };
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
            TypeId::BYTEARRAY => {
                let bytes = unsafe { &mut *(ptr as *mut BytesObject) };
                if let Some(index) = key.as_int() {
                    let byte = match bytearray_assignment_byte(value) {
                        Ok(byte) => byte,
                        Err(err) => return ControlFlow::Error(err),
                    };
                    if bytes.set(index, byte) {
                        return ControlFlow::Continue;
                    }
                    let len = bytes.len();
                    return ControlFlow::Error(RuntimeError::index_error(index, len));
                }

                if let Some(slice) = slice_from_value(key) {
                    let replacement = match bytearray_replacement_bytes(vm, value) {
                        Ok(bytes) => bytes,
                        Err(err) => return ControlFlow::Error(err),
                    };
                    return match bytes.assign_slice(slice, &replacement) {
                        Ok(()) => ControlFlow::Continue,
                        Err(err) => ControlFlow::Error(RuntimeError::value_error(err.to_string())),
                    };
                }

                return ControlFlow::Error(RuntimeError::type_error(
                    "bytearray indices must be integers or slices",
                ));
            }
            TypeId::DICT => {
                // Dict[key] = value
                let dict = unsafe { &mut *(ptr as *mut DictObject) };
                if let Err(err) = dict_set_item_vm(vm, dict, key, value) {
                    return ControlFlow::Error(err);
                }
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
            if let Err(err) = dict_set_item_vm(vm, dict, key, value) {
                return ControlFlow::Error(err);
            }
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
                match dict_remove_item_vm(vm, dict, key) {
                    Ok(Some(_)) => return ControlFlow::Continue,
                    Ok(None) => {
                        return ControlFlow::Error(RuntimeError::key_error("key not found"));
                    }
                    Err(err) => return ControlFlow::Error(err),
                }
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
            match dict_remove_item_vm(vm, dict, key) {
                Ok(Some(_)) => return ControlFlow::Continue,
                Ok(None) => return ControlFlow::Error(RuntimeError::key_error("key not found")),
                Err(err) => return ControlFlow::Error(err),
            }
        }
    }

    fallback_delete_subscr(vm, container, key)
}

fn dict_get_item_vm(
    vm: &mut VirtualMachine,
    dict: &DictObject,
    key: Value,
) -> Result<Option<Value>, RuntimeError> {
    if let Some(value) = dict.get(key) {
        return Ok(Some(value));
    }

    let entries = dict.iter().collect::<Vec<_>>();
    for (candidate, value) in entries {
        if eq_result(vm, candidate, key)? {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

fn dict_set_item_vm(
    vm: &mut VirtualMachine,
    dict: &mut DictObject,
    key: Value,
    value: Value,
) -> Result<(), RuntimeError> {
    if dict.contains_key(key) {
        dict.set(key, value);
        return Ok(());
    }

    let keys = dict.keys().collect::<Vec<_>>();
    for candidate in keys {
        if eq_result(vm, candidate, key)? {
            dict.set(candidate, value);
            return Ok(());
        }
    }

    dict.set(key, value);
    Ok(())
}

fn dict_remove_item_vm(
    vm: &mut VirtualMachine,
    dict: &mut DictObject,
    key: Value,
) -> Result<Option<Value>, RuntimeError> {
    if let Some(value) = dict.remove(key) {
        return Ok(Some(value));
    }

    let keys = dict.keys().collect::<Vec<_>>();
    for candidate in keys {
        if eq_result(vm, candidate, key)? {
            return Ok(dict.remove(candidate));
        }
    }

    Ok(None)
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
