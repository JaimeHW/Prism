//! Subscript opcode handlers for __getitem__/__setitem__/__delitem__.
//!
//! High-performance type-dispatched subscript operations with:
//! - O(1) TypeId dispatch for built-in types (List, Tuple, Dict, String)
//! - SliceObject support with full Python semantics
//! - Negative index normalization
//! - Proper error handling with IndexError/KeyError/TypeError

use crate::VirtualMachine;
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use prism_compiler::bytecode::Instruction;
use prism_core::Value;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;
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
    /// A StringObject that needs GC allocation.
    AllocString(StringObject),
    /// A ListObject that needs GC allocation (from slice operation).
    AllocList(ListObject),
    /// A TupleObject that needs GC allocation (from slice operation).
    AllocTuple(TupleObject),
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
            Ok(subscr_result) => {
                return finish_subscr(vm, dst, subscr_result);
            }
            Err(cf) => return cf,
        }
    }

    // Check if key is a slice object
    if let Some(ptr) = key.as_object_ptr() {
        let header = unsafe { &*(ptr as *const ObjectHeader) };
        if header.type_id == TypeId::SLICE {
            let slice = unsafe { &*(ptr as *const SliceObject) };
            match subscr_slice(container, slice) {
                Ok(subscr_result) => {
                    return finish_subscr(vm, dst, subscr_result);
                }
                Err(cf) => return cf,
            }
        }
    }

    // Dict with any key type (not just integer)
    if let Some(ptr) = container.as_object_ptr() {
        let header = unsafe { &*(ptr as *const ObjectHeader) };
        if header.type_id == TypeId::DICT {
            let dict = unsafe { &*(ptr as *const DictObject) };
            if let Some(value) = dict.get(key) {
                vm.current_frame_mut().set_reg(dst, value);
                return ControlFlow::Continue;
            } else {
                return ControlFlow::Error(RuntimeError::key_error("key not found"));
            }
        }
    }

    // Fallback: __getitem__ protocol (not yet implemented)
    ControlFlow::Error(RuntimeError::type_error(
        "object is not subscriptable or index type not supported",
    ))
}

/// Allocate and store the result of a subscript operation.
///
/// This helper handles the allocation of objects that need to be placed
/// on the GC heap, as well as storing direct values.
#[inline]
fn finish_subscr(vm: &mut VirtualMachine, dst: u8, result: SubscriptResult) -> ControlFlow {
    let value = match result {
        SubscriptResult::Value(v) => v,
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
/// Returns `SubscriptResult::Value` for direct values (list/tuple/dict elements),
/// or `SubscriptResult::NeedsAlloc` for values that need heap allocation (string chars).
#[inline]
fn subscr_integer(container: Value, index: i64) -> Result<SubscriptResult, ControlFlow> {
    if let Some(ptr) = container.as_object_ptr() {
        let header = unsafe { &*(ptr as *const ObjectHeader) };

        match header.type_id {
            TypeId::LIST => {
                let list = unsafe { &*(ptr as *const ListObject) };
                if let Some(value) = list.get(index) {
                    return Ok(SubscriptResult::Value(value));
                }
                let len = list.len();
                return Err(ControlFlow::Error(RuntimeError::index_error(index, len)));
            }
            TypeId::TUPLE => {
                let tuple = unsafe { &*(ptr as *const TupleObject) };
                if let Some(value) = tuple.get(index) {
                    return Ok(SubscriptResult::Value(value));
                }
                let len = tuple.len();
                return Err(ControlFlow::Error(RuntimeError::index_error(index, len)));
            }
            TypeId::STR => {
                let string = unsafe { &*(ptr as *const StringObject) };
                let s = string.as_str();
                let chars: Vec<char> = s.chars().collect();
                let len = chars.len();

                // Normalize negative index
                let normalized = if index < 0 {
                    (len as i64 + index) as usize
                } else {
                    index as usize
                };

                if normalized < len {
                    let ch = chars[normalized];
                    let char_str = StringObject::from_string(ch.to_string());
                    return Ok(SubscriptResult::AllocString(char_str));
                }
                return Err(ControlFlow::Error(RuntimeError::index_error(index, len)));
            }
            TypeId::DICT => {
                // Dict with integer key
                let dict = unsafe { &*(ptr as *const DictObject) };
                let key = Value::int_unchecked(index);
                if let Some(value) = dict.get(key) {
                    return Ok(SubscriptResult::Value(value));
                }
                return Err(ControlFlow::Error(RuntimeError::key_error(format!(
                    "{}",
                    index
                ))));
            }
            _ => {}
        }
    }

    Err(ControlFlow::Error(RuntimeError::type_error(
        "object is not subscriptable",
    )))
}

/// Slice subscript - O(k) where k is slice length.
///
/// Returns `SubscriptResult` variants that need allocation at the call site.
#[inline]
fn subscr_slice(container: Value, slice: &SliceObject) -> Result<SubscriptResult, ControlFlow> {
    if let Some(ptr) = container.as_object_ptr() {
        let header = unsafe { &*(ptr as *const ObjectHeader) };

        match header.type_id {
            TypeId::LIST => {
                let list = unsafe { &*(ptr as *const ListObject) };
                let result = list_slice(list, slice);
                return Ok(SubscriptResult::AllocList(result));
            }
            TypeId::TUPLE => {
                let tuple = unsafe { &*(ptr as *const TupleObject) };
                let result = tuple_slice(tuple, slice);
                return Ok(SubscriptResult::AllocTuple(result));
            }
            TypeId::STR => {
                let string = unsafe { &*(ptr as *const StringObject) };
                let result = string_slice(string, slice);
                return Ok(SubscriptResult::AllocString(result));
            }
            _ => {}
        }
    }

    Err(ControlFlow::Error(RuntimeError::type_error(
        "object does not support slicing",
    )))
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

/// String slice using SliceObject for proper Python semantics.
#[inline]
fn string_slice(string: &StringObject, slice: &SliceObject) -> StringObject {
    let s = string.as_str();
    let len = s.chars().count(); // Character count, not byte count
    let indices = slice.indices(len);

    if indices.length == 0 {
        return StringObject::empty();
    }

    // Collect characters at specified indices
    let chars: Vec<char> = s.chars().collect();
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

/// StoreSubscr: container[key] = value
///
/// Type-dispatched subscript store with proper mutability handling.
/// Only mutable types (List, Dict) support store.
///
/// # Encoding
///
/// - dst: value to store
/// - src1: container
/// - src2: key
pub fn store_subscr(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let value = frame.get_reg(inst.dst().0);
    let container = frame.get_reg(inst.src1().0);
    let key = frame.get_reg(inst.src2().0);

    if let Some(ptr) = container.as_object_ptr() {
        let header = unsafe { &*(ptr as *const ObjectHeader) };

        match header.type_id {
            TypeId::LIST => {
                // List[int] = value
                if let Some(index) = key.as_int() {
                    let list = unsafe { &mut *(ptr as *mut ListObject) };
                    if list.set(index, value) {
                        return ControlFlow::Continue;
                    }
                    let len = list.len();
                    return ControlFlow::Error(RuntimeError::index_error(index, len));
                }

                // List[slice] = iterable - TODO: implement slice assignment
                return ControlFlow::Error(RuntimeError::type_error(
                    "list indices must be integers or slices",
                ));
            }
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
    }

    // Fallback: __setitem__ protocol (not yet implemented)
    ControlFlow::Error(RuntimeError::type_error(
        "object does not support item assignment",
    ))
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

        match header.type_id {
            TypeId::LIST => {
                if let Some(index) = key.as_int() {
                    let list = unsafe { &mut *(ptr as *mut ListObject) };
                    if list.remove(index).is_some() {
                        return ControlFlow::Continue;
                    }
                    let len = list.len();
                    return ControlFlow::Error(RuntimeError::index_error(index, len));
                }
                return ControlFlow::Error(RuntimeError::type_error(
                    "list indices must be integers",
                ));
            }
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
    }

    // Fallback: __delitem__ protocol (not yet implemented)
    ControlFlow::Error(RuntimeError::type_error(
        "object does not support item deletion",
    ))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
}
