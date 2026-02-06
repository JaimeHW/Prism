//! Object operation handlers.
//!
//! Handles attribute access, item access, and iteration with inline caching.
//! All operations use TypeId-based dispatch for type safety and JIT compatibility.

use crate::VirtualMachine;
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use crate::ops::attribute::is_user_defined_type;
use prism_compiler::bytecode::Instruction;
use prism_core::Value;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::iter::IteratorObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::range::RangeObject;
use prism_runtime::types::tuple::TupleObject;

// =============================================================================
// Type Extraction
// =============================================================================

/// Extract TypeId from an object pointer.
///
/// # Safety
/// The pointer must point to a valid object with ObjectHeader at offset 0.
/// All Prism objects use #[repr(C)] layout with ObjectHeader as first field.
///
/// # Performance
/// This is O(1) - a single memory read. JIT code can inline this as:
/// ```asm
/// mov eax, [rdi]  ; Load TypeId (first 4 bytes of object)
/// ```
#[inline(always)]
pub fn extract_type_id(ptr: *const ()) -> TypeId {
    // SAFETY: All objects have ObjectHeader at offset 0 due to #[repr(C)]
    let header = ptr as *const ObjectHeader;
    unsafe { (*header).type_id }
}

// =============================================================================
// Attribute Access (with Inline Caching)
// =============================================================================

/// GetAttr: dst = src.attr[name_idx]
///
/// Attribute lookup follows Python's descriptor protocol with Shape optimization:
/// 1. Check instance Shape for property (O(1) via hidden class)
/// 2. Fall back to type lookup for methods/class attributes
/// 3. Raise AttributeError if not found
///
/// Supports:
/// - OBJECT: ShapedObject with hidden class optimization
/// - List/Dict/Tuple: Built-in method dispatch (future)
/// - Custom types: User-defined objects with __dict__
#[inline(always)]
pub fn get_attr(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let obj = frame.get_reg(inst.src1().0);
    let name_idx = inst.src2().0 as u16;
    let name = frame.get_name(name_idx).clone();

    // Handle different object types
    if let Some(ptr) = obj.as_object_ptr() {
        let type_id = extract_type_id(ptr);

        match type_id {
            TypeId::OBJECT => {
                // Generic ShapedObject - use Shape-based lookup
                let shaped = unsafe { &*(ptr as *const ShapedObject) };

                // Fast path: Check object's property slots via Shape
                if let Some(value) = shaped.get_property(&*name) {
                    // Note: Deleted properties are set to None in the slot
                    // A real None value is distinguishable via separate tracking
                    vm.current_frame_mut().set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }

                // Property not found on instance
                ControlFlow::Error(RuntimeError::attribute_error("object", name))
            }

            TypeId::DICT => {
                // Dict objects have no attributes (use bracket access)
                ControlFlow::Error(RuntimeError::attribute_error("dict", name))
            }

            TypeId::LIST => {
                // List method lookup (append, extend, pop, etc.)
                // TODO: Implement list method binding
                ControlFlow::Error(RuntimeError::attribute_error("list", name))
            }

            TypeId::TUPLE => {
                // Tuple has limited methods (count, index)
                // TODO: Implement tuple method binding
                ControlFlow::Error(RuntimeError::attribute_error("tuple", name))
            }

            TypeId::SET => {
                // Set methods (add, remove, union, etc.)
                // TODO: Implement set method binding
                ControlFlow::Error(RuntimeError::attribute_error("set", name))
            }

            TypeId::FUNCTION | TypeId::CLOSURE => {
                // Function attributes (__name__, __doc__, __code__, etc.)
                // TODO: Implement function attribute access
                ControlFlow::Error(RuntimeError::attribute_error("function", name))
            }

            _ => {
                // User-defined types use ShapedObject storage
                if is_user_defined_type(type_id) {
                    let shaped = unsafe { &*(ptr as *const ShapedObject) };
                    if let Some(value) = shaped.get_property(&*name) {
                        vm.current_frame_mut().set_reg(inst.dst().0, value);
                        return ControlFlow::Continue;
                    }
                }
                // Unknown object type or property not found
                ControlFlow::Error(RuntimeError::attribute_error(type_id.name(), name))
            }
        }
    } else if obj.is_string() {
        // String methods (upper, lower, split, join, etc.)
        // TODO: Implement string method binding
        ControlFlow::Error(RuntimeError::attribute_error("str", name))
    } else if obj.is_none() {
        ControlFlow::Error(RuntimeError::attribute_error("NoneType", name))
    } else if obj.is_bool() {
        ControlFlow::Error(RuntimeError::attribute_error("bool", name))
    } else if obj.is_int() || obj.is_float() {
        // Numeric methods (bit_length, as_integer_ratio, etc.)
        let type_name = if obj.is_int() { "int" } else { "float" };
        ControlFlow::Error(RuntimeError::attribute_error(type_name, name))
    } else {
        ControlFlow::Error(RuntimeError::attribute_error("unknown", name))
    }
}

/// SetAttr: src1.attr[name_idx] = src2
///
/// Sets an attribute on an object. This may cause a Shape transition
/// if the property is new.
///
/// Supports:
/// - OBJECT: ShapedObject with Shape transition support
/// - Custom types: User-defined objects with __dict__
#[inline(always)]
pub fn set_attr(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let obj = frame.get_reg(inst.dst().0);
    let value = frame.get_reg(inst.src2().0);
    let name_idx = inst.src1().0 as u16;
    let name = frame.get_name(name_idx).clone();

    if let Some(ptr) = obj.as_object_ptr() {
        let type_id = extract_type_id(ptr);

        match type_id {
            TypeId::OBJECT => {
                // Generic ShapedObject - use Shape-based property setting
                let shaped = unsafe { &mut *(ptr as *mut ShapedObject) };

                // Get the global shape registry for transitions
                let registry = shape_registry();

                // Set property (may create Shape transition)
                // Intern the name for efficient storage
                let interned_name = prism_core::intern::intern(&*name);
                shaped.set_property(interned_name, value, registry);

                ControlFlow::Continue
            }

            TypeId::DICT | TypeId::LIST | TypeId::TUPLE | TypeId::SET => {
                // Built-in containers don't support attribute assignment
                ControlFlow::Error(RuntimeError::attribute_error(
                    type_id.name(),
                    format!(
                        "'{}' object attribute '{}' is read-only",
                        type_id.name(),
                        name
                    ),
                ))
            }

            TypeId::FUNCTION | TypeId::CLOSURE => {
                // Functions have some settable attributes (__doc__, __name__)
                // TODO: Implement function attribute setting
                ControlFlow::Error(RuntimeError::attribute_error(
                    "function",
                    format!("cannot set '{}' on function", name),
                ))
            }

            _ => {
                // User-defined types use ShapedObject storage
                if is_user_defined_type(type_id) {
                    let shaped = unsafe { &mut *(ptr as *mut ShapedObject) };
                    let registry = shape_registry();
                    let interned_name = prism_core::intern::intern(&*name);
                    shaped.set_property(interned_name, value, registry);
                    return ControlFlow::Continue;
                }
                // Unknown type - reject
                ControlFlow::Error(RuntimeError::attribute_error(
                    type_id.name(),
                    format!("'{}' object has no attribute '{}'", type_id.name(), name),
                ))
            }
        }
    } else {
        // Non-object values (int, float, bool, str, None) don't support setattr
        let type_name = if obj.is_none() {
            "NoneType"
        } else if obj.is_bool() {
            "bool"
        } else if obj.is_int() {
            "int"
        } else if obj.is_float() {
            "float"
        } else if obj.is_string() {
            "str"
        } else {
            "unknown"
        };
        ControlFlow::Error(RuntimeError::attribute_error(
            type_name,
            format!("'{}' object has no attribute '{}'", type_name, name),
        ))
    }
}

/// DelAttr: del src.attr[name_idx]
///
/// Deletes an attribute from an object.
///
/// Note: This doesn't change the object's Shape - the slot is just set to None.
/// A more sophisticated implementation could use "delete shapes" like V8 does
/// for objects that frequently have properties deleted.
///
/// Supports:
/// - OBJECT: ShapedObject with slot-based deletion
/// - Custom types: User-defined objects with __dict__
#[inline(always)]
pub fn del_attr(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let obj = frame.get_reg(inst.src1().0);
    let name_idx = inst.src2().0 as u16;
    let name = frame.get_name(name_idx).clone();

    if let Some(ptr) = obj.as_object_ptr() {
        let type_id = extract_type_id(ptr);

        match type_id {
            TypeId::OBJECT => {
                // Generic ShapedObject - use slot-based deletion
                let shaped = unsafe { &mut *(ptr as *mut ShapedObject) };

                // Try to delete the property (sets slot to None)
                if shaped.delete_property(&*name) {
                    ControlFlow::Continue
                } else {
                    // Property doesn't exist
                    ControlFlow::Error(RuntimeError::attribute_error(
                        "object",
                        format!("'object' object has no attribute '{}'", name),
                    ))
                }
            }

            TypeId::DICT | TypeId::LIST | TypeId::TUPLE | TypeId::SET => {
                // Built-in containers don't support attribute deletion
                ControlFlow::Error(RuntimeError::attribute_error(
                    type_id.name(),
                    format!(
                        "cannot delete attribute '{}' of '{}' object",
                        name,
                        type_id.name()
                    ),
                ))
            }

            TypeId::FUNCTION | TypeId::CLOSURE => {
                // Functions have some deletable attributes
                // TODO: Implement function attribute deletion
                ControlFlow::Error(RuntimeError::attribute_error(
                    "function",
                    format!("cannot delete '{}' from function", name),
                ))
            }

            _ => {
                // User-defined types use ShapedObject storage
                if is_user_defined_type(type_id) {
                    let shaped = unsafe { &mut *(ptr as *mut ShapedObject) };
                    if shaped.delete_property(&*name) {
                        return ControlFlow::Continue;
                    }
                }
                // Unknown type or property not found - reject
                ControlFlow::Error(RuntimeError::attribute_error(
                    type_id.name(),
                    format!("'{}' object has no attribute '{}'", type_id.name(), name),
                ))
            }
        }
    } else {
        // Non-object values don't support delattr
        let type_name = if obj.is_none() {
            "NoneType"
        } else if obj.is_bool() {
            "bool"
        } else if obj.is_int() {
            "int"
        } else if obj.is_float() {
            "float"
        } else if obj.is_string() {
            "str"
        } else {
            "unknown"
        };
        ControlFlow::Error(RuntimeError::attribute_error(
            type_name,
            format!("'{}' object has no attribute '{}'", type_name, name),
        ))
    }
}

// =============================================================================
// Item Access (Type-Discriminated)
// =============================================================================

/// GetItem: dst = src1[src2]
///
/// Supports list/tuple (integer index) and dict (any hashable key).
/// Uses TypeId dispatch for correct type handling.
#[inline(always)]
pub fn get_item(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let container = frame.get_reg(inst.src1().0);
    let key = frame.get_reg(inst.src2().0);
    let dst = inst.dst().0;

    if let Some(ptr) = container.as_object_ptr() {
        let type_id = extract_type_id(ptr);

        match type_id {
            TypeId::LIST => {
                let list = unsafe { &*(ptr as *const ListObject) };
                if let Some(idx) = key.as_int() {
                    if let Some(val) = list.get(idx) {
                        frame.set_reg(dst, val);
                        ControlFlow::Continue
                    } else {
                        ControlFlow::Error(RuntimeError::index_error(idx, list.len()))
                    }
                } else {
                    ControlFlow::Error(RuntimeError::type_error("list indices must be integers"))
                }
            }
            TypeId::TUPLE => {
                let tuple = unsafe { &*(ptr as *const TupleObject) };
                if let Some(idx) = key.as_int() {
                    if let Some(val) = tuple.get(idx) {
                        frame.set_reg(dst, val);
                        ControlFlow::Continue
                    } else {
                        ControlFlow::Error(RuntimeError::index_error(idx, tuple.len()))
                    }
                } else {
                    ControlFlow::Error(RuntimeError::type_error("tuple indices must be integers"))
                }
            }
            TypeId::DICT => {
                let dict = unsafe { &*(ptr as *const DictObject) };
                if let Some(val) = dict.get(key) {
                    frame.set_reg(dst, val);
                    ControlFlow::Continue
                } else {
                    ControlFlow::Error(RuntimeError::key_error("key not found"))
                }
            }
            TypeId::RANGE => {
                let range = unsafe { &*(ptr as *const RangeObject) };
                if let Some(idx) = key.as_int() {
                    if let Some(val) = range.get(idx) {
                        let int_val = Value::int(val).unwrap_or_else(Value::none);
                        frame.set_reg(dst, int_val);
                        ControlFlow::Continue
                    } else {
                        ControlFlow::Error(RuntimeError::index_error(idx, range.len()))
                    }
                } else {
                    ControlFlow::Error(RuntimeError::type_error("range indices must be integers"))
                }
            }
            _ => ControlFlow::Error(RuntimeError::type_error("object is not subscriptable")),
        }
    } else {
        ControlFlow::Error(RuntimeError::type_error("object is not subscriptable"))
    }
}

/// SetItem: src1[dst] = src2 (dst is key register)
///
/// Sets items in mutable containers (list, dict).
#[inline(always)]
pub fn set_item(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let container = frame.get_reg(inst.src1().0);
    let key = frame.get_reg(inst.dst().0);
    let value = frame.get_reg(inst.src2().0);

    if let Some(ptr) = container.as_object_ptr() {
        let type_id = extract_type_id(ptr);

        match type_id {
            TypeId::LIST => {
                let list = unsafe { &mut *(ptr as *mut ListObject) };
                if let Some(idx) = key.as_int() {
                    if list.set(idx, value) {
                        ControlFlow::Continue
                    } else {
                        ControlFlow::Error(RuntimeError::index_error(idx, list.len()))
                    }
                } else {
                    ControlFlow::Error(RuntimeError::type_error("list indices must be integers"))
                }
            }
            TypeId::DICT => {
                let dict = unsafe { &mut *(ptr as *mut DictObject) };
                dict.set(key, value);
                ControlFlow::Continue
            }
            TypeId::TUPLE => ControlFlow::Error(RuntimeError::type_error(
                "'tuple' object does not support item assignment",
            )),
            _ => ControlFlow::Error(RuntimeError::type_error(
                "object does not support item assignment",
            )),
        }
    } else {
        ControlFlow::Error(RuntimeError::type_error(
            "object does not support item assignment",
        ))
    }
}

/// DelItem: del src1[src2]
#[inline(always)]
pub fn del_item(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let container = frame.get_reg(inst.src1().0);
    let key = frame.get_reg(inst.src2().0);

    if let Some(ptr) = container.as_object_ptr() {
        let type_id = extract_type_id(ptr);

        match type_id {
            TypeId::LIST => {
                let list = unsafe { &mut *(ptr as *mut ListObject) };
                if let Some(idx) = key.as_int() {
                    if list.remove(idx).is_some() {
                        ControlFlow::Continue
                    } else {
                        ControlFlow::Error(RuntimeError::index_error(idx, list.len()))
                    }
                } else {
                    ControlFlow::Error(RuntimeError::type_error("list indices must be integers"))
                }
            }
            TypeId::DICT => {
                let dict = unsafe { &mut *(ptr as *mut DictObject) };
                if dict.remove(key).is_some() {
                    ControlFlow::Continue
                } else {
                    ControlFlow::Error(RuntimeError::key_error("key not found"))
                }
            }
            TypeId::TUPLE => ControlFlow::Error(RuntimeError::type_error(
                "'tuple' object does not support item deletion",
            )),
            _ => ControlFlow::Error(RuntimeError::type_error(
                "object does not support item deletion",
            )),
        }
    } else {
        ControlFlow::Error(RuntimeError::type_error(
            "object does not support item deletion",
        ))
    }
}

// =============================================================================
// Iteration (Type-Discriminated)
// =============================================================================

/// GetIter: dst = iter(src)
///
/// Creates an iterator for the given object.
/// Uses TypeId dispatch for type-specific optimized iterators.
#[inline(always)]
pub fn get_iter(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let obj = frame.get_reg(inst.src1().0);
    let dst = inst.dst().0;

    if let Some(ptr) = obj.as_object_ptr() {
        let type_id = extract_type_id(ptr);

        let iter = match type_id {
            TypeId::LIST => {
                // Create Arc reference to list for iterator
                let list = unsafe { &*(ptr as *const ListObject) };
                // Clone values for now - TODO: use Arc<ListObject> properly
                let values: Vec<Value> = list.iter().cloned().collect();
                IteratorObject::from_values(values)
            }
            TypeId::TUPLE => {
                let tuple = unsafe { &*(ptr as *const TupleObject) };
                let values: Vec<Value> = tuple.iter().cloned().collect();
                IteratorObject::from_values(values)
            }
            TypeId::RANGE => {
                let range = unsafe { &*(ptr as *const RangeObject) };
                IteratorObject::from_range(range.iter())
            }
            TypeId::DICT => {
                // Iterate over dict keys
                let dict = unsafe { &*(ptr as *const DictObject) };
                let keys: Vec<Value> = dict.keys().collect();
                IteratorObject::from_values(keys)
            }
            TypeId::ITERATOR => {
                // Already an iterator - return as-is
                frame.set_reg(dst, obj);
                return ControlFlow::Continue;
            }
            _ => {
                return ControlFlow::Error(RuntimeError::type_error(format!(
                    "'{}' object is not iterable",
                    type_id.name()
                )));
            }
        };

        // Allocate iterator on heap
        // TODO: Use GC allocator instead of Box::into_raw
        let iter_box = Box::new(iter);
        let iter_ptr = Box::into_raw(iter_box) as *const ();
        frame.set_reg(dst, Value::object_ptr(iter_ptr));
        ControlFlow::Continue
    } else {
        ControlFlow::Error(RuntimeError::type_error("object is not iterable"))
    }
}

/// ForIter: dst = next(src), jump if StopIteration
///
/// Advances the iterator and jumps to offset if exhausted.
#[inline(always)]
pub fn for_iter(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let iter_val = frame.get_reg(inst.src1().0);
    let dst = inst.dst().0;
    // Offset is encoded in src2 position as 8-bit signed
    let offset = inst.src2().0 as i8 as i16;

    if let Some(ptr) = iter_val.as_object_ptr() {
        let type_id = extract_type_id(ptr);

        if type_id != TypeId::ITERATOR {
            return ControlFlow::Error(RuntimeError::type_error("for loop requires an iterator"));
        }

        let iter = unsafe { &mut *(ptr as *mut IteratorObject) };
        if let Some(val) = iter.next() {
            frame.set_reg(dst, val);
            ControlFlow::Continue
        } else {
            // StopIteration - jump to exit
            ControlFlow::Jump(offset)
        }
    } else {
        ControlFlow::Error(RuntimeError::type_error("object is not an iterator"))
    }
}

// =============================================================================
// Utilities
// =============================================================================

/// Len: dst = len(src)
///
/// Returns the length of a container.
#[inline(always)]
pub fn len(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let obj = frame.get_reg(inst.src1().0);
    let dst = inst.dst().0;

    if let Some(ptr) = obj.as_object_ptr() {
        let type_id = extract_type_id(ptr);

        let len_val = match type_id {
            TypeId::LIST => {
                let list = unsafe { &*(ptr as *const ListObject) };
                list.len() as i64
            }
            TypeId::TUPLE => {
                let tuple = unsafe { &*(ptr as *const TupleObject) };
                tuple.len() as i64
            }
            TypeId::DICT => {
                let dict = unsafe { &*(ptr as *const DictObject) };
                dict.len() as i64
            }
            TypeId::RANGE => {
                let range = unsafe { &*(ptr as *const RangeObject) };
                range.len() as i64
            }
            _ => {
                return ControlFlow::Error(RuntimeError::type_error(format!(
                    "object of type '{}' has no len()",
                    type_id.name()
                )));
            }
        };

        let value = Value::int(len_val).unwrap_or_else(Value::none);
        frame.set_reg(dst, value);
        ControlFlow::Continue
    } else if obj.is_string() {
        // String length - strings are stored differently (interned)
        // TODO: Implement proper string length extraction from InternedString
        // For now, return error as strings need special handling
        ControlFlow::Error(RuntimeError::type_error("string len() not yet implemented"))
    } else {
        ControlFlow::Error(RuntimeError::type_error("object has no len()"))
    }
}

/// IsCallable: dst = callable(src)
#[inline(always)]
pub fn is_callable(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let obj = frame.get_reg(inst.src1().0);

    let is_callable = if let Some(ptr) = obj.as_object_ptr() {
        let type_id = extract_type_id(ptr);
        matches!(
            type_id,
            TypeId::FUNCTION | TypeId::METHOD | TypeId::CLOSURE | TypeId::TYPE
        )
    } else {
        false
    };

    frame.set_reg(inst.dst().0, Value::bool(is_callable));
    ControlFlow::Continue
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_runtime::object::ObjectHeader;

    #[test]
    fn test_extract_type_id() {
        // Create a list and verify TypeId extraction
        let list = Box::new(ListObject::new());
        let ptr = Box::into_raw(list) as *const ();

        let type_id = extract_type_id(ptr);
        assert_eq!(type_id, TypeId::LIST);

        // Clean up
        unsafe {
            drop(Box::from_raw(ptr as *mut ListObject));
        }
    }

    #[test]
    fn test_type_id_layout() {
        // Verify ObjectHeader layout is correct for JIT compatibility
        assert_eq!(std::mem::offset_of!(ObjectHeader, type_id), 0);
        assert_eq!(std::mem::size_of::<TypeId>(), 4);
        assert_eq!(std::mem::size_of::<ObjectHeader>(), 16);
    }
}
