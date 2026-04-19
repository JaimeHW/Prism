//! Python truthiness helpers for VM/runtime values.
//!
//! `prism_core::Value::is_truthy()` intentionally stays lightweight and
//! allocation-free, but the VM needs container-aware semantics for stdlib
//! compatibility. This module layers those richer checks on top of raw values
//! without changing the core tagged-value crate dependency graph.

use crate::stdlib::collections::deque::DequeObject;
use prism_core::Value;
use prism_core::intern::interned_by_ptr;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::list::ListObject;
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
    if value.is_none() {
        return false;
    }
    if let Some(flag) = value.as_bool() {
        return flag;
    }
    if let Some(int_value) = value.as_int() {
        return int_value != 0;
    }
    if let Some(float_value) = value.as_float() {
        return float_value != 0.0;
    }
    if value.is_string() {
        let Some(ptr) = value.as_string_object_ptr() else {
            return true;
        };
        return interned_by_ptr(ptr as *const u8).is_some_and(|s| !s.as_str().is_empty());
    }

    let Some(ptr) = value.as_object_ptr() else {
        return true;
    };

    match extract_type_id(ptr) {
        TypeId::STR => !unsafe { &*(ptr as *const StringObject) }.is_empty(),
        TypeId::LIST => !unsafe { &*(ptr as *const ListObject) }.is_empty(),
        TypeId::TUPLE => !unsafe { &*(ptr as *const TupleObject) }.is_empty(),
        TypeId::DICT => !unsafe { &*(ptr as *const DictObject) }.is_empty(),
        TypeId::SET | TypeId::FROZENSET => !unsafe { &*(ptr as *const SetObject) }.is_empty(),
        TypeId::BYTES | TypeId::BYTEARRAY => !unsafe { &*(ptr as *const BytesObject) }.is_empty(),
        TypeId::DEQUE => !unsafe { &*(ptr as *const DequeObject) }.is_empty(),
        TypeId::RANGE => !unsafe { &*(ptr as *const RangeObject) }.is_empty(),
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::is_truthy;
    use crate::stdlib::collections::deque::DequeObject;
    use prism_core::Value;
    use prism_core::intern::intern;
    use prism_runtime::types::bytes::BytesObject;
    use prism_runtime::types::list::ListObject;
    use prism_runtime::types::tuple::TupleObject;

    #[test]
    fn test_empty_tagged_string_is_falsey() {
        assert!(!is_truthy(Value::string(intern(""))));
        assert!(is_truthy(Value::string(intern("x"))));
    }

    #[test]
    fn test_empty_tuple_and_list_are_falsey() {
        let tuple_ptr = Box::into_raw(Box::new(TupleObject::empty()));
        let list_ptr = Box::into_raw(Box::new(ListObject::new()));

        assert!(!is_truthy(Value::object_ptr(tuple_ptr as *const ())));
        assert!(!is_truthy(Value::object_ptr(list_ptr as *const ())));

        unsafe {
            drop(Box::from_raw(tuple_ptr));
            drop(Box::from_raw(list_ptr));
        }
    }

    #[test]
    fn test_empty_bytes_are_falsey() {
        let bytes_ptr = Box::into_raw(Box::new(BytesObject::new()));
        assert!(!is_truthy(Value::object_ptr(bytes_ptr as *const ())));

        unsafe {
            drop(Box::from_raw(bytes_ptr));
        }
    }

    #[test]
    fn test_empty_deque_is_falsey() {
        let deque_ptr = Box::into_raw(Box::new(DequeObject::new()));
        assert!(!is_truthy(Value::object_ptr(deque_ptr as *const ())));

        unsafe {
            drop(Box::from_raw(deque_ptr));
        }
    }
}
