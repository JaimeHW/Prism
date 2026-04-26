use super::is_truthy;
use crate::stdlib::collections::deque::DequeObject;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::complex::ComplexObject;
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

#[test]
fn test_zero_complex_is_falsey() {
    let zero_ptr = Box::into_raw(Box::new(ComplexObject::new(0.0, 0.0)));
    let nonzero_ptr = Box::into_raw(Box::new(ComplexObject::new(1.0, 0.0)));

    assert!(!is_truthy(Value::object_ptr(zero_ptr as *const ())));
    assert!(is_truthy(Value::object_ptr(nonzero_ptr as *const ())));

    unsafe {
        drop(Box::from_raw(zero_ptr));
        drop(Box::from_raw(nonzero_ptr));
    }
}
