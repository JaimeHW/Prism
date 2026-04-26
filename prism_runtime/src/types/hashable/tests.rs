use super::HashableValue;
use crate::object::descriptor::BoundMethod;
use crate::object::shape::Shape;
use crate::object::shaped_object::ShapedObject;
use crate::object::type_obj::TypeId;
use crate::types::string::StringObject;
use crate::types::tuple::TupleObject;
use num_bigint::BigInt;
use prism_core::Value;
use prism_core::intern::intern;
use std::collections::HashSet;

fn boxed_string_value(s: &str) -> (*mut StringObject, Value) {
    let ptr = Box::into_raw(Box::new(StringObject::new(s)));
    (ptr, Value::object_ptr(ptr as *const ()))
}

#[test]
fn hashable_value_matches_interned_and_heap_strings_by_content() {
    let (_ptr, heap) = boxed_string_value("while");
    let interned = Value::string(intern("while"));
    let mut set = HashSet::new();
    set.insert(HashableValue(heap));
    assert!(set.contains(&HashableValue(interned)));
}

#[test]
fn hashable_value_matches_tuples_structurally() {
    let (_ptr_a, heap_a) = boxed_string_value("while");
    let (_ptr_b, heap_b) = boxed_string_value("while");
    let tuple_a = TupleObject::from_slice(&[heap_a, Value::int(1).unwrap()]);
    let tuple_b = TupleObject::from_slice(&[heap_b, Value::int(1).unwrap()]);
    let tuple_a_ptr = Box::into_raw(Box::new(tuple_a));
    let tuple_b_ptr = Box::into_raw(Box::new(tuple_b));

    let mut set = HashSet::new();
    set.insert(HashableValue(Value::object_ptr(tuple_a_ptr as *const ())));
    assert!(set.contains(&HashableValue(Value::object_ptr(tuple_b_ptr as *const ()))));
}

#[test]
fn hashable_value_matches_bound_methods_by_function_and_instance() {
    let method_a = Box::into_raw(Box::new(BoundMethod::new(
        Value::string(intern("__new__")),
        Value::string(intern("Example")),
    )));
    let method_b = Box::into_raw(Box::new(BoundMethod::new(
        Value::string(intern("__new__")),
        Value::string(intern("Example")),
    )));

    let mut set = HashSet::new();
    set.insert(HashableValue(Value::object_ptr(method_a as *const ())));
    assert!(set.contains(&HashableValue(Value::object_ptr(method_b as *const ()))));
}

#[test]
fn hashable_value_matches_identical_nan_bits() {
    let nan = Value::float(f64::NAN);
    let mut set = HashSet::new();
    set.insert(HashableValue(nan));
    assert!(set.contains(&HashableValue(nan)));
}

#[test]
fn hashable_value_matches_int_subclasses_by_numeric_payload() {
    let object =
        ShapedObject::new_int_backed(TypeId::from_raw(512), Shape::empty(), BigInt::from(7));
    let ptr = Box::into_raw(Box::new(object));
    let value = Value::object_ptr(ptr as *const ());

    let mut set = HashSet::new();
    set.insert(HashableValue(value));
    assert!(set.contains(&HashableValue(Value::int(7).unwrap())));
    assert!(set.contains(&HashableValue(Value::float(7.0))));

    unsafe { drop(Box::from_raw(ptr)) };
}
