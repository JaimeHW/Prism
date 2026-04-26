use super::*;
use crate::object::class::PyClassObject;
use crate::object::mro::ClassId;
use prism_core::intern::intern;
use std::collections::HashMap;
use std::sync::Arc;

/// Test helper to create a simple class hierarchy registry.
fn create_test_registry() -> HashMap<ClassId, Arc<PyClassObject>> {
    let mut registry = HashMap::new();

    // Create class A (base)
    let class_a = Arc::new(PyClassObject::new_simple(intern("A")));
    let class_a_id = class_a.class_id();
    registry.insert(class_a_id, class_a.clone());

    // Create class B(A)
    let class_b = Arc::new(
        PyClassObject::new(intern("B"), &[class_a_id], |id| {
            registry.get(&id).map(|c| c.mro().to_vec().into())
        })
        .unwrap(),
    );
    let class_b_id = class_b.class_id();
    registry.insert(class_b_id, class_b);

    registry
}

#[test]
fn test_super_creation() {
    let class_id = ClassId(100);
    let obj_type = ClassId(101);

    let super_obj = SuperObject::new_instance(class_id, Value::int_unchecked(42), obj_type);

    assert_eq!(super_obj.this_type(), class_id);
    assert_eq!(super_obj.obj_type(), obj_type);
    assert_eq!(super_obj.binding(), SuperBinding::Instance);
    assert!(super_obj.is_bound());
}

#[test]
fn test_super_unbound() {
    let class_id = ClassId(100);

    let super_obj = SuperObject::new_unbound(class_id);

    assert_eq!(super_obj.this_type(), class_id);
    assert_eq!(super_obj.binding(), SuperBinding::Unbound);
    assert!(!super_obj.is_bound());
}

#[test]
fn test_super_type_binding() {
    let this_type = ClassId(100);
    let bound_type = ClassId(101);

    let super_obj = SuperObject::new_type(this_type, Value::int_unchecked(1), bound_type);

    assert_eq!(super_obj.this_type(), this_type);
    assert_eq!(super_obj.obj_type(), bound_type);
    assert_eq!(super_obj.binding(), SuperBinding::Type);
    assert!(super_obj.is_bound());
}

#[test]
fn test_super_descriptor_get() {
    let class_id = ClassId(100);
    let super_obj = SuperObject::new_unbound(class_id);

    // Bind through __get__
    let obj_type = ClassId(101);
    let bound = super_obj.__get__(Some(Value::int_unchecked(42)), obj_type);

    assert!(bound.is_bound());
    assert_eq!(bound.obj_type(), obj_type);
}

#[test]
fn test_super_already_bound_get() {
    let class_id = ClassId(100);
    let obj_type = ClassId(101);
    let super_obj = SuperObject::new_instance(class_id, Value::int_unchecked(42), obj_type);

    // __get__ on already bound super should return same binding
    let result = super_obj.__get__(Some(Value::int_unchecked(99)), ClassId(102));

    // Should keep original binding
    assert_eq!(result.obj_type(), obj_type);
}

#[test]
fn test_super_lookup_attr_basic() {
    let registry = create_test_registry();

    // Get class B
    let class_b = registry
        .values()
        .find(|c| c.name().as_str() == "B")
        .unwrap();
    let class_b_id = class_b.class_id();

    // Get class A
    let class_a = registry
        .values()
        .find(|c| c.name().as_str() == "A")
        .unwrap();
    let class_a_id = class_a.class_id();

    // Set an attribute on A
    class_a.set_attr(intern("test_method"), Value::int_unchecked(42));

    // Create super(B, obj) where obj is instance of B
    let super_obj = SuperObject::new_instance(class_b_id, Value::int_unchecked(1), class_b_id);

    // Lookup through super should find A.test_method
    let result = super_obj.lookup_attr(&intern("test_method"), |id| registry.get(&id).cloned());

    assert!(result.is_some());
    let result = result.unwrap();
    assert_eq!(result.defining_class, class_a_id);
}

#[test]
fn test_super_lookup_not_found() {
    let registry = create_test_registry();

    // Get class B
    let class_b = registry
        .values()
        .find(|c| c.name().as_str() == "B")
        .unwrap();
    let class_b_id = class_b.class_id();

    // Create super(B, obj)
    let super_obj = SuperObject::new_instance(class_b_id, Value::int_unchecked(1), class_b_id);

    // Lookup nonexistent attribute
    let result = super_obj.lookup_attr(&intern("nonexistent"), |id| registry.get(&id).cloned());

    assert!(result.is_none());
}

#[test]
fn test_super_object_size() {
    // SuperObject should be reasonably sized
    assert!(std::mem::size_of::<SuperObject>() <= 48);
}
