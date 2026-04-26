use super::*;
use prism_core::intern::intern;

#[test]
fn test_simple_class_creation() {
    let name = intern("MyClass");
    let class = PyClassObject::new_simple(name.clone());

    assert_eq!(class.name(), &name);
    assert!(class.bases().is_empty());
    assert_eq!(class.mro().len(), 2); // [MyClass, object]
    assert!(!class.is_initialized());
}

#[test]
fn test_header_is_first_field() {
    assert_eq!(std::mem::offset_of!(PyClassObject, header), 0);
}

#[test]
fn test_class_type_id_uniqueness() {
    let class1 = PyClassObject::new_simple(intern("Class1"));
    let class2 = PyClassObject::new_simple(intern("Class2"));
    let class3 = PyClassObject::new_simple(intern("Class3"));

    // Each class should have a unique TypeId
    assert_ne!(class1.class_type_id(), class2.class_type_id());
    assert_ne!(class2.class_type_id(), class3.class_type_id());
    assert_ne!(class1.class_type_id(), class3.class_type_id());
}

#[test]
fn test_class_attributes() {
    let class = PyClassObject::new_simple(intern("Test"));
    let attr_name = intern("my_attr");

    // Initially no attribute
    assert!(!class.has_attr(&attr_name));
    assert!(class.get_attr(&attr_name).is_none());

    // Set attribute
    class.set_attr(attr_name.clone(), Value::int_unchecked(42));

    // Now should exist
    assert!(class.has_attr(&attr_name));
    assert_eq!(class.get_attr(&attr_name), Some(Value::int_unchecked(42)));

    // Delete attribute
    let deleted = class.del_attr(&attr_name);
    assert_eq!(deleted, Some(Value::int_unchecked(42)));
    assert!(!class.has_attr(&attr_name));
}

#[test]
fn test_class_dict_preserves_insertion_order() {
    let namespace = ClassDict::new();
    let alpha = intern("alpha");
    let beta = intern("beta");
    let gamma = intern("gamma");

    namespace.set(alpha.clone(), Value::int_unchecked(1));
    namespace.set(beta.clone(), Value::int_unchecked(2));
    namespace.set(gamma.clone(), Value::int_unchecked(3));

    assert_eq!(
        namespace.keys(),
        vec![alpha.clone(), beta.clone(), gamma.clone()]
    );

    let mut seen = Vec::new();
    namespace.for_each(|name, _| seen.push(name.clone()));
    assert_eq!(seen, vec![alpha, beta, gamma]);
}

#[test]
fn test_class_dict_delete_and_reinsert_moves_name_to_end() {
    let namespace = ClassDict::new();
    let alpha = intern("alpha");
    let beta = intern("beta");
    let gamma = intern("gamma");

    namespace.set(alpha.clone(), Value::int_unchecked(1));
    namespace.set(beta.clone(), Value::int_unchecked(2));
    namespace.set(gamma.clone(), Value::int_unchecked(3));
    assert_eq!(namespace.delete(&beta), Some(Value::int_unchecked(2)));
    namespace.set(beta.clone(), Value::int_unchecked(4));

    assert_eq!(
        namespace.keys(),
        vec![alpha.clone(), gamma.clone(), beta.clone()]
    );

    let mut seen = Vec::new();
    namespace.for_each(|name, value| seen.push((name.clone(), value.as_int())));
    assert_eq!(
        seen,
        vec![(alpha, Some(1)), (gamma, Some(3)), (beta, Some(4))]
    );
}

#[test]
fn test_class_flags() {
    let mut class = PyClassObject::new_simple(intern("Test"));

    assert!(!class.is_initialized());
    class.mark_initialized();
    assert!(class.is_initialized());

    assert!(!class.has_slots());
    class.set_slots(vec![intern("x"), intern("y")]);
    assert!(class.has_slots());
    assert_eq!(class.slot_names().unwrap().len(), 2);
}

#[test]
fn test_class_with_inheritance() {
    use std::collections::HashMap;

    // Create parent class
    let parent = PyClassObject::new_simple(intern("Parent"));
    let parent_id = parent.class_id();
    let parent = Arc::new(parent);

    // Create class registry
    let mut registry: HashMap<ClassId, Arc<PyClassObject>> = HashMap::new();
    registry.insert(parent_id, parent.clone());

    // Create child class
    let child_name = intern("Child");
    let child = PyClassObject::new(child_name.clone(), &[parent_id], |id| {
        registry.get(&id).map(|c| c.mro.clone())
    })
    .unwrap();

    // Child's MRO should include parent
    assert_eq!(child.mro().len(), 3); // [Child, Parent, object]
    assert!(child.bases().contains(&parent_id));
}

#[test]
fn test_method_lookup_in_mro() {
    use std::collections::HashMap;
    use std::sync::Arc;

    // Create parent class with a method
    let parent = PyClassObject::new_simple(intern("Parent"));
    let method_name = intern("greet");
    parent.set_attr(method_name.clone(), Value::int_unchecked(100));
    let parent_id = parent.class_id();
    let parent = Arc::new(parent);

    // Registry
    let mut registry: HashMap<ClassId, Arc<PyClassObject>> = HashMap::new();
    registry.insert(parent_id, parent.clone());

    // Create child class
    let child = PyClassObject::new(intern("Child"), &[parent_id], |id| {
        registry.get(&id).map(|c| c.mro.clone())
    })
    .unwrap();

    // Add child to registry
    let child_id = child.class_id();
    let child = Arc::new(child);
    registry.insert(child_id, child.clone());

    // Look up method from child - should find in parent
    let slot = child.lookup_method(&method_name, |id| registry.get(&id).cloned());
    assert!(slot.is_some());
    let slot = slot.unwrap();
    assert_eq!(slot.defining_class, parent_id);
    assert_eq!(slot.mro_index, 1); // Second in MRO (after Child)
}

#[test]
fn test_method_override() {
    use std::collections::HashMap;
    use std::sync::Arc;

    // Create parent class with a method
    let parent = PyClassObject::new_simple(intern("Parent"));
    let method_name = intern("greet");
    parent.set_attr(method_name.clone(), Value::int_unchecked(100));
    let parent_id = parent.class_id();
    let parent = Arc::new(parent);

    // Registry
    let mut registry: HashMap<ClassId, Arc<PyClassObject>> = HashMap::new();
    registry.insert(parent_id, parent.clone());

    // Create child class with overridden method
    let child = PyClassObject::new(intern("Child"), &[parent_id], |id| {
        registry.get(&id).map(|c| c.mro.clone())
    })
    .unwrap();

    // Override the method in child
    child.set_attr(method_name.clone(), Value::int_unchecked(200));

    let child_id = child.class_id();
    let child = Arc::new(child);
    registry.insert(child_id, child.clone());

    // Look up method - should find child's version
    let slot = child.lookup_method(&method_name, |id| registry.get(&id).cloned());
    assert!(slot.is_some());
    let slot = slot.unwrap();
    assert_eq!(slot.defining_class, child_id);
    assert_eq!(slot.mro_index, 0); // First in MRO (Child itself)
    assert_eq!(slot.value, Value::int_unchecked(200));
}

#[test]
fn test_published_method_layout_rebuilds_inherited_and_overridden_entries() {
    use std::collections::HashMap;
    use std::sync::Arc;

    let parent = PyClassObject::new_simple(intern("ParentPublished"));
    let method_name = intern("greet");
    parent.set_attr(method_name.clone(), Value::int_unchecked(100));
    let parent_id = parent.class_id();
    let parent = Arc::new(parent);

    let mut registry: HashMap<ClassId, Arc<PyClassObject>> = HashMap::new();
    registry.insert(parent_id, Arc::clone(&parent));

    let child = PyClassObject::new(intern("ChildPublished"), &[parent_id], |id| {
        registry.get(&id).map(|class| class.mro.clone())
    })
    .unwrap();
    let child_id = child.class_id();
    let child = Arc::new(child);
    registry.insert(child_id, Arc::clone(&child));

    child.rebuild_method_layout(|id| registry.get(&id).cloned());
    let inherited = child
        .lookup_method_published(&method_name)
        .expect("published layout should expose inherited methods");
    assert_eq!(inherited.value, Value::int_unchecked(100));
    assert_eq!(inherited.defining_class, parent_id);
    assert_eq!(inherited.mro_index, 1);

    child.set_attr(method_name.clone(), Value::int_unchecked(200));
    child.rebuild_method_layout(|id| registry.get(&id).cloned());
    let overridden = child
        .lookup_method_published(&method_name)
        .expect("published layout should prefer direct overrides");
    assert_eq!(overridden.value, Value::int_unchecked(200));
    assert_eq!(overridden.defining_class, child_id);
    assert_eq!(overridden.mro_index, 0);
}

#[test]
fn test_class_dict_thread_safety() {
    use std::sync::Arc;
    use std::thread;

    let class = Arc::new(PyClassObject::new_simple(intern("ThreadTest")));

    // Spawn multiple threads to read/write attributes
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let class = class.clone();
            thread::spawn(move || {
                let attr = intern(&format!("attr_{}", i));
                class.set_attr(attr.clone(), Value::int_unchecked(i as i64));
                assert!(class.has_attr(&attr));
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // All attributes should be set
    for i in 0..4 {
        let attr = intern(&format!("attr_{}", i));
        assert!(class.has_attr(&attr));
    }
}

#[test]
fn test_mro_no_heap_allocation() {
    // For simple classes, MRO should not spill to heap
    let class = PyClassObject::new_simple(intern("Simple"));
    assert!(!class.mro.spilled());
}

// =========================================================================
// Instantiation Protocol Tests
// =========================================================================

#[test]
fn test_mark_has_new() {
    let mut class = PyClassObject::new_simple(intern("MyClass"));
    assert!(!class.has_custom_new());

    class.mark_has_new();
    assert!(class.has_custom_new());
}

#[test]
fn test_mark_has_init() {
    let mut class = PyClassObject::new_simple(intern("MyClass"));
    assert!(!class.has_custom_init());

    class.mark_has_init();
    assert!(class.has_custom_init());
}

#[test]
fn test_instantiation_hint_default() {
    let class = PyClassObject::new_simple(intern("MyClass"));
    // No slots, no init → DefaultInit
    assert_eq!(class.instantiation_hint(), InstantiationHint::DefaultInit);
}

#[test]
fn test_instantiation_hint_with_init() {
    let mut class = PyClassObject::new_simple(intern("MyClass"));
    class.mark_has_init();
    // Has init → Generic
    assert_eq!(class.instantiation_hint(), InstantiationHint::Generic);
}

#[test]
fn test_instantiation_hint_inline_slots() {
    let mut class = PyClassObject::new_simple(intern("MyClass"));
    // Set 4 or fewer slots → InlineSlots
    class.set_slots(vec![intern("x"), intern("y"), intern("z")]);
    assert_eq!(class.instantiation_hint(), InstantiationHint::InlineSlots);
}

#[test]
fn test_instantiation_hint_fixed_slots() {
    let mut class = PyClassObject::new_simple(intern("MyClass"));
    // Set more than 4 slots → FixedSlots
    class.set_slots(vec![
        intern("a"),
        intern("b"),
        intern("c"),
        intern("d"),
        intern("e"),
    ]);
    assert_eq!(class.instantiation_hint(), InstantiationHint::FixedSlots);
}

#[test]
fn test_resolve_new_not_found() {
    use std::collections::HashMap;
    use std::sync::Arc;

    let class = PyClassObject::new_simple(intern("MyClass"));
    let registry: HashMap<ClassId, Arc<PyClassObject>> = HashMap::new();

    // No __new__ defined
    let slot = class.resolve_new(|id| registry.get(&id).cloned());
    assert!(slot.is_none());
}

#[test]
fn test_resolve_init_not_found() {
    use std::collections::HashMap;
    use std::sync::Arc;

    let class = PyClassObject::new_simple(intern("MyClass"));
    let registry: HashMap<ClassId, Arc<PyClassObject>> = HashMap::new();

    // No __init__ defined
    let slot = class.resolve_init(|id| registry.get(&id).cloned());
    assert!(slot.is_none());
}

#[test]
fn test_resolve_new_found() {
    use std::collections::HashMap;
    use std::sync::Arc;

    let class = PyClassObject::new_simple(intern("MyClass"));
    let new_name = intern("__new__");
    class.set_attr(new_name.clone(), Value::int_unchecked(999));

    let registry: HashMap<ClassId, Arc<PyClassObject>> = HashMap::new();

    let slot = class.resolve_new(|id| registry.get(&id).cloned());
    assert!(slot.is_some());
    let slot = slot.unwrap();
    assert_eq!(slot.value, Value::int_unchecked(999));
    assert_eq!(slot.defining_class, class.class_id());
}

#[test]
fn test_resolve_init_found() {
    use std::collections::HashMap;
    use std::sync::Arc;

    let class = PyClassObject::new_simple(intern("MyClass"));
    let init_name = intern("__init__");
    class.set_attr(init_name.clone(), Value::int_unchecked(888));

    let registry: HashMap<ClassId, Arc<PyClassObject>> = HashMap::new();

    let slot = class.resolve_init(|id| registry.get(&id).cloned());
    assert!(slot.is_some());
    let slot = slot.unwrap();
    assert_eq!(slot.value, Value::int_unchecked(888));
    assert_eq!(slot.defining_class, class.class_id());
}

#[test]
fn test_resolve_init_inherited() {
    use std::collections::HashMap;
    use std::sync::Arc;

    // Parent with __init__
    let parent = PyClassObject::new_simple(intern("Parent"));
    let init_name = intern("__init__");
    parent.set_attr(init_name.clone(), Value::int_unchecked(777));
    let parent_id = parent.class_id();
    let parent = Arc::new(parent);

    let mut registry: HashMap<ClassId, Arc<PyClassObject>> = HashMap::new();
    registry.insert(parent_id, parent.clone());

    // Child without __init__
    let child = PyClassObject::new(intern("Child"), &[parent_id], |id| {
        registry.get(&id).map(|c| c.mro.clone())
    })
    .unwrap();
    let child_id = child.class_id();
    let child = Arc::new(child);
    registry.insert(child_id, child.clone());

    // Should find parent's __init__
    let slot = child.resolve_init(|id| registry.get(&id).cloned());
    assert!(slot.is_some());
    let slot = slot.unwrap();
    assert_eq!(slot.value, Value::int_unchecked(777));
    assert_eq!(slot.defining_class, parent_id);
    assert_eq!(slot.mro_index, 1); // Second in child's MRO
}
