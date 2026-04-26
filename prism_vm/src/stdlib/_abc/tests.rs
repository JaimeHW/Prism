use super::*;
use prism_core::intern::intern;
use prism_runtime::types::set::SetObject;

#[test]
fn test_abc_register_increments_cache_token_and_virtual_subclass_state() {
    let cls = builtin_type_object_for_type_id(TypeId::LIST);
    let subclass = builtin_type_object_for_type_id(TypeId::DICT);

    abc_reset_registry(&[cls]).expect("reset registry should succeed");
    abc_reset_caches(&[cls]).expect("reset caches should succeed");

    let before = abc_get_cache_token(&[])
        .expect("token lookup should succeed")
        .as_int()
        .expect("token should be int");
    let registered = abc_register(&[cls, subclass]).expect("register should succeed");
    let after = abc_get_cache_token(&[])
        .expect("token lookup should succeed")
        .as_int()
        .expect("token should be int");

    assert_eq!(registered, subclass);
    assert!(after >= before + 1);

    let is_virtual = abc_subclasscheck(&[cls, subclass])
        .expect("subclasscheck should succeed")
        .as_bool()
        .expect("subclasscheck should produce bool");
    assert!(is_virtual);
}

#[test]
fn test_abc_subclasscheck_uses_real_subclass_relation() {
    let int_cls = builtin_type_object_for_type_id(TypeId::INT);
    let bool_cls = builtin_type_object_for_type_id(TypeId::BOOL);

    let result = abc_subclasscheck(&[int_cls, bool_cls])
        .expect("subclasscheck should succeed")
        .as_bool()
        .expect("subclasscheck should produce bool");
    assert!(result);
}

#[test]
fn test_abc_instancecheck_uses_instance_type() {
    let int_cls = builtin_type_object_for_type_id(TypeId::INT);

    let result = abc_instancecheck(&[int_cls, Value::bool(true)])
        .expect("instancecheck should succeed")
        .as_bool()
        .expect("instancecheck should produce bool");
    assert!(result);
}

#[test]
fn test_abc_init_sets_empty_abstractmethods_for_plain_class() {
    let class = Arc::new(PyClassObject::new_simple(intern("DemoAbc")));
    let class_value = Value::object_ptr(Arc::as_ptr(&class) as *const ());

    abc_init(&[class_value]).expect("init should succeed");

    let abstracts = class
        .get_attr(&intern("__abstractmethods__"))
        .expect("__abstractmethods__ should be present");
    let ptr = abstracts
        .as_object_ptr()
        .expect("abstract methods should be object");
    assert_eq!(extract_type_id(ptr), TypeId::FROZENSET);

    let set = unsafe { &*(ptr as *const SetObject) };
    assert_eq!(set.len(), 0);
}

#[test]
fn test_clear_abc_state_for_class_ids_drops_vm_scoped_values() {
    use prism_runtime::object::type_builtins::{
        SubclassBitmap, register_global_class, unregister_global_class,
    };

    let class = Arc::new(PyClassObject::new_simple(intern("ScopedAbc")));
    let class_id = class.class_id();
    let class_value = Value::object_ptr(Arc::as_ptr(&class) as *const ());
    let mut bitmap = SubclassBitmap::new();
    bitmap.set_bit(class.class_type_id());
    register_global_class(Arc::clone(&class), bitmap);

    with_abc_state_mut(class_value, |state| {
        state.registry.insert(class_value);
        state.cache.insert(class_value);
        state.negative_cache.insert(class_value);
    });
    assert!(
        ABC_STATES
            .read()
            .unwrap()
            .contains_key(&state_key(class_value))
    );

    clear_abc_state_for_class_ids([class_id]);
    assert!(
        !ABC_STATES
            .read()
            .unwrap()
            .contains_key(&state_key(class_value))
    );

    unregister_global_class(class_id);
}

#[test]
fn test_get_dump_returns_four_tuple() {
    let cls = builtin_type_object_for_type_id(TypeId::LIST);
    let dump = abc_get_dump(&[cls]).expect("dump should succeed");
    let ptr = dump.as_object_ptr().expect("dump should be tuple");
    let tuple = unsafe { &*(ptr as *const TupleObject) };

    assert_eq!(tuple.len(), 4);
    assert!(tuple.get(3).and_then(|value| value.as_int()).is_some());
}
