
use super::*;
use crate::builtins::builtin_type;
use crate::ops::calls::{invoke_callable_value, invoke_callable_value_with_keywords};
use crate::ops::objects::{extract_type_id, get_attribute_value, list_storage_ref_from_ptr};

fn dict_from_value(value: Value) -> *mut DictObject {
    value
        .as_object_ptr()
        .expect("dict-backed value should be object") as *mut DictObject
}

fn list_from_value(value: Value) -> &'static ListObject {
    let ptr = value
        .as_object_ptr()
        .expect("list-backed value should be object");
    list_storage_ref_from_ptr(ptr).expect("value should be list-backed")
}

fn tuple_from_value(value: Value) -> &'static TupleObject {
    let ptr = value
        .as_object_ptr()
        .expect("tuple-backed value should be object");
    unsafe { &*(ptr as *const TupleObject) }
}

#[test]
fn test_get_attr_exposes_counter_namedtuple_and_collection_type_objects() {
    let module = CollectionsModule::new();

    assert!(
        module
            .get_attr("Counter")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );
    assert!(
        module
            .get_attr("namedtuple")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );

    for name in [
        "ChainMap",
        "defaultdict",
        "OrderedDict",
        "UserDict",
        "UserList",
        "UserString",
    ] {
        let class_value = module
            .get_attr(name)
            .unwrap_or_else(|_| panic!("collections.{name} should resolve"));
        let class_ptr = class_value
            .as_object_ptr()
            .unwrap_or_else(|| panic!("collections.{name} should be a class"));
        assert_eq!(extract_type_id(class_ptr), TypeId::TYPE);

        unsafe {
            drop(Arc::from_raw(class_ptr as *const PyClassObject));
        }
    }
}

#[test]
fn test_pprint_sensitive_collection_reprs_are_distinct_from_builtin_dispatch_keys() {
    let ordered = ORDEREDDICT_CLASS
        .get_attr(&intern("__repr__"))
        .expect("OrderedDict should define __repr__");
    let default_dict = DEFAULTDICT_CLASS
        .get_attr(&intern("__repr__"))
        .expect("defaultdict should define __repr__");
    let chainmap = CHAINMAP_CLASS
        .get_attr(&intern("__repr__"))
        .expect("ChainMap should define __repr__");
    let user_dict = USERDICT_CLASS
        .get_attr(&intern("__repr__"))
        .expect("UserDict should define __repr__");

    assert_ne!(ordered, default_dict);
    assert_ne!(ordered, chainmap);
    assert_ne!(default_dict, chainmap);
    assert_ne!(user_dict, chainmap);
}

#[test]
fn test_counter_builtin_counts_iterables() {
    let mut vm = VirtualMachine::new();
    let value = builtin_counter(
        &mut vm,
        &[leak_object_value(TupleObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
        ]))],
    )
    .expect("Counter should construct");

    let dict_ptr = dict_from_value(value);
    let dict = unsafe { &*dict_ptr };
    assert_eq!(dict.get(Value::int(1).unwrap()).unwrap().as_int(), Some(2));
    assert_eq!(dict.get(Value::int(2).unwrap()).unwrap().as_int(), Some(1));

    unsafe {
        drop(Box::from_raw(dict_ptr));
    }
}

#[test]
fn test_counter_builtin_accepts_mapping_input() {
    let mut vm = VirtualMachine::new();
    let mut mapping = DictObject::new();
    mapping.set(Value::string(intern("x")), Value::int(3).unwrap());
    let mapping_value = leak_object_value(mapping);

    let value = builtin_counter(&mut vm, &[mapping_value]).expect("Counter should copy mapping");
    let dict_ptr = dict_from_value(value);
    let dict = unsafe { &*dict_ptr };
    assert_eq!(
        dict.get(Value::string(intern("x"))).unwrap().as_int(),
        Some(3)
    );

    unsafe {
        drop(Box::from_raw(
            mapping_value.as_object_ptr().unwrap() as *mut DictObject
        ));
        drop(Box::from_raw(dict_ptr));
    }
}

#[test]
fn test_namedtuple_factory_builds_type_with_declared_fields() {
    let mut vm = VirtualMachine::new();
    let class_value = builtin_namedtuple(
        &mut vm,
        &[
            Value::string(intern("Pair")),
            Value::string(intern("left right")),
        ],
    )
    .expect("namedtuple should construct class");

    let class_ptr = class_value
        .as_object_ptr()
        .expect("class value should be object");
    assert_eq!(
        crate::ops::objects::extract_type_id(class_ptr),
        TypeId::TYPE
    );

    let class = unsafe { &*(class_ptr as *const PyClassObject) };
    assert_eq!(class.name().as_str(), "Pair");
    assert!(class.mro().contains(&ClassId(TypeId::TUPLE.raw())));

    let fields_value = class.get_attr(&intern("_fields")).expect("fields tuple");
    let fields_ptr = fields_value
        .as_object_ptr()
        .expect("_fields should be tuple object");
    let fields = unsafe { &*(fields_ptr as *const TupleObject) };
    assert_eq!(fields.len(), 2);

    unsafe {
        drop(Box::from_raw(fields_ptr as *mut TupleObject));
        drop(Arc::from_raw(class_ptr as *const PyClassObject));
    }
}

#[test]
fn test_namedtuple_factory_unregisters_vm_owned_heap_type_on_vm_drop() {
    let class_id = {
        let mut vm = VirtualMachine::new();
        let class_value = builtin_namedtuple(
            &mut vm,
            &[
                Value::string(intern("ScopedPair")),
                Value::string(intern("left right")),
            ],
        )
        .expect("namedtuple should construct class");

        let class_ptr = class_value
            .as_object_ptr()
            .expect("namedtuple class should be object-backed");
        let class = unsafe { &*(class_ptr as *const PyClassObject) };
        let class_id = class.class_id();
        assert!(global_class(class_id).is_some());

        unsafe {
            drop(Arc::from_raw(class_ptr as *const PyClassObject));
        }
        class_id
    };

    assert!(global_class(class_id).is_none());
}

#[test]
fn test_namedtuple_factory_records_module_and_defaults() {
    let mut vm = VirtualMachine::new();
    let defaults = leak_object_value(TupleObject::from_slice(&[Value::int(7).unwrap()]));
    let class_value = builtin_namedtuple(
        &mut vm,
        &[
            Value::string(intern("Point")),
            Value::string(intern("x y")),
            Value::bool(false),
            defaults,
            Value::string(intern("demo.module")),
        ],
    )
    .expect("namedtuple should construct class with defaults");

    let class_ptr = class_value
        .as_object_ptr()
        .expect("class value should be object");
    let class = unsafe { &*(class_ptr as *const PyClassObject) };

    let module_value = class.get_attr(&intern("__module__")).expect("module attr");
    assert_eq!(extract_string_value(module_value).unwrap(), "demo.module");

    let field_defaults = class
        .get_attr(&intern("_field_defaults"))
        .expect("field defaults attr");
    let dict_ptr = dict_from_value(field_defaults);
    let dict = unsafe { &*dict_ptr };
    assert_eq!(
        dict.get(Value::string(intern("y"))).unwrap().as_int(),
        Some(7)
    );

    unsafe {
        drop(Box::from_raw(
            defaults.as_object_ptr().unwrap() as *mut TupleObject
        ));
        drop(Box::from_raw(dict_ptr));
        drop(Arc::from_raw(class_ptr as *const PyClassObject));
    }
}

#[test]
fn test_namedtuple_instances_bind_positional_fields_and_publish_class_placeholders() {
    let mut vm = VirtualMachine::new();
    let class_value = builtin_namedtuple(
        &mut vm,
        &[
            Value::string(intern("Pair")),
            Value::string(intern("left right")),
        ],
    )
    .expect("namedtuple should construct class");

    let instance = invoke_callable_value(
        &mut vm,
        class_value,
        &[Value::int(10).unwrap(), Value::int(20).unwrap()],
    )
    .expect("namedtuple class should accept positional construction");

    let instance_ptr = instance
        .as_object_ptr()
        .expect("namedtuple instance should be heap allocated");
    let shaped = unsafe { &*(instance_ptr as *const ShapedObject) };
    let tuple = shaped
        .tuple_backing()
        .expect("namedtuple instances should use native tuple storage");
    assert_eq!(tuple.len(), 2);
    assert_eq!(tuple.as_slice()[0].as_int(), Some(10));
    assert_eq!(tuple.as_slice()[1].as_int(), Some(20));
    assert_eq!(
        get_attribute_value(&mut vm, instance, &intern("left"))
            .expect("left descriptor should read tuple storage")
            .as_int(),
        Some(10)
    );
    assert_eq!(
        get_attribute_value(&mut vm, instance, &intern("right"))
            .expect("right descriptor should read tuple storage")
            .as_int(),
        Some(20)
    );

    let class_ptr = class_value
        .as_object_ptr()
        .expect("namedtuple class should be object-backed");
    let class = unsafe { &*(class_ptr as *const PyClassObject) };
    assert!(class.get_attr(&intern("left")).is_some());
    assert!(class.get_attr(&intern("right")).is_some());

    unsafe {
        drop(Box::from_raw(instance_ptr as *mut ShapedObject));
        drop(Arc::from_raw(class_ptr as *const PyClassObject));
    }
}

#[test]
fn test_namedtuple_instances_apply_defaults_and_keywords() {
    let mut vm = VirtualMachine::new();
    let defaults = leak_object_value(TupleObject::from_slice(&[Value::int(7).unwrap()]));
    let class_value = builtin_namedtuple(
        &mut vm,
        &[
            Value::string(intern("Point")),
            Value::string(intern("x y")),
            Value::bool(false),
            defaults,
            Value::string(intern("demo.point")),
        ],
    )
    .expect("namedtuple should construct class");

    let instance = invoke_callable_value_with_keywords(
        &mut vm,
        class_value,
        &[Value::int(3).unwrap()],
        &[("y", Value::int(11).unwrap())],
    )
    .expect("namedtuple class should accept keyword overrides");

    let instance_ptr = instance
        .as_object_ptr()
        .expect("namedtuple instance should be heap allocated");
    let shaped = unsafe { &*(instance_ptr as *const ShapedObject) };
    let tuple = shaped
        .tuple_backing()
        .expect("namedtuple instances should use native tuple storage");
    assert_eq!(tuple.as_slice()[0].as_int(), Some(3));
    assert_eq!(tuple.as_slice()[1].as_int(), Some(11));

    let defaulted = invoke_callable_value(&mut vm, class_value, &[Value::int(5).unwrap()])
        .expect("namedtuple class should fill omitted defaults");
    let defaulted_ptr = defaulted
        .as_object_ptr()
        .expect("defaulted namedtuple instance should be heap allocated");
    let defaulted_shaped = unsafe { &*(defaulted_ptr as *const ShapedObject) };
    let defaulted_tuple = defaulted_shaped
        .tuple_backing()
        .expect("defaulted namedtuple should use native tuple storage");
    assert_eq!(defaulted_tuple.as_slice()[0].as_int(), Some(5));
    assert_eq!(defaulted_tuple.as_slice()[1].as_int(), Some(7));
    assert_eq!(
        get_attribute_value(&mut vm, defaulted, &intern("x"))
            .expect("x descriptor should read tuple storage")
            .as_int(),
        Some(5)
    );
    assert_eq!(
        get_attribute_value(&mut vm, defaulted, &intern("y"))
            .expect("y descriptor should read tuple storage")
            .as_int(),
        Some(7)
    );

    unsafe {
        drop(Box::from_raw(
            defaults.as_object_ptr().unwrap() as *mut TupleObject
        ));
        drop(Box::from_raw(instance_ptr as *mut ShapedObject));
        drop(Box::from_raw(defaulted_ptr as *mut ShapedObject));
        drop(Arc::from_raw(
            class_value.as_object_ptr().unwrap() as *const PyClassObject
        ));
    }
}

#[test]
fn test_namedtuple_subclasses_construct_from_base_field_schema() {
    let mut vm = VirtualMachine::new();
    let base_class = builtin_namedtuple(
        &mut vm,
        &[
            Value::string(intern("BasePair")),
            Value::string(intern("left right")),
        ],
    )
    .expect("namedtuple should construct base class");
    let bases = leak_object_value(TupleObject::from_slice(&[base_class]));
    let widened_fields = leak_object_value(TupleObject::from_slice(&[
        Value::string(intern("left")),
        Value::string(intern("right")),
        Value::string(intern("extra")),
    ]));
    let mut namespace = DictObject::new();
    namespace.set(Value::string(intern("_fields")), widened_fields);
    let namespace_value = leak_object_value(namespace);

    let subclass = builtin_type(&[Value::string(intern("WidenedPair")), bases, namespace_value])
        .expect("namedtuple subclass should be constructible");
    let instance = invoke_callable_value(
        &mut vm,
        subclass,
        &[Value::int(10).unwrap(), Value::int(20).unwrap()],
    )
    .expect("subclass should use the base namedtuple constructor schema");

    let instance_ptr = instance
        .as_object_ptr()
        .expect("subclass instance should be heap allocated");
    let shaped = unsafe { &*(instance_ptr as *const ShapedObject) };
    let tuple = shaped
        .tuple_backing()
        .expect("namedtuple subclass should retain native tuple storage");
    assert_eq!(tuple.len(), 2);
    assert_eq!(tuple.as_slice()[0].as_int(), Some(10));
    assert_eq!(tuple.as_slice()[1].as_int(), Some(20));
    assert_eq!(
        get_attribute_value(&mut vm, instance, &intern("left"))
            .expect("inherited descriptor should read tuple storage")
            .as_int(),
        Some(10)
    );

    unsafe {
        drop(Box::from_raw(instance_ptr as *mut ShapedObject));
        drop(Box::from_raw(
            widened_fields.as_object_ptr().unwrap() as *mut TupleObject
        ));
        drop(Box::from_raw(
            namespace_value.as_object_ptr().unwrap() as *mut DictObject
        ));
        drop(Box::from_raw(
            bases.as_object_ptr().unwrap() as *mut TupleObject
        ));
        drop(Arc::from_raw(
            subclass.as_object_ptr().unwrap() as *const PyClassObject
        ));
        drop(Arc::from_raw(
            base_class.as_object_ptr().unwrap() as *const PyClassObject
        ));
    }
}

#[test]
fn test_chainmap_supports_items_new_child_parents_and_fromkeys() {
    let mut vm = VirtualMachine::new();
    let class_value = class_value(&CHAINMAP_CLASS);
    let class_ptr = class_value
        .as_object_ptr()
        .expect("ChainMap class should be object-backed");

    let mut base = DictObject::new();
    base.set(
        Value::string(intern("scope")),
        Value::string(intern("outer")),
    );
    let mut front = DictObject::new();
    front.set(
        Value::string(intern("scope")),
        Value::string(intern("inner")),
    );

    let chainmap = invoke_callable_value(
        &mut vm,
        class_value,
        &[leak_object_value(front), leak_object_value(base)],
    )
    .expect("ChainMap should construct");

    let getitem = get_attribute_value(&mut vm, chainmap, &intern("__getitem__"))
        .expect("__getitem__ should resolve");
    let scope = invoke_callable_value(&mut vm, getitem, &[Value::string(intern("scope"))])
        .expect("lookup should succeed");
    assert_eq!(extract_string_value(scope).unwrap(), "inner");

    let items =
        get_attribute_value(&mut vm, chainmap, &intern("items")).expect("items should exist");
    let item_values = invoke_callable_value(&mut vm, items, &[]).expect("items() should succeed");
    let items_list = list_from_value(item_values);
    assert_eq!(items_list.len(), 1);
    let item = tuple_from_value(
        items_list
            .get(0)
            .expect("items list should contain a tuple"),
    );
    assert_eq!(
        extract_string_value(item.get(0).expect("tuple key")).unwrap(),
        "scope"
    );
    assert_eq!(
        extract_string_value(item.get(1).expect("tuple value")).unwrap(),
        "inner"
    );

    let new_child = get_attribute_value(&mut vm, chainmap, &intern("new_child"))
        .expect("new_child should exist");
    let child = invoke_callable_value_with_keywords(
        &mut vm,
        new_child,
        &[],
        &[("phase", Value::string(intern("leaf")))],
    )
    .expect("new_child() should succeed");

    let child_items =
        get_attribute_value(&mut vm, child, &intern("items")).expect("child items should exist");
    let child_item_values =
        invoke_callable_value(&mut vm, child_items, &[]).expect("child items() should work");
    let child_items_list = list_from_value(child_item_values);
    assert_eq!(child_items_list.len(), 2);

    let first_child_item = tuple_from_value(
        child_items_list
            .get(0)
            .expect("child items should contain a first tuple"),
    );
    let second_child_item = tuple_from_value(
        child_items_list
            .get(1)
            .expect("child items should contain a second tuple"),
    );
    assert_eq!(
        extract_string_value(first_child_item.get(0).unwrap()).unwrap(),
        "scope"
    );
    assert_eq!(
        extract_string_value(second_child_item.get(0).unwrap()).unwrap(),
        "phase"
    );

    let parents =
        get_attribute_value(&mut vm, child, &intern("parents")).expect("parents should exist");
    let parent_items =
        get_attribute_value(&mut vm, parents, &intern("items")).expect("parent items should exist");
    let parent_item_values =
        invoke_callable_value(&mut vm, parent_items, &[]).expect("parent items() should work");
    let parent_items_list = list_from_value(parent_item_values);
    assert_eq!(parent_items_list.len(), 1);
    let parent_item = tuple_from_value(
        parent_items_list
            .get(0)
            .expect("parents items should contain a tuple"),
    );
    assert_eq!(
        extract_string_value(parent_item.get(0).unwrap()).unwrap(),
        "scope"
    );
    assert_eq!(
        extract_string_value(parent_item.get(1).unwrap()).unwrap(),
        "inner"
    );

    let fromkeys = get_attribute_value(&mut vm, class_value, &intern("fromkeys"))
        .expect("fromkeys should resolve");
    let fromkeys_result = invoke_callable_value(
        &mut vm,
        fromkeys,
        &[
            leak_object_value(ListObject::from_iter([
                Value::string(intern("a")),
                Value::string(intern("b")),
            ])),
            Value::int(5).unwrap(),
        ],
    )
    .expect("fromkeys() should succeed");
    let fromkeys_items = get_attribute_value(&mut vm, fromkeys_result, &intern("items"))
        .expect("fromkeys items should exist");
    let fromkeys_item_values =
        invoke_callable_value(&mut vm, fromkeys_items, &[]).expect("items() should work");
    let fromkeys_items_list = list_from_value(fromkeys_item_values);
    assert_eq!(fromkeys_items_list.len(), 2);
    let first_pair = tuple_from_value(
        fromkeys_items_list
            .get(0)
            .expect("fromkeys items should contain first tuple"),
    );
    let second_pair = tuple_from_value(
        fromkeys_items_list
            .get(1)
            .expect("fromkeys items should contain second tuple"),
    );
    assert_eq!(
        extract_string_value(first_pair.get(0).unwrap()).unwrap(),
        "a"
    );
    assert_eq!(first_pair.get(1).unwrap().as_int(), Some(5));
    assert_eq!(
        extract_string_value(second_pair.get(0).unwrap()).unwrap(),
        "b"
    );
    assert_eq!(second_pair.get(1).unwrap().as_int(), Some(5));

    unsafe {
        drop(Arc::from_raw(class_ptr as *const PyClassObject));
    }
}
