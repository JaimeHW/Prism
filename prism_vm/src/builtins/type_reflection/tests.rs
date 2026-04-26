use super::*;
use crate::builtins::builtin_type_object_for_type_id;
use prism_core::intern::intern;
use prism_runtime::object::PyObject;
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, register_global_class, unregister_global_class,
};
use prism_runtime::object::views::{DescriptorViewObject, MappingProxyObject, MethodWrapperObject};
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::tuple::TupleObject;
use std::sync::Arc;

#[test]
fn test_builtin_type_attr_registry_covers_types_module_surface() {
    assert!(builtin_type_has_attribute(
        TypeId::TYPE,
        &intern("__dict__")
    ));
    assert!(builtin_type_has_attribute(TypeId::TYPE, &intern("__doc__")));
    assert!(builtin_type_has_attribute(
        TypeId::DICT,
        &intern("__name__")
    ));
    assert!(builtin_type_has_attribute(
        TypeId::DICT,
        &intern("__bases__")
    ));
    assert!(builtin_type_has_attribute(
        TypeId::GENERATOR,
        &intern("__mro__")
    ));
    assert!(builtin_type_has_attribute(
        TypeId::OBJECT,
        &intern("__init__")
    ));
    assert!(builtin_type_has_attribute(
        TypeId::OBJECT,
        &intern("__setattr__")
    ));
    assert!(builtin_type_has_attribute(
        TypeId::OBJECT,
        &intern("__delattr__")
    ));
    assert!(builtin_type_has_attribute(TypeId::TYPE, &intern("__new__")));
    assert!(builtin_type_has_attribute(
        TypeId::OBJECT,
        &intern("__new__")
    ));
    assert!(builtin_type_has_attribute(TypeId::INT, &intern("__new__")));
    assert!(builtin_type_has_attribute(
        TypeId::FLOAT,
        &intern("__new__")
    ));
    assert!(builtin_type_has_attribute(TypeId::STR, &intern("__new__")));
    assert!(builtin_type_has_attribute(TypeId::BOOL, &intern("__new__")));
    assert!(builtin_type_has_attribute(
        TypeId::BYTES,
        &intern("__new__")
    ));
    assert!(builtin_type_has_attribute(
        TypeId::BYTEARRAY,
        &intern("__new__")
    ));
    assert!(builtin_type_has_attribute(TypeId::LIST, &intern("__new__")));
    assert!(builtin_type_has_attribute(
        TypeId::TUPLE,
        &intern("__new__")
    ));
    assert!(builtin_type_has_attribute(TypeId::DICT, &intern("__new__")));
    assert!(builtin_type_has_attribute(TypeId::SET, &intern("__new__")));
    assert!(builtin_type_has_attribute(
        TypeId::FROZENSET,
        &intern("__new__")
    ));
    assert!(builtin_type_has_attribute(TypeId::STR, &intern("join")));
    assert!(builtin_type_has_attribute(TypeId::STR, &intern("replace")));
    assert!(builtin_type_has_attribute(
        TypeId::STR,
        &intern("maketrans")
    ));
    assert!(builtin_type_has_attribute(
        TypeId::DICT,
        &intern("fromkeys")
    ));
    assert!(builtin_type_has_attribute(
        TypeId::DICT,
        &intern("__setitem__")
    ));
    assert!(builtin_type_has_attribute(TypeId::DICT, &intern("pop")));
    assert!(builtin_type_has_attribute(
        TypeId::INT,
        &intern("bit_length")
    ));
    assert!(builtin_type_has_attribute(TypeId::INT, &intern("__add__")));
    assert!(builtin_type_has_attribute(
        TypeId::INT,
        &intern("__index__")
    ));
    assert!(builtin_type_has_attribute(
        TypeId::INT,
        &intern("bit_count")
    ));
    assert!(builtin_type_has_attribute(
        TypeId::FUNCTION,
        &intern("__code__")
    ));
    assert!(builtin_type_has_attribute(
        TypeId::FUNCTION,
        &intern("__globals__")
    ));
}

#[test]
fn test_builtin_instance_attr_registry_covers_method_wrapper_surface() {
    assert!(builtin_instance_has_attribute(
        TypeId::NONE,
        &intern("__doc__")
    ));
    assert!(builtin_instance_has_attribute(
        TypeId::LIST,
        &intern("__class__")
    ));
    assert!(builtin_instance_has_attribute(
        TypeId::OBJECT,
        &intern("__str__")
    ));
    assert!(!builtin_instance_has_attribute(
        TypeId::OBJECT,
        &intern("__repr__")
    ));
}

#[test]
fn test_builtin_instance_attribute_value_exposes_builtin_class() {
    let mut vm = VirtualMachine::new();
    let list_ptr = Box::into_raw(Box::new(ListObject::new()));
    let list_value = Value::object_ptr(list_ptr as *const ());

    let class =
        builtin_instance_attribute_value(&mut vm, TypeId::LIST, list_value, &intern("__class__"))
            .expect("list.__class__ lookup should succeed")
            .expect("list instances should expose __class__");

    assert_eq!(class, builtin_type_object_for_type_id(TypeId::LIST));

    unsafe {
        drop(Box::from_raw(list_ptr));
    }
}

#[test]
fn test_builtin_instance_attribute_value_exposes_none_doc() {
    let mut vm = VirtualMachine::new();
    let doc =
        builtin_instance_attribute_value(&mut vm, TypeId::NONE, Value::none(), &intern("__doc__"))
            .expect("None.__doc__ lookup should succeed")
            .expect("None.__doc__ should exist");

    assert_eq!(doc, Value::string(intern(NONE_TYPE_DOC)));
}

#[test]
fn test_builtin_instance_attribute_value_exposes_primitive_type_doc() {
    let mut vm = VirtualMachine::new();
    let doc = builtin_instance_attribute_value(
        &mut vm,
        TypeId::STR,
        Value::string(intern("seed")),
        &intern("__doc__"),
    )
    .expect("str.__doc__ lookup should succeed")
    .expect("str instances should inherit type documentation");

    assert_eq!(doc, Value::string(intern("Text string type.")));
}

#[test]
fn test_builtin_mapping_proxy_names_include_iterator_protocol_methods() {
    let names = builtin_mapping_proxy_names(TypeId::ITERATOR);
    assert!(names.contains(&intern("__iter__")));
    assert!(names.contains(&intern("__next__")));
}

#[test]
fn test_builtin_mapping_proxy_names_include_tuple_sequence_methods() {
    let names = builtin_mapping_proxy_names(TypeId::TUPLE);
    assert!(names.contains(&intern("__iter__")));
    assert!(names.contains(&intern("__len__")));
    assert!(names.contains(&intern("__getitem__")));
    assert!(names.contains(&intern("count")));
    assert!(names.contains(&intern("index")));
}

#[test]
fn test_builtin_mapping_proxy_names_include_regex_match_subscription() {
    let names = builtin_mapping_proxy_names(TypeId::REGEX_MATCH);
    assert!(names.contains(&intern("__getitem__")));
}

#[test]
fn test_builtin_mapping_proxy_names_include_int_bit_operations() {
    let names = builtin_mapping_proxy_names(TypeId::INT);
    assert!(names.contains(&intern("__add__")));
    assert!(names.contains(&intern("__index__")));
    assert!(names.contains(&intern("bit_length")));
    assert!(names.contains(&intern("bit_count")));
}

#[test]
fn test_mapping_proxy_source_round_trip() {
    let proxy = MappingProxyObject::for_builtin_type(TypeId::DICT);
    assert_eq!(
        proxy.source(),
        prism_runtime::object::views::MappingProxySource::BuiltinType(TypeId::DICT)
    );
}

#[test]
fn test_mapping_proxy_supports_heap_class_source() {
    let class = Arc::new(PyClassObject::new_simple(intern("ProxyClass")));
    let proxy = MappingProxyObject::for_user_class(Arc::as_ptr(&class));
    assert_eq!(
        proxy.source(),
        prism_runtime::object::views::MappingProxySource::UserClass(Arc::as_ptr(&class) as usize)
    );
}

#[test]
fn test_mapping_proxy_entry_helpers_cover_heap_class_contents() {
    let class = Arc::new(PyClassObject::new_simple(intern("ProxyEntries")));
    class.set_attr(intern("token"), Value::int(7).unwrap());
    class.set_attr(intern("label"), Value::string(intern("ready")));
    let proxy = MappingProxyObject::for_user_class(Arc::as_ptr(&class));

    let mut keys = builtin_mapping_proxy_keys(&proxy)
        .expect("keys should materialize")
        .into_iter()
        .map(|value| {
            let ptr = value
                .as_string_object_ptr()
                .expect("mappingproxy keys should be interned strings");
            interned_by_ptr(ptr as *const u8)
                .expect("interned string pointer should resolve")
                .as_str()
                .to_string()
        })
        .collect::<Vec<_>>();
    keys.sort();
    assert_eq!(keys, vec!["label".to_string(), "token".to_string()]);

    let entries = builtin_mapping_proxy_entries_static(&proxy).expect("entries should exist");
    assert_eq!(entries.len(), 2);
    assert_eq!(
        builtin_mapping_proxy_len(&proxy).expect("len should succeed"),
        2
    );
}

#[test]
fn test_builtin_mapping_proxy_exposes_core_new_descriptors() {
    for owner in [
        TypeId::TYPE,
        TypeId::OBJECT,
        TypeId::INT,
        TypeId::FLOAT,
        TypeId::STR,
        TypeId::BOOL,
        TypeId::LIST,
        TypeId::TUPLE,
        TypeId::DICT,
        TypeId::SET,
        TypeId::FROZENSET,
    ] {
        let proxy = MappingProxyObject::for_builtin_type(owner);
        assert!(
            builtin_mapping_proxy_contains_key(&proxy, Value::string(intern("__new__")))
                .expect("membership should succeed"),
            "{owner:?}.__dict__ should expose __new__"
        );

        let value = builtin_mapping_proxy_get_item_static(&proxy, Value::string(intern("__new__")))
            .expect("subscript should succeed")
            .expect("__new__ should exist");
        let ptr = value
            .as_object_ptr()
            .expect("__new__ should materialize as a descriptor");
        let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
        assert_eq!(header.type_id, TypeId::WRAPPER_DESCRIPTOR);
    }
}

#[test]
fn test_descriptor_view_uses_expected_runtime_type_ids() {
    let method = DescriptorViewObject::new(TypeId::METHOD_DESCRIPTOR, TypeId::STR, intern("join"));
    let classmethod = DescriptorViewObject::new(
        TypeId::CLASSMETHOD_DESCRIPTOR,
        TypeId::DICT,
        intern("fromkeys"),
    );
    assert_eq!(method.header().type_id, TypeId::METHOD_DESCRIPTOR);
    assert_eq!(classmethod.header().type_id, TypeId::CLASSMETHOD_DESCRIPTOR);
}

#[test]
fn test_mapping_proxy_entries_support_dict_backing() {
    let mut dict = DictObject::new();
    dict.set(Value::string(intern("alpha")), Value::int(1).unwrap());
    dict.set(Value::string(intern("beta")), Value::int(2).unwrap());
    let dict_value = leak_object_value(dict);
    let proxy = MappingProxyObject::for_mapping(dict_value);

    let entries = builtin_mapping_proxy_entries_static(&proxy)
        .expect("dict-backed mappingproxy should expose entries");
    assert_eq!(
        entries,
        vec![
            (Value::string(intern("alpha")), Value::int(1).unwrap()),
            (Value::string(intern("beta")), Value::int(2).unwrap()),
        ]
    );
    assert!(
        builtin_mapping_proxy_contains_key(&proxy, Value::string(intern("alpha")))
            .expect("contains should succeed")
    );
    assert_eq!(
        builtin_mapping_proxy_get_item_static(&proxy, Value::string(intern("beta")))
            .expect("lookup should succeed"),
        Some(Value::int(2).unwrap())
    );
}

#[test]
fn test_method_wrapper_uses_expected_runtime_type_id() {
    let wrapper = MethodWrapperObject::new(
        TypeId::OBJECT,
        intern("__str__"),
        builtin_type_object_for_type_id(TypeId::OBJECT),
    );
    assert_eq!(wrapper.header().type_id, TypeId::METHOD_WRAPPER);
}

#[test]
fn test_builtin_type_method_value_exposes_dict_fromkeys_callable() {
    let value =
        builtin_type_method_value(TypeId::DICT, "fromkeys").expect("dict.fromkeys should resolve");
    let ptr = value
        .as_object_ptr()
        .expect("method should be heap allocated");
    let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
    assert_eq!(header.type_id, TypeId::BUILTIN_FUNCTION);
}

#[test]
fn test_builtin_type_method_value_exposes_str_maketrans_callable() {
    let value =
        builtin_type_method_value(TypeId::STR, "maketrans").expect("str.maketrans should resolve");
    let ptr = value
        .as_object_ptr()
        .expect("method should be heap allocated");
    let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
    assert_eq!(header.type_id, TypeId::BUILTIN_FUNCTION);
}

#[test]
fn test_builtin_type_method_value_exposes_bytes_maketrans_callable() {
    for owner in [TypeId::BYTES, TypeId::BYTEARRAY] {
        let value = builtin_type_method_value(owner, "maketrans")
            .expect("bytes-like maketrans should resolve");
        let ptr = value
            .as_object_ptr()
            .expect("method should be heap allocated");
        let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
        assert_eq!(header.type_id, TypeId::BUILTIN_FUNCTION);
    }
}

#[test]
fn test_builtin_type_method_value_exposes_float_getformat_callable() {
    let value = builtin_type_method_value(TypeId::FLOAT, "__getformat__")
        .expect("float.__getformat__ should resolve");
    let ptr = value
        .as_object_ptr()
        .expect("method should be heap allocated");
    let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
    assert_eq!(header.type_id, TypeId::BUILTIN_FUNCTION);
}

#[test]
fn test_builtin_type_method_value_exposes_unbound_dict_setitem_callable() {
    let value = builtin_type_method_value(TypeId::DICT, "__setitem__")
        .expect("dict.__setitem__ should resolve");
    let ptr = value
        .as_object_ptr()
        .expect("method should be heap allocated");
    let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
    assert_eq!(header.type_id, TypeId::BUILTIN_FUNCTION);
}

#[test]
fn test_builtin_type_method_value_exposes_core_new_callables() {
    for owner in [
        TypeId::TYPE,
        TypeId::OBJECT,
        TypeId::INT,
        TypeId::FLOAT,
        TypeId::STR,
        TypeId::BOOL,
        TypeId::LIST,
        TypeId::TUPLE,
        TypeId::DICT,
        TypeId::SET,
        TypeId::FROZENSET,
    ] {
        let value = builtin_type_method_value(owner, "__new__")
            .unwrap_or_else(|| panic!("{owner:?}.__new__ should resolve"));
        let ptr = value
            .as_object_ptr()
            .expect("method should be heap allocated");
        let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
        assert_eq!(header.type_id, TypeId::BUILTIN_FUNCTION);
    }
}

#[test]
fn test_reflected_descriptor_callable_value_exposes_object_init_builtin() {
    let value = reflected_descriptor_callable_value(
        TypeId::WRAPPER_DESCRIPTOR,
        TypeId::OBJECT,
        &intern("__init__"),
    )
    .expect("object.__init__ descriptor should resolve to a callable");
    let ptr = value
        .as_object_ptr()
        .expect("callable should be heap allocated");
    let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
    assert_eq!(builtin.name(), "object.__init__");
}

#[test]
fn test_reflected_descriptor_callable_value_exposes_type_init_builtin() {
    let value = reflected_descriptor_callable_value(
        TypeId::WRAPPER_DESCRIPTOR,
        TypeId::TYPE,
        &intern("__init__"),
    )
    .expect("type.__init__ descriptor should resolve to a callable");
    let ptr = value
        .as_object_ptr()
        .expect("callable should be heap allocated");
    let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
    assert_eq!(builtin.name(), "type.__init__");
    assert!(
        builtin
            .call(&[builtin_type_object_for_type_id(TypeId::TYPE)])
            .expect("type.__init__ should accept an already-initialized type")
            .is_none()
    );
}

#[test]
fn test_reflected_descriptor_callable_value_exposes_int_add_builtin() {
    let value = reflected_descriptor_callable_value(
        TypeId::WRAPPER_DESCRIPTOR,
        TypeId::INT,
        &intern("__add__"),
    )
    .expect("int.__add__ descriptor should resolve to a callable");
    let ptr = value
        .as_object_ptr()
        .expect("callable should be heap allocated");
    let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
    assert_eq!(builtin.name(), "int.__add__");
}

#[test]
fn test_reflected_descriptor_callable_value_exposes_object_init_subclass_builtin() {
    let value = reflected_descriptor_callable_value(
        TypeId::CLASSMETHOD_DESCRIPTOR,
        TypeId::OBJECT,
        &intern("__init_subclass__"),
    )
    .expect("object.__init_subclass__ descriptor should resolve to a callable");
    let ptr = value
        .as_object_ptr()
        .expect("callable should be heap allocated");
    let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
    assert_eq!(builtin.name(), "object.__init_subclass__");
}

#[test]
fn test_reflected_descriptor_callable_value_exposes_type_prepare_builtin() {
    let value = reflected_descriptor_callable_value(
        TypeId::CLASSMETHOD_DESCRIPTOR,
        TypeId::TYPE,
        &intern("__prepare__"),
    )
    .expect("type.__prepare__ descriptor should resolve to a callable");
    let ptr = value
        .as_object_ptr()
        .expect("callable should be heap allocated");
    let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
    assert_eq!(builtin.name(), "type.__prepare__");
}

#[test]
fn test_builtin_bound_type_attribute_value_binds_type_prepare_receiver() {
    let mut vm = VirtualMachine::new();
    let type_type = builtin_type_object_for_type_id(TypeId::TYPE);
    let method = builtin_bound_type_attribute_value(
        &mut vm,
        TypeId::TYPE,
        type_type,
        &intern("__prepare__"),
    )
    .expect("binding should succeed")
    .expect("bound method should exist");

    let method_ptr = method
        .as_object_ptr()
        .expect("bound method should be allocated");
    let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
    let bases_ptr = Box::into_raw(Box::new(TupleObject::empty()));
    let result = builtin
        .call_with_keywords(
            &[
                Value::string(intern("Prepared")),
                Value::object_ptr(bases_ptr as *const ()),
            ],
            &[("flag", Value::bool(true))],
        )
        .expect("bound type.__prepare__ should be callable");
    let result_ptr = result.as_object_ptr().expect("result should be a dict");
    assert_eq!(
        unsafe { &*(result_ptr as *const prism_runtime::object::ObjectHeader) }.type_id,
        TypeId::DICT
    );

    unsafe {
        drop(Box::from_raw(bases_ptr));
    }
}

#[test]
fn test_builtin_bound_type_attribute_value_binds_dict_fromkeys_receiver() {
    let mut vm = VirtualMachine::new();
    let dict_type = builtin_type_object_for_type_id(TypeId::DICT);
    let method =
        builtin_bound_type_attribute_value(&mut vm, TypeId::DICT, dict_type, &intern("fromkeys"))
            .expect("binding should succeed")
            .expect("bound method should exist");

    let method_ptr = method
        .as_object_ptr()
        .expect("bound method should be allocated");
    let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
    let keys_ptr = Box::into_raw(Box::new(TupleObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
    ])));
    let result = builtin
        .call(&[Value::object_ptr(keys_ptr as *const ())])
        .expect("bound dict.fromkeys should be callable");
    let result_ptr = result.as_object_ptr().expect("result should be a dict");
    let dict = unsafe { &*(result_ptr as *const DictObject) };
    assert!(dict.get(Value::int(1).unwrap()).unwrap().is_none());
    assert!(dict.get(Value::int(2).unwrap()).unwrap().is_none());

    unsafe {
        drop(Box::from_raw(result_ptr as *mut DictObject));
        drop(Box::from_raw(keys_ptr));
    }
}

#[test]
fn test_builtin_bound_type_attribute_value_binds_float_getformat_receiver() {
    let mut vm = VirtualMachine::new();
    let float_type = builtin_type_object_for_type_id(TypeId::FLOAT);
    let method = builtin_bound_type_attribute_value(
        &mut vm,
        TypeId::FLOAT,
        float_type,
        &intern("__getformat__"),
    )
    .expect("binding should succeed")
    .expect("bound method should exist");

    let method_ptr = method
        .as_object_ptr()
        .expect("bound method should be allocated");
    let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
    let result = builtin
        .call(&[Value::string(intern("double"))])
        .expect("bound float.__getformat__ should be callable");
    assert!(result.is_string());
}

#[test]
fn test_builtin_bound_type_attribute_value_returns_unbound_dict_setitem() {
    let mut vm = VirtualMachine::new();
    let dict_type = builtin_type_object_for_type_id(TypeId::DICT);
    let method = builtin_bound_type_attribute_value(
        &mut vm,
        TypeId::DICT,
        dict_type,
        &intern("__setitem__"),
    )
    .expect("lookup should succeed")
    .expect("method should exist");

    let method_ptr = method
        .as_object_ptr()
        .expect("method should be heap allocated");
    let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };

    let dict_ptr = Box::into_raw(Box::new(DictObject::new()));
    let dict_value = Value::object_ptr(dict_ptr as *const ());
    let key = Value::string(intern("ready"));
    builtin
        .call(&[dict_value, key, Value::int(1).unwrap()])
        .expect("dict.__setitem__ should accept an explicit receiver");

    let dict = unsafe { &*(dict_ptr as *const DictObject) };
    assert_eq!(dict.get(key).unwrap().as_int(), Some(1));

    unsafe {
        drop(Box::from_raw(dict_ptr));
    }
}

#[test]
fn test_builtin_bound_type_attribute_value_binds_reflected_object_init_for_instances() {
    let mut vm = VirtualMachine::new();
    let instance = crate::builtins::builtin_object(&[]).expect("object() should succeed");
    let method =
        builtin_bound_type_attribute_value(&mut vm, TypeId::OBJECT, instance, &intern("__init__"))
            .expect("lookup should succeed")
            .expect("object.__init__ should exist");

    let method_ptr = method
        .as_object_ptr()
        .expect("bound method should be heap allocated");
    let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
    assert_eq!(builtin.name(), "object.__init__");
    assert_eq!(builtin.bound_self(), Some(instance));
    assert!(
        builtin
            .call(&[])
            .expect("bound object.__init__ should execute")
            .is_none()
    );
}

#[test]
fn test_builtin_bound_type_attribute_value_binds_object_init_subclass_for_types() {
    let mut vm = VirtualMachine::new();
    let object_type = builtin_type_object_for_type_id(TypeId::OBJECT);
    let method = builtin_bound_type_attribute_value(
        &mut vm,
        TypeId::OBJECT,
        object_type,
        &intern("__init_subclass__"),
    )
    .expect("lookup should succeed")
    .expect("object.__init_subclass__ should exist");

    let method_ptr = method
        .as_object_ptr()
        .expect("bound method should be heap allocated");
    let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
    assert_eq!(builtin.name(), "object.__init_subclass__");
    assert_eq!(builtin.bound_self(), Some(object_type));
    assert!(
        builtin
            .call_with_keywords(&[], &[])
            .expect("bound object.__init_subclass__ should execute")
            .is_none()
    );
}

#[test]
fn test_builtin_bound_type_attribute_value_static_binds_reflected_object_init_for_instances() {
    let instance =
        crate::builtins::builtin_object(&[]).expect("object() should produce an instance");
    let method =
        builtin_bound_type_attribute_value_static(TypeId::OBJECT, instance, &intern("__init__"))
            .expect("lookup should succeed")
            .expect("object.__init__ should exist");

    let method_ptr = method
        .as_object_ptr()
        .expect("bound method should be heap allocated");
    let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
    assert_eq!(builtin.name(), "object.__init__");
    assert_eq!(builtin.bound_self(), Some(instance));
    assert!(
        builtin
            .call(&[])
            .expect("bound object.__init__ should execute")
            .is_none()
    );
}

#[test]
fn test_builtin_bound_type_attribute_value_static_binds_reflected_object_eq_for_primitives() {
    let receiver = Value::int(7).unwrap();
    let method =
        builtin_bound_type_attribute_value_static(TypeId::OBJECT, receiver, &intern("__eq__"))
            .expect("lookup should succeed")
            .expect("object.__eq__ should exist");

    let method_ptr = method
        .as_object_ptr()
        .expect("bound method should be heap allocated");
    let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
    assert_eq!(builtin.name(), "object.__eq__");
    assert_eq!(builtin.bound_self(), Some(receiver));
    assert_eq!(
        builtin
            .call(&[Value::int(7).unwrap()])
            .expect("bound object.__eq__ should execute"),
        Value::bool(true)
    );
}

#[test]
fn test_builtin_bound_type_attribute_value_materializes_doc_slot() {
    let mut vm = VirtualMachine::new();
    let type_object = builtin_type_object_for_type_id(TypeId::TYPE);
    let doc =
        builtin_bound_type_attribute_value(&mut vm, TypeId::TYPE, type_object, &intern("__doc__"))
            .expect("lookup should succeed");

    assert_eq!(
        doc,
        Some(Value::string(intern(
            "Create a new type, or return the type of an object."
        )))
    );
}

#[test]
fn test_builtin_type_attribute_value_materializes_none_doc_slot() {
    let mut vm = VirtualMachine::new();
    let doc = builtin_type_attribute_value(&mut vm, TypeId::NONE, &intern("__doc__"))
        .expect("type(None).__doc__ lookup should succeed")
        .expect("type(None).__doc__ should exist");

    assert_eq!(doc, Value::string(intern(NONE_TYPE_DOC)));
}

#[test]
fn test_builtin_type_attribute_value_materializes_type_init_descriptor() {
    let mut vm = VirtualMachine::new();
    let descriptor = builtin_type_attribute_value(&mut vm, TypeId::TYPE, &intern("__init__"))
        .expect("type.__init__ lookup should succeed")
        .expect("type.__init__ should exist");
    let descriptor_ptr = descriptor
        .as_object_ptr()
        .expect("type.__init__ should be heap allocated");
    let header = unsafe { &*(descriptor_ptr as *const prism_runtime::object::ObjectHeader) };
    assert_eq!(header.type_id, TypeId::WRAPPER_DESCRIPTOR);

    let view = unsafe { &*(descriptor_ptr as *const DescriptorViewObject) };
    assert_eq!(view.owner(), TypeId::TYPE);
    assert_eq!(view.name().as_str(), "__init__");
}

#[test]
fn test_builtin_type_attribute_value_materializes_mro_tuple() {
    let mut vm = VirtualMachine::new();
    let value = builtin_type_attribute_value(&mut vm, TypeId::BOOL, &intern("__mro__"))
        .expect("lookup should succeed")
        .expect("__mro__ should exist");
    let tuple_ptr = value.as_object_ptr().expect("mro should be a tuple object");
    let tuple = unsafe { &*(tuple_ptr as *const TupleObject) };
    assert_eq!(tuple.len(), 3);
    assert_eq!(
        tuple.as_slice()[0],
        builtin_type_object_for_type_id(TypeId::BOOL)
    );
    assert_eq!(
        tuple.as_slice()[1],
        builtin_type_object_for_type_id(TypeId::INT)
    );
    assert_eq!(
        tuple.as_slice()[2],
        builtin_type_object_for_type_id(TypeId::OBJECT)
    );
}

#[test]
fn test_builtin_type_attribute_value_materializes_name_and_bases() {
    let mut vm = VirtualMachine::new();

    let name_value = builtin_type_attribute_value(&mut vm, TypeId::DICT, &intern("__name__"))
        .expect("lookup should succeed")
        .expect("__name__ should exist");
    let name_ptr = name_value
        .as_string_object_ptr()
        .expect("__name__ should be an interned string");
    assert_eq!(
        interned_by_ptr(name_ptr as *const u8).unwrap().as_str(),
        "dict"
    );

    let bases_value = builtin_type_attribute_value(&mut vm, TypeId::BOOL, &intern("__bases__"))
        .expect("lookup should succeed")
        .expect("__bases__ should exist");
    let bases_ptr = bases_value
        .as_object_ptr()
        .expect("__bases__ should be a tuple object");
    let bases = unsafe { &*(bases_ptr as *const TupleObject) };
    assert_eq!(bases.len(), 1);
    assert_eq!(
        bases.as_slice()[0],
        builtin_type_object_for_type_id(TypeId::INT)
    );
}

#[test]
fn test_builtin_type_has_attribute_reports_type_prepare() {
    assert!(builtin_type_has_attribute(
        TypeId::TYPE,
        &intern("__init__")
    ));
    assert!(builtin_type_has_attribute(
        TypeId::TYPE,
        &intern("__prepare__")
    ));
}

#[test]
fn test_heap_type_attribute_value_materializes_mro_tuple_and_dict_proxy() {
    let class = Arc::new(PyClassObject::new_simple(intern("HeapReflect")));
    class.set_attr(intern("token"), Value::int(7).unwrap());

    let mut bitmap = SubclassBitmap::new();
    bitmap.set_bit(TypeId::OBJECT);
    bitmap.set_bit(class.class_type_id());
    register_global_class(Arc::clone(&class), bitmap);

    let class_ptr = Arc::as_ptr(&class);
    let mut vm = VirtualMachine::new();

    let mro_value = heap_type_attribute_value(&mut vm, class_ptr, &intern("__mro__"))
        .expect("lookup should succeed")
        .expect("__mro__ should exist");
    let mro_ptr = mro_value.as_object_ptr().expect("mro should be a tuple");
    let mro = unsafe { &*(mro_ptr as *const TupleObject) };
    assert_eq!(mro.len(), 2);
    assert_eq!(mro.as_slice()[0], Value::object_ptr(class_ptr as *const ()));
    assert_eq!(
        mro.as_slice()[1],
        builtin_type_object_for_type_id(TypeId::OBJECT)
    );

    let dict_value = heap_type_attribute_value(&mut vm, class_ptr, &intern("__dict__"))
        .expect("lookup should succeed")
        .expect("__dict__ should exist");
    let dict_ptr = dict_value
        .as_object_ptr()
        .expect("__dict__ should be a proxy");
    let proxy = unsafe { &*(dict_ptr as *const MappingProxyObject) };
    let token = builtin_mapping_proxy_get_item(&mut vm, proxy, Value::string(intern("token")))
        .expect("subscript should succeed")
        .expect("token should exist");
    assert_eq!(token.as_int(), Some(7));
    let token_static = builtin_mapping_proxy_get_item_static(proxy, Value::string(intern("token")))
        .expect("static subscript should succeed")
        .expect("token should exist");
    assert_eq!(token_static.as_int(), Some(7));
    assert!(
        builtin_mapping_proxy_contains_key(proxy, Value::string(intern("token")))
            .expect("membership should succeed")
    );
}

#[test]
fn test_heap_type_attribute_value_materializes_mro_tuple_when_owner_registry_entry_is_absent() {
    let class = Arc::new(PyClassObject::new_simple(intern("HeapReflectDetached")));

    let mut bitmap = SubclassBitmap::new();
    bitmap.set_bit(TypeId::OBJECT);
    bitmap.set_bit(class.class_type_id());
    register_global_class(Arc::clone(&class), bitmap);
    unregister_global_class(class.class_id());

    let class_ptr = Arc::as_ptr(&class);
    let mut vm = VirtualMachine::new();

    let mro_value = heap_type_attribute_value(&mut vm, class_ptr, &intern("__mro__"))
        .expect("lookup should succeed")
        .expect("__mro__ should exist");
    let mro_ptr = mro_value.as_object_ptr().expect("mro should be a tuple");
    let mro = unsafe { &*(mro_ptr as *const TupleObject) };
    assert_eq!(mro.len(), 2);
    assert_eq!(mro.as_slice()[0], Value::object_ptr(class_ptr as *const ()));
    assert_eq!(
        mro.as_slice()[1],
        builtin_type_object_for_type_id(TypeId::OBJECT)
    );
}

#[test]
fn test_heap_type_attribute_value_inherits_builtin_object_ne() {
    let class = Arc::new(PyClassObject::new_simple(intern("HeapComparable")));

    let mut bitmap = SubclassBitmap::new();
    bitmap.set_bit(TypeId::OBJECT);
    bitmap.set_bit(class.class_type_id());
    register_global_class(Arc::clone(&class), bitmap);

    let class_ptr = Arc::as_ptr(&class);
    let mut vm = VirtualMachine::new();
    let value = heap_type_attribute_value(&mut vm, class_ptr, &intern("__ne__"))
        .expect("lookup should succeed")
        .expect("__ne__ should resolve from object");

    let ptr = value
        .as_object_ptr()
        .expect("object.__ne__ should be materialized as a builtin");
    let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
    assert_eq!(builtin.name(), "object.__ne__");
}

#[test]
fn test_heap_type_attribute_value_materializes_type_metadata() {
    let class = Arc::new(PyClassObject::new_simple(intern("HeapMetadata")));
    class.set_attr(intern("__doc__"), Value::string(intern("heap docs")));
    class.set_attr(intern("__module__"), Value::string(intern("pkg.runtime")));
    class.set_attr(
        intern("__qualname__"),
        Value::string(intern("pkg.runtime.HeapMetadata")),
    );

    let mut bitmap = SubclassBitmap::new();
    bitmap.set_bit(TypeId::OBJECT);
    bitmap.set_bit(class.class_type_id());
    register_global_class(Arc::clone(&class), bitmap);

    let class_ptr = Arc::as_ptr(&class);
    let mut vm = VirtualMachine::new();

    let name = heap_type_attribute_value(&mut vm, class_ptr, &intern("__name__"))
        .expect("lookup should succeed")
        .expect("__name__ should exist");
    let name_ptr = name
        .as_string_object_ptr()
        .expect("__name__ should be an interned string");
    assert_eq!(
        interned_by_ptr(name_ptr as *const u8).unwrap().as_str(),
        "HeapMetadata"
    );

    let module = heap_type_attribute_value(&mut vm, class_ptr, &intern("__module__"))
        .expect("lookup should succeed")
        .expect("__module__ should exist");
    let module_ptr = module
        .as_string_object_ptr()
        .expect("__module__ should be an interned string");
    assert_eq!(
        interned_by_ptr(module_ptr as *const u8).unwrap().as_str(),
        "pkg.runtime"
    );

    let doc = heap_type_attribute_value(&mut vm, class_ptr, &intern("__doc__"))
        .expect("lookup should succeed")
        .expect("__doc__ should exist");
    let doc_ptr = doc
        .as_string_object_ptr()
        .expect("__doc__ should be an interned string");
    assert_eq!(
        interned_by_ptr(doc_ptr as *const u8).unwrap().as_str(),
        "heap docs"
    );

    let bases = heap_type_attribute_value(&mut vm, class_ptr, &intern("__bases__"))
        .expect("lookup should succeed")
        .expect("__bases__ should exist");
    let bases_ptr = bases
        .as_object_ptr()
        .expect("__bases__ should be a tuple object");
    let bases = unsafe { &*(bases_ptr as *const TupleObject) };
    assert_eq!(bases.len(), 1);
    assert_eq!(
        bases.as_slice()[0],
        builtin_type_object_for_type_id(TypeId::OBJECT)
    );
}

#[test]
fn test_heap_type_attribute_value_defaults_doc_to_none() {
    let class = Arc::new(PyClassObject::new_simple(intern("HeapMetadataNoDoc")));

    let mut bitmap = SubclassBitmap::new();
    bitmap.set_bit(TypeId::OBJECT);
    bitmap.set_bit(class.class_type_id());
    register_global_class(Arc::clone(&class), bitmap);

    let class_ptr = Arc::as_ptr(&class);
    let mut vm = VirtualMachine::new();

    let doc = heap_type_attribute_value(&mut vm, class_ptr, &intern("__doc__"))
        .expect("lookup should succeed")
        .expect("__doc__ should exist");
    assert!(doc.is_none());
}
