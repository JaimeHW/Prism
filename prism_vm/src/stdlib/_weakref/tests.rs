use super::*;
use prism_code::CodeObject;

fn builtin_from_value(value: Value) -> &'static BuiltinFunctionObject {
    let ptr = value
        .as_object_ptr()
        .expect("expected builtin function object");
    unsafe { &*(ptr as *const BuiltinFunctionObject) }
}

#[test]
fn test_module_exposes_bootstrap_attributes() {
    let module = WeakRefModule::new();

    assert!(module.get_attr("proxy").unwrap().as_object_ptr().is_some());
    assert!(module.get_attr("ref").unwrap().as_object_ptr().is_some());
    assert!(
        module
            .get_attr("ReferenceType")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );
}

#[test]
fn test_proxy_returns_original_object() {
    let module = WeakRefModule::new();
    let proxy = builtin_from_value(module.get_attr("proxy").expect("proxy should exist"));
    let original = Value::string(intern("ordered-dict-root"));

    assert_eq!(proxy.call(&[original]).unwrap(), original);
}

#[test]
fn test_reference_type_creates_callable_reference_instances() {
    let class_ptr = reference_type_value()
        .as_object_ptr()
        .expect("ReferenceType should be a class object");
    let class = unsafe { &*(class_ptr as *const PyClassObject) };
    let new_builtin = builtin_from_value(
        class
            .get_attr(&intern("__new__"))
            .expect("ReferenceType.__new__ should exist"),
    );
    let call_builtin = builtin_from_value(
        class
            .get_attr(&intern("__call__"))
            .expect("ReferenceType.__call__ should exist"),
    );
    let target = Value::string(intern("cached-module"));

    let instance = new_builtin
        .call(&[reference_type_value(), target])
        .expect("ReferenceType.__new__ should succeed");
    let recalled = call_builtin
        .call(&[instance])
        .expect("ReferenceType instances should be callable");

    assert_eq!(recalled, target);
}

#[test]
fn test_reference_type_is_registered_for_subclass_creation() {
    let base_ptr = reference_type_value()
        .as_object_ptr()
        .expect("ReferenceType should be a class object");
    let base_class = unsafe { &*(base_ptr as *const PyClassObject) };
    let namespace = ClassDict::new();
    let result = type_new(
        intern("KeyedRef"),
        &[base_class.class_id()],
        &namespace,
        global_class_registry(),
    )
    .expect("ReferenceType should support subclass creation");
    register_global_class(result.class.clone(), result.bitmap);

    let subclass_value = Value::object_ptr(Arc::as_ptr(&result.class) as *const ());
    let target = Value::string(intern("bootstrap-entry"));
    let instance = reference_new(&[subclass_value, target])
        .expect("ReferenceType.__new__ should support registered subclasses");
    let instance_ptr = instance
        .as_object_ptr()
        .expect("subclass instance should be heap allocated");

    assert_eq!(
        crate::ops::objects::extract_type_id(instance_ptr),
        result.class.class_type_id()
    );
}

#[test]
fn test_getweakrefs_returns_empty_list() {
    let module = WeakRefModule::new();
    let getweakrefs = builtin_from_value(
        module
            .get_attr("getweakrefs")
            .expect("getweakrefs should exist"),
    );
    let value = getweakrefs.call(&[Value::string(intern("probe"))]).unwrap();
    let ptr = value
        .as_object_ptr()
        .expect("list should be heap allocated");
    let list = unsafe { &*(ptr as *const ListObject) };
    assert!(list.is_empty());
}

#[test]
fn test_remove_dead_weakref_removes_mapping_entry() {
    let module = WeakRefModule::new();
    let remove = builtin_from_value(
        module
            .get_attr("_remove_dead_weakref")
            .expect("remove helper should exist"),
    );

    let mut dict = DictObject::new();
    let key = Value::string(intern("dead"));
    dict.set(key, Value::int(1).unwrap());
    let dict_value = leak_object_value(dict);

    remove.call(&[dict_value, key]).unwrap();

    let dict_ptr = dict_value
        .as_object_ptr()
        .expect("dict should be heap object");
    let dict = unsafe { &*(dict_ptr as *const DictObject) };
    assert!(!dict.contains_key(key));
}

#[test]
fn test_placeholder_types_report_module_name() {
    let value = reference_type_value();
    let ptr = value.as_object_ptr().expect("class should be heap object");
    let class = unsafe { &*(ptr as *const PyClassObject) };
    let module = class
        .get_attr(&intern("__module__"))
        .expect("__module__ should exist");

    let module_name = if module.is_string() {
        let ptr = module
            .as_string_object_ptr()
            .expect("interned string should expose pointer");
        interned_by_ptr(ptr as *const u8)
            .expect("module name should be interned")
            .as_str()
            .to_string()
    } else {
        let ptr = module
            .as_object_ptr()
            .expect("module name should be string object");
        let string = unsafe { &*(ptr as *const StringObject) };
        string.as_str().to_string()
    };

    assert_eq!(module_name, "_weakref");
}

#[test]
fn test_reachability_marker_ignores_misaligned_object_payload() {
    let mut marker = ReachabilityMarker::new();
    marker.push(Value::object_ptr(0x7usize as *const ()));
    marker.drain();

    assert!(marker.reachable.is_empty());
}

#[test]
fn test_unregistered_user_type_does_not_trace_as_shaped_object() {
    #[repr(C)]
    struct UnknownHeapObject {
        header: ObjectHeader,
    }

    let type_id = TypeId::from_raw(TypeId::FIRST_USER_TYPE + 10_000);
    let object = Box::new(UnknownHeapObject {
        header: ObjectHeader::new(type_id),
    });
    let ptr = Box::into_raw(object);
    let value = Value::object_ptr(ptr as *const ());

    assert!(!is_shaped_object(value, type_id));

    let mut marker = ReachabilityMarker::new();
    marker.push(value);
    marker.drain();

    assert!(marker.reachable.contains(&(ptr as usize)));

    unsafe {
        drop(Box::from_raw(ptr));
    }
}

#[test]
fn test_eager_weakref_sweep_clears_unreachable_target() {
    let target = leak_object_value(DictObject::new());
    let reference =
        reference_new(&[reference_type_value(), target]).expect("weakref creation should work");

    let mut code = CodeObject::new("weakref_root", "<test>");
    code.locals = vec![Arc::<str>::from("wr")].into_boxed_slice();
    code.register_count = 1;

    let mut vm = VirtualMachine::new();
    vm.push_frame(Arc::new(code), 0).expect("frame push failed");
    vm.current_frame_mut().set_reg(0, reference);

    clear_unreachable_weakrefs_if_registered(&vm);

    assert!(reference_call(&[reference]).unwrap().is_none());
}
