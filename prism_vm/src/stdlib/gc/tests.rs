use super::*;

fn builtin_from_value(value: Value) -> &'static BuiltinFunctionObject {
    let ptr = value
        .as_object_ptr()
        .expect("expected builtin function object");
    unsafe { &*(ptr as *const BuiltinFunctionObject) }
}

fn reset_state() {
    *GC_STATE.write().unwrap() = GcState::default();
}

#[test]
fn test_module_exposes_expected_attributes() {
    let module = GcModule::new();
    assert_eq!(module.name(), "gc");
    assert!(
        module
            .get_attr("collect")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );
    assert!(
        module
            .get_attr("garbage")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );
    assert!(
        module
            .get_attr("callbacks")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );
}

#[test]
fn test_enable_disable_and_isenabled_roundtrip() {
    reset_state();
    let module = GcModule::new();
    let disable = builtin_from_value(module.get_attr("disable").unwrap());
    let enable = builtin_from_value(module.get_attr("enable").unwrap());
    let isenabled = builtin_from_value(module.get_attr("isenabled").unwrap());

    assert_eq!(isenabled.call(&[]).unwrap(), Value::bool(true));
    disable.call(&[]).unwrap();
    assert_eq!(isenabled.call(&[]).unwrap(), Value::bool(false));
    enable.call(&[]).unwrap();
    assert_eq!(isenabled.call(&[]).unwrap(), Value::bool(true));
}

#[test]
fn test_collect_accepts_optional_generation() {
    reset_state();
    let module = GcModule::new();
    let collect = builtin_from_value(module.get_attr("collect").unwrap());
    let mut vm = VirtualMachine::new();

    assert_eq!(
        collect.call_with_vm(&mut vm, &[]).unwrap().as_int(),
        Some(0)
    );
    assert_eq!(
        collect
            .call_with_vm(&mut vm, &[Value::int(0).unwrap()])
            .unwrap()
            .as_int(),
        Some(0)
    );

    let err = collect
        .call_with_vm(&mut vm, &[Value::int(3).unwrap()])
        .expect_err("invalid generation should fail");
    assert!(err.to_string().contains("invalid generation"));
}

#[test]
fn test_get_and_set_threshold_roundtrip() {
    reset_state();
    let module = GcModule::new();
    let get_threshold = builtin_from_value(module.get_attr("get_threshold").unwrap());
    let set_threshold = builtin_from_value(module.get_attr("set_threshold").unwrap());

    let initial = get_threshold.call(&[]).unwrap();
    let initial_ptr = initial
        .as_object_ptr()
        .expect("thresholds should be a tuple");
    let initial_tuple = unsafe { &*(initial_ptr as *const TupleObject) };
    assert_eq!(
        initial_tuple
            .as_slice()
            .iter()
            .map(|value| value.as_int().unwrap())
            .collect::<Vec<_>>(),
        DEFAULT_THRESHOLDS
    );

    set_threshold
        .call(&[
            Value::int(100).unwrap(),
            Value::int(20).unwrap(),
            Value::int(5).unwrap(),
        ])
        .unwrap();
    assert_eq!(gc_thresholds(), [100, 20, 5]);
}

#[test]
fn test_is_tracked_matches_container_shape() {
    let atomic_tuple = TupleObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    let atomic_tuple_ptr = Box::into_raw(Box::new(atomic_tuple));
    let atomic_tuple_value = Value::object_ptr(atomic_tuple_ptr as *const ());

    let tracked_list = ListObject::from_slice(&[Value::int(1).unwrap()]);
    let tracked_list_ptr = Box::into_raw(Box::new(tracked_list));
    let tracked_list_value = Value::object_ptr(tracked_list_ptr as *const ());

    let tracked_tuple = TupleObject::from_slice(&[tracked_list_value]);
    let tracked_tuple_ptr = Box::into_raw(Box::new(tracked_tuple));
    let tracked_tuple_value = Value::object_ptr(tracked_tuple_ptr as *const ());

    let mut atomic_dict = DictObject::new();
    atomic_dict.set(Value::string(intern("a")), Value::int(1).unwrap());
    let atomic_dict_ptr = Box::into_raw(Box::new(atomic_dict));
    let atomic_dict_value = Value::object_ptr(atomic_dict_ptr as *const ());

    let mut nested_dict = DictObject::new();
    nested_dict.set(Value::string(intern("a")), tracked_list_value);
    let nested_dict_ptr = Box::into_raw(Box::new(nested_dict));
    let nested_dict_value = Value::object_ptr(nested_dict_ptr as *const ());

    let set = SetObject::from_slice(&[Value::int(1).unwrap()]);
    let set_ptr = Box::into_raw(Box::new(set));
    let set_value = Value::object_ptr(set_ptr as *const ());

    assert!(!is_tracked_value(
        Value::int(1).unwrap(),
        &mut FxHashSet::default()
    ));
    assert!(!is_tracked_value(
        Value::string(intern("abc")),
        &mut FxHashSet::default()
    ));
    assert!(!is_tracked_value(
        atomic_tuple_value,
        &mut FxHashSet::default()
    ));
    assert!(is_tracked_value(
        tracked_tuple_value,
        &mut FxHashSet::default()
    ));
    assert!(!is_tracked_value(
        atomic_dict_value,
        &mut FxHashSet::default()
    ));
    assert!(is_tracked_value(
        nested_dict_value,
        &mut FxHashSet::default()
    ));
    assert!(is_tracked_value(set_value, &mut FxHashSet::default()));

    unsafe {
        drop(Box::from_raw(set_ptr));
        drop(Box::from_raw(nested_dict_ptr));
        drop(Box::from_raw(atomic_dict_ptr));
        drop(Box::from_raw(tracked_tuple_ptr));
        drop(Box::from_raw(tracked_list_ptr));
        drop(Box::from_raw(atomic_tuple_ptr));
    }
}

#[test]
fn test_is_tracked_treats_byte_sequences_as_untracked() {
    let bytes = BytesObject::from_slice(b"abc");
    let bytes_ptr = Box::into_raw(Box::new(bytes));
    let bytes_value = Value::object_ptr(bytes_ptr as *const ());

    let bytearray = BytesObject::bytearray_from_slice(b"abc");
    let bytearray_ptr = Box::into_raw(Box::new(bytearray));
    let bytearray_value = Value::object_ptr(bytearray_ptr as *const ());

    assert!(!is_tracked_value(bytes_value, &mut FxHashSet::default()));
    assert!(!is_tracked_value(
        bytearray_value,
        &mut FxHashSet::default()
    ));

    unsafe {
        drop(Box::from_raw(bytearray_ptr));
        drop(Box::from_raw(bytes_ptr));
    }
}
