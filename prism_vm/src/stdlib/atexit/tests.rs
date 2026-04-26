use super::*;

static TEST_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

fn builtin_from_value(value: Value) -> &'static BuiltinFunctionObject {
    let ptr = value
        .as_object_ptr()
        .expect("expected builtin function object");
    unsafe { &*(ptr as *const BuiltinFunctionObject) }
}

fn clear_registry() {
    EXIT_CALLBACKS.lock().unwrap().clear();
}

#[test]
fn test_module_exposes_core_api() {
    let _guard = TEST_LOCK.lock().unwrap();
    clear_registry();
    let module = AtexitModule::new();
    assert!(
        module
            .get_attr("register")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );
    assert!(
        module
            .get_attr("_run_exitfuncs")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );
    clear_registry();
}

#[test]
fn test_register_and_unregister_update_callback_count() {
    let _guard = TEST_LOCK.lock().unwrap();
    clear_registry();
    let module = AtexitModule::new();
    let register = builtin_from_value(module.get_attr("register").unwrap());
    let unregister = builtin_from_value(module.get_attr("unregister").unwrap());
    let ncallbacks = builtin_from_value(module.get_attr("_ncallbacks").unwrap());

    let callback = Value::bool(true);
    assert_eq!(register.call(&[callback]).unwrap(), callback);
    assert_eq!(ncallbacks.call(&[]).unwrap().as_int(), Some(1));

    unregister.call(&[callback]).unwrap();
    assert_eq!(ncallbacks.call(&[]).unwrap().as_int(), Some(0));
    clear_registry();
}

#[test]
fn test_clear_removes_registered_callbacks() {
    let _guard = TEST_LOCK.lock().unwrap();
    clear_registry();
    let module = AtexitModule::new();
    let register = builtin_from_value(module.get_attr("register").unwrap());
    let clear = builtin_from_value(module.get_attr("_clear").unwrap());
    let ncallbacks = builtin_from_value(module.get_attr("_ncallbacks").unwrap());

    register.call(&[Value::bool(false)]).unwrap();
    register.call(&[Value::none()]).unwrap();
    assert_eq!(ncallbacks.call(&[]).unwrap().as_int(), Some(2));

    clear.call(&[]).unwrap();
    assert_eq!(ncallbacks.call(&[]).unwrap().as_int(), Some(0));
    clear_registry();
}
