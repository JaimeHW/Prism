use super::*;

fn builtin_from_value(value: Value) -> &'static BuiltinFunctionObject {
    let ptr = value
        .as_object_ptr()
        .expect("expected builtin function object");
    unsafe { &*(ptr as *const BuiltinFunctionObject) }
}

#[test]
fn test_winreg_module_exposes_importlib_bootstrap_surface() {
    let module = WinregModule::new();

    assert!(module.get_attr("HKEY_CURRENT_USER").is_ok());
    assert!(module.get_attr("HKEY_LOCAL_MACHINE").is_ok());
    assert!(module.get_attr("OpenKey").is_ok());
    assert!(module.get_attr("OpenKeyEx").is_ok());
    assert!(module.get_attr("QueryValue").is_ok());
    assert!(module.get_attr("QueryValueEx").is_ok());
    assert!(module.get_attr("CloseKey").is_ok());
}

#[cfg(not(windows))]
#[test]
fn test_open_key_reports_registry_unavailable_as_oserror() {
    let module = WinregModule::new();
    let open_key = builtin_from_value(module.get_attr("OpenKey").unwrap());
    let err = open_key
        .call(&[Value::int(0).unwrap(), Value::int(0).unwrap()])
        .expect_err("OpenKey should raise OSError");
    assert!(matches!(err, BuiltinError::OSError(_)));
}

#[test]
fn test_open_key_rejects_non_string_subkey() {
    let module = WinregModule::new();
    let open_key = builtin_from_value(module.get_attr("OpenKey").unwrap());
    let err = open_key
        .call(&[Value::int(0).unwrap(), Value::int(0).unwrap()])
        .expect_err("OpenKey should reject non-string subkey");
    assert!(matches!(err, BuiltinError::TypeError(_)));
}

#[test]
fn test_close_key_is_a_no_op_for_bootstrap_cleanup_paths() {
    let module = WinregModule::new();
    let close_key = builtin_from_value(module.get_attr("CloseKey").unwrap());
    let result = close_key
        .call(&[Value::int(0).unwrap()])
        .expect("CloseKey should succeed");
    assert!(result.is_none());
}

#[cfg(windows)]
#[test]
fn test_query_value_ex_reads_windows_current_type() {
    use prism_runtime::types::string::value_as_string_ref;

    let module = WinregModule::new();
    let open_key = builtin_from_value(module.get_attr("OpenKeyEx").unwrap());
    let query_value_ex = builtin_from_value(module.get_attr("QueryValueEx").unwrap());
    let close_key = builtin_from_value(module.get_attr("CloseKey").unwrap());
    let root = module.get_attr("HKEY_LOCAL_MACHINE").unwrap();

    let key = open_key
        .call(&[
            root,
            Value::string(intern(r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")),
        ])
        .expect("CurrentVersion registry key should open");
    let result = query_value_ex
        .call(&[key, Value::string(intern("CurrentType"))])
        .expect("CurrentType should be readable");
    let tuple_ptr = result
        .as_object_ptr()
        .expect("QueryValueEx should return a tuple");
    let tuple = unsafe { &*(tuple_ptr as *const TupleObject) };
    assert_eq!(tuple.len(), 2);
    let value = value_as_string_ref(tuple.as_slice()[0])
        .expect("CurrentType should be a string")
        .as_str()
        .to_string();
    assert!(
        value.contains("Multiprocessor") || value.contains("Uniprocessor"),
        "unexpected CurrentType: {value}"
    );

    close_key.call(&[key]).expect("CloseKey should succeed");
}

#[cfg(windows)]
#[test]
fn test_registry_key_object_exposes_context_manager_methods() {
    let key = key_object_value(0).expect("zero handle should fit");
    let ptr = key.as_object_ptr().expect("key should be a heap object");
    let object = unsafe { &*(ptr as *const ShapedObject) };
    assert!(object.get_property("__enter__").is_some());
    assert!(object.get_property("__exit__").is_some());
    assert!(object.get_property("Close").is_some());
}
