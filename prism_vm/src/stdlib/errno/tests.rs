use super::*;

fn errorcode_mapping(module: &ErrnoModule) -> &'static DictObject {
    let value = module
        .get_attr("errorcode")
        .expect("errorcode should be exported");
    let ptr = value
        .as_object_ptr()
        .expect("errorcode should be a dict object");
    unsafe { &*(ptr as *const DictObject) }
}

#[test]
fn test_errno_module_exports_expected_constants() {
    let module = ErrnoModule::new();
    assert!(module.get_attr("ENOENT").is_ok());
    assert!(module.get_attr("EBADF").is_ok());
    assert!(module.get_attr("EINVAL").is_ok());
    assert!(module.get_attr("errorcode").is_ok());
}

#[test]
fn test_errno_errorcode_maps_back_to_symbol_names() {
    let module = ErrnoModule::new();
    let errorcode = errorcode_mapping(&module);

    for name in ["ENOENT", "EBADF", "EINVAL"] {
        let code = module.get_attr(name).expect("constant should be exported");
        assert_eq!(
            errorcode.get(code),
            Some(Value::string(intern(name))),
            "errorcode should map {} back to its symbol",
            name
        );
    }
}
