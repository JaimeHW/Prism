use super::*;
use prism_runtime::types::tuple::TupleObject;

#[test]
fn test_module_wrapper_exposes_os_path_name() {
    let module = OsPathModule::new();
    assert_eq!(module.name(), "os.path");
    assert!(module.dir().contains(&Arc::from("join")));
}

#[test]
fn test_commonprefix_builtin_matches_string_prefix_semantics() {
    let input = TupleObject::from_slice(&[
        Value::string(intern("interstate")),
        Value::string(intern("interstellar")),
        Value::string(intern("internal")),
    ]);
    let input_ptr = Box::into_raw(Box::new(input));
    let value = builtin_commonprefix(&[Value::object_ptr(input_ptr as *const ())])
        .expect("commonprefix should succeed");

    assert_eq!(
        interned_by_ptr(value.as_string_object_ptr().unwrap() as *const u8)
            .expect("result should resolve")
            .as_str(),
        "inter"
    );

    unsafe {
        drop(Box::from_raw(input_ptr));
    }
}

#[test]
fn test_module_wrapper_exposes_callable_commonprefix() {
    let module = OsPathModule::new();
    let value = module.get_attr("commonprefix").expect("attr should exist");
    assert!(value.as_object_ptr().is_some());
}
