use super::*;
use prism_runtime::types::list::value_as_list_ref;

fn builtin_from_value(value: Value) -> &'static BuiltinFunctionObject {
    let ptr = value
        .as_object_ptr()
        .expect("expected builtin function object");
    unsafe { &*(ptr as *const BuiltinFunctionObject) }
}

fn list(items: &[Value]) -> Value {
    list_value(items.to_vec())
}

#[test]
fn test_select_module_exposes_select_py_bootstrap_surface() {
    let module = SelectModule::new();

    assert_eq!(module.name(), "select");
    assert!(module.get_attr("__doc__").is_ok());
    assert!(module.get_attr("error").is_ok());
    assert!(module.get_attr("select").is_ok());
    assert!(module.get_attr("poll").is_err());
}

#[test]
fn test_select_returns_three_ready_lists() {
    let readers = list(&[Value::int(3).unwrap()]);
    let writers = list(&[Value::int(4).unwrap()]);
    let exceptional = list(&[]);
    let result = select_select(&[readers, writers, exceptional, Value::float(0.0)])
        .expect("select should accept list arguments");

    let tuple = unsafe { &*(result.as_object_ptr().unwrap() as *const TupleObject) };
    assert_eq!(tuple.len(), 3);
    assert_eq!(
        value_as_list_ref(tuple.get(0).unwrap())
            .expect("reader result list")
            .as_slice(),
        &[Value::int(3).unwrap()]
    );
    assert_eq!(
        value_as_list_ref(tuple.get(1).unwrap())
            .expect("writer result list")
            .as_slice(),
        &[Value::int(4).unwrap()]
    );
    assert!(
        value_as_list_ref(tuple.get(2).unwrap())
            .expect("exception result list")
            .is_empty()
    );
}

#[test]
fn test_select_rejects_negative_timeout() {
    let empty = list(&[]);
    let error = select_select(&[empty, empty, empty, Value::float(-0.1)])
        .expect_err("negative timeouts should be rejected");

    assert!(matches!(error, BuiltinError::ValueError(_)));
}

#[test]
fn test_select_builtin_function_is_callable() {
    let function = builtin_from_value(SelectModule::new().get_attr("select").unwrap());
    let empty = list(&[]);

    assert!(function.call(&[empty, empty, empty, Value::none()]).is_ok());
}
