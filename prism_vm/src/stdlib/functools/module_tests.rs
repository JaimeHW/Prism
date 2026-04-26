
use super::*;

#[test]
fn test_get_attr_exposes_wraps_update_wrapper_and_constants() {
    let module = FunctoolsModule::new();

    assert!(module.get_attr("wraps").unwrap().as_object_ptr().is_some());
    assert!(
        module
            .get_attr("update_wrapper")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );

    let assignments = module.get_attr("WRAPPER_ASSIGNMENTS").unwrap();
    let assignments_ptr = assignments.as_object_ptr().expect("tuple should be object");
    let assignments_tuple = unsafe { &*(assignments_ptr as *const TupleObject) };
    assert_eq!(assignments_tuple.len(), WRAPPER_ASSIGNMENTS.len());
}

#[test]
fn test_wraps_returns_identity_decorator() {
    let decorator = builtin_wraps(&[Value::none()]).expect("wraps should succeed");
    let decorator_ptr = decorator
        .as_object_ptr()
        .expect("decorator should be builtin function");
    let builtin = unsafe { &*(decorator_ptr as *const BuiltinFunctionObject) };
    let wrapped = Value::int(7).unwrap();
    assert_eq!(builtin.call(&[wrapped]).unwrap(), wrapped);
}

#[test]
fn test_update_wrapper_returns_wrapper_argument() {
    let wrapper = Value::int(11).unwrap();
    let wrapped = Value::int(12).unwrap();
    assert_eq!(
        builtin_update_wrapper(&[wrapper, wrapped]).unwrap(),
        wrapper
    );
}
