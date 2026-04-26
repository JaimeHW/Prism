
use super::*;

fn builtin_from_value(value: Value) -> &'static BuiltinFunctionObject {
    let ptr = value
        .as_object_ptr()
        .expect("expected builtin function object");
    unsafe { &*(ptr as *const BuiltinFunctionObject) }
}

fn tuple_from_value(value: Value) -> &'static TupleObject {
    let ptr = value.as_object_ptr().expect("expected tuple object");
    unsafe { &*(ptr as *const TupleObject) }
}

#[test]
fn test_time_module_exposes_callable_conversion_helpers() {
    let module = TimeModule::new();
    assert!(module.get_attr("gmtime").unwrap().as_object_ptr().is_some());
    assert!(
        module
            .get_attr("localtime")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );
    assert!(
        module
            .get_attr("strftime")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );
    assert!(
        module
            .get_attr("asctime")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );
}

#[test]
fn test_gmtime_builtin_returns_nine_item_tuple() {
    let module = TimeModule::new();
    let gmtime = builtin_from_value(module.get_attr("gmtime").unwrap());
    let result = gmtime
        .call(&[Value::int(0).unwrap()])
        .expect("gmtime should succeed");
    let tuple = tuple_from_value(result);
    assert_eq!(tuple.len(), 9);
    assert_eq!(tuple.as_slice()[0].as_int(), Some(1970));
    assert_eq!(tuple.as_slice()[1].as_int(), Some(1));
    assert_eq!(tuple.as_slice()[2].as_int(), Some(1));
}

#[test]
fn test_asctime_and_mktime_accept_sequence_arguments() {
    let module = TimeModule::new();
    let asctime = builtin_from_value(module.get_attr("asctime").unwrap());
    let mktime = builtin_from_value(module.get_attr("mktime").unwrap());
    let tuple = tuple_value(&[
        Value::int(1973).unwrap(),
        Value::int(9).unwrap(),
        Value::int(16).unwrap(),
        Value::int(1).unwrap(),
        Value::int(3).unwrap(),
        Value::int(52).unwrap(),
        Value::int(0).unwrap(),
        Value::int(259).unwrap(),
        Value::int(-1).unwrap(),
    ]);

    let rendered = asctime.call(&[tuple]).expect("asctime should succeed");
    let rendered_ptr = rendered
        .as_string_object_ptr()
        .expect("asctime should return a string");
    let rendered_text =
        interned_by_ptr(rendered_ptr as *const u8).expect("asctime result should be interned");
    let text = rendered_text.as_str();
    assert!(text.contains("1973"));

    let timestamp = mktime.call(&[tuple]).expect("mktime should succeed");
    assert!(timestamp.as_float().is_some());
}

#[test]
fn test_time_module_exposes_tzname_tuple() {
    let module = TimeModule::new();
    let value = module.get_attr("tzname").expect("tzname should exist");
    let tuple = tuple_from_value(value);
    assert_eq!(tuple.len(), 2);
}
