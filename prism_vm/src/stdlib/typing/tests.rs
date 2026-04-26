use super::*;
use prism_runtime::types::string::value_as_string_ref;

#[test]
fn test_typing_module_exposes_bootstrap_surface() {
    let module = TypingModule::new();
    assert!(module.get_attr("Union").unwrap().as_object_ptr().is_some());
    assert!(
        module
            .get_attr("Optional")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );
    assert!(module.get_attr("final").unwrap().as_object_ptr().is_some());
    assert!(
        module
            .get_attr("Protocol")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );
    assert_eq!(
        module.get_attr("TYPE_CHECKING").unwrap().as_bool(),
        Some(false)
    );
}

#[test]
fn test_typing_form_getitem_returns_annotation_placeholder_tuple() {
    let result = typing_form_getitem(&[cached_marker_value("Union"), Value::int(7).unwrap()])
        .expect("typing marker subscription should succeed");
    let ptr = result
        .as_object_ptr()
        .expect("typing form result should be heap allocated");
    let tuple = unsafe { &*(ptr as *const TupleObject) };

    assert_eq!(tuple.len(), 2);
    assert_eq!(
        value_as_string_ref(tuple.as_slice()[0])
            .expect("typing placeholder name should be a string")
            .as_str(),
        "Union"
    );
    assert_eq!(tuple.as_slice()[1].as_int(), Some(7));
}

#[test]
fn test_get_origin_returns_marker_for_annotation_placeholder() {
    let alias = typing_form_getitem(&[cached_marker_value("Union"), Value::int(7).unwrap()])
        .expect("typing marker subscription should succeed");

    assert_eq!(
        typing_get_origin(&[alias]).expect("get_origin should succeed"),
        cached_marker_value("Union")
    );
    assert_eq!(
        typing_get_origin(&[Value::int(7).unwrap()]).expect("get_origin should succeed"),
        Value::none()
    );
}

#[test]
fn test_get_args_returns_placeholder_arguments() {
    let alias = typing_form_getitem(&[cached_marker_value("Union"), Value::int(7).unwrap()])
        .expect("typing marker subscription should succeed");
    let args = typing_get_args(&[alias]).expect("get_args should succeed");
    let ptr = args
        .as_object_ptr()
        .expect("typing args should be heap allocated");
    let tuple = unsafe { &*(ptr as *const TupleObject) };

    assert_eq!(tuple.len(), 1);
    assert_eq!(tuple.as_slice()[0].as_int(), Some(7));

    let empty = typing_get_args(&[Value::int(7).unwrap()]).expect("get_args should succeed");
    let empty_ptr = empty
        .as_object_ptr()
        .expect("empty typing args should be heap allocated");
    let empty_tuple = unsafe { &*(empty_ptr as *const TupleObject) };
    assert!(empty_tuple.is_empty());
}

#[test]
fn test_typevar_factory_returns_named_marker() {
    let typevar =
        typing_typevar(&[Value::string(intern("T"))], &[]).expect("TypeVar('T') should succeed");
    assert_eq!(
        marker_name(typevar).expect("typevar marker should carry name"),
        intern("~T")
    );
}
