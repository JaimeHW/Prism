use super::*;
use prism_core::intern::{intern, interned_by_ptr};

#[test]
fn test_warnings_module_imports_as_builtin_anchor() {
    let module = WarningsModule::new();
    assert_eq!(module.name(), "_warnings");
    assert!(module.dir().contains(&Arc::from("filters")));
}

#[test]
fn test_warnings_module_exposes_bootstrap_state() {
    let module = WarningsModule::new();

    let defaultaction = module
        .get_attr("_defaultaction")
        .expect("_defaultaction should exist");
    let defaultaction_ptr = defaultaction
        .as_string_object_ptr()
        .expect("_defaultaction should be an interned string");
    assert_eq!(
        interned_by_ptr(defaultaction_ptr as *const u8)
            .unwrap()
            .as_str(),
        "default"
    );

    let filters = module.get_attr("filters").expect("filters should exist");
    let filters_ptr = filters
        .as_object_ptr()
        .expect("filters should be a list object");
    let filters = unsafe { &*(filters_ptr as *const ListObject) };
    assert!(filters.is_empty());

    let onceregistry = module
        .get_attr("_onceregistry")
        .expect("_onceregistry should exist");
    let onceregistry_ptr = onceregistry
        .as_object_ptr()
        .expect("_onceregistry should be a dict object");
    let onceregistry = unsafe { &*(onceregistry_ptr as *const DictObject) };
    assert!(onceregistry.is_empty());
}

#[test]
fn test_warnings_module_exposes_callable_bootstrap_functions() {
    let module = WarningsModule::new();

    for name in ["_filters_mutated", "warn", "warn_explicit"] {
        let value = module.get_attr(name).expect("callable should exist");
        let ptr = value
            .as_object_ptr()
            .expect("callable should be a builtin function object");
        let header = unsafe { &*(ptr as *const ObjectHeader) };
        assert_eq!(header.type_id, TypeId::BUILTIN_FUNCTION);
    }
}

#[test]
fn test_warn_accepts_cpython_keyword_surface() {
    let message = Value::string(intern("hello"));
    let category = category_value(ExceptionTypeId::DeprecationWarning)
        .expect("DeprecationWarning should be available");

    let call = bind_warn_args(
        &[],
        &[
            ("message", message),
            ("category", category),
            ("stacklevel", Value::int(3).unwrap()),
            ("source", Value::none()),
            ("skip_file_prefixes", Value::none()),
        ],
    )
    .expect("warn should bind CPython keyword arguments");

    assert_eq!(call.message, message);
    assert_eq!(call.category, Some(category));
    assert_eq!(call.stacklevel, 3);
    assert_eq!(warning_stack_depth(call.stacklevel), 2);
}

#[test]
fn test_warn_keyword_binding_rejects_duplicates_and_unknowns() {
    let message = Value::string(intern("hello"));
    let duplicate = bind_warn_args(&[message], &[("message", message)])
        .expect_err("duplicate message should be rejected");
    assert!(
        matches!(duplicate, BuiltinError::TypeError(ref msg) if msg.contains("multiple values"))
    );

    let unknown = bind_warn_args(&[message], &[("bogus", Value::none())])
        .expect_err("unknown keywords should be rejected");
    assert!(
        matches!(unknown, BuiltinError::TypeError(ref msg) if msg.contains("unexpected keyword"))
    );
}

#[test]
fn test_warning_category_accepts_supplemental_heap_warning_classes() {
    let resource_warning = crate::builtins::supplemental_exception_class("ResourceWarning")
        .expect("ResourceWarning supplemental class should exist");
    let value = Value::object_ptr(resource_warning.as_ref() as *const PyClassObject as *const ());

    let category = warning_category_from_value(Some(value))
        .expect("ResourceWarning should be a valid category")
        .expect("category should be present");

    assert_eq!(category.value, value);
    assert_eq!(category.class_id, resource_warning.class_id());
    assert_eq!(category.exception_type_id, None);
    assert!(is_warning_category_class_id(category.class_id));
}

#[test]
fn test_warning_category_matches_builtin_and_heap_category_hierarchy() {
    let resource_warning = crate::builtins::supplemental_exception_class("ResourceWarning")
        .expect("ResourceWarning supplemental class should exist");
    let resource_value =
        Value::object_ptr(resource_warning.as_ref() as *const PyClassObject as *const ());
    let category = warning_category_from_value(Some(resource_value))
        .expect("ResourceWarning should be a valid category")
        .expect("category should be present");

    let warning_value = category_value(ExceptionTypeId::Warning).expect("Warning exists");
    let runtime_warning_value =
        category_value(ExceptionTypeId::RuntimeWarning).expect("RuntimeWarning exists");

    assert!(warning_category_matches(category, warning_value).unwrap());
    assert!(warning_category_matches(category, resource_value).unwrap());
    assert!(!warning_category_matches(category, runtime_warning_value).unwrap());
}

#[test]
fn test_warning_category_rejects_non_warning_heap_classes() {
    let plain_class = PyClassObject::new_simple(intern("PlainCategory"));
    let value = Value::object_ptr(&plain_class as *const PyClassObject as *const ());

    let error = warning_category_from_value(Some(value))
        .expect_err("plain heap classes are not warning categories");

    assert_eq!(error, "warning category must be a Warning subclass");
}
