use super::*;
use prism_core::intern::intern;

#[test]
fn test_builtin_registry() {
    let registry = BuiltinRegistry::with_standard_builtins();

    assert!(registry.get("None").unwrap().is_none());
    assert!(registry.get("True").unwrap().is_truthy());
    assert!(!registry.get("False").unwrap().is_truthy());
    assert!(registry.get("Ellipsis").unwrap().as_object_ptr().is_some());
    assert!(
        registry
            .get("NotImplemented")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );
}

#[test]
fn test_builtin_registry_singletons_have_runtime_type_ids() {
    let registry = BuiltinRegistry::with_standard_builtins();

    let ellipsis = registry
        .get("Ellipsis")
        .expect("Ellipsis should be registered");
    let not_implemented = registry
        .get("NotImplemented")
        .expect("NotImplemented should be registered");

    let ellipsis_ptr = ellipsis
        .as_object_ptr()
        .expect("Ellipsis should be a heap singleton");
    let not_implemented_ptr = not_implemented
        .as_object_ptr()
        .expect("NotImplemented should be a heap singleton");

    let ellipsis_type =
        unsafe { (*(ellipsis_ptr as *const prism_runtime::object::ObjectHeader)).type_id };
    let not_implemented_type =
        unsafe { (*(not_implemented_ptr as *const prism_runtime::object::ObjectHeader)).type_id };

    assert_eq!(ellipsis_type, TypeId::ELLIPSIS);
    assert_eq!(not_implemented_type, TypeId::NOT_IMPLEMENTED);
}

#[test]
fn test_registry_contains_functions() {
    let registry = BuiltinRegistry::with_standard_builtins();

    assert!(registry.is_function("len"));
    assert!(registry.is_function("print"));
    assert!(registry.is_function("range"));
    assert!(registry.is_function("type"));
    assert!(registry.is_function("getattr"));
    assert!(registry.is_function("compile"));
    assert!(!registry.is_function("None")); // Not a function
}

#[test]
fn test_registry_exposes_keyword_aware_compile_builtin() {
    let registry = BuiltinRegistry::with_standard_builtins();

    assert!(registry.get_keyword_function("compile").is_some());
    assert!(
        registry
            .get("compile")
            .and_then(|value| value.as_object_ptr())
            .is_some()
    );

    let compiled = registry
        .call(
            "compile",
            &[
                Value::string(intern("pass")),
                Value::string(intern("<registry>")),
                Value::string(intern("exec")),
            ],
        )
        .expect("keyword-aware builtins should still support positional registry calls");
    let ptr = compiled
        .as_object_ptr()
        .expect("compile should return a code object");
    let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
    assert_eq!(header.type_id, TypeId::CODE);
}

#[test]
fn test_registry_exposes_vm_aware_builtin_function_entries() {
    let registry = BuiltinRegistry::with_standard_builtins();

    assert!(registry.get_vm_function("getattr").is_some());
    assert!(
        registry
            .get("getattr")
            .and_then(|value| value.as_object_ptr())
            .is_some()
    );
    assert!(registry.get_vm_function("__import__").is_some());
    assert!(
        registry
            .get("__import__")
            .and_then(|value| value.as_object_ptr())
            .is_some()
    );
}

#[test]
fn test_registry_contains_memoryview_type_object() {
    let registry = BuiltinRegistry::with_standard_builtins();
    let memoryview = registry
        .get("memoryview")
        .expect("memoryview should be registered");
    let ptr = memoryview
        .as_object_ptr()
        .expect("memoryview should be exposed as a type object");

    let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
    assert_eq!(header.type_id, TypeId::TYPE);
    assert_eq!(
        crate::builtins::builtin_type_object_type_id(ptr),
        Some(TypeId::MEMORYVIEW)
    );
}

#[test]
fn test_registry_contains_slice_type_object() {
    let registry = BuiltinRegistry::with_standard_builtins();
    let slice_type = registry.get("slice").expect("slice should be registered");
    let ptr = slice_type
        .as_object_ptr()
        .expect("slice should be exposed as a type object");

    let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
    assert_eq!(header.type_id, TypeId::TYPE);
    assert_eq!(
        crate::builtins::builtin_type_object_type_id(ptr),
        Some(TypeId::SLICE)
    );
}

#[test]
fn test_registry_contains_super_type_object() {
    let registry = BuiltinRegistry::with_standard_builtins();
    let super_type = registry.get("super").expect("super should be registered");
    let ptr = super_type
        .as_object_ptr()
        .expect("super should be exposed as a type object");

    let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
    assert_eq!(header.type_id, TypeId::TYPE);
    assert_eq!(
        crate::builtins::builtin_type_object_type_id(ptr),
        Some(TypeId::SUPER)
    );
}

#[test]
fn test_registry_contains_supplemental_warning_category_types() {
    let registry = BuiltinRegistry::with_standard_builtins();

    for name in [
        "BytesWarning",
        "FutureWarning",
        "ImportWarning",
        "ResourceWarning",
        "UnicodeWarning",
        "EncodingWarning",
    ] {
        let value = registry
            .get(name)
            .unwrap_or_else(|| panic!("builtin {name} should be registered"));
        let ptr = value
            .as_object_ptr()
            .unwrap_or_else(|| panic!("builtin {name} should be a type object"));
        let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
        assert_eq!(header.type_id, TypeId::TYPE, "{name} should be a heap type");
    }
}

#[test]
fn test_registry_warning_categories_are_warning_subclasses() {
    let registry = BuiltinRegistry::with_standard_builtins();
    let warning = registry.get("Warning").expect("Warning should exist");

    for name in [
        "BytesWarning",
        "FutureWarning",
        "ImportWarning",
        "ResourceWarning",
        "UnicodeWarning",
        "EncodingWarning",
    ] {
        let category = registry
            .get(name)
            .unwrap_or_else(|| panic!("builtin {name} should be registered"));
        let result = builtin_issubclass(&[category, warning])
            .unwrap_or_else(|_| panic!("{name} should be comparable with Warning"));
        assert_eq!(
            result.as_bool(),
            Some(true),
            "{name} should subclass Warning",
        );
    }
}

#[test]
fn test_call_abs() {
    let registry = BuiltinRegistry::with_standard_builtins();
    let result = registry.call("abs", &[Value::int(-42).unwrap()]);
    assert_eq!(result.unwrap().as_int(), Some(42));
}

#[test]
fn test_builtin_function_as_object_ptr() {
    let registry = BuiltinRegistry::with_standard_builtins();

    let abs_val = registry.get("abs").expect("abs should be registered");

    assert!(
        abs_val.as_object_ptr().is_some(),
        "abs should be stored as object_ptr, got value: {:?}",
        abs_val
    );

    let ptr = abs_val.as_object_ptr().unwrap();
    let header_ptr = ptr as *const prism_runtime::object::ObjectHeader;
    let type_id = unsafe { (*header_ptr).type_id };

    assert_eq!(
        type_id,
        prism_runtime::object::type_obj::TypeId::BUILTIN_FUNCTION,
        "abs should have TypeId::BUILTIN_FUNCTION, got {:?}",
        type_id
    );
}
