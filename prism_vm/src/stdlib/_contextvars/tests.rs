use super::*;

fn builtin_from_value(value: Value) -> &'static BuiltinFunctionObject {
    let ptr = value
        .as_object_ptr()
        .expect("expected builtin function object");
    unsafe { &*(ptr as *const BuiltinFunctionObject) }
}

fn value_to_text(value: Value) -> String {
    if value.is_string() {
        let ptr = value
            .as_string_object_ptr()
            .expect("interned string should expose a pointer");
        return interned_by_ptr(ptr as *const u8)
            .expect("string should be interned")
            .as_str()
            .to_string();
    }

    let ptr = value.as_object_ptr().expect("expected string object");
    let string = unsafe { &*(ptr as *const StringObject) };
    string.as_str().to_string()
}

#[test]
fn test_module_exposes_expected_bootstrap_surface() {
    let module = ContextVarsModule::new();
    assert!(
        module
            .get_attr("Context")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );
    assert!(
        module
            .get_attr("ContextVar")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );
    assert!(module.get_attr("Token").unwrap().as_object_ptr().is_some());
    assert!(
        module
            .get_attr("copy_context")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );
}

#[test]
fn test_contextvar_get_set_and_reset_roundtrip() {
    let class_ptr = contextvar_type_value()
        .as_object_ptr()
        .expect("ContextVar should be a class");
    let class = unsafe { &*(class_ptr as *const PyClassObject) };
    let new_builtin = builtin_from_value(
        class
            .get_attr(&intern("__new__"))
            .expect("ContextVar.__new__ should exist"),
    );
    let get_builtin = builtin_from_value(
        class
            .get_attr(&intern("get"))
            .expect("ContextVar.get should exist"),
    );
    let set_builtin = builtin_from_value(
        class
            .get_attr(&intern("set"))
            .expect("ContextVar.set should exist"),
    );
    let reset_builtin = builtin_from_value(
        class
            .get_attr(&intern("reset"))
            .expect("ContextVar.reset should exist"),
    );

    let var = new_builtin
        .call(&[
            contextvar_type_value(),
            Value::string(intern("decimal_context")),
        ])
        .expect("ContextVar.__new__ should succeed");

    let missing = get_builtin
        .call(&[var])
        .expect_err("unset ContextVar.get should raise LookupError");
    assert!(matches!(missing, BuiltinError::Raised(_)));

    let token = set_builtin
        .call(&[var, Value::int(42).expect("integer should fit")])
        .expect("ContextVar.set should return a token");
    assert_eq!(
        get_builtin
            .call(&[var])
            .expect("ContextVar.get should return the stored value")
            .as_int(),
        Some(42)
    );

    reset_builtin
        .call(&[var, token])
        .expect("ContextVar.reset should accept the returned token");

    let after_reset = get_builtin
        .call(&[var, Value::string(intern("fallback"))])
        .expect("ContextVar.get should accept a fallback");
    assert_eq!(value_to_text(after_reset), "fallback");
}

#[test]
fn test_copy_context_returns_context_instance() {
    let module = ContextVarsModule::new();
    let copy = builtin_from_value(
        module
            .get_attr("copy_context")
            .expect("copy_context should exist"),
    );
    let value = copy.call(&[]).expect("copy_context should succeed");
    let ptr = value
        .as_object_ptr()
        .expect("copy_context should return a heap object");
    assert_eq!(
        crate::ops::objects::extract_type_id(ptr),
        CONTEXT_CLASS.class_type_id()
    );
}
