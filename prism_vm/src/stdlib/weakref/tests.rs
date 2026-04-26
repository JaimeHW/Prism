use super::*;

fn builtin_from_value(value: Value) -> &'static BuiltinFunctionObject {
    let ptr = value
        .as_object_ptr()
        .expect("expected builtin function object");
    unsafe { &*(ptr as *const BuiltinFunctionObject) }
}

#[test]
fn test_module_exposes_expected_bootstrap_surface() {
    let module = WeakrefModule::new();

    assert!(
        module
            .get_attr("WeakKeyDictionary")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );
    assert!(
        module
            .get_attr("WeakSet")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );
    assert!(
        module
            .get_attr("ReferenceType")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );
}

#[test]
fn test_weak_key_dictionary_factory_returns_dict_object() {
    let module = WeakrefModule::new();
    let factory = builtin_from_value(
        module
            .get_attr("WeakKeyDictionary")
            .expect("WeakKeyDictionary should exist"),
    );

    let value = factory.call(&[]).expect("factory should succeed");
    let ptr = value.as_object_ptr().expect("dict should be heap object");
    let dict = unsafe { &*(ptr as *const DictObject) };
    assert!(dict.is_empty());
}

#[test]
fn test_weak_set_factory_returns_set_object() {
    let module = WeakrefModule::new();
    let factory = builtin_from_value(module.get_attr("WeakSet").expect("WeakSet should exist"));

    let value = factory.call(&[]).expect("factory should succeed");
    let ptr = value.as_object_ptr().expect("set should be heap object");
    let set = unsafe { &*(ptr as *const SetObject) };
    assert!(set.is_empty());
}

#[test]
fn test_proxy_types_exports_both_proxy_placeholders() {
    let module = WeakrefModule::new();
    let proxy_types = module
        .get_attr("ProxyTypes")
        .expect("ProxyTypes should exist");
    let ptr = proxy_types
        .as_object_ptr()
        .expect("ProxyTypes should be tuple object");
    let tuple = unsafe { &*(ptr as *const TupleObject) };
    assert_eq!(tuple.len(), 2);
    assert_eq!(tuple.as_slice()[0], _weakref::proxy_type_value());
    assert_eq!(tuple.as_slice()[1], _weakref::callable_proxy_type_value());
}

#[test]
fn test_ref_and_weakmethod_alias_reference_type() {
    let module = WeakrefModule::new();

    assert_eq!(
        module.get_attr("ref").expect("ref should exist"),
        _weakref::reference_type_value()
    );
    assert_eq!(
        module
            .get_attr("WeakMethod")
            .expect("WeakMethod should exist"),
        _weakref::reference_type_value()
    );
}
