use super::*;
use crate::VirtualMachine;
use crate::import::ImportResolver;
use prism_core::intern::intern;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::list::ListObject;

fn builtin_from_value(value: Value) -> &'static BuiltinFunctionObject {
    let ptr = value
        .as_object_ptr()
        .expect("builtin function should be an object");
    unsafe { &*(ptr as *const BuiltinFunctionObject) }
}

#[test]
fn test_imp_module_exposes_bootstrap_surface() {
    let module = ImpModule::new();

    assert!(module.get_attr("acquire_lock").is_ok());
    assert!(module.get_attr("create_builtin").is_ok());
    assert!(module.get_attr("exec_builtin").is_ok());
    assert!(module.get_attr("source_hash").is_ok());
    assert!(module.get_attr("check_hash_based_pycs").is_ok());
}

#[test]
fn test_imp_is_builtin_reports_native_bootstrap_modules() {
    let builtin = builtin_from_value(
        ImpModule::new()
            .get_attr("is_builtin")
            .expect("is_builtin should exist"),
    );

    let thread_result = builtin
        .call(&[Value::string(intern("_thread"))])
        .expect("builtin lookup should succeed");
    assert_eq!(thread_result.as_int(), Some(1));

    let re_result = builtin
        .call(&[Value::string(intern("re"))])
        .expect("source-backed lookup should succeed");
    assert_eq!(re_result.as_int(), Some(0));
}

#[test]
fn test_imp_extension_suffixes_returns_empty_list() {
    let builtin = builtin_from_value(
        ImpModule::new()
            .get_attr("extension_suffixes")
            .expect("extension_suffixes should exist"),
    );

    let result = builtin
        .call(&[])
        .expect("extension_suffixes should succeed");
    let ptr = result
        .as_object_ptr()
        .expect("extension_suffixes should return a list");
    let list = unsafe { &*(ptr as *const ListObject) };
    assert!(list.is_empty());
}

#[test]
fn test_imp_source_hash_returns_eight_bytes() {
    let builtin = builtin_from_value(
        ImpModule::new()
            .get_attr("source_hash")
            .expect("source_hash should exist"),
    );
    let source = leak_object_value(BytesObject::from_slice(b"print('hello')"));
    let result = builtin
        .call(&[Value::int(123).unwrap(), source])
        .expect("source_hash should succeed");
    let ptr = result
        .as_object_ptr()
        .expect("source_hash should return bytes");
    let bytes = unsafe { &*(ptr as *const BytesObject) };
    assert_eq!(bytes.as_bytes().len(), 8);
}

#[test]
fn test_imp_create_builtin_loads_native_module_through_vm() {
    let mut vm = VirtualMachine::new();
    let builtin = builtin_from_value(
        ImpModule::new()
            .get_attr("create_builtin")
            .expect("create_builtin should exist"),
    );

    let spec = Box::into_raw(Box::new({
        let registry = prism_runtime::object::shape::shape_registry();
        let mut object = ShapedObject::with_empty_shape(registry.empty_shape());
        object.set_property(intern("name"), Value::string(intern("_thread")), registry);
        object
    }));

    let value = builtin
        .call_with_vm(&mut vm, &[Value::object_ptr(spec as *const ())])
        .expect("create_builtin should succeed");
    let module_ptr = value
        .as_object_ptr()
        .expect("create_builtin should return a module object");
    let module = unsafe { &*(module_ptr as *const ModuleObject) };
    assert_eq!(module.name(), "_thread");
}

#[test]
fn test_import_resolver_can_load_imp_module() {
    let resolver = ImportResolver::new();
    let module = resolver
        .import_module("_imp")
        .expect("_imp should be importable");
    assert!(module.get_attr("create_builtin").is_some());
    assert_eq!(
        module
            .get_attr("check_hash_based_pycs")
            .and_then(|value| value.as_string_object_ptr())
            .and_then(|ptr| interned_by_ptr(ptr as *const u8))
            .map(|value| value.as_ref().to_string()),
        Some("never".to_string())
    );
}
