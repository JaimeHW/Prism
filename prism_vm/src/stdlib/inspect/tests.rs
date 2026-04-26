use super::*;
use prism_code::CodeObject;
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::descriptor::BoundMethod;
use prism_runtime::object::shape::shape_registry;

#[test]
fn test_inspect_module_exposes_bootstrap_functions() {
    let module = InspectModule::new();
    assert!(module.get_attr("get_annotations").is_ok());
    assert!(module.get_attr("signature").is_ok());
    assert!(module.get_attr("ismodule").is_ok());
    assert!(module.get_attr("isclass").is_ok());
    assert!(module.get_attr("isfunction").is_ok());
    assert!(module.get_attr("iscoroutinefunction").is_ok());
    assert!(module.get_attr("isawaitable").is_ok());
    assert!(module.get_attr("ismethod").is_ok());
    assert!(module.get_attr("isroutine").is_ok());
    assert!(module.get_attr("ismethoddescriptor").is_ok());
    assert!(module.get_attr("ismethodwrapper").is_ok());
    assert!(module.get_attr("getmodule").is_ok());
    assert!(module.get_attr("getfile").is_ok());
    assert!(module.get_attr("getsourcefile").is_ok());
    assert!(module.get_attr("unwrap").is_ok());
    assert_eq!(
        module
            .get_attr("CO_GENERATOR")
            .expect("CO_GENERATOR should exist")
            .as_int(),
        Some(0x20)
    );
}

#[test]
fn test_get_annotations_returns_empty_dict_when_missing() {
    let value = inspect_get_annotations(&[Value::bool(true)]).expect("call should succeed");
    let ptr = value
        .as_object_ptr()
        .expect("result should be a dict object");
    let dict = unsafe { &*(ptr as *const DictObject) };
    assert!(dict.is_empty());
}

#[test]
fn test_get_annotations_copies_shape_mapping() {
    let registry = shape_registry();
    let mut annotations = DictObject::new();
    annotations.set(Value::string(intern("value")), Value::string(intern("int")));
    let annotations_value = leak_object_value(annotations);

    let mut shaped = ShapedObject::with_empty_shape(registry.empty_shape());
    shaped.set_property(intern("__annotations__"), annotations_value, registry);
    let object_value = leak_object_value(shaped);

    let copied = inspect_get_annotations(&[object_value]).expect("call should succeed");
    let ptr = copied
        .as_object_ptr()
        .expect("copied annotations should be a dict");
    let dict = unsafe { &*(ptr as *const DictObject) };
    assert_eq!(
        dict.get(Value::string(intern("value"))),
        Some(Value::string(intern("int")))
    );
}

#[test]
fn test_signature_returns_text_signature_when_present() {
    let class = PyClassObject::new_simple(intern("Callable"));
    class.set_attr(
        intern("__text_signature__"),
        Value::string(intern("(x, y=None)")),
    );
    let result = inspect_signature(&[Value::object_ptr(
        Arc::into_raw(Arc::new(class)) as *const ()
    )])
    .expect("signature should succeed");
    assert_eq!(
        interned_by_ptr(result.as_string_object_ptr().unwrap() as *const u8)
            .unwrap()
            .as_str(),
        "(x, y=None)"
    );
}

#[test]
fn test_signature_defaults_to_empty_call_signature() {
    let result = inspect_signature(&[Value::int(1).unwrap()]).expect("signature should succeed");
    assert_eq!(
        interned_by_ptr(result.as_string_object_ptr().unwrap() as *const u8)
            .unwrap()
            .as_str(),
        "()"
    );
}

#[test]
fn test_inspect_predicates_classify_module_function_and_method_values() {
    let module_value =
        Value::object_ptr(Box::into_raw(Box::new(ModuleObject::new("mod"))) as *const ());
    let class_value = Value::object_ptr(Arc::into_raw(Arc::new(PyClassObject::new_simple(intern(
        "Demo",
    )))) as *const ());
    let function = Arc::new(FunctionObject::new(
        Arc::new(CodeObject::new("demo", "demo.py")),
        Arc::from("demo"),
        None,
        None,
    ));
    let function_value = Value::object_ptr(Arc::into_raw(Arc::clone(&function)) as *const ());
    let method_value = Value::object_ptr(Box::into_raw(Box::new(BoundMethod::new(
        function_value,
        Value::none(),
    ))) as *const ());

    assert_eq!(
        inspect_ismodule(&[module_value]).unwrap(),
        Value::bool(true)
    );
    assert_eq!(inspect_isclass(&[class_value]).unwrap(), Value::bool(true));
    assert_eq!(
        inspect_isfunction(&[function_value]).unwrap(),
        Value::bool(true)
    );
    assert_eq!(
        inspect_ismethod(&[method_value]).unwrap(),
        Value::bool(true)
    );
    assert_eq!(
        inspect_isroutine(&[function_value]).unwrap(),
        Value::bool(true)
    );
    assert_eq!(
        inspect_isroutine(&[method_value]).unwrap(),
        Value::bool(true)
    );
    assert_eq!(
        inspect_isroutine(&[module_value]).unwrap(),
        Value::bool(false)
    );
}

#[test]
fn test_inspect_isawaitable_recognizes_coroutine_generators_and_await_protocol() {
    let mut vm = crate::VirtualMachine::new();
    let mut code = CodeObject::new("coro", "demo.py");
    code.flags = CodeFlags::COROUTINE;
    let coroutine_value = leak_object_value(GeneratorObject::from_code(Arc::new(code)));

    assert_eq!(
        inspect_isawaitable(&mut vm, &[coroutine_value]).unwrap(),
        Value::bool(true)
    );
    assert_eq!(
        inspect_isawaitable(&mut vm, &[Value::int(1).unwrap()]).unwrap(),
        Value::bool(false)
    );

    let registry = shape_registry();
    let mut awaitable = ShapedObject::with_empty_shape(registry.empty_shape());
    awaitable.set_property(intern("__await__"), Value::bool(true), registry);
    let awaitable_value = leak_object_value(awaitable);

    assert_eq!(
        inspect_isawaitable(&mut vm, &[awaitable_value]).unwrap(),
        Value::bool(true)
    );
}

#[test]
fn test_inspect_getfile_uses_function_code_filename() {
    let mut vm = crate::VirtualMachine::new();
    let function = Arc::new(FunctionObject::new(
        Arc::new(CodeObject::new("demo", "demo_source.py")),
        Arc::from("demo"),
        None,
        None,
    ));
    let function_value = Value::object_ptr(Arc::into_raw(function) as *const ());

    let result = inspect_getfile(&mut vm, &[function_value]).expect("getfile should succeed");
    assert_eq!(
        interned_by_ptr(result.as_string_object_ptr().unwrap() as *const u8)
            .unwrap()
            .as_str(),
        "demo_source.py"
    );
}
