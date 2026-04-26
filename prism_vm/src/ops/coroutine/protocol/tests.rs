use super::*;
use prism_core::intern::intern;
use prism_runtime::object::descriptor::BoundMethod;
use prism_runtime::object::shape::ShapeRegistry;

fn builtin_identity(args: &[Value]) -> Result<Value, crate::builtins::BuiltinError> {
    if args.len() != 1 {
        return Err(crate::builtins::BuiltinError::TypeError(
            "expected one argument".to_string(),
        ));
    }
    Ok(args[0])
}

#[test]
fn test_lookup_magic_method_from_shaped_object() {
    let registry = ShapeRegistry::new();
    let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());
    obj.set_property(intern("__await__"), Value::int(7).unwrap(), &registry);
    let ptr = Box::into_raw(Box::new(obj)) as *const ();
    let val = Value::object_ptr(ptr);

    let vm = VirtualMachine::new();
    let found = lookup_magic_method(&vm, val, "__await__")
        .expect("lookup should not fail")
        .expect("__await__ should exist");
    assert_eq!(found.as_int(), Some(7));

    unsafe {
        drop(Box::from_raw(ptr as *mut ShapedObject));
    }
}

#[test]
fn test_call_unary_magic_method_builtin() {
    let mut vm = VirtualMachine::new();
    let builtin = BuiltinFunctionObject::new(Arc::from("__await__"), builtin_identity);
    let ptr = Box::into_raw(Box::new(builtin)) as *const ();
    let method = Value::object_ptr(ptr);
    let obj = Value::int(123).unwrap();

    let result = call_unary_magic_method(&mut vm, method, obj, "__await__")
        .expect("builtin unary call should succeed");
    assert_eq!(result.as_int(), Some(123));

    unsafe {
        drop(Box::from_raw(ptr as *mut BuiltinFunctionObject));
    }
}

#[test]
fn test_call_unary_magic_method_bound_builtin_method() {
    let mut vm = VirtualMachine::new();

    let builtin = BuiltinFunctionObject::new(Arc::from("__anext__"), builtin_identity);
    let builtin_ptr = Box::into_raw(Box::new(builtin)) as *const ();
    let builtin_value = Value::object_ptr(builtin_ptr);

    let bound = BoundMethod::new(builtin_value, Value::int(55).unwrap());
    let bound_ptr = Box::into_raw(Box::new(bound)) as *const ();
    let method = Value::object_ptr(bound_ptr);

    let result = call_unary_magic_method(&mut vm, method, Value::int(7).unwrap(), "__anext__")
        .expect("bound builtin unary call should succeed");
    assert_eq!(result.as_int(), Some(55));

    unsafe {
        drop(Box::from_raw(bound_ptr as *mut BoundMethod));
        drop(Box::from_raw(builtin_ptr as *mut BuiltinFunctionObject));
    }
}

#[test]
fn test_call_unary_magic_method_rejects_invalid_bound_target() {
    let mut vm = VirtualMachine::new();
    let bound = BoundMethod::new(Value::int(1).unwrap(), Value::none());
    let bound_ptr = Box::into_raw(Box::new(bound)) as *const ();
    let method = Value::object_ptr(bound_ptr);

    let err = call_unary_magic_method(&mut vm, method, Value::none(), "__aiter__")
        .expect_err("invalid bound target should fail");
    assert!(err.to_string().contains("non-callable"));

    unsafe {
        drop(Box::from_raw(bound_ptr as *mut BoundMethod));
    }
}

#[test]
fn test_call_unary_magic_method_rejects_non_callable() {
    let mut vm = VirtualMachine::new();
    let err = call_unary_magic_method(&mut vm, Value::int(1).unwrap(), Value::none(), "__anext__")
        .expect_err("non-callable should fail");
    assert!(err.to_string().contains("not callable"));
}
