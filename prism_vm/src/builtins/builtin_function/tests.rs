use super::*;
use crate::VirtualMachine;

fn dummy_builtin(_args: &[Value]) -> Result<Value, super::super::BuiltinError> {
    Ok(Value::none())
}

fn dummy_vm_builtin(
    _vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, super::super::BuiltinError> {
    Ok(Value::int(args.len() as i64).unwrap())
}

fn dummy_kw_builtin(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, super::super::BuiltinError> {
    Ok(Value::int((args.len() + keywords.len()) as i64).unwrap())
}

#[test]
fn test_builtin_function_object_creation() {
    let func = BuiltinFunctionObject::new(Arc::from("test"), dummy_builtin);
    assert_eq!(func.name(), "test");
    assert_eq!(func.header.type_id, TypeId::BUILTIN_FUNCTION);
}

#[test]
fn test_builtin_function_object_call() {
    let func = BuiltinFunctionObject::new(Arc::from("test"), dummy_builtin);
    let result = func.call(&[]);
    assert!(result.is_ok());
}

#[test]
fn test_vm_builtin_requires_vm_context() {
    let func = BuiltinFunctionObject::new_vm(Arc::from("test.vm"), dummy_vm_builtin);
    let err = func
        .call(&[])
        .expect_err("vm builtin should reject missing VM");
    assert!(err.to_string().contains("requires VM context"));
}

#[test]
fn test_vm_builtin_function_object_call_with_vm() {
    let func = BuiltinFunctionObject::new_vm(Arc::from("test.vm"), dummy_vm_builtin);
    let mut vm = VirtualMachine::new();
    let result = func
        .call_with_vm(&mut vm, &[Value::int(1).unwrap(), Value::int(2).unwrap()])
        .expect("vm-aware builtin should execute");
    assert_eq!(result.as_int(), Some(2));
}

fn capture_args(args: &[Value]) -> Result<Value, super::super::BuiltinError> {
    Ok(Value::int(args.len() as i64).unwrap())
}

#[test]
fn test_builtin_function_object_bind_prepends_receiver() {
    let func = BuiltinFunctionObject::new(Arc::from("capture"), capture_args);
    let bound = func.bind(Value::int(10).unwrap());
    let result = bound.call(&[Value::int(20).unwrap(), Value::int(30).unwrap()]);
    assert_eq!(result.unwrap().as_int(), Some(3));
    assert_eq!(bound.header.type_id, TypeId::BUILTIN_FUNCTION);
}

#[test]
fn test_vm_builtin_bind_prepends_receiver() {
    let func = BuiltinFunctionObject::new_vm(Arc::from("capture.vm"), dummy_vm_builtin);
    let bound = func.bind(Value::int(10).unwrap());
    let mut vm = VirtualMachine::new();
    let result = bound
        .call_with_vm(&mut vm, &[Value::int(20).unwrap(), Value::int(30).unwrap()])
        .expect("vm-aware bound builtin should execute");
    assert_eq!(result.as_int(), Some(3));
}

#[test]
fn test_keyword_builtin_accepts_keywords_and_binding() {
    let func = BuiltinFunctionObject::new_kw(Arc::from("capture.kw"), dummy_kw_builtin);
    assert!(func.accepts_keywords());

    let result = func
        .call_with_keywords(
            &[Value::int(1).unwrap()],
            &[("name", Value::int(2).unwrap())],
        )
        .expect("keyword-aware builtin should execute");
    assert_eq!(result.as_int(), Some(2));

    let bound = func.bind(Value::int(10).unwrap());
    let bound_result = bound
        .call_with_keywords(&[], &[("flag", Value::bool(true))])
        .expect("bound keyword-aware builtin should execute");
    assert_eq!(bound_result.as_int(), Some(2));
}

#[test]
fn test_bound_builtin_trace_visits_receiver() {
    struct CountingTracer {
        values: usize,
    }

    impl Tracer for CountingTracer {
        fn trace_value(&mut self, _value: Value) {
            self.values += 1;
        }

        fn trace_ptr(&mut self, _ptr: *const ()) {}
    }

    let func = BuiltinFunctionObject::new(Arc::from("capture"), capture_args);
    let bound = func.bind(Value::int(10).unwrap());
    let mut tracer = CountingTracer { values: 0 };

    bound.trace(&mut tracer);

    assert_eq!(tracer.values, 1);
}
