use super::call_ex;
use crate::VirtualMachine;
use crate::builtins::{BuiltinError, BuiltinFunctionObject, builtin_iter, builtin_next};
use crate::dispatch::ControlFlow;
use prism_code::{CodeObject, Instruction, Opcode};
use prism_core::Value;
use prism_runtime::object::descriptor::BoundMethod;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::tuple::TupleObject;
use std::sync::Arc;

fn bound_method_probe(args: &[Value]) -> Result<Value, BuiltinError> {
    assert_eq!(args.len(), 2);
    assert_eq!(args[0].as_int(), Some(7));
    assert_eq!(args[1].as_int(), Some(11));
    Ok(args[1])
}

#[test]
fn test_call_ex_executes_bound_method_with_empty_kwargs_dict() {
    let mut code = CodeObject::new("call_ex_bound_method", "<test>");
    code.instructions = vec![Instruction::new(Opcode::CallKwEx, 3, 0, 0)].into_boxed_slice();

    let mut vm = VirtualMachine::new();
    vm.push_frame(Arc::new(code), 0)
        .expect("frame push should succeed");

    let builtin_ptr = Box::into_raw(Box::new(BuiltinFunctionObject::new(
        Arc::from("test.bound_method_probe_call_ex"),
        bound_method_probe,
    )));
    let method_ptr = Box::into_raw(Box::new(BoundMethod::new(
        Value::object_ptr(builtin_ptr as *const ()),
        Value::int(7).unwrap(),
    )));
    let args_tuple_ptr =
        Box::into_raw(Box::new(TupleObject::from_slice(
            &[Value::int(11).unwrap()],
        )));
    let kwargs_ptr = Box::into_raw(Box::new(DictObject::new()));

    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(method_ptr as *const ()));
    vm.current_frame_mut()
        .set_reg(2, Value::object_ptr(args_tuple_ptr as *const ()));
    vm.current_frame_mut()
        .set_reg(3, Value::object_ptr(kwargs_ptr as *const ()));

    let inst = Instruction::new(Opcode::CallEx, 0, 1, 2);
    assert!(matches!(call_ex(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(11));

    unsafe {
        drop(Box::from_raw(method_ptr));
        drop(Box::from_raw(args_tuple_ptr));
        drop(Box::from_raw(kwargs_ptr));
        drop(Box::from_raw(builtin_ptr));
    }
}

#[test]
fn test_build_tuple_unpack_expands_starred_list() {
    let mut code = CodeObject::new("build_tuple_unpack", "<test>");
    code.instructions = vec![Instruction::new(Opcode::CallKwEx, 1, 0, 0)].into_boxed_slice();

    let mut vm = VirtualMachine::new();
    vm.push_frame(Arc::new(code), 0)
        .expect("frame push should succeed");

    let list_ptr = Box::into_raw(Box::new(ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
    ])));
    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(list_ptr as *const ()));

    let inst = Instruction::new(Opcode::BuildTupleUnpack, 0, 1, 1);
    assert!(matches!(
        super::build_tuple_unpack(&mut vm, inst),
        ControlFlow::Continue
    ));

    let tuple_ptr = vm.current_frame().get_reg(0).as_object_ptr().unwrap();
    let tuple = unsafe { &*(tuple_ptr as *const TupleObject) };
    assert_eq!(tuple.len(), 2);
    assert_eq!(tuple.get(0).unwrap().as_int(), Some(1));
    assert_eq!(tuple.get(1).unwrap().as_int(), Some(2));

    unsafe {
        drop(Box::from_raw(list_ptr));
    }
}

#[test]
fn test_build_tuple_unpack_consumes_starred_iterator_in_place() {
    let mut code = CodeObject::new("build_tuple_unpack_iterator", "<test>");
    code.instructions = vec![Instruction::new(Opcode::CallKwEx, 1, 0, 0)].into_boxed_slice();

    let mut vm = VirtualMachine::new();
    vm.push_frame(Arc::new(code), 0)
        .expect("frame push should succeed");

    let list_ptr = Box::into_raw(Box::new(ListObject::from_slice(&[
        Value::int(4).unwrap(),
        Value::int(5).unwrap(),
    ])));
    let iterator = builtin_iter(&[Value::object_ptr(list_ptr as *const ())])
        .expect("iter(list) should create an iterator");
    vm.current_frame_mut().set_reg(1, iterator);

    let inst = Instruction::new(Opcode::BuildTupleUnpack, 0, 1, 1);
    assert!(matches!(
        super::build_tuple_unpack(&mut vm, inst),
        ControlFlow::Continue
    ));

    let tuple_ptr = vm.current_frame().get_reg(0).as_object_ptr().unwrap();
    let tuple = unsafe { &*(tuple_ptr as *const TupleObject) };
    assert_eq!(
        tuple.as_slice(),
        &[Value::int(4).unwrap(), Value::int(5).unwrap()]
    );

    let next_result = builtin_next(&[iterator]);
    assert!(matches!(next_result, Err(BuiltinError::StopIteration)));

    unsafe {
        drop(Box::from_raw(list_ptr));
    }
}
