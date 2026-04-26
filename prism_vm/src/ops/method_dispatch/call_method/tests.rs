use super::*;
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use prism_code::{CodeObject, Instruction, Opcode, Register};
use prism_runtime::types::function::FunctionObject;
use prism_runtime::types::list::ListObject;
use std::sync::Arc;

fn push_test_frame(vm: &mut VirtualMachine, name: &str) {
    let mut code = CodeObject::new(name, "<test>");
    code.register_count = 16;
    vm.push_frame(Arc::new(code), 0)
        .expect("failed to push test frame");
}

fn make_test_function_value(
    name: &str,
    arg_count: u16,
    return_reg: Register,
) -> (*mut FunctionObject, Value) {
    let mut code = CodeObject::new(name, "<test>");
    code.register_count = 16;
    code.arg_count = arg_count;
    code.instructions = vec![Instruction::op_d(Opcode::Return, return_reg)].into_boxed_slice();
    let func = Box::new(FunctionObject::new(
        Arc::new(code),
        Arc::from(name),
        None,
        None,
    ));
    let ptr = Box::into_raw(func);
    (ptr, Value::object_ptr(ptr as *const ()))
}

fn builtin_arg_count(args: &[Value]) -> Result<Value, BuiltinError> {
    Ok(Value::int(args.len() as i64).expect("arg count should fit in tagged int"))
}

fn make_builtin_value(name: &str) -> (*mut BuiltinFunctionObject, Value) {
    let builtin = Box::new(BuiltinFunctionObject::new(
        Arc::from(name),
        builtin_arg_count,
    ));
    let ptr = Box::into_raw(builtin);
    (ptr, Value::object_ptr(ptr as *const ()))
}

#[test]
fn test_extract_type_id() {
    let list = Box::new(ListObject::new());
    let ptr = Box::into_raw(list) as *const ();

    let type_id = extract_type_id(ptr);
    assert_eq!(type_id, TypeId::LIST);

    // Clean up
    unsafe {
        drop(Box::from_raw(ptr as *mut ListObject));
    }
}

#[test]
fn test_implicit_self_from_slot_none_marker() {
    assert!(implicit_self_from_slot(Value::none()).is_none());
    let self_value = Value::int(7).unwrap();
    assert_eq!(implicit_self_from_slot(self_value), Some(self_value));
}

#[test]
fn test_call_method_user_function_without_implicit_self() {
    let mut vm = VirtualMachine::new();
    push_test_frame(&mut vm, "caller");

    let (func_ptr, func_value) = make_test_function_value("callee", 1, Register::new(0));
    vm.current_frame_mut().set_reg(1, func_value);
    vm.current_frame_mut().set_reg(2, Value::none()); // None marker => no implicit self
    vm.current_frame_mut().set_reg(3, Value::int(42).unwrap());

    let inst = Instruction::new(Opcode::CallMethod, 0, 1, 1);
    let control = call_method(&mut vm, inst);
    assert!(matches!(control, ControlFlow::Continue));
    assert_eq!(vm.call_depth(), 2);
    assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(42));
    assert!(vm.current_frame().get_reg(1).is_none());

    vm.clear_frames();
    unsafe {
        drop(Box::from_raw(func_ptr));
    }
}

#[test]
fn test_call_method_user_function_with_implicit_self() {
    let mut vm = VirtualMachine::new();
    push_test_frame(&mut vm, "caller");

    let (func_ptr, func_value) = make_test_function_value("callee", 2, Register::new(0));
    vm.current_frame_mut().set_reg(1, func_value);
    vm.current_frame_mut().set_reg(2, Value::int(7).unwrap());
    vm.current_frame_mut().set_reg(3, Value::int(42).unwrap());

    let inst = Instruction::new(Opcode::CallMethod, 0, 1, 1);
    let control = call_method(&mut vm, inst);
    assert!(matches!(control, ControlFlow::Continue));
    assert_eq!(vm.call_depth(), 2);
    assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(7));
    assert_eq!(vm.current_frame().get_reg(1).as_int(), Some(42));

    vm.clear_frames();
    unsafe {
        drop(Box::from_raw(func_ptr));
    }
}

#[test]
fn test_call_method_builtin_respects_none_marker() {
    let mut vm = VirtualMachine::new();
    push_test_frame(&mut vm, "caller");

    let (builtin_ptr, builtin_value) = make_builtin_value("argc");
    vm.current_frame_mut().set_reg(1, builtin_value);
    vm.current_frame_mut().set_reg(2, Value::none()); // None marker => no implicit self
    vm.current_frame_mut().set_reg(3, Value::int(99).unwrap());

    let inst = Instruction::new(Opcode::CallMethod, 0, 1, 1);
    let control = call_method(&mut vm, inst);
    assert!(matches!(control, ControlFlow::Continue));
    assert_eq!(vm.call_depth(), 1);
    assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(1));

    vm.clear_frames();
    unsafe {
        drop(Box::from_raw(builtin_ptr));
    }
}

#[test]
fn test_call_method_builtin_includes_implicit_self_when_present() {
    let mut vm = VirtualMachine::new();
    push_test_frame(&mut vm, "caller");

    let (builtin_ptr, builtin_value) = make_builtin_value("argc");
    vm.current_frame_mut().set_reg(1, builtin_value);
    vm.current_frame_mut().set_reg(2, Value::int(1).unwrap());
    vm.current_frame_mut().set_reg(3, Value::int(99).unwrap());

    let inst = Instruction::new(Opcode::CallMethod, 0, 1, 1);
    let control = call_method(&mut vm, inst);
    assert!(matches!(control, ControlFlow::Continue));
    assert_eq!(vm.call_depth(), 1);
    assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(2));

    vm.clear_frames();
    unsafe {
        drop(Box::from_raw(builtin_ptr));
    }
}

#[test]
fn test_call_method_supports_type_objects_without_implicit_self() {
    let mut vm = VirtualMachine::new();
    push_test_frame(&mut vm, "caller");

    vm.current_frame_mut().set_reg(
        1,
        crate::builtins::builtin_type_object_for_type_id(TypeId::DICT),
    );
    vm.current_frame_mut().set_reg(2, Value::none());

    let inst = Instruction::new(Opcode::CallMethod, 0, 1, 0);
    let control = call_method(&mut vm, inst);
    assert!(matches!(control, ControlFlow::Continue));

    let result_ptr = vm
        .current_frame()
        .get_reg(0)
        .as_object_ptr()
        .expect("dict() should return a heap object");
    assert_eq!(extract_type_id(result_ptr), TypeId::DICT);
}

#[test]
fn test_call_method_supports_nested_bound_methods() {
    let mut vm = VirtualMachine::new();
    push_test_frame(&mut vm, "caller");

    let (builtin_ptr, builtin_value) = make_builtin_value("argc");
    let inner_ptr = Box::into_raw(Box::new(BoundMethod::new(
        builtin_value,
        Value::int(11).unwrap(),
    )));
    let inner_value = Value::object_ptr(inner_ptr as *const ());
    let outer_ptr = Box::into_raw(Box::new(BoundMethod::new(
        inner_value,
        Value::int(22).unwrap(),
    )));
    let outer_value = Value::object_ptr(outer_ptr as *const ());

    vm.current_frame_mut().set_reg(1, outer_value);
    vm.current_frame_mut().set_reg(2, Value::none());
    vm.current_frame_mut().set_reg(3, Value::int(99).unwrap());

    let inst = Instruction::new(Opcode::CallMethod, 0, 1, 1);
    let control = call_method(&mut vm, inst);
    assert!(matches!(control, ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(3));

    vm.clear_frames();
    unsafe {
        drop(Box::from_raw(outer_ptr));
        drop(Box::from_raw(inner_ptr));
        drop(Box::from_raw(builtin_ptr));
    }
}
