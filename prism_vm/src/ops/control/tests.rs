use super::*;
use crate::builtins::create_exception_with_args;
use prism_code::{CodeObject, Instruction, Opcode, Register};
use prism_core::Value;
use std::sync::Arc;

fn push_test_frame(vm: &mut VirtualMachine) {
    let mut code = CodeObject::new("control_test", "<test>");
    code.register_count = 4;
    vm.push_frame(Arc::new(code), 0)
        .expect("failed to push test frame");
}

#[test]
fn test_raise_delegates_to_exception_handler() {
    let mut vm = VirtualMachine::new();
    push_test_frame(&mut vm);
    vm.current_frame_mut().set_reg(
        0,
        crate::builtins::create_exception(
            crate::stdlib::exceptions::ExceptionTypeId::ValueError,
            None,
        ),
    );

    let inst = Instruction::op_di(Opcode::Raise, Register::new(0), 0xFFFF);
    let control = raise(&mut vm, inst);
    assert!(matches!(
        control,
        ControlFlow::Exception { handler_pc: 0, .. }
    ));
}

#[test]
fn test_reraise_delegates_to_exception_handler() {
    let mut vm = VirtualMachine::new();
    push_test_frame(&mut vm);

    let inst = Instruction::op(Opcode::Reraise);
    let control = reraise(&mut vm, inst);
    assert!(matches!(control, ControlFlow::Error(_)));
}

#[test]
fn test_stop_iteration_value_reads_exception_payload() {
    let exc = create_exception_with_args(
        ExceptionTypeId::StopIteration,
        None,
        vec![Value::int(99).unwrap()].into_boxed_slice(),
    );
    let err = RuntimeError::raised_exception(
        ExceptionTypeId::StopIteration.as_u8() as u16,
        exc,
        "stop iteration",
    );

    assert_eq!(
        stop_iteration_value(&err).and_then(|value| value.as_int()),
        Some(99)
    );
}
