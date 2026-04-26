use super::*;
use crate::builtins::ExceptionValue;
use crate::error::RuntimeErrorKind;
use crate::stdlib::exceptions::ExceptionTypeId;
use crate::stdlib::generators::LivenessMap;
use prism_code::{CodeFlags, CodeObject, Constant, ExceptionEntry, Instruction, Opcode, Register};
use std::sync::Arc;

fn boxed_constants(constants: Vec<Value>) -> Box<[Constant]> {
    constants
        .into_iter()
        .map(Constant::Value)
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn generator_value_for_state(state: GeneratorState) -> Value {
    let code = Arc::new(CodeObject::new("test_send", "<test>"));
    let mut generator = GeneratorObject::new(code);
    let regs = [Value::none(); 256];

    match state {
        GeneratorState::Created => {}
        GeneratorState::Suspended => {
            generator.try_start();
            generator.suspend(10, 1, &regs, LivenessMap::from_bits(0b1));
        }
        GeneratorState::Running => {
            generator.try_start();
        }
        GeneratorState::Closed => {
            generator.try_start();
            generator.exhaust();
        }
        GeneratorState::NotAGenerator => {}
    }

    let ptr = Box::into_raw(Box::new(generator)) as *const ();
    Value::object_ptr(ptr)
}

fn push_caller_frame(vm: &mut VirtualMachine) {
    let mut caller = CodeObject::new("send_caller", "<test>");
    caller.register_count = 16;
    vm.push_frame(Arc::new(caller), 0)
        .expect("failed to push caller frame");
}

fn runtime_send_generator() -> Value {
    let mut code = CodeObject::new("runtime_send_generator", "<test>");
    code.flags = CodeFlags::GENERATOR;
    code.register_count = 8;
    code.constants = boxed_constants(vec![Value::int(1).unwrap()]);
    code.instructions = vec![
        // r2 = 1
        Instruction::op_di(Opcode::LoadConst, Register::new(2), 0),
        // first yield: yields r2, sent value lands in r1
        Instruction::op_ds(Opcode::Yield, Register::new(1), Register::new(2)),
        // second yield: yields last sent value from r1
        Instruction::op_ds(Opcode::Yield, Register::new(1), Register::new(1)),
        // stop
        Instruction::op(Opcode::ReturnNone),
    ]
    .into_boxed_slice();

    let generator = GeneratorObject::from_code(Arc::new(code));
    let ptr = Box::into_raw(Box::new(generator)) as *const ();
    Value::object_ptr(ptr)
}

fn runtime_raise_generator() -> Value {
    let mut code = CodeObject::new("runtime_raise_generator", "<test>");
    code.flags = CodeFlags::GENERATOR;
    code.register_count = 8;
    code.constants = boxed_constants(vec![create_exception(
        ExceptionTypeId::TypeError,
        Some(Arc::from("boom from generator")),
    )]);
    code.instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register::new(2), 0),
        Instruction::op_di(
            Opcode::Raise,
            Register::new(2),
            ExceptionTypeId::TypeError as u16,
        ),
        Instruction::op(Opcode::ReturnNone),
    ]
    .into_boxed_slice();

    let generator = GeneratorObject::from_code(Arc::new(code));
    let ptr = Box::into_raw(Box::new(generator)) as *const ();
    Value::object_ptr(ptr)
}

fn runtime_handled_raise_generator() -> Value {
    let mut code = CodeObject::new("runtime_handled_raise_generator", "<test>");
    code.flags = CodeFlags::GENERATOR;
    code.register_count = 8;
    code.constants = boxed_constants(vec![
        create_exception(
            ExceptionTypeId::TypeError,
            Some(Arc::from("caught in generator")),
        ),
        Value::int(9).unwrap(),
    ]);
    code.instructions = vec![
        // Raise TypeError from try-range [pc=1, pc=2)
        Instruction::op_di(Opcode::LoadConst, Register::new(2), 0),
        Instruction::op_di(
            Opcode::Raise,
            Register::new(2),
            ExceptionTypeId::TypeError as u16,
        ),
        Instruction::op(Opcode::ReturnNone),
        // Exception handler target: yield sentinel 9 and keep normal generator protocol.
        Instruction::op_di(Opcode::LoadConst, Register::new(3), 1),
        Instruction::op_ds(Opcode::Yield, Register::new(1), Register::new(3)),
        Instruction::op(Opcode::ReturnNone),
    ]
    .into_boxed_slice();
    code.exception_table = vec![ExceptionEntry {
        start_pc: 1,
        end_pc: 2,
        handler_pc: 3,
        finally_pc: u32::MAX,
        depth: 0,
        exception_type_idx: u16::MAX,
    }]
    .into_boxed_slice();

    let generator = GeneratorObject::from_code(Arc::new(code));
    let ptr = Box::into_raw(Box::new(generator)) as *const ();
    Value::object_ptr(ptr)
}

fn runtime_return_value_generator() -> Value {
    let mut code = CodeObject::new("runtime_return_value_generator", "<test>");
    code.flags = CodeFlags::GENERATOR;
    code.register_count = 8;
    code.constants = boxed_constants(vec![Value::int(42).unwrap()]);
    code.instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register::new(2), 0),
        Instruction::op_d(Opcode::Return, Register::new(2)),
    ]
    .into_boxed_slice();

    let generator = GeneratorObject::from_code(Arc::new(code));
    let ptr = Box::into_raw(Box::new(generator)) as *const ();
    Value::object_ptr(ptr)
}

// =========================================================================
// Generator State Tests
// =========================================================================

#[test]
fn test_none_not_generator() {
    assert_eq!(
        get_generator_state(&Value::none()),
        GeneratorState::NotAGenerator
    );
}

#[test]
fn test_int_not_generator() {
    let val = Value::int(42).unwrap();
    assert_eq!(get_generator_state(&val), GeneratorState::NotAGenerator);
}

#[test]
fn test_bool_not_generator() {
    assert_eq!(
        get_generator_state(&Value::bool(true)),
        GeneratorState::NotAGenerator
    );
}

#[test]
fn test_float_not_generator() {
    let val = Value::float(3.14);
    assert_eq!(get_generator_state(&val), GeneratorState::NotAGenerator);
}

#[test]
fn test_generator_created_state_detected() {
    let generator_value = generator_value_for_state(GeneratorState::Created);
    assert_eq!(
        get_generator_state(&generator_value),
        GeneratorState::Created
    );
}

#[test]
fn test_generator_suspended_state_detected() {
    let generator_value = generator_value_for_state(GeneratorState::Suspended);
    assert_eq!(
        get_generator_state(&generator_value),
        GeneratorState::Suspended
    );
}

#[test]
fn test_generator_running_state_detected() {
    let generator_value = generator_value_for_state(GeneratorState::Running);
    assert_eq!(
        get_generator_state(&generator_value),
        GeneratorState::Running
    );
}

#[test]
fn test_generator_closed_state_detected() {
    let generator_value = generator_value_for_state(GeneratorState::Closed);
    assert_eq!(
        get_generator_state(&generator_value),
        GeneratorState::Closed
    );
}

// =========================================================================
// Type Name Tests
// =========================================================================

#[test]
fn test_type_name_none() {
    assert_eq!(type_name(&Value::none()), "NoneType");
}

#[test]
fn test_type_name_bool() {
    assert_eq!(type_name(&Value::bool(true)), "bool");
}

#[test]
fn test_type_name_int() {
    let val = Value::int(42).unwrap();
    assert_eq!(type_name(&val), "int");
}

#[test]
fn test_type_name_float() {
    let val = Value::float(3.14);
    assert_eq!(type_name(&val), "float");
}

#[test]
fn test_type_name_generator() {
    let generator_value = generator_value_for_state(GeneratorState::Created);
    assert_eq!(type_name(&generator_value), "generator");
}

#[test]
fn test_send_resumes_generator_and_yields_sent_value() {
    let mut vm = VirtualMachine::new();
    push_caller_frame(&mut vm);

    let generator = runtime_send_generator();
    vm.current_frame_mut().set_reg(1, generator);

    // First resume must send None and should yield constant 1.
    vm.current_frame_mut().set_reg(2, Value::none());
    let inst = Instruction::new(Opcode::Send, 0, 1, 2);
    let control = send(&mut vm, inst);
    assert!(matches!(control, ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(1));
    assert_eq!(
        GeneratorObject::from_value(generator)
            .expect("generator")
            .state(),
        RuntimeGeneratorState::Suspended
    );

    // Second resume sends 77 and generator immediately yields it back.
    vm.current_frame_mut().set_reg(2, Value::int(77).unwrap());
    let control = send(&mut vm, inst);
    assert!(matches!(control, ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(77));
    assert_eq!(
        GeneratorObject::from_value(generator)
            .expect("generator")
            .state(),
        RuntimeGeneratorState::Suspended
    );

    // Third resume reaches return and raises StopIteration.
    vm.current_frame_mut().set_reg(2, Value::none());
    let control = send(&mut vm, inst);
    match control {
        ControlFlow::Exception { type_id, .. } => {
            assert_eq!(type_id, ExceptionTypeId::StopIteration as u16);
        }
        other => panic!("expected StopIteration, got {other:?}"),
    }
    assert_eq!(
        GeneratorObject::from_value(generator)
            .expect("generator")
            .state(),
        RuntimeGeneratorState::Exhausted
    );
}

#[test]
fn test_send_rejects_non_none_on_initial_resume() {
    let mut vm = VirtualMachine::new();
    push_caller_frame(&mut vm);

    let generator = runtime_send_generator();
    vm.current_frame_mut().set_reg(1, generator);
    vm.current_frame_mut().set_reg(2, Value::int(5).unwrap());

    let inst = Instruction::new(Opcode::Send, 0, 1, 2);
    let control = send(&mut vm, inst);
    match control {
        ControlFlow::Error(err) => {
            assert!(err.to_string().contains("can't send non-None"));
        }
        other => panic!("expected TypeError, got {other:?}"),
    }

    // Generator should remain in created state after protocol violation.
    assert_eq!(
        GeneratorObject::from_value(generator)
            .expect("generator")
            .state(),
        RuntimeGeneratorState::Created
    );
}

#[test]
fn test_send_propagates_uncaught_generator_exception() {
    let mut vm = VirtualMachine::new();
    push_caller_frame(&mut vm);

    let generator = runtime_raise_generator();
    vm.current_frame_mut().set_reg(1, generator);
    vm.current_frame_mut().set_reg(2, Value::none());

    let inst = Instruction::new(Opcode::Send, 0, 1, 2);
    let control = send(&mut vm, inst);
    match control {
        ControlFlow::Error(err) => {
            assert!(matches!(
                err.kind,
                RuntimeErrorKind::Exception { type_id, .. }
                    if type_id == ExceptionTypeId::TypeError as u16
            ));
        }
        other => panic!("expected propagated exception, got {other:?}"),
    }
    assert_eq!(
        GeneratorObject::from_value(generator)
            .expect("generator")
            .state(),
        RuntimeGeneratorState::Exhausted
    );
}

#[test]
fn test_send_handles_generator_exception_table_path() {
    let mut vm = VirtualMachine::new();
    push_caller_frame(&mut vm);

    let generator = runtime_handled_raise_generator();
    vm.current_frame_mut().set_reg(1, generator);
    vm.current_frame_mut().set_reg(2, Value::none());
    let inst = Instruction::new(Opcode::Send, 0, 1, 2);

    // First send triggers raise, catches in generator exception table, then yields 9.
    let control = send(&mut vm, inst);
    assert!(matches!(control, ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(9));
    assert_eq!(
        GeneratorObject::from_value(generator)
            .expect("generator")
            .state(),
        RuntimeGeneratorState::Suspended
    );

    // Next send cleanly completes generator.
    vm.current_frame_mut().set_reg(2, Value::none());
    let control = send(&mut vm, inst);
    match control {
        ControlFlow::Exception { type_id, .. } => {
            assert_eq!(type_id, ExceptionTypeId::StopIteration as u16);
        }
        other => panic!("expected StopIteration, got {other:?}"),
    }
    assert_eq!(
        GeneratorObject::from_value(generator)
            .expect("generator")
            .state(),
        RuntimeGeneratorState::Exhausted
    );
}

#[test]
fn test_send_sets_stop_iteration_value_from_generator_return() {
    let mut vm = VirtualMachine::new();
    push_caller_frame(&mut vm);

    let generator = runtime_return_value_generator();
    vm.current_frame_mut().set_reg(1, generator);
    vm.current_frame_mut().set_reg(2, Value::none());
    let inst = Instruction::new(Opcode::Send, 0, 1, 2);

    let control = send(&mut vm, inst);
    match control {
        ControlFlow::Exception { type_id, .. } => {
            assert_eq!(type_id, ExceptionTypeId::StopIteration as u16);
        }
        other => panic!("expected StopIteration exception flow, got {other:?}"),
    }

    let exc_value = vm
        .get_active_exception()
        .copied()
        .expect("stop iteration should be active");
    let exc = unsafe {
        ExceptionValue::from_value(exc_value)
            .expect("active exception should be an ExceptionValue object")
    };
    let args = exc
        .args
        .as_ref()
        .expect("StopIteration should carry return value in args");
    assert_eq!(args.len(), 1);
    assert_eq!(args[0].as_int(), Some(42));

    assert_eq!(
        GeneratorObject::from_value(generator)
            .expect("generator")
            .state(),
        RuntimeGeneratorState::Exhausted
    );
}
