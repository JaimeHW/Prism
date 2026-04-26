use super::*;
use crate::builtins::{ExceptionFlags, TYPE_ERROR, VALUE_ERROR};
use crate::stdlib::exceptions::ExceptionTypeId;
use prism_code::{CodeObject, Instruction, Opcode, Register};
use std::sync::Arc;

fn push_test_frame(vm: &mut VirtualMachine) {
    let mut code = CodeObject::new("exception_opcode_test", "<test>");
    code.register_count = 4;
    vm.push_frame(Arc::new(code), 0)
        .expect("failed to push test frame");
}

// ════════════════════════════════════════════════════════════════════════
// Type ID Extraction Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_no_type_id_constant() {
    assert_eq!(NO_TYPE_ID, 0xFFFF);
}

#[test]
fn test_no_handler_pc_constant() {
    assert_eq!(NO_HANDLER_PC, 0);
}

// ════════════════════════════════════════════════════════════════════════
// is_subclass Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_is_subclass_self() {
    // Every type is a subclass of itself
    assert!(is_subclass(4, 4)); // Exception is subclass of Exception
}

#[test]
fn test_is_subclass_direct() {
    // TypeError(24) is a subclass of Exception(4)
    assert!(is_subclass(24, 4));
}

#[test]
fn test_is_subclass_invalid_types() {
    // Invalid type IDs should return false
    assert!(!is_subclass(255, 255));
}

#[test]
fn test_is_subclass_not_related() {
    // TypeError(24) is not a subclass of StopIteration(5)
    assert!(!is_subclass(24, 5));
}

// ════════════════════════════════════════════════════════════════════════
// Handler Frame Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_handler_frame_size() {
    // Ensure compact memory layout
    assert_eq!(std::mem::size_of::<HandlerFrame>(), 8);
}

#[test]
fn test_no_handler_sentinel() {
    // Verify sentinel value
    use crate::exception::NO_HANDLER;
    assert_eq!(NO_HANDLER, u16::MAX);
}

// ════════════════════════════════════════════════════════════════════════
// Control Flow Return Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_control_flow_exception_size() {
    let cf = ControlFlow::Exception {
        type_id: 4,
        handler_pc: 100,
    };
    // Ensure ControlFlow is reasonably sized
    assert!(std::mem::size_of_val(&cf) <= 104);
}

#[test]
fn test_control_flow_reraise() {
    let cf = ControlFlow::Reraise;
    match cf {
        ControlFlow::Reraise => {}
        _ => panic!("Expected Reraise"),
    }
}

#[test]
fn test_control_flow_continue() {
    let cf = ControlFlow::Continue;
    match cf {
        ControlFlow::Continue => {}
        _ => panic!("Expected Continue"),
    }
}

// ════════════════════════════════════════════════════════════════════════
// Dynamic Match Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_check_dynamic_match_returns_false() {
    // Currently unimplemented, should return false
    assert!(!check_dynamic_match(None, 4, &Value::none()));
}

#[test]
fn test_check_tuple_match_returns_false() {
    // Currently unimplemented, should return false
    assert!(!check_tuple_match(None, 4, &Value::none()));
}

// ════════════════════════════════════════════════════════════════════════
// Extract Type ID Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_extract_type_id_no_type() {
    use prism_code::{Instruction, Opcode, Register};

    // Create instruction with NO_TYPE_ID
    let inst = Instruction::op_di(Opcode::Raise, Register(0), NO_TYPE_ID);
    let result = extract_type_id(inst, &Value::none());

    // Should return Exception type (4)
    assert_eq!(result, 4);
}

#[test]
fn test_extract_type_id_specific() {
    // Create instruction with specific type ID
    let inst = Instruction::op_di(Opcode::Raise, Register(0), 24); // TypeError
    let result = extract_type_id(inst, &Value::none());

    assert_eq!(result, 24);
}

#[test]
fn test_raise_normalizes_exception_type_object_to_instance() {
    let mut vm = VirtualMachine::new();
    push_test_frame(&mut vm);

    vm.current_frame_mut()
        .set_reg(0, unsafe { TYPE_ERROR.as_value() });

    let inst = Instruction::op_d(Opcode::Raise, Register(0));
    let control = raise(&mut vm, inst);

    assert!(matches!(
        control,
        ControlFlow::Exception {
            type_id: id,
            handler_pc: NO_HANDLER_PC
        } if id == ExceptionTypeId::TypeError as u16
    ));

    let active = vm
        .get_active_exception()
        .copied()
        .expect("active exception should be set");
    let active_exception =
        unsafe { ExceptionValue::from_value(active) }.expect("raise should materialize instance");

    assert_eq!(
        active_exception.exception_type_id,
        ExceptionTypeId::TypeError as u16
    );
    let traceback = active_exception
        .traceback()
        .expect("raised exception should expose traceback");
    assert_eq!(
        crate::ops::objects::extract_type_id(traceback.as_object_ptr().unwrap()),
        TypeId::TRACEBACK
    );
}

#[test]
fn test_raise_from_normalizes_exception_type_objects_to_instances() {
    let mut vm = VirtualMachine::new();
    push_test_frame(&mut vm);

    vm.current_frame_mut()
        .set_reg(0, unsafe { TYPE_ERROR.as_value() });
    vm.current_frame_mut()
        .set_reg(1, unsafe { VALUE_ERROR.as_value() });

    let inst = Instruction::op_ds(Opcode::RaiseFrom, Register(0), Register(1));
    let control = raise_from(&mut vm, inst);

    assert!(matches!(
        control,
        ControlFlow::Exception {
            type_id: id,
            handler_pc: NO_HANDLER_PC
        } if id == ExceptionTypeId::TypeError as u16
    ));

    let active = vm
        .get_active_exception()
        .copied()
        .expect("active exception should be set");
    let active_exception = unsafe { ExceptionValue::from_value(active) }
        .expect("raise_from should materialize instance");

    assert_eq!(
        active_exception.exception_type_id,
        ExceptionTypeId::TypeError as u16
    );
    assert!(active_exception.flags.has(ExceptionFlags::HAS_CAUSE));
    assert!(active_exception.flags.has(ExceptionFlags::SUPPRESS_CONTEXT));
    let traceback = active_exception
        .traceback()
        .expect("raised exception should expose traceback");
    assert_eq!(
        crate::ops::objects::extract_type_id(traceback.as_object_ptr().unwrap()),
        TypeId::TRACEBACK
    );

    let cause = unsafe {
        active_exception
            .cause
            .map(|ptr| &*ptr)
            .expect("explicit cause should be attached")
    };
    assert_eq!(cause.exception_type_id, ExceptionTypeId::ValueError as u16);
}
