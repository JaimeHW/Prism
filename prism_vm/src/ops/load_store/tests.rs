use super::*;
use prism_code::{CodeObject, Instruction, Opcode, Register};
use prism_core::Value;
use prism_runtime::types::Cell;
use std::sync::Arc;

fn vm_with_frame(code: CodeObject) -> VirtualMachine {
    let mut vm = VirtualMachine::new();
    vm.push_frame(Arc::new(code), 0).expect("frame push failed");
    vm
}

// ==========================================================================
// Cell Integration Tests
// ==========================================================================

#[test]
fn test_cell_basic_get_set() {
    let cell = Cell::new(Value::int(42).unwrap());
    assert_eq!(cell.get(), Some(Value::int(42).unwrap()));

    cell.set(Value::int(100).unwrap());
    assert_eq!(cell.get(), Some(Value::int(100).unwrap()));
}

#[test]
fn test_cell_unbound() {
    let cell = Cell::unbound();
    assert!(cell.is_empty());
    assert_eq!(cell.get(), None);
}

#[test]
fn test_cell_clear_makes_unbound() {
    let cell = Cell::new(Value::int(42).unwrap());
    assert!(!cell.is_empty());

    cell.clear();
    assert!(cell.is_empty());
    assert_eq!(cell.get(), None);
}

#[test]
fn test_cell_shared_mutation() {
    // Simulate two closures sharing the same cell
    let cell = Arc::new(Cell::new(Value::int(1).unwrap()));
    let cell_clone = Arc::clone(&cell);

    // Both see initial value
    assert_eq!(cell.get(), Some(Value::int(1).unwrap()));
    assert_eq!(cell_clone.get(), Some(Value::int(1).unwrap()));

    // Mutate through one reference
    cell.set(Value::int(2).unwrap());

    // Both see updated value
    assert_eq!(cell.get(), Some(Value::int(2).unwrap()));
    assert_eq!(cell_clone.get(), Some(Value::int(2).unwrap()));
}

#[test]
fn test_cell_none_value_is_not_unbound() {
    // Python None is different from unbound (deleted)
    let cell = Cell::new(Value::none());
    assert!(!cell.is_empty());
    assert_eq!(cell.get(), Some(Value::none()));
}

#[test]
fn test_cell_atomic_thread_safety() {
    use std::thread;

    let cell = Arc::new(Cell::new(Value::int(0).unwrap()));
    let mut handles = vec![];

    // Spawn multiple threads that read and write
    for i in 0..4 {
        let cell_clone = Arc::clone(&cell);
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                cell_clone.set(Value::int(i).unwrap());
                let _ = cell_clone.get();
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Should not crash, and should have some valid value
    let final_val = cell.get();
    assert!(final_val.is_some());
}

// ==========================================================================
// ClosureEnv Integration Tests
// ==========================================================================

#[test]
fn test_closure_env_inline_storage() {
    use crate::frame::ClosureEnv;

    let cells = vec![
        Arc::new(Cell::new(Value::int(1).unwrap())),
        Arc::new(Cell::new(Value::int(2).unwrap())),
    ];

    let env = ClosureEnv::new(cells);
    assert_eq!(env.len(), 2);
    assert!(env.is_inline()); // <= 4 cells use inline storage

    assert_eq!(env.get(0), Value::int(1).unwrap());
    assert_eq!(env.get(1), Value::int(2).unwrap());
}

#[test]
fn test_closure_env_overflow_storage() {
    use crate::frame::ClosureEnv;

    let cells: Vec<_> = (0..6)
        .map(|i| Arc::new(Cell::new(Value::int(i).unwrap())))
        .collect();

    let env = ClosureEnv::new(cells);
    assert_eq!(env.len(), 6);
    assert!(!env.is_inline()); // > 4 cells use overflow

    for i in 0..6 {
        assert_eq!(env.get(i), Value::int(i as i64).unwrap());
    }
}

#[test]
fn test_closure_env_mutation() {
    use crate::frame::ClosureEnv;

    let cells = vec![Arc::new(Cell::new(Value::int(0).unwrap()))];
    let env = ClosureEnv::new(cells);

    assert_eq!(env.get(0), Value::int(0).unwrap());
    env.set(0, Value::int(42).unwrap());
    assert_eq!(env.get(0), Value::int(42).unwrap());
}

#[test]
fn test_closure_env_get_cell_for_unbound_check() {
    use crate::frame::ClosureEnv;

    let cells = vec![Arc::new(Cell::unbound())];
    let env = ClosureEnv::new(cells);

    // get_cell returns the cell, allowing unbound check
    let cell = env.get_cell(0);
    assert!(cell.is_empty());
    assert_eq!(cell.get(), None);
}

// ==========================================================================
// Error Type Tests
// ==========================================================================

#[test]
fn test_unbound_local_cell_error_message() {
    let err = crate::error::RuntimeError::unbound_local_cell(3);
    let msg = err.to_string();
    assert!(msg.contains("UnboundLocalError"));
    assert!(msg.contains("cell 3"));
}

#[test]
fn test_unbound_local_named_error_message() {
    let err = crate::error::RuntimeError::unbound_local("x");
    let msg = err.to_string();
    assert!(msg.contains("UnboundLocalError"));
    assert!(msg.contains("'x'"));
}

// ==========================================================================
// Store Opcode Semantics
// ==========================================================================

#[test]
fn test_store_local_uses_dst_register_for_source() {
    let mut vm = vm_with_frame(CodeObject::new("test_store_local", "<test>"));
    let frame = vm.current_frame_mut();
    frame.set_reg(9, Value::int(123).unwrap());
    frame.set_reg(0x23, Value::int(999).unwrap());

    // imm16=0x2345 => src1 byte is 0x23, local slot low byte is 0x45.
    let inst = Instruction::op_di(Opcode::StoreLocal, Register::new(9), 0x2345);
    assert!(matches!(store_local(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(vm.current_frame().get_reg(0x45).as_int(), Some(123));
}

#[test]
fn test_store_global_uses_dst_register_for_source() {
    let mut code = CodeObject::new("test_store_global", "<test>");
    code.names = vec!["unused".into(), "answer".into()].into_boxed_slice();

    let mut vm = vm_with_frame(code);
    let frame = vm.current_frame_mut();
    frame.set_reg(4, Value::int(42).unwrap());
    frame.set_reg(0, Value::int(7).unwrap());

    let inst = Instruction::op_di(Opcode::StoreGlobal, Register::new(4), 1);
    assert!(matches!(store_global(&mut vm, inst), ControlFlow::Continue));
    assert_eq!(vm.globals.get("answer").and_then(|v| v.as_int()), Some(42));
}

#[test]
fn test_load_local_raises_unbound_local_for_unwritten_slot() {
    let mut code = CodeObject::new("test_load_unbound_local", "<test>");
    code.locals = vec!["value".into()].into_boxed_slice();

    let mut vm = vm_with_frame(code);
    let inst = Instruction::op_di(Opcode::LoadLocal, Register::new(1), 0);

    match load_local(&mut vm, inst) {
        ControlFlow::Error(err) => {
            let message = err.to_string();
            assert!(message.contains("UnboundLocalError"));
            assert!(message.contains("'value'"));
        }
        other => panic!("expected unbound local error, got {other:?}"),
    }
}

#[test]
fn test_load_builtin_bypasses_shadowed_globals() {
    let mut code = CodeObject::new("test_load_builtin", "<test>");
    code.names = vec!["complex".into()].into_boxed_slice();

    let mut vm = vm_with_frame(code);
    vm.globals.set("complex".into(), Value::int(99).unwrap());

    let inst = Instruction::op_di(Opcode::LoadBuiltin, Register::new(4), 0);
    assert!(matches!(load_builtin(&mut vm, inst), ControlFlow::Continue));

    let value = vm.current_frame().get_reg(4);
    let ptr = value
        .as_object_ptr()
        .expect("builtin complex should be callable");
    assert_eq!(extract_type_id(ptr), TypeId::TYPE);
}

#[test]
fn test_store_closure_uses_dst_register_for_source() {
    use crate::frame::ClosureEnv;

    let mut vm = vm_with_frame(CodeObject::new("test_store_closure", "<test>"));
    let closure = Arc::new(ClosureEnv::with_unbound_cells(1));

    let frame = vm.current_frame_mut();
    frame.closure = Some(Arc::clone(&closure));
    frame.set_reg(5, Value::int(314).unwrap());
    frame.set_reg(0, Value::int(271).unwrap());

    let inst = Instruction::op_di(Opcode::StoreClosure, Register::new(5), 0);
    assert!(matches!(
        store_closure(&mut vm, inst),
        ControlFlow::Continue
    ));
    assert_eq!(closure.get(0).as_int(), Some(314));
}

// ==========================================================================
// Instruction Format Tests
// ==========================================================================

#[test]
fn test_load_closure_instruction_format() {
    let inst = Instruction::op_di(Opcode::LoadClosure, Register::new(0), 5);
    assert_eq!(inst.opcode(), Opcode::LoadClosure as u8);
    assert_eq!(inst.dst().0, 0);
    assert_eq!(inst.imm16(), 5);
}

#[test]
fn test_store_closure_instruction_format() {
    let inst = Instruction::op_di(Opcode::StoreClosure, Register::new(3), 7);
    assert_eq!(inst.opcode(), Opcode::StoreClosure as u8);
    assert_eq!(inst.dst().0, 3);
    assert_eq!(inst.imm16(), 7);
}

#[test]
fn test_delete_closure_instruction_format() {
    let inst = Instruction::op_di(Opcode::DeleteClosure, Register::new(0), 2);
    assert_eq!(inst.opcode(), Opcode::DeleteClosure as u8);
    assert_eq!(inst.imm16(), 2);
}

#[test]
fn test_load_builtin_instruction_format() {
    let inst = Instruction::op_di(Opcode::LoadBuiltin, Register::new(2), 11);
    assert_eq!(inst.opcode(), Opcode::LoadBuiltin as u8);
    assert_eq!(inst.dst().0, 2);
    assert_eq!(inst.imm16(), 11);
}

// ==========================================================================
// Opcode Registration Tests
// ==========================================================================

#[test]
fn test_delete_closure_opcode_value() {
    assert_eq!(Opcode::DeleteClosure as u8, 0x1D);
}

#[test]
fn test_closure_opcodes_in_load_store_range() {
    // All closure opcodes should be in the 0x10-0x1F range
    assert!(Opcode::LoadClosure as u8 >= 0x10);
    assert!(Opcode::LoadClosure as u8 <= 0x1F);
    assert!(Opcode::StoreClosure as u8 >= 0x10);
    assert!(Opcode::StoreClosure as u8 <= 0x1F);
    assert!(Opcode::DeleteClosure as u8 >= 0x10);
    assert!(Opcode::DeleteClosure as u8 <= 0x1F);
}

#[test]
fn test_opcode_from_u8_roundtrip() {
    let opcodes = [
        Opcode::LoadClosure,
        Opcode::StoreClosure,
        Opcode::DeleteClosure,
        Opcode::LoadBuiltin,
    ];

    for op in opcodes {
        let byte = op as u8;
        let recovered = Opcode::from_u8(byte);
        assert_eq!(recovered, Some(op));
    }
}
