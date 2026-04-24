//! Load and store opcode handlers.
//!
//! Handles loading constants, locals, globals, closures, and register moves.

use crate::VirtualMachine;
use crate::dispatch::ControlFlow;
use crate::error::{RuntimeError, RuntimeErrorKind};
use crate::ops::calls::invoke_callable_value;
use crate::ops::objects::{extract_type_id, get_attribute_value};
use crate::stdlib::exceptions::ExceptionTypeId;
use prism_code::{CodeFlags, Instruction};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use std::sync::Arc;

// =============================================================================
// Constants
// =============================================================================

/// LoadConst: dst = constants[imm16]
#[inline(always)]
pub fn load_const(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let value = frame.get_const(inst.imm16());
    frame.set_reg(inst.dst().0, value);
    ControlFlow::Continue
}

/// LoadNone: dst = None
#[inline(always)]
pub fn load_none(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    vm.current_frame_mut().set_reg(inst.dst().0, Value::none());
    ControlFlow::Continue
}

/// LoadTrue: dst = True
#[inline(always)]
pub fn load_true(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    vm.current_frame_mut()
        .set_reg(inst.dst().0, Value::bool(true));
    ControlFlow::Continue
}

/// LoadFalse: dst = False
#[inline(always)]
pub fn load_false(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    vm.current_frame_mut()
        .set_reg(inst.dst().0, Value::bool(false));
    ControlFlow::Continue
}

// =============================================================================
// Locals
// =============================================================================

/// LoadLocal: dst = frame.registers[imm16]
/// Note: In our register-based VM, locals ARE registers. This opcode
/// may be used for explicit local variable access semantics.
#[inline(always)]
pub fn load_local(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let slot = inst.imm16() as u8;
    let dst = inst.dst().0;

    if let Some(mapping) = vm.current_frame().locals_mapping() {
        let name = mapped_local_name(vm.current_frame(), slot);
        return match load_mapped_local(vm, mapping, &name) {
            Ok(value) => {
                vm.current_frame_mut().set_reg(dst, value);
                ControlFlow::Continue
            }
            Err(err) => ControlFlow::Error(err),
        };
    }

    {
        let frame = vm.current_frame();
        if frame.code.flags.contains(CodeFlags::CLASS) && !frame.reg_is_written(slot) {
            let name = mapped_local_name(frame, slot);
            return match load_module_or_builtin(vm, &name) {
                Ok(value) => {
                    vm.current_frame_mut().set_reg(dst, value);
                    ControlFlow::Continue
                }
                Err(err) => ControlFlow::Error(err),
            };
        }
    }

    let frame = vm.current_frame_mut();
    let value = frame.get_reg(slot);
    frame.set_reg(dst, value);
    ControlFlow::Continue
}

/// StoreLocal: frame.registers[imm16] = dst-register
#[inline(always)]
pub fn store_local(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let slot = inst.imm16() as u8;
    if let Some(mapping) = vm.current_frame().locals_mapping() {
        let (value, name) = {
            let frame = vm.current_frame();
            (frame.get_reg(inst.dst().0), mapped_local_name(frame, slot))
        };
        return match store_mapped_local(vm, mapping, &name, value) {
            Ok(()) => {
                vm.current_frame_mut().set_reg(slot, value);
                ControlFlow::Continue
            }
            Err(err) => ControlFlow::Error(err),
        };
    }

    let frame = vm.current_frame_mut();
    // Store opcodes use DstImm16 encoding where the source register is carried
    // in the dst field and imm16 is the destination slot index.
    let value = frame.get_reg(inst.dst().0);
    frame.set_reg(slot, value);
    ControlFlow::Continue
}

/// DeleteLocal: frame.registers[imm16] = undefined
#[inline(always)]
pub fn delete_local(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let slot = inst.imm16() as u8;
    if let Some(mapping) = vm.current_frame().locals_mapping() {
        let name = mapped_local_name(vm.current_frame(), slot);
        return match delete_mapped_local(vm, mapping, &name) {
            Ok(true) => {
                vm.current_frame_mut().clear_reg(slot);
                ControlFlow::Continue
            }
            Ok(false) => ControlFlow::Error(RuntimeError::name_error(name)),
            Err(err) => ControlFlow::Error(err),
        };
    }

    let frame = vm.current_frame_mut();
    frame.clear_reg(slot);
    ControlFlow::Continue
}

#[inline]
fn mapped_local_name(frame: &crate::frame::Frame, slot: u8) -> Arc<str> {
    Arc::clone(frame.get_local_name(slot as u16))
}

#[inline]
fn mapped_local_key(name: &Arc<str>) -> Value {
    Value::string(intern(name.as_ref()))
}

fn load_mapped_local(
    vm: &mut VirtualMachine,
    mapping: Value,
    name: &Arc<str>,
) -> Result<Value, RuntimeError> {
    if let Some(value) = lookup_mapped_local(vm, mapping, name)? {
        return Ok(value);
    }

    load_module_or_builtin(vm, name)
}

fn load_module_or_builtin(vm: &mut VirtualMachine, name: &Arc<str>) -> Result<Value, RuntimeError> {
    if let Some(value) = vm.module_scope_value(name) {
        return Ok(value);
    }

    if let Some(value) = vm.builtins.get(name) {
        return Ok(value);
    }

    Err(RuntimeError::name_error(Arc::clone(name)))
}

fn lookup_mapped_local(
    vm: &mut VirtualMachine,
    mapping: Value,
    name: &Arc<str>,
) -> Result<Option<Value>, RuntimeError> {
    let key = mapped_local_key(name);

    if let Some(ptr) = mapping.as_object_ptr()
        && extract_type_id(ptr) == TypeId::DICT
    {
        let dict = unsafe { &*(ptr as *const DictObject) };
        return Ok(dict.get(key));
    }

    let getitem_name = intern("__getitem__");
    let getitem = get_attribute_value(vm, mapping, &getitem_name).map_err(|err| {
        if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) {
            RuntimeError::type_error("class namespace mapping must define __getitem__")
        } else {
            err
        }
    })?;

    match invoke_callable_value(vm, getitem, &[key]) {
        Ok(value) => Ok(Some(value)),
        Err(err) if is_missing_mapping_key_error(&err) => Ok(None),
        Err(err) => Err(err),
    }
}

fn store_mapped_local(
    vm: &mut VirtualMachine,
    mapping: Value,
    name: &Arc<str>,
    value: Value,
) -> Result<(), RuntimeError> {
    let key = mapped_local_key(name);

    if let Some(ptr) = mapping.as_object_ptr()
        && extract_type_id(ptr) == TypeId::DICT
    {
        let dict = unsafe { &mut *(ptr as *mut DictObject) };
        dict.set(key, value);
        return Ok(());
    }

    let setitem_name = intern("__setitem__");
    let setitem = get_attribute_value(vm, mapping, &setitem_name).map_err(|err| {
        if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) {
            RuntimeError::type_error("class namespace mapping must define __setitem__")
        } else {
            err
        }
    })?;
    invoke_callable_value(vm, setitem, &[key, value])?;
    Ok(())
}

fn delete_mapped_local(
    vm: &mut VirtualMachine,
    mapping: Value,
    name: &Arc<str>,
) -> Result<bool, RuntimeError> {
    let key = mapped_local_key(name);

    if let Some(ptr) = mapping.as_object_ptr()
        && extract_type_id(ptr) == TypeId::DICT
    {
        let dict = unsafe { &mut *(ptr as *mut DictObject) };
        return Ok(dict.remove(key).is_some());
    }

    let delitem_name = intern("__delitem__");
    let delitem = get_attribute_value(vm, mapping, &delitem_name).map_err(|err| {
        if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) {
            RuntimeError::type_error("class namespace mapping must define __delitem__")
        } else {
            err
        }
    })?;

    match invoke_callable_value(vm, delitem, &[key]) {
        Ok(_) => Ok(true),
        Err(err) if is_missing_mapping_key_error(&err) => Ok(false),
        Err(err) => Err(err),
    }
}

#[inline]
fn is_missing_mapping_key_error(err: &RuntimeError) -> bool {
    matches!(err.kind, RuntimeErrorKind::KeyError { .. })
        || matches!(
            err.kind,
            RuntimeErrorKind::Exception { type_id, .. }
                if type_id == ExceptionTypeId::KeyError.as_u8() as u16
        )
        || matches!(
            &err.kind,
            RuntimeErrorKind::InternalError { message } if message.as_ref() == "key not found"
        )
}

// =============================================================================
// Globals
// =============================================================================

/// LoadGlobal: dst = globals[names[imm16]]
#[inline(always)]
pub fn load_global(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let (name, locals_mapping) = {
        let frame = vm.current_frame();
        (frame.get_name(inst.imm16()).clone(), frame.locals_mapping())
    };

    if let Some(mapping) = locals_mapping {
        return match load_mapped_local(vm, mapping, &name) {
            Ok(value) => {
                vm.current_frame_mut().set_reg(inst.dst().0, value);
                ControlFlow::Continue
            }
            Err(err) => ControlFlow::Error(err),
        };
    }

    match vm.module_scope_value(&name) {
        Some(value) => {
            vm.current_frame_mut().set_reg(inst.dst().0, value);
            ControlFlow::Continue
        }
        None => {
            // Check builtins
            match vm.builtins.get(&name) {
                Some(value) => {
                    vm.current_frame_mut().set_reg(inst.dst().0, value);
                    ControlFlow::Continue
                }
                None => ControlFlow::Error(crate::error::RuntimeError::name_error(name)),
            }
        }
    }
}

/// LoadBuiltin: dst = builtins[names[imm16]]
#[inline(always)]
pub fn load_builtin(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let name = frame.get_name(inst.imm16()).clone();

    match vm.builtins.get(&name) {
        Some(value) => {
            vm.current_frame_mut().set_reg(inst.dst().0, value);
            ControlFlow::Continue
        }
        None => ControlFlow::Error(crate::error::RuntimeError::name_error(name)),
    }
}

/// StoreGlobal: globals[names[imm16]] = dst-register
#[inline(always)]
pub fn store_global(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let name = frame.get_name(inst.imm16()).clone();
    // Store opcodes use DstImm16 encoding where the source register is carried
    // in the dst field and imm16 is the name index.
    let value = frame.get_reg(inst.dst().0);

    vm.set_module_scope_value(name, value);
    ControlFlow::Continue
}

/// DeleteGlobal: del globals[names[imm16]]
#[inline(always)]
pub fn delete_global(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let name = frame.get_name(inst.imm16()).clone();

    match vm.delete_module_scope_value(&name) {
        Some(_) => ControlFlow::Continue,
        None => ControlFlow::Error(crate::error::RuntimeError::name_error(name)),
    }
}

// =============================================================================
// Closures
// =============================================================================

/// LoadClosure: dst = closure[imm16].get()
///
/// Loads the value from a cell in the closure environment.
/// Returns an error if the cell is unbound (UnboundLocalError).
#[inline(always)]
pub fn load_closure(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();

    match &frame.closure {
        Some(env) => {
            let idx = inst.imm16() as usize;
            let cell = env.get_cell(idx);
            match cell.get() {
                Some(value) => {
                    vm.current_frame_mut().set_reg(inst.dst().0, value);
                    ControlFlow::Continue
                }
                None => {
                    // Cell is unbound - variable was deleted or never assigned
                    ControlFlow::Error(crate::error::RuntimeError::unbound_local_cell(idx))
                }
            }
        }
        None => {
            let code = &frame.code;
            ControlFlow::Error(crate::error::RuntimeError::internal(format!(
                "LoadClosure without closure environment in {} (qualname={}, cellvars={}, freevars={})",
                code.name,
                code.qualname,
                code.cellvars.len(),
                code.freevars.len()
            )))
        }
    }
}

/// StoreClosure: closure[imm16].set(dst-register)
///
/// Stores a value into a cell in the closure environment.
/// The cell uses interior mutability (atomic operations), so this works
/// despite the Arc<ClosureEnv> being immutable.
#[inline(always)]
pub fn store_closure(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    // Store opcodes use DstImm16 encoding where the source register is carried
    // in the dst field and imm16 is the closure slot.
    let value = frame.get_reg(inst.dst().0);
    let idx = inst.imm16() as usize;

    match &frame.closure {
        Some(env) => {
            env.set(idx, value);
            ControlFlow::Continue
        }
        None => ControlFlow::Error(crate::error::RuntimeError::internal(
            "StoreClosure without closure environment",
        )),
    }
}

/// DeleteClosure: del closure[imm16]
///
/// Clears a cell in the closure environment, making it unbound.
/// Subsequent reads will raise UnboundLocalError.
#[inline(always)]
pub fn delete_closure(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let idx = inst.imm16() as usize;

    match &frame.closure {
        Some(env) => {
            env.get_cell(idx).clear();
            ControlFlow::Continue
        }
        None => ControlFlow::Error(crate::error::RuntimeError::internal(
            "DeleteClosure without closure environment",
        )),
    }
}

// =============================================================================
// Move
// =============================================================================

/// Move: dst = src1
#[inline(always)]
pub fn move_reg(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let value = frame.get_reg(inst.src1().0);
    frame.set_reg(inst.dst().0, value);
    ControlFlow::Continue
}

#[cfg(test)]
mod tests {
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
}
