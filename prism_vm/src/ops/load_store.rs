//! Load and store opcode handlers.
//!
//! Handles loading constants, locals, globals, closures, and register moves.

use crate::VirtualMachine;
use crate::dispatch::ControlFlow;
use prism_compiler::bytecode::Instruction;
use prism_core::Value;

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
    let frame = vm.current_frame_mut();
    let slot = inst.imm16() as u8;
    let value = frame.get_reg(slot);
    frame.set_reg(inst.dst().0, value);
    ControlFlow::Continue
}

/// StoreLocal: frame.registers[imm16] = src1
#[inline(always)]
pub fn store_local(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let value = frame.get_reg(inst.src1().0);
    let slot = inst.imm16() as u8;
    frame.set_reg(slot, value);
    ControlFlow::Continue
}

/// DeleteLocal: frame.registers[imm16] = undefined
#[inline(always)]
pub fn delete_local(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let slot = inst.imm16() as u8;
    frame.set_reg(slot, Value::none()); // Mark as unbound
    ControlFlow::Continue
}

// =============================================================================
// Globals
// =============================================================================

/// LoadGlobal: dst = globals[names[imm16]]
#[inline(always)]
pub fn load_global(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let name = frame.get_name(inst.imm16()).clone();

    match vm.globals.get_arc(&name) {
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

/// StoreGlobal: globals[names[imm16]] = src1
#[inline(always)]
pub fn store_global(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let name = frame.get_name(inst.imm16()).clone();
    let value = frame.get_reg(inst.src1().0);

    vm.globals.set(name, value);
    ControlFlow::Continue
}

/// DeleteGlobal: del globals[names[imm16]]
#[inline(always)]
pub fn delete_global(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let name = frame.get_name(inst.imm16()).clone();

    match vm.globals.delete(&name) {
        Some(_) => ControlFlow::Continue,
        None => ControlFlow::Error(crate::error::RuntimeError::name_error(name)),
    }
}

// =============================================================================
// Closures
// =============================================================================

/// LoadClosure: dst = closure[imm16]
#[inline(always)]
pub fn load_closure(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();

    match &frame.closure {
        Some(env) => {
            let value = env.get(inst.imm16() as usize);
            vm.current_frame_mut().set_reg(inst.dst().0, value);
            ControlFlow::Continue
        }
        None => ControlFlow::Error(crate::error::RuntimeError::internal(
            "LoadClosure without closure environment",
        )),
    }
}

/// StoreClosure: closure[imm16] = src1
#[inline(always)]
pub fn store_closure(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let value = frame.get_reg(inst.src1().0);
    let idx = inst.imm16() as usize;

    // Need mutable access to closure
    // This is tricky because Arc is immutable - in practice,
    // closures use Cell or RefCell for interior mutability
    // For now, we'll use a simplified model
    match &frame.closure {
        Some(_env) => {
            // TODO: Implement mutable closure cells
            // For now, closures are read-only
            ControlFlow::Error(crate::error::RuntimeError::internal(
                "Mutable closure cells not yet implemented",
            ))
        }
        None => ControlFlow::Error(crate::error::RuntimeError::internal(
            "StoreClosure without closure environment",
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
    use prism_compiler::bytecode::{Opcode, Register};

    // Note: Full tests require VirtualMachine setup which is tested in integration tests
}
