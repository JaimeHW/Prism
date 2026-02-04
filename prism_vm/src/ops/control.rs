//! Control flow opcode handlers.
//!
//! Handles jumps, returns, and exception handling.

use crate::VirtualMachine;
use crate::dispatch::ControlFlow;
use prism_compiler::bytecode::Instruction;
use prism_core::Value;

// =============================================================================
// No-op
// =============================================================================

/// Nop: do nothing
#[inline(always)]
pub fn nop(_vm: &mut VirtualMachine, _inst: Instruction) -> ControlFlow {
    ControlFlow::Continue
}

// =============================================================================
// Returns
// =============================================================================

/// Return: return value from dst register
#[inline(always)]
pub fn return_value(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let value = frame.get_reg(inst.dst().0);
    ControlFlow::Return(value)
}

/// ReturnNone: return None
#[inline(always)]
pub fn return_none(_vm: &mut VirtualMachine, _inst: Instruction) -> ControlFlow {
    ControlFlow::Return(Value::none())
}

// =============================================================================
// Jumps
// =============================================================================

/// Jump: unconditional jump by signed 16-bit offset
#[inline(always)]
pub fn jump(_vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    // imm16 is treated as signed offset
    let offset = inst.imm16() as i16;
    ControlFlow::Jump(offset)
}

/// JumpIfFalse: jump if register is falsy
#[inline(always)]
pub fn jump_if_false(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let value = frame.get_reg(inst.dst().0);

    if !value.is_truthy() {
        let offset = inst.imm16() as i16;
        ControlFlow::Jump(offset)
    } else {
        ControlFlow::Continue
    }
}

/// JumpIfTrue: jump if register is truthy
#[inline(always)]
pub fn jump_if_true(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let value = frame.get_reg(inst.dst().0);

    if value.is_truthy() {
        let offset = inst.imm16() as i16;
        ControlFlow::Jump(offset)
    } else {
        ControlFlow::Continue
    }
}

/// JumpIfNone: jump if register is None
#[inline(always)]
pub fn jump_if_none(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let value = frame.get_reg(inst.dst().0);

    if value.is_none() {
        let offset = inst.imm16() as i16;
        ControlFlow::Jump(offset)
    } else {
        ControlFlow::Continue
    }
}

/// JumpIfNotNone: jump if register is not None
#[inline(always)]
pub fn jump_if_not_none(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let value = frame.get_reg(inst.dst().0);

    if !value.is_none() {
        let offset = inst.imm16() as i16;
        ControlFlow::Jump(offset)
    } else {
        ControlFlow::Continue
    }
}

// =============================================================================
// Exception Handling (Stubs)
// =============================================================================

/// PopExceptHandler: pop exception handler from stack
#[inline(always)]
pub fn pop_except_handler(vm: &mut VirtualMachine, _inst: Instruction) -> ControlFlow {
    // TODO: Implement exception handling
    ControlFlow::Continue
}

/// Raise: raise exception from register
#[inline(always)]
pub fn raise(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let _value = frame.get_reg(inst.dst().0);

    // TODO: Implement exception raising
    ControlFlow::Error(crate::error::RuntimeError::internal(
        "Exception handling not yet implemented",
    ))
}

/// Reraise: re-raise current exception
#[inline(always)]
pub fn reraise(_vm: &mut VirtualMachine, _inst: Instruction) -> ControlFlow {
    // TODO: Implement exception re-raising
    ControlFlow::Error(crate::error::RuntimeError::internal(
        "Exception handling not yet implemented",
    ))
}

/// EndFinally: end finally block
#[inline(always)]
pub fn end_finally(_vm: &mut VirtualMachine, _inst: Instruction) -> ControlFlow {
    // TODO: Implement finally handling
    ControlFlow::Continue
}

// =============================================================================
// Generators (Stubs)
// =============================================================================

/// Yield: yield value from generator
///
/// Suspends the generator and returns the yielded value to the caller.
/// The resume_point is encoded in the instruction's src operand to enable
/// efficient O(1) dispatch on resume via the resume table.
///
/// # Instruction Format
///
/// - dst: Register containing the value to yield
/// - src1: Resume point index (encoded as register number)
/// - src2: Optional result register for sent value on resume
///
/// # Returns
///
/// `ControlFlow::Yield` with the value and resume point.
#[inline(always)]
pub fn yield_value(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let value = frame.get_reg(inst.dst().0);

    // Resume point is encoded in src1 - this is the yield point index
    // that will be used to dispatch back to the correct PC on resume
    let resume_point = inst.src1().0 as u32;

    ControlFlow::Yield {
        value,
        resume_point,
    }
}

/// YieldFrom: yield from sub-generator
///
/// Delegates iteration to a sub-generator or iterable. When the sub-generator
/// yields, the value is passed through to the caller. When it's exhausted,
/// control returns to this generator.
///
/// # Instruction Format
///
/// - dst: Register containing the sub-generator/iterable
/// - src1: Resume point index
///
/// # Returns
///
/// `ControlFlow::Yield` with the value from the sub-generator.
#[inline(always)]
pub fn yield_from(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let sub_gen = frame.get_reg(inst.dst().0);
    let resume_point = inst.src1().0 as u32;

    // For yield from, we need to:
    // 1. Get the next value from the sub-generator
    // 2. If StopIteration, continue execution (extract return value)
    // 3. Otherwise, yield the value to our caller

    // For now, treat the sub_gen value as the immediate yield value
    // Full implementation requires integration with the iterator protocol
    // and proper StopIteration handling
    ControlFlow::Yield {
        value: sub_gen,
        resume_point,
    }
}

#[cfg(test)]
#[allow(unused)]
mod tests {
    use super::*;

    // Control flow tests require full VM setup
}
