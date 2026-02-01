//! Function call opcode handlers.
//!
//! Handles function calls, closures, and tail calls.

use crate::VirtualMachine;
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use prism_compiler::bytecode::Instruction;
use prism_core::Value;

// =============================================================================
// Function Calls
// =============================================================================

/// Call: dst = func(args...)
/// src1 = function, src2 = argc, args in r(dst+1)..r(dst+argc)
#[inline(always)]
pub fn call(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let func = frame.get_reg(inst.src1().0);
    let argc = inst.src2().0;
    let dst_reg = inst.dst().0;

    // Collect arguments from registers dst+1 to dst+argc
    let mut args = Vec::with_capacity(argc as usize);
    for i in 0..argc {
        args.push(frame.get_reg(dst_reg + 1 + i));
    }

    // Check if function is callable and dispatch
    // TODO: Implement proper function calling with frame push
    ControlFlow::Error(RuntimeError::internal("Call not yet implemented"))
}

/// CallKw: call with keyword arguments
#[inline(always)]
pub fn call_kw(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    // TODO: Implement keyword argument calls
    ControlFlow::Error(RuntimeError::internal("CallKw not yet implemented"))
}

/// CallMethod: dst = obj.method(args...)
#[inline(always)]
pub fn call_method(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    // TODO: Implement method calls
    ControlFlow::Error(RuntimeError::internal("CallMethod not yet implemented"))
}

/// TailCall: call reusing current frame
#[inline(always)]
pub fn tail_call(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    // TODO: Implement tail call optimization
    // For now, fall back to regular call
    call(vm, inst)
}

// =============================================================================
// Function Creation
// =============================================================================

/// MakeFunction: create function from code object
/// dst = function, imm16 = code constant index
#[inline(always)]
pub fn make_function(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let code_idx = inst.imm16();
    let _code = frame.get_const(code_idx);

    // TODO: Create function object from code constant
    ControlFlow::Error(RuntimeError::internal("MakeFunction not yet implemented"))
}

/// MakeClosure: create closure with captured variables
#[inline(always)]
pub fn make_closure(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let code_idx = inst.imm16();
    let _code = frame.get_const(code_idx);

    // TODO: Create closure with captured environment
    ControlFlow::Error(RuntimeError::internal("MakeClosure not yet implemented"))
}

#[cfg(test)]
mod tests {
    // Call tests require full VM setup
}
