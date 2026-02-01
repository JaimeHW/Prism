//! Object operation handlers.
//!
//! Handles attribute access, item access, and iteration with inline caching.

use crate::VirtualMachine;
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use prism_compiler::bytecode::Instruction;
use prism_core::Value;

// =============================================================================
// Attribute Access (with Inline Caching)
// =============================================================================

/// GetAttr: dst = src.attr[name_idx]
#[inline(always)]
pub fn get_attr(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let obj = frame.get_reg(inst.src1().0);
    let name_idx = inst.src2().0 as u16;
    let name = frame.get_name(name_idx).clone();

    // TODO: Implement proper object attribute access
    // For now, return error
    ControlFlow::Error(RuntimeError::attribute_error("object", name))
}

/// SetAttr: src1.attr[name_idx] = src2
#[inline(always)]
pub fn set_attr(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let _obj = frame.get_reg(inst.dst().0);
    let _value = frame.get_reg(inst.src2().0);
    let name_idx = inst.src1().0 as u16;
    let _name = frame.get_name(name_idx).clone();

    // TODO: Implement proper object attribute setting
    ControlFlow::Error(RuntimeError::internal("SetAttr not yet implemented"))
}

/// DelAttr: del src.attr[name_idx]
#[inline(always)]
pub fn del_attr(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let _obj = frame.get_reg(inst.src1().0);
    let name_idx = inst.src2().0 as u16;
    let _name = frame.get_name(name_idx).clone();

    // TODO: Implement proper object attribute deletion
    ControlFlow::Error(RuntimeError::internal("DelAttr not yet implemented"))
}

// =============================================================================
// Item Access
// =============================================================================

/// GetItem: dst = src1[src2]
#[inline(always)]
pub fn get_item(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let _container = frame.get_reg(inst.src1().0);
    let _key = frame.get_reg(inst.src2().0);

    // TODO: Implement container item access
    ControlFlow::Error(RuntimeError::internal("GetItem not yet implemented"))
}

/// SetItem: src1[dst] = src2 (dst is key register)
#[inline(always)]
pub fn set_item(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let _container = frame.get_reg(inst.src1().0);
    let _key = frame.get_reg(inst.dst().0);
    let _value = frame.get_reg(inst.src2().0);

    // TODO: Implement container item setting
    ControlFlow::Error(RuntimeError::internal("SetItem not yet implemented"))
}

/// DelItem: del src1[src2]
#[inline(always)]
pub fn del_item(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let _container = frame.get_reg(inst.src1().0);
    let _key = frame.get_reg(inst.src2().0);

    // TODO: Implement container item deletion
    ControlFlow::Error(RuntimeError::internal("DelItem not yet implemented"))
}

// =============================================================================
// Iteration
// =============================================================================

/// GetIter: dst = iter(src)
#[inline(always)]
pub fn get_iter(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let _obj = frame.get_reg(inst.src1().0);

    // TODO: Implement iterator creation
    ControlFlow::Error(RuntimeError::internal("GetIter not yet implemented"))
}

/// ForIter: dst = next(src), jump if StopIteration
#[inline(always)]
pub fn for_iter(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let _iterator = frame.get_reg(inst.dst().0);
    let _offset = inst.imm16() as i16;

    // TODO: Implement iterator next
    ControlFlow::Error(RuntimeError::internal("ForIter not yet implemented"))
}

// =============================================================================
// Utilities
// =============================================================================

/// Len: dst = len(src)
#[inline(always)]
pub fn len(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let _obj = frame.get_reg(inst.src1().0);

    // TODO: Implement len() for containers
    ControlFlow::Error(RuntimeError::internal("Len not yet implemented"))
}

/// IsCallable: dst = callable(src)
#[inline(always)]
pub fn is_callable(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let obj = frame.get_reg(inst.src1().0);

    // Check if value is a function object
    // For now, only function pointers are callable
    let is_callable = obj.is_object(); // Simplified check

    frame.set_reg(inst.dst().0, Value::bool(is_callable));
    ControlFlow::Continue
}

#[cfg(test)]
mod tests {
    // Object tests require full VM setup with object system
}
