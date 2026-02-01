//! Function call opcode handlers.
//!
//! Handles function calls, closures, and tail calls.
//!
//! # Performance Notes
//!
//! - Function objects are heap-allocated with Box::into_raw for stable pointers
//! - Call dispatch uses O(1) type discrimination via ObjectHeader
//! - Arguments are passed via register file, avoiding heap allocation

use crate::VirtualMachine;
use crate::builtins::BuiltinFunctionObject;
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use prism_compiler::bytecode::{CodeObject, Instruction};
use prism_core::Value;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::function::FunctionObject;
use std::sync::Arc;

// =============================================================================
// Type ID Extraction Helper
// =============================================================================

/// Extract TypeId from an object pointer.
///
/// SAFETY: Relies on ObjectHeader being at offset 0 of all PyObject types.
#[inline(always)]
fn extract_type_id(ptr: *const ()) -> TypeId {
    let header_ptr = ptr as *const ObjectHeader;
    unsafe { (*header_ptr).type_id }
}

// =============================================================================
// Function Calls
// =============================================================================

/// Call: dst = func(args...)
/// src1 = function, src2 = argc, args in r(dst+1)..r(dst+argc)
///
/// Dispatches to the appropriate call handler based on the function type.
/// Uses O(1) type discrimination via ObjectHeader for fast dispatch.
#[inline(always)]
pub fn call(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let func_val = vm.current_frame().get_reg(inst.src1().0);
    let argc = inst.src2().0 as usize;
    let dst_reg = inst.dst().0;

    // Check if this is a callable object
    if let Some(ptr) = func_val.as_object_ptr() {
        let type_id = extract_type_id(ptr);

        match type_id {
            TypeId::BUILTIN_FUNCTION => {
                // Fast path: builtin function - call directly without frame push
                let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };

                // Collect arguments from registers
                let frame = vm.current_frame();
                let args: Vec<Value> = (0..argc)
                    .map(|i| frame.get_reg(dst_reg + 1 + i as u8))
                    .collect();

                // Call the builtin function
                match builtin.call(&args) {
                    Ok(result) => {
                        vm.current_frame_mut().set_reg(dst_reg, result);
                        ControlFlow::Continue
                    }
                    Err(e) => ControlFlow::Error(RuntimeError::type_error(e.to_string())),
                }
            }
            TypeId::FUNCTION | TypeId::CLOSURE => {
                // User-defined function - push frame
                let func = unsafe { &*(ptr as *const FunctionObject) };
                let code = Arc::clone(&func.code);

                // Push new frame for function execution
                if let Err(e) = vm.push_frame(Arc::clone(&code), dst_reg) {
                    return ControlFlow::Error(e);
                }

                // Copy arguments to new frame's registers
                let caller_frame_idx = vm.call_depth() - 1;

                // Collect args from caller and set in new frame
                let args: Vec<Value> = (0..argc)
                    .map(|i| vm.frames[caller_frame_idx - 1].get_reg(dst_reg + 1 + i as u8))
                    .collect();

                for (i, arg) in args.into_iter().enumerate() {
                    vm.current_frame_mut().set_reg(i as u8, arg);
                }

                ControlFlow::Continue
            }
            _ => ControlFlow::Error(RuntimeError::type_error(format!(
                "'{}' object is not callable",
                type_id.name()
            ))),
        }
    } else {
        ControlFlow::Error(RuntimeError::type_error("object is not callable"))
    }
}

/// CallKw: call with keyword arguments
#[inline(always)]
pub fn call_kw(_vm: &mut VirtualMachine, _inst: Instruction) -> ControlFlow {
    // TODO: Implement keyword argument calls
    // This requires parsing the kwargs dict and matching to parameter names
    ControlFlow::Error(RuntimeError::internal("CallKw not yet implemented"))
}

/// CallMethod: dst = obj.method(args...)
#[inline(always)]
pub fn call_method(_vm: &mut VirtualMachine, _inst: Instruction) -> ControlFlow {
    // TODO: Implement method calls
    // Similar to Call but with implicit self parameter
    ControlFlow::Error(RuntimeError::internal("CallMethod not yet implemented"))
}

/// TailCall: call reusing current frame
///
/// Optimizes tail-recursive calls by reusing the current frame.
#[inline(always)]
pub fn tail_call(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    // For now, fall back to regular call
    // TODO: Implement true tail call optimization
    call(vm, inst)
}

// =============================================================================
// Function Creation
// =============================================================================

/// MakeFunction: create function from code object
/// dst = function, imm16 = code constant index
///
/// Creates a FunctionObject from a code constant and stores it in dst.
#[inline(always)]
pub fn make_function(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let code_idx = inst.imm16();
    let dst = inst.dst().0;

    // Get code object from constant pool
    // Constants can contain Arc<CodeObject> for nested function definitions
    let code_val = frame.get_const(code_idx);

    // For now, we need to extract the CodeObject from the constant
    // In a full implementation, the constant pool would store CodeObject directly
    // TODO: Properly handle code object constants
    if let Some(_code_ptr) = code_val.as_object_ptr() {
        // Assume this points to a CodeObject wrapper
        // For now, create a placeholder function
        let func = Box::new(FunctionObject::new(
            Arc::new(CodeObject::new("anonymous", "<module>")),
            Arc::from("anonymous"),
            None,
            None,
        ));
        let func_ptr = Box::into_raw(func) as *const ();
        frame.set_reg(dst, Value::object_ptr(func_ptr));
        ControlFlow::Continue
    } else {
        // Handle case where code is stored differently
        ControlFlow::Error(RuntimeError::internal(
            "Invalid code object in constant pool",
        ))
    }
}

/// MakeClosure: create closure with captured variables
///
/// Creates a FunctionObject with a captured closure environment.
#[inline(always)]
pub fn make_closure(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let code_idx = inst.imm16();
    let dst = inst.dst().0;

    // Get code object from constant pool
    let _code_val = frame.get_const(code_idx);

    // TODO: Properly implement closure creation
    // 1. Get the list of free variables from the code object
    // 2. Capture values from the current frame's registers or closure env
    // 3. Create ClosureEnv and FunctionObject

    // Placeholder: create function without closure
    let func = Box::new(FunctionObject::new(
        Arc::new(CodeObject::new("closure", "<module>")),
        Arc::from("closure"),
        None,
        None, // Should be Some(ClosureEnv) with captured values
    ));
    let func_ptr = Box::into_raw(func) as *const ();
    frame.set_reg(dst, Value::object_ptr(func_ptr));
    ControlFlow::Continue
}

#[cfg(test)]
mod tests {
    // Call tests require full VM setup with function objects
}
