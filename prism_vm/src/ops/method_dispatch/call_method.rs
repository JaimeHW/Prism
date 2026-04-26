//! CallMethod opcode: Optimized method invocation.
//!
//! # Encoding
//!
//! `CallMethod dst, method_reg, argc`
//! - `dst`: receives return value
//! - `method_reg`: register containing method (from LoadMethod)
//! - `method_reg+1`: contains self instance or `None` marker
//! - `argc`: number of explicit arguments (after method_reg+1)
//!
//! # Register Layout
//!
//! After LoadMethod, registers are arranged as:
//! ```text
//! [method_reg]:     method/function
//! [method_reg+1]:   self instance or `None` marker
//! [method_reg+2..]: explicit arguments
//! ```
//!
//! # Type-Specialized Dispatch
//!
//! - **FunctionObject**: Push frame, optionally set self as r0
//! - **BuiltinFunction**: Inline call with args
//! - **BoundMethod**: Extract function + instance, recurse
//! - **Closure**: Push frame with captured variables

use crate::VirtualMachine;
use crate::builtins::BuiltinFunctionObject;
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use crate::ops::calls::{call_user_function_from_values, invoke_builtin, invoke_callable_value};
use prism_code::Instruction;
use prism_core::Value;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::descriptor::BoundMethod;
use prism_runtime::object::type_obj::TypeId;
use smallvec::SmallVec;

// =============================================================================
// CallMethod Handler
// =============================================================================

/// CallMethod: Optimized method invocation.
///
/// Avoids BoundMethod allocation by passing self explicitly in registers.
/// Uses TypeId dispatch for fast type-specific call paths.
///
/// # Performance
///
/// - User function: ~15 cycles (frame push + arg copy)
/// - Builtin function: ~10 cycles (inline call)
/// - BoundMethod: ~20 cycles (unpack + recurse)
#[inline(always)]
pub fn call_method(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst = inst.dst().0;
    let method_reg = inst.src1().0;
    let argc = inst.src2().0 as usize;

    let method = vm.current_frame().get_reg(method_reg);
    let self_slot = vm.current_frame().get_reg(method_reg + 1);
    let implicit_self = implicit_self_from_slot(self_slot);

    // Check if method is a callable object
    let Some(ptr) = method.as_object_ptr() else {
        return ControlFlow::Error(RuntimeError::type_error(format!(
            "'{}' object is not callable",
            method.type_name()
        )));
    };

    let type_id = extract_type_id(ptr);

    match type_id {
        TypeId::FUNCTION | TypeId::CLOSURE => {
            call_user_function(vm, ptr, implicit_self, dst, method_reg, argc)
        }
        TypeId::BUILTIN_FUNCTION => {
            call_builtin_function(vm, ptr, implicit_self, dst, method_reg, argc)
        }
        TypeId::METHOD => call_bound_method(vm, ptr, dst, method_reg, argc),
        TypeId::TYPE => call_generic_callable(vm, method, implicit_self, dst, method_reg, argc),
        _ if crate::ops::calls::value_supports_call_protocol(method) => {
            call_generic_callable(vm, method, implicit_self, dst, method_reg, argc)
        }
        _ => ControlFlow::Error(RuntimeError::type_error(format!(
            "'{}' object is not callable",
            type_id.name()
        ))),
    }
}

/// Convert LoadMethod's self slot into an optional implicit self argument.
#[inline(always)]
fn implicit_self_from_slot(self_slot: Value) -> Option<Value> {
    if self_slot.is_none() {
        None
    } else {
        Some(self_slot)
    }
}

// =============================================================================
// Type-Specific Call Paths
// =============================================================================

/// Call a user-defined function with optional implicit self.
///
/// # Register Layout After Push
///
/// New frame:
/// - With implicit self: r0=self, r1..rN=explicit args
/// - Without implicit self: r0..rN=explicit args
#[inline]
fn call_user_function(
    vm: &mut VirtualMachine,
    func_ptr: *const (),
    implicit_self: Option<Value>,
    dst: u8,
    method_reg: u8,
    argc: usize,
) -> ControlFlow {
    let caller_frame = &vm.frames[vm.call_depth() - 1];
    let mut args: SmallVec<[Value; 8]> =
        SmallVec::with_capacity(argc + implicit_self.is_some() as usize);

    if let Some(self_val) = implicit_self {
        args.push(self_val);
    }

    for i in 0..argc {
        args.push(caller_frame.get_reg(method_reg + 2 + i as u8));
    }

    call_user_function_from_values(vm, func_ptr, dst, &args, &[])
}

/// Call a builtin function with optional implicit self + args.
///
/// Collects all arguments into a SmallVec and invokes the builtin directly.
/// No frame push required - builtins execute synchronously.
#[inline]
fn call_builtin_function(
    vm: &mut VirtualMachine,
    func_ptr: *const (),
    implicit_self: Option<Value>,
    dst: u8,
    method_reg: u8,
    argc: usize,
) -> ControlFlow {
    let builtin = unsafe { &*(func_ptr as *const BuiltinFunctionObject) };

    // Collect arguments: implicit self (if present) + explicit args.
    // Use SmallVec to avoid heap allocation for typical calls (≤8 args)
    let caller_frame = &vm.frames[vm.call_depth() - 1];
    let mut args: SmallVec<[Value; 8]> =
        SmallVec::with_capacity(argc + implicit_self.is_some() as usize);
    if let Some(self_val) = implicit_self {
        args.push(self_val);
    }

    // Collect explicit arguments
    for i in 0..argc {
        args.push(caller_frame.get_reg(method_reg + 2 + i as u8));
    }

    // Call the builtin function
    match invoke_builtin(vm, builtin, &args) {
        Ok(result) => {
            vm.current_frame_mut().set_reg(dst, result);
            ControlFlow::Continue
        }
        Err(e) => ControlFlow::Error(e),
    }
}

/// Call a pre-bound method (BoundMethod object).
///
/// Extracts the underlying function and instance, then dispatches
/// to the appropriate call path.
#[inline]
fn call_bound_method(
    vm: &mut VirtualMachine,
    bound_ptr: *const (),
    dst: u8,
    method_reg: u8,
    argc: usize,
) -> ControlFlow {
    let bound = unsafe { &*(bound_ptr as *const BoundMethod) };

    // Extract function and instance from BoundMethod
    let func = bound.function();
    let instance = bound.instance();

    // Get the function pointer
    let Some(func_ptr) = func.as_object_ptr() else {
        return ControlFlow::Error(RuntimeError::type_error(
            "bound method has invalid function",
        ));
    };

    let type_id = extract_type_id(func_ptr);

    // Recurse with extracted function and instance
    match type_id {
        TypeId::FUNCTION | TypeId::CLOSURE => {
            call_user_function(vm, func_ptr, Some(instance), dst, method_reg, argc)
        }
        TypeId::BUILTIN_FUNCTION => {
            call_builtin_function(vm, func_ptr, Some(instance), dst, method_reg, argc)
        }
        _ if crate::ops::calls::value_supports_call_protocol(func) => {
            call_generic_callable(vm, func, Some(instance), dst, method_reg, argc)
        }
        _ => ControlFlow::Error(RuntimeError::type_error(format!(
            "bound method wraps non-callable '{}' object",
            type_id.name()
        ))),
    }
}

#[inline]
fn call_generic_callable(
    vm: &mut VirtualMachine,
    callable: Value,
    implicit_self: Option<Value>,
    dst: u8,
    method_reg: u8,
    argc: usize,
) -> ControlFlow {
    let caller_frame = &vm.frames[vm.call_depth() - 1];
    let mut args: SmallVec<[Value; 8]> =
        SmallVec::with_capacity(argc + implicit_self.is_some() as usize);
    if let Some(self_val) = implicit_self {
        args.push(self_val);
    }

    for i in 0..argc {
        args.push(caller_frame.get_reg(method_reg + 2 + i as u8));
    }

    match invoke_callable_value(vm, callable, &args) {
        Ok(result) => {
            vm.current_frame_mut().set_reg(dst, result);
            ControlFlow::Continue
        }
        Err(err) => ControlFlow::Error(err),
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Extract TypeId from an object pointer.
#[inline(always)]
fn extract_type_id(ptr: *const ()) -> TypeId {
    let header = ptr as *const ObjectHeader;
    unsafe { (*header).type_id }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
