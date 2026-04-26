//! Control flow opcode handlers.
//!
//! Handles jumps, returns, and exception handling.

use crate::VirtualMachine;
use crate::builtins::ExceptionValue;
use crate::dispatch::ControlFlow;
use crate::error::{RuntimeError, RuntimeErrorKind};
use crate::ops::calls::invoke_callable_value;
use crate::ops::iteration::{IterStep, ensure_iterator_value, next_step};
use crate::ops::method_dispatch::load_method::{BoundMethodTarget, resolve_special_method};
use crate::stdlib::exceptions::ExceptionTypeId;
use crate::stdlib::generators::GeneratorObject;
use crate::vm::GeneratorResumeOutcome;
use prism_code::Instruction;
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
    let value = vm.current_frame().get_reg(inst.dst().0);

    match crate::truthiness::try_is_truthy(vm, value) {
        Ok(false) => {
            let offset = inst.imm16() as i16;
            ControlFlow::Jump(offset)
        }
        Ok(true) => ControlFlow::Continue,
        Err(err) => ControlFlow::Error(err),
    }
}

/// JumpIfTrue: jump if register is truthy
#[inline(always)]
pub fn jump_if_true(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let value = vm.current_frame().get_reg(inst.dst().0);

    match crate::truthiness::try_is_truthy(vm, value) {
        Ok(true) => {
            let offset = inst.imm16() as i16;
            ControlFlow::Jump(offset)
        }
        Ok(false) => ControlFlow::Continue,
        Err(err) => ControlFlow::Error(err),
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
pub fn pop_except_handler(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    crate::ops::exception::pop_except_handler(vm, inst)
}

/// Raise: raise exception from register
#[inline(always)]
pub fn raise(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    crate::ops::exception::raise(vm, inst)
}

/// Reraise: re-raise current exception
#[inline(always)]
pub fn reraise(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    crate::ops::exception::reraise(vm, inst)
}

/// EndFinally: end finally block
#[inline(always)]
pub fn end_finally(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    crate::ops::exception::end_finally(vm, inst)
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
/// - `dst`: Register that receives the value passed back in via `send()`
/// - `src1`: Register containing the value to yield
///
/// # Returns
///
/// `ControlFlow::Yield` with:
/// - yielded value from `src1`
/// - resume register encoded in `dst`
#[inline(always)]
pub fn yield_value(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let value = frame.get_reg(inst.src1().0);

    // The destination operand encodes which register should receive the
    // next sent value when this generator resumes.
    let resume_point = inst.dst().0 as u32;

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
/// - `dst`: Register that receives sub-generator completion value
/// - `src1`: Register containing sub-generator/iterable
///
/// # Returns
///
/// `ControlFlow::Yield` with the value from the sub-generator.
#[inline(always)]
pub fn yield_from(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst = inst.dst().0;
    let src = inst.src1().0;

    let delegated = vm.current_frame().get_reg(src);
    let iterator = match ensure_iterator_value(vm, delegated) {
        Ok(iterator) => iterator,
        Err(err) => return ControlFlow::Error(err),
    };
    if iterator != delegated {
        vm.current_frame_mut().set_reg(src, iterator);
    }

    let send_value = vm.current_frame().get_reg(dst);
    match drive_yield_from_delegate(vm, iterator, send_value) {
        Ok(YieldFromStep::Yielded(value)) => {
            rewind_current_instruction(vm);
            ControlFlow::Yield {
                value,
                resume_point: dst as u32,
            }
        }
        Ok(YieldFromStep::Returned(value)) => {
            vm.current_frame_mut().set_reg(dst, value);
            ControlFlow::Continue
        }
        Err(err) => ControlFlow::Error(err),
    }
}

enum YieldFromStep {
    Yielded(Value),
    Returned(Value),
}

fn drive_yield_from_delegate(
    vm: &mut VirtualMachine,
    iterator: Value,
    send_value: Value,
) -> Result<YieldFromStep, RuntimeError> {
    if let Some(generator) = GeneratorObject::from_value_mut(iterator) {
        return match vm.resume_generator_for_send(generator, send_value) {
            Ok(GeneratorResumeOutcome::Yielded(value)) => Ok(YieldFromStep::Yielded(value)),
            Ok(GeneratorResumeOutcome::Returned(value)) => Ok(YieldFromStep::Returned(value)),
            Err(err) if is_stop_iteration(&err) => Ok(YieldFromStep::Returned(
                stop_iteration_value(&err).unwrap_or_else(Value::none),
            )),
            Err(err) => Err(err),
        };
    }

    if send_value.is_none() {
        return match next_step(vm, iterator)? {
            IterStep::Yielded(value) => Ok(YieldFromStep::Yielded(value)),
            IterStep::Exhausted => Ok(YieldFromStep::Returned(Value::none())),
        };
    }

    let send_target = resolve_special_method(iterator, "send")?;
    match invoke_bound_method_target(vm, send_target, &[send_value]) {
        Ok(value) => Ok(YieldFromStep::Yielded(value)),
        Err(err) if is_stop_iteration(&err) => Ok(YieldFromStep::Returned(
            stop_iteration_value(&err).unwrap_or_else(Value::none),
        )),
        Err(err) => Err(err),
    }
}

fn invoke_bound_method_target(
    vm: &mut VirtualMachine,
    target: BoundMethodTarget,
    args: &[Value],
) -> Result<Value, RuntimeError> {
    match (target.implicit_self, args.len()) {
        (Some(implicit_self), 0) => invoke_callable_value(vm, target.callable, &[implicit_self]),
        (Some(implicit_self), 1) => {
            invoke_callable_value(vm, target.callable, &[implicit_self, args[0]])
        }
        (Some(implicit_self), _) => {
            let mut call_args = Vec::with_capacity(args.len() + 1);
            call_args.push(implicit_self);
            call_args.extend_from_slice(args);
            invoke_callable_value(vm, target.callable, &call_args)
        }
        (None, _) => invoke_callable_value(vm, target.callable, args),
    }
}

fn is_stop_iteration(err: &RuntimeError) -> bool {
    matches!(err.kind, RuntimeErrorKind::StopIteration)
        || matches!(
            err.kind,
            RuntimeErrorKind::Exception { type_id, .. }
                if type_id == ExceptionTypeId::StopIteration.as_u8() as u16
        )
}

fn stop_iteration_value(err: &RuntimeError) -> Option<Value> {
    let RuntimeErrorKind::Exception { type_id, .. } = err.kind else {
        return None;
    };
    if type_id != ExceptionTypeId::StopIteration.as_u8() as u16 {
        return None;
    }

    let raised = err.raised_value?;
    let exc = unsafe { ExceptionValue::from_value(raised)? };
    exc.args.as_deref().and_then(|args| args.first()).copied()
}

fn rewind_current_instruction(vm: &mut VirtualMachine) {
    let frame = vm.current_frame_mut();
    frame.ip = frame.ip.saturating_sub(1);
}

#[cfg(test)]
mod tests;
