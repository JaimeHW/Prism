//! Send opcode handler.
//!
//! Implements the coroutine/generator send protocol for sending values
//! into suspended coroutines and generators.
//!
//! # Python Semantics
//!
//! The `send()` method is used to:
//! 1. Resume a suspended coroutine/generator
//! 2. Pass a value to the `yield` expression that suspended it
//! 3. Return the next yielded value (or raise StopIteration)
//!
//! # Protocol Rules
//!
//! - First call must use `send(None)` (or `__next__()`)
//! - Sending non-None to a just-started generator raises TypeError
//! - Sending to an exhausted generator raises StopIteration
//!
//! # Performance
//!
//! - Direct send: ~5 cycles (generator state check + resume)
//! - Protocol validation: ~2 extra cycles for state checks

use crate::VirtualMachine;
use crate::builtins::{create_exception, create_exception_with_args};
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use crate::stdlib::exceptions::ExceptionTypeId;
use crate::stdlib::generators::{GeneratorObject, GeneratorState as RuntimeGeneratorState};
use crate::vm::GeneratorResumeOutcome;
use prism_code::Instruction;
use prism_core::Value;

use super::protocol::type_name;

/// Send: Send value to coroutine/generator.
///
/// Instruction format: `Send dst, gen, value`
/// - `dst`: Destination register for the result (yielded value or return)
/// - `gen`: Register containing the generator/coroutine (src1)
/// - `value`: Register containing the value to send (src2)
#[inline(always)]
pub fn send(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst = inst.dst().0;
    let gen_reg = inst.src1().0;
    let value_reg = inst.src2().0;

    let generator = vm.current_frame().get_reg(gen_reg);
    let value = vm.current_frame().get_reg(value_reg);

    // =========================================================================
    // Validate Generator/Coroutine
    // =========================================================================

    // Check if the object is a generator or coroutine
    match get_generator_state(&generator) {
        GeneratorState::NotAGenerator => {
            return ControlFlow::Error(RuntimeError::type_error(format!(
                "cannot send to non-generator object of type {}",
                type_name(&generator)
            )));
        }
        GeneratorState::Created => {
            // Just-started generator: can only receive None
            if !value.is_none() {
                return ControlFlow::Error(RuntimeError::type_error(
                    "can't send non-None value to a just-started generator",
                ));
            }
        }
        GeneratorState::Suspended => {
            // Normal case: generator is suspended and ready to receive
        }
        GeneratorState::Running => {
            return ControlFlow::Error(RuntimeError::value_error("generator already executing"));
        }
        GeneratorState::Closed => {
            // Exhausted generator
            return ControlFlow::Error(RuntimeError::stop_iteration());
        }
    }

    // =========================================================================
    // Perform Send
    // =========================================================================

    // Resume the generator with the sent value
    // The result will be the next yielded value or StopIteration on completion
    match resume_generator(vm, generator, value) {
        ResumeResult::Yielded(yielded_value) => {
            vm.current_frame_mut().set_reg(dst, yielded_value);
            ControlFlow::Continue
        }
        ResumeResult::Returned(return_value) => {
            // Generator completed: raise StopIteration with the return value payload.
            let stop_iteration_value = if return_value.is_none() {
                create_exception(ExceptionTypeId::StopIteration, None)
            } else {
                create_exception_with_args(
                    ExceptionTypeId::StopIteration,
                    None,
                    vec![return_value].into_boxed_slice(),
                )
            };
            vm.set_active_exception_with_type(
                stop_iteration_value,
                ExceptionTypeId::StopIteration as u16,
            );
            ControlFlow::Exception {
                type_id: ExceptionTypeId::StopIteration as u16,
                handler_pc: 0,
            }
        }
        ResumeResult::Error(e) => ControlFlow::Error(e),
    }
}

// =============================================================================
// Generator State
// =============================================================================

/// State of a generator/coroutine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GeneratorState {
    /// Object is not a generator or coroutine.
    NotAGenerator,
    /// Generator was just created (not yet started).
    Created,
    /// Generator is suspended at a yield point.
    Suspended,
    /// Generator is currently executing.
    Running,
    /// Generator has completed (returned or raised).
    Closed,
}

/// Get the current state of a generator/coroutine.
#[inline]
fn get_generator_state(value: &Value) -> GeneratorState {
    let Some(generator) = GeneratorObject::from_value(*value) else {
        return GeneratorState::NotAGenerator;
    };

    match generator.state() {
        RuntimeGeneratorState::Created => GeneratorState::Created,
        RuntimeGeneratorState::Suspended => GeneratorState::Suspended,
        RuntimeGeneratorState::Running => GeneratorState::Running,
        RuntimeGeneratorState::Exhausted => GeneratorState::Closed,
    }
}

/// Get the type name of a value for error messages.
/// Result of resuming a generator.
enum ResumeResult {
    /// Generator yielded a value.
    Yielded(Value),
    /// Generator returned (completed).
    Returned(Value),
    /// Error occurred during execution.
    Error(RuntimeError),
}

/// Resume a generator with a sent value.
#[inline]
fn resume_generator(vm: &mut VirtualMachine, gen_value: Value, send_value: Value) -> ResumeResult {
    let Some(generator) = GeneratorObject::from_value_mut(gen_value) else {
        return ResumeResult::Error(RuntimeError::type_error(
            "send target is not a generator object",
        ));
    };

    match vm.resume_generator_for_send(generator, send_value) {
        Ok(GeneratorResumeOutcome::Yielded(value)) => ResumeResult::Yielded(value),
        Ok(GeneratorResumeOutcome::Returned(value)) => ResumeResult::Returned(value),
        Err(e) => ResumeResult::Error(e),
    }
}
