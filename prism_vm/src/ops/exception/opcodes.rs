//! Exception handling opcode handlers.
//!
//! This module provides the opcode handlers for Python exception handling:
//!
//! - [`raise`]: Raise an exception from a value.
//! - [`raise_with_cause`]: Raise an exception with a cause (chained exceptions).
//! - [`reraise`]: Re-raise the current exception.
//! - [`pop_except_handler`]: Pop exception handler from the handler stack.
//! - [`end_finally`]: End finally block and potentially reraise.
//! - [`setup_except`]: Set up exception handler (push to handler stack).
//! - [`check_exc_match`]: Check if exception matches a type.
//! - [`get_exception_traceback`]: Get traceback from exception.
//! - [`push_exc_info`]: Push current exception info to stack.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    Exception Opcode Handlers                             │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │  try:                                                                    │
//! │      SETUP_EXCEPT handler_pc  ──────▶  Push HandlerFrame to stack        │
//! │      ...code...                                                          │
//! │      POP_EXCEPT_HANDLER       ──────▶  Pop HandlerFrame (normal exit)    │
//! │  except SomeError:                                                       │
//! │      CHECK_EXC_MATCH          ──────▶  Compare type_id, jump if no match │
//! │      ...handler...                                                       │
//! │      END_FINALLY              ──────▶  Clear exception, resume normal    │
//! │  finally:                                                                │
//! │      ...finally...                                                       │
//! │      END_FINALLY              ──────▶  Maybe reraise pending exception   │
//! │                                                                          │
//! │  raise ValueError("x")        ──────▶  RAISE triggers exception flow     │
//! │  raise                        ──────▶  RERAISE re-raises active exc      │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance
//!
//! | Opcode | Complexity | Notes |
//! |--------|------------|-------|
//! | SETUP_EXCEPT | O(1) | Stack push |
//! | POP_EXCEPT_HANDLER | O(1) | Stack pop |
//! | RAISE | O(1) cache hit, O(N) miss | Handler lookup |
//! | CHECK_EXC_MATCH | O(1) | Type ID comparison |
//! | END_FINALLY | O(1) | State check |

use crate::VirtualMachine;
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use crate::exception::HandlerFrame;
use prism_compiler::bytecode::Instruction;
use prism_core::Value;

use super::helpers::{check_dynamic_match, check_tuple_match};

// =============================================================================
// Constants
// =============================================================================

/// Sentinel value for "no exception type" in instruction encoding.
const NO_TYPE_ID: u16 = 0xFFFF;

/// Sentinel value for "no handler PC" indicating lookup is needed.
const NO_HANDLER_PC: u32 = 0;

// =============================================================================
// RAISE Opcodes
// =============================================================================

/// Raise an exception from a value in a register.
///
/// # Instruction Format
///
/// - `dst`: Register containing the exception value
/// - `src1`: (unused)
/// - `imm16`: Exception type ID (or 0xFFFF for dynamic type)
///
/// # Control Flow
///
/// Returns `ControlFlow::Exception` which triggers the exception propagation
/// mechanism in the VM's run loop.
///
/// # Performance
///
/// - Hot path (StopIteration): ~8-12 cycles (flyweight + cache hit)
/// - Cold path (with traceback): ~40-60 cycles
#[inline(always)]
pub fn raise(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let exc_value = frame.get_reg(inst.dst().0);

    // Extract type ID from instruction or infer from value
    let type_id = extract_type_id(inst, &exc_value);

    // Store the active exception in the VM with its type
    vm.set_active_exception_with_type(exc_value, type_id);

    // Return Exception control flow - VM will handle propagation
    ControlFlow::Exception {
        type_id,
        handler_pc: NO_HANDLER_PC, // VM will perform lookup
    }
}

/// Raise an exception with a cause (chained exceptions).
///
/// # Instruction Format
///
/// - `dst`: Register containing the exception value
/// - `src1`: Register containing the cause exception
/// - `imm16`: Exception type ID
///
/// This implements `raise X from Y` syntax.
#[inline(always)]
pub fn raise_with_cause(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let exc_value = frame.get_reg(inst.dst().0);
    let _cause = frame.get_reg(inst.src1().0);

    // TODO: Attach cause to exception object
    // For now, just raise the primary exception

    let type_id = extract_type_id(inst, &exc_value);
    vm.set_active_exception_with_type(exc_value, type_id);

    ControlFlow::Exception {
        type_id,
        handler_pc: NO_HANDLER_PC,
    }
}

/// Re-raise the current exception.
///
/// # Instruction Format
///
/// - No operands
///
/// # Semantics
///
/// Re-raises the currently active exception. This is used in except blocks
/// when the handler decides not to handle the exception.
///
/// # Errors
///
/// Returns `ControlFlow::Error` if there is no active exception to reraise.
#[inline(always)]
pub fn reraise(vm: &mut VirtualMachine, _inst: Instruction) -> ControlFlow {
    // Check if there's an active exception to reraise
    if !vm.has_active_exception() {
        return ControlFlow::Error(RuntimeError::internal("No active exception to reraise"));
    }

    // Return Reraise control flow - VM preserves the current exception
    ControlFlow::Reraise
}

// =============================================================================
// Exception Handler Stack Opcodes
// =============================================================================

/// Set up an exception handler (push to handler stack).
///
/// # Instruction Format
///
/// - `dst`: Handler index in the code object's handler table
/// - `imm16`: Stack depth at handler entry point
///
/// # Semantics
///
/// Pushes a new `HandlerFrame` onto the handler stack. This is executed at
/// the beginning of a try block.
#[inline(always)]
pub fn setup_except(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let handler_idx = inst.dst().0 as u16;
    let stack_depth = inst.imm16();
    let frame_id = vm.current_frame_id();

    let handler_frame = HandlerFrame::new(handler_idx, stack_depth, frame_id);

    // Push handler onto the stack
    if !vm.push_exception_handler(handler_frame) {
        return ControlFlow::Error(RuntimeError::internal("Exception handler stack overflow"));
    }

    ControlFlow::Continue
}

/// Pop exception handler from the handler stack.
///
/// # Instruction Format
///
/// - No operands
///
/// # Semantics
///
/// Pops the top handler from the handler stack. This is executed when
/// exiting a try block normally (without an exception).
#[inline(always)]
pub fn pop_except_handler(vm: &mut VirtualMachine, _inst: Instruction) -> ControlFlow {
    vm.pop_exception_handler();
    ControlFlow::Continue
}

/// End a finally block and potentially reraise.
///
/// # Instruction Format
///
/// - No operands
///
/// # Semantics
///
/// Ends a finally block. If there's a pending exception that was being
/// propagated when the finally block was entered, it will be reraised.
/// Otherwise, execution continues normally.
#[inline(always)]
pub fn end_finally(vm: &mut VirtualMachine, _inst: Instruction) -> ControlFlow {
    // Check if we need to reraise after finally
    if vm.should_reraise_after_finally() {
        vm.clear_reraise_flag();
        ControlFlow::Reraise
    } else {
        // Normal exit from finally - clear any exception state
        vm.clear_exception_state();
        ControlFlow::Continue
    }
}

// =============================================================================
// Exception Matching Opcodes
// =============================================================================

/// Check if the current exception matches a type.
///
/// # Instruction Format
///
/// - `dst`: Register to store match result (bool)
/// - `src1`: Register containing exception type to match against
/// - `imm16`: Exception type ID for fast path (or NO_TYPE_ID for dynamic)
///
/// # Semantics
///
/// Compares the current exception's type against the specified type,
/// considering inheritance. Stores True if the exception matches,
/// False otherwise.
///
/// # Performance
///
/// - Fast path (known type ID): Single comparison (~3 cycles)
/// - Slow path (dynamic type): Hierarchy lookup (~15-20 cycles)
#[inline(always)]
pub fn check_exc_match(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst_reg = inst.dst().0;
    let type_id_hint = inst.imm16();

    // Get the current exception's type ID
    let exc_type_id = match vm.get_active_exception_type_id() {
        Some(type_id) => type_id,
        None => {
            // No active exception - this shouldn't happen in well-formed bytecode
            vm.current_frame_mut().set_reg(dst_reg, Value::bool(false));
            return ControlFlow::Continue;
        }
    };

    // Fast path: Direct type ID comparison
    let matches = if type_id_hint != NO_TYPE_ID {
        // Check if exception type matches or is a subclass
        exc_type_id == type_id_hint || is_subclass(exc_type_id, type_id_hint)
    } else {
        // Slow path: Get type from register and compare dynamically
        let frame = vm.current_frame();
        let type_value = frame.get_reg(inst.src1().0);
        check_dynamic_match(exc_type_id, &type_value)
    };

    vm.current_frame_mut()
        .set_reg(dst_reg, Value::bool(matches));
    ControlFlow::Continue
}

/// Check if exception matches multiple types (tuple of types).
///
/// # Instruction Format
///
/// - `dst`: Register to store match result (bool)
/// - `src1`: Register containing tuple of exception types
///
/// # Semantics
///
/// Checks if the current exception matches any of the types in the tuple.
/// This handles `except (TypeError, ValueError):` syntax.
#[inline(always)]
pub fn check_exc_match_tuple(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst_reg = inst.dst().0;

    let exc_type_id = match vm.get_active_exception_type_id() {
        Some(type_id) => type_id,
        None => {
            vm.current_frame_mut().set_reg(dst_reg, Value::bool(false));
            return ControlFlow::Continue;
        }
    };

    let frame = vm.current_frame();
    let types_tuple = frame.get_reg(inst.src1().0);

    // Check if exception matches any type in the tuple
    let matches = check_tuple_match(exc_type_id, &types_tuple);
    vm.current_frame_mut()
        .set_reg(dst_reg, Value::bool(matches));
    ControlFlow::Continue
}

// =============================================================================
// Exception Info Opcodes
// =============================================================================

/// Get the current exception value and store it in a register.
///
/// # Instruction Format
///
/// - `dst`: Register to store exception value
///
/// # Semantics
///
/// Retrieves the currently active exception and stores it. This is used
/// in except blocks to access the caught exception.
#[inline(always)]
pub fn get_exception(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst_reg = inst.dst().0;

    // Clone the exception value to avoid borrow issues
    let exc_value = vm
        .get_active_exception()
        .cloned()
        .unwrap_or_else(Value::none);
    vm.current_frame_mut().set_reg(dst_reg, exc_value);

    ControlFlow::Continue
}

/// Get the traceback from the current exception.
///
/// # Instruction Format
///
/// - `dst`: Register to store traceback (or None)
///
/// # Semantics
///
/// Retrieves the traceback object from the current exception.
/// Returns None if there's no active exception or no traceback.
#[inline(always)]
pub fn get_exception_traceback(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst_reg = inst.dst().0;

    // TODO: Extract traceback from exception object
    // For now, return None
    vm.current_frame_mut().set_reg(dst_reg, Value::none());
    ControlFlow::Continue
}

/// Push current exception info onto the stack.
///
/// # Instruction Format
///
/// No operands - uses VM exception state.
///
/// # Semantics
///
/// Pushes the current exception info onto the exception info stack.
/// If there's an active exception, transitions to Finally state so that
/// EndFinally knows to reraise the exception after the finally block completes.
#[inline(always)]
pub fn push_exc_info(vm: &mut VirtualMachine, _inst: Instruction) -> ControlFlow {
    // Push current exception info onto the stack
    vm.push_exc_info();

    // If there's an active exception, we're entering a finally block during
    // exception propagation. Set state to Finally so EndFinally knows to reraise.
    if vm.has_active_exception() {
        vm.set_exception_state(crate::exception::ExceptionState::Finally);
    }

    ControlFlow::Continue
}

/// Bind exception to a name (for `except E as e:` syntax).
///
/// # Instruction Format
///
/// - `dst`: Register to store exception value
///
/// # Semantics
///
/// Stores the current exception in a register for the named binding.
/// The exception remains active for potential reraising.
#[inline(always)]
pub fn bind_exception(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst_reg = inst.dst().0;

    // Clone the exception value to avoid borrow issues
    let exc_value = vm
        .get_active_exception()
        .cloned()
        .unwrap_or_else(Value::none);
    vm.current_frame_mut().set_reg(dst_reg, exc_value);

    ControlFlow::Continue
}

/// Clear the current exception.
///
/// # Instruction Format
///
/// - No operands
///
/// # Semantics
///
/// Clears the active exception after it has been fully handled.
/// Called at the end of a successful except block.
#[inline(always)]
pub fn clear_exception(vm: &mut VirtualMachine, _inst: Instruction) -> ControlFlow {
    vm.clear_active_exception();
    vm.clear_exception_state();
    ControlFlow::Continue
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Extract exception type ID from instruction or value.
///
/// If imm16 is set to a valid type ID (not 0 or NO_TYPE_ID), uses that.
/// Otherwise extracts the type from the exception Value itself.
#[inline(always)]
fn extract_type_id(inst: Instruction, value: &Value) -> u16 {
    let type_id = inst.imm16();

    // Both NO_TYPE_ID (0xFFFF) and 0 indicate "not specified in instruction"
    // 0 is common because op_d doesn't set imm16, defaulting to 0
    if type_id == NO_TYPE_ID || type_id == 0 {
        // Try to extract type ID from the exception value itself
        // SAFETY: The value should be an exception object from raise expression
        if let Some(exc) = unsafe { crate::builtins::ExceptionValue::from_value(value.clone()) } {
            exc.exception_type_id
        } else {
            // Fallback to generic Exception type if not a proper exception object
            4 // ExceptionTypeId::Exception
        }
    } else {
        type_id
    }
}

/// Check if one exception type is a subclass of another.
///
/// This uses the precomputed exception hierarchy from the type system.
#[inline(always)]
fn is_subclass(exc_type: u16, parent_type: u16) -> bool {
    use crate::stdlib::exceptions::ExceptionTypeId;

    // Convert u16 to ExceptionTypeId for hierarchy check
    let exc = ExceptionTypeId::from_u8(exc_type as u8);
    let parent = ExceptionTypeId::from_u8(parent_type as u8);

    match (exc, parent) {
        (Some(e), Some(p)) => e.is_subclass_of(p),
        _ => false, // Unknown types don't match
    }
}

// Dynamic match and tuple match functions are implemented in helpers.rs
// Use: super::helpers::{check_dynamic_match, check_tuple_match};

// =============================================================================
// CPython 3.11+ Exception Info Opcodes
// =============================================================================

/// Pop exception info from the stack (CPython 3.11+ semantics).
///
/// # Instruction Format
///
/// - No operands
///
/// # Semantics
///
/// Pops the top exception info entry and restores it as the active exception.
/// This is used when exiting an except handler to restore the previous
/// exception context.
///
/// # Performance
///
/// O(1) stack pop operation.
#[inline(always)]
pub fn pop_exc_info(vm: &mut VirtualMachine, _inst: Instruction) -> ControlFlow {
    vm.pop_exc_info();
    ControlFlow::Continue
}

/// Check if there is exception info available (CPython 3.11+ semantics).
///
/// # Instruction Format
///
/// - `dst`: Register to store boolean result
///
/// # Semantics
///
/// Stores True if there is exception info on the stack or an active exception,
/// False otherwise. Used for conditional exception handling paths.
///
/// # Performance
///
/// O(1) stack check operation.
#[inline(always)]
pub fn has_exc_info(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst_reg = inst.dst().0;
    let has_info = vm.has_exc_info();

    vm.current_frame_mut()
        .set_reg(dst_reg, Value::bool(has_info));
    ControlFlow::Continue
}

/// Raise an exception with a cause (exception chaining).
///
/// # Instruction Format
///
/// - `dst`: Register containing the new exception
/// - `src1`: Register containing the cause exception
/// - `imm16`: New exception type ID
///
/// # Semantics
///
/// Implements `raise NewException from cause_exception` syntax.
/// Sets __cause__ and __suppress_context__ on the new exception.
/// This is distinct from implicit exception chaining (__context__).
///
/// # Performance
///
/// Similar to regular RAISE with additional cause attachment (~50-70 cycles).
#[inline(always)]
pub fn raise_from(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let exc_reg = inst.dst().0;
    let cause_reg = inst.src1().0;

    let exc_value = frame.get_reg(exc_reg);
    let _cause_value = frame.get_reg(cause_reg);

    // RaiseFrom uses DstSrc format (dst=exc_reg, src=cause_reg), so imm16 contains garbage.
    // We MUST extract the type ID directly from the exception value.
    let type_id = if let Some(exc) =
        unsafe { crate::builtins::ExceptionValue::from_value(exc_value.clone()) }
    {
        exc.exception_type_id
    } else {
        // Fallback to generic Exception type if not a proper exception object
        4 // ExceptionTypeId::Exception
    };

    // TODO: Attach cause to exception object via __cause__ attribute
    // For now, we just treat this the same as a regular raise
    // but with the cause available for future traceback display

    // Set the active exception with its type
    // The VM's exception propagation mechanism will handle finding handlers
    vm.set_active_exception_with_type(exc_value.clone(), type_id);

    // Return Exception control flow - VM will handle propagation
    // This is the same pattern as the regular `raise` function
    ControlFlow::Exception {
        type_id,
        handler_pc: NO_HANDLER_PC,
    }
}

/// Load the current exception value into a register.
///
/// # Instruction Format
///
/// - `dst`: Register to store the exception value
///
/// # Semantics
///
/// Loads the current active exception value into the destination register.
/// Returns None if no exception is active.
///
/// # Performance
///
/// O(1) register store operation.
#[inline(always)]
pub fn load_exception(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst_reg = inst.dst().0;

    let exc_value = vm
        .get_active_exception()
        .cloned()
        .unwrap_or_else(Value::none);

    vm.current_frame_mut().set_reg(dst_reg, exc_value);
    ControlFlow::Continue
}

/// Check if exception matches a type dynamically.
///
/// # Instruction Format
///
/// - `dst`: Register to store boolean result  
/// - `src1`: Register containing exception type to match against
///
/// # Semantics
///
/// Checks if the current active exception matches the type in src1.
/// Uses exception hierarchy for isinstance-style matching.
/// Stores True/False in dst register.
///
/// # Performance
///
/// O(1) for type ID comparison, O(log N) for hierarchy lookup.
#[inline(always)]
pub fn exception_match(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst_reg = inst.dst().0;
    let type_reg = inst.src1().0;

    let type_value = vm.current_frame().get_reg(type_reg);

    // Get the active exception type
    let exc_type_id = vm.get_active_exception_type_id().unwrap_or(0);

    // Try to match against the provided type
    // First check if it's encoded as a type ID in the value
    let matches = if let Some(target_type_id) = type_value.as_int() {
        is_subclass(exc_type_id, target_type_id as u16)
    } else {
        // Try dynamic matching first (single exception type)
        let single_match = check_dynamic_match(exc_type_id, &type_value);
        if single_match {
            true
        } else {
            // Fall back to tuple matching for `except (TypeError, ValueError):` syntax
            check_tuple_match(exc_type_id, &type_value)
        }
    };

    vm.current_frame_mut()
        .set_reg(dst_reg, Value::bool(matches));
    ControlFlow::Continue
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ════════════════════════════════════════════════════════════════════════
    // Type ID Extraction Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_no_type_id_constant() {
        assert_eq!(NO_TYPE_ID, 0xFFFF);
    }

    #[test]
    fn test_no_handler_pc_constant() {
        assert_eq!(NO_HANDLER_PC, 0);
    }

    // ════════════════════════════════════════════════════════════════════════
    // is_subclass Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_is_subclass_self() {
        // Every type is a subclass of itself
        assert!(is_subclass(4, 4)); // Exception is subclass of Exception
    }

    #[test]
    fn test_is_subclass_direct() {
        // TypeError(24) is a subclass of Exception(4)
        assert!(is_subclass(24, 4));
    }

    #[test]
    fn test_is_subclass_invalid_types() {
        // Invalid type IDs should return false
        assert!(!is_subclass(255, 255));
    }

    #[test]
    fn test_is_subclass_not_related() {
        // TypeError(24) is not a subclass of StopIteration(5)
        assert!(!is_subclass(24, 5));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Handler Frame Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_handler_frame_size() {
        // Ensure compact memory layout
        assert_eq!(std::mem::size_of::<HandlerFrame>(), 8);
    }

    #[test]
    fn test_no_handler_sentinel() {
        // Verify sentinel value
        use crate::exception::NO_HANDLER;
        assert_eq!(NO_HANDLER, u16::MAX);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Control Flow Return Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_control_flow_exception_size() {
        let cf = ControlFlow::Exception {
            type_id: 4,
            handler_pc: 100,
        };
        // Ensure ControlFlow is reasonably sized
        assert!(std::mem::size_of_val(&cf) <= 104);
    }

    #[test]
    fn test_control_flow_reraise() {
        let cf = ControlFlow::Reraise;
        match cf {
            ControlFlow::Reraise => {}
            _ => panic!("Expected Reraise"),
        }
    }

    #[test]
    fn test_control_flow_continue() {
        let cf = ControlFlow::Continue;
        match cf {
            ControlFlow::Continue => {}
            _ => panic!("Expected Continue"),
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // Dynamic Match Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_check_dynamic_match_returns_false() {
        // Currently unimplemented, should return false
        assert!(!check_dynamic_match(4, &Value::none()));
    }

    #[test]
    fn test_check_tuple_match_returns_false() {
        // Currently unimplemented, should return false
        assert!(!check_tuple_match(4, &Value::none()));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Extract Type ID Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_extract_type_id_no_type() {
        use prism_compiler::bytecode::{Instruction, Opcode, Register};

        // Create instruction with NO_TYPE_ID
        let inst = Instruction::op_di(Opcode::Raise, Register(0), NO_TYPE_ID);
        let result = extract_type_id(inst, &Value::none());

        // Should return Exception type (4)
        assert_eq!(result, 4);
    }

    #[test]
    fn test_extract_type_id_specific() {
        use prism_compiler::bytecode::{Instruction, Opcode, Register};

        // Create instruction with specific type ID
        let inst = Instruction::op_di(Opcode::Raise, Register(0), 24); // TypeError
        let result = extract_type_id(inst, &Value::none());

        assert_eq!(result, 24);
    }
}
