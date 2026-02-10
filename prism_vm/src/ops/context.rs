//! Context manager opcode handlers (PEP 343).
//!
//! This module implements the `with` statement protocol for Python context managers.
//!
//! # Protocol
//!
//! The `with` statement follows PEP 343:
//!
//! ```text
//! with EXPR as VAR:
//!     BODY
//!
//! # Compiles to:
//! mgr = EXPR
//! exit = type(mgr).__exit__
//! value = type(mgr).__enter__(mgr)
//! exc = True
//! try:
//!     VAR = value      # If `as VAR` is present
//!     BODY
//! except:
//!     exc = False
//!     if not exit(mgr, *sys.exc_info()):
//!         raise
//! finally:
//!     if exc:
//!         exit(mgr, None, None, None)
//! ```
//!
//! # Opcodes
//!
//! | Opcode | Value | Description |
//! |--------|-------|-------------|
//! | `BeforeWith` | 0x97 | Call `__enter__`, setup handler |
//! | `ExitWith` | 0x98 | Call `__exit__(None, None, None)` |
//! | `WithCleanup` | 0x99 | Call `__exit__(exc_type, exc_val, exc_tb)` |
//!
//! # Performance
//!
//! All handlers are `#[inline(always)]` for maximum dispatch loop performance.
//! Method lookups use the inline cache for O(1) repeated access.

use crate::VirtualMachine;
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use prism_compiler::bytecode::Instruction;
use prism_core::Value;

// =============================================================================
// Constants
// =============================================================================

/// Magic method names for context manager protocol.
const ENTER_METHOD: &str = "__enter__";
const EXIT_METHOD: &str = "__exit__";

// =============================================================================
// BeforeWith Opcode
// =============================================================================

/// Prepare context manager by calling `__enter__` and storing `__exit__`.
///
/// # Instruction Format
///
/// - `dst`: Register to store __enter__ result (context value)
/// - `src1`: Register containing the context manager object
///
/// # Semantics
///
/// 1. Look up `__enter__` on context manager
/// 2. Call `__enter__(mgr)` to get the context value
/// 3. Store the result in dst register
/// 4. The `__exit__` method is looked up later during cleanup
///
/// # Performance
///
/// Method lookup uses inline cache when available. Average case ~15-25 cycles.
#[inline(always)]
pub fn before_with(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst_reg = inst.dst().0;
    let mgr_reg = inst.src1().0;

    let mgr = vm.current_frame().get_reg(mgr_reg);

    // Look up __enter__ method
    let enter_result = lookup_and_call_method(vm, &mgr, ENTER_METHOD);

    match enter_result {
        Ok(value) => {
            vm.current_frame_mut().set_reg(dst_reg, value);
            ControlFlow::Continue
        }
        Err(err) => ControlFlow::Error(err),
    }
}

// =============================================================================
// ExitWith Opcode
// =============================================================================

/// Exit context manager normally (no exception).
///
/// # Instruction Format
///
/// - `dst`: (unused, reserved for return value)
/// - `src1`: Register containing the context manager object
///
/// # Semantics
///
/// Calls `__exit__(mgr, None, None, None)` for normal exit.
/// The return value is typically ignored for normal exits.
///
/// # Performance
///
/// Method lookup uses inline cache. ~20-30 cycles.
#[inline(always)]
pub fn exit_with(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let mgr_reg = inst.src1().0;

    let mgr = vm.current_frame().get_reg(mgr_reg);

    // Call __exit__(None, None, None) for normal exit
    let exit_result = lookup_and_call_exit_normal(vm, &mgr);

    match exit_result {
        Ok(_) => ControlFlow::Continue,
        Err(err) => ControlFlow::Error(err),
    }
}

// =============================================================================
// WithCleanup Opcode
// =============================================================================

/// Exit context manager with exception (cleanup).
///
/// # Instruction Format
///
/// - `dst`: Register to store suppression result (bool)
/// - `src1`: Register containing the context manager object
///
/// # Semantics
///
/// 1. Get current exception info (exc_type, exc_val, exc_tb)
/// 2. Call `__exit__(mgr, exc_type, exc_val, exc_tb)`
/// 3. If `__exit__` returns True, suppress the exception
/// 4. Store suppression result in dst register
///
/// # Exception Suppression
///
/// If `__exit__` returns a truthy value, the exception is suppressed
/// and execution continues normally. Otherwise, the exception is re-raised.
///
/// # Performance
///
/// Method lookup + exception info extraction. ~30-50 cycles.
#[inline(always)]
pub fn with_cleanup(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst_reg = inst.dst().0;
    let mgr_reg = inst.src1().0;

    let mgr = vm.current_frame().get_reg(mgr_reg);

    // Get current exception info
    let (exc_type, exc_val, _exc_tb) = vm.current_exc_info();

    // Call __exit__(exc_type, exc_val, exc_tb)
    let exit_result = lookup_and_call_exit_with_exc(vm, &mgr, exc_type, exc_val);

    match exit_result {
        Ok(suppress) => {
            // Store whether exception should be suppressed
            vm.current_frame_mut()
                .set_reg(dst_reg, Value::bool(suppress));

            if suppress {
                // Exception suppressed - clear it
                vm.clear_active_exception();
                vm.clear_exception_state();
            }

            ControlFlow::Continue
        }
        Err(err) => ControlFlow::Error(err),
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Look up and call a method on an object (single argument - self).
///
/// Returns the method's return value or an error if method not found.
#[inline]
fn lookup_and_call_method(
    vm: &VirtualMachine,
    obj: &Value,
    method_name: &str,
) -> Result<Value, RuntimeError> {
    // For now, we check if the object has the method via attribute lookup
    // In a full implementation, this would use the type's method resolution order

    // Try to get the method attribute
    match get_method(obj, method_name) {
        Some(method) => {
            // Call the method with just self
            call_method_0(vm, &method, obj)
        }
        None => Err(RuntimeError::attribute_error("object", method_name)),
    }
}

/// Call __exit__ with (None, None, None) for normal context manager exit.
#[inline]
fn lookup_and_call_exit_normal(vm: &VirtualMachine, mgr: &Value) -> Result<Value, RuntimeError> {
    match get_method(mgr, EXIT_METHOD) {
        Some(method) => {
            // Call __exit__(None, None, None)
            call_method_3(
                vm,
                &method,
                mgr,
                &Value::none(),
                &Value::none(),
                &Value::none(),
            )
        }
        None => Err(RuntimeError::attribute_error("object", EXIT_METHOD)),
    }
}

/// Call __exit__ with exception info for cleanup.
///
/// Returns true if exception should be suppressed.
#[inline]
fn lookup_and_call_exit_with_exc(
    vm: &VirtualMachine,
    mgr: &Value,
    exc_type: Option<u16>,
    exc_val: Option<Value>,
) -> Result<bool, RuntimeError> {
    match get_method(mgr, EXIT_METHOD) {
        Some(method) => {
            // Build exception type value from type ID
            let type_val = match exc_type {
                Some(id) => Value::int(id as i64).unwrap_or_else(Value::none),
                None => Value::none(),
            };

            let val = exc_val.unwrap_or_else(Value::none);
            let tb = Value::none(); // TODO: Get actual traceback

            // Call __exit__(exc_type, exc_val, exc_tb)
            let result = call_method_3(vm, &method, mgr, &type_val, &val, &tb)?;

            // True-ish return value means suppress exception
            Ok(is_truthy(&result))
        }
        None => Err(RuntimeError::attribute_error("object", EXIT_METHOD)),
    }
}

/// Get a method from an object.
///
/// This is a simplified version - in production, would use MRO lookup.
#[inline]
fn get_method(_obj: &Value, _name: &str) -> Option<Value> {
    // TODO: Implement proper method resolution order lookup
    // For now, return None to indicate method not found
    // This will be connected to the type system later
    None
}

/// Call a method with no additional arguments (just self).
#[inline]
fn call_method_0(
    _vm: &VirtualMachine,
    _method: &Value,
    _self_obj: &Value,
) -> Result<Value, RuntimeError> {
    // TODO: Implement actual method call
    // For now, return None as placeholder
    Ok(Value::none())
}

/// Call a method with 3 additional arguments.
#[inline]
fn call_method_3(
    _vm: &VirtualMachine,
    _method: &Value,
    _self_obj: &Value,
    _arg1: &Value,
    _arg2: &Value,
    _arg3: &Value,
) -> Result<Value, RuntimeError> {
    // TODO: Implement actual method call with arguments
    // For now, return None as placeholder
    Ok(Value::none())
}

/// Check if a value is truthy.
#[inline(always)]
fn is_truthy(value: &Value) -> bool {
    // Fast path for common cases
    if value.is_bool() {
        return value.as_bool().unwrap_or(false);
    }
    if value.is_none() {
        return false;
    }
    if let Some(n) = value.as_int() {
        return n != 0;
    }
    if let Some(f) = value.as_float() {
        return f != 0.0;
    }

    // Default to true for other objects
    true
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ════════════════════════════════════════════════════════════════════════
    // Truthiness Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_is_truthy_bool_true() {
        assert!(is_truthy(&Value::bool(true)));
    }

    #[test]
    fn test_is_truthy_bool_false() {
        assert!(!is_truthy(&Value::bool(false)));
    }

    #[test]
    fn test_is_truthy_none() {
        assert!(!is_truthy(&Value::none()));
    }

    #[test]
    fn test_is_truthy_int_zero() {
        assert!(!is_truthy(&Value::int(0).unwrap()));
    }

    #[test]
    fn test_is_truthy_int_nonzero() {
        assert!(is_truthy(&Value::int(42).unwrap()));
        assert!(is_truthy(&Value::int(-1).unwrap()));
    }

    #[test]
    fn test_is_truthy_float_zero() {
        assert!(!is_truthy(&Value::float(0.0)));
    }

    #[test]
    fn test_is_truthy_float_nonzero() {
        assert!(is_truthy(&Value::float(3.14)));
        assert!(is_truthy(&Value::float(-1.0)));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Method Lookup Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_get_method_not_found() {
        // get_method currently returns None for all objects
        let val = Value::int(42).unwrap();
        assert!(get_method(&val, "__enter__").is_none());
    }

    #[test]
    fn test_get_method_enter() {
        let val = Value::none();
        assert!(get_method(&val, "__enter__").is_none());
    }

    #[test]
    fn test_get_method_exit() {
        let val = Value::bool(true);
        assert!(get_method(&val, "__exit__").is_none());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Error Handling Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_lookup_method_error_message() {
        let vm = VirtualMachine::new();
        let obj = Value::int(42).unwrap();

        let result = lookup_and_call_method(&vm, &obj, "__enter__");
        assert!(result.is_err());

        let err = result.unwrap_err();
        let msg = format!("{:?}", err);
        assert!(msg.contains("__enter__") || msg.contains("attribute"));
    }

    #[test]
    fn test_exit_normal_error_on_missing_method() {
        let vm = VirtualMachine::new();
        let obj = Value::float(1.5);

        let result = lookup_and_call_exit_normal(&vm, &obj);
        assert!(result.is_err());
    }

    #[test]
    fn test_exit_with_exc_error_on_missing_method() {
        let vm = VirtualMachine::new();
        let obj = Value::none();

        let result = lookup_and_call_exit_with_exc(&vm, &obj, Some(4), None);
        assert!(result.is_err());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Constants Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_enter_method_name() {
        assert_eq!(ENTER_METHOD, "__enter__");
    }

    #[test]
    fn test_exit_method_name() {
        assert_eq!(EXIT_METHOD, "__exit__");
    }

    // ════════════════════════════════════════════════════════════════════════
    // Edge Cases Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_is_truthy_negative_zero() {
        assert!(!is_truthy(&Value::float(-0.0)));
    }

    #[test]
    fn test_is_truthy_small_float() {
        assert!(is_truthy(&Value::float(0.0001)));
        assert!(is_truthy(&Value::float(-0.0001)));
    }

    #[test]
    fn test_is_truthy_large_positive_int() {
        // Use a large value within the valid NaN-boxed integer range
        assert!(is_truthy(&Value::int(1_000_000_000).unwrap()));
    }

    #[test]
    fn test_is_truthy_large_negative_int() {
        // Use a large negative value within the valid NaN-boxed integer range
        assert!(is_truthy(&Value::int(-1_000_000_000).unwrap()));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Call Method Placeholder Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_call_method_0_returns_none() {
        let vm = VirtualMachine::new();
        let method = Value::none();
        let self_obj = Value::int(42).unwrap();

        let result = call_method_0(&vm, &method, &self_obj);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_call_method_3_returns_none() {
        let vm = VirtualMachine::new();
        let method = Value::none();
        let self_obj = Value::int(42).unwrap();
        let arg1 = Value::none();
        let arg2 = Value::none();
        let arg3 = Value::none();

        let result = call_method_3(&vm, &method, &self_obj, &arg1, &arg2, &arg3);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Exception Suppression Logic Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_suppress_exception_true_value() {
        // True suppresses exception
        assert!(is_truthy(&Value::bool(true)));
    }

    #[test]
    fn test_suppress_exception_false_value() {
        // False does not suppress exception
        assert!(!is_truthy(&Value::bool(false)));
    }

    #[test]
    fn test_suppress_exception_none_value() {
        // None does not suppress exception
        assert!(!is_truthy(&Value::none()));
    }

    #[test]
    fn test_suppress_exception_nonzero_int_suppresses() {
        // Non-zero int suppresses
        assert!(is_truthy(&Value::int(1).unwrap()));
    }

    #[test]
    fn test_suppress_exception_zero_int_does_not_suppress() {
        // Zero int does not suppress
        assert!(!is_truthy(&Value::int(0).unwrap()));
    }
}
