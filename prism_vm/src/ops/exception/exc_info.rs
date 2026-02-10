//! Exception info (sys.exc_info()) support.
//!
//! This module provides functions for building and accessing the exception
//! info triple: (type, value, traceback).
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        Exception Info Triple                             │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │  ┌─────────────┐   ┌─────────────────┐   ┌─────────────────┐             │
//! │  │  exc_type   │   │   exc_value     │   │  exc_traceback  │             │
//! │  │  (Value)    │   │    (Value)      │   │    (Value)      │             │
//! │  │             │   │                 │   │                 │             │
//! │  │ Type class  │   │ Exception       │   │ Traceback       │             │
//! │  │ or None     │   │ instance        │   │ object          │             │
//! │  │             │   │ or None         │   │ or None         │             │
//! │  └─────────────┘   └─────────────────┘   └─────────────────┘             │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! The exception info triple is used by:
//! - `sys.exc_info()` builtin
//! - Exception handlers that need full context
//! - Debugging and logging systems
//!
//! # Performance
//!
//! | Operation | Complexity | Notes |
//! |-----------|------------|-------|
//! | build_exc_info | O(1) | Field access only |
//! | get_exc_type | O(1) | Type ID to Value |
//! | get_exc_value | O(1) | Direct access |
//! | get_exc_traceback | O(1) | Clone if present |

use crate::VirtualMachine;
use crate::stdlib::exceptions::ExceptionTypeId;
use prism_core::Value;

// =============================================================================
// Exception Info Triple
// =============================================================================

/// The exception info triple: (type, value, traceback).
///
/// This struct holds the three components of `sys.exc_info()`:
/// - `exc_type`: The exception class/type
/// - `exc_value`: The exception instance
/// - `exc_traceback`: The traceback object (or None)
#[derive(Clone)]
pub struct ExcInfo {
    /// Exception type (class object or None)
    pub exc_type: Value,
    /// Exception instance (or None)
    pub exc_value: Value,
    /// Traceback object (or None)
    pub exc_traceback: Value,
}

impl ExcInfo {
    /// Creates an empty exception info (all None).
    #[inline(always)]
    pub fn empty() -> Self {
        Self {
            exc_type: Value::none(),
            exc_value: Value::none(),
            exc_traceback: Value::none(),
        }
    }

    /// Creates exception info with just a type and value.
    #[inline]
    pub fn from_type_value(exc_type: Value, exc_value: Value) -> Self {
        Self {
            exc_type,
            exc_value,
            exc_traceback: Value::none(),
        }
    }

    /// Creates exception info with all three components.
    #[inline]
    pub fn new(exc_type: Value, exc_value: Value, exc_traceback: Value) -> Self {
        Self {
            exc_type,
            exc_value,
            exc_traceback,
        }
    }

    /// Returns true if there is no active exception.
    ///
    /// An ExcInfo is considered empty if there's no exception value,
    /// even if a type is set. This follows Python's semantics where
    /// `sys.exc_info()` returns `(None, None, None)` when there's no
    /// active exception.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.exc_value.is_none()
    }

    /// Returns true if there is an active exception.
    #[inline(always)]
    pub fn is_active(&self) -> bool {
        !self.exc_value.is_none()
    }

    /// Returns true if a traceback is present.
    #[inline(always)]
    pub fn has_traceback(&self) -> bool {
        !self.exc_traceback.is_none()
    }

    /// Writes the triple to three consecutive registers.
    ///
    /// This is the most efficient way to pass exc_info to Python code.
    #[inline]
    pub fn write_to_registers(&self, vm: &mut VirtualMachine, base_reg: u8) {
        let frame = vm.current_frame_mut();
        frame.set_reg(base_reg, self.exc_type.clone());
        frame.set_reg(base_reg + 1, self.exc_value.clone());
        frame.set_reg(base_reg + 2, self.exc_traceback.clone());
    }

    /// Converts to a tuple (type, value, traceback).
    pub fn to_tuple(&self) -> (Value, Value, Value) {
        (
            self.exc_type.clone(),
            self.exc_value.clone(),
            self.exc_traceback.clone(),
        )
    }
}

impl std::fmt::Debug for ExcInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExcInfo")
            .field("exc_type", &self.exc_type)
            .field("exc_value", &self.exc_value)
            .field("has_traceback", &self.has_traceback())
            .finish()
    }
}

impl Default for ExcInfo {
    fn default() -> Self {
        Self::empty()
    }
}

// =============================================================================
// Building Exception Info
// =============================================================================

/// Build exception info from the current VM state.
///
/// Returns the (type, value, traceback) triple for `sys.exc_info()`.
///
/// # Returns
///
/// - If there's an active exception: (type, value, traceback)
/// - If no active exception: (None, None, None)
#[inline]
pub fn build_exc_info(vm: &VirtualMachine) -> ExcInfo {
    // Check if there's an active exception
    if !vm.has_active_exception() {
        return ExcInfo::empty();
    }

    // Get the exception value
    let exc_value = match vm.get_active_exception() {
        Some(v) => v,
        None => return ExcInfo::empty(),
    };

    // Build the exception type Value
    let type_id = vm.get_active_exception_type_id().unwrap_or(0);
    let exc_type = build_type_value(type_id);

    // Get traceback (may be None)
    let exc_traceback = build_traceback_value(&exc_value);

    ExcInfo::new(exc_type, exc_value.clone(), exc_traceback)
}

/// Build exception info from explicit components.
///
/// Used when constructing exc_info without VM state (e.g., in tests).
#[inline]
pub fn build_exc_info_explicit(type_id: u16, exc_value: Value, exc_traceback: Value) -> ExcInfo {
    let exc_type = build_type_value(type_id);
    ExcInfo::new(exc_type, exc_value, exc_traceback)
}

// =============================================================================
// Type Value Construction
// =============================================================================

/// Build a Value representing an exception type from its type ID.
///
/// This creates a type object Value that can be compared with `is`
/// or used with `isinstance()`.
#[inline]
fn build_type_value(type_id: u16) -> Value {
    // Convert type ID to ExceptionTypeId
    match ExceptionTypeId::from_u8(type_id as u8) {
        Some(_exc_type) => {
            // TODO: Create proper type object Value
            // For now, use an int as a placeholder for the type ID
            Value::int(type_id as i64).unwrap_or_else(Value::none)
        }
        None => Value::none(),
    }
}

/// Build a traceback Value from an exception value.
///
/// Extracts the traceback from the exception object if present.
#[inline]
fn build_traceback_value(_exc_value: &Value) -> Value {
    // TODO: Extract traceback from exception object
    // For now, return None
    Value::none()
}

// =============================================================================
// Register Operations
// =============================================================================

/// Write exception info to consecutive registers.
///
/// This is the optimized path for `push_exc_info` opcode.
#[inline]
pub fn write_exc_info_to_registers(vm: &mut VirtualMachine, base_reg: u8) {
    let exc_info = build_exc_info(vm);
    exc_info.write_to_registers(vm, base_reg);
}

/// Write empty exception info to consecutive registers.
///
/// Used when there's no active exception.
#[inline]
pub fn write_empty_exc_info(vm: &mut VirtualMachine, base_reg: u8) {
    let frame = vm.current_frame_mut();
    frame.set_reg(base_reg, Value::none());
    frame.set_reg(base_reg + 1, Value::none());
    frame.set_reg(base_reg + 2, Value::none());
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ════════════════════════════════════════════════════════════════════════
    // ExcInfo Construction Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_exc_info_empty() {
        let info = ExcInfo::empty();
        assert!(info.is_empty());
        assert!(!info.is_active());
        assert!(!info.has_traceback());
    }

    #[test]
    fn test_exc_info_empty_values() {
        let info = ExcInfo::empty();
        assert!(info.exc_type.is_none());
        assert!(info.exc_value.is_none());
        assert!(info.exc_traceback.is_none());
    }

    #[test]
    fn test_exc_info_from_type_value() {
        let exc_type = Value::int(24).unwrap(); // TypeError
        let exc_value = Value::int(100).unwrap(); // placeholder
        let info = ExcInfo::from_type_value(exc_type, exc_value);

        assert!(!info.is_empty());
        assert!(info.is_active());
        assert!(!info.has_traceback());
    }

    #[test]
    fn test_exc_info_new() {
        let exc_type = Value::int(24).unwrap(); // TypeError
        let exc_value = Value::int(100).unwrap();
        let exc_traceback = Value::int(200).unwrap();
        let info = ExcInfo::new(exc_type, exc_value, exc_traceback);

        assert!(!info.is_empty());
        assert!(info.is_active());
        // traceback is not none
        assert!(info.has_traceback());
    }

    #[test]
    fn test_exc_info_with_none_traceback() {
        let info = ExcInfo::new(
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::none(),
        );
        assert!(!info.has_traceback());
    }

    // ════════════════════════════════════════════════════════════════════════
    // ExcInfo Method Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_exc_info_is_empty_with_type_only() {
        // Type present but value is None - still considered empty
        let info = ExcInfo::new(Value::int(1).unwrap(), Value::none(), Value::none());
        assert!(info.is_empty());
    }

    #[test]
    fn test_exc_info_to_tuple() {
        let exc_type = Value::int(1).unwrap();
        let exc_value = Value::int(2).unwrap();
        let exc_traceback = Value::int(3).unwrap();
        let info = ExcInfo::new(exc_type.clone(), exc_value.clone(), exc_traceback.clone());

        let (t, v, tb) = info.to_tuple();
        // Compare the values (they should be equal)
        assert!(!t.is_none());
        assert!(!v.is_none());
        assert!(!tb.is_none());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Default Implementation Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_exc_info_default() {
        let info = ExcInfo::default();
        assert!(info.is_empty());
    }

    #[test]
    fn test_exc_info_debug_format() {
        let info = ExcInfo::empty();
        let debug = format!("{:?}", info);
        assert!(debug.contains("ExcInfo"));
        assert!(debug.contains("exc_type"));
        assert!(debug.contains("exc_value"));
        assert!(debug.contains("has_traceback"));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Type Value Construction Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_build_type_value_valid() {
        let type_id = ExceptionTypeId::TypeError as u16;
        let value = build_type_value(type_id);
        // Currently returns int representation
        assert!(!value.is_none());
    }

    #[test]
    fn test_build_type_value_invalid() {
        let type_id = 255u16; // Invalid type ID
        let value = build_type_value(type_id);
        // Invalid type ID returns None
        assert!(value.is_none());
    }

    #[test]
    fn test_build_type_value_exception() {
        let type_id = ExceptionTypeId::Exception as u16;
        let value = build_type_value(type_id);
        assert!(!value.is_none());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Traceback Value Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_build_traceback_value_none() {
        let exc_value = Value::none();
        let tb = build_traceback_value(&exc_value);
        // Currently returns None as placeholder
        assert!(tb.is_none());
    }

    #[test]
    fn test_build_traceback_value_int() {
        let exc_value = Value::int(42).unwrap();
        let tb = build_traceback_value(&exc_value);
        // Not an exception, returns None
        assert!(tb.is_none());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Explicit Build Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_build_exc_info_explicit() {
        let type_id = ExceptionTypeId::ValueError as u16;
        let exc_value = Value::int(42).unwrap();
        let exc_traceback = Value::none();

        let info = build_exc_info_explicit(type_id, exc_value, exc_traceback);

        assert!(!info.exc_type.is_none());
        assert!(!info.exc_value.is_none());
        assert!(info.exc_traceback.is_none());
    }

    #[test]
    fn test_build_exc_info_explicit_with_traceback() {
        let type_id = ExceptionTypeId::TypeError as u16;
        let exc_value = Value::int(100).unwrap();
        let exc_traceback = Value::int(200).unwrap(); // placeholder

        let info = build_exc_info_explicit(type_id, exc_value, exc_traceback);

        assert!(info.has_traceback());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Edge Case Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_exc_info_clone() {
        let info = ExcInfo::new(
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        );
        let cloned = info.clone();

        assert!(!cloned.exc_type.is_none());
        assert!(!cloned.exc_value.is_none());
        assert!(!cloned.exc_traceback.is_none());
    }

    #[test]
    fn test_exc_info_all_none() {
        let info = ExcInfo::new(Value::none(), Value::none(), Value::none());
        assert!(info.is_empty());
    }

    #[test]
    fn test_build_type_value_base_exception() {
        let type_id = ExceptionTypeId::BaseException as u16;
        let value = build_type_value(type_id);
        assert!(!value.is_none());
    }

    #[test]
    fn test_build_type_value_stop_iteration() {
        let type_id = ExceptionTypeId::StopIteration as u16;
        let value = build_type_value(type_id);
        assert!(!value.is_none());
    }
}
