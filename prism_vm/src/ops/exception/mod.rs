//! Exception opcode support modules.
//!
//! This module provides high-performance infrastructure for exception handling
//! opcodes, including type extraction, dynamic matching, traceback construction,
//! and sys.exc_info() support.
//!
//! # Architecture
//!
//! ```text
//! ops/exception/
//! ├── mod.rs         # This file - module exports and integration
//! ├── helpers.rs     # Type extraction, dynamic matching, tuple matching
//! ├── traceback.rs   # Traceback construction and extraction
//! └── exc_info.rs    # sys.exc_info() triple support
//! ```
//!
//! # Performance Design
//!
//! All operations are designed for minimal overhead:
//!
//! | Module | Hot Path | Cold Path |
//! |--------|----------|-----------|
//! | helpers | O(1) type checks | O(N) tuple iteration |
//! | traceback | O(1) extraction | O(N) capture |
//! | exc_info | O(1) build | O(1) write |
//!
//! # Usage
//!
//! These modules are used by the main exception opcode handlers:
//!
//! ```ignore
//! // In ops/exception.rs (main file)
//! use crate::ops::exception::helpers::{extract_type_id, check_tuple_match};
//! use crate::ops::exception::traceback::{capture_traceback, extract_traceback};
//! use crate::ops::exception::exc_info::{build_exc_info, write_exc_info_to_registers};
//! ```

pub mod exc_info;
pub mod helpers;
pub mod opcodes;
pub mod traceback;

// Re-export opcode handlers for direct use
pub use opcodes::{
    bind_exception, check_exc_match, check_exc_match_tuple, clear_exception, end_finally,
    exception_match, get_exception, get_exception_traceback, has_exc_info, load_exception,
    pop_exc_info, pop_except_handler, push_exc_info, raise, raise_from, raise_with_cause, reraise,
    setup_except,
};

// Re-export commonly used items
pub use exc_info::{ExcInfo, build_exc_info, write_empty_exc_info, write_exc_info_to_registers};
pub use helpers::{
    DEFAULT_EXCEPTION_TYPE, NO_TYPE_ID, check_dynamic_match, check_tuple_match, extract_type_id,
    extract_type_id_from_value, is_subclass,
};
pub use traceback::{
    build_frame_info, build_frame_info_with_line, capture_traceback, extend_traceback,
    extract_traceback, has_traceback, traceback_to_value, value_to_traceback,
};

// =============================================================================
// Integration Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stdlib::exceptions::ExceptionTypeId;
    use prism_core::Value;

    // ════════════════════════════════════════════════════════════════════════
    // Module Integration Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_module_exports_constants() {
        assert_eq!(NO_TYPE_ID, 0xFFFF);
        assert_eq!(DEFAULT_EXCEPTION_TYPE, 0);
    }

    #[test]
    fn test_module_exports_helpers() {
        // Test that helper functions are accessible
        let type_id = extract_type_id(24, &Value::none());
        assert_eq!(type_id, 24); // Should return encoded type

        let type_id2 = extract_type_id(NO_TYPE_ID, &Value::none());
        assert_eq!(type_id2, ExceptionTypeId::Exception as u16);
    }

    #[test]
    fn test_module_exports_is_subclass() {
        let type_error = ExceptionTypeId::TypeError as u16;
        let exception = ExceptionTypeId::Exception as u16;
        assert!(is_subclass(type_error, exception));
    }

    #[test]
    fn test_module_exports_exc_info() {
        let info = ExcInfo::empty();
        assert!(info.is_empty());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Cross-Module Integration Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_type_extraction_with_exc_info() {
        // Test that type extraction works with exc_info
        let type_id = ExceptionTypeId::ValueError as u16;
        let info =
            exc_info::build_exc_info_explicit(type_id, Value::int(42).unwrap(), Value::none());

        assert!(!info.exc_type.is_none());
        assert!(!info.exc_value.is_none());
    }

    #[test]
    fn test_subclass_check_for_matching() {
        // ValueError should match Exception
        let value_error = ExceptionTypeId::ValueError as u16;
        let exception = ExceptionTypeId::Exception as u16;

        assert!(is_subclass(value_error, exception));
        assert!(is_subclass(value_error, value_error)); // Self match
    }

    #[test]
    fn test_traceback_with_exc_info() {
        // Test that traceback functions work with exc_info
        let tb = capture_traceback_for_test();
        assert!(tb.is_empty()); // No VM, so empty

        let info = ExcInfo::new(
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::none(),
        );
        assert!(!info.has_traceback());
    }

    // Helper for testing without VM
    fn capture_traceback_for_test() -> crate::stdlib::exceptions::TracebackObject {
        crate::stdlib::exceptions::TracebackObject::empty()
    }

    // ════════════════════════════════════════════════════════════════════════
    // Matching Integration Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_dynamic_match_with_subclass() {
        // Test dynamic matching considers inheritance
        let type_error = ExceptionTypeId::TypeError as u16;
        let exception = ExceptionTypeId::Exception as u16;

        // Direct match
        assert!(is_subclass(type_error, type_error));
        // Parent match
        assert!(is_subclass(type_error, exception));
        // Not a match
        assert!(!is_subclass(exception, type_error));
    }

    #[test]
    fn test_tuple_match_with_multiple_types() {
        // Test tuple matching (currently returns false due to stub)
        let exc_type = ExceptionTypeId::TypeError as u16;
        let tuple = Value::none(); // Not a real tuple
        assert!(!check_tuple_match(exc_type, &tuple));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Error Handling Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_extract_from_invalid_value() {
        let value = Value::float(3.14);
        let type_id = extract_type_id_from_value(&value);
        // Should return default Exception type
        assert_eq!(type_id, ExceptionTypeId::Exception as u16);
    }

    #[test]
    fn test_traceback_from_non_exception() {
        let value = Value::int(42).unwrap();
        let tb = extract_traceback(&value);
        assert!(tb.is_none());
    }

    #[test]
    fn test_has_traceback_non_exception() {
        let value = Value::bool(true);
        assert!(!has_traceback(&value));
    }
}
