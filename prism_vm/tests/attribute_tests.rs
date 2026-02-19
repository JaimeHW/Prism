//! Comprehensive tests for attribute mutation operations.
//!
//! Tests for GetAttr, SetAttr, DelAttr opcodes and getattr/setattr/hasattr/delattr builtins.
//!
//! Coverage:
//! - Builtin function semantics (argument validation, type errors)
//! - Error handling for non-object types

use prism_core::Value;

// =============================================================================
// Builtin Function Tests
// =============================================================================

mod builtin_tests {
    use super::*;
    use prism_core::intern::intern;
    use prism_runtime::object::shape::shape_registry;
    use prism_runtime::object::shaped_object::ShapedObject;
    use prism_runtime::types::string::StringObject;
    use prism_vm::builtins::{
        BuiltinError, builtin_delattr, builtin_getattr, builtin_hasattr, builtin_setattr,
    };

    fn new_object_value() -> (Value, *mut ShapedObject) {
        let object = ShapedObject::with_empty_shape(shape_registry().empty_shape());
        let ptr = Box::into_raw(Box::new(object));
        (Value::object_ptr(ptr as *const ()), ptr)
    }

    unsafe fn drop_boxed<T>(ptr: *mut T) {
        drop(unsafe { Box::from_raw(ptr) });
    }

    // =========================================================================
    // getattr() Argument Validation
    // =========================================================================

    #[test]
    fn test_getattr_too_few_args() {
        let result = builtin_getattr(&[Value::none()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_getattr_too_many_args() {
        let result = builtin_getattr(&[Value::none(), Value::none(), Value::none(), Value::none()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_getattr_non_string_name() {
        // Name must be a string - passing an int should fail with TypeError
        let result = builtin_getattr(&[Value::none(), Value::int(42).unwrap()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_getattr_non_string_name_with_default() {
        // Even with a default, non-string name should fail
        let result = builtin_getattr(&[Value::none(), Value::int(42).unwrap(), Value::bool(true)]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    // =========================================================================
    // setattr() Argument Validation
    // =========================================================================

    #[test]
    fn test_setattr_too_few_args() {
        let result = builtin_setattr(&[Value::none(), Value::none()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_setattr_too_few_single_arg() {
        let result = builtin_setattr(&[Value::none()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_setattr_too_many_args() {
        let result = builtin_setattr(&[Value::none(), Value::none(), Value::none(), Value::none()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_setattr_non_string_name() {
        // Name must be a string
        let result = builtin_setattr(&[Value::none(), Value::int(42).unwrap(), Value::bool(true)]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    // =========================================================================
    // hasattr() Argument Validation
    // =========================================================================

    #[test]
    fn test_hasattr_too_few_args() {
        let result = builtin_hasattr(&[Value::none()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_hasattr_no_args() {
        let result = builtin_hasattr(&[]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_hasattr_too_many_args() {
        let result = builtin_hasattr(&[Value::none(), Value::none(), Value::none()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_hasattr_non_string_name() {
        // Name must be a string
        let result = builtin_hasattr(&[Value::none(), Value::int(42).unwrap()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    // =========================================================================
    // delattr() Argument Validation
    // =========================================================================

    #[test]
    fn test_delattr_too_few_args() {
        let result = builtin_delattr(&[Value::none()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_delattr_no_args() {
        let result = builtin_delattr(&[]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_delattr_too_many_args() {
        let result = builtin_delattr(&[Value::none(), Value::none(), Value::none()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_delattr_non_string_name() {
        // Name must be a string
        let result = builtin_delattr(&[Value::none(), Value::int(42).unwrap()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    // =========================================================================
    // Cross-Type Error Scenarios
    // =========================================================================

    #[test]
    fn test_getattr_bool_name_fails() {
        let result = builtin_getattr(&[Value::int(100).unwrap(), Value::bool(true)]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_setattr_float_name_fails() {
        let result =
            builtin_setattr(&[Value::int(100).unwrap(), Value::float(3.14), Value::none()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_hasattr_none_name_fails() {
        // None is not a valid attribute name
        let result = builtin_hasattr(&[Value::int(100).unwrap(), Value::none()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_delattr_none_name_fails() {
        // None is not a valid attribute name
        let result = builtin_delattr(&[Value::int(100).unwrap(), Value::none()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_attribute_roundtrip_with_tagged_name() {
        let (obj, obj_ptr) = new_object_value();
        let name = Value::string(intern("field"));

        builtin_setattr(&[obj, name, Value::int(10).unwrap()]).unwrap();
        assert_eq!(builtin_getattr(&[obj, name]).unwrap().as_int(), Some(10));
        assert!(builtin_hasattr(&[obj, name]).unwrap().as_bool().unwrap());

        builtin_delattr(&[obj, name]).unwrap();
        assert!(!builtin_hasattr(&[obj, name]).unwrap().as_bool().unwrap());
        assert!(matches!(
            builtin_getattr(&[obj, name]).unwrap_err(),
            BuiltinError::AttributeError(_)
        ));

        unsafe { drop_boxed(obj_ptr) };
    }

    #[test]
    fn test_attribute_name_heap_string_object() {
        let (obj, obj_ptr) = new_object_value();
        let string_ptr = Box::into_raw(Box::new(StringObject::new("heap_name")));
        let name = Value::object_ptr(string_ptr as *const ());

        builtin_setattr(&[obj, name, Value::int(22).unwrap()]).unwrap();
        assert_eq!(builtin_getattr(&[obj, name]).unwrap().as_int(), Some(22));
        builtin_delattr(&[obj, name]).unwrap();
        assert!(!builtin_hasattr(&[obj, name]).unwrap().as_bool().unwrap());

        unsafe { drop_boxed(string_ptr) };
        unsafe { drop_boxed(obj_ptr) };
    }

    #[test]
    fn test_setattr_none_is_not_deletion() {
        let (obj, obj_ptr) = new_object_value();
        let name = Value::string(intern("nullable"));

        builtin_setattr(&[obj, name, Value::none()]).unwrap();
        assert!(builtin_hasattr(&[obj, name]).unwrap().as_bool().unwrap());
        assert!(builtin_getattr(&[obj, name]).unwrap().is_none());

        builtin_delattr(&[obj, name]).unwrap();
        assert!(!builtin_hasattr(&[obj, name]).unwrap().as_bool().unwrap());

        unsafe { drop_boxed(obj_ptr) };
    }
}

// =============================================================================
// Additional Integration Test Notes
// =============================================================================
//
// Full integration tests with ShapedObject require:
// 1. Creating StringObject values for attribute names
// 2. Creating ShapedObject values as test targets
//
// These would be best tested via the VM ops directly in the ops module tests,
// or via end-to-end Python execution tests.
