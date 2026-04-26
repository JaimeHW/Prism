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
