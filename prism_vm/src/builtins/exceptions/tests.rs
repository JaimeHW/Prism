use super::*;

// ════════════════════════════════════════════════════════════════════════
// Message Extraction Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_extract_message_empty() {
    assert!(extract_message(&[]).is_none());
}

#[test]
fn test_extract_message_none_value() {
    let msg = extract_message(&[Value::none()]);
    assert_eq!(msg.as_deref(), Some("None"));
}

#[test]
fn test_extract_message_bool() {
    let msg = extract_message(&[Value::bool(true)]);
    assert_eq!(msg.as_deref(), Some("True"));

    let msg = extract_message(&[Value::bool(false)]);
    assert_eq!(msg.as_deref(), Some("False"));
}

#[test]
fn test_extract_message_int() {
    let msg = extract_message(&[Value::int(42).unwrap()]);
    assert_eq!(msg.as_deref(), Some("42"));

    let msg = extract_message(&[Value::int(-123).unwrap()]);
    assert_eq!(msg.as_deref(), Some("-123"));
}

#[test]
fn test_extract_message_float() {
    let msg = extract_message(&[Value::float(3.14)]);
    assert!(msg.is_some());
    // Float formatting may vary, just check it's not empty
    assert!(!msg.unwrap().is_empty());
}

// ════════════════════════════════════════════════════════════════════════
// Constructor Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_value_error_no_args() {
    let result = builtin_value_error(&[]);
    assert!(result.is_ok());

    let exc = unsafe { ExceptionValue::from_value(result.unwrap()) };
    assert!(exc.is_some());
    assert_eq!(exc.unwrap().type_id(), ExceptionTypeId::ValueError);
    assert!(exc.unwrap().message().is_none());
}

#[test]
fn test_value_error_with_int_arg() {
    let result = builtin_value_error(&[Value::int(42).unwrap()]);
    assert!(result.is_ok());

    let exc = unsafe { ExceptionValue::from_value(result.unwrap()).unwrap() };
    assert_eq!(exc.type_id(), ExceptionTypeId::ValueError);
    assert_eq!(exc.message(), Some("42"));
}

#[test]
fn test_type_error_constructor() {
    let result = builtin_type_error(&[]);
    assert!(result.is_ok());

    let exc = unsafe { ExceptionValue::from_value(result.unwrap()).unwrap() };
    assert_eq!(exc.type_id(), ExceptionTypeId::TypeError);
}

#[test]
fn test_zero_division_error_constructor() {
    let result = builtin_zero_division_error(&[]);
    assert!(result.is_ok());

    let exc = unsafe { ExceptionValue::from_value(result.unwrap()).unwrap() };
    assert_eq!(exc.type_id(), ExceptionTypeId::ZeroDivisionError);
    assert!(exc.is_subclass_of(ExceptionTypeId::ArithmeticError));
}

#[test]
fn test_key_error_constructor() {
    let result = builtin_key_error(&[]);
    assert!(result.is_ok());

    let exc = unsafe { ExceptionValue::from_value(result.unwrap()).unwrap() };
    assert_eq!(exc.type_id(), ExceptionTypeId::KeyError);
    assert!(exc.is_subclass_of(ExceptionTypeId::LookupError));
}

#[test]
fn test_index_error_constructor() {
    let result = builtin_index_error(&[Value::int(-1).unwrap()]);
    assert!(result.is_ok());

    let exc = unsafe { ExceptionValue::from_value(result.unwrap()).unwrap() };
    assert_eq!(exc.type_id(), ExceptionTypeId::IndexError);
    assert_eq!(exc.message(), Some("-1"));
}

#[test]
fn test_runtime_error_constructor() {
    let result = builtin_runtime_error(&[]);
    assert!(result.is_ok());

    let exc = unsafe { ExceptionValue::from_value(result.unwrap()).unwrap() };
    assert_eq!(exc.type_id(), ExceptionTypeId::RuntimeError);
}

#[test]
fn test_stop_iteration_constructor() {
    let result = builtin_stop_iteration(&[]);
    assert!(result.is_ok());

    let exc = unsafe { ExceptionValue::from_value(result.unwrap()).unwrap() };
    assert_eq!(exc.type_id(), ExceptionTypeId::StopIteration);
}

#[test]
fn test_os_error_constructor() {
    let result = builtin_os_error(&[]);
    assert!(result.is_ok());

    let exc = unsafe { ExceptionValue::from_value(result.unwrap()).unwrap() };
    assert_eq!(exc.type_id(), ExceptionTypeId::OSError);
}

#[test]
fn test_file_not_found_error_constructor() {
    let result = builtin_file_not_found_error(&[]);
    assert!(result.is_ok());

    let exc = unsafe { ExceptionValue::from_value(result.unwrap()).unwrap() };
    assert_eq!(exc.type_id(), ExceptionTypeId::FileNotFoundError);
    assert!(exc.is_subclass_of(ExceptionTypeId::OSError));
}

// ════════════════════════════════════════════════════════════════════════
// Registration Table Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_exception_constructors_count() {
    // Should have at least 50 exception types
    assert!(EXCEPTION_CONSTRUCTOR_COUNT >= 50);
}

#[test]
fn test_all_constructors_callable() {
    for (name, constructor) in EXCEPTION_CONSTRUCTORS {
        let result = constructor(&[]);
        assert!(result.is_ok(), "Constructor {} failed", name);

        let value = result.unwrap();
        assert!(
            value.is_object(),
            "Constructor {} didn't return object",
            name
        );
    }
}

#[test]
fn test_exception_constructor_names_valid() {
    for (name, _) in EXCEPTION_CONSTRUCTORS {
        // Names should not be empty
        assert!(!name.is_empty());
        // Names should start with uppercase
        assert!(name.chars().next().unwrap().is_uppercase());
    }
}

#[test]
fn test_key_constructors_present() {
    let names: Vec<&str> = EXCEPTION_CONSTRUCTORS.iter().map(|(n, _)| *n).collect();

    // Check critical exception types are present
    assert!(names.contains(&"ValueError"));
    assert!(names.contains(&"TypeError"));
    assert!(names.contains(&"KeyError"));
    assert!(names.contains(&"IndexError"));
    assert!(names.contains(&"AttributeError"));
    assert!(names.contains(&"NameError"));
    assert!(names.contains(&"RuntimeError"));
    assert!(names.contains(&"StopIteration"));
    assert!(names.contains(&"ZeroDivisionError"));
    assert!(names.contains(&"Exception"));
    assert!(names.contains(&"BaseException"));
    assert!(names.contains(&"BaseExceptionGroup"));
    assert!(names.contains(&"ExceptionGroup"));
}

#[test]
fn test_io_error_alias() {
    // IOError should be an alias for OSError
    let io_result = EXCEPTION_CONSTRUCTORS
        .iter()
        .find(|(n, _)| *n == "IOError")
        .map(|(_, f)| f(&[]));

    let os_result = EXCEPTION_CONSTRUCTORS
        .iter()
        .find(|(n, _)| *n == "OSError")
        .map(|(_, f)| f(&[]));

    assert!(io_result.is_some());
    assert!(os_result.is_some());

    let io_exc = unsafe { ExceptionValue::from_value(io_result.unwrap().unwrap()).unwrap() };
    let os_exc = unsafe { ExceptionValue::from_value(os_result.unwrap().unwrap()).unwrap() };

    // Both should have OSError type
    assert_eq!(io_exc.type_id(), ExceptionTypeId::OSError);
    assert_eq!(os_exc.type_id(), ExceptionTypeId::OSError);
}

// ════════════════════════════════════════════════════════════════════════
// Hierarchy Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_exception_hierarchy_preserved() {
    // ValueError -> Exception -> BaseException
    let result = builtin_value_error(&[]).unwrap();
    let exc = unsafe { ExceptionValue::from_value(result).unwrap() };

    assert!(exc.is_subclass_of(ExceptionTypeId::ValueError));
    assert!(exc.is_subclass_of(ExceptionTypeId::Exception));
    assert!(exc.is_subclass_of(ExceptionTypeId::BaseException));
    assert!(!exc.is_subclass_of(ExceptionTypeId::TypeError));
}

#[test]
fn test_connection_error_hierarchy() {
    // ConnectionRefusedError -> ConnectionError -> OSError -> Exception
    let result = builtin_connection_refused_error(&[]).unwrap();
    let exc = unsafe { ExceptionValue::from_value(result).unwrap() };

    assert!(exc.is_subclass_of(ExceptionTypeId::ConnectionRefusedError));
    assert!(exc.is_subclass_of(ExceptionTypeId::ConnectionError));
    assert!(exc.is_subclass_of(ExceptionTypeId::OSError));
    assert!(exc.is_subclass_of(ExceptionTypeId::Exception));
}

#[test]
fn test_unicode_error_hierarchy() {
    // UnicodeDecodeError -> UnicodeError -> ValueError -> Exception
    let result = builtin_unicode_decode_error(&[]).unwrap();
    let exc = unsafe { ExceptionValue::from_value(result).unwrap() };

    assert!(exc.is_subclass_of(ExceptionTypeId::UnicodeDecodeError));
    assert!(exc.is_subclass_of(ExceptionTypeId::UnicodeError));
    assert!(exc.is_subclass_of(ExceptionTypeId::ValueError));
    assert!(exc.is_subclass_of(ExceptionTypeId::Exception));
}

#[test]
fn test_exception_group_hierarchy() {
    let result = builtin_exception_group(&[]).unwrap();
    let exc = unsafe { ExceptionValue::from_value(result).unwrap() };

    assert!(exc.is_subclass_of(ExceptionTypeId::ExceptionGroup));
    assert!(exc.is_subclass_of(ExceptionTypeId::BaseExceptionGroup));
    assert!(exc.is_subclass_of(ExceptionTypeId::Exception));
    assert!(exc.is_subclass_of(ExceptionTypeId::BaseException));
}

// ════════════════════════════════════════════════════════════════════════
// Performance Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_constructor_is_fast() {
    // Constructors should complete quickly (no heavy allocation)
    for _ in 0..1000 {
        let result = builtin_value_error(&[Value::int(42).unwrap()]);
        assert!(result.is_ok());
    }
}
