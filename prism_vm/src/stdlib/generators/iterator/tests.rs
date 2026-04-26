use super::*;
use prism_code::CodeObject;
use std::sync::Arc;

fn test_code() -> Arc<CodeObject> {
    Arc::new(CodeObject::new("test_generator", "<test>"))
}

// ════════════════════════════════════════════════════════════════════════
// IterResult Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_iter_result_yielded() {
    let result = IterResult::Yielded(Value::int(42).unwrap());
    assert!(result.is_yielded());
    assert!(!result.is_returned());
    assert!(!result.is_raised());
    assert_eq!(result.yielded_value().unwrap().as_int().unwrap(), 42);
}

#[test]
fn test_iter_result_returned() {
    let result = IterResult::Returned(Value::none());
    assert!(!result.is_yielded());
    assert!(result.is_returned());
    assert!(!result.is_raised());
    assert!(result.returned_value().unwrap().is_none());
}

#[test]
fn test_iter_result_raised() {
    let result = IterResult::Raised(GeneratorError::Exhausted);
    assert!(!result.is_yielded());
    assert!(!result.is_returned());
    assert!(result.is_raised());
}

#[test]
fn test_stop_iteration_value() {
    let returned = IterResult::Returned(Value::int(100).unwrap());
    let stop_value = returned.into_stop_iteration_value();
    assert!(stop_value.is_some());
    assert_eq!(stop_value.unwrap().as_int().unwrap(), 100);

    let yielded = IterResult::Yielded(Value::int(50).unwrap());
    assert!(yielded.into_stop_iteration_value().is_none());
}

// ════════════════════════════════════════════════════════════════════════
// GeneratorError Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_error_already_running() {
    let err = GeneratorError::AlreadyRunning;
    assert!(!err.is_exhausted());
    assert!(!err.is_fatal());
    assert!(err.to_string().contains("already executing"));
}

#[test]
fn test_error_exhausted() {
    let err = GeneratorError::Exhausted;
    assert!(err.is_exhausted());
    assert!(!err.is_fatal());
    assert!(err.to_string().contains("StopIteration"));
}

#[test]
fn test_error_cant_send_non_none() {
    let err = GeneratorError::CantSendNonNone;
    assert!(!err.is_exhausted());
    assert!(!err.is_fatal());
    assert!(err.to_string().contains("non-None"));
}

#[test]
fn test_error_runtime() {
    let err = GeneratorError::runtime("test error");
    assert!(!err.is_exhausted());
    assert!(err.is_fatal());
    assert!(err.to_string().contains("test error"));
}

#[test]
fn test_error_stop_iteration_with_value() {
    let err = GeneratorError::StopIteration(Some(Value::int(42).unwrap()));
    assert!(err.is_exhausted());
    assert!(err.to_string().contains("with value"));
}

// ════════════════════════════════════════════════════════════════════════
// GeneratorException Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_exception_new() {
    let exc = GeneratorException::new("ValueError", "invalid value");
    assert_eq!(exc.type_name, "ValueError");
    assert_eq!(exc.message, "invalid value");
    assert!(exc.value.is_none());
    assert!(exc.to_string().contains("ValueError"));
    assert!(exc.to_string().contains("invalid value"));
}

#[test]
fn test_exception_with_value() {
    let exc = GeneratorException::with_value("TypeError", "wrong type", Value::int(99).unwrap());
    assert_eq!(exc.type_name, "TypeError");
    assert!(exc.value.is_some());
    assert_eq!(exc.value.unwrap().as_int().unwrap(), 99);
}

// ════════════════════════════════════════════════════════════════════════
// Validation Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_validate_resumable_created() {
    let code = test_code();
    let generator = GeneratorObject::new(code);
    assert_eq!(validate_resumable(&generator), Ok(GeneratorState::Created));
}

#[test]
fn test_validate_resumable_running() {
    let code = test_code();
    let generator = GeneratorObject::new(code);
    generator.try_start();
    assert_eq!(
        validate_resumable(&generator),
        Err(GeneratorError::AlreadyRunning)
    );
}

#[test]
fn test_validate_resumable_exhausted() {
    let code = test_code();
    let generator = GeneratorObject::new(code);
    generator.try_start();
    generator.exhaust();
    assert_eq!(
        validate_resumable(&generator),
        Err(GeneratorError::Exhausted)
    );
}

#[test]
fn test_validate_send_none_to_created() {
    let code = test_code();
    let generator = GeneratorObject::new(code);
    assert!(validate_send_value(&generator, None).is_ok());
    assert!(validate_send_value(&generator, Some(Value::none())).is_ok());
}

#[test]
fn test_validate_send_value_to_created() {
    let code = test_code();
    let generator = GeneratorObject::new(code);
    assert_eq!(
        validate_send_value(&generator, Some(Value::int(42).unwrap())),
        Err(GeneratorError::CantSendNonNone)
    );
}

#[test]
fn test_validate_send_to_exhausted() {
    let code = test_code();
    let generator = GeneratorObject::new(code);
    generator.try_start();
    generator.exhaust();
    assert_eq!(
        validate_send_value(&generator, None),
        Err(GeneratorError::Exhausted)
    );
}

// ════════════════════════════════════════════════════════════════════════
// Prepare Iteration Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_prepare_first_iteration() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);

    let result = prepare_iteration(&mut generator, None);
    assert_eq!(result, Ok(GeneratorState::Created));
    assert!(generator.is_running());
}

#[test]
fn test_prepare_with_send_value() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);

    // First iteration must be None
    prepare_iteration(&mut generator, None).unwrap();

    // Simulate suspend
    let regs = [Value::none(); 256];
    generator.suspend(
        10,
        0,
        &regs,
        super::super::storage::LivenessMap::from_bits(0),
    );

    // Now we can send a value
    let mut generator_new = GeneratorObject::new(test_code());
    generator_new.try_start();
    generator_new.suspend(
        10,
        0,
        &regs,
        super::super::storage::LivenessMap::from_bits(0),
    );

    let result = prepare_iteration(&mut generator_new, Some(Value::int(42).unwrap()));
    assert_eq!(result, Ok(GeneratorState::Suspended));
    assert_eq!(
        generator_new
            .peek_receive_value()
            .unwrap()
            .as_int()
            .unwrap(),
        42
    );
}

#[test]
fn test_prepare_iteration_already_running() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);
    generator.try_start();

    let result = prepare_iteration(&mut generator, None);
    assert_eq!(result, Err(GeneratorError::AlreadyRunning));
}

#[test]
fn test_prepare_iteration_exhausted() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);
    generator.try_start();
    generator.exhaust();

    let result = prepare_iteration(&mut generator, None);
    assert_eq!(result, Err(GeneratorError::Exhausted));
}

// ════════════════════════════════════════════════════════════════════════
// GeneratorIterator Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_generator_iterator_new() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);
    let iter = GeneratorIterator::new(&mut generator);

    assert!(!iter.is_running());
    assert!(!iter.is_exhausted());
}

#[test]
fn test_generator_iterator_running() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);
    generator.try_start();

    let iter = GeneratorIterator::new(&mut generator);
    assert!(iter.is_running());
}

#[test]
fn test_generator_iterator_exhausted() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);
    generator.try_start();
    generator.exhaust();

    let iter = GeneratorIterator::new(&mut generator);
    assert!(iter.is_exhausted());
}

#[test]
fn test_generator_iterator_access() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);

    let mut iter = GeneratorIterator::new(&mut generator);

    // Can access generator through iterator
    assert_eq!(iter.generator().state(), GeneratorState::Created);
    iter.generator_mut().try_start();
    assert!(iter.is_running());
}
