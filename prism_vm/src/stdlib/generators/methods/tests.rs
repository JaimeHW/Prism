use super::*;
use prism_code::CodeObject;
use std::sync::Arc;

fn test_code() -> Arc<CodeObject> {
    Arc::new(CodeObject::new("test_generator", "<test>"))
}

// ════════════════════════════════════════════════════════════════════════
// SendResult Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_send_result_yielded() {
    let result = SendResult::Yielded(Value::int(42).unwrap());
    assert!(result.is_yielded());
    assert!(!result.is_done());
    assert_eq!(result.yielded().unwrap().as_int().unwrap(), 42);
    assert!(result.returned().is_none());
}

#[test]
fn test_send_result_returned() {
    let result = SendResult::Returned(Value::none());
    assert!(!result.is_yielded());
    assert!(result.is_done());
    assert!(result.yielded().is_none());
    assert!(result.returned().is_some());
}

#[test]
fn test_send_result_error() {
    let result = SendResult::Error(GeneratorError::Exhausted);
    assert!(!result.is_yielded());
    assert!(result.is_done());
}

// ════════════════════════════════════════════════════════════════════════
// Send Protocol Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_prepare_send_none_to_created() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);

    let result = prepare_send(&mut generator, Value::none());
    assert_eq!(result, Ok(GeneratorState::Created));
    assert!(generator.is_running());
}

#[test]
fn test_prepare_send_value_to_created_fails() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);

    let result = prepare_send(&mut generator, Value::int(42).unwrap());
    assert_eq!(result, Err(GeneratorError::CantSendNonNone));
}

#[test]
fn test_prepare_send_to_suspended() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);

    // Start and suspend
    generator.try_start();
    let regs = [Value::none(); 256];
    generator.suspend(10, 0, &regs, LivenessMap::from_bits(0));

    let result = prepare_send(&mut generator, Value::int(42).unwrap());
    assert_eq!(result, Ok(GeneratorState::Suspended));
    assert!(generator.is_running());
    assert_eq!(
        generator.peek_receive_value().unwrap().as_int().unwrap(),
        42
    );
}

#[test]
fn test_prepare_send_to_running() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);
    generator.try_start();

    let result = prepare_send(&mut generator, Value::none());
    assert_eq!(result, Err(GeneratorError::AlreadyRunning));
}

#[test]
fn test_prepare_send_to_exhausted() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);
    generator.try_start();
    generator.exhaust();

    let result = prepare_send(&mut generator, Value::none());
    assert_eq!(result, Err(GeneratorError::Exhausted));
}

#[test]
fn test_complete_send_yielded() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);

    let result = complete_send_yielded(&mut generator, Value::int(100).unwrap());
    assert!(result.is_yielded());
    assert_eq!(result.yielded().unwrap().as_int().unwrap(), 100);
}

#[test]
fn test_complete_send_returned() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);
    generator.try_start();

    let result = complete_send_returned(&mut generator, Value::int(200).unwrap());
    assert!(result.is_done());
    assert_eq!(result.returned().unwrap().as_int().unwrap(), 200);
    assert!(generator.is_exhausted());
}

// ════════════════════════════════════════════════════════════════════════
// ThrowResult Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_throw_result_yielded() {
    let result = ThrowResult::Yielded(Value::int(42).unwrap());
    assert!(result.is_yielded());
    assert!(!result.is_done());
}

#[test]
fn test_throw_result_returned() {
    let result = ThrowResult::Returned(Value::none());
    assert!(!result.is_yielded());
    assert!(result.is_done());
}

#[test]
fn test_throw_result_propagated() {
    let exc = GeneratorException::new("ValueError", "test");
    let result = ThrowResult::Propagated(exc);
    assert!(!result.is_yielded());
    assert!(result.is_done());
}

// ════════════════════════════════════════════════════════════════════════
// Throw Protocol Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_prepare_throw_to_created() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);

    let exc = GeneratorException::new("ValueError", "test");
    let result = prepare_throw(&mut generator, exc.clone());

    assert!(matches!(result, Err(GeneratorError::ThrownException(_))));
    assert!(generator.is_exhausted());
}

#[test]
fn test_prepare_throw_to_suspended() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);

    // Start and suspend
    generator.try_start();
    let regs = [Value::none(); 256];
    generator.suspend(10, 0, &regs, LivenessMap::from_bits(0));

    let exc = GeneratorException::new("ValueError", "test");
    let result = prepare_throw(&mut generator, exc);

    assert_eq!(result, Ok(GeneratorState::Suspended));
    assert!(generator.is_running());
}

#[test]
fn test_complete_throw_yielded() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);

    let result = complete_throw_yielded(&mut generator, Value::int(50).unwrap());
    assert!(result.is_yielded());
}

#[test]
fn test_complete_throw_returned() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);
    generator.try_start();

    let result = complete_throw_returned(&mut generator, Value::none());
    assert!(result.is_done());
    assert!(generator.is_exhausted());
}

#[test]
fn test_complete_throw_propagated() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);
    generator.try_start();

    let exc = GeneratorException::new("RuntimeError", "unhandled");
    let result = complete_throw_propagated(&mut generator, exc);

    assert!(result.is_done());
    assert!(generator.is_exhausted());
}

// ════════════════════════════════════════════════════════════════════════
// CloseResult Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_close_result_closed() {
    let result = CloseResult::Closed;
    assert!(result.is_ok());
    assert!(!result.is_err());
}

#[test]
fn test_close_result_error() {
    let result = CloseResult::RuntimeError(GeneratorException::new("ValueError", "test"));
    assert!(!result.is_ok());
    assert!(result.is_err());
}

#[test]
fn test_close_result_yielded_in_finally() {
    let result = CloseResult::YieldedInFinally(Value::int(42).unwrap());
    assert!(!result.is_ok());
    assert!(result.is_err());
}

// ════════════════════════════════════════════════════════════════════════
// Close Protocol Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_prepare_close_created() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);

    let result = prepare_close(&mut generator);
    assert!(result.is_ok());
    assert!(generator.is_exhausted());
}

#[test]
fn test_prepare_close_exhausted() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);
    generator.try_start();
    generator.exhaust();

    let result = prepare_close(&mut generator);
    assert!(result.is_ok());
}

#[test]
fn test_prepare_close_running() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);
    generator.try_start();

    let result = prepare_close(&mut generator);
    assert!(result.is_err());
}

#[test]
fn test_prepare_close_suspended() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);

    // Start and suspend
    generator.try_start();
    let regs = [Value::none(); 256];
    generator.suspend(10, 0, &regs, LivenessMap::from_bits(0));

    let result = prepare_close(&mut generator);
    // For suspended, we prepare for GeneratorExit throw
    assert!(result.is_ok());
}

#[test]
fn test_complete_close_caught() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);
    generator.try_start();

    let result = complete_close_caught(&mut generator);
    assert!(result.is_ok());
    assert!(generator.is_exhausted());
}

#[test]
fn test_complete_close_yielded() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);
    generator.try_start();

    let result = complete_close_yielded(&mut generator, Value::int(42).unwrap());
    assert!(result.is_err());
    assert!(matches!(result, CloseResult::YieldedInFinally(_)));
}

#[test]
fn test_complete_close_generator_exit() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);
    generator.try_start();

    let exc = generator_exit();
    let result = complete_close_exception(&mut generator, exc);
    assert!(result.is_ok());
}

#[test]
fn test_complete_close_other_exception() {
    let code = test_code();
    let mut generator = GeneratorObject::new(code);
    generator.try_start();

    let exc = GeneratorException::new("RuntimeError", "something else");
    let result = complete_close_exception(&mut generator, exc);
    assert!(result.is_err());
}

// ════════════════════════════════════════════════════════════════════════
// Utility Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_generator_exit() {
    let exc = generator_exit();
    assert_eq!(exc.type_name, "GeneratorExit");
}

#[test]
fn test_stop_iteration() {
    let exc = stop_iteration(None);
    assert_eq!(exc.type_name, "StopIteration");
    assert!(exc.value.is_none());

    let exc_with_value = stop_iteration(Some(Value::int(42).unwrap()));
    assert!(exc_with_value.value.is_some());
}
