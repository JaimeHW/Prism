use super::*;
use crate::stdlib::exceptions::traceback::FrameInfo;

// ════════════════════════════════════════════════════════════════════════
// ExceptionArgs Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_args_empty() {
    let args = ExceptionArgs::empty();
    assert!(args.is_empty());
    assert_eq!(args.len(), 0);
}

#[test]
fn test_args_single() {
    let args = ExceptionArgs::single(Value::none());
    assert!(!args.is_empty());
    assert_eq!(args.len(), 1);
    assert!(args.first().is_some());
}

#[test]
fn test_args_from_slice() {
    let values = [Value::none(), Value::none()];
    let args = ExceptionArgs::from_slice(&values);
    assert_eq!(args.len(), 2);
}

#[test]
fn test_args_from_iter() {
    let args = ExceptionArgs::from_iter(vec![Value::none(), Value::none(), Value::none()]);
    assert_eq!(args.len(), 3);
}

#[test]
fn test_args_as_slice() {
    let args = ExceptionArgs::from_slice(&[Value::none(), Value::none()]);
    assert_eq!(args.as_slice().len(), 2);
}

// ════════════════════════════════════════════════════════════════════════
// ExceptionObject Creation Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_exception_new() {
    let exc = ExceptionObject::new(ExceptionTypeId::TypeError);
    assert_eq!(exc.type_id(), ExceptionTypeId::TypeError);
    assert!(!exc.is_normalized());
    assert!(!exc.is_flyweight());
}

#[test]
fn test_exception_with_message() {
    let exc = ExceptionObject::with_message(ExceptionTypeId::ValueError, "invalid value");
    assert_eq!(exc.type_id(), ExceptionTypeId::ValueError);
    assert!(exc.is_normalized());
    assert!(exc.args().is_some());
}

#[test]
fn test_exception_with_args() {
    let args = ExceptionArgs::from_slice(&[Value::none()]);
    let exc = ExceptionObject::with_args(ExceptionTypeId::KeyError, args);
    assert!(exc.is_normalized());
    assert!(!exc.args_or_empty().is_empty());
}

#[test]
fn test_exception_flyweight() {
    let exc = ExceptionObject::flyweight(ExceptionTypeId::StopIteration);
    assert!(exc.is_flyweight());
    assert!(exc.is_normalized());
}

// ════════════════════════════════════════════════════════════════════════
// Type Information Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_exception_type_name() {
    let exc = ExceptionObject::new(ExceptionTypeId::IndexError);
    assert_eq!(exc.type_name(), "IndexError");
}

#[test]
fn test_exception_is_instance() {
    let exc = ExceptionObject::new(ExceptionTypeId::TypeError);
    assert!(exc.is_instance(ExceptionTypeId::TypeError));
    assert!(!exc.is_instance(ExceptionTypeId::ValueError));
}

#[test]
fn test_exception_is_subclass() {
    let exc = ExceptionObject::new(ExceptionTypeId::FileNotFoundError);
    assert!(exc.is_subclass(ExceptionTypeId::OSError));
    assert!(exc.is_subclass(ExceptionTypeId::Exception));
    assert!(!exc.is_subclass(ExceptionTypeId::TypeError));
}

// ════════════════════════════════════════════════════════════════════════
// Args Access Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_exception_args_none() {
    let exc = ExceptionObject::new(ExceptionTypeId::TypeError);
    assert!(exc.args().is_none());
}

#[test]
fn test_exception_args_or_empty() {
    let exc = ExceptionObject::new(ExceptionTypeId::TypeError);
    let args = exc.args_or_empty();
    assert!(args.is_empty());
}

#[test]
fn test_exception_set_args() {
    let exc = ExceptionObject::new(ExceptionTypeId::TypeError);
    let args = ExceptionArgs::single(Value::none());
    assert!(exc.set_args(args).is_ok());
    assert!(exc.args().is_some());
}

#[test]
fn test_exception_set_args_twice_fails() {
    let exc = ExceptionObject::new(ExceptionTypeId::TypeError);
    let args1 = ExceptionArgs::single(Value::none());
    let args2 = ExceptionArgs::single(Value::none());

    assert!(exc.set_args(args1).is_ok());
    assert!(exc.set_args(args2).is_err());
}

// ════════════════════════════════════════════════════════════════════════
// Traceback Access Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_exception_traceback_none() {
    let exc = ExceptionObject::new(ExceptionTypeId::TypeError);
    assert!(exc.traceback().is_none());
    assert!(!exc.has_traceback());
}

#[test]
fn test_exception_traceback_or_empty() {
    let exc = ExceptionObject::new(ExceptionTypeId::TypeError);
    let tb = exc.traceback_or_empty();
    assert!(tb.is_empty());
}

#[test]
fn test_exception_set_traceback() {
    let exc = ExceptionObject::new(ExceptionTypeId::TypeError);
    let tb = TracebackObject::single(FrameInfo::new(Arc::from("test"), Arc::from("test.py"), 1));

    assert!(exc.set_traceback(tb).is_ok());
    assert!(exc.has_traceback());
}

#[test]
fn test_exception_traceback_mut() {
    let mut exc = ExceptionObject::new(ExceptionTypeId::TypeError);
    let tb = TracebackObject::single(FrameInfo::new(Arc::from("test"), Arc::from("test.py"), 1));
    exc.set_traceback(tb).ok();

    // Should be able to mutate
    let tb_mut = exc.traceback_mut().unwrap();
    tb_mut.push(FrameInfo::new(Arc::from("test2"), Arc::from("test2.py"), 2));

    assert_eq!(exc.traceback().unwrap().len(), 2);
}

// ════════════════════════════════════════════════════════════════════════
// Exception Chaining Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_exception_no_chaining() {
    let exc = ExceptionObject::new(ExceptionTypeId::TypeError);
    assert!(exc.cause().is_none());
    assert!(exc.context().is_none());
}

#[test]
fn test_exception_set_cause() {
    let mut exc = ExceptionObject::new(ExceptionTypeId::RuntimeError);
    let cause = Arc::new(ExceptionObject::new(ExceptionTypeId::ValueError));

    exc.set_cause(cause);
    assert!(exc.cause().is_some());
    assert!(exc.flags().has_cause());
    assert!(exc.suppress_context()); // Should be set when cause is set
}

#[test]
fn test_exception_clear_cause() {
    let mut exc = ExceptionObject::new(ExceptionTypeId::RuntimeError);
    let cause = Arc::new(ExceptionObject::new(ExceptionTypeId::ValueError));

    exc.set_cause(cause);
    assert!(exc.cause().is_some());

    exc.clear_cause();
    assert!(exc.cause().is_none());
    assert!(!exc.flags().has_cause());
}

#[test]
fn test_exception_set_context() {
    let mut exc = ExceptionObject::new(ExceptionTypeId::RuntimeError);
    let context = Arc::new(ExceptionObject::new(ExceptionTypeId::ValueError));

    exc.set_context(context);
    assert!(exc.context().is_some());
    assert!(exc.flags().has_context());
}

// ════════════════════════════════════════════════════════════════════════
// Formatting Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_exception_format_no_message() {
    let exc = ExceptionObject::new(ExceptionTypeId::TypeError);
    let formatted = exc.format();
    assert_eq!(formatted, "TypeError");
}

#[test]
fn test_exception_display() {
    let exc = ExceptionObject::new(ExceptionTypeId::ValueError);
    let display = format!("{}", exc);
    assert!(display.contains("ValueError"));
}

#[test]
fn test_exception_debug() {
    let exc = ExceptionObject::new(ExceptionTypeId::KeyError);
    let debug = format!("{:?}", exc);
    assert!(debug.contains("ExceptionObject"));
    assert!(debug.contains("KeyError"));
}

#[test]
fn test_exception_format_with_traceback() {
    let mut exc = ExceptionObject::new(ExceptionTypeId::TypeError);
    let tb = TracebackObject::single(FrameInfo::new(Arc::from("test"), Arc::from("test.py"), 42));
    exc.set_traceback(tb).ok();

    let formatted = exc.format_with_traceback();
    assert!(formatted.contains("Traceback"));
    assert!(formatted.contains("test.py"));
    assert!(formatted.contains("TypeError"));
}

// ════════════════════════════════════════════════════════════════════════
// Memory Layout Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_exception_header_size() {
    // Type ID + Flags = 2 bytes
    assert_eq!(
        std::mem::size_of::<ExceptionTypeId>() + std::mem::size_of::<ExceptionFlags>(),
        2
    );
}

// ════════════════════════════════════════════════════════════════════════
// Control Flow Exception Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_stop_iteration_is_control_flow() {
    let exc = ExceptionObject::new(ExceptionTypeId::StopIteration);
    assert!(exc.type_id().is_control_flow());
}

#[test]
fn test_generator_exit_is_control_flow() {
    let exc = ExceptionObject::new(ExceptionTypeId::GeneratorExit);
    assert!(exc.type_id().is_control_flow());
}

// ════════════════════════════════════════════════════════════════════════
// Arc Support Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_exception_ref() {
    let exc: ExceptionRef = Arc::new(ExceptionObject::new(ExceptionTypeId::TypeError));
    assert_eq!(exc.type_id(), ExceptionTypeId::TypeError);
}
