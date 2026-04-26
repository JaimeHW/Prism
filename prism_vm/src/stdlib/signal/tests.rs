use super::*;

fn builtin_from_value(value: Value) -> &'static BuiltinFunctionObject {
    let ptr = value
        .as_object_ptr()
        .expect("expected builtin function object");
    unsafe { &*(ptr as *const BuiltinFunctionObject) }
}

#[test]
fn test_module_exposes_expected_signal_attributes() {
    let module = SignalModule::new();

    assert_eq!(
        module.get_attr("SIGINT").unwrap().as_int(),
        Some(SIGINT),
        "SIGINT constant should round-trip"
    );
    assert_eq!(
        module.get_attr("SIGTERM").unwrap().as_int(),
        Some(SIGTERM),
        "SIGTERM constant should round-trip"
    );
    assert_eq!(
        module.get_attr("NSIG").unwrap().as_int(),
        Some(NSIG),
        "NSIG constant should round-trip"
    );
    assert!(module.get_attr("signal").unwrap().as_object_ptr().is_some());
    assert!(
        module
            .get_attr("getsignal")
            .unwrap()
            .as_object_ptr()
            .is_some()
    );
}

#[test]
fn test_signal_handler_roundtrip_for_sigint() {
    let module = SignalModule::new();
    let signal = builtin_from_value(module.get_attr("signal").expect("signal should exist"));
    let getsignal = builtin_from_value(
        module
            .get_attr("getsignal")
            .expect("getsignal should exist"),
    );
    let default_handler = module
        .get_attr("default_int_handler")
        .expect("default_int_handler should exist");
    let original = signal
        .call(&[Value::int(SIGINT).unwrap(), default_handler])
        .expect("signal should accept default handler");

    let previous = getsignal
        .call(&[Value::int(SIGINT).unwrap()])
        .expect("getsignal should succeed");
    assert_eq!(previous.as_object_ptr(), default_handler.as_object_ptr());

    let returned = signal
        .call(&[Value::int(SIGINT).unwrap(), Value::int(SIG_IGN).unwrap()])
        .expect("signal should succeed");
    assert_eq!(returned.as_object_ptr(), default_handler.as_object_ptr());
    assert_eq!(
        getsignal
            .call(&[Value::int(SIGINT).unwrap()])
            .expect("getsignal should succeed")
            .as_int(),
        Some(SIG_IGN)
    );

    let _ = signal.call(&[Value::int(SIGINT).unwrap(), original]);
}

#[test]
fn test_default_int_handler_validates_arity() {
    let module = SignalModule::new();
    let handler = builtin_from_value(
        module
            .get_attr("default_int_handler")
            .expect("default_int_handler should exist"),
    );

    let err = handler
        .call(&[Value::int(SIGINT).unwrap()])
        .expect_err("missing frame argument should error");
    assert!(err.to_string().contains("2 positional arguments"));

    let err = handler
        .call(&[Value::int(SIGINT).unwrap(), Value::none()])
        .expect_err("default SIGINT handler should raise KeyboardInterrupt");
    match err {
        BuiltinError::Raised(runtime) => match runtime.kind() {
            crate::error::RuntimeErrorKind::Exception { type_id, .. } => {
                assert_eq!(*type_id, ExceptionTypeId::KeyboardInterrupt.as_u8() as u16);
            }
            other => panic!("expected KeyboardInterrupt exception, got {other:?}"),
        },
        other => panic!("expected raised KeyboardInterrupt, got {other:?}"),
    }
}
