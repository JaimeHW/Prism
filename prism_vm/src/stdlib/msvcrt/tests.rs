use super::*;

fn builtin_from_value(value: Value) -> &'static BuiltinFunctionObject {
    let ptr = value
        .as_object_ptr()
        .expect("expected builtin function object");
    unsafe { &*(ptr as *const BuiltinFunctionObject) }
}

#[test]
fn test_msvcrt_module_exposes_windows_bootstrap_surface() {
    let module = MsvcrtModule::new();

    assert_eq!(module.name(), "msvcrt");
    assert!(module.get_attr("__doc__").is_ok());
    assert!(module.get_attr("get_osfhandle").is_ok());
    assert!(module.get_attr("open_osfhandle").is_ok());
    assert!(module.get_attr("setmode").is_ok());
    assert!(module.get_attr("GetErrorMode").is_ok());
    assert!(module.get_attr("SetErrorMode").is_ok());
    assert!(module.get_attr("SEM_NOGPFAULTERRORBOX").is_ok());
    assert!(module.get_attr("CrtSetReportMode").is_err());
}

#[test]
fn test_os_handle_helpers_validate_integer_arguments() {
    let err = msvcrt_get_osfhandle(&[Value::string(intern("fd"))])
        .expect_err("get_osfhandle should validate fd type");
    assert!(matches!(err, BuiltinError::TypeError(_)));

    let err = msvcrt_open_osfhandle(&[
        Value::string(intern("handle")),
        Value::int(0).expect("flag should fit"),
    ])
    .expect_err("open_osfhandle should validate handle type");
    assert!(matches!(err, BuiltinError::TypeError(_)));
}

#[test]
fn test_set_error_mode_returns_previous_mode_without_changing_state() {
    let old = msvcrt_get_error_mode(&[]).expect("GetErrorMode should succeed");
    let set = builtin_from_value(MsvcrtModule::new().get_attr("SetErrorMode").unwrap());
    let previous = set
        .call(&[old])
        .expect("SetErrorMode should accept the current mode");

    assert_eq!(previous.as_int(), old.as_int());
}

#[test]
fn test_kbhit_reports_no_pending_console_input_for_bootstrap() {
    let value = msvcrt_kbhit(&[]).expect("kbhit should be callable");
    assert_eq!(value.as_bool(), Some(false));
}
