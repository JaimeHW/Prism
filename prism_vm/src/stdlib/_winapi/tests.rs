use super::*;

#[cfg(windows)]
static WINAPI_TEST_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

#[test]
fn test_winapi_module_exposes_bootstrap_constants_and_functions() {
    let module = WinApiModule::new();

    assert!(module.get_attr("CREATE_NEW_CONSOLE").is_ok());
    assert!(module.get_attr("DUPLICATE_SAME_ACCESS").is_ok());
    assert!(module.get_attr("FILE_TYPE_CHAR").is_ok());
    assert!(module.get_attr("STD_OUTPUT_HANDLE").is_ok());
    assert!(module.get_attr("CloseHandle").is_ok());
    assert!(module.get_attr("CreatePipe").is_ok());
    assert!(module.get_attr("CreateProcess").is_ok());
    assert!(module.get_attr("DuplicateHandle").is_ok());
    assert!(module.get_attr("GetACP").is_ok());
    assert!(module.get_attr("GetCurrentProcess").is_ok());
    assert!(module.get_attr("GetExitCodeProcess").is_ok());
    assert!(module.get_attr("GetFileType").is_ok());
    assert!(module.get_attr("GetStdHandle").is_ok());
    assert!(module.get_attr("NeedCurrentDirectoryForExePath").is_ok());
    assert!(module.get_attr("TerminateProcess").is_ok());
    assert!(module.get_attr("WaitForSingleObject").is_ok());
}

#[test]
fn test_create_process_uses_cpython_312_nine_argument_signature() {
    let args = [
        Value::none(),
        Value::string(intern("cmd.exe /c exit 0")),
        Value::none(),
        Value::none(),
        Value::bool(false),
        Value::int(0).expect("flags should fit"),
        Value::none(),
        Value::none(),
    ];
    let err = winapi_create_process(&args).expect_err("CreateProcess should validate arity");
    assert!(matches!(err, BuiltinError::TypeError(_)));

    let args = [
        Value::none(),
        Value::none(),
        Value::none(),
        Value::none(),
        Value::bool(false),
        Value::int(0).expect("flags should fit"),
        Value::none(),
        Value::none(),
        Value::none(),
    ];
    let err = winapi_create_process(&args)
        .expect_err("nine-argument calls should reach command-line validation");
    assert!(matches!(err, BuiltinError::TypeError(message) if message.contains("command_line")));
}

#[test]
fn test_startupinfo_accepts_heap_instance_type_ids() {
    let registry = prism_runtime::object::shape::shape_registry();
    let mut object = ShapedObject::new(
        TypeId::from_raw(TypeId::FIRST_USER_TYPE),
        registry.empty_shape(),
    );
    object.set_property(
        intern("dwFlags"),
        Value::int(i64::from(STARTF_USESTDHANDLES_FLAG)).expect("flags should fit"),
        registry,
    );
    object.set_property(
        intern("hStdInput"),
        Value::int(11).expect("handle should fit"),
        registry,
    );
    object.set_property(
        intern("hStdOutput"),
        Value::int(12).expect("handle should fit"),
        registry,
    );
    object.set_property(
        intern("hStdError"),
        Value::int(13).expect("handle should fit"),
        registry,
    );
    let value = Value::object_ptr(Box::into_raw(Box::new(object)) as *const ());

    let config = startup_info_config(value).expect("heap startupinfo should parse");
    assert_eq!(config.dw_flags, STARTF_USESTDHANDLES_FLAG);
    assert_eq!(config.h_std_input, 11);
    assert_eq!(config.h_std_output, 12);
    assert_eq!(config.h_std_error, 13);
}

#[test]
#[cfg(windows)]
fn test_create_process_launches_process_and_returns_real_handles() {
    let comspec = std::env::var("ComSpec").unwrap_or_else(|_| "cmd.exe".to_string());
    let command_line = format!("\"{comspec}\" /c exit 7");
    let result = winapi_create_process(&[
        Value::string(intern(&comspec)),
        Value::string(intern(&command_line)),
        Value::none(),
        Value::none(),
        Value::bool(false),
        Value::int(0).expect("creation flags should fit"),
        Value::none(),
        Value::none(),
        Value::none(),
    ])
    .expect("CreateProcess should launch a simple command");

    let tuple = unsafe { &*(result.as_object_ptr().unwrap() as *const TupleObject) };
    assert_eq!(tuple.len(), 4);
    let process = tuple.as_slice()[0];
    let thread = tuple.as_slice()[1];
    winapi_wait_for_single_object(&[
        process,
        Value::int(0xFFFF_FFFF).expect("INFINITE should fit"),
    ])
    .expect("process wait should succeed");
    let exit_code =
        winapi_get_exit_code_process(&[process]).expect("exit code should be available");
    assert_eq!(exit_code.as_int(), Some(7));

    winapi_close_handle(&[thread]).expect("thread handle should close");
    winapi_close_handle(&[process]).expect("process handle should close");
}

#[test]
#[cfg(windows)]
fn test_create_process_closes_parent_pipe_sources_for_startup_duplicates() {
    let _guard = WINAPI_TEST_LOCK
        .lock()
        .expect("_winapi test lock should not be poisoned");
    let current_process =
        winapi_get_current_process(&[]).expect("current process handle should be available");
    let pipe = winapi_create_pipe(&[Value::none(), Value::int(0).expect("size should fit")])
        .expect("CreatePipe should allocate a Windows pipe");
    let pipe = unsafe { &*(pipe.as_object_ptr().unwrap() as *const TupleObject) };
    let source_read = pipe.as_slice()[0];
    let source_write = pipe.as_slice()[1];

    let duplicate_read = winapi_duplicate_handle(&[
        current_process,
        source_read,
        current_process,
        Value::int(0).expect("desired access should fit"),
        Value::bool(true),
        Value::int(i64::from(DUPLICATE_SAME_ACCESS)).expect("duplicate option should fit"),
    ])
    .expect("stdin pipe handle should duplicate");
    let duplicate_write = winapi_duplicate_handle(&[
        current_process,
        source_write,
        current_process,
        Value::int(0).expect("desired access should fit"),
        Value::bool(true),
        Value::int(i64::from(DUPLICATE_SAME_ACCESS)).expect("duplicate option should fit"),
    ])
    .expect("stdout pipe handle should duplicate");
    let source_read_handle =
        value_to_i64(source_read).expect("source read handle should be integer-backed");
    let source_write_handle =
        value_to_i64(source_write).expect("source write handle should be integer-backed");
    let duplicate_read_handle =
        value_to_i64(duplicate_read).expect("duplicate read handle should be integer-backed");
    let duplicate_write_handle =
        value_to_i64(duplicate_write).expect("duplicate write handle should be integer-backed");
    assert_eq!(
        pipe_duplicate_source_for_test(duplicate_read_handle),
        Some(source_read_handle)
    );
    assert_eq!(
        pipe_duplicate_source_for_test(duplicate_write_handle),
        Some(source_write_handle)
    );

    let startupinfo = startupinfo_value(
        STARTF_USESTDHANDLES_FLAG,
        duplicate_read,
        duplicate_write,
        duplicate_write,
    );
    let parsed_startup =
        startup_info_config(startupinfo).expect("startupinfo should parse before launch");
    assert_eq!(parsed_startup.dw_flags, STARTF_USESTDHANDLES_FLAG);
    assert_eq!(parsed_startup.h_std_input, duplicate_read_handle);
    assert_eq!(parsed_startup.h_std_output, duplicate_write_handle);
    assert_eq!(parsed_startup.h_std_error, duplicate_write_handle);
    let comspec = std::env::var("ComSpec").unwrap_or_else(|_| "cmd.exe".to_string());
    let command_line = format!("\"{comspec}\" /c exit 0");
    let process = winapi_create_process(&[
        Value::string(intern(&comspec)),
        Value::string(intern(&command_line)),
        Value::none(),
        Value::none(),
        Value::bool(true),
        Value::int(0).expect("creation flags should fit"),
        Value::none(),
        Value::none(),
        startupinfo,
    ])
    .expect("CreateProcess should launch with duplicated std handles");

    let process = unsafe { &*(process.as_object_ptr().unwrap() as *const TupleObject) };
    let process_handle = process.as_slice()[0];
    let thread_handle = process.as_slice()[1];
    winapi_wait_for_single_object(&[
        process_handle,
        Value::int(0xFFFF_FFFF).expect("INFINITE should fit"),
    ])
    .expect("process wait should succeed");

    let source_read_close = winapi_close_handle(&[source_read]);
    let source_write_close = winapi_close_handle(&[source_write]);
    winapi_close_handle(&[duplicate_read]).expect("duplicated read handle should close");
    winapi_close_handle(&[duplicate_write]).expect("duplicated write handle should close");
    winapi_close_handle(&[thread_handle]).expect("thread handle should close");
    winapi_close_handle(&[process_handle]).expect("process handle should close");

    assert!(
        matches!(source_read_close, Err(BuiltinError::OSError(_))),
        "CreateProcess should close the original pipe read source after duplicating it"
    );
    assert!(
        matches!(source_write_close, Err(BuiltinError::OSError(_))),
        "CreateProcess should close the original pipe write source after duplicating it"
    );
}

#[test]
#[cfg(windows)]
fn test_startup_cleanup_preserves_chained_duplicate_stdout_handles() {
    let _guard = WINAPI_TEST_LOCK
        .lock()
        .expect("_winapi test lock should not be poisoned");
    let current_process =
        winapi_get_current_process(&[]).expect("current process handle should be available");
    let pipe = winapi_create_pipe(&[Value::none(), Value::int(0).expect("size should fit")])
        .expect("CreatePipe should allocate a Windows pipe");
    let pipe = unsafe { &*(pipe.as_object_ptr().unwrap() as *const TupleObject) };
    let source_read = pipe.as_slice()[0];
    let source_write = pipe.as_slice()[1];

    let duplicate_stdout = winapi_duplicate_handle(&[
        current_process,
        source_write,
        current_process,
        Value::int(0).expect("desired access should fit"),
        Value::bool(true),
        Value::int(i64::from(DUPLICATE_SAME_ACCESS)).expect("duplicate option should fit"),
    ])
    .expect("stdout pipe handle should duplicate");
    let duplicate_stderr = winapi_duplicate_handle(&[
        current_process,
        duplicate_stdout,
        current_process,
        Value::int(0).expect("desired access should fit"),
        Value::bool(true),
        Value::int(i64::from(DUPLICATE_SAME_ACCESS)).expect("duplicate option should fit"),
    ])
    .expect("stderr=STDOUT pipe handle should duplicate from stdout duplicate");

    let source_write_handle =
        value_to_i64(source_write).expect("source write handle should be integer-backed");
    let duplicate_stdout_handle =
        value_to_i64(duplicate_stdout).expect("duplicate stdout handle should be integer-backed");
    let duplicate_stderr_handle =
        value_to_i64(duplicate_stderr).expect("duplicate stderr handle should be integer-backed");
    assert_eq!(
        pipe_duplicate_source_for_test(duplicate_stdout_handle),
        Some(source_write_handle)
    );
    assert_eq!(
        pipe_duplicate_source_for_test(duplicate_stderr_handle),
        Some(duplicate_stdout_handle)
    );

    close_parent_pipe_sources_for_startup(StartupInfoConfig {
        dw_flags: STARTF_USESTDHANDLES_FLAG,
        h_std_input: 0,
        h_std_output: duplicate_stdout_handle,
        h_std_error: duplicate_stderr_handle,
        w_show_window: 0,
    });

    let source_write_close = winapi_close_handle(&[source_write]);
    assert!(
        matches!(source_write_close, Err(BuiltinError::OSError(_))),
        "startup cleanup should close the original stdout pipe source"
    );
    winapi_close_handle(&[duplicate_stdout])
        .expect("stdout startup duplicate must remain owned by subprocess cleanup");
    winapi_close_handle(&[duplicate_stderr])
        .expect("stderr startup duplicate must remain owned by subprocess cleanup");
    winapi_close_handle(&[source_read]).expect("source read handle should close");
}

#[test]
#[cfg(windows)]
fn test_forget_tracked_pipe_handle_removes_consumed_source_tracking() {
    let _guard = WINAPI_TEST_LOCK
        .lock()
        .expect("_winapi test lock should not be poisoned");
    let current_process =
        winapi_get_current_process(&[]).expect("current process handle should be available");
    let pipe = winapi_create_pipe(&[Value::none(), Value::int(0).expect("size should fit")])
        .expect("CreatePipe should allocate a Windows pipe");
    let pipe = unsafe { &*(pipe.as_object_ptr().unwrap() as *const TupleObject) };
    let source_read = pipe.as_slice()[0];
    let source_write = pipe.as_slice()[1];
    let duplicate_read = winapi_duplicate_handle(&[
        current_process,
        source_read,
        current_process,
        Value::int(0).expect("desired access should fit"),
        Value::bool(true),
        Value::int(i64::from(DUPLICATE_SAME_ACCESS)).expect("duplicate option should fit"),
    ])
    .expect("read pipe handle should duplicate");

    let source_read_handle =
        value_to_i64(source_read).expect("source read handle should be integer-backed");
    let source_write_handle =
        value_to_i64(source_write).expect("source write handle should be integer-backed");
    let duplicate_read_handle =
        value_to_i64(duplicate_read).expect("duplicate read handle should be integer-backed");
    assert!(pipe_handle_is_tracked_for_test(source_read_handle));
    assert!(pipe_handle_is_tracked_for_test(source_write_handle));
    assert_eq!(
        pipe_duplicate_source_for_test(duplicate_read_handle),
        Some(source_read_handle)
    );

    forget_tracked_pipe_handle(source_read_handle);

    assert!(!pipe_handle_is_tracked_for_test(source_read_handle));
    assert!(pipe_handle_is_tracked_for_test(source_write_handle));
    assert!(pipe_handle_is_tracked_for_test(duplicate_read_handle));
    assert_eq!(pipe_duplicate_source_for_test(duplicate_read_handle), None);

    winapi_close_handle(&[source_read]).expect("source read handle should close");
    winapi_close_handle(&[source_write]).expect("source write handle should close");
    winapi_close_handle(&[duplicate_read]).expect("duplicate read handle should close");
}

#[cfg(windows)]
fn startupinfo_value(
    dw_flags: u32,
    h_std_input: Value,
    h_std_output: Value,
    h_std_error: Value,
) -> Value {
    let registry = prism_runtime::object::shape::shape_registry();
    let mut object = ShapedObject::new(
        TypeId::from_raw(TypeId::FIRST_USER_TYPE),
        registry.empty_shape(),
    );
    object.set_property(
        intern("dwFlags"),
        Value::int(i64::from(dw_flags)).expect("flags should fit"),
        registry,
    );
    object.set_property(intern("hStdInput"), h_std_input, registry);
    object.set_property(intern("hStdOutput"), h_std_output, registry);
    object.set_property(intern("hStdError"), h_std_error, registry);
    Value::object_ptr(Box::into_raw(Box::new(object)) as *const ())
}

#[test]
#[cfg(windows)]
fn test_create_pipe_returns_real_handles_that_can_be_closed() {
    let _guard = WINAPI_TEST_LOCK
        .lock()
        .expect("_winapi test lock should not be poisoned");
    let result = winapi_create_pipe(&[Value::none(), Value::int(0).expect("size should fit")])
        .expect("CreatePipe should allocate a Windows pipe");
    let tuple = unsafe { &*(result.as_object_ptr().unwrap() as *const TupleObject) };
    assert_eq!(tuple.len(), 2);

    for &handle in tuple.as_slice() {
        let close_result =
            winapi_close_handle(&[handle]).expect("pipe handles should be closeable");
        assert!(close_result.is_none());
    }
}

#[test]
#[cfg(not(windows))]
fn test_close_handle_accepts_integer_handles_as_no_op_on_non_windows() {
    let result = winapi_close_handle(&[Value::int(42).expect("tagged int should fit")])
        .expect("non-Windows fallback should accept integer handles");
    assert!(result.is_none());
}

#[test]
fn test_close_handle_rejects_non_integer_handles() {
    let err = winapi_close_handle(&[Value::string(intern("pipe"))])
        .expect_err("CloseHandle should validate argument types");
    assert!(matches!(err, BuiltinError::TypeError(_)));
}

#[test]
fn test_wait_and_exit_helpers_validate_integer_handles() {
    let err = winapi_wait_for_single_object(&[
        Value::string(intern("handle")),
        Value::int(0).expect("timeout should fit"),
    ])
    .expect_err("WaitForSingleObject should validate handle type");
    assert!(matches!(err, BuiltinError::TypeError(_)));

    let err = winapi_get_exit_code_process(&[Value::string(intern("handle"))])
        .expect_err("GetExitCodeProcess should validate handle type");
    assert!(matches!(err, BuiltinError::TypeError(_)));
}

#[test]
fn test_get_current_process_returns_integer_handle() {
    let handle = winapi_get_current_process(&[]).expect("GetCurrentProcess should be callable");
    assert!(handle.as_int().is_some() || prism_runtime::types::int::value_to_i64(handle).is_some());
}

#[test]
fn test_need_current_directory_for_exe_path_rejects_explicit_directories() {
    let value = winapi_need_current_directory_for_exe_path(&[Value::string(intern(
        r"C:\Windows\System32\cmd.exe",
    ))])
    .expect("explicit paths should be accepted");
    assert_eq!(value.as_bool(), Some(false));
}

#[test]
fn test_need_current_directory_for_exe_path_accepts_bare_executable_name() {
    let value = winapi_need_current_directory_for_exe_path(&[Value::string(intern("python"))])
        .expect("bare executable names should be accepted");
    assert_eq!(
        value.as_bool(),
        Some(std::env::var_os("NoDefaultCurrentDirectoryInExePath").is_none())
    );
}

#[test]
fn test_get_acp_returns_positive_code_page() {
    let value = winapi_get_acp(&[]).expect("GetACP should succeed");
    assert!(value.as_int().is_some_and(|codepage| codepage > 0));
}

#[test]
fn test_module_dir_contains_sorted_exports() {
    let module = WinApiModule::new();
    let dir = module.dir();
    assert!(dir.windows(2).all(|window| window[0] <= window[1]));
    assert!(dir.iter().any(|name| name.as_ref() == "GetACP"));
    assert!(dir.iter().any(|name| name.as_ref() == "WAIT_TIMEOUT"));
}
