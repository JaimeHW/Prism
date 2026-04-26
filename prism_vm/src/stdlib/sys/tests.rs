use super::*;
use crate::builtins::BuiltinFunctionObject;
use crate::builtins::create_exception;
use crate::stdlib::exceptions::ExceptionTypeId;
use prism_core::intern::interned_by_ptr;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::views::FrameViewObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::tuple::{TupleObject, value_as_tuple_ref};
use std::sync::{Arc, LazyLock, Mutex};

static HOOK_TEST_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

// =========================================================================
// SysModule Creation Tests
// =========================================================================

#[test]
fn test_sys_module_new() {
    let sys = SysModule::new();
    assert_eq!(sys.name(), "sys");
}

#[test]
fn test_sys_module_with_args() {
    let sys = SysModule::with_args(vec!["test.py".to_string(), "--flag".to_string()]);
    assert_eq!(sys.argv().len(), 2);
}

// =========================================================================
// Integer Attribute Tests
// =========================================================================

#[test]
fn test_hexversion_attribute() {
    let sys = SysModule::new();
    let hexversion = sys.get_attr("hexversion").unwrap();
    let v = hexversion.as_int().unwrap();
    // Python 3.12.0 = 0x030C00F0
    assert!(v > 0);
    assert_eq!(v, HEXVERSION as i64);
}

#[test]
fn test_maxsize_attribute() {
    let sys = SysModule::new();
    let maxsize = sys.get_attr("maxsize").unwrap();
    let m = maxsize.as_int().unwrap();
    // maxsize should be positive and equal to SMALL_INT_MAX
    assert!(m > 0);
    assert_eq!(m, MAX_SIZE);
}

#[test]
fn test_maxunicode_attribute() {
    let sys = SysModule::new();
    let maxunicode = sys.get_attr("maxunicode").unwrap();
    let m = maxunicode.as_int().unwrap();
    assert_eq!(m, 0x10FFFF);
}

#[test]
fn test_api_version_attribute() {
    let sys = SysModule::new();
    let api = sys.get_attr("api_version").unwrap();
    let a = api.as_int().unwrap();
    assert!(a > 0);
}

// =========================================================================
// Recursion Limit Tests
// =========================================================================

#[test]
fn test_getrecursionlimit() {
    let sys = SysModule::new();
    let limit = sys.getrecursionlimit();
    assert_eq!(limit, DEFAULT_RECURSION_LIMIT);
}

#[test]
fn test_setrecursionlimit() {
    let mut sys = SysModule::new();
    sys.setrecursionlimit(2000).unwrap();
    assert_eq!(sys.getrecursionlimit(), 2000);
}

#[test]
fn test_setrecursionlimit_too_low() {
    let mut sys = SysModule::new();
    let result = sys.setrecursionlimit(1);
    assert!(result.is_err());
}

#[test]
fn test_recursion_limit_attribute() {
    let sys = SysModule::new();
    let limit = sys.get_attr("recursion_limit").unwrap();
    assert_eq!(limit.as_int().unwrap(), DEFAULT_RECURSION_LIMIT as i64);
}

// =========================================================================
// Attribute Tests
// =========================================================================

#[test]
fn test_stdin_attribute() {
    let sys = SysModule::new();
    let stdin = sys.get_attr("stdin").unwrap();
    let ptr = stdin
        .as_object_ptr()
        .expect("sys.stdin should be a stream object");
    let stream = unsafe { &*(ptr as *const ShapedObject) };
    assert_eq!(
        stream
            .get_property("closed")
            .and_then(|value| value.as_bool()),
        Some(false)
    );
    assert!(stream.get_property("readline").is_some());
}

#[test]
fn test_stdout_attribute() {
    let sys = SysModule::new();
    let stdout = sys.get_attr("stdout").unwrap();
    let ptr = stdout
        .as_object_ptr()
        .expect("sys.stdout should be a stream object");
    let stream = unsafe { &*(ptr as *const ShapedObject) };
    assert_eq!(
        stream
            .get_property("closed")
            .and_then(|value| value.as_bool()),
        Some(false)
    );
    assert!(stream.get_property("write").is_some());
    assert!(stream.get_property("flush").is_some());
}

#[test]
fn test_stderr_attribute() {
    let sys = SysModule::new();
    let stderr = sys.get_attr("stderr").unwrap();
    let ptr = stderr
        .as_object_ptr()
        .expect("sys.stderr should be a stream object");
    let stream = unsafe { &*(ptr as *const ShapedObject) };
    assert_eq!(
        stream
            .get_property("closed")
            .and_then(|value| value.as_bool()),
        Some(false)
    );
    assert!(stream.get_property("write").is_some());
    assert!(stream.get_property("flush").is_some());
}

#[test]
fn test_version_attribute() {
    let sys = SysModule::new();
    let version = sys.get_attr("version").unwrap();
    let ptr = version
        .as_string_object_ptr()
        .expect("sys.version should be an interned string") as *const u8;
    assert!(
        interned_by_ptr(ptr)
            .expect("sys.version should resolve")
            .as_ref()
            .contains("3.12")
    );
}

#[test]
fn test_platform_attribute() {
    let sys = SysModule::new();
    let platform = sys.get_attr("platform").unwrap();
    let ptr = platform
        .as_string_object_ptr()
        .expect("sys.platform should be an interned string") as *const u8;
    let platform_name = interned_by_ptr(ptr)
        .expect("sys.platform should resolve")
        .as_ref()
        .to_string();
    assert!(["win32", "linux", "darwin", "freebsd", "unknown"].contains(&platform_name.as_str()));
}

#[test]
fn test_prefix_family_attributes_are_present_and_consistent() {
    let sys = SysModule::new();

    for attr in ["prefix", "exec_prefix", "base_prefix", "base_exec_prefix"] {
        let value = sys.get_attr(attr).expect("prefix attribute should exist");
        let ptr = value
            .as_string_object_ptr()
            .expect("prefix attribute should be an interned string") as *const u8;
        let resolved = interned_by_ptr(ptr)
            .expect("prefix attribute should resolve")
            .as_ref()
            .to_string();
        assert!(
            !resolved.is_empty(),
            "{attr} should expose a non-empty installation path"
        );
    }

    let prefix = sys.get_attr("prefix").unwrap();
    let base_prefix = sys.get_attr("base_prefix").unwrap();
    assert_eq!(prefix, base_prefix);

    let exec_prefix = sys.get_attr("exec_prefix").unwrap();
    let base_exec_prefix = sys.get_attr("base_exec_prefix").unwrap();
    assert_eq!(exec_prefix, base_exec_prefix);
}

#[test]
fn test_exit_is_callable() {
    let sys = SysModule::new();
    let exit = sys.get_attr("exit").unwrap();
    assert!(exit.as_object_ptr().is_some());
    assert!(value_supports_call_protocol(exit));
}

#[test]
fn test_intern_callable_returns_interned_string() {
    let sys = SysModule::new();
    let value = sys.get_attr("intern").expect("callable should exist");
    let ptr = value
        .as_object_ptr()
        .expect("sys.intern should be builtin function");
    let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
    let result = builtin
        .call(&[Value::string(intern("CacheInfo"))])
        .expect("call should succeed");
    let result_ptr = result
        .as_string_object_ptr()
        .expect("sys.intern result should be string") as *const u8;
    assert_eq!(
        interned_by_ptr(result_ptr)
            .expect("sys.intern result should resolve")
            .as_ref(),
        "CacheInfo"
    );
}

#[test]
fn test_builtin_module_names_attribute_is_tuple() {
    let sys = SysModule::new();
    let value = sys
        .get_attr("builtin_module_names")
        .expect("builtin_module_names should exist");
    let ptr = value
        .as_object_ptr()
        .expect("builtin_module_names should be a tuple object");
    let tuple = unsafe { &*(ptr as *const TupleObject) };

    assert!(
        tuple
            .iter()
            .any(|value| value == &Value::string(intern("sys")))
    );
    assert!(
        tuple
            .iter()
            .any(|value| value == &Value::string(intern("builtins")))
    );
    assert!(
        tuple
            .iter()
            .any(|value| value == &Value::string(intern("_imp")))
    );
    assert!(
        tuple
            .iter()
            .any(|value| value == &Value::string(intern("_weakref")))
    );
    assert!(
        tuple
            .iter()
            .any(|value| value == &Value::string(intern("_warnings")))
    );
    if cfg!(windows) {
        assert!(
            tuple
                .iter()
                .any(|value| value == &Value::string(intern("nt")))
        );
    }
}

#[test]
fn test_import_bootstrap_collections_are_present_and_mutable() {
    let sys = SysModule::new();

    for list_attr in ["warnoptions", "meta_path", "path_hooks"] {
        let value = sys
            .get_attr(list_attr)
            .expect("list attribute should exist");
        let ptr = value
            .as_object_ptr()
            .expect("bootstrap list attribute should be heap allocated");
        let list = unsafe { &*(ptr as *const ListObject) };
        assert!(list.is_empty(), "{list_attr} should start empty");
    }

    for dict_attr in ["path_importer_cache", "modules"] {
        let value = sys
            .get_attr(dict_attr)
            .expect("dict attribute should exist");
        let ptr = value
            .as_object_ptr()
            .expect("bootstrap dict attribute should be heap allocated");
        let dict = unsafe { &*(ptr as *const DictObject) };
        assert!(dict.is_empty(), "{dict_attr} should start empty");
    }
}

#[test]
fn test_flags_attribute_exposes_site_compatible_fields() {
    let sys = SysModule::new();
    let value = sys.get_attr("flags").expect("sys.flags should exist");
    let ptr = value
        .as_object_ptr()
        .expect("sys.flags should be a heap object");
    let flags = unsafe { &*(ptr as *const ShapedObject) };

    for (name, expected) in [
        ("verbose", Value::int(0).unwrap()),
        ("no_user_site", Value::int(0).unwrap()),
        ("isolated", Value::int(0).unwrap()),
        ("no_site", Value::int(0).unwrap()),
        ("hash_randomization", Value::int(1).unwrap()),
        ("safe_path", Value::bool(false)),
    ] {
        assert_eq!(
            flags.get_property_interned(&intern(name)),
            Some(expected),
            "{name} should be present on sys.flags"
        );
    }
}

#[test]
fn test_winver_attribute_matches_cpython_contract() {
    let sys = SysModule::new();
    let result = sys.get_attr("winver");

    if cfg!(windows) {
        let value = result.expect("sys.winver should exist on Windows");
        let ptr = value
            .as_string_object_ptr()
            .expect("sys.winver should be an interned string") as *const u8;
        assert_eq!(
            interned_by_ptr(ptr)
                .expect("sys.winver should resolve")
                .as_ref(),
            WINVER
        );
    } else {
        assert!(
            matches!(result, Err(ModuleError::AttributeError(_))),
            "sys.winver should be absent off Windows"
        );
    }
}

#[test]
fn test_dir_contains_flags() {
    let sys = SysModule::new();
    let names = sys.dir();
    assert!(names.iter().any(|name| name.as_ref() == "flags"));
}

#[test]
fn test_dir_contains_winver_only_on_windows() {
    let sys = SysModule::new();
    let names = sys.dir();
    let has_winver = names.iter().any(|name| name.as_ref() == "winver");
    assert_eq!(has_winver, cfg!(windows));
}

#[test]
fn test_getfilesystemencoding_callable_returns_utf8() {
    let sys = SysModule::new();
    let value = sys
        .get_attr("getfilesystemencoding")
        .expect("callable should exist");
    let ptr = value
        .as_object_ptr()
        .expect("getfilesystemencoding should be builtin function");
    let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
    let result = builtin.call(&[]).expect("call should succeed");
    let result_ptr = result
        .as_string_object_ptr()
        .expect("filesystem encoding should be string") as *const u8;
    assert_eq!(
        interned_by_ptr(result_ptr)
            .expect("filesystem encoding should resolve")
            .as_ref(),
        "utf-8"
    );
}

#[test]
fn test_getfilesystemencodeerrors_callable_returns_platform_default() {
    let sys = SysModule::new();
    let value = sys
        .get_attr("getfilesystemencodeerrors")
        .expect("callable should exist");
    let ptr = value
        .as_object_ptr()
        .expect("getfilesystemencodeerrors should be builtin function");
    let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
    let result = builtin.call(&[]).expect("call should succeed");
    let result_ptr = result
        .as_string_object_ptr()
        .expect("filesystem encode errors should be string") as *const u8;
    let expected = if cfg!(windows) {
        "surrogatepass"
    } else {
        "surrogateescape"
    };
    assert_eq!(
        interned_by_ptr(result_ptr)
            .expect("filesystem encode errors should resolve")
            .as_ref(),
        expected
    );
}

#[test]
fn test_version_info_attribute_exposes_tuple_storage() {
    let sys = SysModule::new();
    let value = sys
        .get_attr("version_info")
        .expect("version_info should exist");
    let tuple = value_as_tuple_ref(value).expect("version_info should expose tuple storage");
    assert_eq!(tuple.len(), 5);
    assert_eq!(tuple.get(0).and_then(|value| value.as_int()), Some(3));
}

#[test]
fn test_git_attribute_matches_cpython_metadata_tuple_shape() {
    let sys = SysModule::new();
    let value = sys.get_attr("_git").expect("sys._git should exist");
    let ptr = value
        .as_object_ptr()
        .expect("sys._git should be a tuple object");
    let tuple = unsafe { &*(ptr as *const TupleObject) };

    assert_eq!(tuple.len(), 3);
    assert_eq!(tuple.get(0), Some(Value::string(intern("Prism"))));
    assert_eq!(tuple.get(1), Some(Value::string(intern(""))));
    assert_eq!(tuple.get(2), Some(Value::string(intern(""))));
}

#[test]
fn test_implementation_attribute_exposes_namespace_fields() {
    let sys = SysModule::new();
    let value = sys
        .get_attr("implementation")
        .expect("implementation should exist");
    let ptr = value
        .as_object_ptr()
        .expect("implementation should be heap allocated");
    let object = unsafe { &*(ptr as *const ShapedObject) };
    assert_eq!(
        object.get_property("name"),
        Some(Value::string(intern("prism")))
    );
    assert!(object.get_property("version").is_some());
}

#[test]
fn test_hash_info_attribute_exposes_runtime_metadata() {
    let sys = SysModule::new();
    let value = sys.get_attr("hash_info").expect("hash_info should exist");
    let ptr = value
        .as_object_ptr()
        .expect("hash_info should be heap allocated");
    let object = unsafe { &*(ptr as *const ShapedObject) };

    assert_eq!(
        object
            .get_property("width")
            .and_then(|value| value.as_int()),
        Some(i64::from(usize::BITS))
    );
    assert!(object.get_property("algorithm").is_some());
}

#[test]
fn test_platlibdir_attribute_matches_runtime_constant() {
    let sys = SysModule::new();
    let value = sys.get_attr("platlibdir").expect("platlibdir should exist");
    let ptr = value
        .as_string_object_ptr()
        .expect("platlibdir should be an interned string") as *const u8;
    assert_eq!(
        interned_by_ptr(ptr)
            .expect("platlibdir should resolve")
            .as_ref(),
        PLATLIBDIR
    );
}

#[test]
fn test_import_cache_configuration_matches_cpython_defaults() {
    let sys = SysModule::new();

    assert!(
        sys.get_attr("pycache_prefix")
            .expect("pycache_prefix should exist")
            .is_none(),
        "sys.pycache_prefix should default to None"
    );
    assert_eq!(
        sys.get_attr("dont_write_bytecode")
            .expect("dont_write_bytecode should exist"),
        Value::bool(false),
        "sys.dont_write_bytecode should default to False"
    );
}

#[test]
fn test_windows_only_vpath_and_getwindowsversion_surface() {
    let sys = SysModule::new();
    let vpath = sys.get_attr("_vpath");
    let getter = sys.get_attr("getwindowsversion");

    if cfg!(windows) {
        let vpath = vpath.expect("_vpath should exist on Windows");
        let ptr = vpath
            .as_string_object_ptr()
            .expect("_vpath should be an interned string") as *const u8;
        assert_eq!(
            interned_by_ptr(ptr)
                .expect("_vpath should resolve")
                .as_ref(),
            VPATH
        );

        let getter = getter.expect("getwindowsversion should exist on Windows");
        let ptr = getter
            .as_object_ptr()
            .expect("getwindowsversion should be callable");
        let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
        let result = builtin.call(&[]).expect("getwindowsversion should succeed");
        let result_ptr = result
            .as_object_ptr()
            .expect("getwindowsversion should return a heap object");
        let info = unsafe { &*(result_ptr as *const ShapedObject) };
        assert_eq!(
            info.get_property("platform")
                .and_then(|value| value.as_int()),
            Some(2)
        );
    } else {
        assert!(matches!(vpath, Err(ModuleError::AttributeError(_))));
        assert!(matches!(getter, Err(ModuleError::AttributeError(_))));
    }
}

#[test]
fn test_argv_attribute_is_list() {
    let sys = SysModule::new();
    let argv = sys.get_attr("argv").unwrap();
    let ptr = argv
        .as_object_ptr()
        .expect("sys.argv should be a list object");
    let list = unsafe { &*(ptr as *const ListObject) };
    assert!(!list.is_empty(), "sys.argv should include at least argv[0]");
}

#[test]
fn test_with_args_populates_argv_values() {
    let sys = SysModule::with_args(vec![
        "script.py".to_string(),
        "--flag".to_string(),
        "value".to_string(),
    ]);
    let argv = sys.get_attr("argv").unwrap();
    let ptr = argv
        .as_object_ptr()
        .expect("sys.argv should be a list object");
    let list = unsafe { &*(ptr as *const ListObject) };
    assert_eq!(list.len(), 3);

    let first = list.get(0).expect("argv[0] should exist");
    let second = list.get(1).expect("argv[1] should exist");
    let third = list.get(2).expect("argv[2] should exist");

    let first_ptr = first
        .as_string_object_ptr()
        .expect("argv[0] should be interned string") as *const u8;
    let second_ptr = second
        .as_string_object_ptr()
        .expect("argv[1] should be interned string") as *const u8;
    let third_ptr = third
        .as_string_object_ptr()
        .expect("argv[2] should be interned string") as *const u8;

    assert_eq!(
        interned_by_ptr(first_ptr)
            .expect("argv[0] should resolve")
            .as_ref(),
        "script.py"
    );
    assert_eq!(
        interned_by_ptr(second_ptr)
            .expect("argv[1] should resolve")
            .as_ref(),
        "--flag"
    );
    assert_eq!(
        interned_by_ptr(third_ptr)
            .expect("argv[2] should resolve")
            .as_ref(),
        "value"
    );
}

#[test]
fn test_dir_contains_argv_and_path() {
    let sys = SysModule::new();
    let names = sys.dir();
    assert!(names.iter().any(|name| name.as_ref() == "argv"));
    assert!(names.iter().any(|name| name.as_ref() == "path"));
    assert!(names.iter().any(|name| name.as_ref() == "_getframe"));
    assert!(names.iter().any(|name| name.as_ref() == "gettrace"));
    assert!(names.iter().any(|name| name.as_ref() == "base_prefix"));
    assert!(names.iter().any(|name| name.as_ref() == "base_exec_prefix"));
    assert!(names.iter().any(|name| name.as_ref() == "pycache_prefix"));
    assert!(
        names
            .iter()
            .any(|name| name.as_ref() == "dont_write_bytecode")
    );
    assert!(names.iter().any(|name| name.as_ref() == "_git"));
    assert!(
        names
            .iter()
            .any(|name| name.as_ref() == "builtin_module_names")
    );
    assert!(
        names
            .iter()
            .any(|name| name.as_ref() == "getfilesystemencoding")
    );
    assert!(names.iter().any(|name| name.as_ref() == "__displayhook__"));
    assert!(names.iter().any(|name| name.as_ref() == "__excepthook__"));
}

#[test]
fn test_sys_hooks_expose_default_callable_objects() {
    let sys = SysModule::new();
    let displayhook = sys
        .get_attr("displayhook")
        .expect("sys.displayhook should exist");
    let default_displayhook = sys
        .get_attr("__displayhook__")
        .expect("sys.__displayhook__ should exist");
    let excepthook = sys
        .get_attr("excepthook")
        .expect("sys.excepthook should exist");
    let default_excepthook = sys
        .get_attr("__excepthook__")
        .expect("sys.__excepthook__ should exist");

    for hook in [
        displayhook,
        default_displayhook,
        excepthook,
        default_excepthook,
    ] {
        let ptr = hook
            .as_object_ptr()
            .expect("sys hook should be a builtin function");
        let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
        assert!(
            builtin.name().starts_with("sys."),
            "hook should be backed by a sys builtin"
        );
    }

    assert_eq!(displayhook, default_displayhook);
    assert_eq!(excepthook, default_excepthook);
}

#[test]
fn test_sys_excepthook_accepts_standard_signature() {
    let sys = SysModule::new();
    let hook = sys
        .get_attr("excepthook")
        .expect("sys.excepthook should exist");
    let ptr = hook
        .as_object_ptr()
        .expect("sys.excepthook should be a builtin function");
    let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };

    assert!(
        builtin
            .call(&[Value::none(), Value::none(), Value::none()])
            .expect("sys.excepthook should accept exc triplets")
            .is_none()
    );
    assert!(matches!(builtin.call(&[]), Err(BuiltinError::TypeError(_))));
}

#[test]
fn test_sys_exc_info_returns_empty_triple_without_active_exception() {
    let sys = SysModule::new();
    let hook = sys.get_attr("exc_info").expect("sys.exc_info should exist");
    let ptr = hook
        .as_object_ptr()
        .expect("sys.exc_info should be a builtin function");
    let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
    let mut vm = crate::VirtualMachine::new();

    let value = builtin
        .call_with_vm(&mut vm, &[])
        .expect("sys.exc_info should accept zero arguments");
    let tuple_ptr = value
        .as_object_ptr()
        .expect("sys.exc_info should return a tuple");
    let tuple = unsafe { &*(tuple_ptr as *const TupleObject) };

    assert_eq!(tuple.len(), 3);
    assert!(tuple.iter().all(Value::is_none));
}

#[test]
fn test_sys_exc_info_ignores_stale_normal_exception_value() {
    let sys = SysModule::new();
    let hook = sys.get_attr("exc_info").expect("sys.exc_info should exist");
    let ptr = hook
        .as_object_ptr()
        .expect("sys.exc_info should be a builtin function");
    let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
    let mut vm = crate::VirtualMachine::new();
    let stale_exception = create_exception(ExceptionTypeId::IndexError, Some(Arc::from("stale")));
    vm.set_active_exception_with_type(stale_exception, ExceptionTypeId::IndexError.as_u8() as u16);
    vm.clear_exception_state();

    let value = builtin
        .call_with_vm(&mut vm, &[])
        .expect("sys.exc_info should accept zero arguments");
    let tuple_ptr = value
        .as_object_ptr()
        .expect("sys.exc_info should return a tuple");
    let tuple = unsafe { &*(tuple_ptr as *const TupleObject) };

    assert_eq!(tuple.len(), 3);
    assert!(tuple.iter().all(Value::is_none));
}

#[test]
fn test_sys_exception_returns_none_without_active_exception() {
    let sys = SysModule::new();
    let hook = sys
        .get_attr("exception")
        .expect("sys.exception should exist");
    let ptr = hook
        .as_object_ptr()
        .expect("sys.exception should be a builtin function");
    let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
    let mut vm = crate::VirtualMachine::new();

    let value = builtin
        .call_with_vm(&mut vm, &[])
        .expect("sys.exception should accept zero arguments");

    assert!(value.is_none());
}

#[test]
fn test_sys_exception_returns_active_exception_value() {
    let sys = SysModule::new();
    let hook = sys
        .get_attr("exception")
        .expect("sys.exception should exist");
    let ptr = hook
        .as_object_ptr()
        .expect("sys.exception should be a builtin function");
    let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
    let mut vm = crate::VirtualMachine::new();
    let active_exception = create_exception(ExceptionTypeId::TypeError, Some(Arc::from("boom")));
    vm.set_active_exception_with_type(active_exception, ExceptionTypeId::TypeError.as_u8() as u16);

    let value = builtin
        .call_with_vm(&mut vm, &[])
        .expect("sys.exception should return the active exception");

    assert_eq!(value, active_exception);
}

#[test]
fn test_sys_getframe_returns_current_frame_and_back_chain() {
    let sys = SysModule::new();
    let hook = sys
        .get_attr("_getframe")
        .expect("sys._getframe should exist");
    let ptr = hook
        .as_object_ptr()
        .expect("sys._getframe should be a builtin function");
    let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
    let mut vm = crate::VirtualMachine::new();

    let outer = Arc::new(prism_code::CodeObject::new("outer", "<test>"));
    let inner = Arc::new(prism_code::CodeObject::new("inner", "<test>"));
    vm.push_frame(Arc::clone(&outer), 0).unwrap();
    vm.current_frame_mut().ip = 2;
    vm.push_frame(Arc::clone(&inner), 0).unwrap();
    vm.current_frame_mut().ip = 5;

    let value = builtin
        .call_with_vm(&mut vm, &[])
        .expect("sys._getframe should return the current frame");
    let frame_ptr = value
        .as_object_ptr()
        .expect("sys._getframe should return a frame object");
    let frame = unsafe { &*(frame_ptr as *const FrameViewObject) };
    assert_eq!(
        frame
            .code()
            .expect("frame should keep its code object")
            .name
            .as_ref(),
        "inner"
    );
    assert_eq!(frame.lasti(), 5);

    let back = frame.back().expect("current frame should expose a caller");
    let back_ptr = back
        .as_object_ptr()
        .expect("caller frame should be a frame object");
    let caller = unsafe { &*(back_ptr as *const FrameViewObject) };
    assert_eq!(
        caller
            .code()
            .expect("caller frame should keep its code object")
            .name
            .as_ref(),
        "outer"
    );
    assert_eq!(caller.lasti(), 2);
    assert_eq!(caller.back(), None);
}

#[test]
fn test_sys_getframe_honors_depth_and_validates_arguments() {
    let sys = SysModule::new();
    let hook = sys
        .get_attr("_getframe")
        .expect("sys._getframe should exist");
    let ptr = hook
        .as_object_ptr()
        .expect("sys._getframe should be a builtin function");
    let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
    let mut vm = crate::VirtualMachine::new();

    vm.push_frame(Arc::new(prism_code::CodeObject::new("outer", "<test>")), 0)
        .unwrap();
    vm.push_frame(Arc::new(prism_code::CodeObject::new("inner", "<test>")), 0)
        .unwrap();

    let caller = builtin
        .call_with_vm(&mut vm, &[Value::int(1).unwrap()])
        .expect("depth=1 should return the caller frame");
    let caller_ptr = caller
        .as_object_ptr()
        .expect("depth=1 should still return a frame object");
    let caller = unsafe { &*(caller_ptr as *const FrameViewObject) };
    assert_eq!(
        caller
            .code()
            .expect("caller frame should keep its code object")
            .name
            .as_ref(),
        "outer"
    );

    let too_deep = builtin
        .call_with_vm(&mut vm, &[Value::int(2).unwrap()])
        .expect_err("depth beyond the active stack should fail");
    assert!(
        matches!(too_deep, BuiltinError::ValueError(ref message) if message == "call stack is not deep enough")
    );

    let negative = builtin
        .call_with_vm(&mut vm, &[Value::int(-1).unwrap()])
        .expect_err("negative depth should fail");
    assert!(
        matches!(negative, BuiltinError::ValueError(ref message) if message == "call stack is not deep enough")
    );

    let invalid = builtin
        .call_with_vm(&mut vm, &[Value::float(1.5)])
        .expect_err("non-integral depth should fail");
    assert!(
        matches!(invalid, BuiltinError::TypeError(ref message) if message.contains("cannot be interpreted as an integer"))
    );
}

#[test]
fn test_sys_gettrace_exists_and_defaults_to_none() {
    let _guard = HOOK_TEST_LOCK.lock().expect("hook test lock poisoned");
    *CURRENT_TRACE_FUNCTION
        .lock()
        .expect("sys trace hook lock poisoned") = Value::none();
    let sys = SysModule::new();
    let hook = sys.get_attr("gettrace").expect("sys.gettrace should exist");
    let ptr = hook
        .as_object_ptr()
        .expect("sys.gettrace should be a builtin function");
    let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };

    assert!(
        builtin
            .call(&[])
            .expect("gettrace() should succeed")
            .is_none()
    );

    let err = builtin
        .call(&[Value::int(1).unwrap()])
        .expect_err("gettrace() should reject positional arguments");
    assert!(
        matches!(err, BuiltinError::TypeError(ref message) if message == "gettrace() takes no arguments (1 given)")
    );
}

#[test]
fn test_sys_settrace_updates_visible_trace_hook_and_validates_input() {
    let _guard = HOOK_TEST_LOCK.lock().expect("hook test lock poisoned");
    *CURRENT_TRACE_FUNCTION
        .lock()
        .expect("sys trace hook lock poisoned") = Value::none();
    let sys = SysModule::new();
    let gettrace = unsafe {
        &*(sys
            .get_attr("gettrace")
            .expect("sys.gettrace should exist")
            .as_object_ptr()
            .expect("sys.gettrace should be callable") as *const BuiltinFunctionObject)
    };
    let settrace = unsafe {
        &*(sys
            .get_attr("settrace")
            .expect("sys.settrace should exist")
            .as_object_ptr()
            .expect("sys.settrace should be callable") as *const BuiltinFunctionObject)
    };
    let intern = sys.get_attr("intern").expect("sys.intern should exist");

    assert!(
        settrace
            .call(&[intern])
            .expect("settrace should accept callables")
            .is_none()
    );
    assert_eq!(
        gettrace
            .call(&[])
            .expect("gettrace should return current hook"),
        intern
    );

    assert!(
        settrace
            .call(&[Value::none()])
            .expect("settrace should accept None")
            .is_none()
    );
    assert!(
        gettrace
            .call(&[])
            .expect("gettrace should return None after reset")
            .is_none()
    );

    let err = settrace
        .call(&[Value::int(1).unwrap()])
        .expect_err("settrace should reject non-callables");
    assert!(
        matches!(err, BuiltinError::TypeError(ref message) if message == "settrace() argument must be callable or None")
    );
}

#[test]
fn test_sys_profile_hooks_match_trace_hook_contract() {
    let _guard = HOOK_TEST_LOCK.lock().expect("hook test lock poisoned");
    *CURRENT_PROFILE_FUNCTION
        .lock()
        .expect("sys profile hook lock poisoned") = Value::none();
    let sys = SysModule::new();
    let getprofile = unsafe {
        &*(sys
            .get_attr("getprofile")
            .expect("sys.getprofile should exist")
            .as_object_ptr()
            .expect("sys.getprofile should be callable") as *const BuiltinFunctionObject)
    };
    let setprofile = unsafe {
        &*(sys
            .get_attr("setprofile")
            .expect("sys.setprofile should exist")
            .as_object_ptr()
            .expect("sys.setprofile should be callable") as *const BuiltinFunctionObject)
    };
    let intern = sys.get_attr("intern").expect("sys.intern should exist");

    assert!(getprofile.call(&[]).unwrap().is_none());
    assert!(setprofile.call(&[intern]).unwrap().is_none());
    assert_eq!(getprofile.call(&[]).unwrap(), intern);
    assert!(setprofile.call(&[Value::none()]).unwrap().is_none());
    assert!(getprofile.call(&[]).unwrap().is_none());

    let err = setprofile
        .call(&[Value::int(1).unwrap()])
        .expect_err("setprofile should reject non-callables");
    assert!(
        matches!(err, BuiltinError::TypeError(ref message) if message == "setprofile() argument must be callable or None")
    );
}

#[test]
fn test_sys_switch_interval_hooks_round_trip_and_validate_input() {
    let _guard = HOOK_TEST_LOCK.lock().expect("hook test lock poisoned");
    *CURRENT_SWITCH_INTERVAL
        .lock()
        .expect("sys switch interval lock poisoned") = SwitchInterval::new();
    let sys = SysModule::new();
    let getinterval = unsafe {
        &*(sys
            .get_attr("getswitchinterval")
            .expect("sys.getswitchinterval should exist")
            .as_object_ptr()
            .expect("sys.getswitchinterval should be callable")
            as *const BuiltinFunctionObject)
    };
    let setinterval = unsafe {
        &*(sys
            .get_attr("setswitchinterval")
            .expect("sys.setswitchinterval should exist")
            .as_object_ptr()
            .expect("sys.setswitchinterval should be callable")
            as *const BuiltinFunctionObject)
    };

    assert_eq!(
        getinterval
            .call(&[])
            .expect("default interval should be readable")
            .as_float(),
        Some(DEFAULT_SWITCH_INTERVAL)
    );
    assert!(
        setinterval
            .call(&[Value::float(0.01)])
            .expect("positive intervals should be accepted")
            .is_none()
    );
    assert_eq!(
        getinterval
            .call(&[])
            .expect("updated interval should be readable")
            .as_float(),
        Some(0.01)
    );

    assert!(matches!(
        setinterval.call(&[Value::float(0.0)]),
        Err(BuiltinError::ValueError(_))
    ));
    assert!(matches!(
        setinterval.call(&[Value::string(intern("slow"))]),
        Err(BuiltinError::TypeError(_))
    ));

    *CURRENT_SWITCH_INTERVAL
        .lock()
        .expect("sys switch interval lock poisoned") = SwitchInterval::new();
}

#[test]
fn test_sys_getrefcount_is_callable_and_positive() {
    let sys = SysModule::new();
    let getrefcount = unsafe {
        &*(sys
            .get_attr("getrefcount")
            .expect("sys.getrefcount should exist")
            .as_object_ptr()
            .expect("sys.getrefcount should be callable")
            as *const BuiltinFunctionObject)
    };

    assert_eq!(
        getrefcount
            .call(&[Value::none()])
            .expect("getrefcount should accept one argument")
            .as_int(),
        Some(2)
    );

    let err = getrefcount
        .call(&[])
        .expect_err("getrefcount should reject missing arguments");
    assert!(matches!(
        err,
        BuiltinError::TypeError(ref message)
            if message == "getrefcount() takes exactly one argument (0 given)"
    ));
}

#[test]
fn test_sys_all_threads_hook_entry_points_update_shared_hooks() {
    let _guard = HOOK_TEST_LOCK.lock().expect("hook test lock poisoned");
    *CURRENT_TRACE_FUNCTION
        .lock()
        .expect("sys trace hook lock poisoned") = Value::none();
    *CURRENT_PROFILE_FUNCTION
        .lock()
        .expect("sys profile hook lock poisoned") = Value::none();
    let sys = SysModule::new();
    let intern = sys.get_attr("intern").expect("sys.intern should exist");
    let settrace_all = unsafe {
        &*(sys
            .get_attr("_settraceallthreads")
            .expect("sys._settraceallthreads should exist")
            .as_object_ptr()
            .expect("sys._settraceallthreads should be callable")
            as *const BuiltinFunctionObject)
    };
    let setprofile_all = unsafe {
        &*(sys
            .get_attr("_setprofileallthreads")
            .expect("sys._setprofileallthreads should exist")
            .as_object_ptr()
            .expect("sys._setprofileallthreads should be callable")
            as *const BuiltinFunctionObject)
    };

    assert!(settrace_all.call(&[intern]).unwrap().is_none());
    assert_eq!(
        *CURRENT_TRACE_FUNCTION
            .lock()
            .expect("sys trace hook lock poisoned"),
        intern
    );

    assert!(setprofile_all.call(&[intern]).unwrap().is_none());
    assert_eq!(
        *CURRENT_PROFILE_FUNCTION
            .lock()
            .expect("sys profile hook lock poisoned"),
        intern
    );
}

#[test]
fn test_path_attribute_includes_current_directory_entry() {
    let sys = SysModule::new();
    let path = sys.get_attr("path").expect("sys.path should exist");
    let ptr = path
        .as_object_ptr()
        .expect("sys.path should be represented as list object");
    let list = unsafe { &*(ptr as *const ListObject) };
    assert!(
        !list.is_empty(),
        "sys.path should include at least one entry"
    );

    let first = list.get(0).expect("sys.path[0] should exist");
    let first_ptr = first
        .as_string_object_ptr()
        .expect("sys.path[0] should be interned string") as *const u8;
    assert_eq!(
        interned_by_ptr(first_ptr)
            .expect("sys.path[0] should resolve")
            .as_ref(),
        ""
    );
}

// =========================================================================
// Error Handling Tests
// =========================================================================

#[test]
fn test_unknown_attribute_error() {
    let sys = SysModule::new();
    let result = sys.get_attr("unknown_attr");
    assert!(result.is_err());
    match result {
        Err(ModuleError::AttributeError(msg)) => {
            assert!(msg.contains("no attribute"));
        }
        _ => panic!("Expected AttributeError"),
    }
}

// =========================================================================
// Platform and Version Tests (Direct Access)
// =========================================================================

#[test]
fn test_platform_direct() {
    let sys = SysModule::new();
    let platform = sys.platform();
    assert!(matches!(
        platform,
        Platform::Windows | Platform::Linux | Platform::MacOS | Platform::FreeBSD
    ));
}

#[test]
fn test_version_direct() {
    let sys = SysModule::new();
    let version = sys.version();
    assert!(version.contains("3.12"));
}
