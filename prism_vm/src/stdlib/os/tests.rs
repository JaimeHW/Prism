use super::*;
use crate::VirtualMachine;
use prism_runtime::types::bytes::BytesObject;

// =========================================================================
// Module Creation Tests
// =========================================================================

#[test]
fn test_os_module_new() {
    let os = OsModule::new();
    assert_eq!(os.name(), "os");
}

// =========================================================================
// O_* Flag Tests
// =========================================================================

#[test]
fn test_o_rdonly() {
    let os = OsModule::new();
    let val = os.get_attr("O_RDONLY").unwrap();
    assert!(val.as_int().is_some());
}

#[test]
fn test_o_wronly() {
    let os = OsModule::new();
    let val = os.get_attr("O_WRONLY").unwrap();
    assert!(val.as_int().is_some());
}

#[test]
fn test_o_rdwr() {
    let os = OsModule::new();
    let val = os.get_attr("O_RDWR").unwrap();
    assert!(val.as_int().is_some());
}

#[test]
fn test_o_creat() {
    let os = OsModule::new();
    let val = os.get_attr("O_CREAT").unwrap();
    assert!(val.as_int().is_some());
}

#[test]
fn test_o_trunc() {
    let os = OsModule::new();
    let val = os.get_attr("O_TRUNC").unwrap();
    assert!(val.as_int().is_some());
}

#[test]
fn test_o_append() {
    let os = OsModule::new();
    let val = os.get_attr("O_APPEND").unwrap();
    assert!(val.as_int().is_some());
}

#[test]
fn test_o_excl() {
    let os = OsModule::new();
    let val = os.get_attr("O_EXCL").unwrap();
    assert!(val.as_int().is_some());
}

// =========================================================================
// Placeholder Attribute Tests
// =========================================================================

#[test]
fn test_name_placeholder() {
    let os = OsModule::new();
    let name = os.get_attr("name").unwrap();
    assert!(name.is_none()); // Placeholder
}

#[test]
fn test_sep_placeholder() {
    let os = OsModule::new();
    let sep = os.get_attr("sep").unwrap();
    assert!(sep.is_none());
}

#[test]
fn test_getcwd_placeholder() {
    let os = OsModule::new();
    let getcwd = os.get_attr("getcwd").unwrap();
    assert!(getcwd.is_none());
}

#[test]
fn test_path_placeholder() {
    let os = OsModule::new();
    let path = os.get_attr("path").unwrap();
    assert!(path.is_none());
}

#[test]
fn test_environ_placeholder() {
    let os = OsModule::new();
    let environ = os.get_attr("environ").unwrap();
    assert!(environ.is_none());
}

#[test]
fn test_urandom_returns_callable_builtin() {
    let os = OsModule::new();
    assert!(os.get_attr("urandom").unwrap().as_object_ptr().is_some());
}

#[test]
fn test_fspath_returns_callable_builtin() {
    let os = OsModule::new();
    assert!(os.get_attr("fspath").unwrap().as_object_ptr().is_some());
}

#[test]
fn test_os_urandom_returns_requested_number_of_bytes() {
    let result = os_urandom(&[Value::int(24).expect("length should fit")])
        .expect("os.urandom should succeed");
    let ptr = result
        .as_object_ptr()
        .expect("os.urandom should return a bytes object");
    let bytes = unsafe { &*(ptr as *const BytesObject) };
    assert_eq!(bytes.len(), 24);
}

#[test]
fn test_os_fspath_returns_str_and_bytes_unchanged() {
    let mut vm = VirtualMachine::new();
    let string = Value::string(prism_core::intern::intern("path"));
    assert_eq!(
        os_fspath(&mut vm, &[string]).expect("str path should be accepted"),
        string
    );

    let bytes_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b"path")));
    let bytes = Value::object_ptr(bytes_ptr as *const ());
    assert_eq!(
        os_fspath(&mut vm, &[bytes]).expect("bytes path should be accepted"),
        bytes
    );

    unsafe {
        drop(Box::from_raw(bytes_ptr));
    }
}

#[test]
fn test_os_fspath_rejects_bytearray_and_non_path_values() {
    let mut vm = VirtualMachine::new();
    let bytearray_ptr = Box::into_raw(Box::new(BytesObject::bytearray_from_slice(b"path")));
    let bytearray = Value::object_ptr(bytearray_ptr as *const ());
    let bytearray_err = os_fspath(&mut vm, &[bytearray])
        .expect_err("bytearray is not an accepted path protocol result");
    assert!(matches!(bytearray_err, BuiltinError::TypeError(_)));

    let int_err = os_fspath(&mut vm, &[Value::int(7).expect("int should fit")])
        .expect_err("non-path object should be rejected");
    assert!(matches!(int_err, BuiltinError::TypeError(_)));

    unsafe {
        drop(Box::from_raw(bytearray_ptr));
    }
}

// =========================================================================
// Error Handling Tests
// =========================================================================

#[test]
fn test_unknown_attribute_error() {
    let os = OsModule::new();
    let result = os.get_attr("nonexistent");
    assert!(result.is_err());
    match result {
        Err(ModuleError::AttributeError(msg)) => {
            assert!(msg.contains("no attribute"));
        }
        _ => panic!("Expected AttributeError"),
    }
}

// =========================================================================
// Dir Tests
// =========================================================================

#[test]
fn test_dir_contains_name() {
    let os = OsModule::new();
    let attrs = os.dir();
    assert!(attrs.contains(&Arc::from("name")));
}

#[test]
fn test_dir_contains_getcwd() {
    let os = OsModule::new();
    let attrs = os.dir();
    assert!(attrs.contains(&Arc::from("getcwd")));
}

#[test]
fn test_dir_contains_path() {
    let os = OsModule::new();
    let attrs = os.dir();
    assert!(attrs.contains(&Arc::from("path")));
}

#[test]
fn test_dir_contains_environ() {
    let os = OsModule::new();
    let attrs = os.dir();
    assert!(attrs.contains(&Arc::from("environ")));
}

#[test]
fn test_dir_contains_flags() {
    let os = OsModule::new();
    let attrs = os.dir();
    assert!(attrs.contains(&Arc::from("O_RDONLY")));
    assert!(attrs.contains(&Arc::from("O_WRONLY")));
    assert!(attrs.contains(&Arc::from("O_RDWR")));
}

#[test]
fn test_dir_contains_fspath() {
    let os = OsModule::new();
    let attrs = os.dir();
    assert!(attrs.contains(&Arc::from("fspath")));
}

#[test]
fn test_dir_length() {
    let os = OsModule::new();
    let attrs = os.dir();
    assert!(attrs.len() >= 40); // Many attributes
}

// =========================================================================
// Environ Access Tests
// =========================================================================
