use super::*;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::int::value_to_bigint;
use prism_runtime::types::set::SetObject;
use prism_runtime::types::tuple::{TupleObject, value_as_tuple_ref};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_temp_path(label: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("prism_nt_{label}_{}_{}", std::process::id(), nonce))
}

#[test]
fn test_nt_module_exposes_bootstrap_surface() {
    let module = NtModule::new();

    assert!(module.get_attr("__all__").is_ok());
    assert!(module.get_attr("access").is_ok());
    assert!(module.get_attr("open").is_ok());
    assert!(module.get_attr("close").is_ok());
    assert!(module.get_attr("pipe").is_ok());
    assert!(module.get_attr("read").is_ok());
    assert!(module.get_attr("write").is_ok());
    assert!(module.get_attr("stat").is_ok());
    assert!(module.get_attr("lstat").is_ok());
    assert!(module.get_attr("listdir").is_ok());
    assert!(module.get_attr("getpid").is_ok());
    assert!(module.get_attr("cpu_count").is_ok());
    assert!(module.get_attr("fspath").is_ok());
    assert!(module.get_attr("_path_splitroot").is_ok());
    assert!(module.get_attr("scandir").is_ok());
    assert!(module.get_attr("getcwd").is_ok());
    assert!(module.get_attr("getcwdb").is_ok());
    assert!(module.get_attr("get_terminal_size").is_ok());
    assert!(module.get_attr("mkdir").is_ok());
    assert!(module.get_attr("remove").is_ok());
    assert!(module.get_attr("rename").is_ok());
    assert!(module.get_attr("replace").is_ok());
    assert!(module.get_attr("rmdir").is_ok());
    assert!(module.get_attr("unlink").is_ok());
    assert!(module.get_attr("stat_result").is_ok());
    assert!(module.get_attr("terminal_size").is_ok());
    assert!(module.get_attr("environ").is_ok());
    assert!(module.get_attr("_have_functions").is_ok());
    assert!(module.get_attr("_exit").is_ok());
    assert!(module.get_attr("urandom").is_ok());
    assert!(module.get_attr("putenv").is_ok());
    assert!(module.get_attr("unsetenv").is_ok());
    assert!(module.get_attr("O_RDONLY").is_ok());
    assert!(module.get_attr("O_WRONLY").is_ok());
    assert!(module.get_attr("O_RDWR").is_ok());
    assert!(module.get_attr("O_APPEND").is_ok());
    assert!(module.get_attr("O_CREAT").is_ok());
    assert!(module.get_attr("O_TRUNC").is_ok());
    assert!(module.get_attr("O_EXCL").is_ok());
    assert!(module.get_attr("O_BINARY").is_ok());
    assert!(module.get_attr("F_OK").is_ok());
    assert!(module.get_attr("R_OK").is_ok());
    assert!(module.get_attr("W_OK").is_ok());
    assert!(module.get_attr("X_OK").is_ok());
}

#[test]
fn test_nt_module_all_lists_public_exports() {
    let module = NtModule::new();
    let all_value = module.get_attr("__all__").expect("__all__ should exist");
    let ptr = all_value
        .as_object_ptr()
        .expect("__all__ should be a tuple object");
    let tuple = unsafe { &*(ptr as *const TupleObject) };

    assert!(
        tuple
            .iter()
            .any(|value| value == &Value::string(intern("open")))
    );
    assert!(
        tuple
            .iter()
            .any(|value| value == &Value::string(intern("close")))
    );
    assert!(
        tuple
            .iter()
            .any(|value| value == &Value::string(intern("pipe")))
    );
    assert!(
        tuple
            .iter()
            .any(|value| value == &Value::string(intern("read")))
    );
    assert!(
        tuple
            .iter()
            .any(|value| value == &Value::string(intern("write")))
    );
    assert!(
        tuple
            .iter()
            .any(|value| value == &Value::string(intern("O_RDONLY")))
    );
    assert!(
        tuple
            .iter()
            .any(|value| value == &Value::string(intern("O_CREAT")))
    );
    assert!(
        tuple
            .iter()
            .any(|value| value == &Value::string(intern("O_BINARY")))
    );
    assert!(
        tuple
            .iter()
            .any(|value| value == &Value::string(intern("F_OK")))
    );
    assert!(
        tuple
            .iter()
            .any(|value| value == &Value::string(intern("stat")))
    );
    assert!(
        tuple
            .iter()
            .any(|value| value == &Value::string(intern("stat_result")))
    );
    assert!(
        tuple
            .iter()
            .any(|value| value == &Value::string(intern("listdir")))
    );
    assert!(
        tuple
            .iter()
            .any(|value| value == &Value::string(intern("getpid")))
    );
    assert!(
        tuple
            .iter()
            .any(|value| value == &Value::string(intern("cpu_count")))
    );
    assert!(
        tuple
            .iter()
            .any(|value| value == &Value::string(intern("fspath")))
    );
    assert!(
        tuple
            .iter()
            .any(|value| value == &Value::string(intern("getcwd")))
    );
    assert!(
        tuple
            .iter()
            .any(|value| value == &Value::string(intern("terminal_size")))
    );
    assert!(
        tuple
            .iter()
            .any(|value| value == &Value::string(intern("rmdir")))
    );
    assert!(
        tuple
            .iter()
            .any(|value| value == &Value::string(intern("environ")))
    );
    assert!(
        tuple
            .iter()
            .any(|value| value == &Value::string(intern("urandom")))
    );
    assert!(
        tuple
            .iter()
            .any(|value| value == &Value::string(intern("putenv")))
    );
    assert!(
        tuple
            .iter()
            .any(|value| value == &Value::string(intern("unsetenv")))
    );
}

#[test]
fn test_nt_pipe_read_write_round_trips_bytes() {
    let pipe = nt_pipe(&[]).expect("nt.pipe should create a pipe");
    let pipe_ptr = pipe
        .as_object_ptr()
        .expect("nt.pipe should return a tuple object");
    let pipe = unsafe { &*(pipe_ptr as *const TupleObject) };
    assert_eq!(pipe.len(), 2);

    let read_fd = pipe
        .get(0)
        .expect("read fd should exist")
        .as_int()
        .expect("read fd should be an int");
    let write_fd = pipe
        .get(1)
        .expect("write fd should exist")
        .as_int()
        .expect("write fd should be an int");

    let payload = leak_object_value(BytesObject::from_slice(b"x"));
    let written = nt_write(&[Value::int(write_fd).expect("write fd should fit"), payload])
        .expect("nt.write should succeed")
        .as_int()
        .expect("nt.write should return an integer");
    assert_eq!(written, 1);

    let result = nt_read(&[
        Value::int(read_fd).expect("read fd should fit"),
        Value::int(1).expect("read length should fit"),
    ])
    .expect("nt.read should succeed");
    let result_ptr = result
        .as_object_ptr()
        .expect("nt.read should return a bytes object");
    let result = unsafe { &*(result_ptr as *const BytesObject) };
    assert_eq!(result.as_bytes(), b"x");

    let _ = nt_close(&[Value::int(read_fd).expect("read fd should fit")]);
    let _ = nt_close(&[Value::int(write_fd).expect("write fd should fit")]);
}

#[test]
fn test_nt_putenv_and_unsetenv_update_process_environment() {
    let key = format!(
        "PRISM_NT_PUTENV_TEST_{}_{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos()
    );

    unsafe {
        std::env::remove_var(&key);
    }

    let result = nt_putenv(&[
        Value::string(intern(&key)),
        Value::string(intern("configured")),
    ])
    .expect("putenv should accept a valid key and value");
    assert!(result.is_none());
    assert_eq!(
        std::env::var(&key).expect("putenv should set process environment"),
        "configured"
    );

    let result =
        nt_unsetenv(&[Value::string(intern(&key))]).expect("unsetenv should accept a valid key");
    assert!(result.is_none());
    assert!(
        std::env::var_os(&key).is_none(),
        "unsetenv should remove process environment entry"
    );
}

#[test]
fn test_nt_module_environ_is_dict() {
    let module = NtModule::new();
    let environ = module.get_attr("environ").expect("environ should exist");
    let ptr = environ
        .as_object_ptr()
        .expect("environ should be represented as dict object");
    let _dict = unsafe { &*(ptr as *const DictObject) };
}

#[test]
fn test_nt_module_have_functions_is_frozenset() {
    let module = NtModule::new();
    let value = module
        .get_attr("_have_functions")
        .expect("_have_functions should exist");
    let ptr = value
        .as_object_ptr()
        .expect("_have_functions should be a set-like object");
    let set = unsafe { &*(ptr as *const SetObject) };

    assert!(
        set.contains(Value::string(intern("MS_WINDOWS"))),
        "Windows bootstrap should advertise the native platform marker"
    );
    assert!(
        set.contains(Value::string(intern("HAVE_LSTAT"))),
        "Windows bootstrap should advertise lstat availability"
    );
    assert_eq!(set.header.type_id, TypeId::FROZENSET);
}

#[test]
fn test_nt_bootstrap_functions_fail_explicitly_when_called() {
    let err = nt_scandir(&[]).expect_err("nt.scandir should not silently succeed");
    assert!(err.to_string().contains("nt.scandir()"));
}

#[test]
fn test_nt_listdir_returns_directory_entries_as_strings_by_default() {
    let dir = unique_temp_path("listdir");
    fs::create_dir_all(&dir).expect("temp directory should be created");
    let alpha = dir.join("alpha.txt");
    let beta = dir.join("beta.txt");
    fs::write(&alpha, b"alpha").expect("alpha file should be written");
    fs::write(&beta, b"beta").expect("beta file should be written");

    let result = nt_listdir(&[Value::string(intern(&dir.to_string_lossy()))])
        .expect("nt.listdir should succeed for an existing directory");
    let ptr = result
        .as_object_ptr()
        .expect("nt.listdir should return a list object");
    let list = unsafe { &*(ptr as *const ListObject) };

    let mut names = list
        .iter()
        .map(|value| value_to_string(*value).expect("listdir entry should be string"))
        .collect::<Vec<_>>();
    names.sort();

    assert_eq!(names, vec!["alpha.txt".to_string(), "beta.txt".to_string()]);

    let _ = fs::remove_file(alpha);
    let _ = fs::remove_file(beta);
    let _ = fs::remove_dir(dir);
}

#[test]
fn test_nt_listdir_preserves_bytes_flavor_for_bytes_paths() {
    let dir = unique_temp_path("listdir_bytes");
    fs::create_dir_all(&dir).expect("temp directory should be created");
    let gamma = dir.join("gamma.txt");
    fs::write(&gamma, b"gamma").expect("gamma file should be written");

    let result = nt_listdir(&[leak_object_value(BytesObject::from_slice(
        dir.to_string_lossy().as_bytes(),
    ))])
    .expect("nt.listdir should accept bytes paths");
    let ptr = result
        .as_object_ptr()
        .expect("nt.listdir should return a list object");
    let list = unsafe { &*(ptr as *const ListObject) };
    assert_eq!(list.len(), 1);

    let entry_ptr = list
        .get(0)
        .expect("bytes listdir result should have one entry")
        .as_object_ptr()
        .expect("bytes listdir entry should be heap allocated");
    let entry = unsafe { &*(entry_ptr as *const BytesObject) };
    assert_eq!(entry.as_bytes(), b"gamma.txt");

    let _ = fs::remove_file(gamma);
    let _ = fs::remove_dir(dir);
}

#[test]
fn test_nt_listdir_raises_os_error_for_missing_directory() {
    let path = unique_temp_path("listdir_missing");
    let err = nt_listdir(&[Value::string(intern(&path.to_string_lossy()))])
        .expect_err("nt.listdir should fail for missing directories");
    assert!(matches!(err, BuiltinError::OSError(_)));
    assert!(err.to_string().contains("listdir() failed"));
}

#[test]
fn test_nt_getcwd_returns_absolute_string_path() {
    let result = nt_getcwd(&[]).expect("nt.getcwd should succeed");
    let path = value_to_string(result).expect("nt.getcwd should return a string");
    assert!(
        Path::new(&path).is_absolute(),
        "cwd should be absolute: {path}"
    );
}

#[test]
fn test_nt_getcwdb_returns_bytes_path() {
    let result = nt_getcwdb(&[]).expect("nt.getcwdb should succeed");
    let ptr = result
        .as_object_ptr()
        .expect("nt.getcwdb should return a bytes object");
    let bytes = unsafe { &*(ptr as *const BytesObject) };
    let cwd = std::env::current_dir().expect("current_dir should succeed");
    assert_eq!(bytes.as_bytes(), cwd.to_string_lossy().as_bytes());
    assert_eq!(bytes.header.type_id, TypeId::BYTES);
}

#[test]
fn test_nt_urandom_returns_requested_number_of_bytes() {
    let result = nt_urandom(&[Value::int(32).expect("length should fit")])
        .expect("nt.urandom should succeed");
    let ptr = result
        .as_object_ptr()
        .expect("nt.urandom should return a bytes object");
    let bytes = unsafe { &*(ptr as *const BytesObject) };
    assert_eq!(bytes.len(), 32);
    assert_eq!(bytes.header.type_id, TypeId::BYTES);
}

#[test]
fn test_nt_terminal_size_constructor_builds_named_record() {
    let value = nt_terminal_size(&[leak_object_value(TupleObject::from_vec(vec![
        Value::int(120).expect("columns should fit"),
        Value::int(40).expect("lines should fit"),
    ]))])
    .expect("terminal_size constructor should accept a 2-tuple");
    let ptr = value
        .as_object_ptr()
        .expect("terminal_size should return a heap object");
    let shaped = unsafe { &*(ptr as *const ShapedObject) };

    assert_eq!(
        shaped
            .get_property("columns")
            .and_then(|value| value.as_int()),
        Some(120)
    );
    assert_eq!(
        shaped
            .get_property("lines")
            .and_then(|value| value.as_int()),
        Some(40)
    );
}

#[test]
fn test_nt_terminal_size_rejects_non_sequences() {
    let err = nt_terminal_size(&[Value::int(80).expect("integer should fit")])
        .expect_err("terminal_size should validate constructor input");
    assert!(matches!(err, BuiltinError::TypeError(_)));
}

#[test]
fn test_nt_get_terminal_size_rejects_bad_file_descriptors() {
    let err = nt_get_terminal_size(&[Value::int(99).expect("descriptor should fit")])
        .expect_err("unsupported file descriptors should fail");
    assert!(matches!(err, BuiltinError::OSError(_)));
    assert!(err.to_string().contains("bad file descriptor"));
}

#[test]
fn test_nt_getpid_returns_process_identifier() {
    let result = nt_getpid(&[]).expect("nt.getpid should succeed");
    assert_eq!(result.as_int(), Some(std::process::id() as i64));
}

#[test]
fn test_nt_cpu_count_returns_positive_integer_or_none() {
    let result = nt_cpu_count(&[]).expect("nt.cpu_count should succeed");
    if !result.is_none() {
        assert!(
            result.as_int().is_some_and(|count| count >= 1),
            "cpu_count should be positive or None"
        );
    }
}

fn splitroot_strings(path: &str) -> (String, String) {
    let result = nt_path_splitroot(&[Value::string(intern(path))])
        .expect("_path_splitroot should accept str paths");
    let ptr = result
        .as_object_ptr()
        .expect("_path_splitroot should return a tuple");
    let tuple = unsafe { &*(ptr as *const TupleObject) };
    assert_eq!(tuple.len(), 2);
    (
        value_to_string(tuple.get(0).expect("root should exist")).unwrap(),
        value_to_string(tuple.get(1).expect("tail should exist")).unwrap(),
    )
}

#[test]
fn test_nt_path_splitroot_matches_importlib_windows_contract_for_strings() {
    assert_eq!(
        splitroot_strings(r"C:\Users\Barney"),
        (r"C:\".to_string(), "Users\\Barney".to_string())
    );
    assert_eq!(
        splitroot_strings("C:///spam///ham"),
        ("C:/".to_string(), "//spam///ham".to_string())
    );
    assert_eq!(
        splitroot_strings(r"\\server\share\folder"),
        (r"\\server\share\".to_string(), "folder".to_string())
    );
    assert_eq!(
        splitroot_strings(r"Windows\notepad"),
        ("".to_string(), r"Windows\notepad".to_string())
    );
}

#[test]
fn test_nt_path_splitroot_preserves_bytes_paths() {
    let result = nt_path_splitroot(&[leak_object_value(BytesObject::from_slice(br"C:/Temp"))])
        .expect("_path_splitroot should accept bytes paths");
    let ptr = result
        .as_object_ptr()
        .expect("_path_splitroot should return a tuple");
    let tuple = unsafe { &*(ptr as *const TupleObject) };
    assert_eq!(tuple.len(), 2);

    let root_ptr = tuple
        .get(0)
        .expect("root should exist")
        .as_object_ptr()
        .expect("root should be bytes");
    let tail_ptr = tuple
        .get(1)
        .expect("tail should exist")
        .as_object_ptr()
        .expect("tail should be bytes");
    let root = unsafe { &*(root_ptr as *const BytesObject) };
    let tail = unsafe { &*(tail_ptr as *const BytesObject) };

    assert_eq!(root.as_bytes(), b"C:/");
    assert_eq!(tail.as_bytes(), b"Temp");
}

#[test]
fn test_nt_open_returns_closeable_file_descriptor_for_existing_file() {
    let path = unique_temp_path("open_existing");
    fs::write(&path, b"ready").expect("fixture file should be written");

    let fd = nt_open(&[
        Value::string(intern(path.to_str().expect("temp path should be utf-8"))),
        Value::int((O_RDONLY | O_BINARY) as i64).expect("flags should fit"),
    ])
    .expect("nt.open should open existing files")
    .as_int()
    .expect("nt.open should return an integer fd");
    assert!(fd >= 0);

    nt_close(&[Value::int(fd).expect("fd should fit")]).expect("nt.close should close fd");
    let _ = fs::remove_file(path);
}

#[test]
fn test_nt_open_create_truncate_and_exclusive_flags() {
    let path = unique_temp_path("open_create");
    let path_value = Value::string(intern(path.to_str().expect("temp path should be utf-8")));
    let create_flags = (O_WRONLY | O_CREAT | O_TRUNC | O_BINARY) as i64;

    let fd = nt_open(&[
        path_value,
        Value::int(create_flags).expect("flags should fit"),
        Value::int(0o600).expect("mode should fit"),
    ])
    .expect("nt.open should create files")
    .as_int()
    .expect("nt.open should return an integer fd");
    nt_close(&[Value::int(fd).expect("fd should fit")]).expect("nt.close should close fd");
    assert!(path.exists());

    let err = nt_open(&[
        path_value,
        Value::int((O_WRONLY | O_CREAT | O_EXCL | O_BINARY) as i64).expect("flags should fit"),
    ])
    .expect_err("exclusive create should fail for an existing file");
    assert!(matches!(err, BuiltinError::OSError(_)));

    let _ = fs::remove_file(path);
}

#[test]
fn test_nt_open_validates_arguments() {
    let err = nt_open(&[Value::none()]).expect_err("nt.open should validate arity");
    assert!(matches!(err, BuiltinError::TypeError(_)));

    let err = nt_open(&[
        Value::string(intern("ignored")),
        Value::string(intern("bad flags")),
    ])
    .expect_err("nt.open should validate flags");
    assert!(matches!(err, BuiltinError::TypeError(_)));
}

#[test]
fn test_nt_remove_and_unlink_delete_files() {
    let remove_path = unique_temp_path("remove");
    fs::write(&remove_path, b"remove").expect("temp file should be written");
    nt_remove(&[Value::string(intern(&remove_path.to_string_lossy()))])
        .expect("nt.remove should delete files");
    assert!(!remove_path.exists());

    let unlink_path = unique_temp_path("unlink");
    fs::write(&unlink_path, b"unlink").expect("temp file should be written");
    nt_unlink(&[Value::string(intern(&unlink_path.to_string_lossy()))])
        .expect("nt.unlink should delete files");
    assert!(!unlink_path.exists());
}

#[test]
fn test_nt_access_tracks_existence_and_writability() {
    let file = unique_temp_path("access");
    fs::write(&file, b"probe").expect("probe file should be written");

    let exists = nt_access(&[
        Value::string(intern(&file.to_string_lossy())),
        Value::int(0).expect("F_OK should fit"),
    ])
    .expect("access should succeed");
    assert_eq!(exists.as_bool(), Some(true));

    let writable = nt_access(&[
        Value::string(intern(&file.to_string_lossy())),
        Value::int(2).expect("W_OK should fit"),
    ])
    .expect("access should accept W_OK");
    assert_eq!(writable.as_bool(), Some(true));

    let missing = nt_access(&[
        Value::string(intern(
            &unique_temp_path("missing_access").to_string_lossy(),
        )),
        Value::int(0).expect("F_OK should fit"),
    ])
    .expect("access should handle missing paths");
    assert_eq!(missing.as_bool(), Some(false));

    let _ = fs::remove_file(file);
}

#[test]
fn test_nt_mkdir_and_rmdir_manage_directories() {
    let dir = unique_temp_path("mkdir");
    nt_mkdir(&[Value::string(intern(&dir.to_string_lossy()))])
        .expect("nt.mkdir should create directories");
    assert!(dir.is_dir());

    nt_rmdir(&[Value::string(intern(&dir.to_string_lossy()))])
        .expect("nt.rmdir should remove empty directories");
    assert!(!dir.exists());
}

#[test]
fn test_nt_rename_and_replace_move_files() {
    let src = unique_temp_path("rename_src");
    let dst = unique_temp_path("rename_dst");
    fs::write(&src, b"payload").expect("source file should be created");
    nt_rename(&[
        Value::string(intern(&src.to_string_lossy())),
        Value::string(intern(&dst.to_string_lossy())),
    ])
    .expect("nt.rename should move files");
    assert!(!src.exists());
    assert_eq!(
        fs::read(&dst).expect("destination should exist"),
        b"payload"
    );

    let replacement = unique_temp_path("replace_src");
    fs::write(&replacement, b"replacement").expect("replacement file should be created");
    nt_replace(&[
        Value::string(intern(&replacement.to_string_lossy())),
        Value::string(intern(&dst.to_string_lossy())),
    ])
    .expect("nt.replace should atomically replace files");
    assert!(!replacement.exists());
    assert_eq!(
        fs::read(&dst).expect("destination should remain after replace"),
        b"replacement"
    );

    let _ = fs::remove_file(dst);
}

#[test]
fn test_nt_module_stat_result_exposes_cpython_metadata() {
    let module = NtModule::new();
    let class_value = module
        .get_attr("stat_result")
        .expect("stat_result should be exported");
    let class_ptr = class_value
        .as_object_ptr()
        .expect("stat_result should be a heap class");
    let class = unsafe { &*(class_ptr as *const PyClassObject) };

    assert_eq!(
        class
            .get_attr(&intern("n_sequence_fields"))
            .and_then(|value| value.as_int()),
        Some(10)
    );
    assert_eq!(
        class
            .get_attr(&intern("n_fields"))
            .and_then(|value| value.as_int()),
        Some(20)
    );
    assert_eq!(
        class
            .get_attr(&intern("n_unnamed_fields"))
            .and_then(|value| value.as_int()),
        Some(3)
    );
    assert!(class.get_attr(&intern("st_file_attributes")).is_some());
    assert!(class.get_attr(&intern("st_reparse_tag")).is_some());

    let match_args = class
        .get_attr(&intern("__match_args__"))
        .expect("__match_args__ should exist");
    let tuple = value_as_tuple_ref(match_args).expect("__match_args__ should be a tuple");
    assert_eq!(tuple.len(), STAT_RESULT_MATCH_ARGS.len());
}

#[test]
fn test_nt_stat_returns_stat_result_instance_with_expected_fields() {
    let path = unique_temp_path("stat");
    fs::write(&path, b"hello world").expect("temp file should be written");

    let result = nt_stat(&[Value::string(intern(&path.to_string_lossy()))])
        .expect("nt.stat should succeed for an existing file");
    let ptr = result
        .as_object_ptr()
        .expect("nt.stat should return a heap object");
    let shaped = unsafe { &*(ptr as *const ShapedObject) };

    assert_eq!(
        shaped
            .get_property("st_size")
            .and_then(|value| value.as_int()),
        Some(11)
    );
    assert!(
        shaped
            .get_property("st_mode")
            .and_then(|value| value.as_int())
            .is_some()
    );
    assert!(
        shaped
            .get_property("st_mtime")
            .and_then(|value| value.as_float())
            .is_some()
    );
    assert!(
        value_to_bigint(
            shaped
                .get_property("st_mtime_ns")
                .expect("st_mtime_ns should be present")
        )
        .is_some()
    );
    assert!(
        shaped
            .get_property("st_file_attributes")
            .and_then(|value| value.as_int())
            .is_some()
    );

    let _ = fs::remove_file(path);
}
