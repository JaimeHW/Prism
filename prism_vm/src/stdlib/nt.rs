//! Native Windows `nt` module bootstrap surface.
//!
//! CPython's `os.py` selects the platform backend by consulting
//! `sys.builtin_module_names` and then importing either `posix` or `nt`.
//! Prism currently relies on CPython's pure-Python `os.py` when a CPython stdlib
//! tree is available, so the runtime must provide a coherent `nt` module during
//! import bootstrap on Windows.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::stdlib::os::{
    O_APPEND, O_BINARY, O_CREAT, O_EXCL, O_RDONLY, O_RDWR, O_TRUNC, O_WRONLY, os_fspath,
};
use crate::stdlib::secure_random::urandom_value_from_args;
use num_bigint::BigInt;
use prism_core::Value;
use prism_core::intern::{intern, interned_by_ptr};
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, class_id_to_type_id, register_global_class,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::int::bigint_to_value;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::set::SetObject;
use prism_runtime::types::string::StringObject;
use prism_runtime::types::tuple::TupleObject;
use std::fs::Metadata;
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};
use std::time::UNIX_EPOCH;

static NT_OPEN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.open"), nt_open));
static NT_CLOSE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.close"), nt_close));
static NT_PIPE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.pipe"), nt_pipe));
static NT_READ_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.read"), nt_read));
static NT_WRITE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.write"), nt_write));
static NT_STAT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.stat"), nt_stat));
static NT_LSTAT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.lstat"), nt_lstat));
static NT_LISTDIR_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.listdir"), nt_listdir));
static NT_GETPID_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.getpid"), nt_getpid));
static NT_CPU_COUNT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.cpu_count"), nt_cpu_count));
static NT_FSPATH_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("nt.fspath"), os_fspath));
static NT_PATH_SPLITROOT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("nt._path_splitroot"), nt_path_splitroot)
});
static NT_SCANDIR_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.scandir"), nt_scandir));
static NT_GETCWD_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.getcwd"), nt_getcwd));
static NT_GETCWDB_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.getcwdb"), nt_getcwdb));
static NT_GET_TERMINAL_SIZE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("nt.get_terminal_size"), nt_get_terminal_size)
});
static NT_MKDIR_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.mkdir"), nt_mkdir));
static NT_REMOVE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.remove"), nt_remove));
static NT_RENAME_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.rename"), nt_rename));
static NT_REPLACE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.replace"), nt_replace));
static NT_RMDIR_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.rmdir"), nt_rmdir));
static NT_TERMINAL_SIZE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.terminal_size"), nt_terminal_size));
static NT_UNLINK_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.unlink"), nt_unlink));
static NT_EXIT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt._exit"), nt_exit));
static NT_ACCESS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.access"), nt_access));
static NT_URANDOM_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.urandom"), nt_urandom));
static NT_PUTENV_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.putenv"), nt_putenv));
static NT_UNSETENV_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.unsetenv"), nt_unsetenv));
static NT_STAT_RESULT_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(build_stat_result_class);

const STAT_RESULT_MATCH_ARGS: [&str; 7] = [
    "st_mode", "st_ino", "st_dev", "st_nlink", "st_uid", "st_gid", "st_size",
];

const STAT_RESULT_FIELD_NAMES: [&str; 17] = [
    "st_mode",
    "st_ino",
    "st_dev",
    "st_nlink",
    "st_uid",
    "st_gid",
    "st_size",
    "st_atime",
    "st_mtime",
    "st_ctime",
    "st_atime_ns",
    "st_mtime_ns",
    "st_ctime_ns",
    "st_birthtime",
    "st_birthtime_ns",
    "st_file_attributes",
    "st_reparse_tag",
];

/// Minimal native `nt` module required for CPython stdlib bootstrap.
pub struct NtModule {
    attrs: Vec<Arc<str>>,
    all_value: Value,
    environ_value: Value,
    have_functions_value: Value,
}

impl NtModule {
    /// Create a new `nt` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("__all__"),
                Arc::from("access"),
                Arc::from("open"),
                Arc::from("close"),
                Arc::from("pipe"),
                Arc::from("read"),
                Arc::from("write"),
                Arc::from("stat"),
                Arc::from("lstat"),
                Arc::from("listdir"),
                Arc::from("getpid"),
                Arc::from("cpu_count"),
                Arc::from("fspath"),
                Arc::from("_path_splitroot"),
                Arc::from("scandir"),
                Arc::from("getcwd"),
                Arc::from("getcwdb"),
                Arc::from("get_terminal_size"),
                Arc::from("mkdir"),
                Arc::from("remove"),
                Arc::from("rename"),
                Arc::from("replace"),
                Arc::from("rmdir"),
                Arc::from("unlink"),
                Arc::from("stat_result"),
                Arc::from("terminal_size"),
                Arc::from("environ"),
                Arc::from("_have_functions"),
                Arc::from("_exit"),
                Arc::from("urandom"),
                Arc::from("putenv"),
                Arc::from("unsetenv"),
                Arc::from("SEEK_SET"),
                Arc::from("SEEK_CUR"),
                Arc::from("SEEK_END"),
                Arc::from("O_RDONLY"),
                Arc::from("O_WRONLY"),
                Arc::from("O_RDWR"),
                Arc::from("O_APPEND"),
                Arc::from("O_CREAT"),
                Arc::from("O_TRUNC"),
                Arc::from("O_EXCL"),
                Arc::from("O_BINARY"),
                Arc::from("F_OK"),
                Arc::from("R_OK"),
                Arc::from("W_OK"),
                Arc::from("X_OK"),
            ],
            all_value: export_names_value(),
            environ_value: environ_value(),
            have_functions_value: empty_frozenset_value(),
        }
    }
}

impl Default for NtModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for NtModule {
    fn name(&self) -> &str {
        "nt"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__all__" => Ok(self.all_value),
            "access" => Ok(builtin_value(&NT_ACCESS_FUNCTION)),
            "open" => Ok(builtin_value(&NT_OPEN_FUNCTION)),
            "close" => Ok(builtin_value(&NT_CLOSE_FUNCTION)),
            "pipe" => Ok(builtin_value(&NT_PIPE_FUNCTION)),
            "read" => Ok(builtin_value(&NT_READ_FUNCTION)),
            "write" => Ok(builtin_value(&NT_WRITE_FUNCTION)),
            "stat" => Ok(builtin_value(&NT_STAT_FUNCTION)),
            "lstat" => Ok(builtin_value(&NT_LSTAT_FUNCTION)),
            "listdir" => Ok(builtin_value(&NT_LISTDIR_FUNCTION)),
            "getpid" => Ok(builtin_value(&NT_GETPID_FUNCTION)),
            "cpu_count" => Ok(builtin_value(&NT_CPU_COUNT_FUNCTION)),
            "fspath" => Ok(builtin_value(&NT_FSPATH_FUNCTION)),
            "_path_splitroot" => Ok(builtin_value(&NT_PATH_SPLITROOT_FUNCTION)),
            "scandir" => Ok(builtin_value(&NT_SCANDIR_FUNCTION)),
            "getcwd" => Ok(builtin_value(&NT_GETCWD_FUNCTION)),
            "getcwdb" => Ok(builtin_value(&NT_GETCWDB_FUNCTION)),
            "get_terminal_size" => Ok(builtin_value(&NT_GET_TERMINAL_SIZE_FUNCTION)),
            "mkdir" => Ok(builtin_value(&NT_MKDIR_FUNCTION)),
            "remove" => Ok(builtin_value(&NT_REMOVE_FUNCTION)),
            "rename" => Ok(builtin_value(&NT_RENAME_FUNCTION)),
            "replace" => Ok(builtin_value(&NT_REPLACE_FUNCTION)),
            "rmdir" => Ok(builtin_value(&NT_RMDIR_FUNCTION)),
            "unlink" => Ok(builtin_value(&NT_UNLINK_FUNCTION)),
            "stat_result" => Ok(stat_result_class_value()),
            "terminal_size" => Ok(builtin_value(&NT_TERMINAL_SIZE_FUNCTION)),
            "environ" => Ok(self.environ_value),
            "_have_functions" => Ok(self.have_functions_value),
            "_exit" => Ok(builtin_value(&NT_EXIT_FUNCTION)),
            "urandom" => Ok(builtin_value(&NT_URANDOM_FUNCTION)),
            "putenv" => Ok(builtin_value(&NT_PUTENV_FUNCTION)),
            "unsetenv" => Ok(builtin_value(&NT_UNSETENV_FUNCTION)),
            "SEEK_SET" => Ok(Value::int(0).expect("SEEK_SET fits in i64")),
            "SEEK_CUR" => Ok(Value::int(1).expect("SEEK_CUR fits in i64")),
            "SEEK_END" => Ok(Value::int(2).expect("SEEK_END fits in i64")),
            "O_RDONLY" => Ok(Value::int(O_RDONLY as i64).expect("O_RDONLY fits in i64")),
            "O_WRONLY" => Ok(Value::int(O_WRONLY as i64).expect("O_WRONLY fits in i64")),
            "O_RDWR" => Ok(Value::int(O_RDWR as i64).expect("O_RDWR fits in i64")),
            "O_APPEND" => Ok(Value::int(O_APPEND as i64).expect("O_APPEND fits in i64")),
            "O_CREAT" => Ok(Value::int(O_CREAT as i64).expect("O_CREAT fits in i64")),
            "O_TRUNC" => Ok(Value::int(O_TRUNC as i64).expect("O_TRUNC fits in i64")),
            "O_EXCL" => Ok(Value::int(O_EXCL as i64).expect("O_EXCL fits in i64")),
            "O_BINARY" => Ok(Value::int(O_BINARY as i64).expect("O_BINARY fits in i64")),
            "F_OK" => Ok(Value::int(0).expect("F_OK fits in i64")),
            "R_OK" => Ok(Value::int(4).expect("R_OK fits in i64")),
            "W_OK" => Ok(Value::int(2).expect("W_OK fits in i64")),
            "X_OK" => Ok(Value::int(1).expect("X_OK fits in i64")),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'nt' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

#[inline]
fn leak_object_value<T: prism_runtime::Trace + 'static>(object: T) -> Value {
    crate::alloc_managed_value(object)
}

fn export_names_value() -> Value {
    let names = [
        "access",
        "open",
        "close",
        "pipe",
        "read",
        "write",
        "stat",
        "lstat",
        "listdir",
        "getpid",
        "cpu_count",
        "fspath",
        "scandir",
        "getcwd",
        "getcwdb",
        "get_terminal_size",
        "mkdir",
        "remove",
        "rename",
        "replace",
        "rmdir",
        "unlink",
        "stat_result",
        "terminal_size",
        "environ",
        "urandom",
        "putenv",
        "unsetenv",
        "SEEK_SET",
        "SEEK_CUR",
        "SEEK_END",
        "O_RDONLY",
        "O_WRONLY",
        "O_RDWR",
        "O_APPEND",
        "O_CREAT",
        "O_TRUNC",
        "O_EXCL",
        "O_BINARY",
        "F_OK",
        "R_OK",
        "W_OK",
        "X_OK",
    ]
    .into_iter()
    .map(|name| Value::string(intern(name)))
    .collect::<Vec<_>>();
    leak_object_value(TupleObject::from_vec(names))
}

fn environ_value() -> Value {
    let vars = std::env::vars().collect::<Vec<_>>();
    let mut environ = DictObject::with_capacity(vars.len());
    for (key, value) in vars {
        environ.set(Value::string(intern(&key)), Value::string(intern(&value)));
    }
    leak_object_value(environ)
}

fn empty_frozenset_value() -> Value {
    let mut set = SetObject::with_capacity(2);
    set.add(Value::string(intern("HAVE_LSTAT")));
    set.add(Value::string(intern("MS_WINDOWS")));
    set.header.type_id = TypeId::FROZENSET;
    leak_object_value(set)
}

fn stat_result_class_value() -> Value {
    Value::object_ptr(Arc::as_ptr(stat_result_class()) as *const ())
}

fn stat_result_class() -> &'static Arc<PyClassObject> {
    &NT_STAT_RESULT_CLASS
}

fn build_stat_result_class() -> Arc<PyClassObject> {
    let class = Arc::new(PyClassObject::new_simple(intern("stat_result")));

    class.set_attr(intern("__module__"), Value::string(intern("os")));
    class.set_attr(intern("__qualname__"), Value::string(intern("stat_result")));
    class.set_attr(
        intern("__match_args__"),
        leak_object_value(TupleObject::from_vec(
            STAT_RESULT_MATCH_ARGS
                .into_iter()
                .map(|name| Value::string(intern(name)))
                .collect(),
        )),
    );
    class.set_attr(
        intern("n_sequence_fields"),
        Value::int(10).expect("stat_result.n_sequence_fields fits"),
    );
    class.set_attr(
        intern("n_fields"),
        Value::int(20).expect("stat_result.n_fields fits"),
    );
    class.set_attr(
        intern("n_unnamed_fields"),
        Value::int(3).expect("stat_result.n_unnamed_fields fits"),
    );

    for field_name in STAT_RESULT_FIELD_NAMES {
        class.set_attr(intern(field_name), Value::none());
    }

    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(class_id_to_type_id(class_id));
    }
    register_global_class(Arc::clone(&class), bitmap);

    class
}

fn value_to_string(value: Value) -> Option<String> {
    if value.is_string() {
        let ptr = value.as_string_object_ptr()?;
        let interned = interned_by_ptr(ptr as *const u8).or_else(|| {
            let string = unsafe { &*(ptr as *const StringObject) };
            Some(intern(string.as_str()))
        })?;
        return Some(interned.as_str().to_string());
    }

    let ptr = value.as_object_ptr()?;
    if crate::ops::objects::extract_type_id(ptr) != TypeId::STR {
        return None;
    }

    let string = unsafe { &*(ptr as *const StringObject) };
    Some(string.as_str().to_string())
}

fn parse_single_path_arg(args: &[Value], fn_name: &str) -> Result<PathBuf, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "{}() takes exactly one argument ({} given)",
            fn_name,
            args.len()
        )));
    }

    value_to_string(args[0]).map(PathBuf::from).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "{}() path must be str, not {}",
            fn_name,
            args[0].type_name()
        ))
    })
}

fn path_from_value(value: Value, fn_name: &str, parameter: &str) -> Result<PathBuf, BuiltinError> {
    if let Some(path) = value_to_string(value) {
        return Ok(PathBuf::from(path));
    }

    if let Some(bytes) = value_to_bytes(value) {
        return Ok(PathBuf::from(String::from_utf8_lossy(&bytes).into_owned()));
    }

    Err(BuiltinError::TypeError(format!(
        "{fn_name}() {parameter} must be str, bytes, or bytearray, not {}",
        value.type_name()
    )))
}

fn parse_two_path_args(args: &[Value], fn_name: &str) -> Result<(PathBuf, PathBuf), BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "{fn_name}() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    Ok((
        path_from_value(args[0], fn_name, "source path")?,
        path_from_value(args[1], fn_name, "destination path")?,
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PathValueFlavor {
    String,
    Bytes,
}

struct ParsedOptionalPathArg {
    path: PathBuf,
    flavor: PathValueFlavor,
}

fn value_to_bytes(value: Value) -> Option<Vec<u8>> {
    let ptr = value.as_object_ptr()?;
    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::BYTES | TypeId::BYTEARRAY => {
            Some(unsafe { &*(ptr as *const BytesObject) }.as_bytes().to_vec())
        }
        _ => None,
    }
}

fn nt_path_splitroot(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "_path_splitroot() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    if let Some(path) = value_to_string(args[0]) {
        let (root_end, tail_start) = windows_splitroot_indices(path.as_bytes());
        return Ok(leak_object_value(TupleObject::from_slice(&[
            Value::string(intern(&path[..root_end])),
            Value::string(intern(&path[tail_start..])),
        ])));
    }

    if let Some(path) = value_to_bytes(args[0]) {
        let (root_end, tail_start) = windows_splitroot_indices(&path);
        return Ok(leak_object_value(TupleObject::from_slice(&[
            leak_object_value(BytesObject::from_slice(&path[..root_end])),
            leak_object_value(BytesObject::from_slice(&path[tail_start..])),
        ])));
    }

    Err(BuiltinError::TypeError(format!(
        "_path_splitroot() path must be str, bytes, or bytearray, not {}",
        args[0].type_name()
    )))
}

fn windows_splitroot_indices(path: &[u8]) -> (usize, usize) {
    if path.first().is_some_and(|byte| is_windows_sep(*byte)) {
        if path.get(1).is_some_and(|byte| is_windows_sep(*byte)) {
            let start = if starts_with_unc_device_prefix(path) {
                8
            } else {
                2
            };
            let Some(first_sep) = find_windows_sep(path, start) else {
                return (path.len(), path.len());
            };
            let Some(second_sep) = find_windows_sep(path, first_sep + 1) else {
                return (path.len(), path.len());
            };
            let split = second_sep + 1;
            return (split, split);
        }

        return (1, 1);
    }

    if path.get(1) == Some(&b':') {
        if path.get(2).is_some_and(|byte| is_windows_sep(*byte)) {
            return (3, 3);
        }
        return (2, 2);
    }

    (0, 0)
}

#[inline]
fn is_windows_sep(byte: u8) -> bool {
    byte == b'\\' || byte == b'/'
}

fn starts_with_unc_device_prefix(path: &[u8]) -> bool {
    path.len() >= 8
        && is_windows_sep(path[0])
        && is_windows_sep(path[1])
        && path[2] == b'?'
        && is_windows_sep(path[3])
        && path[4].eq_ignore_ascii_case(&b'U')
        && path[5].eq_ignore_ascii_case(&b'N')
        && path[6].eq_ignore_ascii_case(&b'C')
        && is_windows_sep(path[7])
}

fn find_windows_sep(path: &[u8], start: usize) -> Option<usize> {
    path.iter()
        .enumerate()
        .skip(start)
        .find_map(|(index, byte)| is_windows_sep(*byte).then_some(index))
}

fn parse_optional_path_arg(
    args: &[Value],
    fn_name: &str,
) -> Result<ParsedOptionalPathArg, BuiltinError> {
    match args.len() {
        0 => Ok(ParsedOptionalPathArg {
            path: PathBuf::from("."),
            flavor: PathValueFlavor::String,
        }),
        1 => {
            if let Some(path) = value_to_string(args[0]) {
                return Ok(ParsedOptionalPathArg {
                    path: PathBuf::from(path),
                    flavor: PathValueFlavor::String,
                });
            }

            if let Some(bytes) = value_to_bytes(args[0]) {
                return Ok(ParsedOptionalPathArg {
                    path: PathBuf::from(String::from_utf8_lossy(&bytes).into_owned()),
                    flavor: PathValueFlavor::Bytes,
                });
            }

            Err(BuiltinError::TypeError(format!(
                "{}() path must be str, bytes, or bytearray, not {}",
                fn_name,
                args[0].type_name()
            )))
        }
        n => Err(BuiltinError::TypeError(format!(
            "{}() takes at most 1 argument ({} given)",
            fn_name, n
        ))),
    }
}

fn file_type_mode_bits(metadata: &Metadata) -> i64 {
    let file_type = metadata.file_type();
    let mut mode = if file_type.is_dir() {
        0o040000
    } else {
        0o100000
    };
    mode |= 0o444;
    if !metadata.permissions().readonly() {
        mode |= 0o222;
    }
    if file_type.is_dir() {
        mode |= 0o111;
    }
    mode
}

fn system_time_parts(time: Result<std::time::SystemTime, std::io::Error>) -> (f64, i64) {
    let Ok(time) = time else {
        return (0.0, 0);
    };
    let Ok(duration) = time.duration_since(UNIX_EPOCH) else {
        return (0.0, 0);
    };
    let seconds = duration.as_secs_f64();
    let nanos = duration
        .as_secs()
        .saturating_mul(1_000_000_000)
        .saturating_add(duration.subsec_nanos() as u64);
    (seconds, i64::try_from(nanos).unwrap_or(i64::MAX))
}

#[cfg(windows)]
fn metadata_file_attributes(metadata: &Metadata) -> i64 {
    use std::os::windows::fs::MetadataExt;
    i64::from(metadata.file_attributes())
}

#[cfg(not(windows))]
fn metadata_file_attributes(_metadata: &Metadata) -> i64 {
    0
}

#[cfg(windows)]
fn metadata_reparse_tag(_metadata: &Metadata) -> i64 {
    0
}

#[cfg(not(windows))]
fn metadata_reparse_tag(_metadata: &Metadata) -> i64 {
    0
}

fn build_stat_result_value(path: &Path, metadata: &Metadata) -> Value {
    let class = stat_result_class();
    let registry = shape_registry();
    let mut instance = ShapedObject::new(class.class_type_id(), class.instance_shape().clone());

    let (st_atime, st_atime_ns) = system_time_parts(metadata.accessed());
    let (st_mtime, st_mtime_ns) = system_time_parts(metadata.modified());
    let (st_birthtime, st_birthtime_ns) = system_time_parts(metadata.created());

    let st_ctime = st_birthtime;
    let st_ctime_ns = st_birthtime_ns;
    let file_attributes = metadata_file_attributes(metadata);
    let reparse_tag = if path.is_symlink() {
        metadata_reparse_tag(metadata)
    } else {
        0
    };

    let field_values = [
        (
            "st_mode",
            Value::int(file_type_mode_bits(metadata)).expect("st_mode fits"),
        ),
        ("st_ino", Value::int(0).expect("st_ino fits")),
        ("st_dev", Value::int(0).expect("st_dev fits")),
        ("st_nlink", Value::int(0).expect("st_nlink fits")),
        ("st_uid", Value::int(0).expect("st_uid fits")),
        ("st_gid", Value::int(0).expect("st_gid fits")),
        (
            "st_size",
            Value::int(i64::try_from(metadata.len()).unwrap_or(i64::MAX)).expect("st_size fits"),
        ),
        ("st_atime", Value::float(st_atime)),
        ("st_mtime", Value::float(st_mtime)),
        ("st_ctime", Value::float(st_ctime)),
        ("st_atime_ns", bigint_to_value(BigInt::from(st_atime_ns))),
        ("st_mtime_ns", bigint_to_value(BigInt::from(st_mtime_ns))),
        ("st_ctime_ns", bigint_to_value(BigInt::from(st_ctime_ns))),
        ("st_birthtime", Value::float(st_birthtime)),
        (
            "st_birthtime_ns",
            bigint_to_value(BigInt::from(st_birthtime_ns)),
        ),
        (
            "st_file_attributes",
            Value::int(file_attributes).expect("st_file_attributes fits"),
        ),
        (
            "st_reparse_tag",
            Value::int(reparse_tag).expect("st_reparse_tag fits"),
        ),
    ];

    for (name, value) in field_values {
        instance.set_property(intern(name), value, registry);
    }

    leak_object_value(instance)
}

fn build_terminal_size_value(columns: i64, lines: i64) -> Value {
    let registry = shape_registry();
    let mut terminal_size = Box::new(ShapedObject::with_empty_shape(registry.empty_shape()));
    terminal_size.set_property(
        intern("columns"),
        Value::int(columns).expect("terminal_size.columns should fit"),
        registry,
    );
    terminal_size.set_property(
        intern("lines"),
        Value::int(lines).expect("terminal_size.lines should fit"),
        registry,
    );
    Value::object_ptr(Box::into_raw(terminal_size) as *const ())
}

fn terminal_size_components_from_value(
    value: Value,
    fn_name: &str,
) -> Result<(i64, i64), BuiltinError> {
    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "{fn_name}() argument must be a sequence of length 2, not {}",
            value.type_name()
        ))
    })?;

    if crate::ops::objects::extract_type_id(ptr) == TypeId::TUPLE {
        let tuple = unsafe { &*(ptr as *const TupleObject) };
        if tuple.len() != 2 {
            return Err(BuiltinError::TypeError(format!(
                "{fn_name}() takes a 2-sequence ({}-sequence given)",
                tuple.len()
            )));
        }
        return Ok((
            terminal_size_component(tuple.get(0).expect("tuple should contain first item"))?,
            terminal_size_component(tuple.get(1).expect("tuple should contain second item"))?,
        ));
    }

    if crate::ops::objects::extract_type_id(ptr) == TypeId::LIST {
        let list = unsafe { &*(ptr as *const ListObject) };
        if list.len() != 2 {
            return Err(BuiltinError::TypeError(format!(
                "{fn_name}() takes a 2-sequence ({}-sequence given)",
                list.len()
            )));
        }
        return Ok((
            terminal_size_component(list.get(0).expect("list should contain first item"))?,
            terminal_size_component(list.get(1).expect("list should contain second item"))?,
        ));
    }

    Err(BuiltinError::TypeError(format!(
        "{fn_name}() argument must be a tuple or list, not {}",
        value.type_name()
    )))
}

fn terminal_size_component(value: Value) -> Result<i64, BuiltinError> {
    if let Some(integer) = value.as_int() {
        return Ok(integer);
    }
    if let Some(boolean) = value.as_bool() {
        return Ok(i64::from(boolean));
    }

    Err(BuiltinError::TypeError(format!(
        "terminal_size() elements must be integers, not {}",
        value.type_name()
    )))
}

fn stat_impl<F>(args: &[Value], fn_name: &str, stat_fn: F) -> Result<Value, BuiltinError>
where
    F: Fn(&Path) -> std::io::Result<Metadata>,
{
    let path = parse_single_path_arg(args, fn_name)?;
    let metadata = stat_fn(&path).map_err(|err| {
        BuiltinError::OSError(format!(
            "{}() failed for '{}': {}",
            fn_name,
            path.display(),
            err
        ))
    })?;
    Ok(build_stat_result_value(path.as_path(), &metadata))
}

fn not_implemented(function_name: &str, _args: &[Value]) -> Result<Value, BuiltinError> {
    Err(BuiltinError::NotImplemented(format!(
        "{} is not implemented yet",
        function_name
    )))
}

fn integer_arg(value: Value, fn_name: &str, parameter: &str) -> Result<i64, BuiltinError> {
    value
        .as_int()
        .or_else(|| value.as_bool().map(i64::from))
        .ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "{fn_name}() {parameter} must be an integer, not {}",
                value.type_name()
            ))
        })
}

fn configure_open_options(flags: i64, mode: i64) -> Result<OpenOptions, BuiltinError> {
    let flags = u32::try_from(flags).map_err(|_| {
        BuiltinError::OverflowError("open() flags out of range for platform integer".to_string())
    })?;
    let mode = u32::try_from(mode).map_err(|_| {
        BuiltinError::OverflowError("open() mode out of range for platform integer".to_string())
    })?;

    let mut options = OpenOptions::new();
    match flags & 0x3 {
        O_RDONLY => {
            options.read(true);
        }
        O_WRONLY => {
            options.write(true);
        }
        O_RDWR => {
            options.read(true).write(true);
        }
        _ => {
            return Err(BuiltinError::ValueError(
                "open() invalid access mode in flags".to_string(),
            ));
        }
    }

    if flags & O_APPEND != 0 {
        options.append(true).write(true);
    }
    if flags & O_TRUNC != 0 {
        options.truncate(true).write(true);
    }
    if flags & O_CREAT != 0 && flags & O_EXCL != 0 {
        options.create_new(true);
    } else if flags & O_CREAT != 0 {
        options.create(true);
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(mode);
    }

    let _ = mode;
    Ok(options)
}

#[cfg(windows)]
#[link(name = "ucrt")]
unsafe extern "C" {
    #[link_name = "_open_osfhandle"]
    fn crt_open_osfhandle(osfhandle: isize, flags: i32) -> i32;
    #[link_name = "_close"]
    fn crt_close(fd: i32) -> i32;
    #[link_name = "_pipe"]
    fn crt_pipe(fds: *mut i32, size: u32, flags: i32) -> i32;
    #[link_name = "_read"]
    fn crt_read(fd: i32, buffer: *mut core::ffi::c_void, count: u32) -> i32;
    #[link_name = "_write"]
    fn crt_write(fd: i32, buffer: *const core::ffi::c_void, count: u32) -> i32;
}

#[cfg(windows)]
fn file_to_fd(file: std::fs::File, flags: i64) -> Result<i32, std::io::Error> {
    use std::os::windows::io::IntoRawHandle;
    use windows_sys::Win32::Foundation::CloseHandle;

    let handle = file.into_raw_handle();
    let fd = unsafe { crt_open_osfhandle(handle as isize, flags as i32) };
    if fd == -1 {
        let error = std::io::Error::last_os_error();
        unsafe {
            CloseHandle(handle);
        }
        return Err(error);
    }
    Ok(fd)
}

#[cfg(unix)]
fn file_to_fd(file: std::fs::File, _flags: i64) -> Result<i32, std::io::Error> {
    use std::os::fd::IntoRawFd;
    Ok(file.into_raw_fd())
}

fn close_fd(fd: i32) -> Result<(), std::io::Error> {
    #[cfg(windows)]
    {
        if unsafe { crt_close(fd) } == -1 {
            return Err(std::io::Error::last_os_error());
        }
        return Ok(());
    }

    #[cfg(unix)]
    {
        if unsafe { libc::close(fd) } == -1 {
            return Err(std::io::Error::last_os_error());
        }
        return Ok(());
    }

    #[allow(unreachable_code)]
    Ok(())
}

fn pipe_fds() -> Result<(i32, i32), std::io::Error> {
    #[cfg(windows)]
    {
        let mut fds = [0_i32; 2];
        if unsafe { crt_pipe(fds.as_mut_ptr(), 0, O_BINARY as i32) } == -1 {
            return Err(std::io::Error::last_os_error());
        }
        return Ok((fds[0], fds[1]));
    }

    #[cfg(unix)]
    {
        let mut fds = [0_i32; 2];
        if unsafe { libc::pipe(fds.as_mut_ptr()) } == -1 {
            return Err(std::io::Error::last_os_error());
        }
        return Ok((fds[0], fds[1]));
    }

    #[allow(unreachable_code)]
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "pipe is not supported on this platform",
    ))
}

fn read_fd(fd: i32, buffer: &mut [u8]) -> Result<usize, std::io::Error> {
    let count = u32::try_from(buffer.len())
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "read too large"))?;

    #[cfg(windows)]
    {
        let read = unsafe { crt_read(fd, buffer.as_mut_ptr().cast(), count) };
        if read == -1 {
            return Err(std::io::Error::last_os_error());
        }
        return Ok(read as usize);
    }

    #[cfg(unix)]
    {
        let read = unsafe { libc::read(fd, buffer.as_mut_ptr().cast(), count as usize) };
        if read == -1 {
            return Err(std::io::Error::last_os_error());
        }
        return Ok(read as usize);
    }

    #[allow(unreachable_code)]
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "read is not supported on this platform",
    ))
}

fn write_fd(fd: i32, buffer: &[u8]) -> Result<usize, std::io::Error> {
    let count = u32::try_from(buffer.len())
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "write too large"))?;

    #[cfg(windows)]
    {
        let written = unsafe { crt_write(fd, buffer.as_ptr().cast(), count) };
        if written == -1 {
            return Err(std::io::Error::last_os_error());
        }
        return Ok(written as usize);
    }

    #[cfg(unix)]
    {
        let written = unsafe { libc::write(fd, buffer.as_ptr().cast(), count as usize) };
        if written == -1 {
            return Err(std::io::Error::last_os_error());
        }
        return Ok(written as usize);
    }

    #[allow(unreachable_code)]
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "write is not supported on this platform",
    ))
}

fn nt_open(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "open() takes 2 or 3 positional arguments but {} were given",
            args.len()
        )));
    }

    let path = path_from_value(args[0], "open", "path")?;
    let flags = integer_arg(args[1], "open", "flags")?;
    let mode = args
        .get(2)
        .map(|value| integer_arg(*value, "open", "mode"))
        .transpose()?
        .unwrap_or(0o777);

    let file = configure_open_options(flags, mode)?
        .open(&path)
        .map_err(|err| {
            BuiltinError::OSError(format!("open() failed for '{}': {}", path.display(), err))
        })?;
    let fd = file_to_fd(file, flags).map_err(|err| {
        BuiltinError::OSError(format!(
            "open() failed to create file descriptor for '{}': {}",
            path.display(),
            err
        ))
    })?;

    Ok(Value::int(i64::from(fd)).expect("file descriptor should fit in Prism int"))
}

fn nt_close(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "close() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let fd = i32::try_from(integer_arg(args[0], "close", "fd")?)
        .map_err(|_| BuiltinError::OverflowError("close() file descriptor out of range".into()))?;
    close_fd(fd).map_err(|err| BuiltinError::OSError(format!("close() failed for {fd}: {err}")))?;
    Ok(Value::none())
}

fn nt_pipe(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "pipe() takes no arguments ({} given)",
            args.len()
        )));
    }

    let (read_fd, write_fd) =
        pipe_fds().map_err(|err| BuiltinError::OSError(format!("pipe() failed: {err}")))?;
    Ok(leak_object_value(TupleObject::from_slice(&[
        Value::int(i64::from(read_fd)).expect("read fd should fit in Prism int"),
        Value::int(i64::from(write_fd)).expect("write fd should fit in Prism int"),
    ])))
}

fn nt_read(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "read() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    let fd = i32::try_from(integer_arg(args[0], "read", "fd")?)
        .map_err(|_| BuiltinError::OverflowError("read() file descriptor out of range".into()))?;
    let size = integer_arg(args[1], "read", "length")?;
    if size < 0 {
        return Err(BuiltinError::ValueError(
            "read length must be non-negative".to_string(),
        ));
    }
    let size = usize::try_from(size)
        .map_err(|_| BuiltinError::OverflowError("read length out of range".into()))?;

    let mut buffer = vec![0_u8; size];
    let bytes_read = crate::threading_runtime::blocking_operation(|| read_fd(fd, &mut buffer))
        .map_err(|err| BuiltinError::OSError(format!("read() failed for {fd}: {err}")))?;
    buffer.truncate(bytes_read);
    Ok(leak_object_value(BytesObject::from_slice(&buffer)))
}

fn nt_write(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "write() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    let fd = i32::try_from(integer_arg(args[0], "write", "fd")?)
        .map_err(|_| BuiltinError::OverflowError("write() file descriptor out of range".into()))?;
    let buffer = value_to_bytes(args[1]).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "write() argument 2 must be bytes-like, not {}",
            args[1].type_name()
        ))
    })?;
    let bytes_written = crate::threading_runtime::blocking_operation(|| write_fd(fd, &buffer))
        .map_err(|err| BuiltinError::OSError(format!("write() failed for {fd}: {err}")))?;
    Ok(
        Value::int(i64::try_from(bytes_written).expect("written byte count should fit"))
            .expect("written byte count should fit in Prism int"),
    )
}

fn nt_access(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "access() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    let path = path_from_value(args[0], "access", "path")?;
    let mode = args[1]
        .as_int()
        .or_else(|| args[1].as_bool().map(i64::from))
        .ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "access() mode must be an integer, not {}",
                args[1].type_name()
            ))
        })?;

    let exists = path.exists();
    let writable = std::fs::metadata(&path)
        .map(|metadata| !metadata.permissions().readonly())
        .unwrap_or(false);
    let executable = exists;

    let allowed = if mode == 0 {
        exists
    } else {
        let mut result = true;
        if mode & 4 != 0 {
            result &= exists;
        }
        if mode & 2 != 0 {
            result &= writable;
        }
        if mode & 1 != 0 {
            result &= executable;
        }
        result
    };

    Ok(Value::bool(allowed))
}

fn nt_stat(args: &[Value]) -> Result<Value, BuiltinError> {
    stat_impl(args, "nt.stat", |path| std::fs::metadata(path))
}

fn nt_lstat(args: &[Value]) -> Result<Value, BuiltinError> {
    stat_impl(args, "nt.lstat", |path| std::fs::symlink_metadata(path))
}

fn nt_listdir(args: &[Value]) -> Result<Value, BuiltinError> {
    let parsed = parse_optional_path_arg(args, "listdir")?;
    let names = super::os::listdir(&parsed.path).map_err(|err| {
        BuiltinError::OSError(format!(
            "listdir() failed for '{}': {}",
            parsed.path.display(),
            err
        ))
    })?;

    let mut list = ListObject::with_capacity(names.len());
    match parsed.flavor {
        PathValueFlavor::String => {
            for name in names {
                list.push(Value::string(intern(&name)));
            }
        }
        PathValueFlavor::Bytes => {
            for name in names {
                list.push(leak_object_value(BytesObject::from_slice(name.as_bytes())));
            }
        }
    }

    Ok(leak_object_value(list))
}

fn nt_scandir(args: &[Value]) -> Result<Value, BuiltinError> {
    not_implemented("nt.scandir()", args)
}

fn nt_getpid(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "getpid() takes 0 positional arguments but {} were given",
            args.len()
        )));
    }

    Ok(Value::int(std::process::id() as i64).expect("process id fits in i64"))
}

fn nt_cpu_count(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "cpu_count() takes 0 positional arguments but {} were given",
            args.len()
        )));
    }

    match std::thread::available_parallelism() {
        Ok(count) => Ok(Value::int(count.get() as i64).expect("CPU count fits in i64")),
        Err(_) => Ok(Value::none()),
    }
}

fn nt_urandom(args: &[Value]) -> Result<Value, BuiltinError> {
    urandom_value_from_args(args, "urandom")
}

fn validate_env_key(key: &str, fn_name: &str) -> Result<(), BuiltinError> {
    if key.is_empty() {
        return Err(BuiltinError::ValueError(format!(
            "{fn_name}() environment variable name cannot be empty"
        )));
    }
    if key.contains('=') {
        return Err(BuiltinError::ValueError(format!(
            "{fn_name}() environment variable name cannot contain '='"
        )));
    }
    if key.contains('\0') {
        return Err(BuiltinError::ValueError(format!(
            "{fn_name}() embedded null character in environment variable name"
        )));
    }
    Ok(())
}

fn validate_env_value(value: &str, fn_name: &str) -> Result<(), BuiltinError> {
    if value.contains('\0') {
        return Err(BuiltinError::ValueError(format!(
            "{fn_name}() embedded null character in environment variable value"
        )));
    }
    Ok(())
}

fn nt_putenv(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "putenv() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    let key = value_to_string(args[0])
        .ok_or_else(|| BuiltinError::TypeError("putenv() key must be str".to_string()))?;
    let value = value_to_string(args[1])
        .ok_or_else(|| BuiltinError::TypeError("putenv() value must be str".to_string()))?;
    validate_env_key(&key, "putenv")?;
    validate_env_value(&value, "putenv")?;

    unsafe {
        std::env::set_var(key, value);
    }
    Ok(Value::none())
}

fn nt_unsetenv(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "unsetenv() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }

    let key = value_to_string(args[0])
        .ok_or_else(|| BuiltinError::TypeError("unsetenv() key must be str".to_string()))?;
    validate_env_key(&key, "unsetenv")?;

    unsafe {
        std::env::remove_var(key);
    }
    Ok(Value::none())
}

fn remove_impl(args: &[Value], fn_name: &str) -> Result<Value, BuiltinError> {
    let path = parse_single_path_arg(args, fn_name)?;
    std::fs::remove_file(&path).map_err(|err| {
        BuiltinError::OSError(format!(
            "{}() failed for '{}': {}",
            fn_name,
            path.display(),
            err
        ))
    })?;
    Ok(Value::none())
}

fn nt_mkdir(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "mkdir() takes 1 or 2 positional arguments but {} were given",
            args.len()
        )));
    }

    let path = path_from_value(args[0], "mkdir", "path")?;
    if let Some(mode) = args.get(1)
        && mode.as_int().is_none()
        && mode.as_bool().is_none()
    {
        return Err(BuiltinError::TypeError(format!(
            "mkdir() integer mode expected, not {}",
            mode.type_name()
        )));
    }

    std::fs::create_dir(&path).map_err(|err| {
        BuiltinError::OSError(format!("mkdir() failed for '{}': {}", path.display(), err))
    })?;
    Ok(Value::none())
}

fn nt_remove(args: &[Value]) -> Result<Value, BuiltinError> {
    remove_impl(args, "remove")
}

fn rename_impl(args: &[Value], fn_name: &str) -> Result<Value, BuiltinError> {
    let (src, dst) = parse_two_path_args(args, fn_name)?;
    std::fs::rename(&src, &dst).map_err(|err| {
        BuiltinError::OSError(format!(
            "{fn_name}() failed from '{}' to '{}': {}",
            src.display(),
            dst.display(),
            err
        ))
    })?;
    Ok(Value::none())
}

fn nt_rename(args: &[Value]) -> Result<Value, BuiltinError> {
    rename_impl(args, "rename")
}

fn nt_replace(args: &[Value]) -> Result<Value, BuiltinError> {
    rename_impl(args, "replace")
}

fn nt_rmdir(args: &[Value]) -> Result<Value, BuiltinError> {
    let path = parse_single_path_arg(args, "rmdir")?;
    std::fs::remove_dir(&path).map_err(|err| {
        BuiltinError::OSError(format!("rmdir() failed for '{}': {}", path.display(), err))
    })?;
    Ok(Value::none())
}

fn nt_unlink(args: &[Value]) -> Result<Value, BuiltinError> {
    remove_impl(args, "unlink")
}

fn cwd_string() -> Result<String, BuiltinError> {
    super::os::getcwd()
        .map(|cwd| cwd.as_ref().to_owned())
        .map_err(|err| BuiltinError::ValueError(format!("failed to get current directory: {err}")))
}

fn nt_getcwd(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "getcwd() takes 0 positional arguments but {} were given",
            args.len()
        )));
    }

    Ok(Value::string(intern(&cwd_string()?)))
}

fn nt_getcwdb(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "getcwdb() takes 0 positional arguments but {} were given",
            args.len()
        )));
    }

    let cwd = cwd_string()?;
    Ok(leak_object_value(BytesObject::from_slice(cwd.as_bytes())))
}

fn nt_terminal_size(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "terminal_size() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }

    let (columns, lines) = terminal_size_components_from_value(args[0], "terminal_size")?;
    Ok(build_terminal_size_value(columns, lines))
}

fn nt_get_terminal_size(args: &[Value]) -> Result<Value, BuiltinError> {
    let fd = match args {
        [] => 1,
        [value] => value.as_int().ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "get_terminal_size() argument must be an integer, not {}",
                value.type_name()
            ))
        })?,
        _ => {
            return Err(BuiltinError::TypeError(format!(
                "get_terminal_size() takes at most 1 argument ({} given)",
                args.len()
            )));
        }
    };

    #[cfg(windows)]
    {
        use std::mem::MaybeUninit;
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Console::{
            CONSOLE_SCREEN_BUFFER_INFO, GetConsoleScreenBufferInfo, GetStdHandle, STD_ERROR_HANDLE,
            STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
        };

        let std_handle = match fd {
            0 => STD_INPUT_HANDLE,
            1 => STD_OUTPUT_HANDLE,
            2 => STD_ERROR_HANDLE,
            _ => return Err(BuiltinError::OSError(format!("bad file descriptor: {fd}"))),
        };

        let handle = unsafe { GetStdHandle(std_handle) };
        if handle.is_null() || handle == INVALID_HANDLE_VALUE as *mut core::ffi::c_void {
            return Err(BuiltinError::OSError(format!(
                "failed to access console handle for file descriptor {fd}: {}",
                std::io::Error::last_os_error()
            )));
        }

        let mut info = MaybeUninit::<CONSOLE_SCREEN_BUFFER_INFO>::zeroed();
        if unsafe { GetConsoleScreenBufferInfo(handle, info.as_mut_ptr()) } == 0 {
            return Err(BuiltinError::OSError(format!(
                "failed to query console size for file descriptor {fd}: {}",
                std::io::Error::last_os_error()
            )));
        }

        let info = unsafe { info.assume_init() };
        let columns = i64::from(info.srWindow.Right - info.srWindow.Left + 1);
        let lines = i64::from(info.srWindow.Bottom - info.srWindow.Top + 1);
        return Ok(build_terminal_size_value(columns, lines));
    }

    #[allow(unreachable_code)]
    Err(BuiltinError::OSError(
        "get_terminal_size() is only available on Windows builds".to_string(),
    ))
}

fn nt_exit(args: &[Value]) -> Result<Value, BuiltinError> {
    not_implemented("nt._exit()", args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_runtime::object::shaped_object::ShapedObject;
    use prism_runtime::types::dict::DictObject;
    use prism_runtime::types::int::value_to_bigint;
    use prism_runtime::types::set::SetObject;
    use prism_runtime::types::tuple::TupleObject;
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

        let result = nt_unsetenv(&[Value::string(intern(&key))])
            .expect("unsetenv should accept a valid key");
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
        let tuple_ptr = match_args
            .as_object_ptr()
            .expect("__match_args__ should be a tuple");
        let tuple = unsafe { &*(tuple_ptr as *const TupleObject) };
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
}
