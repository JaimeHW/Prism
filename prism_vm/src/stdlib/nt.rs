//! Native Windows `nt` module bootstrap surface.
//!
//! CPython's `os.py` selects the platform backend by consulting
//! `sys.builtin_module_names` and then importing either `posix` or `nt`.
//! Prism currently relies on CPython's pure-Python `os.py` when a CPython stdlib
//! tree is available, so the runtime must provide a coherent `nt` module during
//! import bootstrap on Windows.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
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
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};
use std::time::UNIX_EPOCH;

static NT_OPEN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.open"), nt_open));
static NT_STAT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.stat"), nt_stat));
static NT_LSTAT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.lstat"), nt_lstat));
static NT_LISTDIR_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.listdir"), nt_listdir));
static NT_SCANDIR_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.scandir"), nt_scandir));
static NT_GETCWD_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.getcwd"), nt_getcwd));
static NT_GETCWDB_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt.getcwdb"), nt_getcwdb));
static NT_EXIT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("nt._exit"), nt_exit));
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
                Arc::from("open"),
                Arc::from("stat"),
                Arc::from("lstat"),
                Arc::from("listdir"),
                Arc::from("scandir"),
                Arc::from("getcwd"),
                Arc::from("getcwdb"),
                Arc::from("stat_result"),
                Arc::from("environ"),
                Arc::from("_have_functions"),
                Arc::from("_exit"),
                Arc::from("SEEK_SET"),
                Arc::from("SEEK_CUR"),
                Arc::from("SEEK_END"),
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
            "open" => Ok(builtin_value(&NT_OPEN_FUNCTION)),
            "stat" => Ok(builtin_value(&NT_STAT_FUNCTION)),
            "lstat" => Ok(builtin_value(&NT_LSTAT_FUNCTION)),
            "listdir" => Ok(builtin_value(&NT_LISTDIR_FUNCTION)),
            "scandir" => Ok(builtin_value(&NT_SCANDIR_FUNCTION)),
            "getcwd" => Ok(builtin_value(&NT_GETCWD_FUNCTION)),
            "getcwdb" => Ok(builtin_value(&NT_GETCWDB_FUNCTION)),
            "stat_result" => Ok(stat_result_class_value()),
            "environ" => Ok(self.environ_value),
            "_have_functions" => Ok(self.have_functions_value),
            "_exit" => Ok(builtin_value(&NT_EXIT_FUNCTION)),
            "SEEK_SET" => Ok(Value::int(0).expect("SEEK_SET fits in i64")),
            "SEEK_CUR" => Ok(Value::int(1).expect("SEEK_CUR fits in i64")),
            "SEEK_END" => Ok(Value::int(2).expect("SEEK_END fits in i64")),
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
fn leak_object_value<T>(object: T) -> Value {
    let ptr = Box::into_raw(Box::new(object)) as *const ();
    Value::object_ptr(ptr)
}

fn export_names_value() -> Value {
    let names = [
        "open",
        "stat",
        "lstat",
        "listdir",
        "scandir",
        "getcwd",
        "getcwdb",
        "stat_result",
        "environ",
        "SEEK_SET",
        "SEEK_CUR",
        "SEEK_END",
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

fn stat_impl<F>(args: &[Value], fn_name: &str, stat_fn: F) -> Result<Value, BuiltinError>
where
    F: Fn(&Path) -> std::io::Result<Metadata>,
{
    let path = parse_single_path_arg(args, fn_name)?;
    let metadata = stat_fn(&path).map_err(|err| {
        BuiltinError::ValueError(format!(
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

fn nt_open(args: &[Value]) -> Result<Value, BuiltinError> {
    not_implemented("nt.open()", args)
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
        assert!(module.get_attr("open").is_ok());
        assert!(module.get_attr("stat").is_ok());
        assert!(module.get_attr("lstat").is_ok());
        assert!(module.get_attr("listdir").is_ok());
        assert!(module.get_attr("scandir").is_ok());
        assert!(module.get_attr("getcwd").is_ok());
        assert!(module.get_attr("getcwdb").is_ok());
        assert!(module.get_attr("stat_result").is_ok());
        assert!(module.get_attr("environ").is_ok());
        assert!(module.get_attr("_have_functions").is_ok());
        assert!(module.get_attr("_exit").is_ok());
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
                .any(|value| value == &Value::string(intern("getcwd")))
        );
        assert!(
            tuple
                .iter()
                .any(|value| value == &Value::string(intern("environ")))
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
