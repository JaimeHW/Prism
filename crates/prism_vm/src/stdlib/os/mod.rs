//! Python `os` module implementation.
//!
//! High-performance implementation of Python's os module providing:
//! - Platform-specific constants (name, sep, pathsep)
//! - Environment variable access with lazy loading
//! - Current directory operations with thread-local caching
//! - Path query operations with branch-free stat
//! - File system operations (mkdir, rmdir, remove, rename)
//! - Process information (getpid, getppid)
//!
//! # Performance Characteristics
//!
//! - Zero heap allocation for path operations (stack buffers)
//! - Thread-local cached current directory
//! - Lazy environment variable loading
//! - Direct OS syscalls without abstraction overhead

mod constants;
mod cwd;
mod environ;
mod error;
mod file;
pub mod path;
mod process;

pub use constants::*;
pub use cwd::*;
pub use environ::*;
pub use error::*;
pub use file::*;
pub use process::*;

use super::{Module, ModuleError};
use crate::VirtualMachine;
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::ops::calls::invoke_callable_value;
use crate::ops::method_dispatch::load_method::resolve_special_method;
use crate::ops::objects::extract_type_id;
use crate::stdlib::secure_random::urandom_value_from_args;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::type_builtins::global_class;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::value_as_bytes_ref;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::string::value_as_string_ref;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock};

static OS_URANDOM_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.urandom"), os_urandom));
static OS_FSPATH_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("os.fspath"), os_fspath));
static OS_REMOVE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.remove"), os_remove));
static OS_UNLINK_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.unlink"), os_unlink));
static OS_GETCWD_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.getcwd"), os_getcwd));
static OS_GETPID_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.getpid"), os_getpid));
static OS_GETPPID_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.getppid"), os_getppid));
static OS_GETENV_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.getenv"), os_getenv));
static OS_PUTENV_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.putenv"), os_putenv));
static OS_UNSETENV_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.unsetenv"), os_unsetenv));

/// The os module providing operating system interface.
pub struct OsModule {
    /// Cached environment (lazy-loaded)
    environ: Environ,
}

impl OsModule {
    /// Create a new os module.
    #[inline]
    pub fn new() -> Self {
        Self {
            environ: Environ::new(),
        }
    }

    /// Get the environment dictionary.
    #[inline]
    pub fn environ(&self) -> &Environ {
        &self.environ
    }

    /// Get mutable environment dictionary.
    #[inline]
    pub fn environ_mut(&mut self) -> &mut Environ {
        &mut self.environ
    }
}

impl Default for OsModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for OsModule {
    fn name(&self) -> &str {
        "os"
    }

    fn get_attr(&self, name: &str) -> Result<Value, ModuleError> {
        match name {
            // Platform constants
            "name" => Ok(string_value(OS_NAME)),
            "sep" => Ok(string_value(SEP_STR)),
            "pathsep" => Ok(string_value(PATHSEP_STR)),
            "linesep" => Ok(string_value(LINESEP)),
            "curdir" => Ok(string_value(CURDIR)),
            "pardir" => Ok(string_value(PARDIR)),
            "extsep" => Ok(string_value(".")),
            "altsep" => Ok(ALTSEP
                .map(|separator| string_value(&separator.to_string()))
                .unwrap_or_else(Value::none)),
            "devnull" => Ok(string_value(DEVNULL)),

            // Functions
            "remove" => Ok(builtin_value(&OS_REMOVE_FUNCTION)),
            "unlink" => Ok(builtin_value(&OS_UNLINK_FUNCTION)),
            "getcwd" => Ok(builtin_value(&OS_GETCWD_FUNCTION)),
            "getpid" => Ok(builtin_value(&OS_GETPID_FUNCTION)),
            "getppid" => Ok(builtin_value(&OS_GETPPID_FUNCTION)),
            "getenv" => Ok(builtin_value(&OS_GETENV_FUNCTION)),
            "putenv" => Ok(builtin_value(&OS_PUTENV_FUNCTION)),
            "unsetenv" => Ok(builtin_value(&OS_UNSETENV_FUNCTION)),
            "chdir" | "mkdir" | "makedirs" | "rmdir" | "removedirs" | "rename" | "replace"
            | "stat" | "lstat" | "listdir" | "scandir" | "walk" | "fwalk" | "kill" | "system"
            | "popen" | "access" | "chmod" | "chown" | "link" | "symlink" | "readlink" => {
                Ok(Value::none()) // Placeholder for callable
            }
            "urandom" => Ok(builtin_value(&OS_URANDOM_FUNCTION)),
            "fspath" => Ok(builtin_value(&OS_FSPATH_FUNCTION)),

            // Submodule
            "path" => Ok(Value::none()), // TODO: Return os.path module

            // Environ dict
            "environ" => Ok(environ_dict_value()),

            // O_* flags as integers
            "O_RDONLY" => Ok(Value::int(O_RDONLY as i64).unwrap()),
            "O_WRONLY" => Ok(Value::int(O_WRONLY as i64).unwrap()),
            "O_RDWR" => Ok(Value::int(O_RDWR as i64).unwrap()),
            "O_CREAT" => Ok(Value::int(O_CREAT as i64).unwrap()),
            "O_TRUNC" => Ok(Value::int(O_TRUNC as i64).unwrap()),
            "O_APPEND" => Ok(Value::int(O_APPEND as i64).unwrap()),
            "O_EXCL" => Ok(Value::int(O_EXCL as i64).unwrap()),

            _ => Err(ModuleError::AttributeError(format!(
                "module 'os' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        vec![
            // Constants
            Arc::from("name"),
            Arc::from("sep"),
            Arc::from("pathsep"),
            Arc::from("linesep"),
            Arc::from("curdir"),
            Arc::from("pardir"),
            Arc::from("extsep"),
            Arc::from("altsep"),
            Arc::from("devnull"),
            // Submodule
            Arc::from("path"),
            // Environ
            Arc::from("environ"),
            // Directory operations
            Arc::from("getcwd"),
            Arc::from("chdir"),
            Arc::from("mkdir"),
            Arc::from("makedirs"),
            Arc::from("rmdir"),
            Arc::from("removedirs"),
            Arc::from("listdir"),
            Arc::from("scandir"),
            Arc::from("walk"),
            // File operations
            Arc::from("remove"),
            Arc::from("unlink"),
            Arc::from("rename"),
            Arc::from("replace"),
            Arc::from("stat"),
            Arc::from("lstat"),
            Arc::from("access"),
            Arc::from("chmod"),
            Arc::from("link"),
            Arc::from("symlink"),
            Arc::from("readlink"),
            // Process
            Arc::from("getpid"),
            Arc::from("getppid"),
            Arc::from("kill"),
            Arc::from("system"),
            Arc::from("popen"),
            // Environment
            Arc::from("getenv"),
            Arc::from("putenv"),
            Arc::from("unsetenv"),
            // Misc
            Arc::from("urandom"),
            Arc::from("fspath"),
            // Flags
            Arc::from("O_RDONLY"),
            Arc::from("O_WRONLY"),
            Arc::from("O_RDWR"),
            Arc::from("O_CREAT"),
            Arc::from("O_TRUNC"),
            Arc::from("O_APPEND"),
            Arc::from("O_EXCL"),
        ]
    }
}

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

#[inline]
fn string_value(value: &str) -> Value {
    Value::string(intern(value))
}

fn environ_dict_value() -> Value {
    let vars = std::env::vars().collect::<Vec<_>>();
    let mut dict = DictObject::with_capacity(vars.len());
    for (key, value) in vars {
        dict.set(string_value(&key), string_value(&value));
    }
    crate::alloc_managed_value(dict)
}

fn os_urandom(args: &[Value]) -> Result<Value, BuiltinError> {
    urandom_value_from_args(args, "urandom")
}

fn os_remove(args: &[Value]) -> Result<Value, BuiltinError> {
    remove_path(args, "remove")
}

fn os_unlink(args: &[Value]) -> Result<Value, BuiltinError> {
    remove_path(args, "unlink")
}

fn os_getcwd(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "getcwd() takes no arguments ({} given)",
            args.len()
        )));
    }

    let cwd = getcwd().map_err(|err| BuiltinError::OSError(format!("getcwd() failed: {err}")))?;
    Ok(Value::string(intern(cwd.as_ref())))
}

fn os_getpid(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "getpid() takes no arguments ({} given)",
            args.len()
        )));
    }

    Ok(Value::int(getpid() as i64).expect("process id fits in i64"))
}

fn os_getppid(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "getppid() takes no arguments ({} given)",
            args.len()
        )));
    }

    Ok(Value::int(getppid() as i64).expect("parent process id fits in i64"))
}

fn os_getenv(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "getenv() takes from 1 to 2 positional arguments but {} were given",
            args.len()
        )));
    }

    let key = value_as_string_ref(args[0])
        .ok_or_else(|| BuiltinError::TypeError("getenv() key must be str".to_string()))?;
    validate_env_key(key.as_str(), "getenv")?;

    match environ::getenv(key.as_str()) {
        Some(value) => Ok(Value::string(intern(&value))),
        None if args.len() == 2 => Ok(args[1]),
        None => Ok(Value::none()),
    }
}

fn os_putenv(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "putenv() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    let key = value_as_string_ref(args[0])
        .ok_or_else(|| BuiltinError::TypeError("putenv() key must be str".to_string()))?;
    let value = value_as_string_ref(args[1])
        .ok_or_else(|| BuiltinError::TypeError("putenv() value must be str".to_string()))?;
    validate_env_key(key.as_str(), "putenv")?;
    validate_env_value(value.as_str(), "putenv")?;

    environ::putenv(key.as_str(), value.as_str());
    Ok(Value::none())
}

fn os_unsetenv(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "unsetenv() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }

    let key = value_as_string_ref(args[0])
        .ok_or_else(|| BuiltinError::TypeError("unsetenv() key must be str".to_string()))?;
    validate_env_key(key.as_str(), "unsetenv")?;

    environ::unsetenv(key.as_str());
    Ok(Value::none())
}

fn validate_env_key(key: &str, function: &str) -> Result<(), BuiltinError> {
    if key.is_empty() {
        return Err(BuiltinError::ValueError(format!(
            "{function}() environment variable name cannot be empty"
        )));
    }
    if key.contains('=') {
        return Err(BuiltinError::ValueError(format!(
            "{function}() environment variable name cannot contain '='"
        )));
    }
    if key.contains('\0') {
        return Err(BuiltinError::ValueError(format!(
            "{function}() embedded null character in environment variable name"
        )));
    }
    Ok(())
}

fn validate_env_value(value: &str, function: &str) -> Result<(), BuiltinError> {
    if value.contains('\0') {
        return Err(BuiltinError::ValueError(format!(
            "{function}() embedded null character in environment variable value"
        )));
    }
    Ok(())
}

fn remove_path(args: &[Value], function: &str) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "{function}() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let path = path_from_value(args[0], function)?;
    std::fs::remove_file(&path).map_err(|err| {
        BuiltinError::OSError(format!(
            "{function}() failed for '{}': {}",
            path.display(),
            err
        ))
    })?;
    Ok(Value::none())
}

fn path_from_value(value: Value, function: &str) -> Result<PathBuf, BuiltinError> {
    if let Some(path) = value_as_string_ref(value) {
        return Ok(PathBuf::from(path.as_str()));
    }
    if let Some(bytes) = value_as_bytes_ref(value) {
        let path = std::str::from_utf8(bytes.as_bytes()).map_err(|_| {
            BuiltinError::ValueError(format!("{function}() path bytes are not valid UTF-8"))
        })?;
        return Ok(PathBuf::from(path));
    }
    Err(BuiltinError::TypeError(format!(
        "{function}() path should be str, bytes or os.PathLike, not {}",
        python_type_name(value)
    )))
}

pub(crate) fn os_fspath(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "fspath() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let path = args[0];
    if is_path_protocol_result(path) {
        return Ok(path);
    }

    let target = match resolve_special_method(path, "__fspath__") {
        Ok(target) => target,
        Err(err) if err.is_attribute_error() => {
            return Err(BuiltinError::TypeError(format!(
                "expected str, bytes or os.PathLike object, not {}",
                python_type_name(path)
            )));
        }
        Err(err) => return Err(BuiltinError::Raised(err)),
    };

    let result = match target.implicit_self {
        Some(implicit_self) => invoke_callable_value(vm, target.callable, &[implicit_self]),
        None => invoke_callable_value(vm, target.callable, &[]),
    }
    .map_err(BuiltinError::Raised)?;

    if is_path_protocol_result(result) {
        Ok(result)
    } else {
        Err(BuiltinError::TypeError(format!(
            "expected {}.__fspath__() to return str or bytes, not {}",
            python_type_name(path),
            python_type_name(result)
        )))
    }
}

#[inline]
fn is_path_protocol_result(value: Value) -> bool {
    value_as_string_ref(value).is_some()
        || value_as_bytes_ref(value).is_some_and(|bytes| bytes.header.type_id == TypeId::BYTES)
}

fn python_type_name(value: Value) -> String {
    if value.is_none() {
        return "NoneType".to_string();
    }
    if value.as_bool().is_some() {
        return "bool".to_string();
    }
    if value.as_int().is_some() {
        return "int".to_string();
    }
    if value.as_float().is_some() {
        return "float".to_string();
    }
    if value.is_string() {
        return "str".to_string();
    }

    let Some(ptr) = value.as_object_ptr() else {
        return "object".to_string();
    };

    let type_id = extract_type_id(ptr);
    if type_id.raw() >= TypeId::FIRST_USER_TYPE
        && let Some(class) = global_class(ClassId(type_id.raw()))
    {
        return class.name().as_str().to_string();
    }

    type_id.name().to_string()
}
