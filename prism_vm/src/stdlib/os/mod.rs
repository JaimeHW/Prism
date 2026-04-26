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
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::type_builtins::global_class;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::value_as_bytes_ref;
use prism_runtime::types::string::value_as_string_ref;
use std::sync::{Arc, LazyLock};

static OS_URANDOM_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.urandom"), os_urandom));
static OS_FSPATH_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("os.fspath"), os_fspath));

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
            "name" => Ok(Value::none()), // TODO: Return interned "nt" or "posix"
            "sep" => Ok(Value::none()),  // TODO: Return interned "/" or "\\"
            "pathsep" => Ok(Value::none()), // TODO: Return interned ":" or ";"
            "linesep" => Ok(Value::none()), // TODO: Return interned "\n" or "\r\n"
            "curdir" => Ok(Value::none()), // "."
            "pardir" => Ok(Value::none()), // ".."
            "extsep" => Ok(Value::none()), // "."
            "altsep" => Ok(Value::none()), // "/" on Windows, None on Unix
            "devnull" => Ok(Value::none()), // "/dev/null" or "nul"

            // Functions (return None placeholders for now)
            "getcwd" | "chdir" | "mkdir" | "makedirs" | "rmdir" | "removedirs" | "remove"
            | "unlink" | "rename" | "replace" | "stat" | "lstat" | "listdir" | "scandir"
            | "walk" | "fwalk" | "getenv" | "putenv" | "unsetenv" | "getpid" | "getppid"
            | "kill" | "system" | "popen" | "access" | "chmod" | "chown" | "link" | "symlink"
            | "readlink" => {
                Ok(Value::none()) // Placeholder for callable
            }
            "urandom" => Ok(builtin_value(&OS_URANDOM_FUNCTION)),
            "fspath" => Ok(builtin_value(&OS_FSPATH_FUNCTION)),

            // Submodule
            "path" => Ok(Value::none()), // TODO: Return os.path module

            // Environ dict
            "environ" => Ok(Value::none()), // TODO: Return environ dict

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

fn os_urandom(args: &[Value]) -> Result<Value, BuiltinError> {
    urandom_value_from_args(args, "urandom")
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
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

    #[test]
    fn test_environ_access() {
        let os = OsModule::new();
        let _environ = os.environ();
    }

    #[test]
    fn test_environ_mut_access() {
        let mut os = OsModule::new();
        let _environ = os.environ_mut();
    }
}
