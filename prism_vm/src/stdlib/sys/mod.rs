//! Python `sys` module implementation.
//!
//! High-performance implementation of Python's sys module providing:
//! - Runtime version and platform information
//! - Command-line argument access
//! - Module search path management
//! - Standard stream access
//! - System limits and configuration

mod argv;
mod hooks;
mod internals;
mod limits;
mod paths;
mod runtime;
mod streams;

pub use argv::*;
pub use hooks::*;
pub use internals::*;
pub use limits::*;
pub use paths::*;
pub use runtime::*;
pub use streams::*;

use super::{Module, ModuleError};
use crate::builtins::{BuiltinError, BuiltinFunctionObject, create_exception_with_args};
use crate::error::RuntimeError;
use crate::ops::calls::value_supports_call_protocol;
use crate::ops::objects::{snapshot_frame_globals_dict, snapshot_frame_locals_dict};
use crate::stdlib::exceptions::ExceptionTypeId;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::views::FrameViewObject;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::tuple::TupleObject;
use std::sync::{Arc, LazyLock, Mutex};

/// The sys module providing runtime system configuration.
pub struct SysModule {
    /// Cached platform
    platform: Platform,
    /// Cached executable path
    executable: Arc<str>,
    /// Command-line arguments
    argv: SysArgv,
    /// Cached Python list value for `sys.argv`.
    argv_value: Value,
    /// Cached tuple value for `sys.builtin_module_names`.
    builtin_module_names_value: Value,
    /// Mutable list backing `sys.warnoptions`.
    warnoptions_value: Value,
    /// Mutable list backing `sys.meta_path`.
    meta_path_value: Value,
    /// Mutable list backing `sys.path_hooks`.
    path_hooks_value: Value,
    /// Mutable dict backing `sys.path_importer_cache`.
    path_importer_cache_value: Value,
    /// Fallback dict backing `sys.modules` before import resolver injection.
    modules_value: Value,
    /// Python-visible standard input stream.
    stdin_value: Value,
    /// Python-visible standard output stream.
    stdout_value: Value,
    /// Python-visible standard error stream.
    stderr_value: Value,
    /// Original standard input stream exposed via `__stdin__`.
    original_stdin_value: Value,
    /// Original standard output stream exposed via `__stdout__`.
    original_stdout_value: Value,
    /// Original standard error stream exposed via `__stderr__`.
    original_stderr_value: Value,
    /// Process flags exposed via `sys.flags`.
    flags_value: Value,
    /// Module search paths
    path: SysPaths,
    /// Installation prefix configuration.
    prefixes: SysPrefixes,
    /// Recursion limit
    recursion_limit: RecursionLimit,
}

static GET_FILESYSTEM_ENCODING_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("sys.getfilesystemencoding"),
        sys_getfilesystemencoding,
    )
});
static GET_FILESYSTEM_ENCODEERRORS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| {
        BuiltinFunctionObject::new(
            Arc::from("sys.getfilesystemencodeerrors"),
            sys_getfilesystemencodeerrors,
        )
    });
static EXIT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("sys.exit"), sys_exit));
static INTERN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("sys.intern"), sys_intern));
static AUDIT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("sys.audit"), sys_audit));
static DISPLAYHOOK_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("sys.displayhook"), sys_displayhook));
static EXCEPTHOOK_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("sys.excepthook"), sys_excepthook));
static EXC_INFO_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("sys.exc_info"), sys_exc_info));
static EXCEPTION_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("sys.exception"), sys_exception));
static GETFRAME_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("sys._getframe"), sys_getframe));
static GETTRACE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("sys.gettrace"), sys_gettrace));
static SETTRACE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("sys.settrace"), sys_settrace));
static GETPROFILE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("sys.getprofile"), sys_getprofile));
static SETPROFILE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("sys.setprofile"), sys_setprofile));
static GETSWITCHINTERVAL_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("sys.getswitchinterval"), sys_getswitchinterval)
});
static SETSWITCHINTERVAL_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("sys.setswitchinterval"), sys_setswitchinterval)
});
static SETTRACE_ALL_THREADS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("sys._settraceallthreads"),
        sys_settrace_all_threads,
    )
});
static SETPROFILE_ALL_THREADS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("sys._setprofileallthreads"),
        sys_setprofile_all_threads,
    )
});
static GETWINDOWSVERSION_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("sys.getwindowsversion"), sys_getwindowsversion)
});
static GETREFCOUNT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("sys.getrefcount"), sys_getrefcount));
static CURRENT_TRACE_FUNCTION: LazyLock<Mutex<Value>> = LazyLock::new(|| Mutex::new(Value::none()));
static CURRENT_PROFILE_FUNCTION: LazyLock<Mutex<Value>> =
    LazyLock::new(|| Mutex::new(Value::none()));
static CURRENT_SWITCH_INTERVAL: LazyLock<Mutex<SwitchInterval>> =
    LazyLock::new(|| Mutex::new(SwitchInterval::new()));

impl SysModule {
    /// Create a new sys module with default configuration.
    #[inline]
    pub fn new() -> Self {
        let argv = SysArgv::from_env();
        let path = SysPaths::from_env();
        let platform = Platform::detect();
        let stdin_value = super::io::new_stdin_stream_object();
        let stdout_value = super::io::new_stdout_stream_object();
        let stderr_value = super::io::new_stderr_stream_object();
        let executable_path = std::env::current_exe().ok();
        let executable = executable_path
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(String::new);
        let prefixes = SysPrefixes::from_env(executable_path.as_deref());
        Self {
            platform,
            executable: executable.into(),
            argv_value: argv.to_value(),
            builtin_module_names_value: builtin_module_names_value(platform),
            warnoptions_value: empty_list_value(),
            meta_path_value: empty_list_value(),
            path_hooks_value: empty_list_value(),
            path_importer_cache_value: empty_dict_value(),
            modules_value: empty_dict_value(),
            stdin_value,
            stdout_value,
            stderr_value,
            original_stdin_value: stdin_value,
            original_stdout_value: stdout_value,
            original_stderr_value: stderr_value,
            flags_value: sys_flags_value(),
            argv,
            path,
            prefixes,
            recursion_limit: RecursionLimit::new(),
        }
    }

    /// Create sys module with custom arguments (for testing).
    pub fn with_args(args: Vec<String>) -> Self {
        let argv = SysArgv::new(args);
        let path = SysPaths::from_env();
        let platform = Platform::detect();
        let stdin_value = super::io::new_stdin_stream_object();
        let stdout_value = super::io::new_stdout_stream_object();
        let stderr_value = super::io::new_stderr_stream_object();
        let executable_path = std::env::current_exe().ok();
        let executable = executable_path
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(String::new);
        let prefixes = SysPrefixes::from_env(executable_path.as_deref());
        Self {
            platform,
            executable: executable.into(),
            argv_value: argv.to_value(),
            builtin_module_names_value: builtin_module_names_value(platform),
            warnoptions_value: empty_list_value(),
            meta_path_value: empty_list_value(),
            path_hooks_value: empty_list_value(),
            path_importer_cache_value: empty_dict_value(),
            modules_value: empty_dict_value(),
            stdin_value,
            stdout_value,
            stderr_value,
            original_stdin_value: stdin_value,
            original_stdout_value: stdout_value,
            original_stderr_value: stderr_value,
            flags_value: sys_flags_value(),
            argv,
            path,
            prefixes,
            recursion_limit: RecursionLimit::new(),
        }
    }

    /// Get command-line arguments.
    #[inline]
    pub fn argv(&self) -> &SysArgv {
        &self.argv
    }

    /// Get module search paths.
    #[inline]
    pub fn path(&self) -> &SysPaths {
        &self.path
    }

    /// Get mutable module search paths.
    #[inline]
    pub fn path_mut(&mut self) -> &mut SysPaths {
        &mut self.path
    }

    /// Get the recursion limit.
    #[inline]
    pub fn getrecursionlimit(&self) -> u32 {
        self.recursion_limit.get()
    }

    /// Set the recursion limit.
    #[inline]
    pub fn setrecursionlimit(&mut self, limit: u32) -> Result<(), ModuleError> {
        self.recursion_limit.set(limit)
    }

    /// Get the platform.
    #[inline]
    pub fn platform(&self) -> Platform {
        self.platform
    }

    /// Get the version string.
    #[inline]
    pub fn version(&self) -> &'static str {
        VERSION_STRING
    }
}

impl Default for SysModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for SysModule {
    fn name(&self) -> &str {
        "sys"
    }

    fn get_attr(&self, name: &str) -> Result<Value, ModuleError> {
        match name {
            // Version information (return ints for now, strings need InternedString)
            "hexversion" => Ok(Value::int(HEXVERSION as i64).unwrap()),
            "api_version" => Ok(Value::int(API_VERSION as i64).unwrap()),

            // Limits
            "maxsize" => Ok(Value::int(MAX_SIZE).unwrap()),
            "maxunicode" => Ok(Value::int(MAX_UNICODE as i64).unwrap()),

            // Recursion limit as int
            "recursion_limit" => Ok(Value::int(self.recursion_limit.get() as i64).unwrap()),

            // Functions - return None placeholder (would be callable)
            "getrecursionlimit" | "setrecursionlimit" | "getsizeof" => {
                Ok(Value::none()) // Placeholder for callable
            }

            "getrefcount" => Ok(builtin_value(&GETREFCOUNT_FUNCTION)),
            "exit" => Ok(builtin_value(&EXIT_FUNCTION)),
            "exc_info" => Ok(builtin_value(&EXC_INFO_FUNCTION)),
            "exception" => Ok(builtin_value(&EXCEPTION_FUNCTION)),
            "_getframe" => Ok(builtin_value(&GETFRAME_FUNCTION)),
            "gettrace" => Ok(builtin_value(&GETTRACE_FUNCTION)),
            "settrace" => Ok(builtin_value(&SETTRACE_FUNCTION)),
            "getprofile" => Ok(builtin_value(&GETPROFILE_FUNCTION)),
            "setprofile" => Ok(builtin_value(&SETPROFILE_FUNCTION)),
            "getswitchinterval" => Ok(builtin_value(&GETSWITCHINTERVAL_FUNCTION)),
            "setswitchinterval" => Ok(builtin_value(&SETSWITCHINTERVAL_FUNCTION)),
            "_settraceallthreads" => Ok(builtin_value(&SETTRACE_ALL_THREADS_FUNCTION)),
            "_setprofileallthreads" => Ok(builtin_value(&SETPROFILE_ALL_THREADS_FUNCTION)),
            "intern" => Ok(builtin_value(&INTERN_FUNCTION)),
            "getfilesystemencoding" => Ok(builtin_value(&GET_FILESYSTEM_ENCODING_FUNCTION)),
            "getfilesystemencodeerrors" => Ok(builtin_value(&GET_FILESYSTEM_ENCODEERRORS_FUNCTION)),
            "audit" => Ok(builtin_value(&AUDIT_FUNCTION)),

            // String attributes
            "version" => Ok(Value::string(intern(self.version()))),
            "platform" => Ok(Value::string(intern(self.platform.as_str()))),
            "executable" => Ok(Value::string(intern(self.executable.as_ref()))),
            "prefix" => Ok(Value::string(intern(self.prefixes.prefix()))),
            "exec_prefix" => Ok(Value::string(intern(self.prefixes.exec_prefix()))),
            "base_prefix" => Ok(Value::string(intern(self.prefixes.base_prefix()))),
            "base_exec_prefix" => Ok(Value::string(intern(self.prefixes.base_exec_prefix()))),
            "winver" if self.platform.is_windows() => Ok(Value::string(intern(WINVER))),
            "byteorder" => Ok(Value::string(intern(byte_order()))),
            "copyright" => Ok(Value::string(intern(COPYRIGHT))),
            "platlibdir" => Ok(Value::string(intern(PLATLIBDIR))),
            "_vpath" if self.platform.is_windows() => Ok(Value::string(intern(VPATH))),
            "pycache_prefix" => Ok(Value::none()),
            "dont_write_bytecode" => Ok(Value::bool(false)),

            "version_info" => Ok(version_info_tuple()),
            "implementation" => Ok(implementation_info()),
            "float_info" => Ok(float_info_tuple()),
            "int_info" => Ok(int_info_tuple()),
            "hash_info" => Ok(hash_info_tuple()),
            "_git" => Ok(git_info_tuple()),

            "builtin_module_names" => Ok(self.builtin_module_names_value),
            "warnoptions" => Ok(self.warnoptions_value),
            "meta_path" => Ok(self.meta_path_value),
            "path_hooks" => Ok(self.path_hooks_value),
            "path_importer_cache" => Ok(self.path_importer_cache_value),
            "modules" => Ok(self.modules_value),
            "flags" => Ok(self.flags_value),

            "stdin" => Ok(self.stdin_value),
            "stdout" => Ok(self.stdout_value),
            "stderr" => Ok(self.stderr_value),
            "__stdin__" => Ok(self.original_stdin_value),
            "__stdout__" => Ok(self.original_stdout_value),
            "__stderr__" => Ok(self.original_stderr_value),

            "displayhook" | "__displayhook__" => Ok(builtin_value(&DISPLAYHOOK_FUNCTION)),
            "excepthook" | "__excepthook__" => Ok(builtin_value(&EXCEPTHOOK_FUNCTION)),

            // Path list
            "path" => Ok(self.path.to_value()),
            // Command-line arguments
            "argv" => Ok(self.argv_value),
            "getwindowsversion" if self.platform.is_windows() => {
                Ok(builtin_value(&GETWINDOWSVERSION_FUNCTION))
            }

            _ => Err(ModuleError::AttributeError(format!(
                "module 'sys' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        let mut entries = vec![
            Arc::from("hexversion"),
            Arc::from("api_version"),
            Arc::from("maxsize"),
            Arc::from("maxunicode"),
            Arc::from("recursion_limit"),
            Arc::from("version"),
            Arc::from("platform"),
            Arc::from("executable"),
            Arc::from("prefix"),
            Arc::from("exec_prefix"),
            Arc::from("base_prefix"),
            Arc::from("base_exec_prefix"),
            Arc::from("byteorder"),
            Arc::from("copyright"),
            Arc::from("platlibdir"),
            Arc::from("pycache_prefix"),
            Arc::from("dont_write_bytecode"),
            Arc::from("builtin_module_names"),
            Arc::from("warnoptions"),
            Arc::from("meta_path"),
            Arc::from("path_hooks"),
            Arc::from("path_importer_cache"),
            Arc::from("modules"),
            Arc::from("flags"),
            Arc::from("version_info"),
            Arc::from("implementation"),
            Arc::from("float_info"),
            Arc::from("int_info"),
            Arc::from("hash_info"),
            Arc::from("_git"),
            Arc::from("stdin"),
            Arc::from("stdout"),
            Arc::from("stderr"),
            Arc::from("__stdin__"),
            Arc::from("__stdout__"),
            Arc::from("__stderr__"),
            Arc::from("displayhook"),
            Arc::from("excepthook"),
            Arc::from("__displayhook__"),
            Arc::from("__excepthook__"),
            Arc::from("audit"),
            Arc::from("path"),
            Arc::from("argv"),
            Arc::from("exc_info"),
            Arc::from("exception"),
            Arc::from("_getframe"),
            Arc::from("gettrace"),
            Arc::from("settrace"),
            Arc::from("getprofile"),
            Arc::from("setprofile"),
            Arc::from("getswitchinterval"),
            Arc::from("setswitchinterval"),
            Arc::from("_settraceallthreads"),
            Arc::from("_setprofileallthreads"),
            Arc::from("exit"),
            Arc::from("getfilesystemencoding"),
            Arc::from("getfilesystemencodeerrors"),
            Arc::from("getrecursionlimit"),
            Arc::from("setrecursionlimit"),
            Arc::from("getsizeof"),
            Arc::from("getrefcount"),
            Arc::from("intern"),
        ];
        if self.platform.is_windows() {
            entries.push(Arc::from("winver"));
            entries.push(Arc::from("_vpath"));
            entries.push(Arc::from("getwindowsversion"));
        }
        entries
    }
}

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

fn sys_displayhook(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "displayhook() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    Ok(Value::none())
}

fn sys_excepthook(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 3 {
        return Err(BuiltinError::TypeError(format!(
            "excepthook() takes exactly 3 arguments ({} given)",
            args.len()
        )));
    }

    Ok(Value::none())
}

fn sys_exit(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "exit expected at most 1 argument, got {}",
            args.len()
        )));
    }

    let exception = create_exception_with_args(
        ExceptionTypeId::SystemExit,
        None,
        args.to_vec().into_boxed_slice(),
    );
    let message = if args.is_empty() {
        Arc::<str>::from("")
    } else {
        Arc::<str>::from(system_exit_argument_text(args[0]))
    };
    Err(BuiltinError::Raised(RuntimeError::raised_exception(
        ExceptionTypeId::SystemExit.as_u8() as u16,
        exception,
        message,
    )))
}

fn system_exit_argument_text(value: Value) -> String {
    if value.is_none() {
        return "None".to_string();
    }
    if let Some(boolean) = value.as_bool() {
        return if boolean { "True" } else { "False" }.to_string();
    }
    if let Some(integer) = value.as_int() {
        return integer.to_string();
    }
    if let Some(float) = value.as_float() {
        return float.to_string();
    }
    if value.is_string()
        && let Some(ptr) = value.as_string_object_ptr()
        && let Some(interned) = prism_core::intern::interned_by_ptr(ptr as *const u8)
    {
        return interned.as_str().to_string();
    }
    value.type_name().to_string()
}

fn sys_exc_info(vm: &mut crate::VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "exc_info() takes exactly 0 arguments ({} given)",
            args.len()
        )));
    }

    let exc_info = crate::ops::exception::build_exc_info(vm);
    let (exc_type, exc_value, exc_traceback) = exc_info.to_tuple();
    Ok(leak_object_value(TupleObject::from_slice(&[
        exc_type,
        exc_value,
        exc_traceback,
    ])))
}

fn sys_exception(vm: &mut crate::VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "exception() takes exactly 0 arguments ({} given)",
            args.len()
        )));
    }

    Ok(crate::ops::exception::build_exc_info(vm).exc_value)
}

fn sys_getframe(vm: &mut crate::VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "_getframe() takes at most 1 argument ({} given)",
            args.len()
        )));
    }

    let depth = args
        .first()
        .copied()
        .map(parse_frame_depth)
        .transpose()?
        .unwrap_or(0);
    let Some(frame_index) = vm.frame_index_at_depth(depth) else {
        return Err(BuiltinError::ValueError(
            "call stack is not deep enough".to_string(),
        ));
    };

    Ok(build_frame_view(vm, frame_index).expect("frame index from VM should be valid"))
}

fn sys_gettrace(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "gettrace() takes no arguments ({} given)",
            args.len()
        )));
    }

    Ok(*CURRENT_TRACE_FUNCTION
        .lock()
        .expect("sys trace hook lock poisoned"))
}

fn sys_settrace(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "settrace() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    set_trace_hook(args[0])
}

fn sys_getprofile(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "getprofile() takes no arguments ({} given)",
            args.len()
        )));
    }

    Ok(*CURRENT_PROFILE_FUNCTION
        .lock()
        .expect("sys profile hook lock poisoned"))
}

fn sys_setprofile(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "setprofile() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    set_profile_hook(args[0])
}

fn sys_getswitchinterval(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "getswitchinterval() takes no arguments ({} given)",
            args.len()
        )));
    }

    Ok(Value::float(
        CURRENT_SWITCH_INTERVAL
            .lock()
            .expect("sys switch interval lock poisoned")
            .get(),
    ))
}

fn sys_setswitchinterval(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "setswitchinterval() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let interval = switch_interval_from_value(args[0])?;
    CURRENT_SWITCH_INTERVAL
        .lock()
        .expect("sys switch interval lock poisoned")
        .set(interval)
        .map_err(|err| match err {
            ModuleError::ValueError(message) => BuiltinError::ValueError(message),
            ModuleError::TypeError(message) => BuiltinError::TypeError(message),
            other => BuiltinError::ValueError(other.to_string()),
        })?;
    Ok(Value::none())
}

fn switch_interval_from_value(value: Value) -> Result<f64, BuiltinError> {
    if let Some(boolean) = value.as_bool() {
        return Ok(if boolean { 1.0 } else { 0.0 });
    }
    if let Some(float) = value.as_float() {
        return Ok(float);
    }
    if let Some(integer) = value.as_int() {
        return Ok(integer as f64);
    }
    Err(BuiltinError::TypeError(format!(
        "setswitchinterval() argument must be real number, not {}",
        value.type_name()
    )))
}

fn sys_getrefcount(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "getrefcount() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    // Prism is not a reference-counted runtime. CPython includes the temporary
    // argument reference in this value, so return a conservative positive count
    // that preserves API shape without exposing false GC internals.
    Ok(Value::int(2).expect("small refcount fits in Value::int"))
}

fn sys_settrace_all_threads(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "_settraceallthreads() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    set_trace_hook(args[0])
}

fn sys_setprofile_all_threads(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "_setprofileallthreads() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    set_profile_hook(args[0])
}

fn set_trace_hook(trace: Value) -> Result<Value, BuiltinError> {
    if !trace.is_none() && !value_supports_call_protocol(trace) {
        return Err(BuiltinError::TypeError(
            "settrace() argument must be callable or None".to_string(),
        ));
    }

    *CURRENT_TRACE_FUNCTION
        .lock()
        .expect("sys trace hook lock poisoned") = trace;
    Ok(Value::none())
}

fn set_profile_hook(profile: Value) -> Result<Value, BuiltinError> {
    if !profile.is_none() && !value_supports_call_protocol(profile) {
        return Err(BuiltinError::TypeError(
            "setprofile() argument must be callable or None".to_string(),
        ));
    }

    *CURRENT_PROFILE_FUNCTION
        .lock()
        .expect("sys profile hook lock poisoned") = profile;
    Ok(Value::none())
}

fn sys_intern(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "intern() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let value = args[0];
    if value.is_string() {
        let ptr = value
            .as_string_object_ptr()
            .ok_or_else(|| BuiltinError::TypeError("intern() argument must be str".to_string()))?;
        let resolved = prism_core::intern::interned_by_ptr(ptr as *const u8)
            .ok_or_else(|| BuiltinError::TypeError("intern() argument must be str".to_string()))?;
        return Ok(Value::string(intern(resolved.as_str())));
    }

    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(
            "intern() argument must be str".to_string(),
        ));
    };

    if crate::ops::objects::extract_type_id(ptr) != prism_runtime::object::type_obj::TypeId::STR {
        return Err(BuiltinError::TypeError(
            "intern() argument must be str".to_string(),
        ));
    }

    let string = unsafe { &*(ptr as *const prism_runtime::types::string::StringObject) };
    Ok(Value::string(intern(string.as_str())))
}

fn parse_frame_depth(value: Value) -> Result<usize, BuiltinError> {
    let depth = if let Some(index) = value.as_int() {
        index
    } else if let Some(boolean) = value.as_bool() {
        if boolean { 1 } else { 0 }
    } else {
        return Err(BuiltinError::TypeError(format!(
            "'{}' object cannot be interpreted as an integer",
            value.type_name()
        )));
    };

    if depth < 0 {
        return Err(BuiltinError::ValueError(
            "call stack is not deep enough".to_string(),
        ));
    }

    Ok(depth as usize)
}

fn build_frame_view(vm: &crate::VirtualMachine, frame_index: usize) -> Option<Value> {
    let frame = vm.frames.get(frame_index)?;
    let globals = leak_object_value(snapshot_frame_globals_dict(vm, frame));
    let locals = leak_object_value(snapshot_frame_locals_dict(frame));
    let back = frame
        .return_frame
        .and_then(|back_index| build_frame_view(vm, back_index as usize));
    Some(leak_object_value(FrameViewObject::new(
        Some(Arc::clone(&frame.code)),
        globals,
        locals,
        frame_line_number(frame),
        frame.ip,
        back,
    )))
}

#[inline]
fn frame_line_number(frame: &crate::frame::Frame) -> u32 {
    frame
        .code
        .line_for_pc(frame.ip)
        .or_else(|| {
            frame
                .ip
                .checked_sub(1)
                .and_then(|pc| frame.code.line_for_pc(pc))
        })
        .unwrap_or(frame.code.first_lineno)
}

#[inline]
fn leak_object_value<T: prism_runtime::Trace + 'static>(object: T) -> Value {
    crate::alloc_managed_value(object)
}

#[inline]
fn empty_list_value() -> Value {
    leak_object_value(ListObject::new())
}

#[inline]
fn empty_dict_value() -> Value {
    leak_object_value(DictObject::new())
}

fn git_info_tuple() -> Value {
    leak_object_value(TupleObject::from_slice(&[
        Value::string(intern("Prism")),
        Value::string(intern("")),
        Value::string(intern("")),
    ]))
}

fn sys_flags_value() -> Value {
    let registry = shape_registry();
    let mut flags = Box::new(ShapedObject::with_empty_shape(registry.empty_shape()));

    for (name, value) in [
        ("debug", Value::int(0).unwrap()),
        ("inspect", Value::int(0).unwrap()),
        ("interactive", Value::int(0).unwrap()),
        ("optimize", Value::int(0).unwrap()),
        ("dont_write_bytecode", Value::int(0).unwrap()),
        ("no_user_site", Value::int(0).unwrap()),
        ("no_site", Value::int(0).unwrap()),
        ("ignore_environment", Value::int(0).unwrap()),
        ("verbose", Value::int(0).unwrap()),
        ("bytes_warning", Value::int(0).unwrap()),
        ("quiet", Value::int(0).unwrap()),
        ("hash_randomization", Value::int(1).unwrap()),
        ("isolated", Value::int(0).unwrap()),
        ("dev_mode", Value::bool(false)),
        ("utf8_mode", Value::int(0).unwrap()),
        ("warn_default_encoding", Value::int(0).unwrap()),
        ("safe_path", Value::bool(false)),
        ("int_max_str_digits", Value::int(4300).unwrap()),
    ] {
        flags.set_property(intern(name), value, registry);
    }

    Value::object_ptr(Box::into_raw(flags) as *const ())
}

fn builtin_module_names_value(platform: Platform) -> Value {
    let builtin_names = if platform.is_windows() {
        super::builtin_module_names()
            .iter()
            .copied()
            .collect::<Vec<_>>()
    } else if platform.is_posix() {
        super::builtin_module_names()
            .iter()
            .copied()
            .collect::<Vec<_>>()
    } else {
        super::builtin_module_names()
            .iter()
            .copied()
            .collect::<Vec<_>>()
    };

    let names = builtin_names
        .into_iter()
        .map(|name| Value::string(intern(name)))
        .collect::<Vec<_>>();
    leak_object_value(TupleObject::from_vec(names))
}

#[inline]
fn filesystem_encoding() -> &'static str {
    "utf-8"
}

#[inline]
fn filesystem_encode_errors() -> &'static str {
    if cfg!(windows) {
        "surrogatepass"
    } else {
        "surrogateescape"
    }
}

fn sys_getfilesystemencoding(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "getfilesystemencoding() takes 0 positional arguments but {} were given",
            args.len()
        )));
    }

    Ok(Value::string(intern(filesystem_encoding())))
}

fn sys_getfilesystemencodeerrors(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "getfilesystemencodeerrors() takes 0 positional arguments but {} were given",
            args.len()
        )));
    }

    Ok(Value::string(intern(filesystem_encode_errors())))
}

fn sys_audit(_args: &[Value]) -> Result<Value, BuiltinError> {
    Ok(Value::none())
}

fn sys_getwindowsversion(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "getwindowsversion() takes 0 positional arguments but {} were given",
            args.len()
        )));
    }

    Ok(windows_version_info())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
