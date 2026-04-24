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
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::ops::calls::value_supports_call_protocol;
use crate::ops::objects::{snapshot_frame_globals_dict, snapshot_frame_locals_dict};
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
static GETWINDOWSVERSION_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("sys.getwindowsversion"), sys_getwindowsversion)
});
static CURRENT_TRACE_FUNCTION: LazyLock<Mutex<Value>> = LazyLock::new(|| Mutex::new(Value::none()));

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
            "exit" | "getrecursionlimit" | "setrecursionlimit" | "getsizeof" | "getrefcount" => {
                Ok(Value::none()) // Placeholder for callable
            }

            "exc_info" => Ok(builtin_value(&EXC_INFO_FUNCTION)),
            "exception" => Ok(builtin_value(&EXCEPTION_FUNCTION)),
            "_getframe" => Ok(builtin_value(&GETFRAME_FUNCTION)),
            "gettrace" => Ok(builtin_value(&GETTRACE_FUNCTION)),
            "settrace" => Ok(builtin_value(&SETTRACE_FUNCTION)),
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

    let trace = args[0];
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
fn leak_object_value<T>(object: T) -> Value {
    let ptr = Box::into_raw(Box::new(object)) as *const ();
    Value::object_ptr(ptr)
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
mod tests {
    use super::*;
    use crate::builtins::BuiltinFunctionObject;
    use crate::builtins::create_exception;
    use crate::stdlib::exceptions::ExceptionTypeId;
    use prism_core::intern::interned_by_ptr;
    use prism_runtime::object::shaped_object::ShapedObject;
    use prism_runtime::object::views::FrameViewObject;
    use prism_runtime::types::list::ListObject;
    use prism_runtime::types::tuple::TupleObject;
    use std::sync::Arc;

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

    #[test]
    fn test_sys_module_default() {
        let sys = SysModule::default();
        assert_eq!(sys.name(), "sys");
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
        assert!(
            ["win32", "linux", "darwin", "freebsd", "unknown"].contains(&platform_name.as_str())
        );
    }

    #[test]
    fn test_prefix_family_attributes_are_present_and_consistent() {
        let sys = SysModule::new();

        for attr in ["prefix", "exec_prefix", "base_prefix", "base_exec_prefix"] {
            let value = sys.get_attr(attr).expect("prefix attribute should exist");
            let ptr = value
                .as_string_object_ptr()
                .expect("prefix attribute should be an interned string")
                as *const u8;
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
    fn test_exit_placeholder() {
        let sys = SysModule::new();
        let exit = sys.get_attr("exit").unwrap();
        assert!(exit.is_none()); // Placeholder for callable
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
                .expect("sys.winver should be an interned string")
                as *const u8;
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
            .expect("filesystem encode errors should be string")
            as *const u8;
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
    fn test_version_info_attribute_is_real_tuple() {
        let sys = SysModule::new();
        let value = sys
            .get_attr("version_info")
            .expect("version_info should exist");
        let ptr = value
            .as_object_ptr()
            .expect("version_info should be a tuple object");
        let tuple = unsafe { &*(ptr as *const TupleObject) };
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
        let active_exception =
            create_exception(ExceptionTypeId::TypeError, Some(Arc::from("boom")));
        vm.set_active_exception_with_type(
            active_exception,
            ExceptionTypeId::TypeError.as_u8() as u16,
        );

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
        *CURRENT_TRACE_FUNCTION
            .lock()
            .expect("sys trace hook lock poisoned") = Value::none();
        let sys = SysModule::new();
        let gettrace = unsafe {
            &*(sys
                .get_attr("gettrace")
                .expect("sys.gettrace should exist")
                .as_object_ptr()
                .expect("sys.gettrace should be callable")
                as *const BuiltinFunctionObject)
        };
        let settrace = unsafe {
            &*(sys
                .get_attr("settrace")
                .expect("sys.settrace should exist")
                .as_object_ptr()
                .expect("sys.settrace should be callable")
                as *const BuiltinFunctionObject)
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
}
