//! Native Windows `_winapi` bootstrap surface.
//!
//! CPython's Windows stdlib imports `_winapi` opportunistically from modules
//! such as `shutil`, `subprocess`, and `encodings`. Prism only needs a small
//! subset of that API to bootstrap the pure-Python stdlib correctly, but the
//! exposed surface is structured so additional process and handle primitives can
//! be implemented without changing import behavior.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use prism_core::Value;
use prism_core::intern::{intern, interned_by_ptr};
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::int::value_to_i64;
use prism_runtime::types::string::StringObject;
use prism_runtime::types::tuple::TupleObject;
use rustc_hash::{FxHashMap, FxHashSet};
use std::path::Path;
use std::sync::{Arc, LazyLock, Mutex};

#[cfg(windows)]
use windows_sys::Win32::Foundation::{
    CloseHandle, DUPLICATE_SAME_ACCESS, DuplicateHandle, GetLastError, HANDLE,
    INVALID_HANDLE_VALUE, WAIT_FAILED,
};
#[cfg(windows)]
use windows_sys::Win32::Globalization::GetACP;
#[cfg(windows)]
use windows_sys::Win32::Storage::FileSystem::{
    FILE_TYPE_CHAR, FILE_TYPE_DISK, FILE_TYPE_PIPE, FILE_TYPE_UNKNOWN, GetFileType,
};
#[cfg(windows)]
use windows_sys::Win32::System::Console::GetStdHandle;
#[cfg(windows)]
use windows_sys::Win32::System::Pipes::CreatePipe;
#[cfg(windows)]
use windows_sys::Win32::System::Threading::{
    CreateProcessW, GetCurrentProcess, GetExitCodeProcess, PROCESS_INFORMATION, STARTUPINFOW,
    TerminateProcess, WaitForSingleObject,
};

#[cfg(not(windows))]
const DUPLICATE_SAME_ACCESS: u32 = 0x0000_0002;
#[cfg(not(windows))]
const FILE_TYPE_UNKNOWN: u32 = 0x0000;
#[cfg(not(windows))]
const FILE_TYPE_DISK: u32 = 0x0001;
#[cfg(not(windows))]
const FILE_TYPE_CHAR: u32 = 0x0002;
#[cfg(not(windows))]
const FILE_TYPE_PIPE: u32 = 0x0003;

const EXPORTED_CONSTANTS: &[(&str, i64)] = &[
    ("ABOVE_NORMAL_PRIORITY_CLASS", 0x0000_8000),
    ("BELOW_NORMAL_PRIORITY_CLASS", 0x0000_4000),
    ("COPY_FILE_ALLOW_DECRYPTED_DESTINATION", 0x0000_0008),
    ("COPY_FILE_COPY_SYMLINK", 0x0000_0800),
    ("CREATE_BREAKAWAY_FROM_JOB", 0x0100_0000),
    ("CREATE_DEFAULT_ERROR_MODE", 0x0400_0000),
    ("CREATE_NEW_CONSOLE", 0x0000_0010),
    ("CREATE_NEW_PROCESS_GROUP", 0x0000_0200),
    ("CREATE_NO_WINDOW", 0x0800_0000),
    ("CREATE_UNICODE_ENVIRONMENT", 0x0000_0400),
    ("DETACHED_PROCESS", 0x0000_0008),
    ("DUPLICATE_SAME_ACCESS", DUPLICATE_SAME_ACCESS as i64),
    ("EXTENDED_STARTUPINFO_PRESENT", 0x0008_0000),
    ("FILE_TYPE_CHAR", FILE_TYPE_CHAR as i64),
    ("FILE_TYPE_DISK", FILE_TYPE_DISK as i64),
    ("FILE_TYPE_PIPE", FILE_TYPE_PIPE as i64),
    ("FILE_TYPE_UNKNOWN", FILE_TYPE_UNKNOWN as i64),
    ("HIGH_PRIORITY_CLASS", 0x0000_0080),
    ("IDLE_PRIORITY_CLASS", 0x0000_0040),
    ("INFINITE", 0xFFFF_FFFF),
    ("NORMAL_PRIORITY_CLASS", 0x0000_0020),
    ("NULL", 0),
    ("REALTIME_PRIORITY_CLASS", 0x0000_0100),
    ("STARTF_USESHOWWINDOW", 0x0000_0001),
    ("STARTF_USESTDHANDLES", 0x0000_0100),
    ("STD_ERROR_HANDLE", -12),
    ("STD_INPUT_HANDLE", -10),
    ("STD_OUTPUT_HANDLE", -11),
    ("STILL_ACTIVE", 259),
    ("SW_HIDE", 0),
    ("WAIT_ABANDONED_0", 0x0000_0080),
    ("WAIT_OBJECT_0", 0x0000_0000),
    ("WAIT_TIMEOUT", 258),
];

const CREATE_UNICODE_ENVIRONMENT_FLAG: u32 = 0x0000_0400;
const STARTF_USESTDHANDLES_FLAG: u32 = 0x0000_0100;

#[cfg(windows)]
static PIPE_HANDLE_STATE: LazyLock<Mutex<PipeHandleState>> =
    LazyLock::new(|| Mutex::new(PipeHandleState::default()));

#[cfg(windows)]
#[derive(Default)]
struct PipeHandleState {
    handles: FxHashSet<i64>,
    duplicate_sources: FxHashMap<i64, i64>,
}

static CLOSE_HANDLE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_winapi.CloseHandle"), winapi_close_handle)
});
static CREATE_PIPE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_winapi.CreatePipe"), winapi_create_pipe)
});
static CREATE_PROCESS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_winapi.CreateProcess"), winapi_create_process)
});
static DUPLICATE_HANDLE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_winapi.DuplicateHandle"),
        winapi_duplicate_handle,
    )
});
static GET_ACP_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_winapi.GetACP"), winapi_get_acp));
static GET_CURRENT_PROCESS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_winapi.GetCurrentProcess"),
        winapi_get_current_process,
    )
});
static GET_EXIT_CODE_PROCESS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_winapi.GetExitCodeProcess"),
        winapi_get_exit_code_process,
    )
});
static GET_FILE_TYPE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_winapi.GetFileType"), winapi_get_file_type)
});
static GET_STD_HANDLE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_winapi.GetStdHandle"), winapi_get_std_handle)
});
static NEED_CURDIR_FOR_EXE_PATH_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_winapi.NeedCurrentDirectoryForExePath"),
        winapi_need_current_directory_for_exe_path,
    )
});
static TERMINATE_PROCESS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_winapi.TerminateProcess"),
        winapi_terminate_process,
    )
});
static WAIT_FOR_SINGLE_OBJECT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_winapi.WaitForSingleObject"),
        winapi_wait_for_single_object,
    )
});

/// Native `_winapi` module descriptor.
#[derive(Debug, Clone)]
pub struct WinApiModule {
    attrs: Vec<Arc<str>>,
    values: FxHashMap<Arc<str>, Value>,
}

impl WinApiModule {
    /// Create a new `_winapi` module descriptor.
    pub fn new() -> Self {
        let mut attrs = Vec::with_capacity(EXPORTED_CONSTANTS.len() + 13);
        let mut values = FxHashMap::default();

        for &(name, value) in EXPORTED_CONSTANTS {
            let exported = Value::int(value).expect("_winapi constant should fit in tagged int");
            attrs.push(Arc::from(name));
            values.insert(Arc::from(name), exported);
        }

        for name in [
            "CloseHandle",
            "CreatePipe",
            "CreateProcess",
            "DuplicateHandle",
            "GetACP",
            "GetCurrentProcess",
            "GetExitCodeProcess",
            "GetFileType",
            "GetStdHandle",
            "NeedCurrentDirectoryForExePath",
            "TerminateProcess",
            "WaitForSingleObject",
        ] {
            attrs.push(Arc::from(name));
        }
        attrs.sort_unstable();

        Self { attrs, values }
    }
}

impl Default for WinApiModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for WinApiModule {
    fn name(&self) -> &str {
        "_winapi"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "CloseHandle" => Ok(builtin_value(&CLOSE_HANDLE_FUNCTION)),
            "CreatePipe" => Ok(builtin_value(&CREATE_PIPE_FUNCTION)),
            "CreateProcess" => Ok(builtin_value(&CREATE_PROCESS_FUNCTION)),
            "DuplicateHandle" => Ok(builtin_value(&DUPLICATE_HANDLE_FUNCTION)),
            "GetACP" => Ok(builtin_value(&GET_ACP_FUNCTION)),
            "GetCurrentProcess" => Ok(builtin_value(&GET_CURRENT_PROCESS_FUNCTION)),
            "GetExitCodeProcess" => Ok(builtin_value(&GET_EXIT_CODE_PROCESS_FUNCTION)),
            "GetFileType" => Ok(builtin_value(&GET_FILE_TYPE_FUNCTION)),
            "GetStdHandle" => Ok(builtin_value(&GET_STD_HANDLE_FUNCTION)),
            "NeedCurrentDirectoryForExePath" => {
                Ok(builtin_value(&NEED_CURDIR_FOR_EXE_PATH_FUNCTION))
            }
            "TerminateProcess" => Ok(builtin_value(&TERMINATE_PROCESS_FUNCTION)),
            "WaitForSingleObject" => Ok(builtin_value(&WAIT_FOR_SINGLE_OBJECT_FUNCTION)),
            _ => self.values.get(name).copied().ok_or_else(|| {
                ModuleError::AttributeError(format!("module '_winapi' has no attribute '{}'", name))
            }),
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
    if crate::ops::objects::extract_type_id(ptr) != prism_runtime::object::type_obj::TypeId::STR {
        return None;
    }

    let string = unsafe { &*(ptr as *const StringObject) };
    Some(string.as_str().to_string())
}

fn optional_string_arg(
    value: Value,
    fn_name: &str,
    arg_name: &str,
) -> Result<Option<String>, BuiltinError> {
    if value.is_none() {
        return Ok(None);
    }
    value_to_string(value).map(Some).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "{fn_name}() {arg_name} must be str or None, not {}",
            value.type_name()
        ))
    })
}

fn string_arg(value: Value, fn_name: &str, arg_name: &str) -> Result<String, BuiltinError> {
    value_to_string(value).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "{fn_name}() {arg_name} must be str, not {}",
            value.type_name()
        ))
    })
}

fn parse_exact_arity(args: &[Value], fn_name: &str, expected: usize) -> Result<(), BuiltinError> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(BuiltinError::TypeError(format!(
            "{fn_name}() takes exactly {expected} argument{} ({} given)",
            if expected == 1 { "" } else { "s" },
            args.len()
        )))
    }
}

fn integer_arg(value: Value, fn_name: &str, arg_name: &str) -> Result<i64, BuiltinError> {
    if let Some(flag) = value.as_bool() {
        return Ok(if flag { 1 } else { 0 });
    }
    value_to_i64(value).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "{fn_name}() {arg_name} must be an integer, not {}",
            value.type_name()
        ))
    })
}

fn u32_arg(value: Value, fn_name: &str, arg_name: &str) -> Result<u32, BuiltinError> {
    let integer = integer_arg(value, fn_name, arg_name)?;
    u32::try_from(integer).map_err(|_| {
        BuiltinError::OverflowError(format!(
            "{fn_name}() {arg_name} is outside the unsigned C int range"
        ))
    })
}

fn bool_arg(value: Value, fn_name: &str, arg_name: &str) -> Result<bool, BuiltinError> {
    if let Some(flag) = value.as_bool() {
        return Ok(flag);
    }
    Ok(integer_arg(value, fn_name, arg_name)? != 0)
}

fn int_value(value: i64, context: &str) -> Result<Value, BuiltinError> {
    Value::int(value).ok_or_else(|| BuiltinError::OverflowError(format!("{context} overflow")))
}

#[cfg(windows)]
#[inline]
fn handle_from_i64(value: i64) -> HANDLE {
    value as isize as HANDLE
}

#[cfg(windows)]
#[inline]
fn handle_value(handle: HANDLE) -> Result<Value, BuiltinError> {
    int_value(handle as isize as i64, "Windows handle")
}

#[cfg(windows)]
#[inline]
fn handle_to_i64(handle: HANDLE) -> i64 {
    handle as isize as i64
}

#[cfg(windows)]
fn track_pipe_handles(read_pipe: HANDLE, write_pipe: HANDLE) {
    let mut state = PIPE_HANDLE_STATE
        .lock()
        .expect("_winapi pipe handle state mutex poisoned");
    state.handles.insert(handle_to_i64(read_pipe));
    state.handles.insert(handle_to_i64(write_pipe));
}

#[cfg(windows)]
fn track_pipe_duplicate(source_handle: i64, target_handle: HANDLE) {
    let target_handle = handle_to_i64(target_handle);
    let mut state = PIPE_HANDLE_STATE
        .lock()
        .expect("_winapi pipe handle state mutex poisoned");
    if state.handles.contains(&source_handle) {
        state.handles.insert(target_handle);
        state.duplicate_sources.insert(target_handle, source_handle);
    }
}

#[cfg(windows)]
fn unregister_tracked_handle(handle: i64) {
    let mut state = PIPE_HANDLE_STATE
        .lock()
        .expect("_winapi pipe handle state mutex poisoned");
    state.handles.remove(&handle);
    state.duplicate_sources.remove(&handle);
    state
        .duplicate_sources
        .retain(|_, source_handle| *source_handle != handle);
}

#[cfg(windows)]
pub(crate) fn forget_tracked_pipe_handle(handle: i64) {
    unregister_tracked_handle(handle);
}

#[cfg(windows)]
fn take_pipe_duplicate_source_for_parent_close(
    duplicate_handle: i64,
    preserved_sources: &[i64],
) -> Option<i64> {
    let mut state = PIPE_HANDLE_STATE
        .lock()
        .expect("_winapi pipe handle state mutex poisoned");
    let source_handle = *state.duplicate_sources.get(&duplicate_handle)?;
    if preserved_sources.contains(&source_handle) {
        return None;
    }
    state.duplicate_sources.remove(&duplicate_handle);
    state.handles.remove(&source_handle);
    state
        .duplicate_sources
        .retain(|_, candidate| *candidate != source_handle);
    Some(source_handle)
}

#[cfg(windows)]
fn close_parent_pipe_source_for_duplicate(duplicate_handle: i64, preserved_sources: &[i64]) {
    let Some(source_handle) =
        take_pipe_duplicate_source_for_parent_close(duplicate_handle, preserved_sources)
    else {
        return;
    };
    let _ = unsafe { CloseHandle(handle_from_i64(source_handle)) };
}

#[cfg(windows)]
fn close_parent_pipe_sources_for_startup(startup: StartupInfoConfig) {
    if startup.dw_flags & STARTF_USESTDHANDLES_FLAG == 0 {
        return;
    }
    let startup_handles = [
        startup.h_std_input,
        startup.h_std_output,
        startup.h_std_error,
    ];
    for handle in startup_handles {
        close_parent_pipe_source_for_duplicate(handle, &startup_handles);
    }
}

#[cfg(all(windows, test))]
fn pipe_duplicate_source_for_test(duplicate_handle: i64) -> Option<i64> {
    PIPE_HANDLE_STATE
        .lock()
        .expect("_winapi pipe handle state mutex poisoned")
        .duplicate_sources
        .get(&duplicate_handle)
        .copied()
}

#[cfg(all(windows, test))]
fn pipe_handle_is_tracked_for_test(handle: i64) -> bool {
    PIPE_HANDLE_STATE
        .lock()
        .expect("_winapi pipe handle state mutex poisoned")
        .handles
        .contains(&handle)
}

#[inline]
fn tuple_value(items: Vec<Value>) -> Value {
    crate::alloc_managed_value(TupleObject::from_vec(items))
}

fn wide_null_terminated(text: &str, context: &str) -> Result<Vec<u16>, BuiltinError> {
    if text.encode_utf16().any(|unit| unit == 0) {
        return Err(BuiltinError::ValueError(format!(
            "{context} contains an embedded null character"
        )));
    }
    let mut wide: Vec<u16> = text.encode_utf16().collect();
    wide.push(0);
    Ok(wide)
}

fn environment_block(value: Value) -> Result<Option<Vec<u16>>, BuiltinError> {
    if value.is_none() {
        return Ok(None);
    }

    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "CreateProcess() environment must be a mapping or None, not {}",
            value.type_name()
        ))
    })?;
    if crate::ops::objects::extract_type_id(ptr) != TypeId::DICT {
        return Err(BuiltinError::TypeError(format!(
            "CreateProcess() environment must be a dict or None, not {}",
            value.type_name()
        )));
    }

    let dict = unsafe { &*(ptr as *const DictObject) };
    let mut entries = Vec::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let key = value_to_string(key).ok_or_else(|| {
            BuiltinError::TypeError("environment keys must be strings".to_string())
        })?;
        let value = value_to_string(value).ok_or_else(|| {
            BuiltinError::TypeError("environment values must be strings".to_string())
        })?;
        if key.is_empty() || key.contains('=') || key.contains('\0') {
            return Err(BuiltinError::ValueError(format!(
                "illegal environment variable name: {key:?}"
            )));
        }
        if value.contains('\0') {
            return Err(BuiltinError::ValueError(format!(
                "embedded null character in environment value for {key:?}"
            )));
        }
        entries.push((key, value));
    }

    entries.sort_by(|left, right| left.0.to_uppercase().cmp(&right.0.to_uppercase()));

    let mut block = Vec::new();
    for (key, value) in entries {
        block.extend(format!("{key}={value}").encode_utf16());
        block.push(0);
    }
    block.push(0);
    Ok(Some(block))
}

#[derive(Debug, Clone, Copy)]
struct StartupInfoConfig {
    dw_flags: u32,
    w_show_window: u16,
    h_std_input: i64,
    h_std_output: i64,
    h_std_error: i64,
}

impl Default for StartupInfoConfig {
    fn default() -> Self {
        Self {
            dw_flags: 0,
            w_show_window: 0,
            h_std_input: 0,
            h_std_output: 0,
            h_std_error: 0,
        }
    }
}

fn startup_info_config(value: Value) -> Result<StartupInfoConfig, BuiltinError> {
    if value.is_none() {
        return Ok(StartupInfoConfig::default());
    }

    Ok(StartupInfoConfig {
        dw_flags: startup_u32_property(value, "dwFlags", 0)?,
        w_show_window: startup_u16_property(value, "wShowWindow", 0)?,
        h_std_input: startup_handle_property(value, "hStdInput")?,
        h_std_output: startup_handle_property(value, "hStdOutput")?,
        h_std_error: startup_handle_property(value, "hStdError")?,
    })
}

fn startup_property(value: Value, name: &str) -> Result<Option<Value>, BuiltinError> {
    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "CreateProcess() startupinfo must be an object or None, not {}",
            value.type_name()
        ))
    })?;
    let type_id = crate::ops::objects::extract_type_id(ptr);
    if type_id == TypeId::TYPE {
        return Err(BuiltinError::TypeError(
            "CreateProcess() startupinfo must be an instance, not type".to_string(),
        ));
    }
    if type_id != TypeId::OBJECT && type_id.is_builtin() {
        return Err(BuiltinError::TypeError(format!(
            "CreateProcess() startupinfo must be an object or None, not {}",
            value.type_name()
        )));
    }

    let object = unsafe { &*(ptr as *const ShapedObject) };
    let key = Value::string(intern(name));
    if let Some(dict_value) = object.instance_dict_value() {
        if let Some(dict_ptr) = dict_value.as_object_ptr() {
            if crate::ops::objects::extract_type_id(dict_ptr) == TypeId::DICT {
                let dict = unsafe { &*(dict_ptr as *const DictObject) };
                if let Some(value) = dict.get(key) {
                    return Ok(Some(value));
                }
            }
        }
    }
    Ok(object.get_property(name))
}

fn startup_u32_property(value: Value, name: &str, default: u32) -> Result<u32, BuiltinError> {
    match startup_property(value, name)? {
        Some(value) if !value.is_none() => u32_arg(value, "CreateProcess", name),
        _ => Ok(default),
    }
}

fn startup_u16_property(value: Value, name: &str, default: u16) -> Result<u16, BuiltinError> {
    let integer = match startup_property(value, name)? {
        Some(value) if !value.is_none() => integer_arg(value, "CreateProcess", name)?,
        _ => return Ok(default),
    };
    u16::try_from(integer).map_err(|_| {
        BuiltinError::OverflowError(format!(
            "CreateProcess() {name} is outside the unsigned short range"
        ))
    })
}

fn startup_handle_property(value: Value, name: &str) -> Result<i64, BuiltinError> {
    match startup_property(value, name)? {
        Some(value) if !value.is_none() => integer_arg(value, "CreateProcess", name),
        _ => Ok(0),
    }
}

#[cfg(windows)]
fn last_os_error(context: &str) -> BuiltinError {
    let code = unsafe { GetLastError() };
    if code == 0 {
        BuiltinError::OSError(format!("{context}: {}", std::io::Error::last_os_error()))
    } else {
        BuiltinError::OSError(format!(
            "{context}: {}",
            std::io::Error::from_raw_os_error(code as i32)
        ))
    }
}

fn winapi_close_handle(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_exact_arity(args, "CloseHandle", 1)?;
    let handle = integer_arg(args[0], "CloseHandle", "handle")?;

    #[cfg(windows)]
    {
        if unsafe { CloseHandle(handle_from_i64(handle)) } == 0 {
            return Err(last_os_error("CloseHandle() failed"));
        }
        unregister_tracked_handle(handle);
    }

    Ok(Value::none())
}

fn winapi_create_pipe(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_exact_arity(args, "CreatePipe", 2)?;
    if !args[0].is_none() {
        return Err(BuiltinError::NotImplemented(
            "CreatePipe() security attributes are not supported yet".to_string(),
        ));
    }
    let size = u32_arg(args[1], "CreatePipe", "size")?;

    #[cfg(windows)]
    {
        let mut read_pipe: HANDLE = std::ptr::null_mut();
        let mut write_pipe: HANDLE = std::ptr::null_mut();
        if unsafe { CreatePipe(&mut read_pipe, &mut write_pipe, std::ptr::null(), size) } == 0 {
            return Err(last_os_error("CreatePipe() failed"));
        }
        track_pipe_handles(read_pipe, write_pipe);
        return Ok(tuple_value(vec![
            handle_value(read_pipe)?,
            handle_value(write_pipe)?,
        ]));
    }

    #[cfg(not(windows))]
    {
        let _ = size;
        Ok(tuple_value(vec![
            Value::int(0).expect("handle should fit"),
            Value::int(0).expect("handle should fit"),
        ]))
    }
}

fn winapi_create_process(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_exact_arity(args, "CreateProcess", 9)?;
    let application_name = optional_string_arg(args[0], "CreateProcess", "application_name")?;
    let command_line = string_arg(args[1], "CreateProcess", "command_line")?;
    if !args[2].is_none() {
        return Err(BuiltinError::NotImplemented(
            "CreateProcess() process security attributes are not supported yet".to_string(),
        ));
    }
    if !args[3].is_none() {
        return Err(BuiltinError::NotImplemented(
            "CreateProcess() thread security attributes are not supported yet".to_string(),
        ));
    }
    let inherit_handles = bool_arg(args[4], "CreateProcess", "inherit_handles")?;
    let mut creation_flags = u32_arg(args[5], "CreateProcess", "creation_flags")?;
    let environment = environment_block(args[6])?;
    if environment.is_some() {
        creation_flags |= CREATE_UNICODE_ENVIRONMENT_FLAG;
    }
    let current_directory = optional_string_arg(args[7], "CreateProcess", "current_directory")?;
    let startup = startup_info_config(args[8])?;

    #[cfg(windows)]
    {
        let application_name = application_name
            .as_deref()
            .map(|value| wide_null_terminated(value, "application_name"))
            .transpose()?;
        let mut command_line = wide_null_terminated(&command_line, "command_line")?;
        let current_directory = current_directory
            .as_deref()
            .map(|value| wide_null_terminated(value, "current_directory"))
            .transpose()?;

        let mut startup_info: STARTUPINFOW = unsafe { std::mem::zeroed() };
        startup_info.cb = std::mem::size_of::<STARTUPINFOW>() as u32;
        startup_info.dwFlags = startup.dw_flags;
        startup_info.wShowWindow = startup.w_show_window;
        if startup.dw_flags & STARTF_USESTDHANDLES_FLAG != 0 {
            startup_info.hStdInput = handle_from_i64(startup.h_std_input);
            startup_info.hStdOutput = handle_from_i64(startup.h_std_output);
            startup_info.hStdError = handle_from_i64(startup.h_std_error);
        }

        let mut process_info: PROCESS_INFORMATION = unsafe { std::mem::zeroed() };
        let success = unsafe {
            CreateProcessW(
                application_name
                    .as_ref()
                    .map_or(std::ptr::null(), |value| value.as_ptr()),
                command_line.as_mut_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                if inherit_handles { 1 } else { 0 },
                creation_flags,
                environment
                    .as_ref()
                    .map_or(std::ptr::null(), |value| value.as_ptr().cast()),
                current_directory
                    .as_ref()
                    .map_or(std::ptr::null(), |value| value.as_ptr()),
                &startup_info,
                &mut process_info,
            )
        };
        if success == 0 {
            return Err(last_os_error("CreateProcess() failed"));
        }
        close_parent_pipe_sources_for_startup(startup);

        return Ok(tuple_value(vec![
            handle_value(process_info.hProcess)?,
            handle_value(process_info.hThread)?,
            int_value(i64::from(process_info.dwProcessId), "process id")?,
            int_value(i64::from(process_info.dwThreadId), "thread id")?,
        ]));
    }

    #[cfg(not(windows))]
    {
        let _ = (
            application_name,
            command_line,
            inherit_handles,
            creation_flags,
            environment,
            current_directory,
            startup,
        );
        Err(BuiltinError::NotImplemented(
            "CreateProcess() is only available on Windows".to_string(),
        ))
    }
}

fn winapi_duplicate_handle(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_exact_arity(args, "DuplicateHandle", 6)?;
    let source_process = integer_arg(args[0], "DuplicateHandle", "source_process_handle")?;
    let source_handle = integer_arg(args[1], "DuplicateHandle", "source_handle")?;
    let target_process = integer_arg(args[2], "DuplicateHandle", "target_process_handle")?;
    let desired_access = u32_arg(args[3], "DuplicateHandle", "desired_access")?;
    let inherit_handle = bool_arg(args[4], "DuplicateHandle", "inherit_handle")?;
    let options = u32_arg(args[5], "DuplicateHandle", "options")?;

    #[cfg(windows)]
    {
        let mut target_handle: HANDLE = std::ptr::null_mut();
        if unsafe {
            DuplicateHandle(
                handle_from_i64(source_process),
                handle_from_i64(source_handle),
                handle_from_i64(target_process),
                &mut target_handle,
                desired_access,
                if inherit_handle { 1 } else { 0 },
                options,
            )
        } == 0
        {
            return Err(last_os_error("DuplicateHandle() failed"));
        }
        track_pipe_duplicate(source_handle, target_handle);
        return handle_value(target_handle);
    }

    #[cfg(not(windows))]
    {
        let _ = (
            source_process,
            target_process,
            desired_access,
            inherit_handle,
            options,
        );
        int_value(source_handle, "Windows handle")
    }
}

fn winapi_get_acp(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "GetACP() takes 0 positional arguments but {} were given",
            args.len()
        )));
    }

    #[cfg(windows)]
    {
        return Ok(Value::int(unsafe { GetACP() } as i64).expect("ACP should fit in i64"));
    }

    #[allow(unreachable_code)]
    Ok(Value::int(65001).expect("UTF-8 code page should fit in i64"))
}

fn winapi_get_current_process(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_exact_arity(args, "GetCurrentProcess", 0)?;

    #[cfg(windows)]
    {
        return handle_value(unsafe { GetCurrentProcess() });
    }

    #[cfg(not(windows))]
    {
        int_value(-1, "Windows handle")
    }
}

fn winapi_get_exit_code_process(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_exact_arity(args, "GetExitCodeProcess", 1)?;
    let handle = integer_arg(args[0], "GetExitCodeProcess", "handle")?;

    #[cfg(windows)]
    {
        let mut exit_code = 0u32;
        if unsafe { GetExitCodeProcess(handle_from_i64(handle), &mut exit_code) } == 0 {
            return Err(last_os_error("GetExitCodeProcess() failed"));
        }
        return int_value(i64::from(exit_code), "process exit code");
    }

    #[cfg(not(windows))]
    {
        let _ = handle;
        int_value(0, "process exit code")
    }
}

fn winapi_get_file_type(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_exact_arity(args, "GetFileType", 1)?;
    let handle = integer_arg(args[0], "GetFileType", "handle")?;

    #[cfg(windows)]
    {
        return int_value(
            i64::from(unsafe { GetFileType(handle_from_i64(handle)) }),
            "file type",
        );
    }

    #[cfg(not(windows))]
    {
        let _ = handle;
        int_value(i64::from(FILE_TYPE_UNKNOWN), "file type")
    }
}

fn winapi_get_std_handle(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_exact_arity(args, "GetStdHandle", 1)?;
    let std_handle = integer_arg(args[0], "GetStdHandle", "std_handle")?;
    if !(i64::from(i32::MIN)..=i64::from(u32::MAX)).contains(&std_handle) {
        return Err(BuiltinError::OverflowError(
            "GetStdHandle() std_handle is outside the Windows DWORD range".to_string(),
        ));
    }
    let std_handle = if std_handle < 0 {
        (std_handle as i32) as u32
    } else {
        std_handle as u32
    };

    #[cfg(windows)]
    {
        let handle = unsafe { GetStdHandle(std_handle) };
        if handle.is_null() {
            return Ok(Value::none());
        }
        if handle == INVALID_HANDLE_VALUE {
            return Err(last_os_error("GetStdHandle() failed"));
        }
        return handle_value(handle);
    }

    #[cfg(not(windows))]
    {
        int_value(i64::from(std_handle), "Windows handle")
    }
}

fn winapi_terminate_process(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_exact_arity(args, "TerminateProcess", 2)?;
    let handle = integer_arg(args[0], "TerminateProcess", "handle")?;
    let exit_code = u32_arg(args[1], "TerminateProcess", "exit_code")?;

    #[cfg(windows)]
    {
        if unsafe { TerminateProcess(handle_from_i64(handle), exit_code) } == 0 {
            return Err(last_os_error("TerminateProcess() failed"));
        }
    }

    #[cfg(not(windows))]
    {
        let _ = (handle, exit_code);
    }

    Ok(Value::none())
}

fn winapi_wait_for_single_object(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_exact_arity(args, "WaitForSingleObject", 2)?;
    let handle = integer_arg(args[0], "WaitForSingleObject", "handle")?;
    let timeout = u32_arg(args[1], "WaitForSingleObject", "milliseconds")?;

    #[cfg(windows)]
    {
        let result = unsafe { WaitForSingleObject(handle_from_i64(handle), timeout) };
        if result == WAIT_FAILED {
            return Err(last_os_error("WaitForSingleObject() failed"));
        }
        return int_value(i64::from(result), "wait result");
    }

    #[cfg(not(windows))]
    {
        let _ = (handle, timeout);
        int_value(0, "wait result")
    }
}

fn winapi_need_current_directory_for_exe_path(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_exact_arity(args, "NeedCurrentDirectoryForExePath", 1)?;
    let path = value_to_string(args[0]).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "NeedCurrentDirectoryForExePath() argument must be str, not {}",
            args[0].type_name()
        ))
    })?;

    let candidate = Path::new(&path);
    let has_explicit_directory = candidate.has_root()
        || candidate
            .parent()
            .is_some_and(|parent| parent != Path::new(""));
    let env_allows_current_directory =
        std::env::var_os("NoDefaultCurrentDirectoryInExePath").is_none();

    Ok(Value::bool(
        !has_explicit_directory && env_allows_current_directory,
    ))
}

#[cfg(test)]
mod tests {
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
        assert!(
            matches!(err, BuiltinError::TypeError(message) if message.contains("command_line"))
        );
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
        let duplicate_stdout_handle = value_to_i64(duplicate_stdout)
            .expect("duplicate stdout handle should be integer-backed");
        let duplicate_stderr_handle = value_to_i64(duplicate_stderr)
            .expect("duplicate stderr handle should be integer-backed");
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
        assert!(
            handle.as_int().is_some() || prism_runtime::types::int::value_to_i64(handle).is_some()
        );
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
}
