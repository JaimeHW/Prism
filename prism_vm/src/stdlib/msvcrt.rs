//! Native Windows `msvcrt` bootstrap surface.
//!
//! CPython's pure-Python Windows stdlib uses the presence of `msvcrt` to select
//! Windows subprocess paths. Prism therefore exposes the small, real CRT and
//! process error-mode surface required during stdlib bootstrap while leaving
//! interactive console functions out of the module until they can be backed by
//! correct terminal behavior.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::types::int::value_to_i64;
use rustc_hash::FxHashMap;
use std::sync::{Arc, LazyLock};

#[cfg(not(windows))]
use std::sync::atomic::{AtomicU32, Ordering};

#[cfg(windows)]
use windows_sys::Win32::System::Diagnostics::Debug::{
    GetErrorMode, SEM_FAILCRITICALERRORS, SEM_NOALIGNMENTFAULTEXCEPT, SEM_NOGPFAULTERRORBOX,
    SEM_NOOPENFILEERRORBOX, SetErrorMode,
};

#[cfg(not(windows))]
const SEM_FAILCRITICALERRORS: u32 = 0x0001;
#[cfg(not(windows))]
const SEM_NOGPFAULTERRORBOX: u32 = 0x0002;
#[cfg(not(windows))]
const SEM_NOALIGNMENTFAULTEXCEPT: u32 = 0x0004;
#[cfg(not(windows))]
const SEM_NOOPENFILEERRORBOX: u32 = 0x8000;

#[cfg(not(windows))]
static ERROR_MODE: AtomicU32 = AtomicU32::new(0);

#[cfg(windows)]
#[link(name = "ucrt")]
unsafe extern "C" {
    #[link_name = "_get_osfhandle"]
    fn crt_get_osfhandle(fd: i32) -> isize;
    #[link_name = "_open_osfhandle"]
    fn crt_open_osfhandle(handle: isize, flags: i32) -> i32;
    #[link_name = "_setmode"]
    fn crt_setmode(fd: i32, mode: i32) -> i32;
}

const MODULE_DOC: &str = "Native bootstrap implementation of the msvcrt module.";

const O_TEXT: i64 = 0x4000;
const O_BINARY: i64 = 0x8000;

const EXPORTED_CONSTANTS: &[(&str, i64)] = &[
    ("LK_UNLCK", 0),
    ("LK_LOCK", 1),
    ("LK_NBLCK", 2),
    ("LK_RLCK", 3),
    ("LK_NBRLCK", 4),
    ("O_TEXT", O_TEXT),
    ("O_BINARY", O_BINARY),
    ("SEM_FAILCRITICALERRORS", SEM_FAILCRITICALERRORS as i64),
    ("SEM_NOGPFAULTERRORBOX", SEM_NOGPFAULTERRORBOX as i64),
    (
        "SEM_NOALIGNMENTFAULTEXCEPT",
        SEM_NOALIGNMENTFAULTEXCEPT as i64,
    ),
    ("SEM_NOOPENFILEERRORBOX", SEM_NOOPENFILEERRORBOX as i64),
];

const FUNCTION_NAMES: &[&str] = &[
    "GetErrorMode",
    "SetErrorMode",
    "get_osfhandle",
    "kbhit",
    "open_osfhandle",
    "setmode",
];

static GET_OSFHANDLE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("msvcrt.get_osfhandle"), msvcrt_get_osfhandle)
});
static OPEN_OSFHANDLE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("msvcrt.open_osfhandle"), msvcrt_open_osfhandle)
});
static SETMODE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("msvcrt.setmode"), msvcrt_setmode));
static GET_ERROR_MODE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("msvcrt.GetErrorMode"), msvcrt_get_error_mode)
});
static SET_ERROR_MODE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("msvcrt.SetErrorMode"), msvcrt_set_error_mode)
});
static KBHIT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("msvcrt.kbhit"), msvcrt_kbhit));

/// Native `msvcrt` module descriptor.
#[derive(Debug, Clone)]
pub struct MsvcrtModule {
    attrs: Vec<Arc<str>>,
    values: FxHashMap<Arc<str>, Value>,
}

impl MsvcrtModule {
    /// Create a new `msvcrt` module descriptor.
    pub fn new() -> Self {
        let mut attrs = Vec::with_capacity(EXPORTED_CONSTANTS.len() + FUNCTION_NAMES.len() + 1);
        let mut values = FxHashMap::default();

        attrs.push(Arc::from("__doc__"));
        for &(name, value) in EXPORTED_CONSTANTS {
            let exported = Value::int(value).expect("msvcrt constant should fit in tagged int");
            attrs.push(Arc::from(name));
            values.insert(Arc::from(name), exported);
        }
        for &name in FUNCTION_NAMES {
            attrs.push(Arc::from(name));
        }
        attrs.sort_unstable();

        Self { attrs, values }
    }
}

impl Default for MsvcrtModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for MsvcrtModule {
    fn name(&self) -> &str {
        "msvcrt"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__doc__" => Ok(Value::string(intern(MODULE_DOC))),
            "get_osfhandle" => Ok(builtin_value(&GET_OSFHANDLE_FUNCTION)),
            "open_osfhandle" => Ok(builtin_value(&OPEN_OSFHANDLE_FUNCTION)),
            "setmode" => Ok(builtin_value(&SETMODE_FUNCTION)),
            "GetErrorMode" => Ok(builtin_value(&GET_ERROR_MODE_FUNCTION)),
            "SetErrorMode" => Ok(builtin_value(&SET_ERROR_MODE_FUNCTION)),
            "kbhit" => Ok(builtin_value(&KBHIT_FUNCTION)),
            _ => self.values.get(name).copied().ok_or_else(|| {
                ModuleError::AttributeError(format!("module 'msvcrt' has no attribute '{}'", name))
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

fn i32_arg(value: Value, fn_name: &str, arg_name: &str) -> Result<i32, BuiltinError> {
    let integer = integer_arg(value, fn_name, arg_name)?;
    i32::try_from(integer).map_err(|_| {
        BuiltinError::OverflowError(format!("{fn_name}() {arg_name} is outside the C int range"))
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

fn int_value(value: i64, context: &str) -> Result<Value, BuiltinError> {
    Value::int(value).ok_or_else(|| BuiltinError::OverflowError(format!("{context} overflow")))
}

fn msvcrt_get_osfhandle(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_exact_arity(args, "get_osfhandle", 1)?;
    let fd = i32_arg(args[0], "get_osfhandle", "fd")?;

    #[cfg(windows)]
    {
        let handle = unsafe { crt_get_osfhandle(fd) };
        if handle == -1 {
            return Err(BuiltinError::OSError(format!(
                "get_osfhandle() failed for file descriptor {fd}"
            )));
        }
        return int_value(handle as i64, "Windows handle");
    }

    #[cfg(not(windows))]
    {
        int_value(i64::from(fd), "file descriptor")
    }
}

fn msvcrt_open_osfhandle(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_exact_arity(args, "open_osfhandle", 2)?;
    let handle = integer_arg(args[0], "open_osfhandle", "handle")?;
    let flags = i32_arg(args[1], "open_osfhandle", "flags")?;

    #[cfg(windows)]
    {
        let fd = unsafe { crt_open_osfhandle(handle as isize, flags) };
        if fd == -1 {
            return Err(BuiltinError::OSError(
                "open_osfhandle() failed for Windows handle".to_string(),
            ));
        }
        crate::stdlib::_winapi::forget_tracked_pipe_handle(handle);
        return int_value(i64::from(fd), "file descriptor");
    }

    #[cfg(not(windows))]
    {
        let _ = flags;
        int_value(handle, "file descriptor")
    }
}

fn msvcrt_setmode(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_exact_arity(args, "setmode", 2)?;
    let fd = i32_arg(args[0], "setmode", "fd")?;
    let mode = i32_arg(args[1], "setmode", "mode")?;

    #[cfg(windows)]
    {
        let previous = unsafe { crt_setmode(fd, mode) };
        if previous == -1 {
            return Err(BuiltinError::OSError(format!(
                "setmode() failed for file descriptor {fd}"
            )));
        }
        return int_value(i64::from(previous), "file mode");
    }

    #[cfg(not(windows))]
    {
        let _ = (fd, mode);
        int_value(O_TEXT, "file mode")
    }
}

fn msvcrt_get_error_mode(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_exact_arity(args, "GetErrorMode", 0)?;

    #[cfg(windows)]
    {
        return int_value(unsafe { GetErrorMode() } as i64, "error mode");
    }

    #[cfg(not(windows))]
    {
        int_value(i64::from(ERROR_MODE.load(Ordering::SeqCst)), "error mode")
    }
}

fn msvcrt_set_error_mode(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_exact_arity(args, "SetErrorMode", 1)?;
    let mode = u32_arg(args[0], "SetErrorMode", "mode")?;

    #[cfg(windows)]
    {
        return int_value(unsafe { SetErrorMode(mode) } as i64, "error mode");
    }

    #[cfg(not(windows))]
    {
        int_value(
            i64::from(ERROR_MODE.swap(mode, Ordering::SeqCst)),
            "error mode",
        )
    }
}

fn msvcrt_kbhit(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_exact_arity(args, "kbhit", 0)?;
    Ok(Value::bool(false))
}

#[cfg(test)]
mod tests;
