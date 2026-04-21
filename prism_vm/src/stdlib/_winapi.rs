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
use prism_runtime::types::string::StringObject;
use rustc_hash::FxHashMap;
use std::path::Path;
use std::sync::{Arc, LazyLock};

#[cfg(windows)]
use windows_sys::Win32::Globalization::GetACP;

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
    ("DETACHED_PROCESS", 0x0000_0008),
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

static CLOSE_HANDLE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_winapi.CloseHandle"), winapi_close_handle)
});
static GET_ACP_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_winapi.GetACP"), winapi_get_acp));
static NEED_CURDIR_FOR_EXE_PATH_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_winapi.NeedCurrentDirectoryForExePath"),
        winapi_need_current_directory_for_exe_path,
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
        let mut attrs = Vec::with_capacity(EXPORTED_CONSTANTS.len() + 3);
        let mut values = FxHashMap::default();

        for &(name, value) in EXPORTED_CONSTANTS {
            let exported = Value::int(value).expect("_winapi constant should fit in tagged int");
            attrs.push(Arc::from(name));
            values.insert(Arc::from(name), exported);
        }

        for name in ["CloseHandle", "GetACP", "NeedCurrentDirectoryForExePath"] {
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
            "GetACP" => Ok(builtin_value(&GET_ACP_FUNCTION)),
            "NeedCurrentDirectoryForExePath" => {
                Ok(builtin_value(&NEED_CURDIR_FOR_EXE_PATH_FUNCTION))
            }
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

fn winapi_close_handle(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_exact_arity(args, "CloseHandle", 1)?;
    if args[0].as_int().is_none() && args[0].as_bool().is_none() {
        return Err(BuiltinError::TypeError(format!(
            "CloseHandle() argument must be an integer handle, not {}",
            args[0].type_name()
        )));
    }

    // Prism does not yet materialize Windows kernel handles itself. Accept the
    // cleanup request as a no-op so stdlib bootstrap code can safely finalize
    // placeholder handles during import paths such as `subprocess`.
    Ok(Value::none())
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

    #[test]
    fn test_winapi_module_exposes_bootstrap_constants_and_functions() {
        let module = WinApiModule::new();

        assert!(module.get_attr("CREATE_NEW_CONSOLE").is_ok());
        assert!(module.get_attr("STD_OUTPUT_HANDLE").is_ok());
        assert!(module.get_attr("CloseHandle").is_ok());
        assert!(module.get_attr("GetACP").is_ok());
        assert!(module.get_attr("NeedCurrentDirectoryForExePath").is_ok());
    }

    #[test]
    fn test_close_handle_accepts_integer_handles_as_no_op() {
        let result = winapi_close_handle(&[Value::int(42).expect("tagged int should fit")])
            .expect("CloseHandle should accept integer handles");
        assert!(result.is_none());
    }

    #[test]
    fn test_close_handle_rejects_non_integer_handles() {
        let err = winapi_close_handle(&[Value::string(intern("pipe"))])
            .expect_err("CloseHandle should validate argument types");
        assert!(matches!(err, BuiltinError::TypeError(_)));
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
