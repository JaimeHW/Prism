//! Minimal native `winreg` bootstrap surface.
//!
//! Prism does not currently expose live Windows registry access, but CPython's
//! Windows stdlib imports `winreg` during importlib bootstrap. Providing the
//! module and the expected constants keeps those imports working while callers
//! gracefully fall back when registry queries raise `OSError`.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::string::value_as_string_ref;
use prism_runtime::types::tuple::TupleObject;
use rustc_hash::FxHashMap;
use std::sync::{Arc, LazyLock};

const EXPORTED_CONSTANTS: &[(&str, i64)] = &[
    ("HKEY_CLASSES_ROOT", 0x8000_0000),
    ("HKEY_CURRENT_USER", 0x8000_0001),
    ("HKEY_LOCAL_MACHINE", 0x8000_0002),
    ("KEY_READ", 0x0002_0019),
    ("KEY_WRITE", 0x0002_0006),
    ("KEY_ENUMERATE_SUB_KEYS", 0x0000_0008),
    ("KEY_WOW64_64KEY", 0x0000_0100),
    ("KEY_WOW64_32KEY", 0x0000_0200),
    ("REG_SZ", 1),
    ("REG_EXPAND_SZ", 2),
    ("REG_BINARY", 3),
    ("REG_DWORD", 4),
];

const KEY_HANDLE_ATTR: &str = "__winreg_handle";

static CLOSE_KEY_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("winreg.CloseKey"), winreg_close_key));
static CREATE_KEY_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("winreg.CreateKey"), winreg_create_key));
static ENUM_KEY_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("winreg.EnumKey"), winreg_enum_key));
static ENUM_VALUE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("winreg.EnumValue"), winreg_enum_value));
static OPEN_KEY_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("winreg.OpenKey"), winreg_open_key));
static OPEN_KEY_EX_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("winreg.OpenKeyEx"), winreg_open_key_ex));
static QUERY_VALUE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("winreg.QueryValue"), winreg_query_value)
});
static QUERY_VALUE_EX_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("winreg.QueryValueEx"), winreg_query_value_ex)
});
static KEY_ENTER_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("winreg.PyHKEY.__enter__"), winreg_key_enter)
});
static KEY_EXIT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("winreg.PyHKEY.__exit__"), winreg_key_exit)
});

/// Native `winreg` module descriptor.
#[derive(Debug, Clone)]
pub struct WinregModule {
    attrs: Vec<Arc<str>>,
    values: FxHashMap<Arc<str>, Value>,
}

impl WinregModule {
    /// Create a new `winreg` module descriptor.
    pub fn new() -> Self {
        let mut attrs = Vec::with_capacity(EXPORTED_CONSTANTS.len() + 8);
        let mut values = FxHashMap::default();

        for &(name, value) in EXPORTED_CONSTANTS {
            let exported = Value::int(value).expect("winreg constant should fit in tagged int");
            attrs.push(Arc::from(name));
            values.insert(Arc::from(name), exported);
        }

        for name in [
            "CloseKey",
            "CreateKey",
            "EnumKey",
            "EnumValue",
            "OpenKey",
            "OpenKeyEx",
            "QueryValue",
            "QueryValueEx",
        ] {
            attrs.push(Arc::from(name));
        }
        attrs.sort_unstable();

        Self { attrs, values }
    }
}

impl Default for WinregModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for WinregModule {
    fn name(&self) -> &str {
        "winreg"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "CloseKey" => Ok(builtin_value(&CLOSE_KEY_FUNCTION)),
            "CreateKey" => Ok(builtin_value(&CREATE_KEY_FUNCTION)),
            "EnumKey" => Ok(builtin_value(&ENUM_KEY_FUNCTION)),
            "EnumValue" => Ok(builtin_value(&ENUM_VALUE_FUNCTION)),
            "OpenKey" => Ok(builtin_value(&OPEN_KEY_FUNCTION)),
            "OpenKeyEx" => Ok(builtin_value(&OPEN_KEY_EX_FUNCTION)),
            "QueryValue" => Ok(builtin_value(&QUERY_VALUE_FUNCTION)),
            "QueryValueEx" => Ok(builtin_value(&QUERY_VALUE_EX_FUNCTION)),
            _ => self.values.get(name).copied().ok_or_else(|| {
                ModuleError::AttributeError(format!("module 'winreg' has no attribute '{}'", name))
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

#[inline]
fn leak_object_value<T: prism_runtime::Trace + 'static>(object: T) -> Value {
    crate::alloc_managed_value(object)
}

fn bound_builtin_attr_value(function: &'static BuiltinFunctionObject, receiver: Value) -> Value {
    let bound = function.bind(receiver);
    crate::alloc_managed_value(bound)
}

#[inline]
fn registry_unavailable(fn_name: &str) -> BuiltinError {
    BuiltinError::OSError(format!(
        "{fn_name}() Windows registry access is not available in Prism yet"
    ))
}

#[inline]
fn registry_os_error(fn_name: &str, status: u32) -> BuiltinError {
    let err = std::io::Error::from_raw_os_error(status as i32);
    BuiltinError::OSError(format!("{fn_name}() failed: {err}"))
}

#[inline]
fn string_arg(value: Value, fn_name: &str, arg_name: &str) -> Result<String, BuiltinError> {
    value_as_string_ref(value)
        .map(|text| text.as_str().to_string())
        .ok_or_else(|| BuiltinError::TypeError(format!("{fn_name}() {arg_name} must be str")))
}

#[inline]
fn int_arg(value: Value, fn_name: &str, arg_name: &str) -> Result<i64, BuiltinError> {
    value
        .as_int()
        .ok_or_else(|| BuiltinError::TypeError(format!("{fn_name}() {arg_name} must be int")))
}

fn key_handle_from_value(value: Value, fn_name: &str) -> Result<isize, BuiltinError> {
    if let Some(handle) = value.as_int() {
        return Ok(handle as isize);
    }

    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "{fn_name}() key must be an integer handle or PyHKEY object"
        )));
    };

    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != TypeId::OBJECT {
        return Err(BuiltinError::TypeError(format!(
            "{fn_name}() key must be a PyHKEY object"
        )));
    }
    let object = unsafe { &*(ptr as *const ShapedObject) };
    object
        .get_property(KEY_HANDLE_ATTR)
        .and_then(|value| value.as_int())
        .map(|handle| handle as isize)
        .ok_or_else(|| BuiltinError::TypeError(format!("{fn_name}() key must be a PyHKEY object")))
}

fn set_key_handle(value: Value, handle: isize) -> Result<(), BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Ok(());
    };

    let object = unsafe { &mut *(ptr as *mut ShapedObject) };
    let handle_value = Value::int(handle as i64).ok_or_else(|| {
        BuiltinError::OverflowError("registry handle is too large for Prism int".to_string())
    })?;
    object.set_property(intern(KEY_HANDLE_ATTR), handle_value, shape_registry());
    Ok(())
}

fn key_object_value(handle: isize) -> Result<Value, BuiltinError> {
    let handle_value = Value::int(handle as i64).ok_or_else(|| {
        BuiltinError::OverflowError("registry handle is too large for Prism int".to_string())
    })?;
    let registry = shape_registry();
    let mut key = Box::new(ShapedObject::new(TypeId::OBJECT, registry.empty_shape()));
    let receiver = Value::object_ptr(key.as_mut() as *mut ShapedObject as *const ());

    key.set_property(intern(KEY_HANDLE_ATTR), handle_value, registry);
    key.set_property(
        intern("__enter__"),
        bound_builtin_attr_value(&KEY_ENTER_FUNCTION, receiver),
        registry,
    );
    key.set_property(
        intern("__exit__"),
        bound_builtin_attr_value(&KEY_EXIT_FUNCTION, receiver),
        registry,
    );
    key.set_property(
        intern("Close"),
        bound_builtin_attr_value(&CLOSE_KEY_FUNCTION, receiver),
        registry,
    );

    Ok(Value::object_ptr(Box::into_raw(key) as *const ()))
}

#[inline]
fn is_predefined_hkey(handle: isize) -> bool {
    matches!((handle as u64) & 0xffff_ffff, 0x8000_0000..=0x8000_0006)
}

fn parse_arity_range(
    args: &[Value],
    fn_name: &str,
    min_expected: usize,
    max_expected: usize,
) -> Result<(), BuiltinError> {
    if (min_expected..=max_expected).contains(&args.len()) {
        Ok(())
    } else if min_expected == max_expected {
        Err(BuiltinError::TypeError(format!(
            "{fn_name}() takes exactly {min_expected} argument{} ({} given)",
            if min_expected == 1 { "" } else { "s" },
            args.len()
        )))
    } else {
        Err(BuiltinError::TypeError(format!(
            "{fn_name}() takes from {min_expected} to {max_expected} positional arguments ({} given)",
            args.len()
        )))
    }
}

fn winreg_open_key(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_arity_range(args, "OpenKey", 2, 4)?;
    open_key_impl(args, "OpenKey")
}

fn winreg_open_key_ex(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_arity_range(args, "OpenKeyEx", 2, 4)?;
    open_key_impl(args, "OpenKeyEx")
}

fn winreg_query_value(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_arity_range(args, "QueryValue", 2, 2)?;
    Err(registry_unavailable("QueryValue"))
}

fn winreg_query_value_ex(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_arity_range(args, "QueryValueEx", 2, 2)?;
    query_value_ex_impl(args)
}

fn winreg_enum_key(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_arity_range(args, "EnumKey", 2, 2)?;
    Err(registry_unavailable("EnumKey"))
}

fn winreg_enum_value(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_arity_range(args, "EnumValue", 2, 2)?;
    Err(registry_unavailable("EnumValue"))
}

fn winreg_create_key(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_arity_range(args, "CreateKey", 2, 2)?;
    Err(registry_unavailable("CreateKey"))
}

fn winreg_close_key(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_arity_range(args, "CloseKey", 1, 1)?;
    close_key_impl(args[0])?;
    Ok(Value::none())
}

fn winreg_key_enter(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_arity_range(args, "PyHKEY.__enter__", 1, 1)?;
    Ok(args[0])
}

fn winreg_key_exit(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_arity_range(args, "PyHKEY.__exit__", 4, 4)?;
    close_key_impl(args[0])?;
    Ok(Value::none())
}

#[cfg(windows)]
fn open_key_impl(args: &[Value], fn_name: &str) -> Result<Value, BuiltinError> {
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::ERROR_SUCCESS;
    use windows_sys::Win32::System::Registry::{HKEY, KEY_READ, RegOpenKeyExW};

    let root = key_handle_from_value(args[0], fn_name)? as HKEY;
    let sub_key = string_arg(args[1], fn_name, "sub_key")?;
    let reserved = args
        .get(2)
        .map(|value| int_arg(*value, fn_name, "reserved"))
        .transpose()?
        .unwrap_or(0);
    let access = args
        .get(3)
        .map(|value| int_arg(*value, fn_name, "access"))
        .transpose()?
        .unwrap_or(i64::from(KEY_READ));

    let reserved = u32::try_from(reserved)
        .map_err(|_| BuiltinError::OverflowError(format!("{fn_name}() reserved out of range")))?;
    let access = u32::try_from(access)
        .map_err(|_| BuiltinError::OverflowError(format!("{fn_name}() access out of range")))?;
    let wide_sub_key: Vec<u16> = OsStr::new(&sub_key)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let mut handle: HKEY = std::ptr::null_mut();
    let status =
        unsafe { RegOpenKeyExW(root, wide_sub_key.as_ptr(), reserved, access, &mut handle) };
    if status != ERROR_SUCCESS {
        return Err(registry_os_error(fn_name, status));
    }

    key_object_value(handle as isize)
}

#[cfg(not(windows))]
fn open_key_impl(_args: &[Value], fn_name: &str) -> Result<Value, BuiltinError> {
    Err(registry_unavailable(fn_name))
}

#[cfg(windows)]
fn query_value_ex_impl(args: &[Value]) -> Result<Value, BuiltinError> {
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::{ERROR_MORE_DATA, ERROR_SUCCESS};
    use windows_sys::Win32::System::Registry::{
        HKEY, REG_BINARY, REG_DWORD, REG_EXPAND_SZ, REG_SZ, RegQueryValueExW,
    };

    let handle = key_handle_from_value(args[0], "QueryValueEx")?;
    if handle == 0 {
        return Err(BuiltinError::OSError(
            "QueryValueEx() called with a closed registry key".to_string(),
        ));
    }
    let value_name = string_arg(args[1], "QueryValueEx", "value_name")?;
    let wide_value_name: Vec<u16> = OsStr::new(&value_name)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();

    let mut kind = 0_u32;
    let mut byte_len = 0_u32;
    let mut status = unsafe {
        RegQueryValueExW(
            handle as HKEY,
            wide_value_name.as_ptr(),
            std::ptr::null_mut(),
            &mut kind,
            std::ptr::null_mut(),
            &mut byte_len,
        )
    };
    if status != ERROR_SUCCESS && status != ERROR_MORE_DATA {
        return Err(registry_os_error("QueryValueEx", status));
    }

    let mut data = vec![0_u8; byte_len as usize];
    status = unsafe {
        RegQueryValueExW(
            handle as HKEY,
            wide_value_name.as_ptr(),
            std::ptr::null_mut(),
            &mut kind,
            data.as_mut_ptr(),
            &mut byte_len,
        )
    };
    if status != ERROR_SUCCESS {
        return Err(registry_os_error("QueryValueEx", status));
    }
    data.truncate(byte_len as usize);

    let value = match kind {
        REG_SZ | REG_EXPAND_SZ => Value::string(intern(&utf16_registry_string(&data))),
        REG_DWORD if data.len() >= 4 => {
            let number = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            Value::int(i64::from(number)).expect("REG_DWORD value fits in Prism int")
        }
        REG_BINARY => leak_object_value(BytesObject::from_vec(data)),
        _ => leak_object_value(BytesObject::from_vec(data)),
    };
    let kind_value = Value::int(i64::from(kind)).expect("registry value type fits in Prism int");
    Ok(leak_object_value(TupleObject::from_vec(vec![
        value, kind_value,
    ])))
}

#[cfg(not(windows))]
fn query_value_ex_impl(_args: &[Value]) -> Result<Value, BuiltinError> {
    Err(registry_unavailable("QueryValueEx"))
}

#[cfg(windows)]
fn close_key_impl(value: Value) -> Result<(), BuiltinError> {
    use windows_sys::Win32::Foundation::ERROR_SUCCESS;
    use windows_sys::Win32::System::Registry::{HKEY, RegCloseKey};

    let handle = key_handle_from_value(value, "CloseKey")?;
    if handle == 0 {
        return Ok(());
    }
    set_key_handle(value, 0)?;
    if is_predefined_hkey(handle) {
        return Ok(());
    }

    let status = unsafe { RegCloseKey(handle as HKEY) };
    if status != ERROR_SUCCESS {
        return Err(registry_os_error("CloseKey", status));
    }
    Ok(())
}

#[cfg(not(windows))]
fn close_key_impl(value: Value) -> Result<(), BuiltinError> {
    let _ = key_handle_from_value(value, "CloseKey")?;
    Ok(())
}

#[cfg(windows)]
fn utf16_registry_string(data: &[u8]) -> String {
    let mut units = data
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect::<Vec<_>>();
    while units.last().is_some_and(|unit| *unit == 0) {
        units.pop();
    }
    String::from_utf16_lossy(&units)
}

#[cfg(test)]
mod tests;
