//! Minimal native `winreg` bootstrap surface.
//!
//! Prism does not currently expose live Windows registry access, but CPython's
//! Windows stdlib imports `winreg` during importlib bootstrap. Providing the
//! module and the expected constants keeps those imports working while callers
//! gracefully fall back when registry queries raise `OSError`.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use prism_core::Value;
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
];

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
fn registry_unavailable(fn_name: &str) -> BuiltinError {
    BuiltinError::OSError(format!(
        "{fn_name}() Windows registry access is not available in Prism yet"
    ))
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
    Err(registry_unavailable("OpenKey"))
}

fn winreg_open_key_ex(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_arity_range(args, "OpenKeyEx", 2, 4)?;
    Err(registry_unavailable("OpenKeyEx"))
}

fn winreg_query_value(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_arity_range(args, "QueryValue", 2, 2)?;
    Err(registry_unavailable("QueryValue"))
}

fn winreg_query_value_ex(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_arity_range(args, "QueryValueEx", 2, 2)?;
    Err(registry_unavailable("QueryValueEx"))
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
    Ok(Value::none())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn builtin_from_value(value: Value) -> &'static BuiltinFunctionObject {
        let ptr = value
            .as_object_ptr()
            .expect("expected builtin function object");
        unsafe { &*(ptr as *const BuiltinFunctionObject) }
    }

    #[test]
    fn test_winreg_module_exposes_importlib_bootstrap_surface() {
        let module = WinregModule::new();

        assert!(module.get_attr("HKEY_CURRENT_USER").is_ok());
        assert!(module.get_attr("HKEY_LOCAL_MACHINE").is_ok());
        assert!(module.get_attr("OpenKey").is_ok());
        assert!(module.get_attr("OpenKeyEx").is_ok());
        assert!(module.get_attr("QueryValue").is_ok());
        assert!(module.get_attr("QueryValueEx").is_ok());
        assert!(module.get_attr("CloseKey").is_ok());
    }

    #[test]
    fn test_open_key_reports_registry_unavailable_as_oserror() {
        let module = WinregModule::new();
        let open_key = builtin_from_value(module.get_attr("OpenKey").unwrap());
        let err = open_key
            .call(&[Value::int(0).unwrap(), Value::int(0).unwrap()])
            .expect_err("OpenKey should raise OSError");
        assert!(matches!(err, BuiltinError::OSError(_)));
    }

    #[test]
    fn test_close_key_is_a_no_op_for_bootstrap_cleanup_paths() {
        let module = WinregModule::new();
        let close_key = builtin_from_value(module.get_attr("CloseKey").unwrap());
        let result = close_key
            .call(&[Value::int(0).unwrap()])
            .expect("CloseKey should succeed");
        assert!(result.is_none());
    }
}
