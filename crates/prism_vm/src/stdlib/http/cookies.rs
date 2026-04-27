//! Native `http.cookies` bootstrap surface.

use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::stdlib::{Module, ModuleError, ModuleResult};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::types::list::ListObject;
use std::sync::{Arc, LazyLock};

const EXPORTS: &[&str] = &["CookieError", "BaseCookie", "SimpleCookie", "Morsel"];

static COOKIE_ERROR_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("http.cookies.CookieError"), cookie_error)
});
static BASE_COOKIE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("http.cookies.BaseCookie"), base_cookie));
static SIMPLE_COOKIE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("http.cookies.SimpleCookie"), simple_cookie)
});
static MORSEL_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("http.cookies.Morsel"), morsel));

/// Native `http.cookies` module descriptor.
#[derive(Debug, Clone)]
pub struct CookiesModule {
    attrs: Vec<Arc<str>>,
    all: Value,
}

impl CookiesModule {
    /// Create a native `http.cookies` module.
    pub fn new() -> Self {
        Self {
            attrs: EXPORTS.iter().copied().map(Arc::from).collect(),
            all: string_list_value(EXPORTS),
        }
    }
}

impl Default for CookiesModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for CookiesModule {
    fn name(&self) -> &str {
        "http.cookies"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__all__" => Ok(self.all),
            "CookieError" => Ok(builtin_value(&COOKIE_ERROR_FUNCTION)),
            "BaseCookie" => Ok(builtin_value(&BASE_COOKIE_FUNCTION)),
            "SimpleCookie" => Ok(builtin_value(&SIMPLE_COOKIE_FUNCTION)),
            "Morsel" => Ok(builtin_value(&MORSEL_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'http.cookies' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        let mut attrs = self.attrs.clone();
        attrs.push(Arc::from("__all__"));
        attrs
    }
}

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

fn string_list_value(items: &[&str]) -> Value {
    let values = items
        .iter()
        .copied()
        .map(|item| Value::string(intern(item)))
        .collect::<Vec<_>>();
    crate::alloc_managed_value(ListObject::from_iter(values))
}

fn cookie_error(_args: &[Value]) -> Result<Value, BuiltinError> {
    Err(unsupported("CookieError"))
}

fn base_cookie(_args: &[Value]) -> Result<Value, BuiltinError> {
    Err(unsupported("BaseCookie"))
}

fn simple_cookie(_args: &[Value]) -> Result<Value, BuiltinError> {
    Err(unsupported("SimpleCookie"))
}

fn morsel(_args: &[Value]) -> Result<Value, BuiltinError> {
    Err(unsupported("Morsel"))
}

fn unsupported(name: &str) -> BuiltinError {
    BuiltinError::NotImplemented(format!(
        "http.cookies.{name} is not implemented in Prism yet"
    ))
}
