//! Native subset of CPython's `locale` module.
//!
//! Prism currently runs with C-locale formatting semantics. This module exposes
//! the small API surface needed by early CPython regression tests while keeping
//! locale-sensitive hot formatting paths inside the runtime.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{
    BuiltinError, BuiltinFunctionObject, exception_type_value_for_id, percent_format_string,
};
use crate::stdlib::exceptions::ExceptionTypeId;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::value_as_string_ref;
use std::sync::{Arc, LazyLock, RwLock};

const LC_CTYPE: i64 = 0;
const LC_NUMERIC: i64 = 1;
const LC_TIME: i64 = 2;
const LC_COLLATE: i64 = 3;
const LC_MONETARY: i64 = 4;
const LC_MESSAGES: i64 = 5;
const LC_ALL: i64 = 6;

static CURRENT_LOCALE: LazyLock<RwLock<String>> = LazyLock::new(|| RwLock::new("C".to_string()));
static SETLOCALE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("locale.setlocale"), setlocale));
static LOCALECONV_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("locale.localeconv"), localeconv));
static FORMAT_STRING_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(Arc::from("locale.format_string"), format_string)
});
static GETPREFERREDENCODING_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("locale.getpreferredencoding"),
        getpreferredencoding,
    )
});
static GETENCODING_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("locale.getencoding"), getencoding));

/// Native `locale` module descriptor.
#[derive(Debug, Clone)]
pub struct LocaleModule {
    attrs: Vec<Arc<str>>,
}

impl LocaleModule {
    /// Create a new `locale` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("Error"),
                Arc::from("LC_ALL"),
                Arc::from("LC_COLLATE"),
                Arc::from("LC_CTYPE"),
                Arc::from("LC_MESSAGES"),
                Arc::from("LC_MONETARY"),
                Arc::from("LC_NUMERIC"),
                Arc::from("LC_TIME"),
                Arc::from("format_string"),
                Arc::from("getencoding"),
                Arc::from("getpreferredencoding"),
                Arc::from("localeconv"),
                Arc::from("setlocale"),
            ],
        }
    }
}

impl Default for LocaleModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for LocaleModule {
    fn name(&self) -> &str {
        "locale"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "Error" => Ok(
                exception_type_value_for_id(ExceptionTypeId::ValueError as u16)
                    .expect("ValueError exception type is registered"),
            ),
            "LC_ALL" => Ok(int_value(LC_ALL)),
            "LC_COLLATE" => Ok(int_value(LC_COLLATE)),
            "LC_CTYPE" => Ok(int_value(LC_CTYPE)),
            "LC_MESSAGES" => Ok(int_value(LC_MESSAGES)),
            "LC_MONETARY" => Ok(int_value(LC_MONETARY)),
            "LC_NUMERIC" => Ok(int_value(LC_NUMERIC)),
            "LC_TIME" => Ok(int_value(LC_TIME)),
            "format_string" => Ok(builtin_value(&FORMAT_STRING_FUNCTION)),
            "getencoding" => Ok(builtin_value(&GETENCODING_FUNCTION)),
            "getpreferredencoding" => Ok(builtin_value(&GETPREFERREDENCODING_FUNCTION)),
            "localeconv" => Ok(builtin_value(&LOCALECONV_FUNCTION)),
            "setlocale" => Ok(builtin_value(&SETLOCALE_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'locale' has no attribute '{}'",
                name
            ))),
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
fn int_value(value: i64) -> Value {
    Value::int(value).expect("locale constant fits tagged int")
}

fn setlocale(args: &[Value]) -> Result<Value, BuiltinError> {
    match args.len() {
        1 => Ok(current_locale_value()),
        2 => {
            if args[1].is_none() {
                return Ok(current_locale_value());
            }
            let locale = value_as_string_ref(args[1]).ok_or_else(|| {
                BuiltinError::TypeError("setlocale() locale must be str or None".to_string())
            })?;
            let normalized = if locale.as_str().is_empty() {
                "C".to_string()
            } else {
                locale.as_str().to_string()
            };
            *CURRENT_LOCALE.write().expect("locale lock poisoned") = normalized.clone();
            Ok(Value::string(intern(&normalized)))
        }
        count => Err(BuiltinError::TypeError(format!(
            "setlocale() takes 1 or 2 arguments ({count} given)"
        ))),
    }
}

fn localeconv(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "localeconv() takes no arguments ({} given)",
            args.len()
        )));
    }

    let mut dict = DictObject::with_capacity(18);
    dict.set(str_key("decimal_point"), Value::string(intern(".")));
    dict.set(str_key("thousands_sep"), Value::string(intern("")));
    dict.set(
        str_key("grouping"),
        crate::alloc_managed_value(ListObject::new()),
    );
    dict.set(str_key("int_curr_symbol"), Value::string(intern("")));
    dict.set(str_key("currency_symbol"), Value::string(intern("")));
    dict.set(str_key("mon_decimal_point"), Value::string(intern("")));
    dict.set(str_key("mon_thousands_sep"), Value::string(intern("")));
    dict.set(
        str_key("mon_grouping"),
        crate::alloc_managed_value(ListObject::new()),
    );
    for key in [
        "positive_sign",
        "negative_sign",
        "p_cs_precedes",
        "n_cs_precedes",
        "p_sep_by_space",
        "n_sep_by_space",
        "p_sign_posn",
        "n_sign_posn",
        "frac_digits",
        "int_frac_digits",
    ] {
        dict.set(str_key(key), Value::string(intern("")));
    }
    Ok(crate::alloc_managed_value(dict))
}

fn format_string(args: &[Value], _keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "format_string() takes 2 or 3 positional arguments ({} given)",
            args.len()
        )));
    }
    let format = value_as_string_ref(args[0])
        .ok_or_else(|| BuiltinError::TypeError("format must be str".to_string()))?;
    percent_format_string(format.as_str(), args[1])
}

fn getpreferredencoding(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "getpreferredencoding() takes at most one argument ({} given)",
            args.len()
        )));
    }
    Ok(Value::string(intern("utf-8")))
}

fn getencoding(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "getencoding() takes no arguments ({} given)",
            args.len()
        )));
    }
    Ok(Value::string(intern("utf-8")))
}

#[inline]
fn current_locale_value() -> Value {
    let locale = CURRENT_LOCALE.read().expect("locale lock poisoned");
    Value::string(intern(locale.as_str()))
}

#[inline]
fn str_key(key: &str) -> Value {
    Value::string(intern(key))
}
