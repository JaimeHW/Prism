//! Native `_sre` compatibility bridge for CPython's pure-Python `re` stack.
//!
//! Prism keeps a single native regex engine and exposes the subset of `_sre`
//! that CPython's `Lib/re/*.py` expects while importing and compiling regexes.

use super::{Module, ModuleError, ModuleResult};
use crate::VirtualMachine;
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::stdlib::re::builtin_compile as re_builtin_compile;
use num_traits::ToPrimitive;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::int::value_to_bigint;
use std::sync::{Arc, LazyLock};

const MODULE_DOC: &str = "Prism native compatibility bridge for CPython's _sre module.";
const COPYRIGHT: &str = "Prism _sre compatibility bridge";
const MAGIC: i64 = 20221023;
const CODESIZE: i64 = 4;
const MAXREPEAT: i64 = u32::MAX as i64;
const MAXGROUPS: i64 = (i32::MAX as i64) / 2;

static COMPILE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("_sre.compile"), sre_compile));
static TEMPLATE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_sre.template"), sre_template));
static GETCODESIZE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_sre.getcodesize"), sre_getcodesize));
static ASCII_ISCASED_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_sre.ascii_iscased"), sre_ascii_iscased)
});
static UNICODE_ISCASED_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_sre.unicode_iscased"), sre_unicode_iscased)
});
static ASCII_TOLOWER_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_sre.ascii_tolower"), sre_ascii_tolower)
});
static UNICODE_TOLOWER_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_sre.unicode_tolower"), sre_unicode_tolower)
});

/// Native `_sre` module descriptor.
#[derive(Debug, Clone)]
pub struct SreModule {
    attrs: Vec<Arc<str>>,
}

impl SreModule {
    /// Create a new `_sre` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("__doc__"),
                Arc::from("MAGIC"),
                Arc::from("CODESIZE"),
                Arc::from("MAXREPEAT"),
                Arc::from("MAXGROUPS"),
                Arc::from("copyright"),
                Arc::from("compile"),
                Arc::from("template"),
                Arc::from("getcodesize"),
                Arc::from("ascii_iscased"),
                Arc::from("unicode_iscased"),
                Arc::from("ascii_tolower"),
                Arc::from("unicode_tolower"),
            ],
        }
    }
}

impl Default for SreModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for SreModule {
    fn name(&self) -> &str {
        "_sre"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__doc__" => Ok(Value::string(intern(MODULE_DOC))),
            "MAGIC" => Ok(Value::int(MAGIC).expect("SRE magic should fit in Value::int")),
            "CODESIZE" => Ok(Value::int(CODESIZE).expect("SRE code size should fit in Value::int")),
            "MAXREPEAT" => Ok(Value::int(MAXREPEAT).expect("MAXREPEAT should fit in Value::int")),
            "MAXGROUPS" => Ok(Value::int(MAXGROUPS).expect("MAXGROUPS should fit in Value::int")),
            "copyright" => Ok(Value::string(intern(COPYRIGHT))),
            "compile" => Ok(builtin_value(&COMPILE_FUNCTION)),
            "template" => Ok(builtin_value(&TEMPLATE_FUNCTION)),
            "getcodesize" => Ok(builtin_value(&GETCODESIZE_FUNCTION)),
            "ascii_iscased" => Ok(builtin_value(&ASCII_ISCASED_FUNCTION)),
            "unicode_iscased" => Ok(builtin_value(&UNICODE_ISCASED_FUNCTION)),
            "ascii_tolower" => Ok(builtin_value(&ASCII_TOLOWER_FUNCTION)),
            "unicode_tolower" => Ok(builtin_value(&UNICODE_TOLOWER_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module '_sre' has no attribute '{}'",
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

fn sre_compile(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 6 {
        return Err(BuiltinError::TypeError(format!(
            "compile() takes exactly 6 arguments ({} given)",
            args.len()
        )));
    }

    ensure_exact_list(args[2], "compile() argument 'code' must be a list")?;
    ensure_non_negative_int(
        args[3],
        "compile() argument 'groups' must be an integer",
        "compile() argument 'groups' must be non-negative",
    )?;

    re_builtin_compile(vm, &[args[0], args[1]])
}

fn sre_template(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "template() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    ensure_exact_list(args[1], "template() argument 2 must be list")?;
    Ok(args[1])
}

fn sre_getcodesize(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "getcodesize() takes exactly 0 arguments ({} given)",
            args.len()
        )));
    }

    Ok(Value::int(CODESIZE).expect("SRE code size should fit in Value::int"))
}

fn sre_ascii_iscased(args: &[Value]) -> Result<Value, BuiltinError> {
    let character = parse_character_arg("ascii_iscased", args)?;
    let codepoint = character as u32;
    Ok(Value::bool(
        codepoint < 128 && (codepoint as u8).is_ascii_alphabetic(),
    ))
}

fn sre_unicode_iscased(args: &[Value]) -> Result<Value, BuiltinError> {
    let character = parse_character_arg("unicode_iscased", args)?;
    Ok(Value::bool(
        unicode_scalar(character).is_some_and(|ch| ch.is_lowercase() || ch.is_uppercase()),
    ))
}

fn sre_ascii_tolower(args: &[Value]) -> Result<Value, BuiltinError> {
    let character = parse_character_arg("ascii_tolower", args)?;
    let lowered = if (b'A' as i32..=b'Z' as i32).contains(&character) {
        character + 32
    } else {
        character
    };
    Ok(Value::int(lowered as i64).expect("ASCII lowercase result should fit"))
}

fn sre_unicode_tolower(args: &[Value]) -> Result<Value, BuiltinError> {
    let character = parse_character_arg("unicode_tolower", args)?;
    let lowered = unicode_scalar(character)
        .and_then(|ch| ch.to_lowercase().next())
        .map(|ch| ch as i32)
        .unwrap_or(character);
    Ok(Value::int(lowered as i64).expect("Unicode lowercase result should fit"))
}

fn parse_character_arg(fn_name: &str, args: &[Value]) -> Result<i32, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "{fn_name}() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }

    let Some(raw) = value_to_bigint(args[0]) else {
        return Err(BuiltinError::TypeError(format!(
            "{fn_name}() argument must be an integer"
        )));
    };

    raw.to_i32().ok_or_else(|| {
        BuiltinError::OverflowError("Python int too large to convert to C int".to_string())
    })
}

fn ensure_non_negative_int(
    value: Value,
    type_error: &'static str,
    value_error: &'static str,
) -> Result<(), BuiltinError> {
    let Some(integer) = value_to_bigint(value) else {
        return Err(BuiltinError::TypeError(type_error.to_string()));
    };
    if integer.sign() == num_bigint::Sign::Minus {
        return Err(BuiltinError::ValueError(value_error.to_string()));
    }
    Ok(())
}

fn ensure_exact_list(value: Value, message: &'static str) -> Result<(), BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(message.to_string()));
    };
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id == TypeId::LIST {
        Ok(())
    } else {
        Err(BuiltinError::TypeError(message.to_string()))
    }
}

#[inline]
fn unicode_scalar(character: i32) -> Option<char> {
    u32::try_from(character).ok().and_then(char::from_u32)
}
