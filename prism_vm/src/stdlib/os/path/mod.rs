//! os.path module - Path manipulation operations.
//!
//! High-performance implementation of Python's `os.path` module providing:
//! - Path joining and normalization (`join`, `abspath`, `normpath`, `realpath`)
//! - Path splitting (`basename`, `dirname`, `splitext`, `split`, `splitdrive`)
//! - Path queries (`exists`, `isfile`, `isdir`, `islink`, `isabs`, `lexists`, `ismount`)
//! - Path comparison (`commonpath`, `commonprefix`, `relpath`, `samefile`)
//! - Timestamp access (`getmtime`, `getatime`, `getctime`, `lgetmtime`)

mod compare;
mod join;
mod normalize;
mod query;
mod split;
mod time_access;

pub use compare::*;
pub use join::*;
pub use normalize::*;
pub use query::*;
pub use split::*;
pub use time_access::*;

use super::constants::SEP;
use crate::builtins::{BuiltinError, BuiltinFunctionObject, get_iterator_mut, value_to_iterator};
use crate::stdlib::{Module, ModuleError};
use prism_core::Value;
use prism_core::intern::{intern, interned_by_ptr};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::string::StringObject;
use std::sync::Arc;
use std::sync::LazyLock;

static COMMONPREFIX_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("os.path.commonprefix"), builtin_commonprefix)
});

/// Minimal module wrapper for `os.path`.
///
/// Prism already implements the path algorithms in this module tree; this
/// wrapper exposes the submodule itself to the import system so `import os.path`
/// and `from os import path` behave like CPython.
pub struct OsPathModule;

impl OsPathModule {
    #[inline]
    pub fn new() -> Self {
        Self
    }
}

impl Default for OsPathModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for OsPathModule {
    fn name(&self) -> &str {
        "os.path"
    }

    fn get_attr(&self, name: &str) -> Result<Value, ModuleError> {
        match name {
            "sep" => Ok(Value::string(intern(&SEP.to_string()))),
            "commonprefix" => Ok(Value::object_ptr(
                &*COMMONPREFIX_FUNCTION as *const BuiltinFunctionObject as *const (),
            )),
            "exists" | "lexists" | "isfile" | "isdir" | "isabs" | "ismount" | "join"
            | "abspath" | "normpath" | "realpath" | "basename" | "dirname" | "split"
            | "splitdrive" | "splitext" | "commonpath" | "relpath" | "samefile" | "getmtime"
            | "getatime" | "getctime" => Ok(Value::none()),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'os.path' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        vec![
            Arc::from("sep"),
            Arc::from("exists"),
            Arc::from("lexists"),
            Arc::from("isfile"),
            Arc::from("isdir"),
            Arc::from("isabs"),
            Arc::from("ismount"),
            Arc::from("join"),
            Arc::from("abspath"),
            Arc::from("normpath"),
            Arc::from("realpath"),
            Arc::from("basename"),
            Arc::from("dirname"),
            Arc::from("split"),
            Arc::from("splitdrive"),
            Arc::from("splitext"),
            Arc::from("commonpath"),
            Arc::from("commonprefix"),
            Arc::from("relpath"),
            Arc::from("samefile"),
            Arc::from("getmtime"),
            Arc::from("getatime"),
            Arc::from("getctime"),
        ]
    }
}

fn builtin_commonprefix(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "commonprefix() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }

    let values = collect_iterable_values(args[0])?;
    let strings = values
        .into_iter()
        .map(extract_string)
        .collect::<Result<Vec<_>, _>>()?;
    let refs = strings.iter().map(String::as_str).collect::<Vec<_>>();
    Ok(Value::string(intern(&commonprefix(&refs))))
}

fn collect_iterable_values(value: Value) -> Result<Vec<Value>, BuiltinError> {
    if let Some(iterator) = get_iterator_mut(&value) {
        return Ok(iterator.collect_remaining());
    }

    let mut iterator = value_to_iterator(&value).map_err(BuiltinError::from)?;
    Ok(iterator.collect_remaining())
}

fn extract_string(value: Value) -> Result<String, BuiltinError> {
    if value.is_string() {
        let ptr = value.as_string_object_ptr().ok_or_else(|| {
            BuiltinError::TypeError("commonprefix() argument must be str".to_string())
        })?;
        let interned = interned_by_ptr(ptr as *const u8).ok_or_else(|| {
            BuiltinError::TypeError("commonprefix() argument must be str".to_string())
        })?;
        return Ok(interned.as_str().to_string());
    }

    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("commonprefix() argument must be str".to_string())
    })?;
    if crate::ops::objects::extract_type_id(ptr) != TypeId::STR {
        return Err(BuiltinError::TypeError(
            "commonprefix() argument must be str".to_string(),
        ));
    }

    let string = unsafe { &*(ptr as *const StringObject) };
    Ok(string.as_str().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_runtime::types::tuple::TupleObject;

    #[test]
    fn test_module_wrapper_exposes_os_path_name() {
        let module = OsPathModule::new();
        assert_eq!(module.name(), "os.path");
        assert!(module.dir().contains(&Arc::from("join")));
    }

    #[test]
    fn test_commonprefix_builtin_matches_string_prefix_semantics() {
        let input = TupleObject::from_slice(&[
            Value::string(intern("interstate")),
            Value::string(intern("interstellar")),
            Value::string(intern("internal")),
        ]);
        let input_ptr = Box::into_raw(Box::new(input));
        let value = builtin_commonprefix(&[Value::object_ptr(input_ptr as *const ())])
            .expect("commonprefix should succeed");

        assert_eq!(
            interned_by_ptr(value.as_string_object_ptr().unwrap() as *const u8)
                .expect("result should resolve")
                .as_str(),
            "inter"
        );

        unsafe {
            drop(Box::from_raw(input_ptr));
        }
    }

    #[test]
    fn test_module_wrapper_exposes_callable_commonprefix() {
        let module = OsPathModule::new();
        let value = module.get_attr("commonprefix").expect("attr should exist");
        assert!(value.as_object_ptr().is_some());
    }
}
