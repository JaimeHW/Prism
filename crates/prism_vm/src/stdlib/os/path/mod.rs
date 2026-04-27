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
use prism_runtime::types::TupleObject;
use prism_runtime::types::bytes::value_as_bytes_ref;
use prism_runtime::types::string::StringObject;
use std::sync::Arc;
use std::sync::LazyLock;

static COMMONPREFIX_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("os.path.commonprefix"), builtin_commonprefix)
});
static EXISTS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.path.exists"), builtin_exists));
static LEXISTS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.path.lexists"), builtin_lexists));
static ISFILE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.path.isfile"), builtin_isfile));
static ISDIR_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.path.isdir"), builtin_isdir));
static ISABS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.path.isabs"), builtin_isabs));
static ISMOUNT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.path.ismount"), builtin_ismount));
static JOIN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.path.join"), builtin_join));
static ABSPATH_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.path.abspath"), builtin_abspath));
static NORMPATH_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.path.normpath"), builtin_normpath));
static REALPATH_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.path.realpath"), builtin_realpath));
static BASENAME_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.path.basename"), builtin_basename));
static DIRNAME_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.path.dirname"), builtin_dirname));
static SPLIT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.path.split"), builtin_split));
static SPLITDRIVE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("os.path.splitdrive"), builtin_splitdrive)
});
static SPLITEXT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.path.splitext"), builtin_splitext));
static COMMONPATH_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("os.path.commonpath"), builtin_commonpath)
});
static RELPATH_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.path.relpath"), builtin_relpath));
static SAMEFILE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.path.samefile"), builtin_samefile));
static GETMTIME_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.path.getmtime"), builtin_getmtime));
static GETATIME_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.path.getatime"), builtin_getatime));
static GETCTIME_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("os.path.getctime"), builtin_getctime));

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
            "exists" => Ok(builtin_value(&EXISTS_FUNCTION)),
            "lexists" => Ok(builtin_value(&LEXISTS_FUNCTION)),
            "isfile" => Ok(builtin_value(&ISFILE_FUNCTION)),
            "isdir" => Ok(builtin_value(&ISDIR_FUNCTION)),
            "isabs" => Ok(builtin_value(&ISABS_FUNCTION)),
            "ismount" => Ok(builtin_value(&ISMOUNT_FUNCTION)),
            "join" => Ok(builtin_value(&JOIN_FUNCTION)),
            "abspath" => Ok(builtin_value(&ABSPATH_FUNCTION)),
            "normpath" => Ok(builtin_value(&NORMPATH_FUNCTION)),
            "realpath" => Ok(builtin_value(&REALPATH_FUNCTION)),
            "basename" => Ok(builtin_value(&BASENAME_FUNCTION)),
            "dirname" => Ok(builtin_value(&DIRNAME_FUNCTION)),
            "split" => Ok(builtin_value(&SPLIT_FUNCTION)),
            "splitdrive" => Ok(builtin_value(&SPLITDRIVE_FUNCTION)),
            "splitext" => Ok(builtin_value(&SPLITEXT_FUNCTION)),
            "commonpath" => Ok(builtin_value(&COMMONPATH_FUNCTION)),
            "commonprefix" => Ok(builtin_value(&COMMONPREFIX_FUNCTION)),
            "relpath" => Ok(builtin_value(&RELPATH_FUNCTION)),
            "samefile" => Ok(builtin_value(&SAMEFILE_FUNCTION)),
            "getmtime" => Ok(builtin_value(&GETMTIME_FUNCTION)),
            "getatime" => Ok(builtin_value(&GETATIME_FUNCTION)),
            "getctime" => Ok(builtin_value(&GETCTIME_FUNCTION)),
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

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

fn builtin_exists(args: &[Value]) -> Result<Value, BuiltinError> {
    let path = unary_path_arg(args, "exists")?;
    Ok(Value::bool(query::exists(&path)))
}

fn builtin_lexists(args: &[Value]) -> Result<Value, BuiltinError> {
    let path = unary_path_arg(args, "lexists")?;
    Ok(Value::bool(query::lexists(&path)))
}

fn builtin_isfile(args: &[Value]) -> Result<Value, BuiltinError> {
    let path = unary_path_arg(args, "isfile")?;
    Ok(Value::bool(query::isfile(&path)))
}

fn builtin_isdir(args: &[Value]) -> Result<Value, BuiltinError> {
    let path = unary_path_arg(args, "isdir")?;
    Ok(Value::bool(query::isdir(&path)))
}

fn builtin_isabs(args: &[Value]) -> Result<Value, BuiltinError> {
    let path = unary_path_arg(args, "isabs")?;
    Ok(Value::bool(query::isabs(&path)))
}

fn builtin_ismount(args: &[Value]) -> Result<Value, BuiltinError> {
    let path = unary_path_arg(args, "ismount")?;
    Ok(Value::bool(query::ismount(&path)))
}

fn builtin_join(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "join() missing 1 required positional argument: 'path'".to_string(),
        ));
    }

    let mut parts = Vec::with_capacity(args.len());
    for arg in args {
        parts.push(path_string(*arg, "join")?);
    }
    Ok(Value::string(intern(&join::join_many(&parts))))
}

fn builtin_abspath(args: &[Value]) -> Result<Value, BuiltinError> {
    let path = unary_path_arg(args, "abspath")?;
    Ok(Value::string(intern(&normalize::abspath(&path))))
}

fn builtin_normpath(args: &[Value]) -> Result<Value, BuiltinError> {
    let path = unary_path_arg(args, "normpath")?;
    Ok(Value::string(intern(&normalize::normpath(&path))))
}

fn builtin_realpath(args: &[Value]) -> Result<Value, BuiltinError> {
    let path = unary_path_arg(args, "realpath")?;
    let resolved = normalize::realpath(&path)
        .unwrap_or_else(|_| normalize::normpath(normalize::abspath(&path)));
    Ok(Value::string(intern(&resolved)))
}

fn builtin_basename(args: &[Value]) -> Result<Value, BuiltinError> {
    let path = unary_path_arg(args, "basename")?;
    Ok(Value::string(intern(&split::basename(&path))))
}

fn builtin_dirname(args: &[Value]) -> Result<Value, BuiltinError> {
    let path = unary_path_arg(args, "dirname")?;
    Ok(Value::string(intern(&split::dirname(&path))))
}

fn builtin_split(args: &[Value]) -> Result<Value, BuiltinError> {
    let path = unary_path_arg(args, "split")?;
    let (head, tail) = split::split(&path);
    Ok(tuple2(
        Value::string(intern(&head)),
        Value::string(intern(&tail)),
    ))
}

fn builtin_splitdrive(args: &[Value]) -> Result<Value, BuiltinError> {
    let path = unary_path_arg(args, "splitdrive")?;
    let (drive, tail) = split::splitdrive(&path);
    Ok(tuple2(
        Value::string(intern(&drive)),
        Value::string(intern(&tail)),
    ))
}

fn builtin_splitext(args: &[Value]) -> Result<Value, BuiltinError> {
    let path = unary_path_arg(args, "splitext")?;
    let (root, ext) = split::splitext(&path);
    Ok(tuple2(
        Value::string(intern(&root)),
        Value::string(intern(&ext)),
    ))
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

fn builtin_commonpath(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "commonpath() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }

    let paths = collect_iterable_values(args[0])?
        .into_iter()
        .map(|value| path_string(value, "commonpath"))
        .collect::<Result<Vec<_>, _>>()?;
    compare::commonpath(&paths)
        .map(|path| Value::string(intern(&path)))
        .map_err(|err| BuiltinError::ValueError(err.to_string()))
}

fn builtin_relpath(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "relpath() takes 1 or 2 arguments ({} given)",
            args.len()
        )));
    }

    let path = path_string(args[0], "relpath")?;
    let start = if args.len() == 2 {
        path_string(args[1], "relpath")?
    } else {
        ".".to_string()
    };
    Ok(Value::string(intern(&compare::relpath(&path, &start))))
}

fn builtin_samefile(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "samefile() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    let left = path_string(args[0], "samefile")?;
    let right = path_string(args[1], "samefile")?;
    Ok(Value::bool(compare::samefile(&left, &right)))
}

fn builtin_getmtime(args: &[Value]) -> Result<Value, BuiltinError> {
    let path = unary_path_arg(args, "getmtime")?;
    time_access::getmtime(&path)
        .map(Value::float)
        .map_err(|err| BuiltinError::OSError(err.to_string()))
}

fn builtin_getatime(args: &[Value]) -> Result<Value, BuiltinError> {
    let path = unary_path_arg(args, "getatime")?;
    time_access::getatime(&path)
        .map(Value::float)
        .map_err(|err| BuiltinError::OSError(err.to_string()))
}

fn builtin_getctime(args: &[Value]) -> Result<Value, BuiltinError> {
    let path = unary_path_arg(args, "getctime")?;
    time_access::getctime(&path)
        .map(Value::float)
        .map_err(|err| BuiltinError::OSError(err.to_string()))
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

fn unary_path_arg(args: &[Value], function: &str) -> Result<String, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "{function}() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }
    path_string(args[0], function)
}

fn path_string(value: Value, function: &str) -> Result<String, BuiltinError> {
    if let Some(string) = prism_runtime::types::string::value_as_string_ref(value) {
        return Ok(string.as_str().to_string());
    }

    if let Some(bytes) = value_as_bytes_ref(value) {
        return std::str::from_utf8(bytes.as_bytes())
            .map(str::to_owned)
            .map_err(|_| {
                BuiltinError::ValueError(format!("{function}() path bytes are not UTF-8"))
            });
    }

    Err(BuiltinError::TypeError(format!(
        "{function}() argument must be str, bytes, or os.PathLike, not '{}'",
        value.type_name()
    )))
}

fn tuple2(left: Value, right: Value) -> Value {
    crate::alloc_managed_value(TupleObject::from_iter([left, right]))
}
