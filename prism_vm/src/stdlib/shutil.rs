//! Native `shutil` module bootstrap surface.
//!
//! Prism implements the filesystem operations that CPython's regression helpers
//! need without importing the large pure-Python `shutil` module. The hot paths
//! delegate directly to Rust's platform filesystem APIs.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::truthiness::is_truthy;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::types::bytes::value_as_bytes_ref;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::value_as_string_ref;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock};

const EXPORTS: &[&str] = &["copyfile", "copy", "copy2", "move", "rmtree"];

static COPYFILE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("shutil.copyfile"), copyfile));
static COPY_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("shutil.copy"), copy));
static COPY2_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("shutil.copy2"), copy2));
static MOVE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("shutil.move"), move_path));
static RMTREE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("shutil.rmtree"), rmtree));

/// Native `shutil` module descriptor.
#[derive(Debug, Clone)]
pub struct ShutilModule {
    attrs: Vec<Arc<str>>,
    all: Value,
}

impl ShutilModule {
    /// Create a native `shutil` module.
    pub fn new() -> Self {
        Self {
            attrs: EXPORTS.iter().copied().map(Arc::from).collect(),
            all: string_list_value(EXPORTS),
        }
    }
}

impl Default for ShutilModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ShutilModule {
    fn name(&self) -> &str {
        "shutil"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__all__" => Ok(self.all),
            "copyfile" => Ok(builtin_value(&COPYFILE_FUNCTION)),
            "copy" => Ok(builtin_value(&COPY_FUNCTION)),
            "copy2" => Ok(builtin_value(&COPY2_FUNCTION)),
            "move" => Ok(builtin_value(&MOVE_FUNCTION)),
            "rmtree" => Ok(builtin_value(&RMTREE_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'shutil' has no attribute '{}'",
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

fn copyfile(args: &[Value]) -> Result<Value, BuiltinError> {
    let (src, dst) = two_path_args("copyfile", args)?;
    copy_file_impl(&src, &dst)?;
    Ok(args[1])
}

fn copy(args: &[Value]) -> Result<Value, BuiltinError> {
    let (src, dst) = two_path_args("copy", args)?;
    copy_file_impl(&src, &dst)?;
    Ok(args[1])
}

fn copy2(args: &[Value]) -> Result<Value, BuiltinError> {
    let (src, dst) = two_path_args("copy2", args)?;
    copy_file_impl(&src, &dst)?;
    Ok(args[1])
}

fn move_path(args: &[Value]) -> Result<Value, BuiltinError> {
    let (src, dst) = two_path_args("move", args)?;
    if std::fs::rename(&src, &dst).is_err() {
        copy_file_impl(&src, &dst)?;
        std::fs::remove_file(&src)
            .map_err(|err| os_error(format!("move() failed removing source: {err}")))?;
    }
    Ok(args[1])
}

fn rmtree(args: &[Value]) -> Result<Value, BuiltinError> {
    if !(1..=3).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "rmtree() takes from 1 to 3 positional arguments but {} were given",
            args.len()
        )));
    }

    let path = path_arg(args[0], "rmtree")?;
    let ignore_errors = args.get(1).copied().is_some_and(is_truthy);
    match std::fs::remove_dir_all(&path) {
        Ok(()) => Ok(Value::none()),
        Err(_) if ignore_errors => Ok(Value::none()),
        Err(err) => Err(os_error(format!(
            "rmtree() failed for '{}': {err}",
            path.display()
        ))),
    }
}

fn two_path_args(function: &str, args: &[Value]) -> Result<(PathBuf, PathBuf), BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "{function}() takes exactly 2 positional arguments ({} given)",
            args.len()
        )));
    }
    Ok((path_arg(args[0], function)?, path_arg(args[1], function)?))
}

fn path_arg(value: Value, function: &str) -> Result<PathBuf, BuiltinError> {
    if let Some(text) = value_as_string_ref(value) {
        return Ok(PathBuf::from(text.as_str()));
    }

    if let Some(bytes) = value_as_bytes_ref(value) {
        return Ok(PathBuf::from(
            String::from_utf8_lossy(bytes.as_bytes()).into_owned(),
        ));
    }

    Err(BuiltinError::TypeError(format!(
        "{function}() path arguments must be str, bytes, or os.PathLike"
    )))
}

fn copy_file_impl(src: &PathBuf, dst: &PathBuf) -> Result<(), BuiltinError> {
    std::fs::copy(src, dst).map(|_| ()).map_err(|err| {
        os_error(format!(
            "copy failed from '{}' to '{}': {err}",
            src.display(),
            dst.display()
        ))
    })
}

#[inline]
fn os_error(message: String) -> BuiltinError {
    BuiltinError::OSError(message)
}
