//! Native `importlib` bootstrap surface.
//!
//! Prism's resolver is already the authoritative import engine. This module
//! exposes the public `importlib` helpers that CPython stdlib code calls while
//! keeping the actual import path in Rust and avoiding a second bootstrap
//! implementation.

use super::{Module, ModuleError, ModuleResult};
use crate::VirtualMachine;
use crate::builtins::{BuiltinError, BuiltinFunctionObject, runtime_error_to_builtin_error};
use crate::import::{ModuleObject, resolve_relative_import};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::value_as_string_ref;
use std::sync::{Arc, LazyLock};

const CACHE_TAG: &str = "prism-312";
const MAGIC_NUMBER: [u8; 4] = [0xcb, 0x0d, 0x0d, 0x0a];

static IMPORT_MODULE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("importlib.import_module"), import_module)
});
static INVALIDATE_CACHES_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("importlib.invalidate_caches"), invalidate_caches)
});
static RELOAD_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("importlib.reload"), reload));

static CACHE_FROM_SOURCE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(
        Arc::from("importlib.util.cache_from_source"),
        cache_from_source,
    )
});
static SOURCE_FROM_CACHE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("importlib.util.source_from_cache"),
        source_from_cache,
    )
});
static SOURCE_HASH_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("importlib.util.source_hash"), source_hash)
});
static FIND_SPEC_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("importlib.util.find_spec"), find_spec)
});

/// Native `importlib` module descriptor.
#[derive(Debug, Clone)]
pub struct ImportlibModule {
    attrs: Vec<Arc<str>>,
    all: Value,
}

impl ImportlibModule {
    /// Create a new `importlib` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("__all__"),
                Arc::from("import_module"),
                Arc::from("invalidate_caches"),
                Arc::from("reload"),
            ],
            all: string_list_value(&["import_module", "invalidate_caches", "reload"]),
        }
    }
}

impl Default for ImportlibModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ImportlibModule {
    fn name(&self) -> &str {
        "importlib"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__all__" => Ok(self.all),
            "import_module" => Ok(builtin_value(&IMPORT_MODULE_FUNCTION)),
            "invalidate_caches" => Ok(builtin_value(&INVALIDATE_CACHES_FUNCTION)),
            "reload" => Ok(builtin_value(&RELOAD_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'importlib' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

/// Native `importlib.util` module descriptor.
#[derive(Debug, Clone)]
pub struct ImportlibUtilModule {
    attrs: Vec<Arc<str>>,
    magic_number: Value,
}

impl ImportlibUtilModule {
    /// Create a new `importlib.util` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("MAGIC_NUMBER"),
                Arc::from("cache_from_source"),
                Arc::from("find_spec"),
                Arc::from("source_from_cache"),
                Arc::from("source_hash"),
            ],
            magic_number: bytes_value(&MAGIC_NUMBER),
        }
    }
}

impl Default for ImportlibUtilModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ImportlibUtilModule {
    fn name(&self) -> &str {
        "importlib.util"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "MAGIC_NUMBER" => Ok(self.magic_number),
            "cache_from_source" => Ok(builtin_value(&CACHE_FROM_SOURCE_FUNCTION)),
            "find_spec" => Ok(builtin_value(&FIND_SPEC_FUNCTION)),
            "source_from_cache" => Ok(builtin_value(&SOURCE_FROM_CACHE_FUNCTION)),
            "source_hash" => Ok(builtin_value(&SOURCE_HASH_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'importlib.util' has no attribute '{}'",
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
fn leak_object_value<T: prism_runtime::Trace + 'static>(object: T) -> Value {
    crate::alloc_managed_value(object)
}

fn string_list_value(items: &[&str]) -> Value {
    leak_object_value(ListObject::from_iter(
        items
            .iter()
            .copied()
            .map(|item| Value::string(intern(item))),
    ))
}

fn bytes_value(bytes: &[u8]) -> Value {
    leak_object_value(BytesObject::from_slice(bytes))
}

fn import_module(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "import_module() takes from 1 to 2 positional arguments but {} were given",
            args.len()
        )));
    }

    let resolved = resolve_import_name(args[0], args.get(1).copied(), "import_module")?;

    let module = vm
        .import_module_named(&resolved)
        .map_err(runtime_error_to_builtin_error)?;
    Ok(Value::object_ptr(Arc::as_ptr(&module) as *const ()))
}

fn invalidate_caches(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "invalidate_caches() takes 0 positional arguments but {} were given",
            args.len()
        )));
    }
    Ok(Value::none())
}

fn reload(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "reload() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }

    let Some(module_ptr) = args[0].as_object_ptr() else {
        return Err(BuiltinError::TypeError(
            "reload() argument must be a module".to_string(),
        ));
    };
    let Some(module) = vm.import_resolver.module_from_ptr(module_ptr) else {
        return Err(BuiltinError::TypeError(
            "reload() argument must be an imported module".to_string(),
        ));
    };
    let module = vm
        .import_module_named(module.name())
        .map_err(runtime_error_to_builtin_error)?;
    Ok(Value::object_ptr(Arc::as_ptr(&module) as *const ()))
}

fn cache_from_source(
    positional: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if positional.is_empty() || positional.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "cache_from_source() takes from 1 to 2 positional arguments but {} were given",
            positional.len()
        )));
    }

    let path = value_as_string_ref(positional[0])
        .ok_or_else(|| BuiltinError::TypeError("path must be str".to_string()))?;
    let mut optimization = None;

    if let Some(debug_override) = positional.get(1)
        && !debug_override.is_none()
    {
        optimization = Some(if crate::truthiness::is_truthy(*debug_override) {
            String::new()
        } else {
            "0".to_string()
        });
    }

    for (name, value) in keywords {
        match *name {
            "optimization" => {
                optimization = optimization_string(*value)?;
            }
            "debug_override" => {
                if !value.is_none() {
                    optimization = Some(if crate::truthiness::is_truthy(*value) {
                        String::new()
                    } else {
                        "0".to_string()
                    });
                }
            }
            other => {
                return Err(BuiltinError::TypeError(format!(
                    "cache_from_source() got an unexpected keyword argument '{}'",
                    other
                )));
            }
        }
    }

    Ok(Value::string(intern(&cache_path_for_source(
        path.as_str(),
        optimization.as_deref(),
    )?)))
}

fn source_from_cache(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "source_from_cache() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }
    let path = value_as_string_ref(args[0])
        .ok_or_else(|| BuiltinError::TypeError("path must be str".to_string()))?;
    let source = source_path_from_cache(path.as_str())?;
    Ok(Value::string(intern(&source)))
}

fn source_hash(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "source_hash() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }
    let imp = vm
        .import_module_named("_imp")
        .map_err(runtime_error_to_builtin_error)?;
    let source_hash = imp
        .get_attr("source_hash")
        .ok_or_else(|| BuiltinError::AttributeError("_imp.source_hash".to_string()))?;
    crate::ops::calls::invoke_callable_value(
        vm,
        source_hash,
        &[Value::int(3531).expect("raw magic number fits"), args[0]],
    )
    .map_err(runtime_error_to_builtin_error)
}

fn find_spec(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "find_spec() takes from 1 to 2 positional arguments but {} were given",
            args.len()
        )));
    }
    let resolved = resolve_import_name(args[0], args.get(1).copied(), "find_spec")?;
    if !vm.import_resolver.module_exists(&resolved) {
        return Ok(Value::none());
    }
    Ok(module_spec_value(&resolved))
}

fn resolve_import_name(
    name: Value,
    package: Option<Value>,
    function_name: &str,
) -> Result<String, BuiltinError> {
    let name = value_as_string_ref(name)
        .ok_or_else(|| BuiltinError::TypeError(format!("{function_name}() name must be str")))?;
    if !name.as_str().starts_with('.') {
        return Ok(name.as_str().to_string());
    }

    let package = package
        .filter(|value| !value.is_none())
        .and_then(value_as_string_ref)
        .ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "the 'package' argument is required to perform a relative import for '{}'",
                name.as_str()
            ))
        })?;
    let level = name
        .as_str()
        .as_bytes()
        .iter()
        .take_while(|byte| **byte == b'.')
        .count();
    resolve_relative_import(&name.as_str()[level..], level as u32, package.as_str())
        .map_err(|err| BuiltinError::ImportError(err.to_string()))
}

fn module_spec_value(name: &str) -> Value {
    let spec = ModuleObject::new("importlib.machinery.ModuleSpec");
    spec.set_attr("name", Value::string(intern(name)));
    spec.set_attr("loader", Value::none());
    spec.set_attr("origin", Value::string(intern("built-in")));
    spec.set_attr("submodule_search_locations", Value::none());
    spec.set_attr("cached", Value::none());
    spec.set_attr("has_location", Value::bool(false));
    let ptr = Box::leak(Box::new(spec)) as *mut ModuleObject as *const ();
    Value::object_ptr(ptr)
}

fn optimization_string(value: Value) -> Result<Option<String>, BuiltinError> {
    if value.is_none() {
        return Ok(None);
    }

    let rendered = if let Some(text) = value_as_string_ref(value) {
        text.as_str().to_string()
    } else if let Some(number) = value.as_int() {
        number.to_string()
    } else if let Some(flag) = value.as_bool() {
        flag.to_string()
    } else {
        return Err(BuiltinError::TypeError(
            "optimization must be str, int, bool, or None".to_string(),
        ));
    };

    if !rendered.chars().all(|ch| ch.is_ascii_alphanumeric()) {
        return Err(BuiltinError::ValueError(format!(
            "optimization portion of filename must be alphanumeric: {}",
            rendered
        )));
    }
    Ok(Some(rendered))
}

fn cache_path_for_source(path: &str, optimization: Option<&str>) -> Result<String, BuiltinError> {
    if path.is_empty() {
        return Err(BuiltinError::ValueError("empty path".to_string()));
    }

    let (dir, filename) = split_dir_filename(path);
    let stem = filename
        .strip_suffix(".py")
        .or_else(|| filename.strip_suffix(".pyw"))
        .unwrap_or(filename);

    let cache_name = match optimization.filter(|value| !value.is_empty()) {
        Some(opt) => format!("{stem}.{CACHE_TAG}.opt-{opt}.pyc"),
        None => format!("{stem}.{CACHE_TAG}.pyc"),
    };

    Ok(match dir {
        Some(dir) if dir.is_empty() => format!("__pycache__{MAIN_SEPARATOR}{cache_name}"),
        Some(dir) => format!("{dir}{MAIN_SEPARATOR}__pycache__{MAIN_SEPARATOR}{cache_name}"),
        None => format!("__pycache__{MAIN_SEPARATOR}{cache_name}"),
    })
}

fn source_path_from_cache(path: &str) -> Result<String, BuiltinError> {
    let marker = format!("{MAIN_SEPARATOR}__pycache__{MAIN_SEPARATOR}");
    let Some((dir, cached)) = path.rsplit_once(&marker) else {
        return Err(BuiltinError::ValueError(format!(
            "not a pycache path: {}",
            path
        )));
    };

    let Some((stem, _tag_and_suffix)) = cached.split_once('.') else {
        return Err(BuiltinError::ValueError(format!(
            "expected cache filename with tag: {}",
            cached
        )));
    };

    Ok(if dir.is_empty() {
        format!("{stem}.py")
    } else {
        format!("{dir}{MAIN_SEPARATOR}{stem}.py")
    })
}

fn split_dir_filename(path: &str) -> (Option<&str>, &str) {
    match path.rfind(['/', '\\']) {
        Some(index) => (Some(&path[..index]), &path[index + 1..]),
        None => (None, path),
    }
}

#[cfg(windows)]
const MAIN_SEPARATOR: char = '\\';
#[cfg(not(windows))]
const MAIN_SEPARATOR: char = '/';
