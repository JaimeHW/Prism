//! Native `platform` module.
//!
//! The CPython `platform.py` module is a broad source-backed introspection
//! layer. Prism keeps the common interpreter and host identity queries native so
//! startup-sensitive code can answer them without importing the filesystem-heavy
//! implementation.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::types::tuple::TupleObject;
use std::sync::{Arc, LazyLock};

const PYTHON_VERSION: &str = "3.12.0";
const PYTHON_VERSION_TUPLE: &[&str] = &["3", "12", "0"];

static ARCHITECTURE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("platform.architecture"), platform_architecture)
});
static MACHINE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("platform.machine"), platform_machine));
static PLATFORM_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("platform.platform"), platform_platform));
static PYTHON_IMPLEMENTATION_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("platform.python_implementation"),
        platform_python_implementation,
    )
});
static PYTHON_VERSION_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("platform.python_version"),
        platform_python_version,
    )
});
static PYTHON_VERSION_TUPLE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("platform.python_version_tuple"),
        platform_python_version_tuple,
    )
});
static RELEASE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("platform.release"), platform_release));
static SYSTEM_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("platform.system"), platform_system));
static VERSION_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("platform.version"), platform_version));

/// Native `platform` module descriptor.
#[derive(Debug, Clone)]
pub struct PlatformModule {
    attrs: Vec<Arc<str>>,
}

impl PlatformModule {
    /// Create a new `platform` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("architecture"),
                Arc::from("machine"),
                Arc::from("platform"),
                Arc::from("python_implementation"),
                Arc::from("python_version"),
                Arc::from("python_version_tuple"),
                Arc::from("release"),
                Arc::from("system"),
                Arc::from("version"),
            ],
        }
    }
}

impl Default for PlatformModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for PlatformModule {
    fn name(&self) -> &str {
        "platform"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "architecture" => Ok(builtin_value(&ARCHITECTURE_FUNCTION)),
            "machine" => Ok(builtin_value(&MACHINE_FUNCTION)),
            "platform" => Ok(builtin_value(&PLATFORM_FUNCTION)),
            "python_implementation" => Ok(builtin_value(&PYTHON_IMPLEMENTATION_FUNCTION)),
            "python_version" => Ok(builtin_value(&PYTHON_VERSION_FUNCTION)),
            "python_version_tuple" => Ok(builtin_value(&PYTHON_VERSION_TUPLE_FUNCTION)),
            "release" => Ok(builtin_value(&RELEASE_FUNCTION)),
            "system" => Ok(builtin_value(&SYSTEM_FUNCTION)),
            "version" => Ok(builtin_value(&VERSION_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'platform' has no attribute '{}'",
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
fn string_value(text: &str) -> Value {
    Value::string(intern(text))
}

#[inline]
fn tuple_value(items: impl IntoIterator<Item = Value>) -> Value {
    crate::alloc_managed_value(TupleObject::from_iter(items))
}

#[inline]
fn expect_no_args(function: &str, args: &[Value]) -> Result<(), BuiltinError> {
    if args.is_empty() {
        Ok(())
    } else {
        Err(BuiltinError::TypeError(format!(
            "{function}() takes 0 positional arguments but {} were given",
            args.len()
        )))
    }
}

fn platform_architecture(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args("architecture", args)?;
    Ok(tuple_value([
        string_value(if cfg!(target_pointer_width = "64") {
            "64bit"
        } else {
            "32bit"
        }),
        string_value(executable_format()),
    ]))
}

fn platform_machine(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args("machine", args)?;
    Ok(string_value(machine()))
}

fn platform_platform(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args("platform", args)?;
    let mut parts = vec![system()];
    let release = release();
    if !release.is_empty() {
        parts.push(release);
    }
    let machine = machine();
    if !machine.is_empty() {
        parts.push(machine);
    }
    Ok(string_value(&parts.join("-")))
}

fn platform_python_implementation(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args("python_implementation", args)?;
    Ok(string_value("Prism"))
}

fn platform_python_version(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args("python_version", args)?;
    Ok(string_value(PYTHON_VERSION))
}

fn platform_python_version_tuple(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args("python_version_tuple", args)?;
    Ok(tuple_value(
        PYTHON_VERSION_TUPLE.iter().copied().map(string_value),
    ))
}

fn platform_release(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args("release", args)?;
    Ok(string_value(release()))
}

fn platform_system(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args("system", args)?;
    Ok(string_value(system()))
}

fn platform_version(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args("version", args)?;
    Ok(string_value(release()))
}

#[inline]
fn system() -> &'static str {
    if cfg!(windows) {
        "Windows"
    } else if cfg!(target_os = "macos") {
        "Darwin"
    } else if cfg!(target_os = "linux") {
        "Linux"
    } else if cfg!(target_os = "freebsd") {
        "FreeBSD"
    } else {
        ""
    }
}

#[inline]
fn release() -> &'static str {
    std::env::consts::OS
}

#[inline]
fn machine() -> &'static str {
    match std::env::consts::ARCH {
        "x86_64" => "AMD64",
        "x86" => "x86",
        "aarch64" => "ARM64",
        "arm" => "arm",
        "riscv64" => "riscv64",
        other => other,
    }
}

#[inline]
fn executable_format() -> &'static str {
    if cfg!(windows) {
        "WindowsPE"
    } else if cfg!(target_os = "macos") {
        "Mach-O"
    } else {
        "ELF"
    }
}
