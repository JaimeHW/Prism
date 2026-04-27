//! Native `threading` module bootstrap surface.
//!
//! Prism's low-level `_thread` module owns real thread primitives. This module
//! exposes the high-level `threading` import surface needed by stdlib helpers
//! while failing explicitly for object types that require VM-aware scheduling.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::types::list::ListObject;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, LazyLock};

const EXPORTS: &[&str] = &[
    "get_ident",
    "get_native_id",
    "active_count",
    "current_thread",
    "enumerate",
    "main_thread",
    "Condition",
    "Event",
    "Lock",
    "RLock",
    "Semaphore",
    "BoundedSemaphore",
    "Thread",
    "Barrier",
    "TIMEOUT_MAX",
];

static GET_IDENT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("threading.get_ident"), get_ident));
static GET_NATIVE_ID_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("threading.get_native_id"), get_ident));
static ACTIVE_COUNT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("threading.active_count"), active_count));
static CURRENT_THREAD_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("threading.current_thread"), current_thread)
});
static ENUMERATE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("threading.enumerate"), enumerate));
static MAIN_THREAD_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("threading.main_thread"), main_thread));
static CONDITION_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("threading.Condition"), condition));
static EVENT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("threading.Event"), event));
static LOCK_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("threading.Lock"), lock));
static RLOCK_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("threading.RLock"), rlock));
static SEMAPHORE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("threading.Semaphore"), semaphore));
static BOUNDED_SEMAPHORE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("threading.BoundedSemaphore"), bounded_semaphore)
});
static THREAD_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("threading.Thread"), thread));
static BARRIER_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("threading.Barrier"), barrier));

/// Native `threading` module descriptor.
#[derive(Debug, Clone)]
pub struct ThreadingModule {
    attrs: Vec<Arc<str>>,
    all: Value,
}

impl ThreadingModule {
    /// Create a native `threading` module.
    pub fn new() -> Self {
        Self {
            attrs: EXPORTS.iter().copied().map(Arc::from).collect(),
            all: string_list_value(EXPORTS),
        }
    }
}

impl Default for ThreadingModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ThreadingModule {
    fn name(&self) -> &str {
        "threading"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__all__" => Ok(self.all),
            "get_ident" => Ok(builtin_value(&GET_IDENT_FUNCTION)),
            "get_native_id" => Ok(builtin_value(&GET_NATIVE_ID_FUNCTION)),
            "active_count" => Ok(builtin_value(&ACTIVE_COUNT_FUNCTION)),
            "current_thread" => Ok(builtin_value(&CURRENT_THREAD_FUNCTION)),
            "enumerate" => Ok(builtin_value(&ENUMERATE_FUNCTION)),
            "main_thread" => Ok(builtin_value(&MAIN_THREAD_FUNCTION)),
            "Condition" => Ok(builtin_value(&CONDITION_FUNCTION)),
            "Event" => Ok(builtin_value(&EVENT_FUNCTION)),
            "Lock" => Ok(builtin_value(&LOCK_FUNCTION)),
            "RLock" => Ok(builtin_value(&RLOCK_FUNCTION)),
            "Semaphore" => Ok(builtin_value(&SEMAPHORE_FUNCTION)),
            "BoundedSemaphore" => Ok(builtin_value(&BOUNDED_SEMAPHORE_FUNCTION)),
            "Thread" => Ok(builtin_value(&THREAD_FUNCTION)),
            "Barrier" => Ok(builtin_value(&BARRIER_FUNCTION)),
            "TIMEOUT_MAX" => Ok(Value::float(f64::MAX)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'threading' has no attribute '{}'",
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

fn get_ident(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args("get_ident", args)?;
    let mut hasher = DefaultHasher::new();
    std::thread::current().id().hash(&mut hasher);
    let id = (hasher.finish() & 0x0000_7fff_ffff_ffff) as i64;
    Ok(Value::int(id).unwrap_or_else(|| Value::int_unchecked(1)))
}

fn active_count(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args("active_count", args)?;
    Ok(Value::int_unchecked(1))
}

fn enumerate(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args("enumerate", args)?;
    Err(unsupported("enumerate"))
}

fn current_thread(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args("current_thread", args)?;
    Err(unsupported("current_thread"))
}

fn main_thread(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args("main_thread", args)?;
    Err(unsupported("main_thread"))
}

fn condition(_args: &[Value]) -> Result<Value, BuiltinError> {
    Err(unsupported("Condition"))
}

fn event(_args: &[Value]) -> Result<Value, BuiltinError> {
    Err(unsupported("Event"))
}

fn lock(_args: &[Value]) -> Result<Value, BuiltinError> {
    Err(unsupported("Lock"))
}

fn rlock(_args: &[Value]) -> Result<Value, BuiltinError> {
    Err(unsupported("RLock"))
}

fn semaphore(_args: &[Value]) -> Result<Value, BuiltinError> {
    Err(unsupported("Semaphore"))
}

fn bounded_semaphore(_args: &[Value]) -> Result<Value, BuiltinError> {
    Err(unsupported("BoundedSemaphore"))
}

fn thread(_args: &[Value]) -> Result<Value, BuiltinError> {
    Err(unsupported("Thread"))
}

fn barrier(_args: &[Value]) -> Result<Value, BuiltinError> {
    Err(unsupported("Barrier"))
}

fn expect_no_args(function: &str, args: &[Value]) -> Result<(), BuiltinError> {
    if args.is_empty() {
        Ok(())
    } else {
        Err(BuiltinError::TypeError(format!(
            "{function}() takes no arguments ({} given)",
            args.len()
        )))
    }
}

fn unsupported(feature: &str) -> BuiltinError {
    BuiltinError::NotImplemented(format!(
        "threading.{feature} requires the high-level threading runtime"
    ))
}
