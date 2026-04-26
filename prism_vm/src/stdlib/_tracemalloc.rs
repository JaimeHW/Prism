//! Minimal native `_tracemalloc` compatibility hooks.
//!
//! CPython's `tracemalloc.py` imports these C-extension functions at module
//! import time. Prism does not yet trace allocation stacks, but it must expose
//! the inert API surface so CPython support code can reliably detect that
//! tracing is disabled.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use prism_core::Value;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::tuple::TupleObject;
use std::sync::{Arc, LazyLock};

static IS_TRACING_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_tracemalloc.is_tracing"), tracemalloc_is_tracing)
});
static START_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_tracemalloc.start"), tracemalloc_start)
});
static STOP_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_tracemalloc.stop"), tracemalloc_stop));
static CLEAR_TRACES_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_tracemalloc.clear_traces"),
        tracemalloc_clear_traces,
    )
});
static RESET_PEAK_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_tracemalloc.reset_peak"), tracemalloc_reset_peak)
});
static GET_TRACEBACK_LIMIT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_tracemalloc.get_traceback_limit"),
        tracemalloc_get_traceback_limit,
    )
});
static GET_TRACED_MEMORY_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_tracemalloc.get_traced_memory"),
        tracemalloc_get_traced_memory,
    )
});
static GET_OBJECT_TRACEBACK_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_tracemalloc._get_object_traceback"),
        tracemalloc_get_object_traceback,
    )
});
static GET_TRACES_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_tracemalloc._get_traces"),
        tracemalloc_get_traces,
    )
});

/// Native `_tracemalloc` module descriptor.
#[derive(Debug, Clone)]
pub struct TraceMallocModule {
    attrs: Vec<Arc<str>>,
}

impl TraceMallocModule {
    /// Create a new `_tracemalloc` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("_get_object_traceback"),
                Arc::from("_get_traces"),
                Arc::from("clear_traces"),
                Arc::from("get_traceback_limit"),
                Arc::from("get_traced_memory"),
                Arc::from("is_tracing"),
                Arc::from("reset_peak"),
                Arc::from("start"),
                Arc::from("stop"),
            ],
        }
    }
}

impl Default for TraceMallocModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for TraceMallocModule {
    fn name(&self) -> &str {
        "_tracemalloc"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "_get_object_traceback" => Ok(builtin_value(&GET_OBJECT_TRACEBACK_FUNCTION)),
            "_get_traces" => Ok(builtin_value(&GET_TRACES_FUNCTION)),
            "clear_traces" => Ok(builtin_value(&CLEAR_TRACES_FUNCTION)),
            "get_traceback_limit" => Ok(builtin_value(&GET_TRACEBACK_LIMIT_FUNCTION)),
            "get_traced_memory" => Ok(builtin_value(&GET_TRACED_MEMORY_FUNCTION)),
            "is_tracing" => Ok(builtin_value(&IS_TRACING_FUNCTION)),
            "reset_peak" => Ok(builtin_value(&RESET_PEAK_FUNCTION)),
            "start" => Ok(builtin_value(&START_FUNCTION)),
            "stop" => Ok(builtin_value(&STOP_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module '_tracemalloc' has no attribute '{}'",
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

fn leak_object_value<T: prism_runtime::Trace + 'static>(object: T) -> Value {
    crate::alloc_managed_value(object)
}

fn tracemalloc_is_tracing(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "is_tracing() takes no arguments ({} given)",
            args.len()
        )));
    }
    Ok(Value::bool(false))
}

fn tracemalloc_start(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "start() takes at most 1 argument ({} given)",
            args.len()
        )));
    }
    Ok(Value::none())
}

fn tracemalloc_stop(args: &[Value]) -> Result<Value, BuiltinError> {
    no_arg_none("stop", args)
}

fn tracemalloc_clear_traces(args: &[Value]) -> Result<Value, BuiltinError> {
    no_arg_none("clear_traces", args)
}

fn tracemalloc_reset_peak(args: &[Value]) -> Result<Value, BuiltinError> {
    no_arg_none("reset_peak", args)
}

fn tracemalloc_get_traceback_limit(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "get_traceback_limit() takes no arguments ({} given)",
            args.len()
        )));
    }
    Ok(Value::int(1).expect("traceback limit should fit"))
}

fn tracemalloc_get_traced_memory(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "get_traced_memory() takes no arguments ({} given)",
            args.len()
        )));
    }
    Ok(leak_object_value(TupleObject::from_slice(&[
        Value::int(0).expect("zero should fit"),
        Value::int(0).expect("zero should fit"),
    ])))
}

fn tracemalloc_get_object_traceback(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "_get_object_traceback() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }
    Ok(Value::none())
}

fn tracemalloc_get_traces(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "_get_traces() takes no arguments ({} given)",
            args.len()
        )));
    }
    Ok(leak_object_value(ListObject::new()))
}

fn no_arg_none(name: &str, args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "{name}() takes no arguments ({} given)",
            args.len()
        )));
    }
    Ok(Value::none())
}

#[cfg(test)]
mod tests;
