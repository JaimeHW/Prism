//! Native `atexit` module bootstrap surface.
//!
//! Modules like `weakref`, `logging`, and `site` expect `atexit` to exist
//! during import. Prism provides a compact native implementation with a real
//! callback registry so those imports work and direct `atexit` tests have a
//! compatible runtime surface.

use super::{Module, ModuleError, ModuleResult};
use crate::VirtualMachine;
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::ops::calls::invoke_callable_value_with_keywords;
use prism_core::Value;
use std::sync::{Arc, LazyLock, Mutex};

static REGISTER_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("atexit.register"), builtin_register));
static UNREGISTER_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("atexit.unregister"), builtin_unregister)
});
static CLEAR_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("atexit._clear"), builtin_clear));
static NCALLBACKS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("atexit._ncallbacks"), builtin_ncallbacks)
});
static RUN_EXITFUNCS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("atexit._run_exitfuncs"), builtin_run_exitfuncs)
});
static EXIT_CALLBACKS: LazyLock<Mutex<Vec<RegisteredCallback>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

#[derive(Clone, Debug)]
struct RegisteredCallback {
    callable: Value,
    args: Vec<Value>,
    keywords: Vec<(Arc<str>, Value)>,
}

/// Native `atexit` module descriptor.
#[derive(Debug, Clone)]
pub struct AtexitModule {
    attrs: Vec<Arc<str>>,
}

impl AtexitModule {
    /// Create a new `atexit` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("register"),
                Arc::from("unregister"),
                Arc::from("_clear"),
                Arc::from("_ncallbacks"),
                Arc::from("_run_exitfuncs"),
            ],
        }
    }
}

impl Default for AtexitModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for AtexitModule {
    fn name(&self) -> &str {
        "atexit"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "register" => Ok(builtin_value(&REGISTER_FUNCTION)),
            "unregister" => Ok(builtin_value(&UNREGISTER_FUNCTION)),
            "_clear" => Ok(builtin_value(&CLEAR_FUNCTION)),
            "_ncallbacks" => Ok(builtin_value(&NCALLBACKS_FUNCTION)),
            "_run_exitfuncs" => Ok(builtin_value(&RUN_EXITFUNCS_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'atexit' has no attribute '{}'",
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

fn builtin_register(args: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    let Some(&callable) = args.first() else {
        return Err(BuiltinError::TypeError(
            "register() missing required argument 'func'".to_string(),
        ));
    };

    EXIT_CALLBACKS.lock().unwrap().push(RegisteredCallback {
        callable,
        args: args[1..].to_vec(),
        keywords: keywords
            .iter()
            .map(|(name, value)| (Arc::from(*name), *value))
            .collect(),
    });
    Ok(callable)
}

fn builtin_unregister(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "unregister() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let callable = args[0];
    EXIT_CALLBACKS
        .lock()
        .unwrap()
        .retain(|registered| registered.callable.to_bits() != callable.to_bits());
    Ok(Value::none())
}

fn builtin_clear(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "_clear() takes no arguments ({} given)",
            args.len()
        )));
    }

    EXIT_CALLBACKS.lock().unwrap().clear();
    Ok(Value::none())
}

fn builtin_ncallbacks(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "_ncallbacks() takes no arguments ({} given)",
            args.len()
        )));
    }

    Value::int(EXIT_CALLBACKS.lock().unwrap().len() as i64)
        .ok_or_else(|| BuiltinError::OverflowError("callback count overflow".to_string()))
}

pub(crate) fn run_exitfuncs(vm: &mut VirtualMachine) -> Result<Value, BuiltinError> {
    builtin_run_exitfuncs(vm, &[])
}

fn builtin_run_exitfuncs(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "_run_exitfuncs() takes no arguments ({} given)",
            args.len()
        )));
    }

    let callbacks = std::mem::take(&mut *EXIT_CALLBACKS.lock().unwrap());
    let mut last_error = None;

    for callback in callbacks.into_iter().rev() {
        let keyword_refs = callback
            .keywords
            .iter()
            .map(|(name, value)| (name.as_ref(), *value))
            .collect::<Vec<_>>();
        if let Err(err) = invoke_callable_value_with_keywords(
            vm,
            callback.callable,
            &callback.args,
            &keyword_refs,
        ) {
            last_error = Some(err);
        }
    }

    match last_error {
        Some(err) => Err(BuiltinError::Raised(err)),
        None => Ok(Value::none()),
    }
}
