//! Native `signal` module bootstrap surface.
//!
//! CPython's `unittest.signals` imports this module during test discovery. A
//! small native implementation keeps the import path short and avoids pulling
//! in CPython's `_signal` extension expectations.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::error::RuntimeError;
use crate::ops::calls::value_supports_call_protocol;
use crate::stdlib::exceptions::ExceptionTypeId;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::types::tuple::TupleObject;
use rustc_hash::FxHashMap;
use std::sync::{Arc, LazyLock, RwLock};

pub(crate) const SIG_DFL: i64 = 0;
pub(crate) const SIG_IGN: i64 = 1;
pub(crate) const SIGINT: i64 = 2;
pub(crate) const SIGTERM: i64 = 15;
pub(crate) const NSIG: i64 = 65;

static DEFAULT_INT_HANDLER_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("signal.default_int_handler"),
        builtin_default_int_handler,
    )
});
static GETSIGNAL_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("signal.getsignal"), builtin_getsignal));
static SIGNAL_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("signal.signal"), builtin_signal));
static SIGNAL_STATE: LazyLock<RwLock<SignalState>> =
    LazyLock::new(|| RwLock::new(SignalState::new()));

#[derive(Default)]
struct SignalState {
    handlers: FxHashMap<i64, Value>,
}

impl SignalState {
    fn new() -> Self {
        let mut handlers = FxHashMap::default();
        handlers.insert(SIGINT, builtin_value(&DEFAULT_INT_HANDLER_FUNCTION));
        handlers.insert(
            SIGTERM,
            Value::int(SIG_DFL).expect("signal default constant fits in Value::int"),
        );
        Self { handlers }
    }

    fn get(&self, signum: i64) -> Value {
        self.handlers
            .get(&signum)
            .copied()
            .unwrap_or_else(|| Value::int(SIG_DFL).expect("signal default constant fits"))
    }

    fn set(&mut self, signum: i64, handler: Value) -> Value {
        let previous = self.get(signum);
        self.handlers.insert(signum, handler);
        previous
    }
}

/// Native `signal` module descriptor.
pub struct SignalModule {
    attrs: Vec<Arc<str>>,
    all_value: Value,
}

impl SignalModule {
    /// Create a new `signal` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("__all__"),
                Arc::from("SIG_DFL"),
                Arc::from("SIG_IGN"),
                Arc::from("SIGINT"),
                Arc::from("SIGTERM"),
                Arc::from("NSIG"),
                Arc::from("default_int_handler"),
                Arc::from("getsignal"),
                Arc::from("signal"),
            ],
            all_value: export_names_value(),
        }
    }
}

impl Default for SignalModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for SignalModule {
    fn name(&self) -> &str {
        "signal"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__all__" => Ok(self.all_value),
            "SIG_DFL" => Ok(Value::int(SIG_DFL).expect("SIG_DFL fits in Value::int")),
            "SIG_IGN" => Ok(Value::int(SIG_IGN).expect("SIG_IGN fits in Value::int")),
            "SIGINT" => Ok(Value::int(SIGINT).expect("SIGINT fits in Value::int")),
            "SIGTERM" => Ok(Value::int(SIGTERM).expect("SIGTERM fits in Value::int")),
            "NSIG" => Ok(Value::int(NSIG).expect("NSIG fits in Value::int")),
            "default_int_handler" => Ok(builtin_value(&DEFAULT_INT_HANDLER_FUNCTION)),
            "getsignal" => Ok(builtin_value(&GETSIGNAL_FUNCTION)),
            "signal" => Ok(builtin_value(&SIGNAL_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'signal' has no attribute '{}'",
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

fn export_names_value() -> Value {
    leak_object_value(TupleObject::from_vec(
        [
            "SIG_DFL",
            "SIG_IGN",
            "SIGINT",
            "SIGTERM",
            "NSIG",
            "default_int_handler",
            "getsignal",
            "signal",
        ]
        .into_iter()
        .map(|name| Value::string(intern(name)))
        .collect(),
    ))
}

fn parse_signal_number(value: Value, fn_name: &str) -> Result<i64, BuiltinError> {
    let Some(signum) = value.as_int() else {
        return Err(BuiltinError::TypeError(format!(
            "{fn_name}() signal number must be int"
        )));
    };

    if is_valid_signal_number(signum) {
        Ok(signum)
    } else {
        Err(BuiltinError::ValueError(format!(
            "{fn_name}() signal number out of range"
        )))
    }
}

#[inline]
pub(crate) fn is_valid_signal_number(signum: i64) -> bool {
    signum > 0 && signum < NSIG
}

#[inline]
pub(crate) fn handler_for_signal(signum: i64) -> Value {
    SIGNAL_STATE.read().unwrap().get(signum)
}

#[inline]
pub(crate) fn is_default_or_ignored_handler(handler: Value) -> bool {
    matches!(handler.as_int(), Some(SIG_DFL | SIG_IGN))
}

fn validate_handler(handler: Value) -> Result<(), BuiltinError> {
    if matches!(handler.as_int(), Some(SIG_DFL | SIG_IGN)) || value_supports_call_protocol(handler)
    {
        Ok(())
    } else {
        Err(BuiltinError::TypeError(
            "signal handler must be signal.SIG_IGN, signal.SIG_DFL, or a callable object"
                .to_string(),
        ))
    }
}

fn builtin_default_int_handler(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "default_int_handler() takes 2 positional arguments ({} given)",
            args.len()
        )));
    }

    Err(BuiltinError::Raised(RuntimeError::exception(
        ExceptionTypeId::KeyboardInterrupt.as_u8() as u16,
        "",
    )))
}

fn builtin_getsignal(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "getsignal() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let signum = parse_signal_number(args[0], "getsignal")?;
    Ok(SIGNAL_STATE.read().unwrap().get(signum))
}

fn builtin_signal(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "signal() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    let signum = parse_signal_number(args[0], "signal")?;
    validate_handler(args[1])?;
    Ok(SIGNAL_STATE.write().unwrap().set(signum, args[1]))
}

#[cfg(test)]
mod tests;
