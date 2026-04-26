//! Native `select` module bootstrap surface.
//!
//! CPython's `selectors.py` imports this extension module unconditionally and
//! falls back to `SelectSelector` when poll-style primitives are absent. Prism
//! exposes the `select.select` entry point and error alias needed for that
//! import path while keeping unimplemented polling APIs out of the module
//! surface.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::tuple::TupleObject;
use std::sync::{Arc, LazyLock};

const MODULE_DOC: &str = "Native bootstrap implementation of the select module.";

static SELECT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("select.select"), select_select));

/// Native `select` module descriptor.
#[derive(Debug, Clone)]
pub struct SelectModule {
    attrs: Vec<Arc<str>>,
}

impl SelectModule {
    /// Create a new `select` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("__doc__"),
                Arc::from("error"),
                Arc::from("select"),
            ],
        }
    }
}

impl Default for SelectModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for SelectModule {
    fn name(&self) -> &str {
        "select"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__doc__" => Ok(Value::string(intern(MODULE_DOC))),
            "error" => Ok(os_error_type_value()),
            "select" => Ok(builtin_value(&SELECT_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'select' has no attribute '{}'",
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
fn os_error_type_value() -> Value {
    Value::object_ptr((&*crate::builtins::OS_ERROR) as *const _ as *const ())
}

fn select_select(args: &[Value]) -> Result<Value, BuiltinError> {
    if !(3..=4).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "select.select() takes 3 or 4 arguments ({} given)",
            args.len()
        )));
    }

    let readers = sequence_arg(args[0], "rlist")?;
    let writers = sequence_arg(args[1], "wlist")?;
    let exceptional = sequence_arg(args[2], "xlist")?;
    if let Some(timeout) = args.get(3).copied() {
        validate_timeout(timeout)?;
    }

    Ok(tuple_value(vec![
        list_value(readers),
        list_value(writers),
        list_value(exceptional),
    ]))
}

fn validate_timeout(value: Value) -> Result<(), BuiltinError> {
    if value.is_none() {
        return Ok(());
    }

    let seconds = if let Some(float) = value.as_float() {
        float
    } else if let Some(integer) = value.as_int() {
        integer as f64
    } else {
        return Err(BuiltinError::TypeError(
            "timeout must be a number or None".to_string(),
        ));
    };

    if seconds < 0.0 {
        return Err(BuiltinError::ValueError(
            "timeout must be non-negative".to_string(),
        ));
    }
    Ok(())
}

fn sequence_arg(value: Value, name: &str) -> Result<Vec<Value>, BuiltinError> {
    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "{name} must be a list or tuple of file descriptors"
        ))
    })?;

    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::LIST => Ok(unsafe { &*(ptr as *const ListObject) }.as_slice().to_vec()),
        TypeId::TUPLE => Ok(unsafe { &*(ptr as *const TupleObject) }.as_slice().to_vec()),
        _ => Err(BuiltinError::TypeError(format!(
            "{name} must be a list or tuple of file descriptors"
        ))),
    }
}

fn tuple_value(items: Vec<Value>) -> Value {
    crate::alloc_managed_value(TupleObject::from_vec(items))
}

fn list_value(items: Vec<Value>) -> Value {
    crate::alloc_managed_value(ListObject::from_iter(items))
}
