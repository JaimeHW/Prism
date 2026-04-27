//! Native subset of CPython's `copy` module.
//!
//! This provides the identity-preserving fast paths needed by early regression
//! tests. Rich object graph copying belongs in the full stdlib implementation;
//! immutable runtime objects can use these native identity paths directly.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use prism_core::Value;
use std::sync::{Arc, LazyLock};

static COPY_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("copy.copy"), copy_value));
static DEEPCOPY_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("copy.deepcopy"), deepcopy_value));

/// Native `copy` module descriptor.
#[derive(Debug, Clone)]
pub struct CopyModule {
    attrs: Vec<Arc<str>>,
}

impl CopyModule {
    /// Create a new `copy` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![Arc::from("copy"), Arc::from("deepcopy")],
        }
    }
}

impl Default for CopyModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for CopyModule {
    fn name(&self) -> &str {
        "copy"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "copy" => Ok(builtin_value(&COPY_FUNCTION)),
            "deepcopy" => Ok(builtin_value(&DEEPCOPY_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'copy' has no attribute '{}'",
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

fn copy_value(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "copy() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    Ok(args[0])
}

fn deepcopy_value(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "deepcopy() takes 1 or 2 arguments ({} given)",
            args.len()
        )));
    }
    Ok(args[0])
}
