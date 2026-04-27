//! Native `pickletools` bootstrap module.
//!
//! The full CPython module is a pickle disassembler and optimizer. Prism exposes
//! the public module surface natively so shared regression helpers can import it
//! without a source stdlib tree. Unsupported analysis operations fail loudly
//! until the full opcode walker is implemented.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::types::list::ListObject;
use std::sync::{Arc, LazyLock};

const EXPORTS: &[&str] = &["dis", "genops", "optimize"];

static DIS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("pickletools.dis"), dis));
static GENOPS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("pickletools.genops"), genops));
static OPTIMIZE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("pickletools.optimize"), optimize));

/// Native `pickletools` module descriptor.
#[derive(Debug, Clone)]
pub struct PickleToolsModule {
    attrs: Vec<Arc<str>>,
    all: Value,
}

impl PickleToolsModule {
    /// Create a native `pickletools` module.
    pub fn new() -> Self {
        Self {
            attrs: EXPORTS.iter().copied().map(Arc::from).collect(),
            all: string_list_value(EXPORTS),
        }
    }
}

impl Default for PickleToolsModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for PickleToolsModule {
    fn name(&self) -> &str {
        "pickletools"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__all__" => Ok(self.all),
            "dis" => Ok(builtin_value(&DIS_FUNCTION)),
            "genops" => Ok(builtin_value(&GENOPS_FUNCTION)),
            "optimize" => Ok(builtin_value(&OPTIMIZE_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'pickletools' has no attribute '{}'",
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

fn dis(_args: &[Value]) -> Result<Value, BuiltinError> {
    Err(unsupported("dis"))
}

fn genops(_args: &[Value]) -> Result<Value, BuiltinError> {
    Err(unsupported("genops"))
}

fn optimize(_args: &[Value]) -> Result<Value, BuiltinError> {
    Err(unsupported("optimize"))
}

fn unsupported(function: &str) -> BuiltinError {
    BuiltinError::NotImplemented(format!(
        "pickletools.{function} is not implemented in Prism yet"
    ))
}
