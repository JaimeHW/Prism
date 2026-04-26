//! Native `types` module bootstrap surface.
//!
//! CPython's `types.py` mostly exposes names for runtime-owned object kinds.
//! Prism keeps the object constructors native so compatibility imports do not
//! require a large Python source module on startup.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::BuiltinFunctionObject;
use prism_core::Value;
use std::sync::{Arc, LazyLock};

static MODULE_TYPE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("types.ModuleType"),
        crate::builtins::builtin_module,
    )
});
static METHOD_TYPE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("types.MethodType"),
        crate::builtins::builtin_methodtype,
    )
});

/// Native `types` module descriptor.
#[derive(Debug, Clone)]
pub struct TypesModule {
    attrs: Vec<Arc<str>>,
}

impl TypesModule {
    /// Create a new `types` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![Arc::from("ModuleType"), Arc::from("MethodType")],
        }
    }
}

impl Default for TypesModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for TypesModule {
    fn name(&self) -> &str {
        "types"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "ModuleType" => Ok(builtin_value(&MODULE_TYPE_FUNCTION)),
            "MethodType" => Ok(builtin_value(&METHOD_TYPE_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'types' has no attribute '{}'",
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
