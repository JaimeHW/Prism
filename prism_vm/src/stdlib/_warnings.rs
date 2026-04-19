//! Native `_warnings` bootstrap module.
//!
//! CPython's `warnings.py` expects a small C-accelerated compatibility surface
//! to exist during startup. Prism still relies on the Python stdlib module for
//! higher-level warning semantics, but this native module provides the stable
//! bootstrap state and callables that let that import complete cleanly.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::list::ListObject;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};

static FILTERS_MUTATED_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_warnings._filters_mutated"), filters_mutated)
});
static WARN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_warnings.warn"), warn));
static WARN_EXPLICIT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_warnings.warn_explicit"), warn_explicit)
});

static FILTERS_VERSION: AtomicUsize = AtomicUsize::new(1);

/// Native `_warnings` module descriptor.
#[derive(Debug, Clone)]
pub struct WarningsModule {
    filters_value: Value,
    defaultaction_value: Value,
    onceregistry_value: Value,
    attrs: Vec<Arc<str>>,
}

impl WarningsModule {
    /// Create a new `_warnings` module descriptor.
    pub fn new() -> Self {
        Self {
            filters_value: leak_object_value(ListObject::new()),
            defaultaction_value: Value::string(intern("default")),
            onceregistry_value: leak_object_value(DictObject::new()),
            attrs: vec![
                Arc::from("_defaultaction"),
                Arc::from("_filters_mutated"),
                Arc::from("_onceregistry"),
                Arc::from("filters"),
                Arc::from("warn"),
                Arc::from("warn_explicit"),
            ],
        }
    }
}

impl Default for WarningsModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for WarningsModule {
    fn name(&self) -> &str {
        "_warnings"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "filters" => Ok(self.filters_value),
            "_defaultaction" => Ok(self.defaultaction_value),
            "_onceregistry" => Ok(self.onceregistry_value),
            "_filters_mutated" => Ok(builtin_value(&FILTERS_MUTATED_FUNCTION)),
            "warn" => Ok(builtin_value(&WARN_FUNCTION)),
            "warn_explicit" => Ok(builtin_value(&WARN_EXPLICIT_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module '_warnings' has no attribute '{}'",
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
fn leak_object_value<T>(object: T) -> Value {
    Value::object_ptr(Box::into_raw(Box::new(object)) as *const ())
}

fn filters_mutated(_args: &[Value]) -> Result<Value, BuiltinError> {
    FILTERS_VERSION.fetch_add(1, Ordering::Relaxed);
    Ok(Value::none())
}

fn warn(_args: &[Value]) -> Result<Value, BuiltinError> {
    Ok(Value::none())
}

fn warn_explicit(_args: &[Value]) -> Result<Value, BuiltinError> {
    Ok(Value::none())
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::intern::interned_by_ptr;

    #[test]
    fn test_warnings_module_imports_as_builtin_anchor() {
        let module = WarningsModule::new();
        assert_eq!(module.name(), "_warnings");
        assert!(module.dir().contains(&Arc::from("filters")));
    }

    #[test]
    fn test_warnings_module_exposes_bootstrap_state() {
        let module = WarningsModule::new();

        let defaultaction = module
            .get_attr("_defaultaction")
            .expect("_defaultaction should exist");
        let defaultaction_ptr = defaultaction
            .as_string_object_ptr()
            .expect("_defaultaction should be an interned string");
        assert_eq!(
            interned_by_ptr(defaultaction_ptr as *const u8)
                .unwrap()
                .as_str(),
            "default"
        );

        let filters = module.get_attr("filters").expect("filters should exist");
        let filters_ptr = filters
            .as_object_ptr()
            .expect("filters should be a list object");
        let filters = unsafe { &*(filters_ptr as *const ListObject) };
        assert!(filters.is_empty());

        let onceregistry = module
            .get_attr("_onceregistry")
            .expect("_onceregistry should exist");
        let onceregistry_ptr = onceregistry
            .as_object_ptr()
            .expect("_onceregistry should be a dict object");
        let onceregistry = unsafe { &*(onceregistry_ptr as *const DictObject) };
        assert!(onceregistry.is_empty());
    }

    #[test]
    fn test_warnings_module_exposes_callable_bootstrap_functions() {
        let module = WarningsModule::new();

        for name in ["_filters_mutated", "warn", "warn_explicit"] {
            let value = module.get_attr(name).expect("callable should exist");
            let ptr = value
                .as_object_ptr()
                .expect("callable should be a builtin function object");
            let header = unsafe { &*(ptr as *const ObjectHeader) };
            assert_eq!(header.type_id, TypeId::BUILTIN_FUNCTION);
        }
    }
}
