//! Native `weakref` compatibility module.
//!
//! This module provides the bootstrap surface needed by early CPython stdlib
//! imports such as `unittest.signals`, while Prism's lower-level `_weakref`
//! support continues to grow underneath it.

use super::{_weakref, Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject, builtin_dict, builtin_set};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::set::SetObject;
use prism_runtime::types::tuple::TupleObject;
use std::sync::{Arc, LazyLock};

static WEAKKEYDICTIONARY_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("weakref.WeakKeyDictionary"),
        builtin_weak_key_dictionary,
    )
});
static WEAKVALUEDICTIONARY_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("weakref.WeakValueDictionary"),
        builtin_weak_value_dictionary,
    )
});
static WEAKSET_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("weakref.WeakSet"), builtin_weak_set));
static FINALIZE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("weakref.finalize"), builtin_finalize));
static PROXY_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("weakref.proxy"), _weakref::builtin_proxy)
});
static GETWEAKREFCOUNT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("weakref.getweakrefcount"),
        _weakref::builtin_getweakrefcount,
    )
});
static GETWEAKREFS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("weakref.getweakrefs"),
        _weakref::builtin_getweakrefs,
    )
});

/// Native `weakref` module descriptor.
pub struct WeakrefModule {
    attrs: Vec<Arc<str>>,
    all_value: Value,
    proxy_types_value: Value,
}

impl WeakrefModule {
    /// Create a new `weakref` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("__all__"),
                Arc::from("ref"),
                Arc::from("proxy"),
                Arc::from("getweakrefcount"),
                Arc::from("getweakrefs"),
                Arc::from("WeakKeyDictionary"),
                Arc::from("WeakValueDictionary"),
                Arc::from("WeakSet"),
                Arc::from("WeakMethod"),
                Arc::from("finalize"),
                Arc::from("ReferenceType"),
                Arc::from("ProxyType"),
                Arc::from("CallableProxyType"),
                Arc::from("ProxyTypes"),
            ],
            all_value: export_names_value(),
            proxy_types_value: export_proxy_types_value(),
        }
    }
}

impl Default for WeakrefModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for WeakrefModule {
    fn name(&self) -> &str {
        "weakref"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__all__" => Ok(self.all_value),
            "ref" => Ok(_weakref::reference_type_value()),
            "proxy" => Ok(builtin_value(&PROXY_FUNCTION)),
            "getweakrefcount" => Ok(builtin_value(&GETWEAKREFCOUNT_FUNCTION)),
            "getweakrefs" => Ok(builtin_value(&GETWEAKREFS_FUNCTION)),
            "WeakKeyDictionary" => Ok(builtin_value(&WEAKKEYDICTIONARY_FUNCTION)),
            "WeakValueDictionary" => Ok(builtin_value(&WEAKVALUEDICTIONARY_FUNCTION)),
            "WeakSet" => Ok(builtin_value(&WEAKSET_FUNCTION)),
            "WeakMethod" => Ok(_weakref::reference_type_value()),
            "finalize" => Ok(builtin_value(&FINALIZE_FUNCTION)),
            "ReferenceType" => Ok(_weakref::reference_type_value()),
            "ProxyType" => Ok(_weakref::proxy_type_value()),
            "CallableProxyType" => Ok(_weakref::callable_proxy_type_value()),
            "ProxyTypes" => Ok(self.proxy_types_value),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'weakref' has no attribute '{}'",
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
            "ref",
            "proxy",
            "getweakrefcount",
            "getweakrefs",
            "WeakKeyDictionary",
            "ReferenceType",
            "ProxyType",
            "CallableProxyType",
            "ProxyTypes",
            "WeakValueDictionary",
            "WeakSet",
            "WeakMethod",
            "finalize",
        ]
        .into_iter()
        .map(|name| Value::string(intern(name)))
        .collect(),
    ))
}

fn export_proxy_types_value() -> Value {
    leak_object_value(TupleObject::from_vec(vec![
        _weakref::proxy_type_value(),
        _weakref::callable_proxy_type_value(),
    ]))
}

fn builtin_weak_key_dictionary(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_dict(args)
}

fn builtin_weak_value_dictionary(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_dict(args)
}

fn builtin_weak_set(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_set(args)
}

fn builtin_finalize(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 {
        return Err(BuiltinError::TypeError(format!(
            "finalize() needs at least 2 arguments ({} given)",
            args.len()
        )));
    }

    Ok(Value::none())
}
