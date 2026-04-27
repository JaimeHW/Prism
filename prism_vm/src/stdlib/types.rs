//! Native `types` module bootstrap surface.
//!
//! CPython's `types.py` mostly exposes names for runtime-owned object kinds.
//! Prism keeps the object constructors native so compatibility imports do not
//! require a large Python source module on startup.

use super::{Module, ModuleError, ModuleResult};
use crate::VirtualMachine;
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::ops::calls::invoke_callable_value;
use crate::ops::objects::dict_storage_ref_from_ptr;
use prism_core::Value;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::string::value_as_string_ref;
use prism_runtime::types::tuple::TupleObject;
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
static NEW_CLASS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm_kw(Arc::from("types.new_class"), new_class));

/// Native `types` module descriptor.
#[derive(Debug, Clone)]
pub struct TypesModule {
    attrs: Vec<Arc<str>>,
}

impl TypesModule {
    /// Create a new `types` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("ModuleType"),
                Arc::from("MethodType"),
                Arc::from("new_class"),
            ],
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
            "new_class" => Ok(builtin_value(&NEW_CLASS_FUNCTION)),
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

#[derive(Clone, Copy)]
struct NewClassArgs {
    name: Value,
    bases: Value,
    kwds: Value,
    exec_body: Value,
}

fn new_class(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let bound = bind_new_class_args(args, keywords)?;
    let namespace = crate::alloc_managed_value(DictObject::new());

    if !bound.exec_body.is_none() {
        invoke_callable_value(vm, bound.exec_body, &[namespace])
            .map_err(crate::builtins::runtime_error_to_builtin_error)?;
    }

    let type_args = [bound.name, bound.bases, namespace];
    let kwd_entries = new_class_keyword_entries(bound.kwds)?;
    if kwd_entries.is_empty() {
        return crate::builtins::builtin_type_with_vm(vm, &type_args);
    }

    let kwd_refs = kwd_entries
        .iter()
        .map(|(name, value)| (name.as_str(), *value))
        .collect::<Vec<_>>();
    crate::builtins::call_builtin_type_kw_with_vm(
        vm,
        prism_runtime::object::type_obj::TypeId::TYPE,
        &type_args,
        &kwd_refs,
    )
}

fn bind_new_class_args(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<NewClassArgs, BuiltinError> {
    if args.len() > 4 {
        return Err(BuiltinError::TypeError(format!(
            "new_class() takes from 1 to 4 positional arguments but {} were given",
            args.len()
        )));
    }

    let mut name = args.first().copied();
    let mut bases = args.get(1).copied();
    let mut kwds = args.get(2).copied();
    let mut exec_body = args.get(3).copied();

    for &(keyword, value) in keywords {
        match keyword {
            "name" => assign_new_class_keyword(&mut name, value, "name")?,
            "bases" => assign_new_class_keyword(&mut bases, value, "bases")?,
            "kwds" => assign_new_class_keyword(&mut kwds, value, "kwds")?,
            "exec_body" => assign_new_class_keyword(&mut exec_body, value, "exec_body")?,
            _ => {
                return Err(BuiltinError::TypeError(format!(
                    "new_class() got an unexpected keyword argument '{}'",
                    keyword
                )));
            }
        }
    }

    let name = name.ok_or_else(|| {
        BuiltinError::TypeError("new_class() missing required argument 'name'".to_string())
    })?;

    Ok(NewClassArgs {
        name,
        bases: bases.unwrap_or_else(empty_tuple_value),
        kwds: kwds.unwrap_or_else(Value::none),
        exec_body: exec_body.unwrap_or_else(Value::none),
    })
}

#[inline]
fn assign_new_class_keyword(
    slot: &mut Option<Value>,
    value: Value,
    name: &str,
) -> Result<(), BuiltinError> {
    if slot.is_some() {
        return Err(BuiltinError::TypeError(format!(
            "new_class() got multiple values for argument '{}'",
            name
        )));
    }
    *slot = Some(value);
    Ok(())
}

fn new_class_keyword_entries(value: Value) -> Result<Vec<(String, Value)>, BuiltinError> {
    if value.is_none() {
        return Ok(Vec::new());
    }

    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("new_class() argument 'kwds' must be a dict or None".to_string())
    })?;
    let dict = dict_storage_ref_from_ptr(ptr).ok_or_else(|| {
        BuiltinError::TypeError("new_class() argument 'kwds' must be a dict or None".to_string())
    })?;

    let mut entries = Vec::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let name = value_as_string_ref(key)
            .ok_or_else(|| BuiltinError::TypeError("keywords must be strings".to_string()))?;
        entries.push((name.as_str().to_string(), value));
    }
    Ok(entries)
}

#[inline]
fn empty_tuple_value() -> Value {
    crate::alloc_managed_value(TupleObject::empty())
}
