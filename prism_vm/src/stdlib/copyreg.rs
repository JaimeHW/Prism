//! Native `copyreg` module.
//!
//! `copyreg` is small but semantically important: `copy`, `pickle`, and user
//! code all share its reducer registry. Prism keeps the public registries as
//! normal module dictionaries so they are interpreter-local Python objects,
//! while the hot functions stay native and avoid importing a pure-Python module
//! on startup-sensitive paths.

use super::{Module, ModuleError, ModuleResult};
use crate::VirtualMachine;
use crate::builtins::{
    BuiltinError, BuiltinFunctionObject, allocate_heap_instance_for_class,
    runtime_error_to_builtin_error,
};
use crate::import::ModuleObject;
use crate::ops::calls::value_supports_call_protocol;
use crate::ops::dict_access::{dict_get_item, dict_remove_item, dict_set_item};
use crate::ops::objects::{dict_storage_mut_from_ptr, dict_storage_ref_from_ptr, extract_type_id};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::int::value_to_i64;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::tuple::TupleObject;
use std::sync::{Arc, LazyLock};

const EXPORTS: &[&str] = &[
    "pickle",
    "constructor",
    "add_extension",
    "remove_extension",
    "clear_extension_cache",
];

static PICKLE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("copyreg.pickle"), pickle));
static CONSTRUCTOR_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("copyreg.constructor"), constructor));
static ADD_EXTENSION_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("copyreg.add_extension"), add_extension)
});
static REMOVE_EXTENSION_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("copyreg.remove_extension"), remove_extension)
});
static CLEAR_EXTENSION_CACHE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("copyreg.clear_extension_cache"),
        clear_extension_cache,
    )
});
static NEWOBJ_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("copyreg.__newobj__"), newobj));
static NEWOBJ_EX_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("copyreg.__newobj_ex__"), newobj_ex));
static RECONSTRUCTOR_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("copyreg._reconstructor"), reconstructor)
});
static SLOTNAMES_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("copyreg._slotnames"), slotnames));

/// Native `copyreg` module descriptor.
#[derive(Debug, Clone)]
pub struct CopyRegModule {
    attrs: Vec<Arc<str>>,
}

impl CopyRegModule {
    /// Create a new `copyreg` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: [
                "__all__",
                "pickle",
                "constructor",
                "dispatch_table",
                "add_extension",
                "remove_extension",
                "clear_extension_cache",
                "_extension_registry",
                "_inverted_registry",
                "_extension_cache",
                "__newobj__",
                "__newobj_ex__",
                "_reconstructor",
                "_slotnames",
            ]
            .into_iter()
            .map(Arc::from)
            .collect(),
        }
    }
}

impl Default for CopyRegModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for CopyRegModule {
    fn name(&self) -> &str {
        "copyreg"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__all__" => Ok(string_list_value(EXPORTS)),
            "pickle" => Ok(builtin_value(&PICKLE_FUNCTION)),
            "constructor" => Ok(builtin_value(&CONSTRUCTOR_FUNCTION)),
            "dispatch_table"
            | "_extension_registry"
            | "_inverted_registry"
            | "_extension_cache" => Ok(crate::alloc_managed_value(DictObject::new())),
            "add_extension" => Ok(builtin_value(&ADD_EXTENSION_FUNCTION)),
            "remove_extension" => Ok(builtin_value(&REMOVE_EXTENSION_FUNCTION)),
            "clear_extension_cache" => Ok(builtin_value(&CLEAR_EXTENSION_CACHE_FUNCTION)),
            "__newobj__" => Ok(builtin_value(&NEWOBJ_FUNCTION)),
            "__newobj_ex__" => Ok(builtin_value(&NEWOBJ_EX_FUNCTION)),
            "_reconstructor" => Ok(builtin_value(&RECONSTRUCTOR_FUNCTION)),
            "_slotnames" => Ok(builtin_value(&SLOTNAMES_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'copyreg' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }

    fn populate(&self, module: &ModuleObject) -> Result<(), ModuleError> {
        module.set_attr("__all__", string_list_value(EXPORTS));
        module.set_attr("pickle", builtin_value(&PICKLE_FUNCTION));
        module.set_attr("constructor", builtin_value(&CONSTRUCTOR_FUNCTION));
        module.set_attr("dispatch_table", dict_value());
        module.set_attr("add_extension", builtin_value(&ADD_EXTENSION_FUNCTION));
        module.set_attr(
            "remove_extension",
            builtin_value(&REMOVE_EXTENSION_FUNCTION),
        );
        module.set_attr(
            "clear_extension_cache",
            builtin_value(&CLEAR_EXTENSION_CACHE_FUNCTION),
        );
        module.set_attr("_extension_registry", dict_value());
        module.set_attr("_inverted_registry", dict_value());
        module.set_attr("_extension_cache", dict_value());
        module.set_attr("__newobj__", builtin_value(&NEWOBJ_FUNCTION));
        module.set_attr("__newobj_ex__", builtin_value(&NEWOBJ_EX_FUNCTION));
        module.set_attr("_reconstructor", builtin_value(&RECONSTRUCTOR_FUNCTION));
        module.set_attr("_slotnames", builtin_value(&SLOTNAMES_FUNCTION));
        Ok(())
    }
}

/// Look up a reducer in the active interpreter's `copyreg.dispatch_table`.
pub(crate) fn dispatch_reducer(
    vm: &mut VirtualMachine,
    type_value: Value,
) -> Result<Option<Value>, BuiltinError> {
    let module = copyreg_module(vm)?;
    let table_value = module_attr(&module, "dispatch_table")?;
    let table = dict_ref(table_value, "copyreg.dispatch_table")?;
    dict_get_item(vm, table, type_value).map_err(runtime_error_to_builtin_error)
}

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

#[inline]
fn dict_value() -> Value {
    crate::alloc_managed_value(DictObject::new())
}

fn string_list_value(items: &[&str]) -> Value {
    crate::alloc_managed_value(ListObject::from_iter(
        items
            .iter()
            .copied()
            .map(|item| Value::string(intern(item))),
    ))
}

fn pickle(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if !(2..=3).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "pickle() takes 2 or 3 arguments ({} given)",
            args.len()
        )));
    }
    if !value_supports_call_protocol(args[1]) {
        return Err(BuiltinError::TypeError(
            "reduction functions must be callable".to_string(),
        ));
    }
    if let Some(&constructor_ob) = args.get(2)
        && !constructor_ob.is_none()
    {
        constructor(&[constructor_ob])?;
    }

    let module = copyreg_module(vm)?;
    let table_value = module_attr(&module, "dispatch_table")?;
    let table = dict_mut(table_value, "copyreg.dispatch_table")?;
    dict_set_item(vm, table, args[0], args[1]).map_err(runtime_error_to_builtin_error)?;
    Ok(Value::none())
}

fn constructor(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "constructor() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    if !value_supports_call_protocol(args[0]) {
        return Err(BuiltinError::TypeError(
            "constructors must be callable".to_string(),
        ));
    }
    Ok(Value::none())
}

fn add_extension(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 3 {
        return Err(BuiltinError::TypeError(format!(
            "add_extension() takes exactly 3 arguments ({} given)",
            args.len()
        )));
    }
    let code = extension_code(args[2])?;
    let module = copyreg_module(vm)?;
    let key = extension_key(args[0], args[1]);
    let code_value = Value::int(code).expect("validated extension code fits tagged int");

    let registry_value = module_attr(&module, "_extension_registry")?;
    let inverted_value = module_attr(&module, "_inverted_registry")?;
    let registry = dict_ref(registry_value, "copyreg._extension_registry")?;
    let inverted = dict_ref(inverted_value, "copyreg._inverted_registry")?;

    if let Some(existing) =
        dict_get_item(vm, registry, key).map_err(runtime_error_to_builtin_error)?
        && existing != code_value
    {
        return Err(BuiltinError::ValueError(format!(
            "key {:?} is already registered with code {:?}",
            key, existing
        )));
    }
    if let Some(existing) =
        dict_get_item(vm, inverted, code_value).map_err(runtime_error_to_builtin_error)?
        && existing != key
    {
        return Err(BuiltinError::ValueError(format!(
            "code {} is already in use for key {:?}",
            code, existing
        )));
    }

    let registry = dict_mut(registry_value, "copyreg._extension_registry")?;
    dict_set_item(vm, registry, key, code_value).map_err(runtime_error_to_builtin_error)?;
    let inverted = dict_mut(inverted_value, "copyreg._inverted_registry")?;
    dict_set_item(vm, inverted, code_value, key).map_err(runtime_error_to_builtin_error)?;
    Ok(Value::none())
}

fn remove_extension(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 3 {
        return Err(BuiltinError::TypeError(format!(
            "remove_extension() takes exactly 3 arguments ({} given)",
            args.len()
        )));
    }
    let code = extension_code(args[2])?;
    let module = copyreg_module(vm)?;
    let key = extension_key(args[0], args[1]);
    let code_value = Value::int(code).expect("validated extension code fits tagged int");

    let registry_value = module_attr(&module, "_extension_registry")?;
    let inverted_value = module_attr(&module, "_inverted_registry")?;
    let cache_value = module_attr(&module, "_extension_cache")?;
    let registry = dict_ref(registry_value, "copyreg._extension_registry")?;
    let inverted = dict_ref(inverted_value, "copyreg._inverted_registry")?;
    let registered = dict_get_item(vm, registry, key).map_err(runtime_error_to_builtin_error)?;
    let inverted_key =
        dict_get_item(vm, inverted, code_value).map_err(runtime_error_to_builtin_error)?;

    if registered != Some(code_value) || inverted_key != Some(key) {
        return Err(BuiltinError::ValueError(format!(
            "key {:?} is not registered with code {}",
            key, code
        )));
    }

    let registry = dict_mut(registry_value, "copyreg._extension_registry")?;
    dict_remove_item(vm, registry, key).map_err(runtime_error_to_builtin_error)?;
    let inverted = dict_mut(inverted_value, "copyreg._inverted_registry")?;
    dict_remove_item(vm, inverted, code_value).map_err(runtime_error_to_builtin_error)?;
    let cache = dict_mut(cache_value, "copyreg._extension_cache")?;
    dict_remove_item(vm, cache, code_value).map_err(runtime_error_to_builtin_error)?;
    Ok(Value::none())
}

fn clear_extension_cache(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "clear_extension_cache() takes no arguments ({} given)",
            args.len()
        )));
    }
    let module = copyreg_module(vm)?;
    let cache_value = module_attr(&module, "_extension_cache")?;
    dict_mut(cache_value, "copyreg._extension_cache")?.clear();
    Ok(Value::none())
}

fn newobj(_vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    let Some(&class_value) = args.first() else {
        return Err(BuiltinError::TypeError(
            "__newobj__() missing class argument".to_string(),
        ));
    };
    let class = heap_class_from_value(class_value, "__newobj__")?;
    Ok(crate::alloc_managed_value(
        allocate_heap_instance_for_class(class),
    ))
}

fn newobj_ex(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 3 {
        return Err(BuiltinError::TypeError(format!(
            "__newobj_ex__() takes exactly 3 arguments ({} given)",
            args.len()
        )));
    }
    newobj(vm, &[args[0]])
}

fn reconstructor(_vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 3 {
        return Err(BuiltinError::TypeError(format!(
            "_reconstructor() takes exactly 3 arguments ({} given)",
            args.len()
        )));
    }
    let class = heap_class_from_value(args[0], "_reconstructor")?;
    Ok(crate::alloc_managed_value(
        allocate_heap_instance_for_class(class),
    ))
}

fn slotnames(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "_slotnames() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    Ok(crate::alloc_managed_value(ListObject::new()))
}

fn copyreg_module(vm: &mut VirtualMachine) -> Result<Arc<ModuleObject>, BuiltinError> {
    vm.import_resolver
        .get_cached("copyreg")
        .or_else(|| vm.import_resolver.import_module("copyreg").ok())
        .ok_or_else(|| BuiltinError::ModuleNotFoundError("No module named 'copyreg'".to_string()))
}

#[inline]
fn module_attr(module: &ModuleObject, name: &'static str) -> Result<Value, BuiltinError> {
    module.get_attr(name).ok_or_else(|| {
        BuiltinError::AttributeError(format!("module 'copyreg' has no attribute '{}'", name))
    })
}

#[inline]
fn dict_ref(value: Value, context: &'static str) -> Result<&'static DictObject, BuiltinError> {
    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError(format!("{context} must be a dict")))?;
    dict_storage_ref_from_ptr(ptr)
        .ok_or_else(|| BuiltinError::TypeError(format!("{context} must be a dict")))
}

#[inline]
fn dict_mut(value: Value, context: &'static str) -> Result<&'static mut DictObject, BuiltinError> {
    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError(format!("{context} must be a dict")))?;
    dict_storage_mut_from_ptr(ptr)
        .ok_or_else(|| BuiltinError::TypeError(format!("{context} must be a dict")))
}

#[inline]
fn extension_code(value: Value) -> Result<i64, BuiltinError> {
    let code = value_to_i64(value)
        .ok_or_else(|| BuiltinError::TypeError("extension code must be an integer".to_string()))?;
    if !(1..=0x7fff_ffff).contains(&code) {
        return Err(BuiltinError::ValueError("code out of range".to_string()));
    }
    Ok(code)
}

#[inline]
fn extension_key(module: Value, name: Value) -> Value {
    crate::alloc_managed_value(TupleObject::from_vec(vec![module, name]))
}

fn heap_class_from_value(
    value: Value,
    context: &'static str,
) -> Result<&'static PyClassObject, BuiltinError> {
    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError(format!("{context} first argument must be a type"))
    })?;
    if extract_type_id(ptr) != TypeId::TYPE
        || crate::builtins::builtin_type_object_type_id(ptr).is_some()
    {
        return Err(BuiltinError::TypeError(format!(
            "{context} currently requires a heap type"
        )));
    }
    Ok(unsafe { &*(ptr as *const PyClassObject) })
}
