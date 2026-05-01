//! Native `weakref` compatibility module.
//!
//! This module provides the bootstrap surface needed by early CPython stdlib
//! imports such as `unittest.signals`, while Prism's lower-level `_weakref`
//! support continues to grow underneath it.

use super::{_weakref, Module, ModuleError, ModuleResult};
use crate::VirtualMachine;
use crate::builtins::{
    BuiltinError, BuiltinFunctionObject, iterator_to_value, runtime_error_to_builtin_error,
};
use crate::ops::iteration::collect_iterable_values;
use crate::ops::objects::{
    dict_storage_mut_from_ptr, dict_storage_ref_from_ptr, extract_type_id, set_storage_mut_from_ptr,
};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::class::{ClassFlags, PyClassObject};
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, global_class_bitmap, register_global_class,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::iter::IteratorObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::set::SetObject;
use prism_runtime::types::tuple::TupleObject;
use rustc_hash::FxHashSet;
use std::sync::{Arc, LazyLock, Mutex};

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
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("weakref.WeakSet"), builtin_weak_set));
static WEAKDICT_LEN_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("weakref.WeakDictionary.__len__"), weak_dict_len)
});
static WEAKDICT_ITEMS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("weakref.WeakDictionary.items"), weak_dict_items)
});
static WEAKDICT_CLEAR_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("weakref.WeakDictionary.clear"), weak_dict_clear)
});
static WEAKSET_LEN_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("weakref.WeakSet.__len__"), weak_set_len)
});
static WEAKSET_ITER_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("weakref.WeakSet.__iter__"), weak_set_iter)
});
static WEAKSET_CONTAINS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("weakref.WeakSet.__contains__"), weak_set_contains)
});
static WEAKSET_ADD_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("weakref.WeakSet.add"), weak_set_add));
static WEAKSET_DISCARD_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("weakref.WeakSet.discard"), weak_set_discard)
});
static WEAKSET_REMOVE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("weakref.WeakSet.remove"), weak_set_remove)
});
static WEAKSET_CLEAR_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("weakref.WeakSet.clear"), weak_set_clear)
});
static WEAKSET_UPDATE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("weakref.WeakSet.update"), weak_set_update)
});
static WEAKSET_COPY_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("weakref.WeakSet.copy"), weak_set_copy));
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
static WEAK_KEY_DICTIONARY_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_weak_dict_type("WeakKeyDictionary"));
static WEAK_VALUE_DICTIONARY_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_weak_dict_type("WeakValueDictionary"));
static WEAK_SET_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_weak_set_type("WeakSet"));
static WEAK_DICTS: LazyLock<Mutex<Vec<usize>>> = LazyLock::new(|| Mutex::new(Vec::new()));
static WEAK_SETS: LazyLock<Mutex<Vec<usize>>> = LazyLock::new(|| Mutex::new(Vec::new()));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WeakDictKind {
    Key,
    Value,
}

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
    new_weak_dict_from_args(WeakDictKind::Key, args)
}

fn builtin_weak_value_dictionary(args: &[Value]) -> Result<Value, BuiltinError> {
    new_weak_dict_from_args(WeakDictKind::Value, args)
}

fn builtin_weak_set(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    new_weak_set_from_args(vm, args)
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

fn build_weak_dict_type(name: &str) -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern(name));
    class.set_attr(intern("__module__"), Value::string(intern("weakref")));
    class.set_attr(intern("__len__"), builtin_value(&WEAKDICT_LEN_FUNCTION));
    class.set_attr(intern("items"), builtin_value(&WEAKDICT_ITEMS_FUNCTION));
    class.set_attr(intern("clear"), builtin_value(&WEAKDICT_CLEAR_FUNCTION));
    class.add_flags(ClassFlags::INITIALIZED | ClassFlags::NATIVE_HEAPTYPE);

    let mut bitmap = SubclassBitmap::new();
    bitmap.set_bit(class.class_type_id());
    bitmap.set_bit(TypeId::OBJECT);
    bitmap.set_bit(TypeId::DICT);

    let class = Arc::new(class);
    register_global_class(class.clone(), bitmap);
    class
}

fn build_weak_set_type(name: &str) -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern(name));
    class.set_attr(intern("__module__"), Value::string(intern("weakref")));
    class.set_attr(intern("__len__"), builtin_value(&WEAKSET_LEN_FUNCTION));
    class.set_attr(intern("__iter__"), builtin_value(&WEAKSET_ITER_FUNCTION));
    class.set_attr(
        intern("__contains__"),
        builtin_value(&WEAKSET_CONTAINS_FUNCTION),
    );
    class.set_attr(intern("add"), builtin_value(&WEAKSET_ADD_FUNCTION));
    class.set_attr(intern("discard"), builtin_value(&WEAKSET_DISCARD_FUNCTION));
    class.set_attr(intern("remove"), builtin_value(&WEAKSET_REMOVE_FUNCTION));
    class.set_attr(intern("clear"), builtin_value(&WEAKSET_CLEAR_FUNCTION));
    class.set_attr(intern("update"), builtin_value(&WEAKSET_UPDATE_FUNCTION));
    class.set_attr(intern("copy"), builtin_value(&WEAKSET_COPY_FUNCTION));
    class.add_flags(ClassFlags::INITIALIZED | ClassFlags::NATIVE_HEAPTYPE);

    let mut bitmap = SubclassBitmap::new();
    bitmap.set_bit(class.class_type_id());
    bitmap.set_bit(TypeId::OBJECT);
    bitmap.set_bit(TypeId::SET);

    let class = Arc::new(class);
    register_global_class(class.clone(), bitmap);
    class
}

#[inline]
fn weak_dict_class(kind: WeakDictKind) -> &'static Arc<PyClassObject> {
    match kind {
        WeakDictKind::Key => &WEAK_KEY_DICTIONARY_CLASS,
        WeakDictKind::Value => &WEAK_VALUE_DICTIONARY_CLASS,
    }
}

pub(crate) fn new_weak_set() -> Value {
    let class = &WEAK_SET_CLASS;
    let object = ShapedObject::new_set_backed(
        class.class_type_id(),
        Arc::clone(class.instance_shape()),
        TypeId::SET,
    );
    let value = leak_object_value(object);
    register_weak_set(value);
    value
}

fn new_weak_set_from_args(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "WeakSet() takes at most 1 argument ({} given)",
            args.len()
        )));
    }

    let value = new_weak_set();
    if let Some(source) = args.first().copied() {
        weak_set_update_from_iterable(vm, value, source)?;
    }
    Ok(value)
}

pub(crate) fn new_weak_dict(kind: WeakDictKind) -> Value {
    let class = weak_dict_class(kind);
    let object =
        ShapedObject::new_dict_backed(class.class_type_id(), Arc::clone(class.instance_shape()));
    let value = leak_object_value(object);
    register_weak_dict(value);
    value
}

fn new_weak_dict_from_args(kind: WeakDictKind, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "{}() takes at most 1 argument ({} given)",
            weak_dict_type_name(kind),
            args.len()
        )));
    }

    let value = new_weak_dict(kind);
    let Some(source) = args.first().copied() else {
        return Ok(value);
    };

    let Some(source_ptr) = source.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "{}() argument must be a mapping",
            weak_dict_type_name(kind)
        )));
    };
    let source_dict = dict_storage_ref_from_ptr(source_ptr).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "{}() argument must be a mapping",
            weak_dict_type_name(kind)
        ))
    })?;
    let target_ptr = value
        .as_object_ptr()
        .expect("weak dictionary instances are object pointers");
    let target = dict_storage_mut_from_ptr(target_ptr)
        .expect("weak dictionary instances always carry dict storage");
    for (key, item) in source_dict.iter() {
        target.set(key, item);
    }

    Ok(value)
}

#[inline]
fn weak_dict_type_name(kind: WeakDictKind) -> &'static str {
    match kind {
        WeakDictKind::Key => "WeakKeyDictionary",
        WeakDictKind::Value => "WeakValueDictionary",
    }
}

fn weak_set_len(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "__len__() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let set = weak_set_storage_mut(args[0], "__len__")?;
    prune_dead_weak_set_refs(set);
    Value::int(set.len() as i64)
        .ok_or_else(|| BuiltinError::OverflowError("set length does not fit in int".into()))
}

fn weak_set_iter(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "__iter__() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let set = weak_set_storage_mut(args[0], "__iter__")?;
    let values = live_weak_set_values(set);
    Ok(iterator_to_value(IteratorObject::from_values(values)))
}

fn weak_set_contains(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "__contains__() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    let set = weak_set_storage_mut(args[0], "__contains__")?;
    prune_dead_weak_set_refs(set);
    let Some(reference) = weak_set_probe_reference(args[1])? else {
        return Ok(Value::bool(false));
    };
    let contains = crate::ops::set_access::set_contains_item(vm, set, reference)
        .map_err(runtime_error_to_builtin_error)?;
    Ok(Value::bool(contains))
}

fn weak_set_add(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "add() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    let set = weak_set_storage_mut(args[0], "add")?;
    let reference = _weakref::new_reference(args[1], None)?;
    crate::ops::set_access::set_add_item(vm, set, reference)
        .map_err(runtime_error_to_builtin_error)?;
    Ok(Value::none())
}

fn weak_set_discard(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "discard() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    let set = weak_set_storage_mut(args[0], "discard")?;
    let _ = weak_set_remove_probe(vm, set, args[1], false)?;
    Ok(Value::none())
}

fn weak_set_remove(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "remove() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    let set = weak_set_storage_mut(args[0], "remove")?;
    if weak_set_remove_probe(vm, set, args[1], true)? {
        Ok(Value::none())
    } else {
        Err(BuiltinError::KeyError("item not in WeakSet".to_string()))
    }
}

fn weak_set_clear(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "clear() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let set = weak_set_storage_mut(args[0], "clear")?;
    set.clear();
    Ok(Value::none())
}

fn weak_set_update(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    let Some(self_value) = args.first().copied() else {
        return Err(BuiltinError::TypeError(
            "unbound WeakSet.update()".to_string(),
        ));
    };
    weak_set_storage_mut(self_value, "update")?;

    for iterable in &args[1..] {
        weak_set_update_from_iterable(vm, self_value, *iterable)?;
    }
    Ok(Value::none())
}

fn weak_set_copy(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "copy() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let source = weak_set_storage_mut(args[0], "copy")?;
    let refs = live_weak_set_refs(source);
    let copy = new_weak_set();
    let target = weak_set_storage_mut(copy, "copy")?;
    for reference in refs {
        target.add(reference);
    }
    Ok(copy)
}

fn weak_dict_len(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "__len__() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let dict = weak_dict_storage(args[0], "__len__")?;
    Value::int(dict.len() as i64)
        .ok_or_else(|| BuiltinError::OverflowError("dictionary length does not fit in int".into()))
}

fn weak_dict_items(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "items() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let dict = weak_dict_storage(args[0], "items")?;
    let mut items = ListObject::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let pair = leak_object_value(TupleObject::from_slice(&[key, value]));
        items.push(pair);
    }
    Ok(leak_object_value(items))
}

fn weak_dict_clear(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "clear() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let dict = weak_dict_storage_mut(args[0], "clear")?;
    dict.clear();
    Ok(Value::none())
}

fn weak_dict_storage(value: Value, method_name: &str) -> Result<&'static DictObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "descriptor '{method_name}' requires a weak dictionary"
        )));
    };
    dict_storage_ref_from_ptr(ptr).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "descriptor '{method_name}' requires a weak dictionary"
        ))
    })
}

fn weak_dict_storage_mut(
    value: Value,
    method_name: &str,
) -> Result<&'static mut DictObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "descriptor '{method_name}' requires a weak dictionary"
        )));
    };
    dict_storage_mut_from_ptr(ptr).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "descriptor '{method_name}' requires a weak dictionary"
        ))
    })
}

fn weak_set_storage_mut(
    value: Value,
    method_name: &str,
) -> Result<&'static mut SetObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "descriptor '{method_name}' requires a WeakSet"
        )));
    };
    if !is_weak_set_type_id(extract_type_id(ptr)) {
        return Err(BuiltinError::TypeError(format!(
            "descriptor '{method_name}' requires a WeakSet"
        )));
    }
    set_storage_mut_from_ptr(ptr).ok_or_else(|| {
        BuiltinError::TypeError(format!("descriptor '{method_name}' requires a WeakSet"))
    })
}

fn weak_set_update_from_iterable(
    vm: &mut VirtualMachine,
    set_value: Value,
    iterable: Value,
) -> Result<(), BuiltinError> {
    let values = collect_iterable_values(vm, iterable).map_err(runtime_error_to_builtin_error)?;
    let set = weak_set_storage_mut(set_value, "update")?;
    for value in values {
        let reference = _weakref::new_reference(value, None)?;
        crate::ops::set_access::set_add_item(vm, set, reference)
            .map_err(runtime_error_to_builtin_error)?;
    }
    Ok(())
}

fn weak_set_probe_reference(value: Value) -> Result<Option<Value>, BuiltinError> {
    match _weakref::new_reference(value, None) {
        Ok(reference) => Ok(Some(reference)),
        Err(BuiltinError::TypeError(_)) => Ok(None),
        Err(err) => Err(err),
    }
}

fn weak_set_remove_probe(
    vm: &mut VirtualMachine,
    set: &mut SetObject,
    item: Value,
    propagate_type_error: bool,
) -> Result<bool, BuiltinError> {
    prune_dead_weak_set_refs(set);
    let reference = match _weakref::new_reference(item, None) {
        Ok(reference) => reference,
        Err(BuiltinError::TypeError(_)) if !propagate_type_error => return Ok(false),
        Err(err) => return Err(err),
    };
    crate::ops::set_access::set_remove_item(vm, set, reference)
        .map_err(runtime_error_to_builtin_error)
}

fn live_weak_set_values(set: &mut SetObject) -> Vec<Value> {
    live_weak_set_entries(set)
        .into_iter()
        .map(|(_, target)| target)
        .collect()
}

fn live_weak_set_refs(set: &mut SetObject) -> Vec<Value> {
    live_weak_set_entries(set)
        .into_iter()
        .map(|(reference, _)| reference)
        .collect()
}

fn live_weak_set_entries(set: &mut SetObject) -> Vec<(Value, Value)> {
    let mut dead_refs = Vec::new();
    let mut live_refs = Vec::with_capacity(set.len());
    for reference in set.iter().collect::<Vec<_>>() {
        match _weakref::reference_target(reference) {
            Some(target) if !target.is_none() => live_refs.push((reference, target)),
            _ => dead_refs.push(reference),
        }
    }
    remove_weak_set_refs(set, dead_refs);
    live_refs
}

fn prune_dead_weak_set_refs(set: &mut SetObject) {
    let dead_refs = set
        .iter()
        .filter(|reference| {
            _weakref::reference_target(*reference).is_none_or(|target| target.is_none())
        })
        .collect::<Vec<_>>();
    remove_weak_set_refs(set, dead_refs);
}

fn remove_weak_set_refs(set: &mut SetObject, refs: Vec<Value>) {
    for reference in refs {
        set.remove(reference);
    }
}

fn register_weak_dict(value: Value) {
    if let Some(ptr) = aligned_object_ptr(value) {
        let ptr = ptr as usize;
        let mut weak_dicts = WEAK_DICTS
            .lock()
            .expect("weak dictionary registry lock poisoned");
        if !weak_dicts.contains(&ptr) {
            weak_dicts.push(ptr);
        }
    }
}

fn register_weak_set(value: Value) {
    if let Some(ptr) = aligned_object_ptr(value) {
        let ptr = ptr as usize;
        let mut weak_sets = WEAK_SETS.lock().expect("weak set registry lock poisoned");
        if !weak_sets.contains(&ptr) {
            weak_sets.push(ptr);
        }
    }
}

#[inline]
fn aligned_object_ptr(value: Value) -> Option<*const ()> {
    let ptr = value.as_object_ptr()?;
    let addr = ptr as usize;
    (addr != 0 && addr % std::mem::align_of::<ObjectHeader>() == 0).then_some(ptr)
}

pub(crate) fn has_registered_weak_containers() -> bool {
    !WEAK_DICTS
        .lock()
        .expect("weak dictionary registry lock poisoned")
        .is_empty()
        || !WEAK_SETS
            .lock()
            .expect("weak set registry lock poisoned")
            .is_empty()
}

pub(crate) fn weak_dict_kind(value: Value) -> Option<WeakDictKind> {
    let ptr = aligned_object_ptr(value)?;
    weak_dict_kind_for_type_id(extract_type_id(ptr))
}

pub(crate) fn weak_dict_kind_for_type_id(type_id: TypeId) -> Option<WeakDictKind> {
    if type_id == WEAK_KEY_DICTIONARY_CLASS.class_type_id() {
        return Some(WeakDictKind::Key);
    }
    if type_id == WEAK_VALUE_DICTIONARY_CLASS.class_type_id() {
        return Some(WeakDictKind::Value);
    }
    if type_id.raw() < TypeId::FIRST_USER_TYPE {
        return None;
    }

    let bitmap = global_class_bitmap(ClassId(type_id.raw()))?;
    if bitmap.is_subclass_of(WEAK_KEY_DICTIONARY_CLASS.class_type_id()) {
        Some(WeakDictKind::Key)
    } else if bitmap.is_subclass_of(WEAK_VALUE_DICTIONARY_CLASS.class_type_id()) {
        Some(WeakDictKind::Value)
    } else {
        None
    }
}

pub(crate) fn is_weak_set_type_id(type_id: TypeId) -> bool {
    if type_id == WEAK_SET_CLASS.class_type_id() {
        return true;
    }
    if type_id.raw() < TypeId::FIRST_USER_TYPE {
        return false;
    }

    global_class_bitmap(ClassId(type_id.raw()))
        .is_some_and(|bitmap| bitmap.is_subclass_of(WEAK_SET_CLASS.class_type_id()))
}

pub(crate) fn clear_unreachable_weak_containers(reachable: &FxHashSet<usize>) {
    clear_unreachable_weak_dicts(reachable);
    clear_unreachable_weak_sets(reachable);
}

fn clear_unreachable_weak_dicts(reachable: &FxHashSet<usize>) {
    let mut weak_dicts = WEAK_DICTS
        .lock()
        .expect("weak dictionary registry lock poisoned");

    weak_dicts.retain(|ptr| {
        if !reachable.contains(ptr) {
            return false;
        }

        let value = Value::object_ptr(*ptr as *const ());
        let Some(kind) = weak_dict_kind(value) else {
            return false;
        };
        let Some(dict) = dict_storage_mut_from_ptr(*ptr as *const ()) else {
            return false;
        };

        let dead_keys = dict
            .iter()
            .filter_map(|(key, item)| {
                weak_dict_entry_is_dead(kind, key, item, reachable).then_some(key)
            })
            .collect::<Vec<_>>();
        for key in dead_keys {
            dict.remove(key);
        }

        true
    });
}

fn clear_unreachable_weak_sets(reachable: &FxHashSet<usize>) {
    let mut weak_sets = WEAK_SETS.lock().expect("weak set registry lock poisoned");

    weak_sets.retain(|ptr| {
        if !reachable.contains(ptr) {
            return false;
        }

        let Some(set) = set_storage_mut_from_ptr(*ptr as *const ()) else {
            return false;
        };
        prune_dead_weak_set_refs(set);
        true
    });
}

#[inline]
fn weak_dict_entry_is_dead(
    kind: WeakDictKind,
    key: Value,
    value: Value,
    reachable: &FxHashSet<usize>,
) -> bool {
    match kind {
        WeakDictKind::Key => weak_side_is_unreachable(key, reachable),
        WeakDictKind::Value => weak_side_is_unreachable(value, reachable),
    }
}

#[inline]
fn weak_side_is_unreachable(value: Value, reachable: &FxHashSet<usize>) -> bool {
    value
        .as_object_ptr()
        .map(|ptr| !reachable.contains(&(ptr as usize)))
        .unwrap_or(false)
}
