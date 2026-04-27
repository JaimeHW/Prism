//! Native `weakref` compatibility module.
//!
//! This module provides the bootstrap surface needed by early CPython stdlib
//! imports such as `unittest.signals`, while Prism's lower-level `_weakref`
//! support continues to grow underneath it.

use super::{_weakref, Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject, builtin_set};
use crate::ops::objects::{dict_storage_mut_from_ptr, dict_storage_ref_from_ptr, extract_type_id};
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
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("weakref.WeakSet"), builtin_weak_set));
static WEAKDICT_LEN_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("weakref.WeakDictionary.__len__"), weak_dict_len)
});
static WEAKDICT_ITEMS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("weakref.WeakDictionary.items"), weak_dict_items)
});
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
static WEAK_DICTS: LazyLock<Mutex<Vec<usize>>> = LazyLock::new(|| Mutex::new(Vec::new()));

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

fn build_weak_dict_type(name: &str) -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern(name));
    class.set_attr(intern("__module__"), Value::string(intern("weakref")));
    class.set_attr(intern("__len__"), builtin_value(&WEAKDICT_LEN_FUNCTION));
    class.set_attr(intern("items"), builtin_value(&WEAKDICT_ITEMS_FUNCTION));
    class.add_flags(ClassFlags::INITIALIZED | ClassFlags::NATIVE_HEAPTYPE);

    let mut bitmap = SubclassBitmap::new();
    bitmap.set_bit(class.class_type_id());
    bitmap.set_bit(TypeId::OBJECT);
    bitmap.set_bit(TypeId::DICT);

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

#[inline]
fn aligned_object_ptr(value: Value) -> Option<*const ()> {
    let ptr = value.as_object_ptr()?;
    let addr = ptr as usize;
    (addr != 0 && addr % std::mem::align_of::<ObjectHeader>() == 0).then_some(ptr)
}

pub(crate) fn has_registered_weak_dicts() -> bool {
    !WEAK_DICTS
        .lock()
        .expect("weak dictionary registry lock poisoned")
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

pub(crate) fn clear_unreachable_weak_dicts(reachable: &FxHashSet<usize>) {
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
