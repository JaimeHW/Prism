//! Native `_abc` accelerator module.
//!
//! CPython's `abc.py` prefers this accelerator and only falls back to
//! `_py_abc` when `_abc` is unavailable. Providing the accelerator keeps the
//! import path short and avoids pulling in `_weakrefset` during bootstrap.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{
    BuiltinError, BuiltinFunctionObject, builtin_issubclass, builtin_type,
    builtin_type_object_for_type_id, builtin_type_object_type_id,
};
use crate::ops::objects::{descriptor_is_abstract, extract_type_id};
use prism_core::Value;
use prism_core::intern::{InternedString, intern, interned_by_ptr};
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::type_builtins::{class_id_to_type_id, global_class};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::set::SetObject;
use prism_runtime::types::string::StringObject;
use prism_runtime::types::tuple::TupleObject;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, RwLock};

#[derive(Clone, Default)]
struct AbcState {
    registry: FxHashSet<Value>,
    cache: FxHashSet<Value>,
    negative_cache: FxHashSet<Value>,
    negative_cache_version: u64,
}

static ABC_INVALIDATION_COUNTER: AtomicU64 = AtomicU64::new(0);
static ABC_STATES: LazyLock<RwLock<FxHashMap<u64, AbcState>>> =
    LazyLock::new(|| RwLock::new(FxHashMap::default()));

static GET_CACHE_TOKEN_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_abc.get_cache_token"), abc_get_cache_token)
});
static ABC_INIT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_abc._abc_init"), abc_init));
static ABC_REGISTER_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_abc._abc_register"), abc_register));
static ABC_INSTANCECHECK_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_abc._abc_instancecheck"), abc_instancecheck)
});
static ABC_SUBCLASSCHECK_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_abc._abc_subclasscheck"), abc_subclasscheck)
});
static GET_DUMP_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_abc._get_dump"), abc_get_dump));
static RESET_REGISTRY_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_abc._reset_registry"), abc_reset_registry)
});
static RESET_CACHES_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_abc._reset_caches"), abc_reset_caches));

pub struct AbcModule {
    attrs: Vec<Arc<str>>,
}

impl AbcModule {
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("get_cache_token"),
                Arc::from("_abc_init"),
                Arc::from("_abc_register"),
                Arc::from("_abc_instancecheck"),
                Arc::from("_abc_subclasscheck"),
                Arc::from("_get_dump"),
                Arc::from("_reset_registry"),
                Arc::from("_reset_caches"),
            ],
        }
    }
}

impl Default for AbcModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for AbcModule {
    fn name(&self) -> &str {
        "_abc"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "get_cache_token" => Ok(builtin_value(&GET_CACHE_TOKEN_FUNCTION)),
            "_abc_init" => Ok(builtin_value(&ABC_INIT_FUNCTION)),
            "_abc_register" => Ok(builtin_value(&ABC_REGISTER_FUNCTION)),
            "_abc_instancecheck" => Ok(builtin_value(&ABC_INSTANCECHECK_FUNCTION)),
            "_abc_subclasscheck" => Ok(builtin_value(&ABC_SUBCLASSCHECK_FUNCTION)),
            "_get_dump" => Ok(builtin_value(&GET_DUMP_FUNCTION)),
            "_reset_registry" => Ok(builtin_value(&RESET_REGISTRY_FUNCTION)),
            "_reset_caches" => Ok(builtin_value(&RESET_CACHES_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module '_abc' has no attribute '{}'",
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
    let ptr = Box::leak(Box::new(object)) as *mut T as *const ();
    Value::object_ptr(ptr)
}

#[inline]
fn to_frozenset_value(mut set: SetObject) -> Value {
    set.header.type_id = TypeId::FROZENSET;
    leak_object_value(set)
}

#[inline]
fn state_key(class_value: Value) -> u64 {
    class_value.raw_bits()
}

#[inline]
fn current_counter() -> u64 {
    ABC_INVALIDATION_COUNTER.load(Ordering::Relaxed)
}

fn ensure_class_value(value: Value, message: &str) -> Result<Value, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(message.to_string()));
    };

    match extract_type_id(ptr) {
        TypeId::TYPE | TypeId::EXCEPTION_TYPE => Ok(value),
        _ => Err(BuiltinError::TypeError(message.to_string())),
    }
}

fn user_class_from_value(value: Value) -> Option<&'static PyClassObject> {
    let ptr = value.as_object_ptr()?;
    if extract_type_id(ptr) != TypeId::TYPE || builtin_type_object_type_id(ptr).is_some() {
        return None;
    }

    Some(unsafe { &*(ptr as *const PyClassObject) })
}

fn class_value_from_class_id(class_id: ClassId) -> Option<Value> {
    if let Some(class) = global_class(class_id) {
        return Some(Value::object_ptr(Arc::as_ptr(&class) as *const ()));
    }

    let type_id = class_id_to_type_id(class_id);
    (type_id.raw() < TypeId::FIRST_USER_TYPE).then(|| builtin_type_object_for_type_id(type_id))
}

pub(crate) fn clear_abc_state_for_class_ids(class_ids: impl IntoIterator<Item = ClassId>) {
    let stale_values = class_ids
        .into_iter()
        .filter_map(class_value_from_class_id)
        .map(state_key)
        .collect::<FxHashSet<_>>();
    if stale_values.is_empty() {
        return;
    }

    let mut states = ABC_STATES.write().unwrap();
    states.retain(|class_key, state| {
        if stale_values.contains(class_key) {
            return false;
        }

        state
            .registry
            .retain(|value| !stale_values.contains(&value.raw_bits()));
        state
            .cache
            .retain(|value| !stale_values.contains(&value.raw_bits()));
        state
            .negative_cache
            .retain(|value| !stale_values.contains(&value.raw_bits()));
        true
    });
}

fn value_to_interned_string(value: Value) -> Option<InternedString> {
    if value.is_string() {
        let ptr = value.as_string_object_ptr()? as *const u8;
        return interned_by_ptr(ptr);
    }

    let ptr = value.as_object_ptr()?;
    if extract_type_id(ptr) != TypeId::STR {
        return None;
    }

    let string = unsafe { &*(ptr as *const StringObject) };
    Some(intern(string.as_str()))
}

fn names_from_abstract_value(value: Value) -> FxHashSet<InternedString> {
    let mut names = FxHashSet::default();
    let Some(ptr) = value.as_object_ptr() else {
        return names;
    };

    match extract_type_id(ptr) {
        TypeId::SET | TypeId::FROZENSET => {
            let set = unsafe { &*(ptr as *const SetObject) };
            for item in set.iter() {
                if let Some(name) = value_to_interned_string(item) {
                    names.insert(name);
                }
            }
        }
        _ => {}
    }

    names
}

fn abstract_method_names(class: &PyClassObject) -> FxHashSet<InternedString> {
    let mut names = FxHashSet::default();
    class.for_each_attr(|name, value| {
        if descriptor_is_abstract(value) {
            names.insert(name.clone());
        }
    });

    for &base_id in class.bases() {
        let Some(base_value) = class_value_from_class_id(base_id) else {
            continue;
        };
        let Some(base_class) = user_class_from_value(base_value) else {
            continue;
        };
        let Some(base_abstracts) = base_class.get_attr(&intern("__abstractmethods__")) else {
            continue;
        };

        for name in names_from_abstract_value(base_abstracts) {
            match class.get_attr(&name) {
                Some(value) if !descriptor_is_abstract(value) => {}
                _ => {
                    names.insert(name);
                }
            }
        }
    }

    names
}

fn frozenset_of_strings(names: impl IntoIterator<Item = InternedString>) -> Value {
    let values: Vec<Value> = names.into_iter().map(Value::string).collect();
    to_frozenset_value(SetObject::from_iter(values))
}

fn frozenset_of_values(values: impl IntoIterator<Item = Value>) -> Value {
    to_frozenset_value(SetObject::from_iter(values))
}

fn abc_state_snapshot(class_value: Value) -> AbcState {
    ABC_STATES
        .read()
        .unwrap()
        .get(&state_key(class_value))
        .cloned()
        .unwrap_or_else(|| AbcState {
            negative_cache_version: current_counter(),
            ..AbcState::default()
        })
}

fn with_abc_state_mut<R>(class_value: Value, f: impl FnOnce(&mut AbcState) -> R) -> R {
    let mut states = ABC_STATES.write().unwrap();
    let state = states
        .entry(state_key(class_value))
        .or_insert_with(|| AbcState {
            negative_cache_version: current_counter(),
            ..AbcState::default()
        });
    f(state)
}

fn python_issubclass(subclass: Value, cls: Value) -> Result<bool, BuiltinError> {
    builtin_issubclass(&[subclass, cls]).map(|value| value.as_bool().unwrap_or(false))
}

fn abc_subclasscheck_value(cls: Value, subclass: Value) -> Result<Value, BuiltinError> {
    let counter = current_counter();
    let mut registry_snapshot = Vec::new();

    {
        let mut states = ABC_STATES.write().unwrap();
        let state = states.entry(state_key(cls)).or_insert_with(|| AbcState {
            negative_cache_version: counter,
            ..AbcState::default()
        });

        if state.cache.contains(&subclass) {
            return Ok(Value::bool(true));
        }

        if state.negative_cache_version < counter {
            state.negative_cache.clear();
            state.negative_cache_version = counter;
        } else if state.negative_cache.contains(&subclass) {
            return Ok(Value::bool(false));
        }

        registry_snapshot.extend(state.registry.iter().copied());
    }

    if python_issubclass(subclass, cls)? {
        with_abc_state_mut(cls, |state| {
            state.cache.insert(subclass);
        });
        return Ok(Value::bool(true));
    }

    for registered in registry_snapshot {
        if python_issubclass(subclass, registered)? {
            with_abc_state_mut(cls, |state| {
                state.cache.insert(subclass);
            });
            return Ok(Value::bool(true));
        }
    }

    with_abc_state_mut(cls, |state| {
        state.negative_cache.insert(subclass);
        state.negative_cache_version = counter;
    });
    Ok(Value::bool(false))
}

fn abc_get_cache_token(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "get_cache_token() takes 0 arguments ({} given)",
            args.len()
        )));
    }

    Value::int(current_counter() as i64)
        .ok_or_else(|| BuiltinError::OverflowError("ABC cache token overflow".to_string()))
}

fn abc_init(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "_abc_init() takes 1 argument ({} given)",
            args.len()
        )));
    }

    let cls = ensure_class_value(args[0], "_abc_init expects a class")?;
    if let Some(class) = user_class_from_value(cls) {
        let abstracts = abstract_method_names(class);
        class.set_attr(
            intern("__abstractmethods__"),
            frozenset_of_strings(abstracts),
        );
    }

    with_abc_state_mut(cls, |state| {
        state.cache.clear();
        state.negative_cache.clear();
        state.negative_cache_version = current_counter();
    });
    Ok(Value::none())
}

fn abc_register(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "_abc_register() takes 2 arguments ({} given)",
            args.len()
        )));
    }

    let cls = ensure_class_value(args[0], "Can only register classes")?;
    let subclass = ensure_class_value(args[1], "Can only register classes")?;

    if python_issubclass(subclass, cls)? {
        return Ok(subclass);
    }

    with_abc_state_mut(cls, |state| {
        state.registry.insert(subclass);
    });
    ABC_INVALIDATION_COUNTER.fetch_add(1, Ordering::Relaxed);
    Ok(subclass)
}

fn abc_instancecheck(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "_abc_instancecheck() takes 2 arguments ({} given)",
            args.len()
        )));
    }

    let cls = ensure_class_value(args[0], "_abc_instancecheck expects a class")?;
    let instance_type = builtin_type(&[args[1]])?;
    abc_subclasscheck_value(cls, instance_type)
}

fn abc_subclasscheck(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "_abc_subclasscheck() takes 2 arguments ({} given)",
            args.len()
        )));
    }

    let cls = ensure_class_value(args[0], "_abc_subclasscheck expects a class")?;
    let subclass = ensure_class_value(args[1], "issubclass() arg 1 must be a class")?;
    abc_subclasscheck_value(cls, subclass)
}

fn abc_get_dump(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "_get_dump() takes 1 argument ({} given)",
            args.len()
        )));
    }

    let cls = ensure_class_value(args[0], "_get_dump expects a class")?;
    let state = abc_state_snapshot(cls);
    let version = Value::int(state.negative_cache_version as i64).ok_or_else(|| {
        BuiltinError::OverflowError("ABC negative cache version overflow".to_string())
    })?;
    Ok(leak_object_value(TupleObject::from_vec(vec![
        frozenset_of_values(state.registry.into_iter()),
        frozenset_of_values(state.cache.into_iter()),
        frozenset_of_values(state.negative_cache.into_iter()),
        version,
    ])))
}

fn abc_reset_registry(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "_reset_registry() takes 1 argument ({} given)",
            args.len()
        )));
    }

    let cls = ensure_class_value(args[0], "_reset_registry expects a class")?;
    with_abc_state_mut(cls, |state| {
        state.registry.clear();
    });
    Ok(Value::none())
}

fn abc_reset_caches(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "_reset_caches() takes 1 argument ({} given)",
            args.len()
        )));
    }

    let cls = ensure_class_value(args[0], "_reset_caches expects a class")?;
    with_abc_state_mut(cls, |state| {
        state.cache.clear();
        state.negative_cache.clear();
        state.negative_cache_version = current_counter();
    });
    Ok(Value::none())
}

#[cfg(test)]
mod tests;
