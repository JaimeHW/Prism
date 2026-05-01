//! Native `threading` module bootstrap surface.
//!
//! Prism's low-level `_thread` module owns real thread primitives. This module
//! exposes the high-level `threading` import surface needed by stdlib helpers
//! while failing explicitly for object types that require VM-aware scheduling.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::python_numeric::int_like_value;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::allocation_context::alloc_static_value;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::list::ListObject;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

const EXPORTS: &[&str] = &[
    "get_ident",
    "get_native_id",
    "active_count",
    "current_thread",
    "enumerate",
    "main_thread",
    "local",
    "Condition",
    "Event",
    "Lock",
    "RLock",
    "Semaphore",
    "BoundedSemaphore",
    "Thread",
    "Barrier",
    "TIMEOUT_MAX",
];

static GET_IDENT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("threading.get_ident"), get_ident));
static GET_NATIVE_ID_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("threading.get_native_id"), get_ident));
static ACTIVE_COUNT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("threading.active_count"), active_count));
static CURRENT_THREAD_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("threading.current_thread"), current_thread)
});
static ENUMERATE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("threading.enumerate"), enumerate));
static MAIN_THREAD_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("threading.main_thread"), main_thread));
static CONDITION_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("threading.Condition"), condition));
static EVENT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("threading.Event"), event));
static LOCK_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("threading.Lock"), lock));
static RLOCK_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("threading.RLock"), rlock));
static SEMAPHORE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("threading.Semaphore"), semaphore));
static BOUNDED_SEMAPHORE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("threading.BoundedSemaphore"), bounded_semaphore)
});
static THREAD_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("threading.Thread"), thread));
static BARRIER_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("threading.Barrier"), barrier));
static THREAD_IS_ALIVE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("threading.Thread.is_alive"), is_alive));
static THREAD_OBJECTS: LazyLock<Mutex<HashMap<u64, Value>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Native `threading` module descriptor.
#[derive(Debug, Clone)]
pub struct ThreadingModule {
    attrs: Vec<Arc<str>>,
    all: Value,
}

impl ThreadingModule {
    /// Create a native `threading` module.
    pub fn new() -> Self {
        Self {
            attrs: EXPORTS.iter().copied().map(Arc::from).collect(),
            all: string_list_value(EXPORTS),
        }
    }
}

impl Default for ThreadingModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ThreadingModule {
    fn name(&self) -> &str {
        "threading"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__all__" => Ok(self.all),
            "get_ident" => Ok(builtin_value(&GET_IDENT_FUNCTION)),
            "get_native_id" => Ok(builtin_value(&GET_NATIVE_ID_FUNCTION)),
            "active_count" => Ok(builtin_value(&ACTIVE_COUNT_FUNCTION)),
            "current_thread" => Ok(builtin_value(&CURRENT_THREAD_FUNCTION)),
            "enumerate" => Ok(builtin_value(&ENUMERATE_FUNCTION)),
            "main_thread" => Ok(builtin_value(&MAIN_THREAD_FUNCTION)),
            "local" => Ok(super::_thread::local_type_value()),
            "Condition" => Ok(builtin_value(&CONDITION_FUNCTION)),
            "Event" => Ok(builtin_value(&EVENT_FUNCTION)),
            "Lock" => Ok(builtin_value(&LOCK_FUNCTION)),
            "RLock" => Ok(builtin_value(&RLOCK_FUNCTION)),
            "Semaphore" => Ok(builtin_value(&SEMAPHORE_FUNCTION)),
            "BoundedSemaphore" => Ok(builtin_value(&BOUNDED_SEMAPHORE_FUNCTION)),
            "Thread" => Ok(builtin_value(&THREAD_FUNCTION)),
            "Barrier" => Ok(builtin_value(&BARRIER_FUNCTION)),
            "TIMEOUT_MAX" => Ok(Value::float(f64::MAX)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'threading' has no attribute '{}'",
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

fn get_ident(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args("get_ident", args)?;
    thread_ident_value(super::_thread::current_thread_ident())
}

fn active_count(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args("active_count", args)?;
    Ok(Value::int(super::_thread::active_thread_count() as i64)
        .expect("thread count should fit in i64"))
}

fn enumerate(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args("enumerate", args)?;
    let mut idents = super::_thread::active_thread_idents();
    let main_ident = super::_thread::main_thread_ident();
    if !idents.contains(&main_ident) {
        idents.push(main_ident);
    }
    idents.sort_unstable();
    idents.dedup();

    let values = idents
        .into_iter()
        .map(thread_object_for_ident)
        .collect::<Vec<_>>();
    Ok(crate::alloc_managed_value(ListObject::from_iter(values)))
}

fn current_thread(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args("current_thread", args)?;
    Ok(thread_object_for_ident(
        super::_thread::current_thread_ident(),
    ))
}

fn main_thread(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args("main_thread", args)?;
    Ok(thread_object_for_ident(super::_thread::main_thread_ident()))
}

fn condition(_args: &[Value]) -> Result<Value, BuiltinError> {
    Err(unsupported("Condition"))
}

fn event(_args: &[Value]) -> Result<Value, BuiltinError> {
    Err(unsupported("Event"))
}

fn lock(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args("Lock", args)?;
    super::_thread::new_lock_value()
}

fn rlock(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args("RLock", args)?;
    super::_thread::new_rlock_value()
}

fn semaphore(_args: &[Value]) -> Result<Value, BuiltinError> {
    Err(unsupported("Semaphore"))
}

fn bounded_semaphore(_args: &[Value]) -> Result<Value, BuiltinError> {
    Err(unsupported("BoundedSemaphore"))
}

fn thread(_args: &[Value]) -> Result<Value, BuiltinError> {
    Err(unsupported("Thread"))
}

fn barrier(_args: &[Value]) -> Result<Value, BuiltinError> {
    Err(unsupported("Barrier"))
}

fn is_alive(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "Thread.is_alive() takes no arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let ident = thread_object_ident(args[0])?;
    Ok(Value::bool(
        ident == super::_thread::main_thread_ident()
            || super::_thread::active_thread_idents().contains(&ident),
    ))
}

fn thread_object_for_ident(ident: u64) -> Value {
    let mut objects = THREAD_OBJECTS
        .lock()
        .expect("threading thread-object cache lock poisoned");
    if let Some(value) = objects.get(&ident).copied() {
        return value;
    }

    let value = build_thread_object(ident);
    objects.insert(ident, value);
    value
}

fn build_thread_object(ident: u64) -> Value {
    let main_ident = super::_thread::main_thread_ident();
    let is_main = ident == main_ident;
    let name = if is_main {
        "MainThread".to_string()
    } else {
        format!("Dummy-{}", ident)
    };

    let registry = shape_registry();
    let mut object = Box::new(ShapedObject::new(TypeId::OBJECT, registry.empty_shape()));
    let value = Value::object_ptr(object.as_mut() as *mut ShapedObject as *const ());
    let ident_value = thread_ident_value(ident).expect("thread identifiers should fit in i64");

    object.set_property(intern("name"), Value::string(intern(&name)), registry);
    object.set_property(intern("_name"), Value::string(intern(&name)), registry);
    object.set_property(intern("ident"), ident_value, registry);
    object.set_property(intern("_ident"), ident_value, registry);
    object.set_property(intern("native_id"), ident_value, registry);
    object.set_property(intern("daemon"), Value::bool(!is_main), registry);
    object.set_property(intern("_daemonic"), Value::bool(!is_main), registry);
    object.set_property(
        intern("is_alive"),
        alloc_static_value(THREAD_IS_ALIVE_FUNCTION.bind(value)),
        registry,
    );

    Value::object_ptr(Box::into_raw(object) as *const ())
}

fn thread_object_ident(value: Value) -> Result<u64, BuiltinError> {
    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "descriptor 'Thread.is_alive' requires a thread object but received '{}'",
            value.type_name()
        ))
    })?;
    let object = unsafe { &*(ptr as *const ShapedObject) };
    let ident = object
        .get_property("ident")
        .and_then(int_like_value)
        .ok_or_else(|| {
            BuiltinError::TypeError(
                "descriptor 'Thread.is_alive' requires a thread object".to_string(),
            )
        })?;
    u64::try_from(ident)
        .map_err(|_| BuiltinError::OverflowError("thread identifier is out of range".to_string()))
}

fn thread_ident_value(ident: u64) -> Result<Value, BuiltinError> {
    let ident = i64::try_from(ident)
        .map_err(|_| BuiltinError::OverflowError("thread identifier is too large".to_string()))?;
    Value::int(ident)
        .ok_or_else(|| BuiltinError::OverflowError("thread identifier is too large".to_string()))
}

fn expect_no_args(function: &str, args: &[Value]) -> Result<(), BuiltinError> {
    if args.is_empty() {
        Ok(())
    } else {
        Err(BuiltinError::TypeError(format!(
            "{function}() takes no arguments ({} given)",
            args.len()
        )))
    }
}

fn unsupported(feature: &str) -> BuiltinError {
    BuiltinError::NotImplemented(format!(
        "threading.{feature} requires the high-level threading runtime"
    ))
}
