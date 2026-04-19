//! Native `_thread` module primitives.
//!
//! This module provides the lowest-level thread identity hooks needed by the
//! CPython stdlib bootstrap path. The implementation is intentionally small
//! but structured so additional synchronization primitives can be layered onto
//! the same module without changing the import surface.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject, ExceptionTypeObject, RUNTIME_ERROR};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use rustc_hash::FxHashMap;
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex};
use std::time::{Duration, Instant};

static NEXT_THREAD_IDENT: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static THREAD_IDENT: u64 = NEXT_THREAD_IDENT.fetch_add(1, Ordering::Relaxed);
}

static GET_IDENT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_thread.get_ident"), thread_get_ident));
static RLOCK_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_thread.RLock"), thread_rlock));
static RLOCK_ACQUIRE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_thread.RLock.acquire"), rlock_acquire));
static RLOCK_RELEASE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_thread.RLock.release"), rlock_release));
static RLOCK_ENTER_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_thread.RLock.__enter__"), rlock_enter));
static RLOCK_EXIT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_thread.RLock.__exit__"), rlock_exit));
static RLOCKS_BY_PTR: LazyLock<Mutex<FxHashMap<usize, Arc<NativeRLock>>>> =
    LazyLock::new(|| Mutex::new(FxHashMap::default()));

#[derive(Debug, Default)]
struct NativeRLockState {
    owner: Option<u64>,
    depth: usize,
}

#[derive(Debug, Default)]
struct NativeRLock {
    state: Mutex<NativeRLockState>,
    available: Condvar,
}

/// Native `_thread` module surface.
#[derive(Debug, Clone)]
pub struct ThreadModule {
    attrs: Vec<Arc<str>>,
}

impl ThreadModule {
    /// Create a new `_thread` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("RLock"),
                Arc::from("error"),
                Arc::from("get_ident"),
            ],
        }
    }
}

impl Default for ThreadModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ThreadModule {
    fn name(&self) -> &str {
        "_thread"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "RLock" => Ok(builtin_value(&RLOCK_FUNCTION)),
            "get_ident" => Ok(builtin_value(&GET_IDENT_FUNCTION)),
            "error" => Ok(runtime_error_type_value()),
            _ => Err(ModuleError::AttributeError(format!(
                "module '_thread' has no attribute '{}'",
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
fn runtime_error_type_value() -> Value {
    let exception_type: &ExceptionTypeObject = RUNTIME_ERROR.deref();
    Value::object_ptr(exception_type as *const ExceptionTypeObject as *const ())
}

#[inline]
fn current_thread_ident() -> u64 {
    THREAD_IDENT.with(|ident| *ident)
}

#[inline]
fn thread_get_ident(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "_thread.get_ident() takes 0 positional arguments but {} were given",
            args.len()
        )));
    }

    Ok(Value::int(current_thread_ident() as i64).expect("thread identifiers should fit in i64"))
}

#[inline]
fn builtin_attr_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

#[inline]
fn bound_builtin_attr_value(function: &'static BuiltinFunctionObject, receiver: Value) -> Value {
    let bound = Box::new(function.bind(receiver));
    Value::object_ptr(Box::leak(bound) as *mut BuiltinFunctionObject as *const ())
}

fn thread_rlock(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "_thread.RLock() takes 0 positional arguments but {} were given",
            args.len()
        )));
    }

    let registry = shape_registry();
    let mut object = Box::new(ShapedObject::with_empty_shape(registry.empty_shape()));
    let ptr = object.as_mut() as *mut ShapedObject;
    let value = Value::object_ptr(ptr as *const ());

    object.set_property(
        intern("acquire"),
        bound_builtin_attr_value(&RLOCK_ACQUIRE_FUNCTION, value),
        registry,
    );
    object.set_property(
        intern("release"),
        bound_builtin_attr_value(&RLOCK_RELEASE_FUNCTION, value),
        registry,
    );
    object.set_property(
        intern("__enter__"),
        bound_builtin_attr_value(&RLOCK_ENTER_FUNCTION, value),
        registry,
    );
    object.set_property(
        intern("__exit__"),
        bound_builtin_attr_value(&RLOCK_EXIT_FUNCTION, value),
        registry,
    );

    let ptr = Box::into_raw(object);
    RLOCKS_BY_PTR
        .lock()
        .expect("RLock registry lock poisoned")
        .insert(ptr as usize, Arc::new(NativeRLock::default()));
    Ok(value)
}

fn rlock_for_value(receiver: Value) -> Result<Arc<NativeRLock>, BuiltinError> {
    let ptr = receiver.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("RLock methods require an RLock instance".to_string())
    })?;
    RLOCKS_BY_PTR
        .lock()
        .expect("RLock registry lock poisoned")
        .get(&(ptr as usize))
        .cloned()
        .ok_or_else(|| {
            BuiltinError::TypeError("RLock methods require an RLock instance".to_string())
        })
}

fn parse_blocking_arg(value: Value) -> Result<bool, BuiltinError> {
    if let Some(flag) = value.as_bool() {
        return Ok(flag);
    }
    if let Some(flag) = value.as_int() {
        return Ok(flag != 0);
    }
    Err(BuiltinError::TypeError(
        "blocking argument must be a bool or int".to_string(),
    ))
}

fn parse_timeout_arg(value: Value) -> Result<Option<Duration>, BuiltinError> {
    if let Some(timeout) = value.as_int() {
        if timeout < 0 {
            return Ok(None);
        }
        return Ok(Some(Duration::from_secs(timeout as u64)));
    }
    if let Some(timeout) = value.as_float() {
        if timeout < 0.0 {
            return Ok(None);
        }
        return Ok(Some(Duration::from_secs_f64(timeout)));
    }
    Err(BuiltinError::TypeError(
        "timeout value must be an int or float".to_string(),
    ))
}

fn acquire_native_rlock(
    lock: &NativeRLock,
    blocking: bool,
    timeout: Option<Duration>,
) -> Result<bool, BuiltinError> {
    let current = current_thread_ident();
    let mut state = lock.state.lock().expect("RLock state lock poisoned");

    if state.owner.is_none() || state.owner == Some(current) {
        state.owner = Some(current);
        state.depth += 1;
        return Ok(true);
    }

    if !blocking {
        return Ok(false);
    }

    if let Some(timeout) = timeout {
        let deadline = Instant::now() + timeout;
        loop {
            let remaining = match deadline.checked_duration_since(Instant::now()) {
                Some(remaining) if !remaining.is_zero() => remaining,
                _ => return Ok(false),
            };
            let (next_state, wait_result) = lock
                .available
                .wait_timeout(state, remaining)
                .expect("RLock wait poisoned");
            state = next_state;
            if state.owner.is_none() {
                state.owner = Some(current);
                state.depth = 1;
                return Ok(true);
            }
            if wait_result.timed_out() {
                return Ok(false);
            }
        }
    }

    while state.owner.is_some() && state.owner != Some(current) {
        state = lock.available.wait(state).expect("RLock wait poisoned");
    }
    state.owner = Some(current);
    state.depth += 1;
    Ok(true)
}

fn release_native_rlock(lock: &NativeRLock) -> Result<(), BuiltinError> {
    let current = current_thread_ident();
    let mut state = lock.state.lock().expect("RLock state lock poisoned");
    if state.owner != Some(current) || state.depth == 0 {
        return Err(BuiltinError::TypeError(
            "cannot release un-acquired lock".to_string(),
        ));
    }

    state.depth -= 1;
    if state.depth == 0 {
        state.owner = None;
        lock.available.notify_one();
    }
    Ok(())
}

fn rlock_acquire(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "RLock.acquire() takes from 0 to 2 positional arguments but {} were given",
            args.len().saturating_sub(1)
        )));
    }

    let receiver = args[0];
    let blocking = args
        .get(1)
        .copied()
        .map(parse_blocking_arg)
        .transpose()?
        .unwrap_or(true);
    let timeout = args
        .get(2)
        .copied()
        .map(parse_timeout_arg)
        .transpose()?
        .flatten();
    let lock = rlock_for_value(receiver)?;
    Ok(Value::bool(acquire_native_rlock(&lock, blocking, timeout)?))
}

fn rlock_release(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "RLock.release() takes 0 positional arguments but {} were given",
            args.len().saturating_sub(1)
        )));
    }

    let lock = rlock_for_value(args[0])?;
    release_native_rlock(&lock)?;
    Ok(Value::none())
}

fn rlock_enter(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "RLock.__enter__() takes 0 positional arguments but {} were given",
            args.len().saturating_sub(1)
        )));
    }

    let receiver = args[0];
    let lock = rlock_for_value(receiver)?;
    acquire_native_rlock(&lock, true, None)?;
    Ok(receiver)
}

fn rlock_exit(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 4 {
        return Err(BuiltinError::TypeError(format!(
            "RLock.__exit__() takes 3 positional arguments but {} were given",
            args.len().saturating_sub(1)
        )));
    }

    let lock = rlock_for_value(args[0])?;
    release_native_rlock(&lock)?;
    Ok(Value::bool(false))
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_runtime::object::shaped_object::ShapedObject;

    #[test]
    fn test_thread_module_exposes_expected_attrs() {
        let module = ThreadModule::new();

        assert!(module.get_attr("RLock").is_ok());
        assert!(module.get_attr("get_ident").is_ok());
        assert!(module.get_attr("error").is_ok());
        assert_eq!(
            module
                .get_attr("missing")
                .expect_err("missing attribute should error")
                .to_string(),
            "AttributeError: module '_thread' has no attribute 'missing'"
        );
    }

    #[test]
    fn test_get_ident_is_stable_within_single_thread() {
        let first = thread_get_ident(&[])
            .expect("get_ident should succeed")
            .as_int()
            .expect("get_ident should return an int");
        let second = thread_get_ident(&[])
            .expect("get_ident should succeed")
            .as_int()
            .expect("get_ident should return an int");

        assert_eq!(first, second);
        assert!(first > 0);
    }

    #[test]
    fn test_get_ident_is_unique_across_threads() {
        let main_thread = thread_get_ident(&[])
            .expect("get_ident should succeed")
            .as_int()
            .expect("get_ident should return an int");
        let worker = std::thread::spawn(|| {
            thread_get_ident(&[])
                .expect("get_ident should succeed in worker")
                .as_int()
                .expect("worker identifier should be an int")
        })
        .join()
        .expect("worker thread should join");

        assert_ne!(main_thread, worker);
    }

    #[test]
    fn test_error_alias_points_to_runtime_error_type() {
        let module = ThreadModule::new();
        let error = module.get_attr("error").expect("error alias should exist");
        let expected = runtime_error_type_value();

        assert_eq!(error.as_object_ptr(), expected.as_object_ptr());
    }

    #[test]
    fn test_rlock_factory_installs_context_manager_methods() {
        let lock = thread_rlock(&[]).expect("RLock() should succeed");
        let ptr = lock
            .as_object_ptr()
            .expect("RLock() should return an object");
        let object = unsafe { &*(ptr as *const ShapedObject) };

        assert!(object.get_property("acquire").is_some());
        assert!(object.get_property("release").is_some());
        assert!(object.get_property("__enter__").is_some());
        assert!(object.get_property("__exit__").is_some());
    }

    #[test]
    fn test_rlock_factory_installs_bound_context_manager_methods() {
        let lock = thread_rlock(&[]).expect("RLock() should succeed");
        let ptr = lock
            .as_object_ptr()
            .expect("RLock() should return an object");
        let object = unsafe { &*(ptr as *const ShapedObject) };

        let enter = object
            .get_property("__enter__")
            .expect("__enter__ should be installed");
        let enter_ptr = enter
            .as_object_ptr()
            .expect("__enter__ should be a builtin function");
        let enter_fn = unsafe { &*(enter_ptr as *const BuiltinFunctionObject) };
        assert_eq!(enter_fn.bound_self(), Some(lock));

        let acquire = object
            .get_property("acquire")
            .expect("acquire should be installed");
        let acquire_ptr = acquire
            .as_object_ptr()
            .expect("acquire should be a builtin function");
        let acquire_fn = unsafe { &*(acquire_ptr as *const BuiltinFunctionObject) };
        assert_eq!(acquire_fn.bound_self(), Some(lock));
    }

    #[test]
    fn test_rlock_supports_reentrant_enter_and_release() {
        let lock = thread_rlock(&[]).expect("RLock() should succeed");

        assert_eq!(rlock_enter(&[lock]).unwrap(), lock);
        assert!(rlock_acquire(&[lock]).unwrap().as_bool().unwrap());
        assert!(rlock_release(&[lock]).unwrap().is_none());
        assert!(
            !rlock_exit(&[lock, Value::none(), Value::none(), Value::none()])
                .unwrap()
                .as_bool()
                .unwrap()
        );
    }

    #[test]
    fn test_rlock_nonblocking_acquire_detects_contention_across_threads() {
        let lock = thread_rlock(&[]).expect("RLock() should succeed");
        assert!(rlock_acquire(&[lock]).unwrap().as_bool().unwrap());

        let worker = std::thread::spawn(move || {
            rlock_acquire(&[lock, Value::bool(false)])
                .expect("non-blocking acquire should not error")
                .as_bool()
                .expect("acquire should return a bool")
        })
        .join()
        .expect("worker thread should join");

        assert!(!worker);
        assert!(rlock_release(&[lock]).unwrap().is_none());
    }
}
