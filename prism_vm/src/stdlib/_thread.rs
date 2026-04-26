//! Native `_thread` module primitives.
//!
//! This module provides the lowest-level thread identity hooks needed by the
//! CPython stdlib bootstrap path. The implementation is intentionally small
//! but structured so additional synchronization primitives can be layered onto
//! the same module without changing the import surface.

use super::{Module, ModuleError, ModuleResult};
use crate::VirtualMachine;
use crate::builtins::{
    BuiltinError, BuiltinFunctionObject, ExceptionTypeObject, RUNTIME_ERROR,
    runtime_error_to_builtin_error,
};
use crate::error::RuntimeError;
use crate::import::ModuleObject;
use crate::ops::calls::{
    invoke_callable_value, invoke_callable_value_with_keywords, value_supports_call_protocol,
};
use crate::stdlib::exceptions::ExceptionTypeId;
use prism_code::CodeObject;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::class::{ClassFlags, PyClassObject};
use prism_runtime::object::descriptor::{BoundMethod, StaticMethodDescriptor};
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, class_id_to_type_id, register_global_class,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::function::FunctionObject;
use prism_runtime::types::string::value_as_string_ref;
use prism_runtime::types::tuple::{TupleObject, value_as_tuple_ref};
use rustc_hash::{FxHashMap, FxHashSet};
use std::cell::{Cell, RefCell};
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

static NEXT_THREAD_IDENT: AtomicU64 = AtomicU64::new(1);
static ACTIVE_THREAD_COUNT: AtomicU64 = AtomicU64::new(1);
static NEXT_INTERRUPT_TARGET: AtomicU64 = AtomicU64::new(1);
static PENDING_MAIN_INTERRUPTS: LazyLock<Mutex<FxHashMap<u64, i64>>> =
    LazyLock::new(|| Mutex::new(FxHashMap::default()));
static ACTIVE_THREAD_IDENTS: LazyLock<Mutex<FxHashSet<u64>>> =
    LazyLock::new(|| Mutex::new(FxHashSet::default()));
static PENDING_ASYNC_EXCEPTIONS: LazyLock<Mutex<FxHashMap<u64, Value>>> =
    LazyLock::new(|| Mutex::new(FxHashMap::default()));
thread_local! {
    static THREAD_IDENT: Cell<Option<u64>> = const { Cell::new(None) };
    static THREAD_SENTINELS: RefCell<Vec<Arc<NativeLock>>> = const { RefCell::new(Vec::new()) };
    static THREAD_INTERRUPT_TARGET: Cell<u64> = const { Cell::new(0) };
    static THREAD_IS_PYTHON_WORKER: Cell<bool> = const { Cell::new(false) };
}

pub(crate) type SharedThreadGroup = Arc<ThreadGroup>;

#[derive(Debug, Default)]
pub(crate) struct ThreadGroup {
    handles: Mutex<FxHashMap<u64, JoinHandle<()>>>,
}

impl ThreadGroup {
    #[inline]
    fn register(&self, ident: u64, handle: JoinHandle<()>) {
        self.handles
            .lock()
            .expect("thread handle registry lock poisoned")
            .insert(ident, handle);
    }

    #[inline]
    pub(crate) fn handle_count(&self) -> u64 {
        self.handles
            .lock()
            .expect("thread handle registry lock poisoned")
            .len() as u64
    }

    pub(crate) fn join_finished(&self) {
        let mut finished = Vec::new();
        {
            let mut handles = self
                .handles
                .lock()
                .expect("thread handle registry lock poisoned");
            let finished_idents = handles
                .iter()
                .filter_map(|(ident, handle)| handle.is_finished().then_some(*ident))
                .collect::<Vec<_>>();
            for ident in finished_idents {
                if let Some(handle) = handles.remove(&ident) {
                    finished.push(handle);
                }
            }
        }

        for handle in finished {
            let _ = handle.join();
        }
    }

    pub(crate) fn join_all_finished_before(&self, deadline: Instant) -> bool {
        loop {
            self.join_finished();
            if self.handle_count() == 0 {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            thread::sleep(Duration::from_millis(1));
        }
    }
}

pub(crate) fn new_thread_group() -> SharedThreadGroup {
    Arc::new(ThreadGroup::default())
}

static GET_IDENT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_thread.get_ident"), thread_get_ident));
static GET_NATIVE_ID_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_thread.get_native_id"), thread_get_native_id)
});
static COUNT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_thread._count"), thread_count));
static IS_MAIN_INTERPRETER_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_thread._is_main_interpreter"),
        thread_is_main_interpreter,
    )
});
static LOCAL_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(build_local_class);
static START_NEW_THREAD_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("_thread.start_new_thread"),
        thread_start_new_thread,
    )
});
static INTERRUPT_MAIN_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("_thread.interrupt_main"), thread_interrupt_main)
});
static DAEMON_THREADS_ALLOWED_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_thread.daemon_threads_allowed"),
        thread_daemon_threads_allowed,
    )
});
static STACK_SIZE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_thread.stack_size"), thread_stack_size)
});
static ALLOCATE_LOCK_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_thread.allocate_lock"), thread_allocate_lock)
});
static SET_SENTINEL_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_thread._set_sentinel"), thread_set_sentinel)
});
static RLOCK_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_thread.RLock"), thread_rlock));
static LOCK_ACQUIRE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(Arc::from("_thread.lock.acquire"), lock_acquire_kw)
});
static LOCK_RELEASE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_thread.lock.release"), lock_release));
static LOCK_LOCKED_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_thread.lock.locked"), lock_locked));
static LOCK_REPR_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_thread.lock.__repr__"), lock_repr));
static LOCK_ENTER_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_thread.lock.__enter__"), lock_enter));
static LOCK_EXIT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_thread.lock.__exit__"), lock_exit));
static LOCK_AT_FORK_REINIT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_thread.lock._at_fork_reinit"),
        lock_at_fork_reinit,
    )
});
static RLOCK_ACQUIRE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(Arc::from("_thread.RLock.acquire"), rlock_acquire_kw)
});
static RLOCK_RELEASE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_thread.RLock.release"), rlock_release));
static RLOCK_LOCKED_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_thread.RLock.locked"), rlock_locked));
static RLOCK_IS_OWNED_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_thread.RLock._is_owned"), rlock_is_owned)
});
static RLOCK_RECURSION_COUNT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_thread.RLock._recursion_count"),
        rlock_recursion_count,
    )
});
static RLOCK_RELEASE_SAVE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_thread.RLock._release_save"), rlock_release_save)
});
static RLOCK_ACQUIRE_RESTORE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_thread.RLock._acquire_restore"),
        rlock_acquire_restore,
    )
});
static RLOCK_AT_FORK_REINIT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_thread.RLock._at_fork_reinit"),
        rlock_at_fork_reinit,
    )
});
static RLOCK_REPR_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_thread.RLock.__repr__"), rlock_repr));
static RLOCK_ENTER_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_thread.RLock.__enter__"), rlock_enter));
static RLOCK_EXIT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_thread.RLock.__exit__"), rlock_exit));
static TIMEOUT_MAX_VALUE: LazyLock<Value> = LazyLock::new(|| Value::float(i32::MAX as f64));
static STACK_SIZE_BYTES: LazyLock<Mutex<usize>> = LazyLock::new(|| Mutex::new(0));
static LOCKS_BY_PTR: LazyLock<Mutex<FxHashMap<usize, Arc<NativeLock>>>> =
    LazyLock::new(|| Mutex::new(FxHashMap::default()));
static RLOCKS_BY_PTR: LazyLock<Mutex<FxHashMap<usize, Arc<NativeRLock>>>> =
    LazyLock::new(|| Mutex::new(FxHashMap::default()));

#[derive(Debug, Default)]
struct NativeLockState {
    locked: bool,
}

#[derive(Debug, Default)]
struct NativeLock {
    state: Mutex<NativeLockState>,
    available: Condvar,
}

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
                Arc::from("TIMEOUT_MAX"),
                Arc::from("_count"),
                Arc::from("_is_main_interpreter"),
                Arc::from("_local"),
                Arc::from("_set_sentinel"),
                Arc::from("allocate_lock"),
                Arc::from("daemon_threads_allowed"),
                Arc::from("error"),
                Arc::from("get_ident"),
                Arc::from("get_native_id"),
                Arc::from("interrupt_main"),
                Arc::from("start_new_thread"),
                Arc::from("stack_size"),
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
            "TIMEOUT_MAX" => Ok(*TIMEOUT_MAX_VALUE),
            "_count" => Ok(builtin_value(&COUNT_FUNCTION)),
            "_is_main_interpreter" => Ok(builtin_value(&IS_MAIN_INTERPRETER_FUNCTION)),
            "_local" => Ok(local_type_value()),
            "_set_sentinel" => Ok(builtin_value(&SET_SENTINEL_FUNCTION)),
            "allocate_lock" => Ok(builtin_value(&ALLOCATE_LOCK_FUNCTION)),
            "daemon_threads_allowed" => Ok(builtin_value(&DAEMON_THREADS_ALLOWED_FUNCTION)),
            "get_ident" => Ok(builtin_value(&GET_IDENT_FUNCTION)),
            "get_native_id" => Ok(builtin_value(&GET_NATIVE_ID_FUNCTION)),
            "interrupt_main" => Ok(builtin_value(&INTERRUPT_MAIN_FUNCTION)),
            "start_new_thread" => Ok(builtin_value(&START_NEW_THREAD_FUNCTION)),
            "stack_size" => Ok(builtin_value(&STACK_SIZE_FUNCTION)),
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
    THREAD_IDENT.with(|ident| {
        if let Some(existing) = ident.get() {
            existing
        } else {
            let assigned = NEXT_THREAD_IDENT.fetch_add(1, Ordering::Relaxed);
            ident.set(Some(assigned));
            register_thread_ident(assigned);
            assigned
        }
    })
}

#[inline]
fn set_current_thread_ident(ident: u64) {
    THREAD_IDENT.with(|slot| slot.set(Some(ident)));
    register_thread_ident(ident);
}

#[inline]
fn register_thread_ident(ident: u64) {
    ACTIVE_THREAD_IDENTS
        .lock()
        .expect("active thread registry lock poisoned")
        .insert(ident);
}

fn unregister_thread_ident(ident: u64) {
    ACTIVE_THREAD_IDENTS
        .lock()
        .expect("active thread registry lock poisoned")
        .remove(&ident);
    PENDING_ASYNC_EXCEPTIONS
        .lock()
        .expect("pending async exception registry lock poisoned")
        .remove(&ident);
}

pub(crate) fn new_main_interrupt_target() -> u64 {
    let _ = current_thread_ident();
    let target = NEXT_INTERRUPT_TARGET.fetch_add(1, Ordering::Relaxed);
    THREAD_INTERRUPT_TARGET.with(|slot| slot.set(target));
    THREAD_IS_PYTHON_WORKER.with(|slot| slot.set(false));
    target
}

#[inline]
fn set_interrupt_context(target: u64, is_python_worker: bool) {
    THREAD_INTERRUPT_TARGET.with(|slot| slot.set(target));
    THREAD_IS_PYTHON_WORKER.with(|slot| slot.set(is_python_worker));
}

#[inline]
fn current_thread_is_python_worker() -> bool {
    THREAD_IS_PYTHON_WORKER.with(Cell::get)
}

#[inline]
fn thread_get_native_id(args: &[Value]) -> Result<Value, BuiltinError> {
    thread_get_ident(args)
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
fn thread_count(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "_thread._count() takes 0 positional arguments but {} were given",
            args.len()
        )));
    }

    Ok(
        Value::int(ACTIVE_THREAD_COUNT.load(Ordering::SeqCst) as i64)
            .expect("thread count should fit in i64"),
    )
}

fn thread_is_main_interpreter(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "_thread._is_main_interpreter() takes 0 positional arguments but {} were given",
            args.len()
        )));
    }

    Ok(Value::bool(true))
}

fn build_local_class() -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern("_local"));
    class.set_attr(intern("__module__"), Value::string(intern("_thread")));
    class.set_attr(intern("__qualname__"), Value::string(intern("_local")));
    class.add_flags(ClassFlags::INITIALIZED | ClassFlags::NATIVE_HEAPTYPE);

    let class = Arc::new(class);
    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(class_id_to_type_id(class_id));
    }
    register_global_class(Arc::clone(&class), bitmap);
    class
}

#[inline]
fn local_type_value() -> Value {
    Value::object_ptr(Arc::as_ptr(&LOCAL_CLASS) as *const ())
}

#[cfg(test)]
pub(crate) fn active_thread_count() -> u64 {
    ACTIVE_THREAD_COUNT.load(Ordering::SeqCst)
}

#[cfg(test)]
pub(crate) fn wait_for_active_thread_count_at_most(target: u64, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        if active_thread_count() <= target {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        thread::sleep(Duration::from_millis(1));
    }
}

fn thread_interrupt_main(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "_thread.interrupt_main() takes at most 1 argument ({} given)",
            args.len()
        )));
    }

    let signum = match args.first().copied() {
        Some(value) => parse_interrupt_signal(value)?,
        None => super::signal::SIGINT,
    };

    if current_thread_is_python_worker() {
        queue_pending_main_interrupt(vm.thread_interrupt_target(), signum);
    } else {
        deliver_interrupt_signal(vm, signum).map_err(runtime_error_to_builtin_error)?;
    }

    Ok(Value::none())
}

#[inline]
fn parse_interrupt_signal(value: Value) -> Result<i64, BuiltinError> {
    let Some(signum) = value.as_int() else {
        return Err(BuiltinError::TypeError(
            "interrupt_main() signal number must be int".to_string(),
        ));
    };
    if super::signal::is_valid_signal_number(signum) {
        Ok(signum)
    } else {
        Err(BuiltinError::ValueError(
            "interrupt_main() signal number out of range".to_string(),
        ))
    }
}

#[inline]
fn queue_pending_main_interrupt(target: u64, signum: i64) {
    PENDING_MAIN_INTERRUPTS
        .lock()
        .expect("pending interrupt registry lock poisoned")
        .insert(target, signum);
}

pub(crate) fn take_pending_main_interrupt(target: u64) -> Option<i64> {
    if current_thread_is_python_worker() {
        return None;
    }

    PENDING_MAIN_INTERRUPTS
        .lock()
        .expect("pending interrupt registry lock poisoned")
        .remove(&target)
}

pub(crate) fn set_pending_async_exception_for_ident(ident: u64, exception: Value) -> bool {
    let is_active = ACTIVE_THREAD_IDENTS
        .lock()
        .expect("active thread registry lock poisoned")
        .contains(&ident);
    if !is_active {
        return false;
    }

    PENDING_ASYNC_EXCEPTIONS
        .lock()
        .expect("pending async exception registry lock poisoned")
        .insert(ident, exception);
    true
}

pub(crate) fn take_pending_async_exception_for_current_thread() -> Option<Value> {
    let ident = current_thread_ident();
    PENDING_ASYNC_EXCEPTIONS
        .lock()
        .expect("pending async exception registry lock poisoned")
        .remove(&ident)
}

pub(crate) fn deliver_interrupt_signal(
    vm: &mut VirtualMachine,
    signum: i64,
) -> Result<(), RuntimeError> {
    let handler = super::signal::handler_for_signal(signum);
    if super::signal::is_default_or_ignored_handler(handler) {
        return Ok(());
    }

    let signum_value = Value::int(signum).expect("signal number should fit in Value::int");
    invoke_callable_value(vm, handler, &[signum_value, Value::none()])?;
    Ok(())
}

fn thread_start_new_thread(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "_thread.start_new_thread() takes 2 or 3 positional arguments but {} were given",
            args.len()
        )));
    }

    let call_args = value_as_tuple_ref(args[1])
        .ok_or_else(|| BuiltinError::TypeError("2nd arg must be a tuple".to_string()))?
        .as_slice()
        .to_vec();
    let keywords = thread_kwargs(args.get(2).copied())?;
    let callable = args[0];
    if !value_supports_call_protocol(callable) {
        return Err(BuiltinError::TypeError(
            "first arg must be callable".to_string(),
        ));
    }

    let ident = NEXT_THREAD_IDENT.fetch_add(1, Ordering::Relaxed);
    let interrupt_target = vm.thread_interrupt_target();
    let builtins = vm.builtins.clone();
    let import_resolver = vm.import_resolver.clone();
    let shared_heap = vm.shared_heap();
    let class_publications = vm.class_publications();
    let codec_registry = vm.shared_codec_registry();
    let thread_group = vm.thread_group();
    let worker_thread_group = Arc::clone(&thread_group);

    ACTIVE_THREAD_COUNT.fetch_add(1, Ordering::SeqCst);
    let spawn_result = thread::Builder::new()
        .name(format!("prism-python-thread-{ident}"))
        .spawn(move || {
            set_current_thread_ident(ident);
            set_interrupt_context(interrupt_target, true);

            run_thread_callable(
                ident,
                callable,
                call_args,
                keywords,
                shared_heap,
                interrupt_target,
                builtins,
                import_resolver,
                class_publications,
                codec_registry,
                worker_thread_group,
            );

            release_thread_sentinels();
            unregister_thread_ident(ident);
            ACTIVE_THREAD_COUNT.fetch_sub(1, Ordering::SeqCst);
        });
    match spawn_result {
        Ok(handle) => thread_group.register(ident, handle),
        Err(err) => {
            ACTIVE_THREAD_COUNT.fetch_sub(1, Ordering::SeqCst);
            return Err(runtime_error_builtin(format!(
                "can't start new thread: {err}"
            )));
        }
    }

    Ok(Value::int(ident as i64).expect("thread identifiers should fit in i64"))
}

fn runtime_error_builtin(message: impl Into<Arc<str>>) -> BuiltinError {
    BuiltinError::Raised(RuntimeError::exception(
        ExceptionTypeId::RuntimeError.as_u8() as u16,
        message,
    ))
}

fn thread_kwargs(value: Option<Value>) -> Result<Vec<(String, Value)>, BuiltinError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    if value.is_none() {
        return Ok(Vec::new());
    }

    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("optional 3rd arg must be a dict".to_string()))?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != TypeId::DICT {
        return Err(BuiltinError::TypeError(
            "optional 3rd arg must be a dict".to_string(),
        ));
    }

    let dict = unsafe { &*(ptr as *const DictObject) };
    let mut keywords = Vec::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let key = value_as_string_ref(key)
            .ok_or_else(|| BuiltinError::TypeError("keywords must be strings".to_string()))?;
        keywords.push((key.as_str().to_string(), value));
    }
    Ok(keywords)
}

fn run_thread_callable(
    ident: u64,
    callable: Value,
    args: Vec<Value>,
    keywords: Vec<(String, Value)>,
    shared_heap: crate::vm::SharedManagedHeap,
    interrupt_target: u64,
    builtins: crate::builtins::BuiltinRegistry,
    import_resolver: crate::import::ImportResolver,
    class_publications: crate::vm::SharedClassPublications,
    codec_registry: crate::stdlib::_codecs::SharedCodecRegistry,
    thread_group: SharedThreadGroup,
) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _execution_region = crate::threading_runtime::enter_execution_region();
        let mut vm = VirtualMachine::with_shared_heap_and_import_resolver(
            shared_heap,
            interrupt_target,
            builtins,
            import_resolver,
            class_publications,
            codec_registry,
            thread_group,
            false,
        );

        let module = callable_module(callable);
        if let Some(module) = module.as_ref() {
            vm.import_resolver
                .insert_module(module.name(), Arc::clone(module));
        }

        let caller = Arc::new(CodeObject::new("<thread bootstrap>", "<thread>"));
        vm.push_frame_with_module(caller, 0, module.clone())?;

        if keywords.is_empty() {
            invoke_callable_value(&mut vm, callable, &args)
        } else {
            let keyword_refs = keywords
                .iter()
                .map(|(name, value)| (name.as_str(), *value))
                .collect::<Vec<_>>();
            invoke_callable_value_with_keywords(&mut vm, callable, &args, &keyword_refs)
        }
    }));

    if let Ok(Err(err)) = result {
        eprintln!("Exception in Prism thread {ident}: {err}");
    } else if result.is_err() {
        eprintln!("Exception in Prism thread {ident}: native thread panicked");
    }
}

fn callable_module(callable: Value) -> Option<Arc<ModuleObject>> {
    let ptr = callable.as_object_ptr()?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    match header.type_id {
        TypeId::FUNCTION | TypeId::CLOSURE => function_module(ptr),
        TypeId::METHOD => {
            let method = unsafe { &*(ptr as *const BoundMethod) };
            callable_module(method.function())
        }
        TypeId::STATICMETHOD => {
            let descriptor = unsafe { &*(ptr as *const StaticMethodDescriptor) };
            callable_module(descriptor.function())
        }
        _ => None,
    }
}

fn function_module(ptr: *const ()) -> Option<Arc<ModuleObject>> {
    let func = unsafe { &*(ptr as *const FunctionObject) };
    let module_ptr = func.globals_ptr();
    if module_ptr.is_null() {
        return None;
    }

    let module = unsafe { Arc::from_raw(module_ptr as *const ModuleObject) };
    let cloned = Arc::clone(&module);
    std::mem::forget(module);
    Some(cloned)
}

#[inline]
fn thread_daemon_threads_allowed(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "_thread.daemon_threads_allowed() takes 0 positional arguments but {} were given",
            args.len()
        )));
    }
    Ok(Value::bool(true))
}

#[inline]
fn thread_stack_size(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "_thread.stack_size() takes at most 1 argument ({} given)",
            args.len()
        )));
    }

    let mut stack_size = STACK_SIZE_BYTES
        .lock()
        .expect("thread stack size lock poisoned");
    if let Some(value) = args.first().copied() {
        let requested = value
            .as_int()
            .ok_or_else(|| BuiltinError::TypeError("size must be an integer".to_string()))?;
        if requested < 0 {
            return Err(BuiltinError::ValueError(
                "size must be 0 or a positive value".to_string(),
            ));
        }
        *stack_size = requested as usize;
    }

    Ok(Value::int(*stack_size as i64).expect("stack size should fit in i64"))
}

#[inline]
fn thread_allocate_lock(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "_thread.allocate_lock() takes 0 positional arguments but {} were given",
            args.len()
        )));
    }
    new_lock_object()
}

#[inline]
fn thread_set_sentinel(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "_thread._set_sentinel() takes 0 positional arguments but {} were given",
            args.len()
        )));
    }
    let (value, lock) = new_lock_object_with_state()?;
    THREAD_SENTINELS.with(|sentinels| sentinels.borrow_mut().push(lock));
    Ok(value)
}

fn release_thread_sentinels() {
    THREAD_SENTINELS.with(|sentinels| {
        for lock in sentinels.borrow_mut().drain(..) {
            if native_lock_locked(&lock) {
                let _ = release_native_lock(&lock);
            }
        }
    });
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

fn new_lock_object() -> Result<Value, BuiltinError> {
    new_lock_object_with_state().map(|(value, _)| value)
}

fn new_lock_object_with_state() -> Result<(Value, Arc<NativeLock>), BuiltinError> {
    let registry = shape_registry();
    let mut object = Box::new(ShapedObject::with_empty_shape(registry.empty_shape()));
    let ptr = object.as_mut() as *mut ShapedObject;
    let value = Value::object_ptr(ptr as *const ());

    object.set_property(
        intern("acquire"),
        bound_builtin_attr_value(&LOCK_ACQUIRE_FUNCTION, value),
        registry,
    );
    object.set_property(
        intern("release"),
        bound_builtin_attr_value(&LOCK_RELEASE_FUNCTION, value),
        registry,
    );
    object.set_property(
        intern("locked"),
        bound_builtin_attr_value(&LOCK_LOCKED_FUNCTION, value),
        registry,
    );
    object.set_property(
        intern("__repr__"),
        bound_builtin_attr_value(&LOCK_REPR_FUNCTION, value),
        registry,
    );
    object.set_property(
        intern("__enter__"),
        bound_builtin_attr_value(&LOCK_ENTER_FUNCTION, value),
        registry,
    );
    object.set_property(
        intern("__exit__"),
        bound_builtin_attr_value(&LOCK_EXIT_FUNCTION, value),
        registry,
    );
    object.set_property(
        intern("_at_fork_reinit"),
        bound_builtin_attr_value(&LOCK_AT_FORK_REINIT_FUNCTION, value),
        registry,
    );

    let ptr = Box::into_raw(object);
    let lock = Arc::new(NativeLock::default());
    LOCKS_BY_PTR
        .lock()
        .expect("lock registry lock poisoned")
        .insert(ptr as usize, Arc::clone(&lock));
    Ok((value, lock))
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
        intern("locked"),
        bound_builtin_attr_value(&RLOCK_LOCKED_FUNCTION, value),
        registry,
    );
    object.set_property(
        intern("_is_owned"),
        bound_builtin_attr_value(&RLOCK_IS_OWNED_FUNCTION, value),
        registry,
    );
    object.set_property(
        intern("_recursion_count"),
        bound_builtin_attr_value(&RLOCK_RECURSION_COUNT_FUNCTION, value),
        registry,
    );
    object.set_property(
        intern("_release_save"),
        bound_builtin_attr_value(&RLOCK_RELEASE_SAVE_FUNCTION, value),
        registry,
    );
    object.set_property(
        intern("_acquire_restore"),
        bound_builtin_attr_value(&RLOCK_ACQUIRE_RESTORE_FUNCTION, value),
        registry,
    );
    object.set_property(
        intern("_at_fork_reinit"),
        bound_builtin_attr_value(&RLOCK_AT_FORK_REINIT_FUNCTION, value),
        registry,
    );
    object.set_property(
        intern("__repr__"),
        bound_builtin_attr_value(&RLOCK_REPR_FUNCTION, value),
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

fn parse_timeout_value(value: Value) -> Result<Option<Duration>, BuiltinError> {
    if let Some(timeout) = value.as_int() {
        if timeout == -1 {
            return Ok(None);
        }
        if timeout < 0 {
            return Err(BuiltinError::ValueError(
                "timeout value must be positive".to_string(),
            ));
        }
        if timeout > i32::MAX as i64 {
            return Err(BuiltinError::OverflowError(
                "timeout value is too large".to_string(),
            ));
        }
        return Ok(Some(Duration::from_secs(timeout as u64)));
    }
    if let Some(timeout) = value.as_float() {
        if timeout == -1.0 {
            return Ok(None);
        }
        if timeout.is_nan() || timeout < 0.0 {
            return Err(BuiltinError::ValueError(
                "timeout value must be positive".to_string(),
            ));
        }
        if !timeout.is_finite() || timeout > i32::MAX as f64 {
            return Err(BuiltinError::OverflowError(
                "timeout value is too large".to_string(),
            ));
        }
        return Ok(Some(Duration::from_secs_f64(timeout)));
    }
    Err(BuiltinError::TypeError(
        "timeout value must be an int or float".to_string(),
    ))
}

fn parse_acquire_call_args(
    method_name: &str,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<(Value, bool, Option<Duration>), BuiltinError> {
    if args.is_empty() || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "{method_name}() takes from 0 to 2 positional arguments but {} were given",
            args.len().saturating_sub(1)
        )));
    }

    let receiver = args[0];
    let mut blocking_arg = args.get(1).copied();
    let mut timeout_arg = args.get(2).copied();

    for (name, value) in keywords {
        match *name {
            "blocking" => {
                if blocking_arg.is_some() {
                    return Err(BuiltinError::TypeError(format!(
                        "{method_name}() got multiple values for argument 'blocking'"
                    )));
                }
                blocking_arg = Some(*value);
            }
            "timeout" => {
                if timeout_arg.is_some() {
                    return Err(BuiltinError::TypeError(format!(
                        "{method_name}() got multiple values for argument 'timeout'"
                    )));
                }
                timeout_arg = Some(*value);
            }
            other => {
                return Err(BuiltinError::TypeError(format!(
                    "{method_name}() got an unexpected keyword argument '{other}'"
                )));
            }
        }
    }

    let blocking = blocking_arg
        .map(parse_blocking_arg)
        .transpose()?
        .unwrap_or(true);
    let timeout = match timeout_arg {
        Some(value) => {
            let timeout = parse_timeout_value(value)?;
            if !blocking && timeout.is_some() {
                return Err(BuiltinError::ValueError(
                    "can't specify a timeout for a non-blocking call".to_string(),
                ));
            }
            timeout
        }
        None => None,
    };

    Ok((receiver, blocking, timeout))
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
        return Ok(crate::threading_runtime::blocking_operation(|| {
            loop {
                let remaining = match deadline.checked_duration_since(Instant::now()) {
                    Some(remaining) if !remaining.is_zero() => remaining,
                    _ => return false,
                };
                let (next_state, wait_result) = lock
                    .available
                    .wait_timeout(state, remaining)
                    .expect("RLock wait poisoned");
                state = next_state;
                if state.owner.is_none() {
                    state.owner = Some(current);
                    state.depth = 1;
                    return true;
                }
                if wait_result.timed_out() {
                    return false;
                }
            }
        }));
    }

    Ok(crate::threading_runtime::blocking_operation(|| {
        while state.owner.is_some() && state.owner != Some(current) {
            state = lock.available.wait(state).expect("RLock wait poisoned");
        }
        state.owner = Some(current);
        state.depth += 1;
        true
    }))
}

fn lock_for_value(receiver: Value) -> Result<Arc<NativeLock>, BuiltinError> {
    let ptr = receiver.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("lock methods require a lock instance".to_string())
    })?;
    LOCKS_BY_PTR
        .lock()
        .expect("lock registry lock poisoned")
        .get(&(ptr as usize))
        .cloned()
        .ok_or_else(|| BuiltinError::TypeError("lock methods require a lock instance".to_string()))
}

fn acquire_native_lock(
    lock: &NativeLock,
    blocking: bool,
    timeout: Option<Duration>,
) -> Result<bool, BuiltinError> {
    let mut state = lock.state.lock().expect("lock state lock poisoned");
    if !state.locked {
        state.locked = true;
        return Ok(true);
    }

    if !blocking {
        return Ok(false);
    }

    if let Some(timeout) = timeout {
        let deadline = Instant::now() + timeout;
        return Ok(crate::threading_runtime::blocking_operation(|| {
            loop {
                let remaining = match deadline.checked_duration_since(Instant::now()) {
                    Some(remaining) if !remaining.is_zero() => remaining,
                    _ => return false,
                };
                let (next_state, wait_result) = lock
                    .available
                    .wait_timeout(state, remaining)
                    .expect("lock wait poisoned");
                state = next_state;
                if !state.locked {
                    state.locked = true;
                    return true;
                }
                if wait_result.timed_out() {
                    return false;
                }
            }
        }));
    }

    Ok(crate::threading_runtime::blocking_operation(|| {
        while state.locked {
            state = lock.available.wait(state).expect("lock wait poisoned");
        }
        state.locked = true;
        true
    }))
}

fn release_native_lock(lock: &NativeLock) -> Result<(), BuiltinError> {
    let mut state = lock.state.lock().expect("lock state lock poisoned");
    if !state.locked {
        return Err(runtime_error_builtin("release unlocked lock"));
    }
    state.locked = false;
    lock.available.notify_one();
    Ok(())
}

fn native_lock_locked(lock: &NativeLock) -> bool {
    lock.state.lock().expect("lock state lock poisoned").locked
}

fn reinit_native_lock(lock: &NativeLock) {
    let mut state = lock.state.lock().expect("lock state lock poisoned");
    state.locked = false;
    lock.available.notify_all();
}

fn release_native_rlock(lock: &NativeRLock) -> Result<(), BuiltinError> {
    let current = current_thread_ident();
    let mut state = lock.state.lock().expect("RLock state lock poisoned");
    if state.owner != Some(current) || state.depth == 0 {
        return Err(runtime_error_builtin("cannot release un-acquired lock"));
    }

    state.depth -= 1;
    if state.depth == 0 {
        state.owner = None;
        lock.available.notify_one();
    }
    Ok(())
}

fn native_rlock_locked(lock: &NativeRLock) -> bool {
    lock.state
        .lock()
        .expect("RLock state lock poisoned")
        .owner
        .is_some()
}

fn native_rlock_is_owned(lock: &NativeRLock) -> bool {
    let current = current_thread_ident();
    let state = lock.state.lock().expect("RLock state lock poisoned");
    state.depth > 0 && state.owner == Some(current)
}

fn native_rlock_recursion_count(lock: &NativeRLock) -> usize {
    let current = current_thread_ident();
    let state = lock.state.lock().expect("RLock state lock poisoned");
    if state.owner == Some(current) {
        state.depth
    } else {
        0
    }
}

fn release_save_native_rlock(lock: &NativeRLock) -> Result<(usize, u64), BuiltinError> {
    let mut state = lock.state.lock().expect("RLock state lock poisoned");
    let Some(owner) = state.owner else {
        return Err(runtime_error_builtin("cannot release un-acquired lock"));
    };
    if state.depth == 0 {
        return Err(runtime_error_builtin("cannot release un-acquired lock"));
    }

    let depth = state.depth;
    state.owner = None;
    state.depth = 0;
    lock.available.notify_one();
    Ok((depth, owner))
}

fn acquire_restore_native_rlock(
    lock: &NativeRLock,
    owner: u64,
    depth: usize,
) -> Result<(), BuiltinError> {
    let mut state = lock.state.lock().expect("RLock state lock poisoned");
    crate::threading_runtime::blocking_operation(|| {
        while state.owner.is_some() {
            state = lock.available.wait(state).expect("RLock wait poisoned");
        }
        state.owner = Some(owner);
        state.depth = depth;
    });
    Ok(())
}

fn reinit_native_rlock(lock: &NativeRLock) {
    let mut state = lock.state.lock().expect("RLock state lock poisoned");
    state.owner = None;
    state.depth = 0;
    lock.available.notify_all();
}

fn lock_repr_text(ptr: usize, lock: &NativeLock) -> String {
    let state = if native_lock_locked(lock) {
        "locked"
    } else {
        "unlocked"
    };
    format!("<{state} _thread.lock object at 0x{ptr:x}>")
}

fn rlock_repr_text(ptr: usize, lock: &NativeRLock) -> String {
    let state = lock.state.lock().expect("RLock state lock poisoned");
    let status = if state.depth > 0 {
        "locked"
    } else {
        "unlocked"
    };
    let owner = state.owner.unwrap_or(0);
    let count = state.depth;
    format!("<{status} _thread.RLock object owner={owner} count={count} at 0x{ptr:x}>")
}

pub(crate) fn native_thread_object_repr(value: Value) -> Option<String> {
    let ptr = value.as_object_ptr()? as usize;
    if let Some(lock) = LOCKS_BY_PTR
        .lock()
        .expect("lock registry lock poisoned")
        .get(&ptr)
        .cloned()
    {
        return Some(lock_repr_text(ptr, &lock));
    }
    RLOCKS_BY_PTR
        .lock()
        .expect("RLock registry lock poisoned")
        .get(&ptr)
        .cloned()
        .map(|lock| rlock_repr_text(ptr, &lock))
}

pub(crate) fn is_native_thread_object(value: Value) -> bool {
    let Some(ptr) = value.as_object_ptr().map(|ptr| ptr as usize) else {
        return false;
    };

    LOCKS_BY_PTR
        .lock()
        .expect("lock registry lock poisoned")
        .contains_key(&ptr)
        || RLOCKS_BY_PTR
            .lock()
            .expect("RLock registry lock poisoned")
            .contains_key(&ptr)
}

fn lock_acquire(args: &[Value]) -> Result<Value, BuiltinError> {
    lock_acquire_kw(args, &[])
}

fn lock_acquire_kw(args: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    let (receiver, blocking, timeout) = parse_acquire_call_args("lock.acquire", args, keywords)?;
    let lock = lock_for_value(receiver)?;
    Ok(Value::bool(acquire_native_lock(&lock, blocking, timeout)?))
}

fn lock_release(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "lock.release() takes 0 positional arguments but {} were given",
            args.len().saturating_sub(1)
        )));
    }

    let lock = lock_for_value(args[0])?;
    release_native_lock(&lock)?;
    Ok(Value::none())
}

fn lock_locked(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "lock.locked() takes 0 positional arguments but {} were given",
            args.len().saturating_sub(1)
        )));
    }

    let lock = lock_for_value(args[0])?;
    Ok(Value::bool(native_lock_locked(&lock)))
}

fn lock_repr(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "lock.__repr__() takes 0 positional arguments but {} were given",
            args.len().saturating_sub(1)
        )));
    }

    let ptr = args[0].as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("lock methods require a lock instance".to_string())
    })? as usize;
    let lock = lock_for_value(args[0])?;
    Ok(Value::string(intern(&lock_repr_text(ptr, &lock))))
}

fn lock_enter(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "lock.__enter__() takes 0 positional arguments but {} were given",
            args.len().saturating_sub(1)
        )));
    }

    let receiver = args[0];
    let lock = lock_for_value(receiver)?;
    Ok(Value::bool(acquire_native_lock(&lock, true, None)?))
}

fn lock_exit(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 4 {
        return Err(BuiltinError::TypeError(format!(
            "lock.__exit__() takes 3 positional arguments but {} were given",
            args.len().saturating_sub(1)
        )));
    }

    let lock = lock_for_value(args[0])?;
    release_native_lock(&lock)?;
    Ok(Value::bool(false))
}

fn lock_at_fork_reinit(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "lock._at_fork_reinit() takes 0 positional arguments but {} were given",
            args.len().saturating_sub(1)
        )));
    }

    let lock = lock_for_value(args[0])?;
    reinit_native_lock(&lock);
    Ok(Value::none())
}

fn rlock_acquire(args: &[Value]) -> Result<Value, BuiltinError> {
    rlock_acquire_kw(args, &[])
}

fn rlock_acquire_kw(args: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    let (receiver, blocking, timeout) = parse_acquire_call_args("RLock.acquire", args, keywords)?;
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

fn rlock_locked(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "RLock.locked() takes 0 positional arguments but {} were given",
            args.len().saturating_sub(1)
        )));
    }

    let lock = rlock_for_value(args[0])?;
    Ok(Value::bool(native_rlock_locked(&lock)))
}

fn rlock_is_owned(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "RLock._is_owned() takes 0 positional arguments but {} were given",
            args.len().saturating_sub(1)
        )));
    }

    let lock = rlock_for_value(args[0])?;
    Ok(Value::bool(native_rlock_is_owned(&lock)))
}

fn rlock_recursion_count(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "RLock._recursion_count() takes 0 positional arguments but {} were given",
            args.len().saturating_sub(1)
        )));
    }

    let lock = rlock_for_value(args[0])?;
    let count = native_rlock_recursion_count(&lock);
    let count = i64::try_from(count)
        .map_err(|_| BuiltinError::OverflowError("recursion count is too large".to_string()))?;
    Value::int(count)
        .ok_or_else(|| BuiltinError::OverflowError("recursion count is too large".to_string()))
}

fn rlock_release_save(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "RLock._release_save() takes 0 positional arguments but {} were given",
            args.len().saturating_sub(1)
        )));
    }

    let lock = rlock_for_value(args[0])?;
    let (depth, owner) = release_save_native_rlock(&lock)?;
    let depth = i64::try_from(depth)
        .map_err(|_| BuiltinError::OverflowError("recursion count is too large".to_string()))?;
    let owner = i64::try_from(owner)
        .map_err(|_| BuiltinError::OverflowError("thread identifier is too large".to_string()))?;
    let depth = Value::int(depth)
        .ok_or_else(|| BuiltinError::OverflowError("recursion count is too large".to_string()))?;
    let owner = Value::int(owner)
        .ok_or_else(|| BuiltinError::OverflowError("thread identifier is too large".to_string()))?;
    let tuple = TupleObject::from_slice(&[depth, owner]);
    Ok(crate::alloc_managed_value(tuple))
}

fn parse_rlock_restore_state(value: Value) -> Result<(usize, u64), BuiltinError> {
    let tuple = value_as_tuple_ref(value)
        .ok_or_else(|| BuiltinError::TypeError("RLock state must be a 2-tuple".to_string()))?;
    if tuple.len() != 2 {
        return Err(BuiltinError::TypeError(
            "RLock state must be a 2-tuple".to_string(),
        ));
    }

    let count = tuple
        .get(0)
        .and_then(|value| value.as_int())
        .ok_or_else(|| BuiltinError::TypeError("RLock count must be an int".to_string()))?;
    let owner = tuple
        .get(1)
        .and_then(|value| value.as_int())
        .ok_or_else(|| BuiltinError::TypeError("RLock owner must be an int".to_string()))?;

    if count <= 0 {
        return Err(BuiltinError::ValueError(
            "RLock count must be positive".to_string(),
        ));
    }
    if owner <= 0 {
        return Err(BuiltinError::ValueError(
            "RLock owner must be positive".to_string(),
        ));
    }

    let count = usize::try_from(count)
        .map_err(|_| BuiltinError::OverflowError("RLock count is too large".to_string()))?;
    let owner = u64::try_from(owner)
        .map_err(|_| BuiltinError::OverflowError("RLock owner is too large".to_string()))?;
    Ok((count, owner))
}

fn rlock_acquire_restore(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "RLock._acquire_restore() takes 1 positional argument but {} were given",
            args.len().saturating_sub(1)
        )));
    }

    let lock = rlock_for_value(args[0])?;
    let (depth, owner) = parse_rlock_restore_state(args[1])?;
    acquire_restore_native_rlock(&lock, owner, depth)?;
    Ok(Value::none())
}

fn rlock_at_fork_reinit(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "RLock._at_fork_reinit() takes 0 positional arguments but {} were given",
            args.len().saturating_sub(1)
        )));
    }

    let lock = rlock_for_value(args[0])?;
    reinit_native_rlock(&lock);
    Ok(Value::none())
}

fn rlock_repr(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "RLock.__repr__() takes 0 positional arguments but {} were given",
            args.len().saturating_sub(1)
        )));
    }

    let ptr = args[0].as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("RLock methods require an RLock instance".to_string())
    })? as usize;
    let lock = rlock_for_value(args[0])?;
    Ok(Value::string(intern(&rlock_repr_text(ptr, &lock))))
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
    Ok(Value::bool(acquire_native_rlock(&lock, true, None)?))
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
mod tests;
