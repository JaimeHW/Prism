use super::*;
use prism_runtime::object::shaped_object::ShapedObject;

static THREAD_COUNT_TEST_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

#[test]
fn test_thread_module_exposes_expected_attrs() {
    let module = ThreadModule::new();

    assert!(module.get_attr("RLock").is_ok());
    assert!(module.get_attr("TIMEOUT_MAX").is_ok());
    assert!(module.get_attr("_count").is_ok());
    assert!(module.get_attr("_is_main_interpreter").is_ok());
    assert!(module.get_attr("_local").is_ok());
    assert!(module.get_attr("_set_sentinel").is_ok());
    assert!(module.get_attr("allocate_lock").is_ok());
    assert!(module.get_attr("daemon_threads_allowed").is_ok());
    assert!(module.get_attr("get_ident").is_ok());
    assert!(module.get_attr("get_native_id").is_ok());
    assert!(module.get_attr("interrupt_main").is_ok());
    assert!(module.get_attr("start_new_thread").is_ok());
    assert!(module.get_attr("stack_size").is_ok());
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
fn test_pending_async_exception_round_trips_for_active_thread() {
    let ident = thread_get_ident(&[])
        .expect("get_ident should succeed")
        .as_int()
        .expect("get_ident should return an int") as u64;
    let exception = Value::string(intern("AsyncExc"));

    assert!(set_pending_async_exception_for_ident(ident, exception));
    assert_eq!(
        take_pending_async_exception_for_current_thread(),
        Some(exception)
    );
    assert_eq!(take_pending_async_exception_for_current_thread(), None);
}

#[test]
fn test_pending_async_exception_rejects_unknown_thread() {
    assert!(!set_pending_async_exception_for_ident(
        u64::MAX,
        Value::string(intern("AsyncExc"))
    ));
}

#[test]
fn test_count_reports_active_thread_baseline() {
    let baseline = thread_count(&[])
        .expect("_count should succeed")
        .as_int()
        .expect("_count should return an int");

    assert!(baseline >= 1);
}

#[test]
fn test_is_main_interpreter_reports_true_for_single_runtime() {
    assert_eq!(
        thread_is_main_interpreter(&[]).expect("_is_main_interpreter should succeed"),
        Value::bool(true)
    );

    let err = thread_is_main_interpreter(&[Value::none()])
        .expect_err("_is_main_interpreter should validate arity");
    assert!(matches!(err, BuiltinError::TypeError(_)));
}

#[test]
fn test_thread_local_type_returns_attribute_capable_object() {
    let local_type = local_type_value();
    let class_ptr = local_type
        .as_object_ptr()
        .expect("_local should be exported as a class");
    let class = unsafe { &*(class_ptr as *const PyClassObject) };
    assert_eq!(
        class.get_attr(&intern("__module__")),
        Some(Value::string(intern("_thread")))
    );

    let mut vm = VirtualMachine::new();
    let local = invoke_callable_value(&mut vm, local_type, &[])
        .expect("_local should construct a local namespace");
    let ptr = local
        .as_object_ptr()
        .expect("_local should return an object");
    let object = unsafe { &mut *(ptr as *mut ShapedObject) };
    let registry = shape_registry();

    object.set_property(
        intern("marker"),
        Value::int(7).expect("marker should fit"),
        registry,
    );
    assert_eq!(
        object
            .get_property("marker")
            .and_then(|value| value.as_int()),
        Some(7)
    );

    let err = invoke_callable_value(&mut vm, local_type, &[Value::none()])
        .expect_err("_local should validate arity");
    assert!(matches!(
        err.kind,
        crate::error::RuntimeErrorKind::TypeError { .. }
    ));
}

#[test]
fn test_count_validates_arity() {
    let err = thread_count(&[Value::none()]).expect_err("_count should reject arguments");

    match err {
        BuiltinError::TypeError(message) => {
            assert_eq!(
                message,
                "_thread._count() takes 0 positional arguments but 1 were given"
            );
        }
        other => panic!("unexpected error type: {other:?}"),
    }
}

#[test]
fn test_count_guard_tracks_native_thread_lifetime() {
    let _test_guard = THREAD_COUNT_TEST_LOCK
        .lock()
        .expect("thread count test lock should not be poisoned");
    let baseline = ACTIVE_THREAD_COUNT.load(Ordering::SeqCst);

    ACTIVE_THREAD_COUNT.fetch_add(1, Ordering::SeqCst);
    assert_eq!(
        thread_count(&[]).unwrap().as_int().unwrap(),
        (baseline + 1) as i64
    );
    ACTIVE_THREAD_COUNT.fetch_sub(1, Ordering::SeqCst);

    assert_eq!(
        thread_count(&[]).unwrap().as_int().unwrap(),
        baseline as i64
    );
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
fn test_get_native_id_matches_get_ident() {
    assert_eq!(
        thread_get_native_id(&[]).unwrap(),
        thread_get_ident(&[]).unwrap()
    );
}

#[test]
fn test_interrupt_main_validates_signal_range() {
    let mut vm = VirtualMachine::new();
    let err = thread_interrupt_main(&mut vm, &[Value::int(-1).unwrap()])
        .expect_err("negative signal should be rejected");

    assert!(matches!(err, BuiltinError::ValueError(_)));
}

#[test]
fn test_stack_size_defaults_to_zero_and_updates() {
    assert_eq!(thread_stack_size(&[]).unwrap().as_int(), Some(0));
    assert_eq!(
        thread_stack_size(&[Value::int(64 * 1024).unwrap()])
            .unwrap()
            .as_int(),
        Some(64 * 1024)
    );
    assert_eq!(thread_stack_size(&[]).unwrap().as_int(), Some(64 * 1024));
    assert_eq!(
        thread_stack_size(&[Value::int(0).unwrap()])
            .unwrap()
            .as_int(),
        Some(0)
    );
}

#[test]
fn test_allocate_lock_installs_expected_methods() {
    let lock = thread_allocate_lock(&[]).expect("allocate_lock() should succeed");
    let ptr = lock
        .as_object_ptr()
        .expect("allocate_lock() should return an object");
    let object = unsafe { &*(ptr as *const ShapedObject) };

    for name in [
        "acquire",
        "release",
        "locked",
        "__repr__",
        "__enter__",
        "__exit__",
        "_at_fork_reinit",
    ] {
        assert!(
            object.get_property(name).is_some(),
            "{name} should be installed"
        );
    }
}

#[test]
fn test_allocate_lock_supports_acquire_release_and_reinit() {
    let lock = thread_allocate_lock(&[]).expect("allocate_lock() should succeed");
    assert!(!lock_locked(&[lock]).unwrap().as_bool().unwrap());
    assert!(lock_acquire(&[lock]).unwrap().as_bool().unwrap());
    assert!(lock_locked(&[lock]).unwrap().as_bool().unwrap());
    assert!(lock_release(&[lock]).unwrap().is_none());
    assert!(!lock_locked(&[lock]).unwrap().as_bool().unwrap());

    assert!(lock_acquire(&[lock]).unwrap().as_bool().unwrap());
    assert!(lock_at_fork_reinit(&[lock]).unwrap().is_none());
    assert!(!lock_locked(&[lock]).unwrap().as_bool().unwrap());
}

#[test]
fn test_lock_acquire_accepts_keywords_and_validates_timeout() {
    let lock = thread_allocate_lock(&[]).expect("allocate_lock() should succeed");

    let err = lock_acquire_kw(&[lock, Value::bool(false), Value::int(1).unwrap()], &[])
        .expect_err("non-blocking acquire with a timeout should fail");
    assert!(matches!(err, BuiltinError::ValueError(_)));

    let err = lock_acquire_kw(&[lock], &[("timeout", Value::int(-100).unwrap())])
        .expect_err("negative timeout other than -1 should fail");
    assert!(matches!(err, BuiltinError::ValueError(_)));

    let err = lock_acquire_kw(&[lock], &[("timeout", Value::float(1e100))])
        .expect_err("huge timeout should fail");
    assert!(matches!(err, BuiltinError::OverflowError(_)));

    assert!(
        lock_acquire_kw(&[lock], &[("timeout", Value::float(i32::MAX as f64))])
            .unwrap()
            .as_bool()
            .unwrap()
    );
    assert!(lock_release(&[lock]).unwrap().is_none());
}

#[test]
fn test_lock_and_rlock_repr_describe_native_state() {
    let lock = thread_allocate_lock(&[]).expect("allocate_lock() should succeed");
    let text = lock_repr(&[lock]).expect("__repr__ should succeed");
    let text_object = value_as_string_ref(text).unwrap();
    let text = text_object.as_str();
    assert!(text.starts_with("<unlocked _thread.lock object at 0x"));
    assert_eq!(native_thread_object_repr(lock).as_deref(), Some(text));

    assert!(lock_acquire(&[lock]).unwrap().as_bool().unwrap());
    let text = native_thread_object_repr(lock).expect("native repr should exist");
    assert!(text.starts_with("<locked _thread.lock object at 0x"));
    assert!(lock_release(&[lock]).unwrap().is_none());

    let rlock = thread_rlock(&[]).expect("RLock() should succeed");
    let text = rlock_repr(&[rlock]).expect("__repr__ should succeed");
    let text_object = value_as_string_ref(text).unwrap();
    let text = text_object.as_str();
    assert!(text.starts_with("<unlocked _thread.RLock object owner=0 count=0 at 0x"));

    assert!(rlock_acquire(&[rlock]).unwrap().as_bool().unwrap());
    let text = native_thread_object_repr(rlock).expect("native repr should exist");
    assert!(text.starts_with("<locked _thread.RLock object owner="));
    assert!(text.contains(" count=1 at 0x"));
    assert!(rlock_release(&[rlock]).unwrap().is_none());
}

#[test]
fn test_set_sentinel_returns_lock_like_object() {
    let sentinel = thread_set_sentinel(&[]).expect("_set_sentinel() should succeed");
    assert!(lock_acquire(&[sentinel]).unwrap().as_bool().unwrap());
    assert!(lock_locked(&[sentinel]).unwrap().as_bool().unwrap());
}

#[test]
fn test_daemon_threads_allowed_is_true() {
    assert_eq!(
        thread_daemon_threads_allowed(&[]).unwrap(),
        Value::bool(true)
    );
}

#[test]
fn test_start_new_thread_returns_thread_identifier() {
    let _test_guard = THREAD_COUNT_TEST_LOCK
        .lock()
        .expect("thread count test lock should not be poisoned");
    let baseline = ACTIVE_THREAD_COUNT.load(Ordering::SeqCst);
    let mut vm = VirtualMachine::new();
    assert_eq!(vm.thread_group().handle_count(), 0);
    let args = prism_runtime::types::tuple::TupleObject::from_slice(&[]);
    let args_ptr = Box::into_raw(Box::new(args));
    let token = thread_start_new_thread(
        &mut vm,
        &[
            builtin_value(&GET_IDENT_FUNCTION),
            Value::object_ptr(args_ptr as *const ()),
        ],
    )
    .expect("start_new_thread should succeed")
    .as_int()
    .expect("start_new_thread should return an int");
    assert!(token > 0);

    assert!(
        vm.join_owned_threads(Duration::from_secs(2)),
        "worker thread should finish and join"
    );
    assert!(
        wait_for_active_thread_count_at_most(baseline, Duration::from_secs(2)),
        "worker thread count should return to baseline"
    );
    assert_eq!(ACTIVE_THREAD_COUNT.load(Ordering::SeqCst), baseline);
    assert_eq!(vm.thread_group().handle_count(), 0);

    unsafe {
        drop(Box::from_raw(args_ptr));
    }
}

#[test]
fn test_error_alias_points_to_runtime_error_type() {
    let module = ThreadModule::new();
    let error = module.get_attr("error").expect("error alias should exist");
    let expected = runtime_error_type_value();

    assert_eq!(error.as_object_ptr(), expected.as_object_ptr());
}

#[test]
fn test_rlock_factory_installs_cpython_methods() {
    let lock = thread_rlock(&[]).expect("RLock() should succeed");
    let ptr = lock
        .as_object_ptr()
        .expect("RLock() should return an object");
    let object = unsafe { &*(ptr as *const ShapedObject) };

    for name in [
        "acquire",
        "release",
        "locked",
        "_is_owned",
        "_recursion_count",
        "_release_save",
        "_acquire_restore",
        "_at_fork_reinit",
        "__repr__",
        "__enter__",
        "__exit__",
    ] {
        assert!(
            object.get_property(name).is_some(),
            "{name} should be installed"
        );
    }
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

    assert!(rlock_enter(&[lock]).unwrap().as_bool().unwrap());
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

#[test]
fn test_rlock_is_owned_and_recursion_count_track_current_owner() {
    let lock = thread_rlock(&[]).expect("RLock() should succeed");
    assert!(!rlock_is_owned(&[lock]).unwrap().as_bool().unwrap());
    assert_eq!(rlock_recursion_count(&[lock]).unwrap().as_int(), Some(0));

    assert!(rlock_acquire(&[lock]).unwrap().as_bool().unwrap());
    assert!(rlock_acquire(&[lock]).unwrap().as_bool().unwrap());
    assert!(rlock_is_owned(&[lock]).unwrap().as_bool().unwrap());
    assert_eq!(rlock_recursion_count(&[lock]).unwrap().as_int(), Some(2));

    let worker = std::thread::spawn(move || {
        let owned = rlock_is_owned(&[lock])
            .expect("_is_owned should succeed")
            .as_bool()
            .expect("_is_owned should return bool");
        let count = rlock_recursion_count(&[lock])
            .expect("_recursion_count should succeed")
            .as_int()
            .expect("_recursion_count should return int");
        (owned, count)
    })
    .join()
    .expect("worker thread should join");

    assert_eq!(worker, (false, 0));
    assert!(rlock_release(&[lock]).unwrap().is_none());
    assert!(rlock_release(&[lock]).unwrap().is_none());
    assert!(!rlock_is_owned(&[lock]).unwrap().as_bool().unwrap());
}

#[test]
fn test_rlock_release_save_and_acquire_restore_round_trip() {
    let lock = thread_rlock(&[]).expect("RLock() should succeed");
    assert!(rlock_acquire(&[lock]).unwrap().as_bool().unwrap());
    assert!(rlock_acquire(&[lock]).unwrap().as_bool().unwrap());

    let state = rlock_release_save(&[lock]).expect("_release_save should succeed");
    assert!(!rlock_locked(&[lock]).unwrap().as_bool().unwrap());
    assert!(!rlock_is_owned(&[lock]).unwrap().as_bool().unwrap());
    assert_eq!(rlock_recursion_count(&[lock]).unwrap().as_int(), Some(0));

    let tuple = value_as_tuple_ref(state).expect("_release_save should return a tuple");
    assert_eq!(tuple.len(), 2);
    assert_eq!(tuple.get(0).unwrap().as_int(), Some(2));
    assert_eq!(
        tuple.get(1).unwrap().as_int(),
        Some(current_thread_ident() as i64)
    );

    assert!(rlock_acquire_restore(&[lock, state]).unwrap().is_none());
    assert!(rlock_is_owned(&[lock]).unwrap().as_bool().unwrap());
    assert_eq!(rlock_recursion_count(&[lock]).unwrap().as_int(), Some(2));
    assert!(rlock_release(&[lock]).unwrap().is_none());
    assert!(rlock_release(&[lock]).unwrap().is_none());
}

#[test]
fn test_rlock_private_hooks_validate_unowned_and_reinit_state() {
    let lock = thread_rlock(&[]).expect("RLock() should succeed");
    let err = rlock_release(&[lock]).expect_err("unowned release should fail");
    assert!(matches!(err, BuiltinError::Raised(_)));
    assert!(err.to_string().contains("cannot release un-acquired lock"));

    let err = rlock_release_save(&[lock]).expect_err("unowned _release_save should fail");
    assert!(matches!(err, BuiltinError::Raised(_)));
    assert!(err.to_string().contains("cannot release un-acquired lock"));

    assert!(rlock_acquire(&[lock]).unwrap().as_bool().unwrap());
    assert!(rlock_at_fork_reinit(&[lock]).unwrap().is_none());
    assert!(!rlock_locked(&[lock]).unwrap().as_bool().unwrap());
    assert!(!rlock_is_owned(&[lock]).unwrap().as_bool().unwrap());
    assert_eq!(rlock_recursion_count(&[lock]).unwrap().as_int(), Some(0));
}

#[test]
fn test_rlock_acquire_accepts_keywords_and_validates_timeout() {
    let lock = thread_rlock(&[]).expect("RLock() should succeed");

    let err = rlock_acquire_kw(&[lock, Value::bool(false), Value::int(1).unwrap()], &[])
        .expect_err("non-blocking acquire with a timeout should fail");
    assert!(matches!(err, BuiltinError::ValueError(_)));

    let err = rlock_acquire_kw(&[lock], &[("timeout", Value::int(-2).unwrap())])
        .expect_err("negative timeout other than -1 should fail");
    assert!(matches!(err, BuiltinError::ValueError(_)));

    let err = rlock_acquire_kw(&[lock], &[("timeout", Value::float(1e100))])
        .expect_err("huge timeout should fail");
    assert!(matches!(err, BuiltinError::OverflowError(_)));

    assert!(
        rlock_acquire_kw(&[lock], &[("blocking", Value::bool(true))])
            .unwrap()
            .as_bool()
            .unwrap()
    );
    assert!(rlock_release(&[lock]).unwrap().is_none());
}
