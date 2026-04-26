use super::*;
use prism_core::intern::intern;

// =========================================================================
// CallError Tests
// =========================================================================

#[test]
fn test_call_error_display() {
    let err = CallError::ClassNotFound {
        class_id: ClassId(123),
    };
    assert!(err.to_string().contains("123"));

    let err = CallError::NewReturnedWrongType {
        expected: "MyClass".to_string(),
        actual: "OtherClass".to_string(),
    };
    assert!(err.to_string().contains("MyClass"));
    assert!(err.to_string().contains("OtherClass"));

    let err = CallError::InitReturnedNonNone;
    assert!(err.to_string().contains("None"));

    let err = CallError::ArgumentError {
        message: "too many args".to_string(),
    };
    assert!(err.to_string().contains("too many args"));

    let err = CallError::NotCallable {
        type_name: "int".to_string(),
    };
    assert!(err.to_string().contains("int"));
}

// =========================================================================
// MethodSlot Tests
// =========================================================================

#[test]
fn test_method_slot_new() {
    let slot = MethodSlot::new(ClassId(42), 5);
    assert_eq!(slot.class_id, ClassId(42));
    assert_eq!(slot.method_name, 5);
}

// =========================================================================
// CallSpecialization Tests
// =========================================================================

#[test]
fn test_specialization_is_fast_path() {
    assert!(CallSpecialization::DefaultBoth.is_fast_path());
    assert!(CallSpecialization::Singleton(Value::int_unchecked(0)).is_fast_path());
    assert!(
        !CallSpecialization::DefaultNew {
            init_class: ClassId(1)
        }
        .is_fast_path()
    );
    assert!(
        !CallSpecialization::CustomNew {
            new_class: ClassId(1)
        }
        .is_fast_path()
    );
    assert!(
        !CallSpecialization::CustomBoth {
            new_class: ClassId(1),
            init_class: ClassId(2),
        }
        .is_fast_path()
    );
}

#[test]
fn test_specialization_needs_new() {
    assert!(!CallSpecialization::DefaultBoth.needs_new());
    assert!(
        !CallSpecialization::DefaultNew {
            init_class: ClassId(1)
        }
        .needs_new()
    );
    assert!(
        CallSpecialization::CustomNew {
            new_class: ClassId(1)
        }
        .needs_new()
    );
    assert!(
        CallSpecialization::CustomBoth {
            new_class: ClassId(1),
            init_class: ClassId(2),
        }
        .needs_new()
    );
}

#[test]
fn test_specialization_needs_init() {
    assert!(!CallSpecialization::DefaultBoth.needs_init());
    assert!(
        CallSpecialization::DefaultNew {
            init_class: ClassId(1)
        }
        .needs_init()
    );
    assert!(
        !CallSpecialization::CustomNew {
            new_class: ClassId(1)
        }
        .needs_init()
    );
    assert!(
        CallSpecialization::CustomBoth {
            new_class: ClassId(1),
            init_class: ClassId(2),
        }
        .needs_init()
    );
}

#[test]
fn test_specialization_from_class_default_both() {
    use super::super::class::PyClassObject;
    let class = PyClassObject::new_simple(intern("Simple"));
    let spec = CallSpecialization::from_class(&class);
    assert_eq!(spec, CallSpecialization::DefaultBoth);
}

#[test]
fn test_specialization_from_class_with_init() {
    use super::super::class::PyClassObject;
    let mut class = PyClassObject::new_simple(intern("WithInit"));
    class.mark_has_init();
    let spec = CallSpecialization::from_class(&class);
    match spec {
        CallSpecialization::DefaultNew { init_class } => {
            assert_eq!(init_class, class.class_id());
        }
        _ => panic!("Expected DefaultNew, got {:?}", spec),
    }
}

#[test]
fn test_specialization_from_class_with_new() {
    use super::super::class::PyClassObject;
    let mut class = PyClassObject::new_simple(intern("WithNew"));
    class.mark_has_new();
    let spec = CallSpecialization::from_class(&class);
    match spec {
        CallSpecialization::CustomNew { new_class } => {
            assert_eq!(new_class, class.class_id());
        }
        _ => panic!("Expected CustomNew, got {:?}", spec),
    }
}

#[test]
fn test_specialization_from_class_with_both() {
    use super::super::class::PyClassObject;
    let mut class = PyClassObject::new_simple(intern("WithBoth"));
    class.mark_has_new();
    class.mark_has_init();
    let spec = CallSpecialization::from_class(&class);
    match spec {
        CallSpecialization::CustomBoth {
            new_class,
            init_class,
        } => {
            assert_eq!(new_class, class.class_id());
            assert_eq!(init_class, class.class_id());
        }
        _ => panic!("Expected CustomBoth, got {:?}", spec),
    }
}

// =========================================================================
// CallDispatcher Tests
// =========================================================================

#[test]
fn test_dispatcher_new() {
    let dispatcher = CallDispatcher::new();
    assert!(dispatcher.is_empty());
    assert_eq!(dispatcher.len(), 0);
}

#[test]
fn test_dispatcher_register_and_get() {
    let dispatcher = CallDispatcher::new();

    dispatcher.register(ClassId(100), CallSpecialization::DefaultBoth);
    dispatcher.register(
        ClassId(101),
        CallSpecialization::DefaultNew {
            init_class: ClassId(101),
        },
    );

    assert_eq!(
        dispatcher.get_specialization(ClassId(100)),
        Some(CallSpecialization::DefaultBoth)
    );
    assert_eq!(
        dispatcher.get_specialization(ClassId(101)),
        Some(CallSpecialization::DefaultNew {
            init_class: ClassId(101)
        })
    );
    assert_eq!(dispatcher.get_specialization(ClassId(999)), None);
    assert_eq!(dispatcher.len(), 2);
}

#[test]
fn test_dispatcher_analyze() {
    use super::super::class::PyClassObject;

    let dispatcher = CallDispatcher::new();
    let class = PyClassObject::new_simple(intern("AnalyzeTest"));

    let spec = dispatcher.analyze(&class);
    assert_eq!(spec, CallSpecialization::DefaultBoth);

    // Should be cached now
    assert_eq!(
        dispatcher.get_specialization(class.class_id()),
        Some(CallSpecialization::DefaultBoth)
    );
    assert_eq!(dispatcher.len(), 1);
}

#[test]
fn test_dispatcher_invalidate() {
    let dispatcher = CallDispatcher::new();

    dispatcher.register(ClassId(100), CallSpecialization::DefaultBoth);
    assert!(dispatcher.get_specialization(ClassId(100)).is_some());

    dispatcher.invalidate(ClassId(100));
    assert!(dispatcher.get_specialization(ClassId(100)).is_none());
}

#[test]
fn test_dispatcher_clear() {
    let dispatcher = CallDispatcher::new();

    dispatcher.register(ClassId(100), CallSpecialization::DefaultBoth);
    dispatcher.register(ClassId(101), CallSpecialization::DefaultBoth);
    assert_eq!(dispatcher.len(), 2);

    dispatcher.clear();
    assert!(dispatcher.is_empty());
}

// =========================================================================
// CallContext Tests
// =========================================================================

#[test]
fn test_call_context_new() {
    let args = vec![Value::int_unchecked(1), Value::int_unchecked(2)];
    let ctx = CallContext::new(&args);

    assert!(ctx.has_args());
    assert!(!ctx.has_kwargs());
    assert_eq!(ctx.arg_count(), 2);
    assert!(!ctx.is_super_call);
}

#[test]
fn test_call_context_empty() {
    let args: Vec<Value> = vec![];
    let ctx = CallContext::new(&args);

    assert!(!ctx.has_args());
    assert!(!ctx.has_kwargs());
    assert_eq!(ctx.arg_count(), 0);
}

#[test]
fn test_call_context_with_kwargs() {
    let args = vec![Value::int_unchecked(1)];
    let kwargs = vec![
        (intern("x"), Value::int_unchecked(10)),
        (intern("y"), Value::int_unchecked(20)),
    ];
    let ctx = CallContext::with_kwargs(&args, &kwargs);

    assert!(ctx.has_args());
    assert!(ctx.has_kwargs());
    assert_eq!(ctx.arg_count(), 3);
}

#[test]
fn test_call_context_as_super_call() {
    let args = vec![Value::int_unchecked(1)];
    let ctx = CallContext::new(&args).as_super_call();

    assert!(ctx.is_super_call);
}

// =========================================================================
// CallStats Tests
// =========================================================================

#[test]
fn test_call_stats_new() {
    let stats = CallStats::new();
    use std::sync::atomic::Ordering;
    assert_eq!(stats.default_both_calls.load(Ordering::Relaxed), 0);
    assert_eq!(stats.cache_hits.load(Ordering::Relaxed), 0);
}

#[test]
fn test_call_stats_record_call() {
    let stats = CallStats::new();
    use std::sync::atomic::Ordering;

    stats.record_call(&CallSpecialization::DefaultBoth);
    stats.record_call(&CallSpecialization::DefaultBoth);
    stats.record_call(&CallSpecialization::DefaultNew {
        init_class: ClassId(1),
    });

    assert_eq!(stats.default_both_calls.load(Ordering::Relaxed), 2);
    assert_eq!(stats.default_new_calls.load(Ordering::Relaxed), 1);
}

#[test]
fn test_call_stats_record_cache() {
    let stats = CallStats::new();
    use std::sync::atomic::Ordering;

    stats.record_cache_hit();
    stats.record_cache_hit();
    stats.record_cache_miss();

    assert_eq!(stats.cache_hits.load(Ordering::Relaxed), 2);
    assert_eq!(stats.cache_misses.load(Ordering::Relaxed), 1);
}
