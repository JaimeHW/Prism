use super::*;
use prism_gc::trace::tracer::CountingTracer;
use prism_runtime::object::views::TracebackViewObject;
use prism_runtime::{size_of_object, trace_object};

// ════════════════════════════════════════════════════════════════════════
// Construction Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_exception_value_new() {
    let exc = ExceptionValue::new(ExceptionTypeId::ValueError, Some(Arc::from("test")));
    assert_eq!(exc.type_id(), ExceptionTypeId::ValueError);
    assert_eq!(exc.message(), Some("test"));
    assert_eq!(exc.header.type_id, TypeId::EXCEPTION);
}

#[test]
fn test_exception_value_no_message() {
    let exc = ExceptionValue::new(ExceptionTypeId::TypeError, None);
    assert_eq!(exc.type_id(), ExceptionTypeId::TypeError);
    assert!(exc.message().is_none());
}

#[test]
fn test_exception_value_import_details_round_trip() {
    let exc = ExceptionValue::new(
        ExceptionTypeId::ModuleNotFoundError,
        Some(Arc::from("No module named 'pkg.missing'")),
    )
    .with_import_details(Some(Arc::from("pkg.missing")), None);
    assert_eq!(exc.import_name(), Some("pkg.missing"));
    assert!(exc.import_path().is_none());
}

#[test]
fn test_exception_value_syntax_details_round_trip() {
    let exc = ExceptionValue::new(
        ExceptionTypeId::SyntaxError,
        Some(Arc::from("invalid syntax")),
    )
    .with_syntax_details(SyntaxErrorDetails::new(
        Some(Arc::from("demo.py")),
        Some(3),
        Some(7),
        Some(Arc::from("value =\n")),
        Some(3),
        Some(8),
    ));
    assert_eq!(exc.syntax_filename(), Some("demo.py"));
    assert_eq!(exc.syntax_lineno(), Some(3));
    assert_eq!(exc.syntax_offset(), Some(7));
    assert_eq!(exc.syntax_text(), Some("value =\n"));
    assert_eq!(exc.syntax_end_lineno(), Some(3));
    assert_eq!(exc.syntax_end_offset(), Some(8));
}

#[test]
fn test_exception_value_with_args() {
    let args = vec![Value::int(1).unwrap(), Value::int(2).unwrap()].into_boxed_slice();
    let exc = ExceptionValue::with_args(ExceptionTypeId::KeyError, Some(Arc::from("key")), args);
    assert_eq!(exc.type_id(), ExceptionTypeId::KeyError);
    assert!(exc.args.is_some());
    assert_eq!(exc.args.as_ref().unwrap().len(), 2);
}

#[test]
fn test_exception_type_name() {
    let exc = ExceptionValue::new(ExceptionTypeId::ZeroDivisionError, None);
    assert_eq!(exc.type_name(), "ZeroDivisionError");
}

// ════════════════════════════════════════════════════════════════════════
// Flags Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_exception_flags_none() {
    let flags = ExceptionFlags::NONE;
    assert!(!flags.has(ExceptionFlags::HAS_CAUSE));
    assert!(!flags.has(ExceptionFlags::SUPPRESS_CONTEXT));
}

#[test]
fn test_exception_flags_with() {
    let flags = ExceptionFlags::NONE.with(ExceptionFlags::HAS_CAUSE);
    assert!(flags.has(ExceptionFlags::HAS_CAUSE));
    assert!(!flags.has(ExceptionFlags::SUPPRESS_CONTEXT));
}

#[test]
fn test_exception_flags_without() {
    let flags = ExceptionFlags::NONE
        .with(ExceptionFlags::HAS_CAUSE)
        .with(ExceptionFlags::SUPPRESS_CONTEXT)
        .without(ExceptionFlags::HAS_CAUSE);
    assert!(!flags.has(ExceptionFlags::HAS_CAUSE));
    assert!(flags.has(ExceptionFlags::SUPPRESS_CONTEXT));
}

#[test]
fn test_exception_flags_multiple() {
    let flags = ExceptionFlags::NONE
        .with(ExceptionFlags::HAS_CAUSE)
        .with(ExceptionFlags::HANDLING)
        .with(ExceptionFlags::FLYWEIGHT);
    assert!(flags.has(ExceptionFlags::HAS_CAUSE));
    assert!(flags.has(ExceptionFlags::HANDLING));
    assert!(flags.has(ExceptionFlags::FLYWEIGHT));
    assert!(!flags.has(ExceptionFlags::SUPPRESS_CONTEXT));
}

// ════════════════════════════════════════════════════════════════════════
// Cause/Context Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_exception_set_cause() {
    let cause = Box::leak(Box::new(ExceptionValue::new(
        ExceptionTypeId::OSError,
        Some(Arc::from("original")),
    )));

    let mut exc = ExceptionValue::new(ExceptionTypeId::ValueError, Some(Arc::from("wrapped")));
    exc.set_cause(cause);

    assert!(exc.flags.has(ExceptionFlags::HAS_CAUSE));
    assert!(exc.cause.is_some());
}

#[test]
fn test_exception_suppress_context() {
    let mut exc = ExceptionValue::new(ExceptionTypeId::ValueError, None);
    exc.suppress_context();
    assert!(exc.flags.has(ExceptionFlags::SUPPRESS_CONTEXT));
}

#[test]
fn test_exception_set_traceback_value() {
    let mut exc = ExceptionValue::new(ExceptionTypeId::ValueError, None);
    let traceback = Value::int(7).unwrap();
    exc.set_traceback(traceback);

    assert_eq!(exc.traceback(), Some(traceback));
    assert_ne!(exc.traceback_id, 0);

    exc.clear_traceback();
    assert!(exc.traceback().is_none());
    assert_eq!(exc.traceback_id, 0);
}

#[test]
fn test_exception_replace_traceback_accepts_traceback_objects_and_none() {
    let traceback = Value::object_ptr(Box::into_raw(Box::new(TracebackViewObject::new(
        Value::none(),
        None,
        12,
        3,
    ))) as *const ());
    let mut exc = ExceptionValue::new(ExceptionTypeId::ValueError, None);

    exc.replace_traceback(traceback)
        .expect("traceback objects should be accepted");
    assert_eq!(exc.traceback(), Some(traceback));

    exc.replace_traceback(Value::none())
        .expect("None should clear traceback");
    assert!(exc.traceback().is_none());
}

#[test]
fn test_exception_replace_traceback_rejects_non_tracebacks() {
    let mut exc = ExceptionValue::new(ExceptionTypeId::ValueError, None);
    let err = exc
        .replace_traceback(Value::int(7).unwrap())
        .expect_err("non-traceback values should be rejected");
    assert_eq!(err, TRACEBACK_TYPE_ERROR_MESSAGE);
}

// ════════════════════════════════════════════════════════════════════════
// Subclass Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_is_subclass_of_self() {
    let exc = ExceptionValue::new(ExceptionTypeId::ValueError, None);
    assert!(exc.is_subclass_of(ExceptionTypeId::ValueError));
}

#[test]
fn test_is_subclass_of_parent() {
    let exc = ExceptionValue::new(ExceptionTypeId::ValueError, None);
    assert!(exc.is_subclass_of(ExceptionTypeId::Exception));
    assert!(exc.is_subclass_of(ExceptionTypeId::BaseException));
}

#[test]
fn test_is_not_subclass() {
    let exc = ExceptionValue::new(ExceptionTypeId::ValueError, None);
    assert!(!exc.is_subclass_of(ExceptionTypeId::TypeError));
    assert!(!exc.is_subclass_of(ExceptionTypeId::OSError));
}

#[test]
fn test_zero_division_is_arithmetic() {
    let exc = ExceptionValue::new(ExceptionTypeId::ZeroDivisionError, None);
    assert!(exc.is_subclass_of(ExceptionTypeId::ArithmeticError));
    assert!(exc.is_subclass_of(ExceptionTypeId::Exception));
}

// ════════════════════════════════════════════════════════════════════════
// Value Conversion Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_exception_into_value() {
    let exc = ExceptionValue::new(ExceptionTypeId::RuntimeError, Some(Arc::from("test")));
    let value = exc.into_value();

    assert!(value.is_object());
    assert!(value.as_object_ptr().is_some());
}

#[test]
fn test_exception_into_value_uses_bound_vm_heap_when_available() {
    let baseline = PINNED_EXCEPTION_VALUES.with(|store| store.borrow().len());
    let _vm = VirtualMachine::new();

    let value = ExceptionValue::new(
        ExceptionTypeId::RuntimeError,
        Some(Arc::from("managed allocation")),
    )
    .into_value();

    assert_eq!(
        PINNED_EXCEPTION_VALUES.with(|store| store.borrow().len()),
        baseline
    );
    let recovered =
        unsafe { ExceptionValue::from_value(value).expect("exception should downcast") };
    assert_eq!(recovered.message(), Some("managed allocation"));
}

#[test]
fn test_exception_into_value_survives_vm_move_after_binding() {
    fn relocate_vm(vm: VirtualMachine) -> VirtualMachine {
        vm
    }

    let baseline = PINNED_EXCEPTION_VALUES.with(|store| store.borrow().len());
    let _vm = relocate_vm(VirtualMachine::new());

    let value = ExceptionValue::new(
        ExceptionTypeId::RuntimeError,
        Some(Arc::from("moved vm allocation")),
    )
    .into_value();

    assert_eq!(
        PINNED_EXCEPTION_VALUES.with(|store| store.borrow().len()),
        baseline
    );
    let recovered =
        unsafe { ExceptionValue::from_value(value).expect("exception should downcast") };
    assert_eq!(recovered.message(), Some("moved vm allocation"));
}

#[test]
fn test_exception_from_value() {
    let exc = ExceptionValue::new(ExceptionTypeId::IndexError, Some(Arc::from("out of range")));
    let value = exc.into_value();

    let recovered = unsafe { ExceptionValue::from_value(value) };
    assert!(recovered.is_some());

    let recovered = recovered.unwrap();
    assert_eq!(recovered.type_id(), ExceptionTypeId::IndexError);
    assert_eq!(recovered.message(), Some("out of range"));
}

#[test]
fn test_exception_from_value_mut_allows_in_place_traceback_updates() {
    let traceback = Value::object_ptr(Box::into_raw(Box::new(TracebackViewObject::new(
        Value::none(),
        None,
        21,
        5,
    ))) as *const ());
    let value = ExceptionValue::new(ExceptionTypeId::RuntimeError, None).into_value();

    let exception = unsafe {
        ExceptionValue::from_value_mut(value).expect("exception value should downcast mutably")
    };
    exception
        .replace_traceback(traceback)
        .expect("mutable exception should accept traceback");

    let observed =
        unsafe { ExceptionValue::from_value(value).expect("exception should remain valid") };
    assert_eq!(observed.traceback(), Some(traceback));
}

#[test]
fn test_exception_from_non_exception_value() {
    let value = Value::int(42).unwrap();
    let result = unsafe { ExceptionValue::from_value(value) };
    assert!(result.is_none());
}

// ════════════════════════════════════════════════════════════════════════
// Display Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_exception_display_with_message() {
    let exc = ExceptionValue::new(
        ExceptionTypeId::ValueError,
        Some(Arc::from("invalid input")),
    );
    let display = format!("{}", exc);
    assert_eq!(display, "invalid input");
}

#[test]
fn test_exception_display_no_message() {
    let exc = ExceptionValue::new(ExceptionTypeId::StopIteration, None);
    let display = format!("{}", exc);
    assert_eq!(display, "");
}

#[test]
fn test_exception_debug() {
    let exc = ExceptionValue::new(ExceptionTypeId::TypeError, Some(Arc::from("test")));
    let debug = format!("{:?}", exc);
    assert!(debug.contains("ExceptionValue"));
    assert!(debug.contains("TypeError"));
}

// ════════════════════════════════════════════════════════════════════════
// Helper Function Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_create_exception() {
    let value = create_exception(ExceptionTypeId::NameError, Some(Arc::from("undefined")));
    assert!(value.is_object());

    let exc = unsafe { ExceptionValue::from_value(value).unwrap() };
    assert_eq!(exc.type_id(), ExceptionTypeId::NameError);
    assert_eq!(exc.message(), Some("undefined"));
}

#[test]
fn test_create_exception_no_message() {
    let value = create_exception(ExceptionTypeId::MemoryError, None);
    let exc = unsafe { ExceptionValue::from_value(value).unwrap() };
    assert_eq!(exc.type_id(), ExceptionTypeId::MemoryError);
    assert!(exc.message().is_none());
}

#[test]
fn test_create_exception_with_args() {
    let args = vec![Value::int(1).unwrap()].into_boxed_slice();
    let value =
        create_exception_with_args(ExceptionTypeId::SystemExit, Some(Arc::from("exit")), args);

    let exc = unsafe { ExceptionValue::from_value(value).unwrap() };
    assert_eq!(exc.type_id(), ExceptionTypeId::SystemExit);
    assert!(exc.args.is_some());
}

#[test]
fn test_create_exception_with_import_details() {
    let value = create_exception_with_import_details(
        ExceptionTypeId::ModuleNotFoundError,
        Some(Arc::from("No module named 'pkg.missing'")),
        Some(Arc::from("pkg.missing")),
        None,
    );

    let exc = unsafe { ExceptionValue::from_value(value).unwrap() };
    assert_eq!(exc.type_id(), ExceptionTypeId::ModuleNotFoundError);
    assert_eq!(exc.import_name(), Some("pkg.missing"));
    assert!(exc.import_path().is_none());
}

#[test]
fn test_create_exception_with_syntax_details() {
    let value = create_exception_with_syntax_details(
        ExceptionTypeId::SyntaxError,
        Some(Arc::from("expected ':'")),
        SyntaxErrorDetails::new(
            Some(Arc::from("sample.py")),
            Some(5),
            Some(9),
            Some(Arc::from("if True\n")),
            Some(5),
            Some(10),
        ),
    );

    let exc = unsafe { ExceptionValue::from_value(value).unwrap() };
    assert_eq!(exc.type_id(), ExceptionTypeId::SyntaxError);
    assert_eq!(exc.syntax_filename(), Some("sample.py"));
    assert_eq!(exc.syntax_lineno(), Some(5));
    assert_eq!(exc.syntax_offset(), Some(9));
    assert_eq!(exc.syntax_text(), Some("if True\n"));
    assert_eq!(exc.syntax_end_lineno(), Some(5));
    assert_eq!(exc.syntax_end_offset(), Some(10));
}

#[test]
fn test_display_text_prefers_single_string_arg() {
    let args = vec![Value::string(prism_core::intern::intern("boom"))].into_boxed_slice();
    let exc = ExceptionValue::with_args(ExceptionTypeId::ValueError, None, args);
    assert_eq!(exc.display_text(), "boom");
}

#[test]
fn test_repr_text_uses_exception_type_and_args() {
    let args = vec![Value::string(prism_core::intern::intern("boom"))].into_boxed_slice();
    let exc = ExceptionValue::with_args(ExceptionTypeId::ValueError, None, args);
    assert_eq!(exc.repr_text(), "ValueError('boom')");
}

#[test]
fn test_exception_method_value_exposes_core_base_exception_builtins() {
    for (name, expected) in [
        ("__new__", "BaseException.__new__"),
        ("__init__", "BaseException.__init__"),
        ("__str__", "BaseException.__str__"),
        ("__repr__", "BaseException.__repr__"),
        ("with_traceback", "BaseException.with_traceback"),
    ] {
        let method = exception_method_value(name).expect("base exception builtin should resolve");
        let ptr = method
            .as_object_ptr()
            .expect("base exception builtin should be heap allocated");
        let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
        assert_eq!(builtin.name(), expected);
    }
}

#[test]
fn test_exception_display_and_repr_helpers_cover_native_exceptions() {
    let exc = ExceptionValue::with_args(
        ExceptionTypeId::ValueError,
        None,
        vec![Value::string(prism_core::intern::intern("boom"))].into_boxed_slice(),
    )
    .into_value();

    assert_eq!(
        exception_display_text_for_value(exc).as_deref(),
        Some("boom")
    );
    assert_eq!(
        exception_repr_text_for_value(exc).as_deref(),
        Some("ValueError('boom')")
    );
}

#[test]
fn test_exception_gc_dispatch_traces_args_links_and_traceback() {
    let vm = VirtualMachine::new();
    let cause =
        ExceptionValue::new(ExceptionTypeId::ValueError, Some(Arc::from("cause"))).into_value();
    let context =
        ExceptionValue::new(ExceptionTypeId::TypeError, Some(Arc::from("context"))).into_value();

    let mut exc = ExceptionValue::with_args(
        ExceptionTypeId::RuntimeError,
        Some(Arc::from("boom")),
        vec![
            Value::int(1).unwrap(),
            Value::string(prism_core::intern::intern("arg")),
        ]
        .into_boxed_slice(),
    );
    let expected_size = std::mem::size_of::<ExceptionValue>()
        + std::mem::size_of_val(exc.args.as_deref().expect("args should exist"));
    let cause_ptr = unsafe { ExceptionValue::from_value(cause).expect("cause should downcast") }
        as *const ExceptionValue;
    let context_ptr =
        unsafe { ExceptionValue::from_value(context).expect("context should downcast") }
            as *const ExceptionValue;
    exc.set_cause(cause_ptr);
    exc.set_context(context_ptr);
    exc.set_traceback(Value::string(prism_core::intern::intern("traceback")));

    let value = exc
        .into_gc_value(&vm)
        .expect("managed exception allocation should succeed");
    let ptr = value
        .as_object_ptr()
        .expect("managed exception should be object-backed");
    let mut tracer = CountingTracer::new();

    unsafe {
        trace_object(ptr, TypeId::EXCEPTION, &mut tracer);
    }

    assert_eq!(tracer.value_count, 3);
    assert_eq!(tracer.ptr_count, 2);
    let size = unsafe { size_of_object(ptr, TypeId::EXCEPTION) };
    assert_eq!(size, expected_size);
}

// ════════════════════════════════════════════════════════════════════════
// Memory Layout Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_exception_value_size() {
    // Verify the struct is reasonably sized
    let size = std::mem::size_of::<ExceptionValue>();
    // Should be <= 96 bytes for cache efficiency
    assert!(
        size <= 128,
        "ExceptionValue is {} bytes, expected <= 128",
        size
    );
}

#[test]
fn test_exception_value_alignment() {
    let align = std::mem::align_of::<ExceptionValue>();
    // Should be 8-byte aligned for pointer fields
    assert!(
        align >= 8,
        "ExceptionValue alignment is {}, expected >= 8",
        align
    );
}

#[test]
fn test_exception_flags_size() {
    assert_eq!(std::mem::size_of::<ExceptionFlags>(), 2);
}

// ════════════════════════════════════════════════════════════════════════
// All Exception Types Test
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_all_exception_types_constructible() {
    // Test that all exception types can be constructed
    let types = [
        ExceptionTypeId::BaseException,
        ExceptionTypeId::Exception,
        ExceptionTypeId::ValueError,
        ExceptionTypeId::TypeError,
        ExceptionTypeId::KeyError,
        ExceptionTypeId::IndexError,
        ExceptionTypeId::AttributeError,
        ExceptionTypeId::NameError,
        ExceptionTypeId::ZeroDivisionError,
        ExceptionTypeId::RuntimeError,
        ExceptionTypeId::StopIteration,
        ExceptionTypeId::OSError,
        ExceptionTypeId::FileNotFoundError,
        ExceptionTypeId::PermissionError,
        ExceptionTypeId::MemoryError,
        ExceptionTypeId::RecursionError,
        ExceptionTypeId::ImportError,
        ExceptionTypeId::ModuleNotFoundError,
        ExceptionTypeId::SyntaxError,
        ExceptionTypeId::IndentationError,
    ];

    for type_id in types {
        let exc = ExceptionValue::new(type_id, Some(Arc::from("test")));
        assert_eq!(exc.type_id(), type_id);
        assert_eq!(exc.type_name(), type_id.name());
    }
}
