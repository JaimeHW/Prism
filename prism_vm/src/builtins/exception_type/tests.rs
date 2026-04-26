use super::*;
use crate::builtins::ExceptionValue;
use prism_gc::heap::GcHeap;
use prism_runtime::allocation_context::RuntimeHeapBinding;

// ════════════════════════════════════════════════════════════════════════
// Memory Layout Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_exception_type_object_size() {
    assert_eq!(std::mem::size_of::<ExceptionTypeObject>(), 48);
}

#[test]
fn test_exception_type_object_alignment() {
    assert_eq!(std::mem::align_of::<ExceptionTypeObject>(), 8);
}

#[test]
fn test_header_offset() {
    assert_eq!(std::mem::offset_of!(ExceptionTypeObject, header), 0);
}

#[test]
fn test_type_id_offset() {
    // type_id should be at offset 16 (after ObjectHeader)
    assert_eq!(
        std::mem::offset_of!(ExceptionTypeObject, exception_type_id),
        16
    );
}

#[test]
fn test_name_offset() {
    // name should be at offset 24
    assert_eq!(std::mem::offset_of!(ExceptionTypeObject, name), 24);
}

// ════════════════════════════════════════════════════════════════════════
// Construction Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_new_value_error() {
    let exc_type = ExceptionTypeObject::new(ExceptionTypeId::ValueError, "ValueError");
    assert_eq!(
        exc_type.exception_type_id,
        ExceptionTypeId::ValueError as u16
    );
    assert_eq!(exc_type.name(), "ValueError");
    assert_eq!(exc_type.header.type_id, EXCEPTION_TYPE_ID);
}

#[test]
fn test_new_type_error() {
    let exc_type = ExceptionTypeObject::new(ExceptionTypeId::TypeError, "TypeError");
    assert_eq!(
        exc_type.exception_type_id,
        ExceptionTypeId::TypeError as u16
    );
    assert_eq!(exc_type.name(), "TypeError");
}

#[test]
fn test_new_system_exit() {
    let exc_type = ExceptionTypeObject::new(ExceptionTypeId::SystemExit, "SystemExit");
    assert!(exc_type.is_system_exit());
    assert!(!exc_type.caught_by_bare_except());
}

#[test]
fn test_new_keyboard_interrupt() {
    let exc_type =
        ExceptionTypeObject::new(ExceptionTypeId::KeyboardInterrupt, "KeyboardInterrupt");
    assert!(
        exc_type
            .flags
            .contains(ExceptionTypeFlags::KEYBOARD_INTERRUPT)
    );
    assert!(!exc_type.caught_by_bare_except());
}

#[test]
fn test_new_generator_exit() {
    let exc_type = ExceptionTypeObject::new(ExceptionTypeId::GeneratorExit, "GeneratorExit");
    assert!(exc_type.flags.contains(ExceptionTypeFlags::GENERATOR_EXIT));
}

#[test]
fn test_regular_exception_caught_by_bare_except() {
    let exc_type = ExceptionTypeObject::new(ExceptionTypeId::ValueError, "ValueError");
    assert!(exc_type.caught_by_bare_except());
}

// ════════════════════════════════════════════════════════════════════════
// Type ID Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_type_id_extraction() {
    let exc_type = ExceptionTypeObject::new(ExceptionTypeId::KeyError, "KeyError");
    assert_eq!(exc_type.type_id(), ExceptionTypeId::KeyError as u16);
}

#[test]
fn test_exception_type_enum() {
    let exc_type = ExceptionTypeObject::new(ExceptionTypeId::IndexError, "IndexError");
    assert_eq!(exc_type.exception_type(), Some(ExceptionTypeId::IndexError));
}

#[test]
fn test_base_type_id() {
    // ValueError's parent is Exception
    let exc_type = ExceptionTypeObject::new(ExceptionTypeId::ValueError, "ValueError");
    assert_eq!(exc_type.base_type_id(), ExceptionTypeId::Exception as u16);
}

#[test]
fn test_base_exception_has_self_as_base() {
    let exc_type = ExceptionTypeObject::new(ExceptionTypeId::BaseException, "BaseException");
    assert_eq!(
        exc_type.base_type_id(),
        ExceptionTypeId::BaseException as u16
    );
}

// ════════════════════════════════════════════════════════════════════════
// Subclass Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_is_subclass_of_self() {
    let exc_type = ExceptionTypeObject::new(ExceptionTypeId::ValueError, "ValueError");
    assert!(exc_type.is_subclass_of(ExceptionTypeId::ValueError as u16));
}

#[test]
fn test_value_error_is_subclass_of_exception() {
    let exc_type = ExceptionTypeObject::new(ExceptionTypeId::ValueError, "ValueError");
    assert!(exc_type.is_subclass_of(ExceptionTypeId::Exception as u16));
}

#[test]
fn test_value_error_is_subclass_of_base_exception() {
    let exc_type = ExceptionTypeObject::new(ExceptionTypeId::ValueError, "ValueError");
    assert!(exc_type.is_subclass_of(ExceptionTypeId::BaseException as u16));
}

#[test]
fn test_type_error_not_subclass_of_value_error() {
    let exc_type = ExceptionTypeObject::new(ExceptionTypeId::TypeError, "TypeError");
    assert!(!exc_type.is_subclass_of(ExceptionTypeId::ValueError as u16));
}

#[test]
fn test_zero_division_is_subclass_of_arithmetic() {
    let exc_type =
        ExceptionTypeObject::new(ExceptionTypeId::ZeroDivisionError, "ZeroDivisionError");
    assert!(exc_type.is_subclass_of(ExceptionTypeId::ArithmeticError as u16));
}

#[test]
fn test_key_error_is_subclass_of_lookup_error() {
    let exc_type = ExceptionTypeObject::new(ExceptionTypeId::KeyError, "KeyError");
    assert!(exc_type.is_subclass_of(ExceptionTypeId::LookupError as u16));
}

#[test]
fn test_exception_group_is_subclass_of_exception_and_base_exception_group() {
    let exc_type = ExceptionTypeObject::new(ExceptionTypeId::ExceptionGroup, "ExceptionGroup");
    assert!(exc_type.is_subclass_of(ExceptionTypeId::Exception as u16));
    assert!(exc_type.is_subclass_of(ExceptionTypeId::BaseExceptionGroup as u16));
    assert!(exc_type.is_subclass_of(ExceptionTypeId::BaseException as u16));
}

// ════════════════════════════════════════════════════════════════════════
// Construction/Call Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_construct_no_args() {
    let exc_type = ExceptionTypeObject::new(ExceptionTypeId::ValueError, "ValueError");
    let exc = exc_type.construct(&[]);
    assert!(exc.is_object());
}

#[test]
fn test_construct_with_int_arg() {
    let exc_type = ExceptionTypeObject::new(ExceptionTypeId::ValueError, "ValueError");
    let exc = exc_type.construct(&[Value::int(42).unwrap()]);
    assert!(exc.is_object());
}

#[test]
fn test_construct_with_string_arg_preserves_original_args() {
    let exc_type = ExceptionTypeObject::new(ExceptionTypeId::ValueError, "ValueError");
    let exc = exc_type.construct(&[Value::string(intern("boom"))]);
    let exc = unsafe { ExceptionValue::from_value(exc).expect("exception instance") };

    assert!(exc.message().is_none());
    let args = exc
        .args
        .as_deref()
        .expect("constructor should preserve args");
    assert_eq!(args.len(), 1);
    assert!(args[0].is_string());
}

#[test]
fn test_call_returns_ok() {
    let exc_type = ExceptionTypeObject::new(ExceptionTypeId::TypeError, "TypeError");
    let result = exc_type.call(&[]);
    assert!(result.is_ok());
}

// ════════════════════════════════════════════════════════════════════════
// Static Type Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_static_value_error() {
    assert_eq!(
        VALUE_ERROR.exception_type_id,
        ExceptionTypeId::ValueError as u16
    );
    assert_eq!(VALUE_ERROR.name(), "ValueError");
}

#[test]
fn test_static_type_error() {
    assert_eq!(
        TYPE_ERROR.exception_type_id,
        ExceptionTypeId::TypeError as u16
    );
}

#[test]
fn test_static_system_exit() {
    assert!(SYSTEM_EXIT.is_system_exit());
}

#[test]
fn test_static_keyboard_interrupt() {
    assert!(
        KEYBOARD_INTERRUPT
            .flags
            .contains(ExceptionTypeFlags::KEYBOARD_INTERRUPT)
    );
}

// ════════════════════════════════════════════════════════════════════════
// Table Lookup Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_get_exception_type_value_error() {
    let exc_type = get_exception_type("ValueError");
    assert!(exc_type.is_some());
    assert_eq!(exc_type.unwrap().name(), "ValueError");
}

#[test]
fn test_get_exception_type_type_error() {
    let exc_type = get_exception_type("TypeError");
    assert!(exc_type.is_some());
}

#[test]
fn test_get_exception_type_ioerror_alias() {
    let exc_type = get_exception_type("IOError");
    assert!(exc_type.is_some());
    // IOError is an alias for OSError
    assert_eq!(
        exc_type.unwrap().exception_type_id,
        ExceptionTypeId::OSError as u16
    );
}

#[test]
fn test_get_exception_type_environment_error_alias() {
    let exc_type = get_exception_type("EnvironmentError");
    assert!(exc_type.is_some());
    assert_eq!(
        exc_type.unwrap().exception_type_id,
        ExceptionTypeId::OSError as u16
    );
}

#[test]
fn test_get_exception_type_nonexistent() {
    let exc_type = get_exception_type("FooBarError");
    assert!(exc_type.is_none());
}

#[test]
fn test_get_exception_type_by_id_value_error() {
    let exc_type = get_exception_type_by_id(ExceptionTypeId::ValueError as u16);
    assert!(exc_type.is_some());
    assert_eq!(exc_type.unwrap().name(), "ValueError");
}

#[test]
fn test_get_exception_type_by_id_invalid() {
    let exc_type = get_exception_type_by_id(255);
    assert!(exc_type.is_none());
}

#[test]
fn test_exception_proxy_class_bridge_preserves_exception_hierarchy() {
    let exception_id =
        exception_proxy_class_id(ExceptionTypeId::Exception as u16).expect("Exception proxy");
    let runtime_error_id =
        exception_proxy_class_id(ExceptionTypeId::RuntimeError as u16).expect("RuntimeError proxy");
    let warning_id =
        exception_proxy_class_id(ExceptionTypeId::Warning as u16).expect("Warning proxy");

    let runtime_error_bitmap =
        prism_runtime::object::type_builtins::global_class_bitmap(runtime_error_id)
            .expect("RuntimeError proxy should be registered");
    let warning_bitmap = prism_runtime::object::type_builtins::global_class_bitmap(warning_id)
        .expect("Warning proxy should be registered");

    assert!(runtime_error_bitmap.is_subclass_of(TypeId::from_raw(exception_id.0)));
    assert!(!runtime_error_bitmap.is_subclass_of(TypeId::from_raw(warning_id.0)));
    assert!(warning_bitmap.is_subclass_of(TypeId::from_raw(exception_id.0)));
}

#[test]
fn test_exception_group_proxy_class_preserves_dual_exception_bases() {
    let exception_id =
        exception_proxy_class_id(ExceptionTypeId::Exception as u16).expect("Exception proxy");
    let base_group_id = exception_proxy_class_id(ExceptionTypeId::BaseExceptionGroup as u16)
        .expect("BaseExceptionGroup proxy");
    let group_id = exception_proxy_class_id(ExceptionTypeId::ExceptionGroup as u16)
        .expect("ExceptionGroup proxy");

    let group_bitmap = prism_runtime::object::type_builtins::global_class_bitmap(group_id)
        .expect("ExceptionGroup proxy should be registered");

    assert!(group_bitmap.is_subclass_of(TypeId::from_raw(exception_id.0)));
    assert!(group_bitmap.is_subclass_of(TypeId::from_raw(base_group_id.0)));
}

#[test]
fn test_exception_proxy_class_id_round_trips_to_builtin_exception_value() {
    let runtime_error_id =
        exception_proxy_class_id(ExceptionTypeId::RuntimeError as u16).expect("RuntimeError proxy");
    let value = exception_type_value_for_proxy_class_id(runtime_error_id)
        .expect("RuntimeError proxy should map back to builtin exception type value");
    let ptr = value
        .as_object_ptr()
        .expect("builtin exception type should be a heap object");
    let exc_type = unsafe { &*(ptr as *const ExceptionTypeObject) };

    assert_eq!(
        exc_type.exception_type_id,
        ExceptionTypeId::RuntimeError as u16
    );
}

#[test]
fn test_exception_type_attribute_value_exposes_class_metadata() {
    let warning = get_exception_type("Warning").expect("Warning type should exist");

    let name = exception_type_attribute_value(warning, &intern("__name__"))
        .expect("__name__ should exist");
    let name_ptr = name
        .as_string_object_ptr()
        .expect("__name__ should be an interned string");
    assert_eq!(
        prism_core::intern::interned_by_ptr(name_ptr as *const u8)
            .unwrap()
            .as_str(),
        "Warning"
    );

    let bases = exception_type_attribute_value(warning, &intern("__bases__"))
        .expect("__bases__ should exist");
    let bases_ptr = bases
        .as_object_ptr()
        .expect("__bases__ should be a tuple object");
    let bases = unsafe { &*(bases_ptr as *const TupleObject) };
    assert_eq!(bases.len(), 1);

    let base_ptr = bases.as_slice()[0]
        .as_object_ptr()
        .expect("base should be an exception type object");
    let base = unsafe { &*(base_ptr as *const ExceptionTypeObject) };
    assert_eq!(base.name(), "Exception");
}

#[test]
fn test_exception_proxy_classes_expose_base_exception_slots() {
    let proxy = exception_proxy_class(ExceptionTypeId::Exception);

    for name in [
        "__new__",
        "__init__",
        "__str__",
        "__repr__",
        "with_traceback",
    ] {
        assert!(
            proxy.get_attr(&intern(name)).is_some(),
            "exception proxy should expose {name}"
        );
    }
}

#[test]
fn test_exception_proxy_new_staticmethod_ignores_bound_vm_heap() {
    let heap = GcHeap::with_defaults();
    let _binding = RuntimeHeapBinding::register(&heap);
    let method = crate::builtins::exception_method_value("__new__")
        .expect("BaseException.__new__ should be registered");

    let value = exception_proxy_namespace_value("__new__", method);
    let ptr = value
        .as_object_ptr()
        .expect("staticmethod descriptor should be object-backed");
    let descriptor = unsafe { &*(ptr as *const StaticMethodDescriptor) };

    assert_eq!(descriptor.header.type_id, TypeId::STATICMETHOD);
    assert_eq!(descriptor.function(), method);
    assert!(!heap.contains(ptr));
}

#[test]
fn test_exception_proxy_classes_are_native_heaptypes() {
    let proxy = exception_proxy_class(ExceptionTypeId::Exception);
    assert!(proxy.is_native_heaptype());
}

#[test]
fn test_exception_type_attribute_value_exposes_base_exception_methods() {
    let base = get_exception_type("BaseException").expect("BaseException type should exist");

    for (name, expected) in [
        ("__new__", "BaseException.__new__"),
        ("__init__", "BaseException.__init__"),
        ("__str__", "BaseException.__str__"),
        ("__repr__", "BaseException.__repr__"),
        ("with_traceback", "BaseException.with_traceback"),
    ] {
        let value = exception_type_attribute_value(base, &intern(name))
            .unwrap_or_else(|| panic!("{name} should resolve from exception type metadata"));
        let ptr = value
            .as_object_ptr()
            .expect("base exception methods should be heap allocated builtins");
        let builtin = unsafe { &*(ptr as *const crate::builtins::BuiltinFunctionObject) };
        assert_eq!(builtin.name(), expected);
        assert_eq!(builtin.bound_self(), None);
    }
}

#[test]
fn test_exception_group_type_metadata_exposes_both_bases_and_full_mro() {
    let exc_group = get_exception_type("ExceptionGroup").expect("ExceptionGroup type should exist");

    let bases = exception_type_attribute_value(exc_group, &intern("__bases__"))
        .expect("__bases__ should exist");
    let bases_ptr = bases
        .as_object_ptr()
        .expect("__bases__ should be a tuple object");
    let bases = unsafe { &*(bases_ptr as *const TupleObject) };
    assert_eq!(bases.len(), 2);

    let first_base = unsafe {
        &*(bases.as_slice()[0]
            .as_object_ptr()
            .expect("first base should be an exception type object")
            as *const ExceptionTypeObject)
    };
    let second_base = unsafe {
        &*(bases.as_slice()[1]
            .as_object_ptr()
            .expect("second base should be an exception type object")
            as *const ExceptionTypeObject)
    };
    assert_eq!(first_base.name(), "BaseExceptionGroup");
    assert_eq!(second_base.name(), "Exception");

    let mro = exception_type_attribute_value(exc_group, &intern("__mro__"))
        .expect("__mro__ should exist");
    let mro_ptr = mro
        .as_object_ptr()
        .expect("__mro__ should be a tuple object");
    let mro = unsafe { &*(mro_ptr as *const TupleObject) };
    let mro_names = mro
        .as_slice()
        .iter()
        .map(|value| {
            let ptr = value
                .as_object_ptr()
                .expect("mro entries should be type objects");
            if crate::ops::objects::extract_type_id(ptr) == TypeId::TYPE
                && let Some(type_id) = crate::builtins::builtin_type_object_type_id(ptr)
            {
                return type_id.name().to_string();
            }

            unsafe { &*(ptr as *const ExceptionTypeObject) }
                .name()
                .to_string()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        mro_names,
        vec![
            "ExceptionGroup",
            "BaseExceptionGroup",
            "Exception",
            "BaseException",
            "object",
        ]
    );
}

// ════════════════════════════════════════════════════════════════════════
// Table Completeness Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_supplemental_warning_category_table_contains_python_builtin_names() {
    let names: Vec<_> = SUPPLEMENTAL_EXCEPTION_CLASS_TABLE
        .iter()
        .map(|(name, _)| *name)
        .collect();

    assert!(names.contains(&"BytesWarning"));
    assert!(names.contains(&"ImportWarning"));
    assert!(names.contains(&"ResourceWarning"));
    assert!(names.contains(&"EncodingWarning"));
}

#[test]
fn test_supplemental_warning_categories_inherit_warning_proxy_class() {
    let warning_id =
        exception_proxy_class_id(ExceptionTypeId::Warning as u16).expect("Warning proxy");

    for name in [
        "BytesWarning",
        "FutureWarning",
        "ImportWarning",
        "ResourceWarning",
        "UnicodeWarning",
        "EncodingWarning",
    ] {
        let class = supplemental_exception_class(name)
            .unwrap_or_else(|| panic!("missing supplemental warning category {name}"));
        assert!(
            prism_runtime::object::type_builtins::issubclass(
                class.class_id(),
                warning_id,
                prism_runtime::object::type_builtins::global_class_bitmap,
            ),
            "{name} should inherit from Warning",
        );
    }
}

#[test]
fn test_supplemental_warning_categories_are_type_objects() {
    for name in [
        "BytesWarning",
        "FutureWarning",
        "ImportWarning",
        "ResourceWarning",
        "UnicodeWarning",
        "EncodingWarning",
    ] {
        let class = supplemental_exception_class(name)
            .unwrap_or_else(|| panic!("missing supplemental warning category {name}"));
        assert_eq!(
            class.header.type_id,
            TypeId::TYPE,
            "{name} should be a type"
        );
    }
}

#[test]
fn test_exception_type_table_length() {
    // 52 unique types + 2 aliases + some warnings
    assert!(EXCEPTION_TYPE_TABLE.len() >= 60);
}

#[test]
fn test_exception_type_table_has_base_exception() {
    assert!(get_exception_type("BaseException").is_some());
}

#[test]
fn test_exception_type_table_has_exception() {
    assert!(get_exception_type("Exception").is_some());
}

#[test]
fn test_exception_type_table_has_all_common_types() {
    let common_types = [
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "AttributeError",
        "NameError",
        "RuntimeError",
        "SystemError",
        "StopIteration",
        "OSError",
        "ZeroDivisionError",
    ];

    for name in common_types {
        assert!(
            get_exception_type(name).is_some(),
            "Missing exception type: {}",
            name
        );
    }
}

// ════════════════════════════════════════════════════════════════════════
// Display/Debug Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_display() {
    let exc_type = ExceptionTypeObject::new(ExceptionTypeId::ValueError, "ValueError");
    let display = format!("{}", exc_type);
    assert_eq!(display, "<class 'ValueError'>");
}

#[test]
fn test_debug() {
    let exc_type = ExceptionTypeObject::new(ExceptionTypeId::TypeError, "TypeError");
    let debug = format!("{:?}", exc_type);
    assert!(debug.contains("ExceptionTypeObject"));
    assert!(debug.contains("TypeError"));
}

// ════════════════════════════════════════════════════════════════════════
// Flags Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_exception_type_flags_combine() {
    let flags = ExceptionTypeFlags::ABSTRACT | ExceptionTypeFlags::CUSTOM_INIT;
    assert!(flags.contains(ExceptionTypeFlags::ABSTRACT));
    assert!(flags.contains(ExceptionTypeFlags::CUSTOM_INIT));
    assert!(!flags.contains(ExceptionTypeFlags::SYSTEM_EXIT));
}

// ════════════════════════════════════════════════════════════════════════
// EXCEPTION_TYPE_ID Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_exception_type_id_constant() {
    assert_eq!(EXCEPTION_TYPE_ID.0, 27);
}

#[test]
fn test_exception_type_object_uses_correct_type_id() {
    let exc_type = ExceptionTypeObject::new(ExceptionTypeId::ValueError, "ValueError");
    assert_eq!(exc_type.header.type_id, EXCEPTION_TYPE_ID);
}
