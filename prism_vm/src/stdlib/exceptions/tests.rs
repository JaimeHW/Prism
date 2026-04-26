use super::*;

// ════════════════════════════════════════════════════════════════════════
// Integration Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_type_id_and_flags_integration() {
    // Simulate creating a TypeError with args
    let type_id = ExceptionTypeId::TypeError;
    let flags = ExceptionFlags::new_exception()
        .set_normalized()
        .set_has_args();

    assert_eq!(type_id.name(), "TypeError");
    assert!(flags.is_normalized());
    assert!(flags.has_args());
}

#[test]
fn test_hierarchy_and_type_id_integration() {
    // Test that hierarchy functions work with type IDs
    let type_id = ExceptionTypeId::ConnectionRefusedError;

    // Using type's is_subclass_of method
    assert!(type_id.is_subclass_of(ExceptionTypeId::OSError));

    // Using hierarchy's is_subclass function
    assert!(is_subclass(type_id, ExceptionTypeId::OSError));
    assert!(is_subclass(type_id, ExceptionTypeId::Exception));
}

#[test]
fn test_exception_type_set_with_ids() {
    let mut set = ExceptionTypeSet::empty();
    set.insert(ExceptionTypeId::TypeError);
    set.insert(ExceptionTypeId::ValueError);

    assert!(set.contains(ExceptionTypeId::TypeError));
    assert!(set.contains(ExceptionTypeId::ValueError));
    assert!(!set.contains(ExceptionTypeId::KeyError));
}

#[test]
fn test_descendants_function() {
    let lookup_descendants = descendants(ExceptionTypeId::LookupError);

    assert!(lookup_descendants.contains(ExceptionTypeId::LookupError));
    assert!(lookup_descendants.contains(ExceptionTypeId::IndexError));
    assert!(lookup_descendants.contains(ExceptionTypeId::KeyError));
}

#[test]
fn test_common_ancestor_function() {
    let ancestor = common_ancestor(
        ExceptionTypeId::FileNotFoundError,
        ExceptionTypeId::PermissionError,
    );
    assert_eq!(ancestor, ExceptionTypeId::OSError);
}

#[test]
fn test_flyweight_flags() {
    let flags = ExceptionFlags::flyweight();

    // Flyweight exceptions are pre-normalized and marked as flyweight
    assert!(flags.is_flyweight());
    assert!(flags.is_normalized());
    assert!(!flags.has_args());
    assert!(!flags.has_traceback());
}

#[test]
fn test_control_flow_exceptions() {
    // Control flow exceptions should be detected
    assert!(ExceptionTypeId::StopIteration.is_control_flow());
    assert!(ExceptionTypeId::GeneratorExit.is_control_flow());

    // Regular exceptions are not control flow
    assert!(!ExceptionTypeId::TypeError.is_control_flow());
}

#[test]
fn test_exception_chaining_flags() {
    // Simulate `raise X from Y` pattern
    let flags = ExceptionFlags::new_exception()
        .set_normalized()
        .set_has_args()
        .set_has_cause()
        .set_suppress_context();

    assert!(flags.has_cause());
    assert!(flags.suppress_context());
    assert!(!flags.has_context()); // Only explicit cause, not implicit context
}

#[test]
fn test_memory_layout() {
    // Verify compact memory layout for performance
    assert_eq!(std::mem::size_of::<ExceptionTypeId>(), 1);
    assert_eq!(std::mem::size_of::<ExceptionFlags>(), 1);
    assert_eq!(std::mem::size_of::<ExceptionTypeSet>(), 8);

    // Combined header should be 2 bytes
    assert_eq!(
        std::mem::size_of::<ExceptionTypeId>() + std::mem::size_of::<ExceptionFlags>(),
        2
    );
}

// ════════════════════════════════════════════════════════════════════════
// Real-World Scenario Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_except_clause_matching() {
    // Simulate: except OSError as e:
    let raised = ExceptionTypeId::FileNotFoundError;
    let catch_type = ExceptionTypeId::OSError;

    // Check if we should catch this
    assert!(is_subclass(raised, catch_type));
}

#[test]
fn test_except_multiple_types() {
    // Simulate: except (TypeError, ValueError):
    let catch_types = ExceptionTypeSet::empty()
        .with(ExceptionTypeId::TypeError)
        .with(ExceptionTypeId::ValueError);

    // TypeError should match
    assert!(catch_types.contains(ExceptionTypeId::TypeError));

    // UnicodeError (subclass of ValueError) should NOT match directly
    // (Python requires matching the exact type or using issubclass)
    assert!(!catch_types.contains(ExceptionTypeId::UnicodeError));

    // But UnicodeError IS a subclass of ValueError
    assert!(is_subclass(
        ExceptionTypeId::UnicodeError,
        ExceptionTypeId::ValueError
    ));
}

#[test]
fn test_bare_except_catches_all() {
    // Simulate: except: (bare except)
    let catch_all = descendants(ExceptionTypeId::BaseException);

    // Should catch everything
    assert!(catch_all.contains(ExceptionTypeId::TypeError));
    assert!(catch_all.contains(ExceptionTypeId::KeyboardInterrupt));
    assert!(catch_all.contains(ExceptionTypeId::SystemExit));
}

#[test]
fn test_except_exception_skips_system() {
    // Simulate: except Exception:
    let catch_exception = descendants(ExceptionTypeId::Exception);

    // Should catch normal exceptions
    assert!(catch_exception.contains(ExceptionTypeId::TypeError));
    assert!(catch_exception.contains(ExceptionTypeId::ValueError));

    // Should NOT catch system-exiting exceptions
    assert!(!catch_exception.contains(ExceptionTypeId::SystemExit));
    assert!(!catch_exception.contains(ExceptionTypeId::KeyboardInterrupt));
    assert!(!catch_exception.contains(ExceptionTypeId::GeneratorExit));
}
