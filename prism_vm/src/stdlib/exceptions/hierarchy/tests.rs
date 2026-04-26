use super::*;

// ════════════════════════════════════════════════════════════════════════
// ExceptionTypeSet Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_empty_set() {
    let set = ExceptionTypeSet::empty();
    assert!(set.is_empty());
    assert_eq!(set.len(), 0);
}

#[test]
fn test_all_set() {
    let set = ExceptionTypeSet::all();
    assert!(!set.is_empty());
    assert_eq!(set.len(), 64);
}

#[test]
fn test_singleton() {
    let set = ExceptionTypeSet::singleton(ExceptionTypeId::TypeError);
    assert!(set.contains(ExceptionTypeId::TypeError));
    assert!(!set.contains(ExceptionTypeId::ValueError));
    assert_eq!(set.len(), 1);
}

#[test]
fn test_contains() {
    let set =
        ExceptionTypeSet::singleton(ExceptionTypeId::TypeError).with(ExceptionTypeId::ValueError);

    assert!(set.contains(ExceptionTypeId::TypeError));
    assert!(set.contains(ExceptionTypeId::ValueError));
    assert!(!set.contains(ExceptionTypeId::KeyError));
}

#[test]
fn test_union() {
    let a = ExceptionTypeSet::singleton(ExceptionTypeId::TypeError);
    let b = ExceptionTypeSet::singleton(ExceptionTypeId::ValueError);
    let c = a.union(b);

    assert!(c.contains(ExceptionTypeId::TypeError));
    assert!(c.contains(ExceptionTypeId::ValueError));
    assert_eq!(c.len(), 2);
}

#[test]
fn test_intersection() {
    let a =
        ExceptionTypeSet::singleton(ExceptionTypeId::TypeError).with(ExceptionTypeId::ValueError);
    let b =
        ExceptionTypeSet::singleton(ExceptionTypeId::ValueError).with(ExceptionTypeId::KeyError);
    let c = a.intersection(b);

    assert!(!c.contains(ExceptionTypeId::TypeError));
    assert!(c.contains(ExceptionTypeId::ValueError));
    assert!(!c.contains(ExceptionTypeId::KeyError));
    assert_eq!(c.len(), 1);
}

#[test]
fn test_with() {
    let set = ExceptionTypeSet::empty()
        .with(ExceptionTypeId::TypeError)
        .with(ExceptionTypeId::ValueError);

    assert!(set.contains(ExceptionTypeId::TypeError));
    assert!(set.contains(ExceptionTypeId::ValueError));
}

#[test]
fn test_without() {
    let set = ExceptionTypeSet::singleton(ExceptionTypeId::TypeError)
        .with(ExceptionTypeId::ValueError)
        .without(ExceptionTypeId::TypeError);

    assert!(!set.contains(ExceptionTypeId::TypeError));
    assert!(set.contains(ExceptionTypeId::ValueError));
}

#[test]
fn test_insert_remove() {
    let mut set = ExceptionTypeSet::empty();
    set.insert(ExceptionTypeId::TypeError);
    assert!(set.contains(ExceptionTypeId::TypeError));

    set.remove(ExceptionTypeId::TypeError);
    assert!(!set.contains(ExceptionTypeId::TypeError));
}

#[test]
fn test_set_size() {
    // u64 is 8 bytes
    assert_eq!(std::mem::size_of::<ExceptionTypeSet>(), 8);
}

// ════════════════════════════════════════════════════════════════════════
// Descendants Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_base_exception_descendants_contain_all() {
    let desc = descendants(ExceptionTypeId::BaseException);

    // All exception types should be descendants of BaseException
    assert!(desc.contains(ExceptionTypeId::Exception));
    assert!(desc.contains(ExceptionTypeId::TypeError));
    assert!(desc.contains(ExceptionTypeId::SystemExit));
    assert!(desc.contains(ExceptionTypeId::KeyboardInterrupt));
}

#[test]
fn test_exception_descendants() {
    let desc = descendants(ExceptionTypeId::Exception);

    // Should contain Exception subclasses
    assert!(desc.contains(ExceptionTypeId::TypeError));
    assert!(desc.contains(ExceptionTypeId::ValueError));
    assert!(desc.contains(ExceptionTypeId::KeyError));

    // Should NOT contain non-Exception types
    assert!(!desc.contains(ExceptionTypeId::BaseException));
    assert!(!desc.contains(ExceptionTypeId::SystemExit));
    assert!(!desc.contains(ExceptionTypeId::KeyboardInterrupt));
    assert!(!desc.contains(ExceptionTypeId::GeneratorExit));
}

#[test]
fn test_lookup_error_descendants() {
    let desc = descendants(ExceptionTypeId::LookupError);

    assert!(desc.contains(ExceptionTypeId::LookupError));
    assert!(desc.contains(ExceptionTypeId::IndexError));
    assert!(desc.contains(ExceptionTypeId::KeyError));

    assert!(!desc.contains(ExceptionTypeId::TypeError));
    assert!(!desc.contains(ExceptionTypeId::ValueError));
}

#[test]
fn test_arithmetic_error_descendants() {
    let desc = descendants(ExceptionTypeId::ArithmeticError);

    assert!(desc.contains(ExceptionTypeId::ArithmeticError));
    assert!(desc.contains(ExceptionTypeId::OverflowError));
    assert!(desc.contains(ExceptionTypeId::ZeroDivisionError));
    assert!(desc.contains(ExceptionTypeId::FloatingPointError));

    assert!(!desc.contains(ExceptionTypeId::TypeError));
}

#[test]
fn test_os_error_descendants() {
    let desc = descendants(ExceptionTypeId::OSError);

    assert!(desc.contains(ExceptionTypeId::OSError));
    assert!(desc.contains(ExceptionTypeId::FileNotFoundError));
    assert!(desc.contains(ExceptionTypeId::PermissionError));
    assert!(desc.contains(ExceptionTypeId::ConnectionError));
    assert!(desc.contains(ExceptionTypeId::ConnectionRefusedError));

    assert!(!desc.contains(ExceptionTypeId::TypeError));
}

#[test]
fn test_syntax_error_descendants() {
    let desc = descendants(ExceptionTypeId::SyntaxError);

    assert!(desc.contains(ExceptionTypeId::SyntaxError));
    assert!(desc.contains(ExceptionTypeId::IndentationError));
    assert!(desc.contains(ExceptionTypeId::TabError));

    assert!(!desc.contains(ExceptionTypeId::TypeError));
}

#[test]
fn test_value_error_descendants() {
    let desc = descendants(ExceptionTypeId::ValueError);

    assert!(desc.contains(ExceptionTypeId::ValueError));
    assert!(desc.contains(ExceptionTypeId::UnicodeError));
    assert!(desc.contains(ExceptionTypeId::UnicodeDecodeError));
    assert!(desc.contains(ExceptionTypeId::UnicodeEncodeError));

    assert!(!desc.contains(ExceptionTypeId::TypeError));
}

#[test]
fn test_warning_descendants() {
    let desc = descendants(ExceptionTypeId::Warning);

    assert!(desc.contains(ExceptionTypeId::Warning));
    assert!(desc.contains(ExceptionTypeId::DeprecationWarning));
    assert!(desc.contains(ExceptionTypeId::RuntimeWarning));
    assert!(desc.contains(ExceptionTypeId::SyntaxWarning));

    assert!(!desc.contains(ExceptionTypeId::TypeError));
}

#[test]
fn test_base_exception_group_descendants() {
    let desc = descendants(ExceptionTypeId::BaseExceptionGroup);

    assert!(desc.contains(ExceptionTypeId::BaseExceptionGroup));
    assert!(desc.contains(ExceptionTypeId::ExceptionGroup));
    assert!(!desc.contains(ExceptionTypeId::TypeError));
}

#[test]
fn test_leaf_type_descendants() {
    // Leaf types should only contain themselves
    let desc = descendants(ExceptionTypeId::TypeError);
    assert!(desc.contains(ExceptionTypeId::TypeError));
    assert_eq!(desc.len(), 1);

    let desc = descendants(ExceptionTypeId::KeyError);
    assert!(desc.contains(ExceptionTypeId::KeyError));
    assert_eq!(desc.len(), 1);
}

// ════════════════════════════════════════════════════════════════════════
// is_subclass Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_is_subclass_self() {
    assert!(is_subclass(
        ExceptionTypeId::TypeError,
        ExceptionTypeId::TypeError
    ));
    assert!(is_subclass(
        ExceptionTypeId::Exception,
        ExceptionTypeId::Exception
    ));
}

#[test]
fn test_is_subclass_exception_group() {
    assert!(is_subclass(
        ExceptionTypeId::ExceptionGroup,
        ExceptionTypeId::BaseExceptionGroup
    ));
    assert!(is_subclass(
        ExceptionTypeId::ExceptionGroup,
        ExceptionTypeId::Exception
    ));
}

#[test]
fn test_is_subclass_direct_parent() {
    assert!(is_subclass(
        ExceptionTypeId::TypeError,
        ExceptionTypeId::Exception
    ));
    assert!(is_subclass(
        ExceptionTypeId::IndexError,
        ExceptionTypeId::LookupError
    ));
}

#[test]
fn test_is_subclass_indirect_parent() {
    assert!(is_subclass(
        ExceptionTypeId::TypeError,
        ExceptionTypeId::BaseException
    ));
    assert!(is_subclass(
        ExceptionTypeId::TabError,
        ExceptionTypeId::Exception
    ));
    assert!(is_subclass(
        ExceptionTypeId::ConnectionRefusedError,
        ExceptionTypeId::Exception
    ));
}

#[test]
fn test_is_not_subclass() {
    assert!(!is_subclass(
        ExceptionTypeId::TypeError,
        ExceptionTypeId::ValueError
    ));
    assert!(!is_subclass(
        ExceptionTypeId::Exception,
        ExceptionTypeId::TypeError
    ));
    assert!(!is_subclass(
        ExceptionTypeId::SystemExit,
        ExceptionTypeId::Exception
    ));
}

// ════════════════════════════════════════════════════════════════════════
// common_ancestor Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_common_ancestor_same_type() {
    let ancestor = common_ancestor(ExceptionTypeId::TypeError, ExceptionTypeId::TypeError);
    assert_eq!(ancestor, ExceptionTypeId::TypeError);
}

#[test]
fn test_common_ancestor_parent_child() {
    let ancestor = common_ancestor(ExceptionTypeId::TypeError, ExceptionTypeId::Exception);
    assert_eq!(ancestor, ExceptionTypeId::Exception);

    let ancestor = common_ancestor(ExceptionTypeId::Exception, ExceptionTypeId::TypeError);
    assert_eq!(ancestor, ExceptionTypeId::Exception);
}

#[test]
fn test_common_ancestor_siblings() {
    let ancestor = common_ancestor(ExceptionTypeId::TypeError, ExceptionTypeId::ValueError);
    assert_eq!(ancestor, ExceptionTypeId::Exception);
}

#[test]
fn test_common_ancestor_cousins() {
    let ancestor = common_ancestor(ExceptionTypeId::IndexError, ExceptionTypeId::KeyError);
    assert_eq!(ancestor, ExceptionTypeId::LookupError);
}

#[test]
fn test_common_ancestor_distant() {
    let ancestor = common_ancestor(
        ExceptionTypeId::TabError,
        ExceptionTypeId::ConnectionRefusedError,
    );
    assert_eq!(ancestor, ExceptionTypeId::Exception);
}

#[test]
fn test_common_ancestor_system_exit() {
    // SystemExit and TypeError only share BaseException
    let ancestor = common_ancestor(ExceptionTypeId::SystemExit, ExceptionTypeId::TypeError);
    assert_eq!(ancestor, ExceptionTypeId::BaseException);
}

// ════════════════════════════════════════════════════════════════════════
// Debug Format Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_debug_empty_set() {
    let set = ExceptionTypeSet::empty();
    let debug = format!("{:?}", set);
    assert_eq!(debug, "{}");
}

#[test]
fn test_debug_singleton_set() {
    let set = ExceptionTypeSet::singleton(ExceptionTypeId::TypeError);
    let debug = format!("{:?}", set);
    assert!(debug.contains("TypeError"));
}
