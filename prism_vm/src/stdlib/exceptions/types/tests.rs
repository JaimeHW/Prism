use super::*;

// ════════════════════════════════════════════════════════════════════════
// Basic Type ID Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_type_id_as_u8() {
    assert_eq!(ExceptionTypeId::BaseException.as_u8(), 0);
    assert_eq!(ExceptionTypeId::Exception.as_u8(), 4);
    assert_eq!(ExceptionTypeId::TypeError.as_u8(), 52);
    assert_eq!(ExceptionTypeId::ValueError.as_u8(), 53);
    assert_eq!(ExceptionTypeId::UserWarning.as_u8(), 63);
}

#[test]
fn test_type_id_from_u8_valid() {
    assert_eq!(
        ExceptionTypeId::from_u8(0),
        Some(ExceptionTypeId::BaseException)
    );
    assert_eq!(
        ExceptionTypeId::from_u8(4),
        Some(ExceptionTypeId::Exception)
    );
    assert_eq!(
        ExceptionTypeId::from_u8(52),
        Some(ExceptionTypeId::TypeError)
    );
    assert_eq!(
        ExceptionTypeId::from_u8(53),
        Some(ExceptionTypeId::ValueError)
    );
}

#[test]
fn test_type_id_from_u8_reserved() {
    // Reserved IDs should return None
    assert_eq!(ExceptionTypeId::from_u8(23), None);
    assert_eq!(ExceptionTypeId::from_u8(46), None);
    assert_eq!(ExceptionTypeId::from_u8(47), None);
}

#[test]
fn test_type_id_from_u8_invalid() {
    // IDs beyond 63 are not built-in
    assert_eq!(ExceptionTypeId::from_u8(64), None);
    assert_eq!(ExceptionTypeId::from_u8(100), None);
    assert_eq!(ExceptionTypeId::from_u8(128), None);
    assert_eq!(ExceptionTypeId::from_u8(255), None);
}

#[test]
fn test_type_id_roundtrip() {
    // All valid IDs should roundtrip through as_u8/from_u8
    for id in 0..=63u8 {
        if let Some(type_id) = ExceptionTypeId::from_u8(id) {
            assert_eq!(type_id.as_u8(), id);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Name Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_type_id_names() {
    assert_eq!(ExceptionTypeId::BaseException.name(), "BaseException");
    assert_eq!(ExceptionTypeId::Exception.name(), "Exception");
    assert_eq!(ExceptionTypeId::TypeError.name(), "TypeError");
    assert_eq!(ExceptionTypeId::ValueError.name(), "ValueError");
    assert_eq!(ExceptionTypeId::StopIteration.name(), "StopIteration");
    assert_eq!(ExceptionTypeId::KeyError.name(), "KeyError");
    assert_eq!(ExceptionTypeId::IndexError.name(), "IndexError");
    assert_eq!(
        ExceptionTypeId::BaseExceptionGroup.name(),
        "BaseExceptionGroup"
    );
    assert_eq!(ExceptionTypeId::ExceptionGroup.name(), "ExceptionGroup");
    assert_eq!(
        ExceptionTypeId::ZeroDivisionError.name(),
        "ZeroDivisionError"
    );
}

#[test]
fn test_type_id_display() {
    assert_eq!(format!("{}", ExceptionTypeId::TypeError), "TypeError");
    assert_eq!(format!("{}", ExceptionTypeId::ValueError), "ValueError");
    assert_eq!(format!("{}", ExceptionTypeId::OSError), "OSError");
}

#[test]
fn test_all_names_non_empty() {
    for id in 0..=63u8 {
        if let Some(type_id) = ExceptionTypeId::from_u8(id) {
            assert!(!type_id.name().is_empty(), "ID {} has empty name", id);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Hierarchy Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_base_exception_has_no_parent() {
    assert_eq!(ExceptionTypeId::BaseException.parent(), None);
}

#[test]
fn test_exception_parent_is_base_exception() {
    assert_eq!(
        ExceptionTypeId::Exception.parent(),
        Some(ExceptionTypeId::BaseException)
    );
}

#[test]
fn test_type_error_parent_chain() {
    // TypeError → Exception → BaseException → None
    let type_error = ExceptionTypeId::TypeError;
    assert_eq!(type_error.parent(), Some(ExceptionTypeId::Exception));
    assert_eq!(
        type_error.parent().unwrap().parent(),
        Some(ExceptionTypeId::BaseException)
    );
    assert_eq!(
        type_error.parent().unwrap().parent().unwrap().parent(),
        None
    );
}

#[test]
fn test_tab_error_parent_chain() {
    // TabError → IndentationError → SyntaxError → Exception → BaseException
    let tab_error = ExceptionTypeId::TabError;
    assert_eq!(tab_error.parent(), Some(ExceptionTypeId::IndentationError));
    assert_eq!(
        tab_error.parent().unwrap().parent(),
        Some(ExceptionTypeId::SyntaxError)
    );
    assert_eq!(
        tab_error.parent().unwrap().parent().unwrap().parent(),
        Some(ExceptionTypeId::Exception)
    );
}

#[test]
fn test_exception_group_parent_chain() {
    assert_eq!(
        ExceptionTypeId::BaseExceptionGroup.parent(),
        Some(ExceptionTypeId::BaseException)
    );
    assert_eq!(
        ExceptionTypeId::ExceptionGroup.parent(),
        Some(ExceptionTypeId::BaseExceptionGroup)
    );
    assert_eq!(
        ExceptionTypeId::ExceptionGroup.secondary_parent(),
        Some(ExceptionTypeId::Exception)
    );
}

#[test]
fn test_os_error_hierarchy() {
    // FileNotFoundError → OSError → Exception → BaseException
    assert_eq!(
        ExceptionTypeId::FileNotFoundError.parent(),
        Some(ExceptionTypeId::OSError)
    );
    assert_eq!(
        ExceptionTypeId::OSError.parent(),
        Some(ExceptionTypeId::Exception)
    );
}

#[test]
fn test_connection_error_hierarchy() {
    // ConnectionRefusedError → ConnectionError → OSError → Exception
    assert_eq!(
        ExceptionTypeId::ConnectionRefusedError.parent(),
        Some(ExceptionTypeId::ConnectionError)
    );
    assert_eq!(
        ExceptionTypeId::ConnectionError.parent(),
        Some(ExceptionTypeId::OSError)
    );
}

#[test]
fn test_arithmetic_error_hierarchy() {
    // ZeroDivisionError → ArithmeticError → Exception → BaseException
    assert_eq!(
        ExceptionTypeId::ZeroDivisionError.parent(),
        Some(ExceptionTypeId::ArithmeticError)
    );
    assert_eq!(
        ExceptionTypeId::ArithmeticError.parent(),
        Some(ExceptionTypeId::Exception)
    );
}

#[test]
fn test_unicode_error_hierarchy() {
    // UnicodeDecodeError → UnicodeError → ValueError → Exception
    assert_eq!(
        ExceptionTypeId::UnicodeDecodeError.parent(),
        Some(ExceptionTypeId::UnicodeError)
    );
    assert_eq!(
        ExceptionTypeId::UnicodeError.parent(),
        Some(ExceptionTypeId::ValueError)
    );
    assert_eq!(
        ExceptionTypeId::ValueError.parent(),
        Some(ExceptionTypeId::Exception)
    );
}

#[test]
fn test_all_types_have_valid_parent_chain() {
    // Every type should eventually reach BaseException
    for id in 0..=63u8 {
        if let Some(type_id) = ExceptionTypeId::from_u8(id) {
            let mut current = type_id;
            let mut depth = 0;
            while let Some(parent) = current.parent() {
                current = parent;
                depth += 1;
                assert!(
                    depth <= 10,
                    "Infinite loop in parent chain for {:?}",
                    type_id
                );
            }
            // Should always end at BaseException (which has no parent)
            assert_eq!(
                current,
                ExceptionTypeId::BaseException,
                "Type {:?} doesn't trace back to BaseException",
                type_id
            );
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Subclass Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_is_subclass_of_self() {
    assert!(ExceptionTypeId::TypeError.is_subclass_of(ExceptionTypeId::TypeError));
    assert!(ExceptionTypeId::Exception.is_subclass_of(ExceptionTypeId::Exception));
    assert!(ExceptionTypeId::BaseException.is_subclass_of(ExceptionTypeId::BaseException));
}

#[test]
fn test_is_subclass_of_direct_parent() {
    assert!(ExceptionTypeId::TypeError.is_subclass_of(ExceptionTypeId::Exception));
    assert!(ExceptionTypeId::Exception.is_subclass_of(ExceptionTypeId::BaseException));
    assert!(ExceptionTypeId::IndexError.is_subclass_of(ExceptionTypeId::LookupError));
}

#[test]
fn test_is_subclass_of_indirect_parent() {
    // TypeError is subclass of BaseException (via Exception)
    assert!(ExceptionTypeId::TypeError.is_subclass_of(ExceptionTypeId::BaseException));

    // TabError is subclass of Exception (via Indentation → Syntax → Exception)
    assert!(ExceptionTypeId::TabError.is_subclass_of(ExceptionTypeId::Exception));

    // ConnectionRefusedError is subclass of Exception (via Connection → OS → Exception)
    assert!(ExceptionTypeId::ConnectionRefusedError.is_subclass_of(ExceptionTypeId::Exception));
}

#[test]
fn test_exception_group_multiple_inheritance_relationships() {
    assert!(ExceptionTypeId::ExceptionGroup.is_subclass_of(ExceptionTypeId::ExceptionGroup));
    assert!(ExceptionTypeId::ExceptionGroup.is_subclass_of(ExceptionTypeId::Exception));
    assert!(ExceptionTypeId::ExceptionGroup.is_subclass_of(ExceptionTypeId::BaseExceptionGroup));
    assert!(ExceptionTypeId::ExceptionGroup.is_subclass_of(ExceptionTypeId::BaseException));
    assert!(!ExceptionTypeId::BaseExceptionGroup.is_subclass_of(ExceptionTypeId::Exception));
}

#[test]
fn test_is_not_subclass() {
    // TypeError is not a subclass of ValueError
    assert!(!ExceptionTypeId::TypeError.is_subclass_of(ExceptionTypeId::ValueError));

    // OSError is not a subclass of TypeError
    assert!(!ExceptionTypeId::OSError.is_subclass_of(ExceptionTypeId::TypeError));

    // Exception is not a subclass of TypeError (parent, not child)
    assert!(!ExceptionTypeId::Exception.is_subclass_of(ExceptionTypeId::TypeError));
}

#[test]
fn test_base_exception_is_not_subclass_of_exception() {
    // BaseException is NOT a subclass of Exception
    assert!(!ExceptionTypeId::BaseException.is_subclass_of(ExceptionTypeId::Exception));
}

#[test]
fn test_system_exit_is_not_subclass_of_exception() {
    // SystemExit is a direct child of BaseException, not Exception
    assert!(!ExceptionTypeId::SystemExit.is_subclass_of(ExceptionTypeId::Exception));
    assert!(ExceptionTypeId::SystemExit.is_subclass_of(ExceptionTypeId::BaseException));
}

// ════════════════════════════════════════════════════════════════════════
// Depth Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_depth_base_exception() {
    assert_eq!(ExceptionTypeId::BaseException.depth(), 0);
}

#[test]
fn test_depth_exception() {
    assert_eq!(ExceptionTypeId::Exception.depth(), 1);
}

#[test]
fn test_depth_type_error() {
    // TypeError → Exception → BaseException = depth 2
    assert_eq!(ExceptionTypeId::TypeError.depth(), 2);
}

#[test]
fn test_depth_tab_error() {
    // TabError → Indentation → Syntax → Exception → BaseException = depth 4
    assert_eq!(ExceptionTypeId::TabError.depth(), 4);
}

#[test]
fn test_depth_connection_refused() {
    // ConnectionRefusedError → ConnectionError → OSError → Exception → BaseException = depth 4
    assert_eq!(ExceptionTypeId::ConnectionRefusedError.depth(), 4);
}

#[test]
fn test_depth_exception_group_uses_shortest_builtin_chain() {
    assert_eq!(ExceptionTypeId::BaseExceptionGroup.depth(), 1);
    assert_eq!(ExceptionTypeId::ExceptionGroup.depth(), 2);
}

#[test]
fn test_max_depth_is_bounded() {
    // Python's exception hierarchy has max depth 4
    for id in 0..=63u8 {
        if let Some(type_id) = ExceptionTypeId::from_u8(id) {
            assert!(
                type_id.depth() <= 5,
                "Depth of {:?} is too deep: {}",
                type_id,
                type_id.depth()
            );
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Classification Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_is_control_flow() {
    assert!(ExceptionTypeId::StopIteration.is_control_flow());
    assert!(ExceptionTypeId::StopAsyncIteration.is_control_flow());
    assert!(ExceptionTypeId::GeneratorExit.is_control_flow());

    assert!(!ExceptionTypeId::TypeError.is_control_flow());
    assert!(!ExceptionTypeId::KeyboardInterrupt.is_control_flow());
}

#[test]
fn test_is_system_exit() {
    assert!(ExceptionTypeId::SystemExit.is_system_exit());
    assert!(ExceptionTypeId::KeyboardInterrupt.is_system_exit());

    assert!(!ExceptionTypeId::Exception.is_system_exit());
    assert!(!ExceptionTypeId::GeneratorExit.is_system_exit());
}

#[test]
fn test_is_warning() {
    assert!(ExceptionTypeId::Warning.is_warning());
    assert!(ExceptionTypeId::DeprecationWarning.is_warning());
    assert!(ExceptionTypeId::RuntimeWarning.is_warning());
    assert!(ExceptionTypeId::SyntaxWarning.is_warning());
    assert!(ExceptionTypeId::UserWarning.is_warning());
    assert!(ExceptionTypeId::PendingDeprecationWarning.is_warning());

    assert!(!ExceptionTypeId::Exception.is_warning());
    assert!(!ExceptionTypeId::TypeError.is_warning());
}

#[test]
fn test_is_os_error() {
    assert!(ExceptionTypeId::OSError.is_os_error());
    assert!(ExceptionTypeId::FileNotFoundError.is_os_error());
    assert!(ExceptionTypeId::PermissionError.is_os_error());
    assert!(ExceptionTypeId::ConnectionError.is_os_error());
    assert!(ExceptionTypeId::TimeoutError.is_os_error());

    assert!(!ExceptionTypeId::TypeError.is_os_error());
    assert!(!ExceptionTypeId::ValueError.is_os_error());
}

// ════════════════════════════════════════════════════════════════════════
// Comparison and Ordering Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_type_id_ordering() {
    assert!(ExceptionTypeId::BaseException < ExceptionTypeId::Exception);
    assert!(ExceptionTypeId::Exception < ExceptionTypeId::TypeError);
}

// ════════════════════════════════════════════════════════════════════════
// Constants Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_count_constant() {
    assert_eq!(ExceptionTypeId::COUNT, 64);
}

#[test]
fn test_user_defined_start() {
    assert_eq!(ExceptionTypeId::USER_DEFINED_START, 128);
}

#[test]
fn test_max_id() {
    assert_eq!(ExceptionTypeId::MAX_ID, 255);
}

// ════════════════════════════════════════════════════════════════════════
// Default Tests
// ════════════════════════════════════════════════════════════════════════

// ════════════════════════════════════════════════════════════════════════
// Debug Tests
// ════════════════════════════════════════════════════════════════════════

// ════════════════════════════════════════════════════════════════════════
// Size Tests (Performance Verification)
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_size_is_one_byte() {
    // Critical for JIT performance: must be exactly 1 byte
    assert_eq!(std::mem::size_of::<ExceptionTypeId>(), 1);
}

#[test]
fn test_alignment_is_one_byte() {
    // Should have single-byte alignment for optimal packing
    assert_eq!(std::mem::align_of::<ExceptionTypeId>(), 1);
}

#[test]
fn test_option_size() {
    // Option<ExceptionTypeId> should use niche optimization
    // Can't guarantee this, but it's worth checking
    assert!(std::mem::size_of::<Option<ExceptionTypeId>>() <= 2);
}
