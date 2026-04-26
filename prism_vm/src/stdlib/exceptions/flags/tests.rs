use super::*;

// ════════════════════════════════════════════════════════════════════════
// Constructor Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_empty_flags() {
    let flags = ExceptionFlags::empty();
    assert!(flags.is_empty());
    assert_eq!(flags.as_raw(), 0);
}

#[test]
fn test_all_flags() {
    let flags = ExceptionFlags::all();
    assert!(!flags.is_empty());
    assert_eq!(flags.as_raw(), 0xFF);
}

#[test]
fn test_from_raw() {
    let flags = ExceptionFlags::from_raw(0b00000011);
    assert!(flags.is_normalized());
    assert!(flags.has_args());
    assert!(!flags.has_traceback());
}

#[test]
fn test_flyweight_preset() {
    let flags = ExceptionFlags::flyweight();
    assert!(flags.is_flyweight());
    assert!(flags.is_normalized());
    assert!(!flags.has_args());
}

#[test]
fn test_new_exception_preset() {
    let flags = ExceptionFlags::new_exception();
    assert!(flags.is_empty());
    assert!(!flags.is_normalized());
}

#[test]
fn test_with_args_preset() {
    let flags = ExceptionFlags::with_args();
    assert!(flags.has_args());
    assert!(flags.is_normalized());
}

// ════════════════════════════════════════════════════════════════════════
// Individual Flag Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_normalized_flag() {
    let flags = ExceptionFlags::empty().set_normalized();
    assert!(flags.is_normalized());
    assert!(!flags.has_args());
}

#[test]
fn test_has_args_flag() {
    let flags = ExceptionFlags::empty().set_has_args();
    assert!(flags.has_args());
    assert!(!flags.is_normalized());
}

#[test]
fn test_has_traceback_flag() {
    let flags = ExceptionFlags::empty().set_has_traceback();
    assert!(flags.has_traceback());
    assert!(!flags.has_args());
}

#[test]
fn test_has_cause_flag() {
    let flags = ExceptionFlags::empty().set_has_cause();
    assert!(flags.has_cause());
    assert!(!flags.has_context());
}

#[test]
fn test_has_context_flag() {
    let flags = ExceptionFlags::empty().set_has_context();
    assert!(flags.has_context());
    assert!(!flags.has_cause());
}

#[test]
fn test_suppress_context_flag() {
    let flags = ExceptionFlags::empty().set_suppress_context();
    assert!(flags.suppress_context());
}

#[test]
fn test_flyweight_flag() {
    let flags = ExceptionFlags::empty().set_flyweight();
    assert!(flags.is_flyweight());
}

#[test]
fn test_reraised_flag() {
    let flags = ExceptionFlags::empty().set_reraised();
    assert!(flags.was_reraised());
}

// ════════════════════════════════════════════════════════════════════════
// Clear Flag Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_clear_has_traceback() {
    let flags = ExceptionFlags::all().clear_has_traceback();
    assert!(!flags.has_traceback());
    assert!(flags.has_args()); // Other flags preserved
}

#[test]
fn test_clear_has_cause() {
    let flags = ExceptionFlags::all().clear_has_cause();
    assert!(!flags.has_cause());
    assert!(flags.has_context()); // Other flags preserved
}

#[test]
fn test_clear_has_context() {
    let flags = ExceptionFlags::all().clear_has_context();
    assert!(!flags.has_context());
    assert!(flags.has_cause()); // Other flags preserved
}

#[test]
fn test_clear_suppress_context() {
    let flags = ExceptionFlags::all().clear_suppress_context();
    assert!(!flags.suppress_context());
}

#[test]
fn test_clear_reraised() {
    let flags = ExceptionFlags::all().clear_reraised();
    assert!(!flags.was_reraised());
}

// ════════════════════════════════════════════════════════════════════════
// Mutation Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_mark_normalized() {
    let mut flags = ExceptionFlags::empty();
    flags.mark_normalized();
    assert!(flags.is_normalized());
}

#[test]
fn test_mark_has_args() {
    let mut flags = ExceptionFlags::empty();
    flags.mark_has_args();
    assert!(flags.has_args());
}

#[test]
fn test_mark_has_traceback() {
    let mut flags = ExceptionFlags::empty();
    flags.mark_has_traceback();
    assert!(flags.has_traceback());
}

#[test]
fn test_mark_has_cause() {
    let mut flags = ExceptionFlags::empty();
    flags.mark_has_cause();
    assert!(flags.has_cause());
}

#[test]
fn test_mark_has_context() {
    let mut flags = ExceptionFlags::empty();
    flags.mark_has_context();
    assert!(flags.has_context());
}

#[test]
fn test_mark_suppress_context() {
    let mut flags = ExceptionFlags::empty();
    flags.mark_suppress_context();
    assert!(flags.suppress_context());
}

#[test]
fn test_mark_reraised() {
    let mut flags = ExceptionFlags::empty();
    flags.mark_reraised();
    assert!(flags.was_reraised());
}

// ════════════════════════════════════════════════════════════════════════
// Combination Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_multiple_flags() {
    let flags = ExceptionFlags::empty()
        .set_normalized()
        .set_has_args()
        .set_has_traceback();

    assert!(flags.is_normalized());
    assert!(flags.has_args());
    assert!(flags.has_traceback());
    assert!(!flags.has_cause());
}

#[test]
fn test_chained_exception_flags() {
    // Simulating a chained exception
    let flags = ExceptionFlags::empty()
        .set_normalized()
        .set_has_args()
        .set_has_context()
        .set_has_traceback();

    assert!(flags.is_normalized());
    assert!(flags.has_context());
    assert!(!flags.has_cause()); // Implicit chaining only
}

#[test]
fn test_explicit_chained_exception() {
    // Simulating `raise X from Y`
    let flags = ExceptionFlags::empty()
        .set_normalized()
        .set_has_args()
        .set_has_cause()
        .set_suppress_context();

    assert!(flags.has_cause());
    assert!(flags.suppress_context());
}

// ════════════════════════════════════════════════════════════════════════
// Bitwise Operation Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_with_operation() {
    let a = ExceptionFlags::empty().set_normalized();
    let b = ExceptionFlags::empty().set_has_args();
    let c = a.with(b);

    assert!(c.is_normalized());
    assert!(c.has_args());
}

#[test]
fn test_without_operation() {
    let flags = ExceptionFlags::all();
    let mask = ExceptionFlags::from_raw(ExceptionFlags::HAS_ARGS | ExceptionFlags::HAS_TRACEBACK);
    let result = flags.without(mask);

    assert!(!result.has_args());
    assert!(!result.has_traceback());
    assert!(result.is_normalized()); // Other flags preserved
}

#[test]
fn test_contains() {
    let flags = ExceptionFlags::empty()
        .set_normalized()
        .set_has_args()
        .set_has_traceback();

    let mask = ExceptionFlags::empty().set_normalized().set_has_args();

    assert!(flags.contains(mask));
}

#[test]
fn test_contains_fails() {
    let flags = ExceptionFlags::empty().set_normalized();
    let mask = ExceptionFlags::empty().set_normalized().set_has_args();

    assert!(!flags.contains(mask));
}

#[test]
fn test_intersects() {
    let a = ExceptionFlags::empty().set_normalized().set_has_args();
    let b = ExceptionFlags::empty().set_has_args().set_has_traceback();

    assert!(a.intersects(b)); // Both have HAS_ARGS
}

#[test]
fn test_intersects_fails() {
    let a = ExceptionFlags::empty().set_normalized();
    let b = ExceptionFlags::empty().set_has_args();

    assert!(!a.intersects(b)); // No common flags
}

#[test]
fn test_count_ones() {
    assert_eq!(ExceptionFlags::empty().count_ones(), 0);
    assert_eq!(ExceptionFlags::empty().set_normalized().count_ones(), 1);
    assert_eq!(
        ExceptionFlags::empty()
            .set_normalized()
            .set_has_args()
            .count_ones(),
        2
    );
    assert_eq!(ExceptionFlags::all().count_ones(), 8);
}

// ════════════════════════════════════════════════════════════════════════
// Operator Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_bitor_operator() {
    let a = ExceptionFlags::from_raw(0b00000001);
    let b = ExceptionFlags::from_raw(0b00000010);
    let c = a | b;

    assert_eq!(c.as_raw(), 0b00000011);
}

#[test]
fn test_bitor_assign_operator() {
    let mut flags = ExceptionFlags::from_raw(0b00000001);
    flags |= ExceptionFlags::from_raw(0b00000010);

    assert_eq!(flags.as_raw(), 0b00000011);
}

#[test]
fn test_bitand_operator() {
    let a = ExceptionFlags::from_raw(0b00000011);
    let b = ExceptionFlags::from_raw(0b00000010);
    let c = a & b;

    assert_eq!(c.as_raw(), 0b00000010);
}

#[test]
fn test_bitand_assign_operator() {
    let mut flags = ExceptionFlags::from_raw(0b00000011);
    flags &= ExceptionFlags::from_raw(0b00000010);

    assert_eq!(flags.as_raw(), 0b00000010);
}

#[test]
fn test_not_operator() {
    let flags = ExceptionFlags::from_raw(0b00001111);
    let inverted = !flags;

    assert_eq!(inverted.as_raw(), 0b11110000);
}

// ════════════════════════════════════════════════════════════════════════
// Debug Format Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_debug_empty() {
    let flags = ExceptionFlags::empty();
    let debug = format!("{:?}", flags);
    assert_eq!(debug, "ExceptionFlags(empty)");
}

#[test]
fn test_debug_single_flag() {
    let flags = ExceptionFlags::empty().set_normalized();
    let debug = format!("{:?}", flags);
    assert_eq!(debug, "ExceptionFlags(NORMALIZED)");
}

#[test]
fn test_debug_multiple_flags() {
    let flags = ExceptionFlags::empty().set_normalized().set_has_args();
    let debug = format!("{:?}", flags);
    assert_eq!(debug, "ExceptionFlags(NORMALIZED | HAS_ARGS)");
}

// ════════════════════════════════════════════════════════════════════════
// Trait Tests
// ════════════════════════════════════════════════════════════════════════

// ════════════════════════════════════════════════════════════════════════
// Size Tests (Performance Verification)
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_size_is_one_byte() {
    assert_eq!(std::mem::size_of::<ExceptionFlags>(), 1);
}

#[test]
fn test_alignment_is_one_byte() {
    assert_eq!(std::mem::align_of::<ExceptionFlags>(), 1);
}

// ════════════════════════════════════════════════════════════════════════
// Edge Case Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_set_same_flag_twice() {
    let flags = ExceptionFlags::empty().set_normalized().set_normalized();

    assert!(flags.is_normalized());
    assert_eq!(flags.count_ones(), 1);
}

#[test]
fn test_clear_unset_flag() {
    let flags = ExceptionFlags::empty().clear_has_traceback();
    assert!(!flags.has_traceback());
    assert!(flags.is_empty());
}

#[test]
fn test_all_flags_are_independent() {
    // Each flag should be independently settable
    let flag_setters: [fn(ExceptionFlags) -> ExceptionFlags; 8] = [
        ExceptionFlags::set_normalized,
        ExceptionFlags::set_has_args,
        ExceptionFlags::set_has_traceback,
        ExceptionFlags::set_has_cause,
        ExceptionFlags::set_has_context,
        ExceptionFlags::set_suppress_context,
        ExceptionFlags::set_flyweight,
        ExceptionFlags::set_reraised,
    ];

    for (i, setter) in flag_setters.iter().enumerate() {
        let flags = setter(ExceptionFlags::empty());
        assert_eq!(flags.count_ones(), 1, "Flag {} should set exactly 1 bit", i);
    }
}
