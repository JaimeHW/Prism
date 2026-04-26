use super::*;
use crate::ops::generator::suspend::capture_generator_frame;

// ════════════════════════════════════════════════════════════════════════
// ResumeResult Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_resume_result_new() {
    let result = ResumeResult::new(100, 5);
    assert_eq!(result.target_pc, 100);
    assert_eq!(result.registers_restored, 5);
    assert!(!result.had_send_value);
}

#[test]
fn test_resume_result_with_send_value() {
    let result = ResumeResult::new(100, 5).with_send_value();
    assert!(result.had_send_value);
}

// ════════════════════════════════════════════════════════════════════════
// ResumeDispatcher Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_dispatcher_new() {
    let dispatcher = ResumeDispatcher::new();
    assert_eq!(dispatcher.fallback_pc, 0);
}

#[test]
fn test_dispatcher_with_fallback() {
    let dispatcher = ResumeDispatcher::with_fallback(999);
    assert_eq!(dispatcher.fallback_pc, 999);
}

#[test]
fn test_dispatcher_no_table() {
    let dispatcher = ResumeDispatcher::with_fallback(42);
    let pc = dispatcher.dispatch(None, 0);
    assert_eq!(pc, 42);
}

#[test]
fn test_dispatcher_with_table() {
    use crate::ops::generator::resume_cache::ResumeTable;

    let mut table = ResumeTable::new();
    table.insert(0, 100);
    table.insert(1, 200);
    table.insert(2, 300);

    let dispatcher = ResumeDispatcher::new();

    assert_eq!(dispatcher.dispatch(Some(&table), 0), 100);
    assert_eq!(dispatcher.dispatch(Some(&table), 1), 200);
    assert_eq!(dispatcher.dispatch(Some(&table), 2), 300);
}

#[test]
fn test_dispatcher_invalid_index_uses_fallback() {
    use crate::ops::generator::resume_cache::ResumeTable;

    let table = ResumeTable::new();
    let dispatcher = ResumeDispatcher::with_fallback(999);

    assert_eq!(dispatcher.dispatch(Some(&table), 100), 999);
}

#[test]
fn test_dispatcher_direct() {
    let dispatcher = ResumeDispatcher::new();
    assert_eq!(dispatcher.dispatch_direct(42), 42);
    assert_eq!(dispatcher.dispatch_direct(0), 0);
}

// ════════════════════════════════════════════════════════════════════════
// restore_generator_frame Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_restore_empty() {
    let suspended = SuspendResult::new(0, 0);
    let mut registers = vec![Value::int(99).unwrap(); 4];

    restore_generator_frame(&suspended, &mut registers);

    // Nothing restored, values unchanged
    for reg in &registers {
        assert_eq!(reg.as_int(), Some(99));
    }
}

#[test]
fn test_restore_all() {
    let original = vec![
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ];
    let suspended = capture_generator_frame(&original, 0b111, 0).unwrap();

    let mut restored = vec![Value::none(); 3];
    restore_generator_frame(&suspended, &mut restored);

    assert_eq!(restored[0].as_int(), Some(1));
    assert_eq!(restored[1].as_int(), Some(2));
    assert_eq!(restored[2].as_int(), Some(3));
}

#[test]
fn test_restore_sparse() {
    let original = vec![
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
        Value::int(4).unwrap(),
    ];
    // Only registers 0 and 3 live
    let suspended = capture_generator_frame(&original, 0b1001, 0).unwrap();

    let mut restored = vec![Value::none(); 4];
    restore_generator_frame(&suspended, &mut restored);

    assert_eq!(restored[0].as_int(), Some(1));
    assert!(restored[1].is_none()); // Not live
    assert!(restored[2].is_none()); // Not live
    assert_eq!(restored[3].as_int(), Some(4));
}

#[test]
fn test_restore_preserves_non_live() {
    let original = vec![
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ];
    // Only register 1 live
    let suspended = capture_generator_frame(&original, 0b010, 0).unwrap();

    // Pre-fill with different values
    let mut restored = vec![
        Value::int(100).unwrap(),
        Value::int(200).unwrap(),
        Value::int(300).unwrap(),
    ];
    restore_generator_frame(&suspended, &mut restored);

    assert_eq!(restored[0].as_int(), Some(100)); // Unchanged
    assert_eq!(restored[1].as_int(), Some(2)); // Restored
    assert_eq!(restored[2].as_int(), Some(300)); // Unchanged
}

// ════════════════════════════════════════════════════════════════════════
// dispatch_to_resume_point Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_dispatch_no_table() {
    let result = dispatch_to_resume_point(None, 0);
    assert!(result.is_none());
}

#[test]
fn test_dispatch_valid_index() {
    use crate::ops::generator::resume_cache::ResumeTable;

    let mut table = ResumeTable::new();
    table.insert(0, 100);

    let result = dispatch_to_resume_point(Some(&table), 0);
    assert_eq!(result, Some(100));
}

#[test]
fn test_dispatch_invalid_index() {
    use crate::ops::generator::resume_cache::ResumeTable;

    let table = ResumeTable::new();

    let result = dispatch_to_resume_point(Some(&table), 99);
    assert!(result.is_none());
}

// ════════════════════════════════════════════════════════════════════════
// inject_send_value Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_inject_send_value() {
    let mut registers = vec![Value::none(); 4];

    inject_send_value(&mut registers, 2, Value::int(42).unwrap());

    assert!(registers[0].is_none());
    assert!(registers[1].is_none());
    assert_eq!(registers[2].as_int(), Some(42));
    assert!(registers[3].is_none());
}

#[test]
fn test_inject_send_value_first_register() {
    let mut registers = vec![Value::none(); 4];

    inject_send_value(&mut registers, 0, Value::bool(true));

    assert_eq!(registers[0].as_bool(), Some(true));
}

#[test]
fn test_inject_send_value_out_of_bounds() {
    let mut registers = vec![Value::none(); 4];

    // Should not panic, just do nothing
    inject_send_value(&mut registers, 100, Value::int(42).unwrap());

    // All still None
    for reg in &registers {
        assert!(reg.is_none());
    }
}

// ════════════════════════════════════════════════════════════════════════
// prepare_throw Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_prepare_throw() {
    let mut registers = vec![Value::none(); 4];

    prepare_throw(&mut registers, 1, Value::int(999).unwrap());

    assert!(registers[0].is_none());
    assert_eq!(registers[1].as_int(), Some(999));
    assert!(registers[2].is_none());
}

#[test]
fn test_prepare_throw_out_of_bounds() {
    let mut registers = vec![Value::none(); 4];

    // Should not panic
    prepare_throw(&mut registers, 100, Value::int(999).unwrap());
}

// ════════════════════════════════════════════════════════════════════════
// ResumeError Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_resume_error_display() {
    assert_eq!(
        ResumeError::NotSuspended.to_string(),
        "Generator is not suspended"
    );
    assert_eq!(
        ResumeError::GeneratorExhausted.to_string(),
        "Generator has already completed"
    );
    assert_eq!(
        ResumeError::InvalidResumeIndex(42).to_string(),
        "Invalid resume index: 42"
    );
    assert_eq!(
        ResumeError::CantSendNonNone.to_string(),
        "Can't send non-None value to a just-started generator"
    );
    assert_eq!(
        ResumeError::NoSavedState.to_string(),
        "No saved state to restore"
    );
}

// ════════════════════════════════════════════════════════════════════════
// Round-trip Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_capture_restore_round_trip() {
    let original = vec![
        Value::int(10).unwrap(),
        Value::int(20).unwrap(),
        Value::int(30).unwrap(),
        Value::int(40).unwrap(),
        Value::none(),
        Value::bool(true),
    ];
    let liveness = 0b111011; // All except register 2

    let suspended = capture_generator_frame(&original, liveness, 42).unwrap();
    assert_eq!(suspended.resume_index, 42);

    let mut restored = vec![Value::none(); 6];
    restore_generator_frame(&suspended, &mut restored);

    assert_eq!(restored[0].as_int(), Some(10));
    assert_eq!(restored[1].as_int(), Some(20));
    assert!(restored[2].is_none()); // Not live
    assert_eq!(restored[3].as_int(), Some(40));
    assert!(restored[4].is_none());
    assert_eq!(restored[5].as_bool(), Some(true));
}

#[test]
fn test_full_resume_workflow() {
    use crate::ops::generator::resume_cache::ResumeTable;

    // Setup
    let original = vec![
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ];

    // Simulate yield (suspend)
    let liveness = 0b111;
    let resume_index = 1;
    let suspended = capture_generator_frame(&original, liveness, resume_index).unwrap();

    // Build resume table
    let mut table = ResumeTable::new();
    table.insert(0, 100);
    table.insert(1, 200); // Our resume point
    table.insert(2, 300);

    // Simulate resume (restore)
    let mut registers = vec![Value::none(); 3];
    restore_generator_frame(&suspended, &mut registers);

    // Inject send value
    inject_send_value(&mut registers, 0, Value::int(42).unwrap());

    // Dispatch
    let target_pc = dispatch_to_resume_point(Some(&table), resume_index);

    // Verify
    assert_eq!(target_pc, Some(200));
    assert_eq!(registers[0].as_int(), Some(42)); // Send value overwrote
    assert_eq!(registers[1].as_int(), Some(2));
    assert_eq!(registers[2].as_int(), Some(3));
}

// ════════════════════════════════════════════════════════════════════════
// Edge Case Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_restore_to_smaller_array() {
    let original = vec![
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
        Value::int(4).unwrap(),
    ];
    let suspended = capture_generator_frame(&original, 0b1111, 0).unwrap();

    // Restore to smaller array
    let mut restored = vec![Value::none(); 2];
    restore_generator_frame(&suspended, &mut restored);

    // Only first 2 restored
    assert_eq!(restored[0].as_int(), Some(1));
    assert_eq!(restored[1].as_int(), Some(2));
}

#[test]
fn test_restore_to_larger_array() {
    let original = vec![Value::int(1).unwrap(), Value::int(2).unwrap()];
    let suspended = capture_generator_frame(&original, 0b11, 0).unwrap();

    // Restore to larger array
    let mut restored = vec![Value::int(99).unwrap(); 4];
    restore_generator_frame(&suspended, &mut restored);

    assert_eq!(restored[0].as_int(), Some(1));
    assert_eq!(restored[1].as_int(), Some(2));
    assert_eq!(restored[2].as_int(), Some(99)); // Unchanged
    assert_eq!(restored[3].as_int(), Some(99)); // Unchanged
}
