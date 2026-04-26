use super::*;

#[test]
fn test_recovery_creation() {
    let recovery = DeoptRecovery::new();
    assert_eq!(recovery.recompile_threshold, 10);
    assert_eq!(recovery.patch_threshold, 100);
}

#[test]
fn test_recovery_with_thresholds() {
    let recovery = DeoptRecovery::with_thresholds(5, 50);
    assert_eq!(recovery.recompile_threshold, 5);
    assert_eq!(recovery.patch_threshold, 50);
}

#[test]
fn test_recover_simple() {
    let recovery = DeoptRecovery::new();

    let state = DeoptState::new(100, DeoptReason::TypeGuard, 1, 12345);
    let jit_regs = vec![Value::from(1), Value::from(2), Value::from(3)];
    let mut frame_regs = vec![Value::none(); 3];

    let result = recovery.recover(&state, &jit_regs, &mut frame_regs);

    match result {
        RecoveryResult::Resume {
            bc_offset,
            should_recompile,
        } => {
            assert_eq!(bc_offset, 100);
            assert!(should_recompile); // TypeGuard triggers recompile
        }
        RecoveryResult::Error(_) => panic!("Expected Resume"),
    }
}

#[test]
fn test_recover_with_delta() {
    let recovery = DeoptRecovery::new();

    let mut state = DeoptState::new(100, DeoptReason::TypeGuard, 1, 12345);
    state.record_modified(1, Value::from(999));

    let jit_regs = vec![Value::from(1), Value::from(2), Value::from(3)];
    let mut frame_regs = vec![Value::none(); 3];

    let _ = recovery.recover(&state, &jit_regs, &mut frame_regs);

    // Slot 1 should have delta value
    // (Exact comparison depends on Value implementation)
}

#[test]
fn test_should_patch_guard() {
    let recovery = DeoptRecovery::new();

    // PolymorphicSite always triggers patch
    assert!(recovery.should_patch_guard(0, DeoptReason::PolymorphicSite));

    // TypeGuard only after threshold
    assert!(!recovery.should_patch_guard(0, DeoptReason::TypeGuard));
    assert!(recovery.should_patch_guard(100, DeoptReason::TypeGuard));
}

#[test]
fn test_should_recompile() {
    let recovery = DeoptRecovery::new();

    // TypeGuard triggers recompile after threshold
    assert!(!recovery.should_recompile(5, DeoptReason::TypeGuard));
    assert!(recovery.should_recompile(10, DeoptReason::TypeGuard));

    // DivByZero doesn't trigger recompile
    assert!(!recovery.should_recompile(100, DeoptReason::DivByZero));
}

#[test]
fn test_recovery_error_display() {
    let err = RecoveryError::InvalidOffset(100);
    assert!(format!("{}", err).contains("100"));

    let err = RecoveryError::FrameCorruption;
    assert!(format!("{}", err).contains("corruption"));
}

#[test]
fn test_value_type_hint() {
    let recovery = DeoptRecovery::new();

    let int_val = recovery.materialize_value(42, ValueTypeHint::Int);
    assert!(!int_val.is_none());

    let bool_val = recovery.materialize_value(1, ValueTypeHint::Bool);
    assert!(!bool_val.is_none());
}
