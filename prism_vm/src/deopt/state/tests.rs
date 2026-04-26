use super::*;

#[test]
fn test_deopt_reason_from_u8() {
    assert_eq!(DeoptReason::from_u8(0), Some(DeoptReason::TypeGuard));
    assert_eq!(DeoptReason::from_u8(5), Some(DeoptReason::DivByZero));
    assert_eq!(DeoptReason::from_u8(255), None);
}

#[test]
fn test_deopt_reason_display() {
    assert_eq!(format!("{}", DeoptReason::TypeGuard), "type guard");
    assert_eq!(format!("{}", DeoptReason::Overflow), "overflow");
}

#[test]
fn test_deopt_reason_should_patch_guard() {
    assert!(DeoptReason::PolymorphicSite.should_patch_guard());
    assert!(DeoptReason::UncommonTrap.should_patch_guard());
    assert!(!DeoptReason::TypeGuard.should_patch_guard());
}

#[test]
fn test_delta_entry() {
    let entry = DeltaEntry::new(5, Value::from(42));
    assert_eq!(entry.slot, 5);
}

#[test]
fn test_deopt_delta_record() {
    let mut delta = DeoptDelta::new();
    assert!(delta.is_empty());

    delta.record(0, Value::from(10));
    delta.record(5, Value::from(20));

    assert_eq!(delta.len(), 2);
    assert!(delta.get(0).is_some());
    assert!(delta.get(5).is_some());
    assert!(delta.get(3).is_none());
}

#[test]
fn test_deopt_delta_update_existing() {
    let mut delta = DeoptDelta::new();

    delta.record(0, Value::from(10));
    delta.record(0, Value::from(20));

    assert_eq!(delta.len(), 1);
    // Value should be updated
}

#[test]
fn test_deopt_state_creation() {
    let state = DeoptState::new(100, DeoptReason::TypeGuard, 1, 12345);

    assert_eq!(state.bc_offset, 100);
    assert_eq!(state.reason, DeoptReason::TypeGuard);
    assert_eq!(state.deopt_id, 1);
    assert_eq!(state.code_id, 12345);
    assert!(state.delta.is_empty());
}

#[test]
fn test_deopt_state_record_modified() {
    let mut state = DeoptState::new(100, DeoptReason::TypeGuard, 1, 12345);

    state.record_modified(0, Value::from(42));
    state.record_modified(5, Value::from(100));

    assert_eq!(state.delta.len(), 2);
}
