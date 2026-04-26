use super::*;

// -------------------------------------------------------------------------
// CallIcData Tests
// -------------------------------------------------------------------------

#[test]
fn test_call_ic_data_function() {
    let data = CallIcData::function(100, 0x1000, 2);
    assert_eq!(data.callee_id, 100);
    assert_eq!(data.code_ptr, 0x1000);
    assert_eq!(data.expected_argc, 2);
    assert!(!data.is_method);
    assert!(data.receiver_shape.is_none());
}

#[test]
fn test_call_ic_data_method() {
    let data = CallIcData::method(100, 0x2000, 3, ShapeId(42));
    assert_eq!(data.callee_id, 100);
    assert_eq!(data.code_ptr, 0x2000);
    assert_eq!(data.expected_argc, 3);
    assert!(data.is_method);
    assert_eq!(data.receiver_shape, Some(ShapeId(42)));
}

#[test]
fn test_call_ic_data_matches_function() {
    let data = CallIcData::function(100, 0x1000, 2);
    assert!(data.matches_function(100));
    assert!(!data.matches_function(101));
}

#[test]
fn test_call_ic_data_matches_method() {
    let data = CallIcData::method(100, 0x1000, 2, ShapeId(42));
    assert!(data.matches_method(100, ShapeId(42)));
    assert!(!data.matches_method(100, ShapeId(43)));
    assert!(!data.matches_method(101, ShapeId(42)));
}

// -------------------------------------------------------------------------
// CallIc Tests
// -------------------------------------------------------------------------

#[test]
fn test_call_ic_new() {
    let ic = CallIc::new();
    assert_eq!(ic.state(), CallIcState::Uninitialized);
    assert_eq!(ic.hits(), 0);
    assert_eq!(ic.misses(), 0);
}

#[test]
fn test_call_ic_lookup_uninitialized() {
    let mut ic = CallIc::new();
    assert!(ic.lookup_function(100).is_none());
    assert_eq!(ic.misses(), 1);
}

#[test]
fn test_call_ic_update_to_mono() {
    let mut ic = CallIc::new();
    ic.update_function(100, 0x1000, 2);

    assert_eq!(ic.state(), CallIcState::Monomorphic);

    let result = ic.lookup_function(100);
    assert_eq!(result, Some(0x1000));
    assert_eq!(ic.hits(), 1);
}

#[test]
fn test_call_ic_mono_miss() {
    let mut ic = CallIc::new();
    ic.update_function(100, 0x1000, 2);

    assert!(ic.lookup_function(101).is_none());
    assert_eq!(ic.misses(), 1);
}

#[test]
fn test_call_ic_mono_to_poly() {
    let mut ic = CallIc::new();

    ic.update_function(100, 0x1000, 2);
    assert_eq!(ic.state(), CallIcState::Monomorphic);

    ic.update_function(200, 0x2000, 3);
    assert_eq!(ic.state(), CallIcState::Polymorphic);

    // Both should be accessible
    assert_eq!(ic.lookup_function(100), Some(0x1000));
    assert_eq!(ic.lookup_function(200), Some(0x2000));
}

#[test]
fn test_call_ic_poly_to_mega() {
    let mut ic = CallIc::new();

    // Fill beyond capacity
    for i in 0..(POLY_CALL_ENTRIES + 2) {
        ic.update_function(i as u64, 0x1000 + i * 0x100, 0);
    }

    assert_eq!(ic.state(), CallIcState::Megamorphic);
}

#[test]
fn test_call_ic_method_lookup() {
    let mut ic = CallIc::new();
    ic.update_method(100, 0x1000, 2, ShapeId(42));

    let result = ic.lookup_method(100, ShapeId(42));
    assert_eq!(result, Some(0x1000));
    assert_eq!(ic.hits(), 1);

    // Wrong shape
    assert!(ic.lookup_method(100, ShapeId(43)).is_none());
}

#[test]
fn test_call_ic_reset() {
    let mut ic = CallIc::new();
    ic.update_function(100, 0x1000, 2);
    ic.lookup_function(100);
    ic.lookup_function(200);

    ic.reset();

    assert_eq!(ic.state(), CallIcState::Uninitialized);
    assert_eq!(ic.hits(), 0);
    assert_eq!(ic.misses(), 0);
}

#[test]
fn test_call_ic_hit_rate() {
    let mut ic = CallIc::new();
    ic.update_function(100, 0x1000, 0);

    ic.lookup_function(100); // hit
    ic.lookup_function(100); // hit
    ic.lookup_function(100); // hit
    ic.lookup_function(200); // miss

    assert!((ic.hit_rate() - 0.75).abs() < 0.001);
}

#[test]
fn test_call_ic_same_callee_no_transition() {
    let mut ic = CallIc::new();

    ic.update_function(100, 0x1000, 2);
    assert_eq!(ic.state(), CallIcState::Monomorphic);

    // Same callee shouldn't transition
    ic.update_function(100, 0x1000, 2);
    assert_eq!(ic.state(), CallIcState::Monomorphic);
}

#[test]
fn test_call_ic_poly_dedup() {
    let mut ic = CallIc::new();

    ic.update_function(100, 0x1000, 0);
    ic.update_function(200, 0x2000, 0);
    assert_eq!(ic.poly_count, 2);

    // Same callee shouldn't add new entry
    ic.update_function(200, 0x2000, 0);
    assert_eq!(ic.poly_count, 2);
}

#[test]
fn test_call_ic_state_from_ic_state() {
    assert_eq!(
        CallIcState::from(IcState::Uninitialized),
        CallIcState::Uninitialized
    );
    assert_eq!(
        CallIcState::from(IcState::Monomorphic),
        CallIcState::Monomorphic
    );
    assert_eq!(
        CallIcState::from(IcState::Polymorphic),
        CallIcState::Polymorphic
    );
    assert_eq!(
        CallIcState::from(IcState::Megamorphic),
        CallIcState::Megamorphic
    );
}

#[test]
fn test_poly_call_entry() {
    let data = CallIcData::function(100, 0x1000, 2);
    let mut entry = PolyCallEntry::new(data);

    assert!(entry.valid);
    assert_eq!(entry.access_count, 0);

    entry.touch();
    assert_eq!(entry.access_count, 1);
}

#[test]
fn test_poly_call_entry_empty() {
    let entry = PolyCallEntry::empty();
    assert!(!entry.valid);
}
