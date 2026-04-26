use super::*;

// -------------------------------------------------------------------------
// FieldIndex Tests
// -------------------------------------------------------------------------

#[test]
fn test_field_index_constant() {
    let idx = FieldIndex::Constant(5);
    assert!(idx.is_constant());
    assert_eq!(idx.as_constant(), Some(5));
}

#[test]
fn test_field_index_dynamic() {
    let idx = FieldIndex::Dynamic;
    assert!(!idx.is_constant());
    assert_eq!(idx.as_constant(), None);
}

// -------------------------------------------------------------------------
// FieldAccess Tests
// -------------------------------------------------------------------------

#[test]
fn test_field_access_load() {
    let access = FieldAccess {
        node: NodeId::new(1),
        object: NodeId::new(0),
        field: FieldIndex::Constant(2),
        kind: FieldAccessKind::Load,
        value: None,
        value_type: ValueType::Int64,
        control: None,
    };

    assert!(access.is_load());
    assert!(!access.is_store());
}

#[test]
fn test_field_access_store() {
    let access = FieldAccess {
        node: NodeId::new(1),
        object: NodeId::new(0),
        field: FieldIndex::Constant(2),
        kind: FieldAccessKind::Store,
        value: Some(NodeId::new(3)),
        value_type: ValueType::Int64,
        control: Some(NodeId::new(4)),
    };

    assert!(!access.is_load());
    assert!(access.is_store());
    assert_eq!(access.value, Some(NodeId::new(3)));
}

// -------------------------------------------------------------------------
// FieldState Tests
// -------------------------------------------------------------------------

#[test]
fn test_field_state_uninitialized() {
    let state = FieldState::Uninitialized;
    assert!(state.as_value().is_none());
    assert!(state.values().is_empty());
}

#[test]
fn test_field_state_value() {
    let state = FieldState::Value(NodeId::new(5));
    assert_eq!(state.as_value(), Some(NodeId::new(5)));
    assert_eq!(state.values().len(), 1);
}

#[test]
fn test_field_state_phi() {
    let mut values = SmallVec::new();
    values.push(NodeId::new(1));
    values.push(NodeId::new(2));
    let state = FieldState::Phi(values);

    assert!(state.as_value().is_none());
    assert_eq!(state.values().len(), 2);
}

#[test]
fn test_field_state_merge_uninit() {
    let uninit = FieldState::Uninitialized;
    let value = FieldState::Value(NodeId::new(5));

    let merged = uninit.merge(&value);
    assert_eq!(merged.as_value(), Some(NodeId::new(5)));

    let merged2 = value.merge(&FieldState::Uninitialized);
    assert_eq!(merged2.as_value(), Some(NodeId::new(5)));
}

#[test]
fn test_field_state_merge_same_value() {
    let v1 = FieldState::Value(NodeId::new(5));
    let v2 = FieldState::Value(NodeId::new(5));

    let merged = v1.merge(&v2);
    assert_eq!(merged.as_value(), Some(NodeId::new(5)));
}

#[test]
fn test_field_state_merge_different_values() {
    let v1 = FieldState::Value(NodeId::new(5));
    let v2 = FieldState::Value(NodeId::new(6));

    let merged = v1.merge(&v2);
    assert!(merged.as_value().is_none());
    assert_eq!(merged.values().len(), 2);
}

#[test]
fn test_field_state_merge_value_with_phi() {
    let value = FieldState::Value(NodeId::new(5));
    let mut phi_values = SmallVec::new();
    phi_values.push(NodeId::new(1));
    phi_values.push(NodeId::new(2));
    let phi = FieldState::Phi(phi_values);

    let merged = value.merge(&phi);
    assert_eq!(merged.values().len(), 3);
}

#[test]
fn test_field_state_merge_phi_with_phi() {
    let mut phi1_values = SmallVec::new();
    phi1_values.push(NodeId::new(1));
    phi1_values.push(NodeId::new(2));
    let phi1 = FieldState::Phi(phi1_values);

    let mut phi2_values = SmallVec::new();
    phi2_values.push(NodeId::new(2));
    phi2_values.push(NodeId::new(3));
    let phi2 = FieldState::Phi(phi2_values);

    let merged = phi1.merge(&phi2);
    // Should have 1, 2, 3 (2 is deduplicated)
    assert_eq!(merged.values().len(), 3);
}

// -------------------------------------------------------------------------
// FieldMap Tests
// -------------------------------------------------------------------------

#[test]
fn test_field_map_new() {
    let map = FieldMap::new();
    assert!(map.can_scalar_replace());
    assert_eq!(map.num_fields(), 0);
}

#[test]
fn test_field_map_store_and_load() {
    let mut map = FieldMap::new();

    map.store(FieldIndex::Constant(0), NodeId::new(10));
    map.store(FieldIndex::Constant(1), NodeId::new(11));

    assert_eq!(map.num_fields(), 2);

    let state0 = map.load(FieldIndex::Constant(0)).unwrap();
    assert_eq!(state0.as_value(), Some(NodeId::new(10)));

    let state1 = map.load(FieldIndex::Constant(1)).unwrap();
    assert_eq!(state1.as_value(), Some(NodeId::new(11)));

    assert!(map.load(FieldIndex::Constant(2)).is_none());
}

#[test]
fn test_field_map_dynamic_access() {
    let mut map = FieldMap::new();

    map.store(FieldIndex::Constant(0), NodeId::new(10));
    assert!(map.can_scalar_replace());

    map.store(FieldIndex::Dynamic, NodeId::new(11));
    assert!(!map.can_scalar_replace());
}

#[test]
fn test_field_map_max_field() {
    let mut map = FieldMap::new();

    map.store(FieldIndex::Constant(5), NodeId::new(10));
    assert_eq!(map.max_field_index(), 5);

    map.store(FieldIndex::Constant(2), NodeId::new(11));
    assert_eq!(map.max_field_index(), 5);

    map.store(FieldIndex::Constant(10), NodeId::new(12));
    assert_eq!(map.max_field_index(), 10);
}

#[test]
fn test_field_map_merge() {
    let mut map1 = FieldMap::new();
    map1.store(FieldIndex::Constant(0), NodeId::new(10));
    map1.store(FieldIndex::Constant(1), NodeId::new(11));

    let mut map2 = FieldMap::new();
    map2.store(FieldIndex::Constant(0), NodeId::new(20));
    map2.store(FieldIndex::Constant(2), NodeId::new(22));

    map1.merge(&map2);

    // Field 0 should be phi (two different values)
    let state0 = map1.get(0).unwrap();
    assert_eq!(state0.values().len(), 2);

    // Field 1 should still be single value
    let state1 = map1.get(1).unwrap();
    assert_eq!(state1.as_value(), Some(NodeId::new(11)));

    // Field 2 should be from map2
    let state2 = map1.get(2).unwrap();
    assert_eq!(state2.as_value(), Some(NodeId::new(22)));
}

#[test]
fn test_field_map_is_initialized() {
    let mut map = FieldMap::new();

    assert!(!map.is_initialized(0));

    map.store(FieldIndex::Constant(0), NodeId::new(10));

    assert!(map.is_initialized(0));
    assert!(!map.is_initialized(1));
}

// -------------------------------------------------------------------------
// FieldTracker Tests
// -------------------------------------------------------------------------

#[test]
fn test_field_tracker_new() {
    let tracker = FieldTracker::new(NodeId::new(0));

    assert_eq!(tracker.allocation, NodeId::new(0));
    assert!(tracker.accesses.is_empty());
    assert!(!tracker.has_dynamic_access);
    assert!(!tracker.has_unknown_uses);
}

#[test]
fn test_field_tracker_empty_cannot_scalar_replace() {
    let tracker = FieldTracker::new(NodeId::new(0));
    // Empty tracker (no accesses) cannot be scalar replaced
    assert!(!tracker.can_scalar_replace());
}

#[test]
fn test_field_tracker_with_dynamic_access() {
    let mut tracker = FieldTracker::new(NodeId::new(0));
    tracker.has_dynamic_access = true;

    assert!(!tracker.can_scalar_replace());
}

#[test]
fn test_field_tracker_with_unknown_uses() {
    let mut tracker = FieldTracker::new(NodeId::new(0));
    tracker.has_unknown_uses = true;

    assert!(!tracker.can_scalar_replace());
}

#[test]
fn test_field_tracker_num_fields() {
    let mut tracker = FieldTracker::new(NodeId::new(0));

    // Manually add some field accesses
    tracker.stores_by_field.insert(0, SmallVec::new());
    tracker.stores_by_field.insert(1, SmallVec::new());
    tracker.loads_by_field.insert(2, SmallVec::new());

    assert_eq!(tracker.num_fields(), 3);
}

#[test]
fn test_field_tracker_field_type() {
    let mut tracker = FieldTracker::new(NodeId::new(0));

    tracker.field_types.insert(0, ValueType::Int64);
    tracker.field_types.insert(1, ValueType::Float64);

    assert_eq!(tracker.field_type(0), ValueType::Int64);
    assert_eq!(tracker.field_type(1), ValueType::Float64);
    assert_eq!(tracker.field_type(2), ValueType::Top); // Default
}

#[test]
fn test_field_tracker_stores_and_loads_iterators() {
    let mut tracker = FieldTracker::new(NodeId::new(0));

    // Add some accesses
    let store_access = FieldAccess {
        node: NodeId::new(1),
        object: NodeId::new(0),
        field: FieldIndex::Constant(0),
        kind: FieldAccessKind::Store,
        value: Some(NodeId::new(5)),
        value_type: ValueType::Int64,
        control: None,
    };

    let load_access = FieldAccess {
        node: NodeId::new(2),
        object: NodeId::new(0),
        field: FieldIndex::Constant(0),
        kind: FieldAccessKind::Load,
        value: None,
        value_type: ValueType::Int64,
        control: None,
    };

    tracker.accesses.push(store_access);
    tracker.accesses.push(load_access);
    tracker.stores_by_field.entry(0).or_default().push(0);
    tracker.loads_by_field.entry(0).or_default().push(1);

    let stores: Vec<_> = tracker.stores_for_field(0).collect();
    assert_eq!(stores.len(), 1);
    assert!(stores[0].is_store());

    let loads: Vec<_> = tracker.loads_for_field(0).collect();
    assert_eq!(loads.len(), 1);
    assert!(loads[0].is_load());
}

// -------------------------------------------------------------------------
// Integration Tests
// -------------------------------------------------------------------------

#[test]
fn test_field_map_field_indices() {
    let mut map = FieldMap::new();

    map.store(FieldIndex::Constant(5), NodeId::new(10));
    map.store(FieldIndex::Constant(1), NodeId::new(11));
    map.store(FieldIndex::Constant(3), NodeId::new(12));

    let indices: Vec<u32> = map.field_indices().collect();
    assert_eq!(indices.len(), 3);
    assert!(indices.contains(&1));
    assert!(indices.contains(&3));
    assert!(indices.contains(&5));
}
