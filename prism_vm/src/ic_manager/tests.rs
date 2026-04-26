use super::*;

#[test]
fn test_ic_entry_state_transitions() {
    let mut entry = ICEntry::new_attr(0, true);

    // Start empty
    assert_eq!(entry.classification(), ICClassification::Uninitialized);

    // First access: becomes monomorphic
    entry.record(1, 10);
    assert_eq!(entry.classification(), ICClassification::Monomorphic);

    // Same type: still monomorphic
    entry.record(1, 10);
    assert_eq!(entry.classification(), ICClassification::Monomorphic);

    // Different type: becomes bimorphic
    entry.record(2, 20);
    assert_eq!(entry.classification(), ICClassification::Bimorphic);

    // Third type: becomes polymorphic
    entry.record(3, 30);
    assert_eq!(entry.classification(), ICClassification::Polymorphic);

    // Fourth type: still polymorphic
    entry.record(4, 40);
    assert_eq!(entry.classification(), ICClassification::Polymorphic);
    assert_eq!(entry.type_count(), 4);

    // Fifth type: becomes megamorphic
    entry.record(5, 50);
    assert_eq!(entry.classification(), ICClassification::Megamorphic);
}

#[test]
fn test_ic_entry_hit_tracking() {
    let mut entry = ICEntry::new_attr(0, true);

    // Record initial type
    entry.record(42, 100);

    // Access with same type should hit
    for _ in 0..100 {
        let result = entry.access(42);
        assert_eq!(result, ICAccessResult::Hit(100));
    }

    // Hit rate should be 100%
    assert!(entry.hit_rate() > 99.0);
}

#[test]
fn test_ic_manager_basic() {
    let mut manager = ICManager::new();
    let site = ICSiteId::new(CodeId(1), 10);

    // Initial access: miss
    let result = manager.access(site, 1);
    assert_eq!(result, ICAccessResult::Miss);

    // Record the result
    manager.record(site, 1, 100, 0, true);

    // Now access should hit
    let result = manager.access(site, 1);
    assert_eq!(result, ICAccessResult::Hit(100));

    // Different type: miss again
    let result = manager.access(site, 2);
    assert_eq!(result, ICAccessResult::Miss);
}

#[test]
fn test_ic_manager_jit_queries() {
    let mut manager = ICManager::new();
    let site = ICSiteId::new(CodeId(1), 20);

    // Record monomorphic case
    manager.record(site, 5, 50, 0, true);

    // Check classification
    assert_eq!(
        manager.get_classification(site),
        ICClassification::Monomorphic
    );

    // Get monomorphic type for JIT
    let (type_id, slot) = manager.get_monomorphic_type(site).unwrap();
    assert_eq!(type_id, 5);
    assert_eq!(slot, 50);

    // Add more types to make polymorphic
    manager.record(site, 6, 60, 0, true);
    manager.record(site, 7, 70, 0, true);

    let types = manager.get_polymorphic_types(site).unwrap();
    assert_eq!(types.len(), 3);
}

#[test]
fn test_ic_stats() {
    let mut manager = ICManager::new();

    // Create various IC states
    for i in 0..10 {
        let site = ICSiteId::new(CodeId(i), 0);
        manager.record(site, 1, 100, 0, true);
    }

    // Add polymorphic sites
    for i in 10..15 {
        let site = ICSiteId::new(CodeId(i), 0);
        manager.record(site, 1, 100, 0, true);
        manager.record(site, 2, 200, 0, true);
        manager.record(site, 3, 300, 0, true);
    }

    let stats = manager.classification_breakdown();
    assert_eq!(stats.monomorphic, 10);
    assert_eq!(stats.polymorphic, 5);
}
