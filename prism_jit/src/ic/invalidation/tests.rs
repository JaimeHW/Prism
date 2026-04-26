use super::*;

// -------------------------------------------------------------------------
// ShapeVersion Tests
// -------------------------------------------------------------------------

#[test]
fn test_shape_version_new() {
    let v = ShapeVersion::new(42);
    assert_eq!(v.value(), 42);
}

#[test]
fn test_shape_version_ordering() {
    let v1 = ShapeVersion::new(1);
    let v2 = ShapeVersion::new(2);

    assert!(v1 < v2);
    assert!(v2 > v1);
    assert_eq!(v1, ShapeVersion::new(1));
}

#[test]
fn test_shape_version_bump() {
    let before = ShapeVersion::current();
    let after = ShapeVersion::bump();
    assert!(after > before);
}

// -------------------------------------------------------------------------
// IcDependency Tests
// -------------------------------------------------------------------------

#[test]
fn test_ic_dependency_new() {
    let dep = IcDependency::new(ShapeId(42), 1, 0);
    assert_eq!(dep.shape_id, ShapeId(42));
    assert_eq!(dep.manager_id, 1);
    assert_eq!(dep.site_index, 0);
}

#[test]
fn test_ic_dependency_stale() {
    let dep = IcDependency::new(ShapeId(1), 1, 0);
    assert!(!dep.is_stale()); // Just created

    ShapeVersion::bump();
    assert!(dep.is_stale()); // Now stale
}

// -------------------------------------------------------------------------
// InvalidationEvent Tests
// -------------------------------------------------------------------------

#[test]
fn test_invalidation_event_new() {
    let before = ShapeVersion::current();
    let event = InvalidationEvent::new(ShapeId(1), InvalidationReason::ShapeTransition);

    assert_eq!(event.shape_id, ShapeId(1));
    assert_eq!(event.reason, InvalidationReason::ShapeTransition);
    assert!(event.new_version > before);
}

// -------------------------------------------------------------------------
// IcInvalidator Tests
// -------------------------------------------------------------------------

#[test]
fn test_invalidator_new() {
    let inv = IcInvalidator::new();
    assert_eq!(inv.dependency_count(), 0);
    assert_eq!(inv.invalidation_count(), 0);
}

#[test]
fn test_invalidator_register_dependency() {
    let inv = IcInvalidator::new();
    let dep = IcDependency::new(ShapeId(1), 100, 0);

    inv.register_dependency(dep);
    assert_eq!(inv.dependency_count(), 1);
}

#[test]
fn test_invalidator_register_multiple() {
    let inv = IcInvalidator::new();
    let deps = vec![
        IcDependency::new(ShapeId(1), 100, 0),
        IcDependency::new(ShapeId(1), 100, 1),
        IcDependency::new(ShapeId(2), 100, 2),
    ];

    inv.register_dependencies(deps);
    assert_eq!(inv.dependency_count(), 3);

    // Check grouped by shape
    assert_eq!(inv.get_dependencies(ShapeId(1)).len(), 2);
    assert_eq!(inv.get_dependencies(ShapeId(2)).len(), 1);
    assert!(inv.get_dependencies(ShapeId(99)).is_empty());
}

#[test]
fn test_invalidator_remove_manager() {
    let inv = IcInvalidator::new();

    // Register deps for two managers
    inv.register_dependency(IcDependency::new(ShapeId(1), 100, 0));
    inv.register_dependency(IcDependency::new(ShapeId(1), 100, 1));
    inv.register_dependency(IcDependency::new(ShapeId(1), 200, 0));

    // Remove manager 100
    inv.remove_manager_dependencies(100);

    assert_eq!(inv.dependency_count(), 1);
    assert_eq!(inv.get_dependencies(ShapeId(1)).len(), 1);
}

#[test]
fn test_invalidator_invalidate_shape() {
    let inv = IcInvalidator::new();

    inv.register_dependency(IcDependency::new(ShapeId(1), 100, 0));
    inv.register_dependency(IcDependency::new(ShapeId(1), 100, 1));
    inv.register_dependency(IcDependency::new(ShapeId(2), 100, 2));

    let event = InvalidationEvent::new(ShapeId(1), InvalidationReason::PropertyDeletion);
    let count = inv.invalidate_shape(event);

    assert_eq!(count, 2);
    assert_eq!(inv.invalidation_count(), 1);
}

#[test]
fn test_invalidator_invalidate_shapes_batch() {
    let inv = IcInvalidator::new();

    inv.register_dependency(IcDependency::new(ShapeId(1), 100, 0));
    inv.register_dependency(IcDependency::new(ShapeId(2), 100, 1));
    inv.register_dependency(IcDependency::new(ShapeId(3), 100, 2));

    let count = inv.invalidate_shapes(
        &[ShapeId(1), ShapeId(3)],
        InvalidationReason::PrototypeChange,
    );

    assert_eq!(count, 2);
}

#[test]
fn test_invalidator_prune_stale() {
    let inv = IcInvalidator::new();

    // Create deps at current version
    let v1 = ShapeVersion::current();
    inv.register_dependency(IcDependency::new(ShapeId(1), 100, 0));

    // Bump version and create more deps
    ShapeVersion::bump();
    let v2 = ShapeVersion::current();
    inv.register_dependency(IcDependency::new(ShapeId(2), 100, 1));

    // Prune older than v2
    inv.prune_stale(v2);

    // Should only have the newer dep
    assert_eq!(inv.dependency_count(), 1);
    assert!(inv.get_dependencies(ShapeId(1)).is_empty());
    assert_eq!(inv.get_dependencies(ShapeId(2)).len(), 1);
}

#[test]
fn test_invalidator_clear() {
    let inv = IcInvalidator::new();

    inv.register_dependency(IcDependency::new(ShapeId(1), 100, 0));
    inv.register_dependency(IcDependency::new(ShapeId(2), 100, 1));

    inv.clear();

    assert_eq!(inv.dependency_count(), 0);
    assert!(inv.get_dependencies(ShapeId(1)).is_empty());
}

#[test]
fn test_invalidator_stats() {
    let inv = IcInvalidator::new();

    inv.register_dependency(IcDependency::new(ShapeId(1), 100, 0));
    inv.register_dependency(IcDependency::new(ShapeId(1), 100, 1));
    inv.register_dependency(IcDependency::new(ShapeId(2), 100, 2));

    let event = InvalidationEvent::new(ShapeId(1), InvalidationReason::Manual);
    inv.invalidate_shape(event);

    let stats = inv.stats();
    assert_eq!(stats.dependency_count, 3);
    assert_eq!(stats.unique_shapes, 2);
    assert_eq!(stats.invalidation_count, 1);
}

// -------------------------------------------------------------------------
// Concurrent Tests
// -------------------------------------------------------------------------

#[test]
fn test_invalidator_concurrent_register() {
    use std::sync::Arc;
    use std::thread;

    let inv = Arc::new(IcInvalidator::new());
    let mut handles = vec![];

    for manager_id in 0..10u64 {
        let i = Arc::clone(&inv);
        handles.push(thread::spawn(move || {
            for site_idx in 0..100u32 {
                let shape_id = ShapeId((manager_id * 100 + site_idx as u64) as u32);
                i.register_dependency(IcDependency::new(shape_id, manager_id, site_idx));
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(inv.dependency_count(), 1000);
}

#[test]
fn test_shape_version_concurrent_bump() {
    use std::sync::Arc;
    use std::thread;

    let before = ShapeVersion::current();
    let mut handles = vec![];

    for _ in 0..100 {
        handles.push(thread::spawn(|| {
            for _ in 0..100 {
                ShapeVersion::bump();
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let after = ShapeVersion::current();
    // Each bump is atomic, so we should have at least 10000 increments
    // (may be more if other concurrent tests also bumped the version)
    assert!(
        after.value() - before.value() >= 10000,
        "Expected at least 10000 bumps, got {}",
        after.value() - before.value()
    );
}

// -------------------------------------------------------------------------
// Global Invalidator Tests
// -------------------------------------------------------------------------

#[test]
fn test_global_invalidator() {
    init_invalidator();
    let inv = global_invalidator();

    // Should be the same instance
    assert!(std::ptr::eq(inv, global_invalidator()));
}
