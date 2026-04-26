use super::*;
use crate::ic::invalidation::IcDependency;
use std::sync::Arc;
use std::thread;

// -------------------------------------------------------------------------
// JitShapeListener Construction Tests
// -------------------------------------------------------------------------

#[test]
fn test_jit_listener_new() {
    let listener = JitShapeListener::new();
    assert!(!listener.explicit_on_transition());
    assert_eq!(listener.stats().transition_count(), 0);
}

#[test]
fn test_jit_listener_with_explicit_invalidation() {
    let listener = JitShapeListener::with_explicit_invalidation();
    assert!(listener.explicit_on_transition());
}

#[test]
fn test_jit_listener_set_explicit() {
    let listener = JitShapeListener::new();
    assert!(!listener.explicit_on_transition());

    listener.set_explicit_on_transition(true);
    assert!(listener.explicit_on_transition());

    listener.set_explicit_on_transition(false);
    assert!(!listener.explicit_on_transition());
}

// -------------------------------------------------------------------------
// Transition Tests
// -------------------------------------------------------------------------

#[test]
fn test_jit_listener_transition_bumps_version() {
    let listener = JitShapeListener::new();
    let before = ShapeVersion::current();

    listener.on_transition(ShapeId(1), ShapeId(2));

    let after = ShapeVersion::current();
    assert!(after > before);
    assert_eq!(listener.stats().transition_count(), 1);
    assert_eq!(listener.stats().version_bump_count(), 1);
}

#[test]
fn test_jit_listener_transition_explicit_mode() {
    let listener = JitShapeListener::with_explicit_invalidation();
    let before = ShapeVersion::current();

    listener.on_transition(ShapeId(1), ShapeId(2));

    let after = ShapeVersion::current();
    assert!(after > before);
    assert_eq!(listener.stats().transition_count(), 1);
    assert_eq!(listener.stats().explicit_invalidation_count(), 1);
}

#[test]
fn test_jit_listener_multiple_transitions() {
    let listener = JitShapeListener::new();

    for i in 0..10 {
        listener.on_transition(ShapeId(i), ShapeId(i + 1));
    }

    assert_eq!(listener.stats().transition_count(), 10);
    assert_eq!(listener.stats().version_bump_count(), 10);
}

// -------------------------------------------------------------------------
// Property Deletion Tests
// -------------------------------------------------------------------------

#[test]
fn test_jit_listener_property_delete_bumps_version() {
    let listener = JitShapeListener::new();
    let before = ShapeVersion::current();

    listener.on_property_delete(ShapeId(1), "foo");

    let after = ShapeVersion::current();
    assert!(after > before);
    assert_eq!(listener.stats().deletion_count(), 1);
    assert_eq!(listener.stats().version_bump_count(), 1);
    assert_eq!(listener.stats().explicit_invalidation_count(), 1);
}

#[test]
fn test_jit_listener_property_delete_invalidates() {
    let listener = JitShapeListener::new();

    // Register a dependency to be invalidated
    let dep = IcDependency::new(ShapeId(42), 100, 0);
    global_invalidator().register_dependency(dep);

    listener.on_property_delete(ShapeId(42), "prop");

    assert_eq!(listener.stats().deletion_count(), 1);
    // IC should have been invalidated
    assert!(listener.stats().ics_invalidated_count() >= 1);
}

// -------------------------------------------------------------------------
// Prototype Change Tests
// -------------------------------------------------------------------------

#[test]
fn test_jit_listener_prototype_change_bumps_version() {
    let listener = JitShapeListener::new();
    let before = ShapeVersion::current();

    listener.on_prototype_change(ShapeId(1));

    let after = ShapeVersion::current();
    assert!(after > before);
    assert_eq!(listener.stats().prototype_change_count(), 1);
    assert_eq!(listener.stats().version_bump_count(), 1);
    assert_eq!(listener.stats().explicit_invalidation_count(), 1);
}

#[test]
fn test_jit_listener_prototype_change_invalidates() {
    let listener = JitShapeListener::new();

    // Register a dependency
    let dep = IcDependency::new(ShapeId(99), 200, 5);
    global_invalidator().register_dependency(dep);

    listener.on_prototype_change(ShapeId(99));

    assert_eq!(listener.stats().prototype_change_count(), 1);
}

// -------------------------------------------------------------------------
// Accessor Installation Tests
// -------------------------------------------------------------------------

#[test]
fn test_jit_listener_accessor_installed_bumps_version() {
    let listener = JitShapeListener::new();
    let before = ShapeVersion::current();

    listener.on_accessor_installed(ShapeId(1), "getValue");

    let after = ShapeVersion::current();
    assert!(after > before);
    assert_eq!(listener.stats().accessor_install_count(), 1);
    assert_eq!(listener.stats().version_bump_count(), 1);
    assert_eq!(listener.stats().explicit_invalidation_count(), 1);
}

// -------------------------------------------------------------------------
// Batch Transition Tests
// -------------------------------------------------------------------------

#[test]
fn test_jit_listener_batch_transition_empty() {
    let listener = JitShapeListener::new();
    let before = ShapeVersion::current();

    let count = listener.on_batch_transition(&[]);

    let after = ShapeVersion::current();
    assert_eq!(count, 0);
    // No version bump for empty batch
    assert_eq!(before, after);
}

#[test]
fn test_jit_listener_batch_transition_single_bump() {
    let listener = JitShapeListener::new();
    let before = ShapeVersion::current();

    let transitions = vec![
        (ShapeId(1), ShapeId(2)),
        (ShapeId(2), ShapeId(3)),
        (ShapeId(3), ShapeId(4)),
    ];

    let count = listener.on_batch_transition(&transitions);

    let after = ShapeVersion::current();
    assert_eq!(count, 3);
    // Single version bump for all
    assert_eq!(listener.stats().version_bump_count(), 1);
    assert_eq!(listener.stats().batch_transition_count(), 3);
    assert!(after > before);
}

#[test]
fn test_jit_listener_batch_transition_explicit_mode() {
    let listener = JitShapeListener::with_explicit_invalidation();

    let transitions = vec![(ShapeId(1), ShapeId(2)), (ShapeId(2), ShapeId(3))];

    let count = listener.on_batch_transition(&transitions);

    assert_eq!(count, 2);
    // Should have done explicit invalidation
    assert_eq!(listener.stats().explicit_invalidation_count(), 1);
}

// -------------------------------------------------------------------------
// Statistics Tests
// -------------------------------------------------------------------------

#[test]
fn test_jit_listener_stats_new() {
    let stats = JitListenerStats::new();
    assert_eq!(stats.transition_count(), 0);
    assert_eq!(stats.deletion_count(), 0);
    assert_eq!(stats.prototype_change_count(), 0);
    assert_eq!(stats.accessor_install_count(), 0);
    assert_eq!(stats.batch_transition_count(), 0);
    assert_eq!(stats.version_bump_count(), 0);
    assert_eq!(stats.explicit_invalidation_count(), 0);
    assert_eq!(stats.ics_invalidated_count(), 0);
    assert_eq!(stats.total_events(), 0);
}

#[test]
fn test_jit_listener_stats_snapshot() {
    let listener = JitShapeListener::new();

    listener.on_transition(ShapeId(1), ShapeId(2));
    listener.on_property_delete(ShapeId(2), "x");
    listener.on_prototype_change(ShapeId(3));
    listener.on_accessor_installed(ShapeId(4), "y");

    let snap = listener.stats().snapshot();
    assert_eq!(snap.transitions, 1);
    assert_eq!(snap.deletions, 1);
    assert_eq!(snap.prototype_changes, 1);
    assert_eq!(snap.accessor_installs, 1);
    assert!(snap.version_bumps >= 4);
    assert_eq!(snap.explicit_invalidations, 3); // delete + proto + accessor
}

#[test]
fn test_jit_listener_stats_total_events() {
    let listener = JitShapeListener::new();

    listener.on_transition(ShapeId(1), ShapeId(2));
    listener.on_transition(ShapeId(2), ShapeId(3));
    listener.on_property_delete(ShapeId(3), "a");
    listener.on_prototype_change(ShapeId(4));
    listener.on_accessor_installed(ShapeId(5), "b");

    assert_eq!(listener.stats().total_events(), 5);
}

// -------------------------------------------------------------------------
// Thread Safety Tests
// -------------------------------------------------------------------------

#[test]
fn test_jit_listener_concurrent_transitions() {
    let listener = Arc::new(JitShapeListener::new());
    let mut handles = vec![];

    for t in 0..10 {
        let l = Arc::clone(&listener);
        handles.push(thread::spawn(move || {
            for i in 0..100u32 {
                let base = t * 1000 + i;
                l.on_transition(ShapeId(base), ShapeId(base + 1));
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(listener.stats().transition_count(), 1000);
    assert_eq!(listener.stats().version_bump_count(), 1000);
}

#[test]
fn test_jit_listener_concurrent_mixed_events() {
    let listener = Arc::new(JitShapeListener::new());
    let mut handles = vec![];

    // Transition threads
    for t in 0..5 {
        let l = Arc::clone(&listener);
        handles.push(thread::spawn(move || {
            for i in 0..50u32 {
                l.on_transition(ShapeId(t * 100 + i), ShapeId(t * 100 + i + 1));
            }
        }));
    }

    // Deletion threads
    for t in 0..3 {
        let l = Arc::clone(&listener);
        handles.push(thread::spawn(move || {
            for i in 0..30u32 {
                l.on_property_delete(ShapeId(t * 100 + i), "prop");
            }
        }));
    }

    // Prototype change threads
    for t in 0..2 {
        let l = Arc::clone(&listener);
        handles.push(thread::spawn(move || {
            for i in 0..20u32 {
                l.on_prototype_change(ShapeId(t * 100 + i));
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // 5*50 = 250 transitions
    assert_eq!(listener.stats().transition_count(), 250);
    // 3*30 = 90 deletions
    assert_eq!(listener.stats().deletion_count(), 90);
    // 2*20 = 40 prototype changes
    assert_eq!(listener.stats().prototype_change_count(), 40);
    // Total: 380 events
    assert_eq!(listener.stats().total_events(), 380);
}

// -------------------------------------------------------------------------
// Global Listener Tests
// -------------------------------------------------------------------------

#[test]
fn test_global_jit_listener_singleton() {
    let l1 = global_jit_listener();
    let l2 = global_jit_listener();
    // Same instance
    assert!(std::ptr::eq(l1, l2));
}

// -------------------------------------------------------------------------
// Edge Case Tests
// -------------------------------------------------------------------------

#[test]
fn test_jit_listener_same_shape_transition() {
    let listener = JitShapeListener::new();
    let before = ShapeVersion::current();

    // Transition to same shape (edge case)
    listener.on_transition(ShapeId(1), ShapeId(1));

    let after = ShapeVersion::current();
    // Should still bump version
    assert!(after > before);
}

#[test]
fn test_jit_listener_max_shape_id() {
    let listener = JitShapeListener::new();

    listener.on_transition(ShapeId(u32::MAX), ShapeId(0));
    listener.on_property_delete(ShapeId(u32::MAX), "x");
    listener.on_prototype_change(ShapeId(u32::MAX));

    assert_eq!(listener.stats().total_events(), 3);
}

#[test]
fn test_jit_listener_empty_property_name() {
    let listener = JitShapeListener::new();

    listener.on_property_delete(ShapeId(1), "");
    listener.on_accessor_installed(ShapeId(1), "");

    assert_eq!(listener.stats().deletion_count(), 1);
    assert_eq!(listener.stats().accessor_install_count(), 1);
}

#[test]
fn test_jit_listener_unicode_property_name() {
    let listener = JitShapeListener::new();

    listener.on_property_delete(ShapeId(1), "日本語");
    listener.on_accessor_installed(ShapeId(1), "𝓤𝓷𝓲𝓬𝓸𝓭𝓮");

    assert_eq!(listener.stats().deletion_count(), 1);
    assert_eq!(listener.stats().accessor_install_count(), 1);
}

#[test]
fn test_jit_listener_large_batch() {
    let listener = JitShapeListener::new();

    let transitions: Vec<_> = (0..1000u32).map(|i| (ShapeId(i), ShapeId(i + 1))).collect();

    let count = listener.on_batch_transition(&transitions);

    assert_eq!(count, 1000);
    assert_eq!(listener.stats().batch_transition_count(), 1000);
    // Only one version bump for entire batch
    assert_eq!(listener.stats().version_bump_count(), 1);
}
