use super::*;
use std::thread;

// =========================================================================
// InvalidationReason Tests
// =========================================================================

#[test]
fn test_reason_should_cascade() {
    assert!(InvalidationReason::TierUpgrade.should_cascade());
    assert!(InvalidationReason::Deoptimization.should_cascade());
    assert!(InvalidationReason::SourceChanged.should_cascade());
    assert!(InvalidationReason::CalleeCascade.should_cascade());
    assert!(InvalidationReason::Manual.should_cascade());
    assert!(!InvalidationReason::MemoryPressure.should_cascade());
}

#[test]
fn test_reason_is_error() {
    assert!(!InvalidationReason::TierUpgrade.is_error());
    assert!(InvalidationReason::Deoptimization.is_error());
    assert!(!InvalidationReason::Manual.is_error());
}

// =========================================================================
// Registry Basic Tests
// =========================================================================

#[test]
fn test_registry_new() {
    let reg = InvalidationRegistry::new();
    assert_eq!(reg.function_count(), 0);
    assert_eq!(reg.dependency_count(), 0);
}

#[test]
fn test_register_function() {
    let reg = InvalidationRegistry::new();
    reg.register(1);
    assert_eq!(reg.function_count(), 1);
    assert_eq!(reg.get_version(1), 0);
}

#[test]
fn test_unregistered_version() {
    let reg = InvalidationRegistry::new();
    assert_eq!(reg.get_version(999), 0);
}

// =========================================================================
// Version Tests
// =========================================================================

#[test]
fn test_invalidation_increments_version() {
    let reg = InvalidationRegistry::new();
    reg.register(1);

    assert_eq!(reg.get_version(1), 0);
    reg.invalidate(1, InvalidationReason::Manual);
    assert_eq!(reg.get_version(1), 1);
}

#[test]
fn test_multiple_invalidations() {
    let reg = InvalidationRegistry::new();
    reg.register(1);

    for i in 1..=5 {
        reg.invalidate(1, InvalidationReason::Manual);
        assert_eq!(reg.get_version(1), i);
    }
}

#[test]
fn test_is_stale() {
    let reg = InvalidationRegistry::new();
    reg.register(1);

    assert!(!reg.is_stale(1, 0));
    reg.invalidate(1, InvalidationReason::Manual);
    assert!(reg.is_stale(1, 0));
    assert!(!reg.is_stale(1, 1));
}

// =========================================================================
// Dependency Tests
// =========================================================================

#[test]
fn test_record_inline() {
    let reg = InvalidationRegistry::new();
    reg.record_inline(1, 2); // 1 inlined 2

    let dependents = reg.get_dependents(2);
    assert_eq!(dependents.len(), 1);
    assert!(dependents.contains(&1));

    let callees = reg.get_callees(1);
    assert_eq!(callees.len(), 1);
    assert!(callees.contains(&2));
}

#[test]
fn test_multiple_dependents() {
    let reg = InvalidationRegistry::new();
    reg.record_inline(1, 10);
    reg.record_inline(2, 10);
    reg.record_inline(3, 10);

    let dependents = reg.get_dependents(10);
    assert_eq!(dependents.len(), 3);
    assert!(dependents.contains(&1));
    assert!(dependents.contains(&2));
    assert!(dependents.contains(&3));
}

#[test]
fn test_multiple_callees() {
    let reg = InvalidationRegistry::new();
    reg.record_inline(1, 10);
    reg.record_inline(1, 11);
    reg.record_inline(1, 12);

    let callees = reg.get_callees(1);
    assert_eq!(callees.len(), 3);
}

#[test]
fn test_remove_inline() {
    let reg = InvalidationRegistry::new();
    reg.record_inline(1, 2);
    reg.remove_inline(1, 2);

    assert!(reg.get_dependents(2).is_empty());
    assert!(reg.get_callees(1).is_empty());
}

#[test]
fn test_clear_dependencies() {
    let reg = InvalidationRegistry::new();
    reg.record_inline(1, 10);
    reg.record_inline(1, 11);
    reg.record_inline(1, 12);

    reg.clear_dependencies(1);

    assert!(reg.get_callees(1).is_empty());
    assert!(reg.get_dependents(10).is_empty());
    assert!(reg.get_dependents(11).is_empty());
    assert!(reg.get_dependents(12).is_empty());
}

// =========================================================================
// Cascade Tests
// =========================================================================

#[test]
fn test_cascade_invalidation() {
    let reg = InvalidationRegistry::new();
    reg.record_inline(1, 10); // 1 inlined 10

    // Invalidate 10, should cascade to 1
    let events = reg.invalidate(10, InvalidationReason::TierUpgrade);

    assert_eq!(events.len(), 2);
    assert!(reg.is_stale(10, 0));
    assert!(reg.is_stale(1, 0));
}

#[test]
fn test_multi_level_cascade() {
    let reg = InvalidationRegistry::new();
    // A inlined B, B inlined C
    reg.record_inline(1, 2);
    reg.record_inline(2, 3);

    // Invalidate C, should cascade to B, then to A
    let events = reg.invalidate(3, InvalidationReason::Deoptimization);

    assert_eq!(events.len(), 3);
    assert!(reg.is_stale(3, 0));
    assert!(reg.is_stale(2, 0));
    assert!(reg.is_stale(1, 0));
}

#[test]
fn test_no_cascade_memory_pressure() {
    let reg = InvalidationRegistry::new();
    reg.record_inline(1, 10);

    let events = reg.invalidate(10, InvalidationReason::MemoryPressure);

    // Only 10 invalidated, not 1
    assert_eq!(events.len(), 1);
    assert!(reg.is_stale(10, 0));
    assert!(!reg.is_stale(1, 0));
}

#[test]
fn test_cascade_diamond_pattern() {
    let reg = InvalidationRegistry::new();
    // Diamond: A and B both inline C
    reg.record_inline(1, 3);
    reg.record_inline(2, 3);

    let events = reg.invalidate(3, InvalidationReason::Manual);

    assert_eq!(events.len(), 3); // 3, 1, 2 (or 3, 2, 1)
}

#[test]
fn test_cascade_cycle() {
    let reg = InvalidationRegistry::new();
    // Cycle: A -> B -> C -> A
    reg.record_inline(1, 2);
    reg.record_inline(2, 3);
    reg.record_inline(3, 1);

    // Should not infinite loop
    let events = reg.invalidate(1, InvalidationReason::Manual);

    assert_eq!(events.len(), 3);
}

// =========================================================================
// Batch Tests
// =========================================================================

#[test]
fn test_invalidate_batch() {
    let reg = InvalidationRegistry::new();
    reg.register(1);
    reg.register(2);
    reg.register(3);

    let events = reg.invalidate_batch(&[1, 2, 3], InvalidationReason::Manual);

    assert_eq!(events.len(), 3);
    assert!(reg.is_stale(1, 0));
    assert!(reg.is_stale(2, 0));
    assert!(reg.is_stale(3, 0));
}

#[test]
fn test_invalidate_batch_dedupes() {
    let reg = InvalidationRegistry::new();
    reg.record_inline(1, 3);
    reg.record_inline(2, 3);

    // Invalidating 3 cascades to 1 and 2
    // Then invalidating 1 and 2 directly shouldn't double-count
    let events = reg.invalidate_batch(&[3, 1, 2], InvalidationReason::Manual);

    // Each function only once
    let func_ids: HashSet<_> = events.iter().map(|e| e.func_id).collect();
    assert_eq!(func_ids.len(), events.len());
}

// =========================================================================
// Event History Tests
// =========================================================================

#[test]
fn test_event_history() {
    let reg = InvalidationRegistry::new();
    reg.register(1);

    reg.invalidate(1, InvalidationReason::TierUpgrade);
    reg.invalidate(1, InvalidationReason::Deoptimization);

    let events = reg.get_recent_events(10);
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].reason, InvalidationReason::TierUpgrade);
    assert_eq!(events[1].reason, InvalidationReason::Deoptimization);
}

#[test]
fn test_function_events() {
    let reg = InvalidationRegistry::new();
    reg.register(1);
    reg.register(2);

    reg.invalidate(1, InvalidationReason::Manual);
    reg.invalidate(2, InvalidationReason::Manual);
    reg.invalidate(1, InvalidationReason::Manual);

    let events = reg.get_function_events(1);
    assert_eq!(events.len(), 2);
}

#[test]
fn test_total_invalidations() {
    let reg = InvalidationRegistry::new();
    reg.register(1);

    assert_eq!(reg.total_invalidations(), 0);
    reg.invalidate(1, InvalidationReason::Manual);
    assert_eq!(reg.total_invalidations(), 1);
}

#[test]
fn test_history_limit() {
    let reg = InvalidationRegistry::with_history_size(5);
    reg.register(1);

    for _ in 0..10 {
        reg.invalidate(1, InvalidationReason::Manual);
    }

    let events = reg.get_recent_events(100);
    assert!(events.len() <= 5);
}

// =========================================================================
// Last Reason Tests
// =========================================================================

#[test]
fn test_get_last_reason() {
    let reg = InvalidationRegistry::new();
    reg.register(1);

    assert!(reg.get_last_reason(1).is_none());

    reg.invalidate(1, InvalidationReason::TierUpgrade);
    assert_eq!(
        reg.get_last_reason(1),
        Some(InvalidationReason::TierUpgrade)
    );

    reg.invalidate(1, InvalidationReason::Deoptimization);
    assert_eq!(
        reg.get_last_reason(1),
        Some(InvalidationReason::Deoptimization)
    );
}

// =========================================================================
// Cleanup Tests
// =========================================================================

#[test]
fn test_remove() {
    let reg = InvalidationRegistry::new();
    reg.record_inline(1, 2);

    reg.remove(2);

    assert!(reg.get_dependents(2).is_empty());
    assert!(reg.get_callees(1).is_empty());
    assert_eq!(reg.function_count(), 1);
}

#[test]
fn test_clear() {
    let reg = InvalidationRegistry::new();
    reg.record_inline(1, 2);
    reg.invalidate(1, InvalidationReason::Manual);

    reg.clear();

    assert_eq!(reg.function_count(), 0);
    assert_eq!(reg.dependency_count(), 0);
    assert!(reg.get_recent_events(100).is_empty());
}

// =========================================================================
// Summary Tests
// =========================================================================

#[test]
fn test_summary() {
    let reg = InvalidationRegistry::new();
    reg.record_inline(1, 10);
    reg.record_inline(2, 10);

    reg.invalidate(10, InvalidationReason::TierUpgrade);
    reg.invalidate(1, InvalidationReason::Deoptimization);

    let summary = reg.summary();
    assert_eq!(summary.function_count, 3);
    assert_eq!(summary.dependency_count, 2);
    assert!(summary.tier_upgrade_count >= 1);
    assert!(summary.deoptimization_count >= 1);
}

// =========================================================================
// Clone Tests
// =========================================================================

#[test]
fn test_clone() {
    let reg = InvalidationRegistry::new();
    reg.record_inline(1, 2);
    reg.invalidate(1, InvalidationReason::Manual);

    let cloned = reg.clone();

    assert_eq!(cloned.get_version(1), 1);
    assert!(!cloned.get_dependents(2).is_empty());
}

// =========================================================================
// Thread Safety Tests
// =========================================================================

#[test]
fn test_concurrent_invalidation() {
    let reg = std::sync::Arc::new(InvalidationRegistry::new());

    for i in 0..10 {
        reg.register(i);
    }

    let mut handles = vec![];

    for i in 0..8 {
        let r = reg.clone();
        handles.push(thread::spawn(move || {
            for j in 0..100 {
                r.invalidate((j % 10) as u64, InvalidationReason::Manual);
                if i % 2 == 0 {
                    r.record_inline(i as u64, (i + 1) as u64);
                }
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // Should complete without panicking
    let _ = reg.summary();
}

#[test]
fn test_concurrent_dependency_tracking() {
    let reg = std::sync::Arc::new(InvalidationRegistry::new());
    let mut handles = vec![];

    for i in 0..4 {
        let r = reg.clone();
        handles.push(thread::spawn(move || {
            for j in 0..50 {
                r.record_inline(i as u64, (i * 100 + j) as u64);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // Each thread added 50 dependencies
    assert!(reg.dependency_count() >= 200);
}

// =========================================================================
// Event Timestamp Tests
// =========================================================================

#[test]
fn test_event_timestamps_monotonic() {
    let reg = InvalidationRegistry::new();
    reg.register(1);
    reg.register(2);

    let events1 = reg.invalidate(1, InvalidationReason::Manual);
    let events2 = reg.invalidate(2, InvalidationReason::Manual);

    assert!(events2[0].timestamp > events1[0].timestamp);
}

// =========================================================================
// Edge Case Tests
// =========================================================================

#[test]
fn test_self_inline() {
    let reg = InvalidationRegistry::new();
    reg.record_inline(1, 1); // Self-reference

    // Should not infinite loop
    let events = reg.invalidate(1, InvalidationReason::Manual);
    assert_eq!(events.len(), 1);
}

#[test]
fn test_invalidate_unregistered() {
    let reg = InvalidationRegistry::new();

    // Should handle gracefully
    let events = reg.invalidate(999, InvalidationReason::Manual);
    assert_eq!(events.len(), 1);
    assert_eq!(reg.get_version(999), 1);
}

#[test]
fn test_dependency_count() {
    let reg = InvalidationRegistry::new();
    reg.record_inline(1, 10);
    reg.record_inline(2, 10);
    reg.record_inline(1, 20);

    // 3 total inlining relationships
    assert_eq!(reg.dependency_count(), 3);
}
